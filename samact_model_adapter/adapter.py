# SAMACT Model Adapter
# Copyright (C) 2026 Maviss Design
#
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
#
# If you modify this software and make it available over a network,
# you must provide the corresponding source code to users (AGPLv3 §13).
#
# For commercial licensing options, contact:
# solution-sales@maviss-design.com
#
# See the LICENSE file for full details.

"""SAMACTModelAdapter dataclass and invariants."""

import logging
from dataclasses import dataclass
from typing import Callable
from pathlib import Path

import numpy as np
import torch
from samact.ISAMACTModel import ITrainableSAMACTModel
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from samact_model_adapter import inspection, persistence
from samact_model_adapter.config import FitResult, SAMACTConfig
from samact_model_adapter.data import (
    Array,
    build_dataloader,
    collect_predict_samples,
    extract_input,
    extract_target,
    freeze_loader_order,
    to_numpy,
)
from samact_model_adapter.exceptions import InvalidSAMACTConfigError
from samact_model_adapter.hooks import collect_bottleneck_outputs
from samact_model_adapter.metrics import (
    calc_accuracy,
    calc_correct_summary,
    calc_mae,
    calc_mse,
    calc_rmse,
    model_disagreement_rate,
)
from samact_model_adapter.scaling import transform_with_scaler

_logger = logging.getLogger(__name__)

_VALID_MODES = ("compression", "standalone")
_VALID_TASKS = ("compression", "classification", "regression")


def _as_model_input_tensor(x: object) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    raise TypeError(
        f"model input must be a torch.Tensor or np.ndarray: {type(x)!r}"
    )


def _collect_pytorch_outputs_and_targets(
    model: nn.Module,
    loader: DataLoader[object],
    input_getter: Callable[[object], object] | None = None,
    target_getter: Callable[[object], object] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """design_spec.md 13.3/13.4/13.5: 元 PyTorch モデルの forward 出力と target を収集する。

    `hooks.collect_bottleneck_outputs()` は指定層の forward hook 収集専用であるため、
    ここでは最終出力そのものを収集する専用実装とする。training state の保存・復元、
    `torch.no_grad()` 下での実行は design_spec.md 13.5 の方針を踏襲する。
    """
    was_training = model.training
    outputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    try:
        model.eval()
        with torch.no_grad():
            for batch in loader:
                x = extract_input(batch, input_getter)
                y = extract_target(batch, target_getter)
                out = model(_as_model_input_tensor(x))
                outputs.append(out.detach().cpu().numpy())
                targets.append(to_numpy(y))  # type: ignore[arg-type]
    finally:
        model.train(was_training)
    return np.concatenate(outputs, axis=0), np.concatenate(targets, axis=0)


@dataclass
class SAMACTModelAdapter:  # pylint: disable=too-many-instance-attributes
    """学習済み SAMACT モデルと、その学習・推論に必要な情報を保持するクラス。

    ``compress_from_pytorch()`` / ``translate_from_pytorch()`` /
    ``load_from_artifact()`` の戻り値として得られる。

    Attributes:
        samact_model: 学習済みの PySAMACT モデル。
        mode: ``"compression"``(特徴量圧縮モード)または ``"standalone"``
            (スタンドアロンモード)。
        task: ``"compression"``、``"classification"``、``"regression"``
            のいずれか。
        input_dim: SAMACT モデルの入力次元数。
        input_scaler: 入力を ``[0, 1]`` に正規化する ``MinMaxScaler``。
        target_scaler: 教師信号を ``[0, 1]`` に正規化する ``MinMaxScaler``。
            ``mode="standalone"`` かつ ``task="classification"`` の場合は
            ``None``。
        config: 学習に使用した ``SAMACTConfig``。
        metadata: artifact 保存・復元用のメタデータ。
        fit_result: 学習結果を保持する ``FitResult``。
        output_dim: スタンドアロンモードでの出力次元数。特徴量圧縮モードでは
            ``None``。
        bottleneck_layer_name: 特徴量圧縮モードで使用したボトルネック層名。
            スタンドアロンモードでは ``None``。
        bottleneck_dim: 特徴量圧縮モードでのボトルネック層の出力次元数。
            スタンドアロンモードでは ``None``。
    """

    # ユーザがインスタンスすることはない(=mypyで拾える)ので、動的な型チェックは不要
    samact_model: ITrainableSAMACTModel
    mode: str
    task: str
    input_dim: int
    input_scaler: MinMaxScaler
    target_scaler: MinMaxScaler | None
    config: SAMACTConfig
    # design_spec.md 14.3 は「少なくとも以下を含める」という open-ended な項目一覧であり、
    # 実際のキー集合は persistence.py(#25/#26)側で確定する。そちらで型が固まった時点で、
    # dict[str, object] のままにするか専用型にするかを改めて判断する。
    metadata: dict[str, object]
    fit_result: FitResult | None = None
    output_dim: int | None = None
    bottleneck_layer_name: str | None = None
    bottleneck_dim: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise InvalidSAMACTConfigError(
                f"Invalid value specified for mode: {self.mode!r}. "
                f"Valid values are {_VALID_MODES}."
            )
        if self.task not in _VALID_TASKS:
            raise InvalidSAMACTConfigError(
                f"Invalid value specified for task: {self.task!r}. "
                f"Valid values are {_VALID_TASKS}."
            )
        if self.mode == "compression":
            if self.target_scaler is None:
                raise InvalidSAMACTConfigError(
                    "target_scaler is required when mode='compression'."
                )
            if self.bottleneck_layer_name is None:
                raise InvalidSAMACTConfigError(
                    "bottleneck_layer_name is required when mode='compression'."
                )
            if self.bottleneck_dim is None:
                raise InvalidSAMACTConfigError(
                    "bottleneck_dim is required when mode='compression'."
                )
        if self.mode == "standalone" and self.task == "regression":
            if self.target_scaler is None:
                raise InvalidSAMACTConfigError(
                    "target_scaler is required when mode='standalone', task='regression'."
                )
            if self.output_dim is None:
                raise InvalidSAMACTConfigError(
                    "output_dim is required when mode='standalone', task='regression'."
                )
        if self.mode == "standalone" and self.task == "classification":
            if self.target_scaler is not None:
                raise InvalidSAMACTConfigError(
                    "target_scaler must be None when mode='standalone', task='classification'."
                )

    def save_artifact(self, path: str | Path, overwrite: bool = False) -> None:
        """adapter を artifact directory として保存する。

        保存されるファイル:

        - ``samact_model.h5``: 学習済み SAMACT モデル。
        - ``input_scaler.pkl``: 入力用の ``MinMaxScaler`` (pickle)。
        - ``target_scaler.pkl``: 教師信号用の ``MinMaxScaler`` (pickle)。
          ``target_scaler`` が存在する場合のみ保存される。
        - ``metadata.json``: ``mode`` / ``task`` /各種設定などのメタデータ。

        Args:
            path: 保存先ディレクトリのパス。
            overwrite: ``True`` の場合、既存の保存先を上書きする。``False``
                (既定)の場合、既存の保存先が存在するとエラーになる。

        Raises:
            FileExistsError: 保存先ディレクトリが既に存在し、
                ``overwrite=False`` の場合。
            ArtifactPersistenceError: SAMACT モデル、scaler、
                ``metadata.json`` のいずれかの保存に失敗した場合。
        """
        persistence.save_artifact(self, path, overwrite=overwrite)

    def _normalize_predict_input(
        self,
        data: Array | DataLoader[object] | Dataset[object],
        input_getter: Callable[[object], object] | None = None,
        dataloader_kwargs: dict[str, object] | None = None,
    ) -> np.ndarray:
        """design_spec.md 12.1 / 12.2 共通処理 手順 1-4(predict() 入力正規化)を行う。

        `predict()` 本体(SAMACT `Predict()` の呼び出しおよび mode/task 別の後処理、
        design_spec.md 12.3-12.6)は本 Issue のスコープ外であり、ここでは data を
        検証済み・scaling 済みの shape `(N, input_dim)` の 2D `np.ndarray` に
        正規化するところまでを担う。

        `input_getter` / `dataloader_kwargs` は design_spec.md 12.1 の `predict()` が
        受け取るパラメータをそのまま中継する。学習時の
        `compress_from_pytorch()` / `translate_from_pytorch()` の `input_getter` とは
        独立しており、adapter は学習時の `input_getter` を保持・再利用しない。
        """
        x = collect_predict_samples(
            data,
            self.input_dim,
            input_getter=input_getter,
            dataloader_kwargs=dataloader_kwargs,
            torch_seed=self.config.torch_seed,
        )
        return transform_with_scaler(
            self.input_scaler, x, clip=self.config.clip_scaled_values
        )

    def predict(
        self,
        data: Array | DataLoader[object] | Dataset[object],
        input_getter: Callable[[object], object] | None = None,
        dataloader_kwargs: dict[str, object] | None = None,
    ) -> np.ndarray | int:
        """入力データに対する SAMACT モデルの推論を行う。

        Args:
            data: 入力データ。``torch.Tensor``、``np.ndarray``、
                ``DataLoader``、``Dataset`` のいずれかを指定できる。
            input_getter: ``data`` が ``DataLoader`` / ``Dataset`` の場合に、
                batch から入力を抽出する関数。未指定の場合、batch 自体が
                ``torch.Tensor`` / ``np.ndarray`` であればそれをそのまま
                入力とし、それ以外は batch を ``(x, y)`` とみなし
                ``batch[0]`` を入力として扱う。``compress_from_pytorch()``
                / ``translate_from_pytorch()`` に渡した ``input_getter``
                とは独立しており、学習時の設定は再利用されない。
            dataloader_kwargs: ``data`` が ``Dataset`` の場合に、内部で
                ``DataLoader`` を構築する際のキーワード引数。``data`` が
                ``DataLoader`` の場合は無視される。

        Returns:
            ``mode="compression"`` の場合は shape ``(N, bottleneck_dim)``
            の ``np.ndarray``。``mode="standalone"`` かつ
            ``task="regression"`` の場合は shape ``(N, output_dim)`` の
            ``np.ndarray``。``mode="standalone"`` かつ
            ``task="classification"`` の場合、``N > 1`` では shape
            ``(N,)`` の int 配列、``N == 1`` では int scalar。

        Raises:
            UnsupportedDatasetError: ``data`` が ``IterableDataset``、
                または ``IterableDataset`` ベースの ``DataLoader`` である
                場合。
            UnsupportedInputShapeError: 抽出した入力が 3 次元以上である
                場合。
            InputDimensionMismatchError: 抽出した入力の次元数が
                ``input_dim`` と一致しない場合。
            UnsupportedBatchFormatError: ``input_getter`` 未指定時に
                batch を既定形式 ``(x, y)`` として解釈できない場合、
                または ``input_getter`` の実行に失敗した場合。
        """
        x = self._normalize_predict_input(
            data, input_getter=input_getter, dataloader_kwargs=dataloader_kwargs
        )
        raw = np.stack([self.samact_model.Predict(sample) for sample in x])
        return self._postprocess_predict_output(raw)

    def _postprocess_predict_output(self, raw: np.ndarray) -> np.ndarray | int:
        """design_spec.md 12.3-12.5(mode / task 別の predict() 戻り値後処理)を行う。

        `raw` は SAMACT `Predict()` を 1 サンプルずつ呼び出した結果を
        `np.stack` した配列であり、task="classification" では shape `(N,)` の
        int 配列、それ以外(compression / standalone regression)では shape
        `(N, dim)` の配列である。
        """
        if self.task == "classification":
            if raw.shape[0] == 1:
                return int(raw[0])
            return raw
        # compression (12.3) / standalone regression (12.4): 両方とも
        # target_scaler.inverse_transform() を適用し shape (N, dim) を返す。
        # target_scaler は __post_init__ の invariant により非 None が保証される。
        assert self.target_scaler is not None
        inverse_transformed: np.ndarray = self.target_scaler.inverse_transform(raw)
        return inverse_transformed

    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader[object] | Dataset[object],
        input_getter: Callable[[object], object] | None = None,
        target_getter: Callable[[object], object] | None = None,
        dataloader_kwargs: dict[str, object] | None = None,
    ) -> dict[str, float]:
        """元の PyTorch モデルと SAMACT モデルの予測性能を比較する。

        ``mode`` / ``task`` に応じて処理を切り替える。処理前後で
        ``model.training`` は変化しない。

        Args:
            model: 比較対象の PyTorch モデル。
            loader: 評価データ。``DataLoader`` または ``Dataset``。
            input_getter: batch から入力を抽出する関数。未指定時は batch
                を ``(x, y)`` とみなし ``batch[0]`` を入力として扱う。
            target_getter: batch から target を抽出する関数。未指定時は
                batch を ``(x, y)`` とみなし ``batch[1]`` を target として
                扱う。
            dataloader_kwargs: ``loader`` が ``Dataset`` の場合に、内部で
                ``DataLoader`` を構築する際のキーワード引数。

        Returns:
            ``mode`` / ``task`` に応じたキーを持つ dict。

            - ``mode="compression"``: ``samact_mse``、``samact_rmse``、
              ``samact_mae``
            - ``mode="standalone"`` かつ ``task="classification"``:
              ``pytorch_accuracy``、``samact_accuracy``
            - ``mode="standalone"`` かつ ``task="regression"``:
              ``pytorch_mse``、``samact_mse``、``pytorch_mae``、
              ``samact_mae``

        Raises:
            UnsupportedArchitectureError: ``model`` に未対応レイヤーが
                含まれる場合、またはスタンドアロンモードで必要な Linear
                層構成を確認できない場合。
            BottleneckDetectionError: 特徴量圧縮モードで
                ``self.bottleneck_layer_name`` が ``model`` に存在しない
                場合、または ``nn.Linear`` ではない場合。
            UnsupportedDatasetError: ``loader`` が ``IterableDataset``、
                または ``IterableDataset`` ベースの ``DataLoader`` である
                場合。
            UnsupportedInputShapeError: 抽出した入力が 3 次元以上である
                場合。
            InputDimensionMismatchError: 抽出した入力の次元数が
                ``input_dim`` と一致しない場合。
            UnsupportedBatchFormatError: ``input_getter`` /
                ``target_getter`` 未指定時に batch を既定形式 ``(x, y)``
                として解釈できない場合、またはそれらの実行に失敗した場合。
        """
        if self.mode == "compression":
            return self._evaluate_compression(
                model, loader, input_getter=input_getter, dataloader_kwargs=dataloader_kwargs
            )
        if self.task == "classification":
            return self._evaluate_standalone_classification(
                model,
                loader,
                input_getter=input_getter,
                target_getter=target_getter,
                dataloader_kwargs=dataloader_kwargs,
            )
        return self._evaluate_standalone_regression(
            model,
            loader,
            input_getter=input_getter,
            target_getter=target_getter,
            dataloader_kwargs=dataloader_kwargs,
        )

    def _evaluate_standalone_classification(
        self,
        model: nn.Module,
        loader: DataLoader[object] | Dataset[object],
        input_getter: Callable[[object], object] | None = None,
        target_getter: Callable[[object], object] | None = None,
        dataloader_kwargs: dict[str, object] | None = None,
    ) -> dict[str, float]:
        """design_spec.md 13.3 を参照。"""
        inspection.extract_linear_dims(model)

        eval_loader = build_dataloader(
            loader, dataloader_kwargs=dataloader_kwargs, torch_seed=self.config.torch_seed
        )
        # pytorch 側の出力/target と samact 側の predict() を別々の走査で取得するため、
        # shuffle=True の loader だと走査ごとに順序が変わり対応関係が崩れる。
        eval_loader = freeze_loader_order(eval_loader)
        pytorch_logits, y_true = _collect_pytorch_outputs_and_targets(
            model, eval_loader, input_getter=input_getter, target_getter=target_getter
        )
        pytorch_pred = pytorch_logits.argmax(axis=-1)
        y_true_int = y_true.astype(np.int64)

        samact_pred = np.atleast_1d(
            np.asarray(self.predict(eval_loader, input_getter=input_getter))
        )

        correct_summary = calc_correct_summary(pytorch_pred, samact_pred, y_true_int)
        disagreement_rate = model_disagreement_rate(pytorch_pred, samact_pred)
        _logger.info(
            "Classification 4-quadrant summary: %s, inter-model prediction disagreement rate: %.4f",
            correct_summary,
            disagreement_rate,
        )

        return {
            "pytorch_accuracy": calc_accuracy(pytorch_pred, y_true_int),
            "samact_accuracy": calc_accuracy(samact_pred, y_true_int),
        }

    def _evaluate_standalone_regression(
        self,
        model: nn.Module,
        loader: DataLoader[object] | Dataset[object],
        input_getter: Callable[[object], object] | None = None,
        target_getter: Callable[[object], object] | None = None,
        dataloader_kwargs: dict[str, object] | None = None,
    ) -> dict[str, float]:
        """design_spec.md 13.4 を参照。"""
        inspection.extract_linear_dims(model)

        eval_loader = build_dataloader(
            loader, dataloader_kwargs=dataloader_kwargs, torch_seed=self.config.torch_seed
        )
        # pytorch 側の出力/target と samact 側の predict() を別々の走査で取得するため、
        # shuffle=True の loader だと走査ごとに順序が変わり対応関係が崩れる。
        eval_loader = freeze_loader_order(eval_loader)
        pytorch_pred, y_true = _collect_pytorch_outputs_and_targets(
            model, eval_loader, input_getter=input_getter, target_getter=target_getter
        )
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        samact_pred = self.predict(eval_loader, input_getter=input_getter)
        assert isinstance(samact_pred, np.ndarray)

        prediction_mae = calc_mae(samact_pred, pytorch_pred)
        _logger.info("MAE between original model and SAMACT predictions: %.6f", prediction_mae)

        return {
            "pytorch_mse": calc_mse(pytorch_pred, y_true),
            "samact_mse": calc_mse(samact_pred, y_true),
            "pytorch_mae": calc_mae(pytorch_pred, y_true),
            "samact_mae": calc_mae(samact_pred, y_true),
        }

    def _evaluate_compression(
        self,
        model: nn.Module,
        loader: DataLoader[object] | Dataset[object],
        input_getter: Callable[[object], object] | None = None,
        dataloader_kwargs: dict[str, object] | None = None,
    ) -> dict[str, float]:
        """design_spec.md 13.2 / 13.5 / 13.6 を参照。

        `adapter.bottleneck_layer_name` の存在・`nn.Linear` 検証は
        `inspection.resolve_bottleneck_layer()` に委譲する(未対応アーキテクチャ検証も
        あわせて行われる)。PyTorch モデルの training state 管理は
        `collect_bottleneck_outputs()` に委譲する。
        """
        assert self.bottleneck_layer_name is not None
        inspection.resolve_bottleneck_layer(model, self.bottleneck_layer_name)

        eval_loader = build_dataloader(
            loader, dataloader_kwargs=dataloader_kwargs, torch_seed=self.config.torch_seed
        )
        # pytorch 側のbottleneck出力とsamact側のpredict()を別々の走査で取得するため、
        # shuffle=True の loader だと走査ごとに順序が変わり対応関係が崩れる。
        eval_loader = freeze_loader_order(eval_loader)
        pytorch_outputs = collect_bottleneck_outputs(
            model, eval_loader, self.bottleneck_layer_name, input_getter=input_getter
        )
        samact_outputs = self.predict(eval_loader, input_getter=input_getter)

        return {
            "samact_mse": calc_mse(samact_outputs, pytorch_outputs),
            "samact_rmse": calc_rmse(samact_outputs, pytorch_outputs),
            "samact_mae": calc_mae(samact_outputs, pytorch_outputs),
        }
