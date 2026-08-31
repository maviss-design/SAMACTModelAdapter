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

"""公開ファクトリ関数。"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from samact_model_adapter import inspection, persistence
from samact_model_adapter.adapter import SAMACTModelAdapter
from samact_model_adapter.config import FitResult, SAMACTConfig
from samact_model_adapter.data import (
    build_dataloader,
    extract_input,
    extract_target,
    freeze_loader_order,
    to_numpy,
    validate_input_shape,
)
from samact_model_adapter.exceptions import InvalidSAMACTConfigError
from samact_model_adapter.hooks import collect_bottleneck_outputs
from samact_model_adapter.samact_builder import (
    build_encoder,
    build_learning_property,
    build_majority_decoder,
    build_neuron_focus_decoder,
    build_samlayer,
    build_sequential,
)
from samact_model_adapter.scaling import fit_minmax_scaler, transform_with_scaler

_VALID_TRANSLATE_TASKS = ("classification", "regression")


def load_from_artifact(artifact_path: str) -> SAMACTModelAdapter:
    """artifact directory から ``SAMACTModelAdapter`` を復元する。

    Args:
        artifact_path: ``SAMACTModelAdapter.save_artifact()`` で保存した
            artifact directory のパス。

    Returns:
        復元された ``SAMACTModelAdapter``。

    Raises:
        ArtifactLoadError: ``artifact_path`` が存在しない場合、
            ``metadata.json`` が存在しない、または JSON として読み込めない
            場合、metadata に記載された必須ファイルが存在しない場合、
            ``mode`` / ``task`` に応じた必須 metadata が不足している場合、
            SAMACT モデルまたは scaler の復元に失敗した場合、metadata と
            実ファイルの整合性が取れない場合。

    Warning:
        artifact のロードには pickle を使用するため、信頼できる artifact
        のみを読み込むこと。
    """
    return persistence.load_artifact(artifact_path)


def _collect_training_inputs(
    loader: DataLoader[object],
    input_dim: int,
    input_getter: Callable[[object], object] | None = None,
) -> np.ndarray:
    """design_spec.md 7.4 (SAMACT 入力 x の収集)を行う。

    `collect_predict_samples`(data.py, 12.1/12.2 用)と同様に、batch から入力を
    抽出・検証して 2D 配列へ連結するが、学習時は `input_getter` のみを受け取り、
    predict() 側の「batch 自体が Array であれば無条件でそのまま扱う」優先順位は
    適用しない(9.4 節の学習データ既定形式 `(x, y)` に従う)。
    """
    samples: list[np.ndarray] = []
    for batch in loader:
        x = to_numpy(extract_input(batch, input_getter))  # type: ignore[arg-type]
        samples.append(validate_input_shape(x, input_dim))  # type: ignore[arg-type]
    return np.concatenate(samples, axis=0)


def _collect_training_labels(
    loader: DataLoader[object],
    target_getter: Callable[[object], object] | None = None,
) -> np.ndarray:
    """design_spec.md 8.3(分類): DataLoader から `y` を収集し `(N,)` の整数配列にする。

    DataLoader から取得した `y` が float tensor であっても、ここで int に変換する
    (design_spec.md 8.3 / 要件仕様 F-07)。
    """
    samples: list[np.ndarray] = []
    for batch in loader:
        y = to_numpy(extract_target(batch, target_getter))  # type: ignore[arg-type]
        samples.append(np.asarray(y).reshape(-1))
    return np.concatenate(samples, axis=0).astype(np.int64)


def _collect_training_targets(
    loader: DataLoader[object],
    target_getter: Callable[[object], object] | None = None,
) -> np.ndarray:
    """design_spec.md 8.3(回帰): DataLoader から `y` を収集し `(N, output_dim)` の float配列にする。"""
    samples: list[np.ndarray] = []
    for batch in loader:
        y = to_numpy(extract_target(batch, target_getter))  # type: ignore[arg-type]
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        samples.append(y)
    return np.concatenate(samples, axis=0)


def translate_from_pytorch(  # pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
    model: nn.Module,
    loader: DataLoader[object] | Dataset[object],
    config: SAMACTConfig | None = None,
    task: str = "classification",
    val_loader: DataLoader[object] | Dataset[object] | None = None,
    input_getter: Callable[[object], object] | None = None,
    target_getter: Callable[[object], object] | None = None,
    dataloader_kwargs: dict[str, object] | None = None,
) -> SAMACTModelAdapter:
    """スタンドアロンモードで SAMACT モデルを構築・学習する。

    PyTorch モデルの先頭 ``nn.Linear`` から入力次元、末尾 ``nn.Linear`` から
    出力次元のみを抽出し、同じ入出力次元を持つ SAMACT モデルを新規に構築・
    学習する。PyTorch モデルの重み・中間構造・活性化関数は SAMACT へ変換
    されない。

    Args:
        model: 入出力次元の抽出元となる PyTorch モデル。
        loader: 学習データ。``DataLoader`` または ``Dataset``。
        config: SAMACT モデルの構築・学習設定。未指定の場合は既定値の
            ``SAMACTConfig`` を使用する。
        task: ``"classification"`` または ``"regression"``。
        val_loader: ``IterableDataset`` 検証のためだけに使用される。
            エポックごとの検証データ評価には使用されない。
        input_getter: batch から入力を抽出する関数。未指定時は batch を
            ``(x, y)`` とみなし ``batch[0]`` を入力として扱う。
        target_getter: batch から target を抽出する関数。未指定時は
            batch を ``(x, y)`` とみなし ``batch[1]`` を target として
            扱う。
        dataloader_kwargs: ``loader`` または ``val_loader`` が ``Dataset``
            の場合に、内部で ``DataLoader`` を構築する際のキーワード引数。

    Returns:
        ``mode="standalone"``、``task`` は指定どおりの、学習済み
        ``SAMACTModelAdapter``。

    Raises:
        UnsupportedArchitectureError: ``model`` に未対応レイヤーが含まれる
            場合、または Linear 層が 2 層未満で入出力次元を抽出できない
            場合。
        UnsupportedDatasetError: ``loader`` または ``val_loader`` が
            ``IterableDataset``、または ``IterableDataset`` ベースの
            ``DataLoader`` である場合。
        UnsupportedInputShapeError: 学習データから抽出した入力が 3 次元
            以上である場合。
        InputDimensionMismatchError: 学習データから抽出した入力の次元数が
            抽出済みの入力次元と一致しない場合。
        UnsupportedBatchFormatError: ``input_getter`` / ``target_getter``
            未指定時に batch を既定形式 ``(x, y)`` として解釈できない
            場合、またはそれらの実行に失敗した場合。
        InvalidSAMACTConfigError: ``task`` が ``"classification"`` /
            ``"regression"`` 以外の場合、``config.samlayer_kwargs`` に
            PySAMACT ``SAMLayer`` コンストラクタに存在しないキーが指定
            された場合。``config`` 自体の値の妥当性は ``SAMACTConfig``
            のコンストラクタで検証済みであるため、本関数の実行中に
            ``config`` の値が原因でこの例外が送出されることはない。
    """
    if task not in _VALID_TRANSLATE_TASKS:
        raise InvalidSAMACTConfigError(
            f"Invalid value specified for task: {task!r}. "
            f"Valid values are {_VALID_TRANSLATE_TASKS}."
        )
    if config is None:
        config = SAMACTConfig()

    dims = inspection.extract_linear_dims(model)
    input_dim = dims.input_dim
    output_dim = dims.output_dim

    train_loader = build_dataloader(
        loader, dataloader_kwargs=dataloader_kwargs, torch_seed=config.torch_seed
    )
    # x と y を別々の走査で収集するため、shuffle=True の loader だと走査ごとに
    # 順序が変わり対応関係が崩れる。1 度だけ順序を確定させてから使う。
    train_loader = freeze_loader_order(train_loader)
    if val_loader is not None:
        # design_spec.md 8.1: val_loader によるエポックごとの検証はサポートしない。
        # ただし 5.2 の送出例外(UnsupportedDatasetError)はここでも検証する。
        build_dataloader(
            val_loader, dataloader_kwargs=dataloader_kwargs, torch_seed=config.torch_seed
        )

    x = _collect_training_inputs(train_loader, input_dim, input_getter=input_getter)
    input_scaler = fit_minmax_scaler(x)
    x_scaled = transform_with_scaler(input_scaler, x, clip=config.clip_scaled_values)

    encoder = build_encoder(config, input_dim)

    target_scaler: MinMaxScaler | None
    if task == "classification":
        y_train: np.ndarray = _collect_training_labels(train_loader, target_getter=target_getter)
        decoder = build_majority_decoder()
        units = output_dim
        fit_metrics = "accuracy"
        target_scaler = None
    else:
        y_targets = _collect_training_targets(train_loader, target_getter=target_getter)
        target_scaler = fit_minmax_scaler(y_targets)
        y_train = transform_with_scaler(target_scaler, y_targets, clip=config.clip_scaled_values)
        decoder = build_neuron_focus_decoder(output_dim)
        units = output_dim * config.n_output_multiplier
        fit_metrics = "mse"

    hidden_layer = build_samlayer(config, units=config.hidden_units)
    final_layer = build_samlayer(config, units=units, is_final=True)
    samact_model = build_sequential(config, encoder, decoder, [hidden_layer, final_layer])
    learning_property = build_learning_property(config)

    start = time.perf_counter()
    fit_result_raw = samact_model.Fit(
        x_scaled,
        y_train,
        epochs=config.epochs,
        learningProperty=learning_property,
        metrics=fit_metrics,
    )
    elapsed_seconds = time.perf_counter() - start

    fit_result = FitResult(
        epochs=config.epochs,
        elapsed_seconds=elapsed_seconds,
        mse_per_epoch=list(fit_result_raw.metrics) if task == "regression" else None,
        accuracy_per_epoch=list(fit_result_raw.metrics) if task == "classification" else None,
    )

    return SAMACTModelAdapter(
        samact_model=samact_model,
        mode="standalone",
        task=task,
        input_dim=input_dim,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
        config=config,
        metadata={},
        fit_result=fit_result,
        output_dim=output_dim,
    )


def compress_from_pytorch(  # pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
    model: nn.Module,
    loader: DataLoader[object] | Dataset[object],
    config: SAMACTConfig | None = None,
    bottleneck_layer_name: str | None = None,
    val_loader: DataLoader[object] | Dataset[object] | None = None,
    input_getter: Callable[[object], object] | None = None,
    dataloader_kwargs: dict[str, object] | None = None,
) -> SAMACTModelAdapter:
    """特徴量圧縮モードで SAMACT モデルを構築・学習する。

    PyTorch モデルの入力からボトルネック層出力までの前段処理を、SAMACT
    モデルで近似するよう回帰学習する。学習後の adapter の ``predict()`` は
    最終タスクの予測ではなく、SAMACT が再現したボトルネック表現を返す。

    Args:
        model: ボトルネック層を検出・使用する PyTorch モデル。
        loader: 学習データ。``DataLoader`` または ``Dataset``。
        config: SAMACT モデルの構築・学習設定。未指定の場合は既定値の
            ``SAMACTConfig`` を使用する。
        bottleneck_layer_name: 使用するボトルネック層名。未指定の場合、
            出力層候補を除いた ``nn.Linear`` 層のうち ``out_features`` が
            最小の層を自動検出する。
        val_loader: ``IterableDataset`` 検証のためだけに使用される。
            エポックごとの検証データ評価には使用されない。
        input_getter: batch から入力を抽出する関数。未指定時は batch を
            ``(x, y)`` とみなし ``batch[0]`` を入力として扱う。
        dataloader_kwargs: ``loader`` または ``val_loader`` が ``Dataset``
            の場合に、内部で ``DataLoader`` を構築する際のキーワード引数。

    Returns:
        ``mode="compression"``、``task="compression"`` の、学習済み
        ``SAMACTModelAdapter``。

    Raises:
        UnsupportedArchitectureError: ``model`` に未対応レイヤーが含まれる
            場合。``bottleneck_layer_name`` を手動指定した場合も検証
            される。
        BottleneckDetectionError: ``bottleneck_layer_name`` 未指定時に
            ボトルネック候補が存在しない場合、指定した層名が ``model``
            に存在しない場合、指定した層が ``nn.Linear`` ではない場合。
        UnsupportedDatasetError: ``loader`` または ``val_loader`` が
            ``IterableDataset``、または ``IterableDataset`` ベースの
            ``DataLoader`` である場合。
        UnsupportedInputShapeError: 学習データから抽出した入力が 3 次元
            以上である場合。
        InputDimensionMismatchError: 学習データから抽出した入力の次元数が
            ``model`` から抽出した入力次元と一致しない場合。
        UnsupportedBatchFormatError: ``input_getter`` 未指定時に batch を
            既定形式 ``(x, y)`` として解釈できない場合、または
            ``input_getter`` の実行に失敗した場合。
        InvalidSAMACTConfigError: ``config.samlayer_kwargs`` に PySAMACT
            ``SAMLayer`` コンストラクタに存在しないキーが指定された場合。
            ``config`` 自体の値の妥当性は ``SAMACTConfig`` のコンスト
            ラクタで検証済みであるため、本関数の実行中に ``config`` の
            値が原因でこの例外が送出されることはない。
    """
    if config is None:
        config = SAMACTConfig()

    inspection.validate_supported_architecture(model)
    bottleneck_info = inspection.resolve_bottleneck_layer(model, bottleneck_layer_name)
    input_dim = inspection.extract_linear_dims(model).input_dim

    train_loader = build_dataloader(
        loader, dataloader_kwargs=dataloader_kwargs, torch_seed=config.torch_seed
    )
    # x と bottleneck 出力 y を別々の走査で収集するため、shuffle=True の loader
    # だと走査ごとに順序が変わり対応関係が崩れる。1 度だけ順序を確定させてから使う。
    train_loader = freeze_loader_order(train_loader)
    if val_loader is not None:
        # design_spec.md 7.6: val_loader によるエポックごとの検証はサポートしない。
        # ただし 5.1 の送出例外(UnsupportedDatasetError)はここでも検証する。
        build_dataloader(
            val_loader, dataloader_kwargs=dataloader_kwargs, torch_seed=config.torch_seed
        )

    x = _collect_training_inputs(train_loader, input_dim, input_getter=input_getter)
    y = collect_bottleneck_outputs(
        model, train_loader, bottleneck_info.layer_name, input_getter=input_getter
    )
    bottleneck_dim = y.shape[1]

    input_scaler = fit_minmax_scaler(x)
    target_scaler = fit_minmax_scaler(y)
    x_scaled = transform_with_scaler(input_scaler, x, clip=config.clip_scaled_values)
    y_scaled = transform_with_scaler(target_scaler, y, clip=config.clip_scaled_values)

    encoder = build_encoder(config, input_dim)
    decoder = build_neuron_focus_decoder(bottleneck_dim)
    hidden_layer = build_samlayer(config, units=config.hidden_units)
    final_layer = build_samlayer(
        config, units=bottleneck_dim * config.n_output_multiplier, is_final=True
    )
    samact_model = build_sequential(config, encoder, decoder, [hidden_layer, final_layer])
    learning_property = build_learning_property(config)

    start = time.perf_counter()
    fit_result_raw = samact_model.Fit(
        x_scaled,
        y_scaled,
        epochs=config.epochs,
        learningProperty=learning_property,
        metrics="mse",
    )
    elapsed_seconds = time.perf_counter() - start

    fit_result = FitResult(
        epochs=config.epochs,
        elapsed_seconds=elapsed_seconds,
        mse_per_epoch=list(fit_result_raw.metrics),
        accuracy_per_epoch=None,
    )

    return SAMACTModelAdapter(
        samact_model=samact_model,
        mode="compression",
        task="compression",
        input_dim=input_dim,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
        config=config,
        metadata={},
        fit_result=fit_result,
        bottleneck_layer_name=bottleneck_info.layer_name,
        bottleneck_dim=bottleneck_dim,
    )
