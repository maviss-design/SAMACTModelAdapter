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

"""Dataset / DataLoader 準備処理。

design_spec.md 9. Dataset / DataLoader 連携設計 を参照。
"""

from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from samact_model_adapter.exceptions import (
    InputDimensionMismatchError,
    UnsupportedBatchFormatError,
    UnsupportedDatasetError,
    UnsupportedInputShapeError,
)

_DEFAULT_DATALOADER_KWARGS: dict[str, object] = {
    "batch_size": 1,
    "shuffle": False,
    "num_workers": 0,
}

Array = torch.Tensor | np.ndarray


def _raise_if_iterable_dataset(dataset: object) -> None:
    if isinstance(dataset, IterableDataset):
        raise UnsupportedDatasetError(
            "IterableDataset is not supported (design_spec.md 9.3). "
            "Use a map-style Dataset because scaler fitting requires two passes over the data."
        )


def build_dataloader(
    loader: DataLoader[object] | Dataset[object],
    dataloader_kwargs: dict[str, object] | None = None,
    torch_seed: int | None = None,
) -> DataLoader[object]:
    """design_spec.md 9.2 / 9.3, 16.1, 4.1 を参照して DataLoader を準備する。

    - `loader` が DataLoader の場合、既存の sampler / collate_fn / num_workers /
      pin_memory 等を尊重し、再構築せずそのまま返す。
    - `loader` が Dataset の場合、`dataloader_kwargs` を用いて DataLoader を構築する。
      `dataloader_kwargs` が None の場合は既定値
      (`batch_size=1, shuffle=False, num_workers=0`) を使用する。
    - `torch_seed` が指定され、`dataloader_kwargs` に `generator` が含まれない場合、
      seed 付き `torch.Generator` を設定する。
    - IterableDataset (または IterableDataset を dataset に持つ DataLoader) は
      `UnsupportedDatasetError` を送出する。
    """
    if isinstance(loader, DataLoader):
        _raise_if_iterable_dataset(loader.dataset)
        return loader

    if isinstance(loader, Dataset):
        _raise_if_iterable_dataset(loader)
        kwargs: dict[str, object] = (
            dict(_DEFAULT_DATALOADER_KWARGS)
            if dataloader_kwargs is None
            else dict(dataloader_kwargs)
        )
        if torch_seed is not None and "generator" not in kwargs:
            generator = torch.Generator()
            generator.manual_seed(torch_seed)
            kwargs["generator"] = generator
        return DataLoader(loader, **kwargs)  # type: ignore[arg-type]

    raise TypeError(
        "loader must be a torch.utils.data.DataLoader or Dataset: "
        f"{type(loader)!r}"
    )


class _FrozenBatchSequence(Dataset[object]):
    """`freeze_loader_order()` が確定させた batch 列をそのまま保持する内部 Dataset。"""

    def __init__(self, batches: list[object]) -> None:
        self._batches = batches

    def __len__(self) -> int:
        return len(self._batches)

    def __getitem__(self, index: int) -> object:
        return self._batches[index]


def freeze_loader_order(loader: DataLoader[object]) -> DataLoader[object]:
    """`loader` を 1 回だけ走査して batch 列を確定し、再イテレートしても常に
    同じ順序で同じ batch を返す新しい `DataLoader` を返す。

    入力 `x` とラベル/ターゲット `y`(または bottleneck 出力)を、同じ `loader`
    に対する別々の走査で収集する処理(`_collect_training_inputs()` と
    `_collect_training_labels()`/`_collect_training_targets()`、
    `collect_bottleneck_outputs()`、`evaluate()` 内の `predict()` 呼び出しなど)は、
    `loader` が `shuffle=True` の場合、走査のたびに異なる順序を観測してしまい、
    `x` と `y` の対応関係が崩れる。1 度だけ `list(loader)` で確定させ、
    `shuffle=False` の新しい `DataLoader` でラップし直すことでこれを防ぐ。
    """
    batches = list(loader)
    return DataLoader(_FrozenBatchSequence(batches), batch_size=None, shuffle=False)


def _unsupported_batch_format_error(
    batch: object,
    *,
    getter_name: str | None = None,
    cause: Exception | None = None,
) -> UnsupportedBatchFormatError:
    """要件仕様 7.8 / design_spec.md 15.7 が求めるメッセージ要素を含む例外を組み立てる。

    メッセージには以下を含める。
    - 実際の batch 型
    - 期待する既定形式 `(x, y)`
    - dict / namedtuple / 追加情報付き batch を使う場合の
      `input_getter` / `target_getter` 指定の案内
    """
    guidance = (
        "If you use a batch with extra information such as dict / namedtuple / "
        "(x, y, sample_weight), specify input_getter / target_getter."
    )
    if getter_name is not None and cause is not None:
        message = (
            f"Extraction via {getter_name} failed: "
            f"actual batch type={type(batch)!r}, expected default format=(x, y), "
            f"details={cause!r}. {guidance}"
        )
    else:
        message = (
            f"Could not interpret batch as the default format (x, y): "
            f"actual batch type={type(batch)!r}, batch={batch!r}. {guidance}"
        )
    return UnsupportedBatchFormatError(message)


def extract_input(
    batch: object, input_getter: Callable[[object], object] | None = None
) -> object:
    """design_spec.md 9.4 を参照し、batch から入力を抽出する。

    `input_getter` 未指定時は batch を `(x, y)` とみなし `batch[0]` を入力とする。
    抽出に失敗した場合は `UnsupportedBatchFormatError` を送出する。
    """
    if input_getter is not None:
        try:
            return input_getter(batch)
        except Exception as exc:
            raise _unsupported_batch_format_error(
                batch, getter_name="input_getter", cause=exc
            ) from exc
    try:
        return batch[0]  # type: ignore[index]
    except (TypeError, IndexError, KeyError) as exc:
        raise _unsupported_batch_format_error(batch) from exc


def extract_target(
    batch: object, target_getter: Callable[[object], object] | None = None
) -> object:
    """design_spec.md 9.4 を参照し、batch から target を抽出する。

    `target_getter` 未指定時は batch を `(x, y)` とみなし `batch[1]` を target とする。
    抽出に失敗した場合は `UnsupportedBatchFormatError` を送出する。
    """
    if target_getter is not None:
        try:
            return target_getter(batch)
        except Exception as exc:
            raise _unsupported_batch_format_error(
                batch, getter_name="target_getter", cause=exc
            ) from exc
    try:
        return batch[1]  # type: ignore[index]
    except (TypeError, IndexError, KeyError) as exc:
        raise _unsupported_batch_format_error(batch) from exc


def to_numpy(x: Array) -> np.ndarray:
    """design_spec.md 11.5 / 要件仕様 F-09 を参照し、`Array` を `np.ndarray` に変換する。

    `torch.Tensor` は forward hook 実装 (design_spec.md 6 節) と同様に
    `detach().cpu().numpy()` の順で変換する。`np.ndarray` はそのまま返す。
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    raise TypeError(f"x must be a torch.Tensor or np.ndarray: {type(x)!r}")


def _input_dimension_mismatch_error(
    x: Array, input_dim: int
) -> InputDimensionMismatchError:
    """要件仕様 7.7 / design_spec.md 15.5 が求めるメッセージ要素を含む例外を組み立てる。

    メッセージには以下を含める。
    - 抽出した `input_dim`
    - 実入力 shape
    - 期待する shape `(batch_size, input_dim)`
    - モデル/データの組み合わせ見直しまたは flatten / 特徴抽出の案内
    """
    return InputDimensionMismatchError(
        f"Input dimension mismatch: extracted input_dim={input_dim}, "
        f"actual input shape={x.shape}, expected shape=(batch_size, {input_dim}). "
        "Reconsider the model/data combination, "
        "or flatten / extract features before passing data to the adapter."
    )


def validate_input_shape(x: Array, input_dim: int) -> Array:
    """design_spec.md 9.5 / 15.4 / 15.5 を参照した入力 shape 検証・正規化。

    - `x.ndim == 1`: 単一サンプル `(input_dim,)` とみなし `(1, input_dim)` に reshape する。
      要件仕様 4.1 のエラーハンドリング原則(入力次元不一致には明示的な例外を送出する)に基づき、
      長さが `input_dim` と異なる場合は `InputDimensionMismatchError` を送出する。
    - `x.ndim == 2`: `x.shape[1] == input_dim` を検証する。
    - `x.ndim >= 3`: 自動 flatten せず `UnsupportedInputShapeError` を送出する。

    例外メッセージには要件仕様 7.3/7.7・design_spec.md 15.4/15.5 が求める必須要素
    (実際の shape、期待する shape `(batch_size, input_dim)`、回避策の案内) を含める。
    """
    if x.ndim == 1:
        if x.shape[0] != input_dim:
            raise _input_dimension_mismatch_error(x, input_dim)
        return x.reshape(1, input_dim)
    if x.ndim == 2:
        if x.shape[1] != input_dim:
            raise _input_dimension_mismatch_error(x, input_dim)
        return x
    raise UnsupportedInputShapeError(
        f"Unsupported input shape: actual shape={x.shape}, "
        f"expected shape=(batch_size, {input_dim}). Automatic flattening is not performed. "
        "Flatten or extract features before passing data to the adapter."
    )


def collect_predict_samples(
    data: Array | DataLoader[object] | Dataset[object],
    input_dim: int,
    input_getter: Callable[[object], object] | None = None,
    dataloader_kwargs: dict[str, object] | None = None,
    torch_seed: int | None = None,
) -> np.ndarray:
    """design_spec.md 12.1 / 12.2 手順 1-2(反復可能なサンプル列への正規化・shape 検証)を行う。

    `data` が `DataLoader` / `Dataset` の場合、batch から以下の優先順で入力を抽出する。

    1. `input_getter` が指定されていれば、それを使う。
    2. 未指定かつ batch 自体が `Array` (`Tensor` / `np.ndarray`) であれば、
       batch はそのまま入力(ラベル無し batch)とみなす。batch が `(x, y)` のような
       tuple / list / dict になり得るのは、`Array` 自体ではなく `Array` を要素として
       含むコンテナの場合のみであるため、この判定に矛盾は生じない。
    3. それ以外は 9.4 節の既定形式 `(x, y)` とみなし `batch[0]` を入力として扱う。

    `dataloader_kwargs` は `data` が `Dataset` の場合のみ `build_dataloader()` に渡す
    (`data` が `DataLoader` の場合は無視される)。

    戻り値は常に shape `(N, input_dim)` の 2D `np.ndarray` である。
    """
    if isinstance(data, (torch.Tensor, np.ndarray)):
        return validate_input_shape(to_numpy(data), input_dim)  # type: ignore[return-value]

    loader = build_dataloader(data, dataloader_kwargs=dataloader_kwargs, torch_seed=torch_seed)
    samples: list[np.ndarray] = []
    for batch in loader:
        if input_getter is None and isinstance(batch, (torch.Tensor, np.ndarray)):
            x = to_numpy(batch)
        else:
            # extract_input() は batch 形式に依存するため object を返す。input_getter
            # 未指定時は design_spec.md 9.4 の既定形式 (x, y) を前提に
            # Array (Tensor | np.ndarray) として扱う。
            x = to_numpy(extract_input(batch, input_getter))  # type: ignore[arg-type]
        samples.append(validate_input_shape(x, input_dim))  # type: ignore[arg-type]
    return np.concatenate(samples, axis=0)
