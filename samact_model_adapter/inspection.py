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

"""PyTorch モデルから nn.Linear 層を列挙し、入出力次元を抽出するユーティリティ。"""

import logging
from dataclasses import dataclass

from torch import nn

from samact_model_adapter.exceptions import (
    BottleneckDetectionError,
    UnsupportedArchitectureError,
)

_logger = logging.getLogger(__name__)

# F-10 / design_spec.md 15.3: 検出した場合に UnsupportedArchitectureError を送出する非対応レイヤー型。
_UNSUPPORTED_LAYER_TYPES: tuple[type[nn.Module], ...] = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.RNN,
    nn.LSTM,
    nn.GRU,
    nn.MultiheadAttention,
    nn.Transformer,
    nn.TransformerEncoder,
    nn.TransformerDecoder,
)

_MLP_ONLY_NOTE = "The current version only supports MLP / nn.Linear-based models."
_WORKAROUND_UNSUPPORTED_LAYER = (
    "Replace it with an nn.Linear-based model that does not contain unsupported layers, "
    "or pre-extract/transform the unsupported layer's output into features "
    "and pass it as an nn.Linear-based model."
)
_WORKAROUND_TOO_FEW_LINEAR = (
    "Pass a model with at least two nn.Linear layers so that the input and output "
    "dimensions can be extracted."
)
_BOTTLENECK_NO_CANDIDATES_NOTE = (
    "There are no bottleneck layer candidates "
    "(no nn.Linear layers remain after excluding the output layer candidate). "
)


@dataclass(frozen=True)
class LinearLayerDims:
    """Linear 層列挙から抽出した入出力次元。"""

    input_dim: int
    output_dim: int


@dataclass(frozen=True)
class BottleneckLayerInfo:
    """ボトルネック層解決の結果。"""

    layer_name: str
    in_features: int
    out_features: int
    excluded_output_layer_name: str | None


def validate_supported_architecture(model: nn.Module) -> None:
    """非対応レイヤー（Conv/RNN/Attention 系）を検出した場合に UnsupportedArchitectureError を送出する。"""
    for name, module in model.named_modules():
        if isinstance(module, _UNSUPPORTED_LAYER_TYPES):
            layer_type = type(module).__name__
            raise UnsupportedArchitectureError(
                f"Detected an unsupported layer: name='{name}', type={layer_type}. "
                f"{_MLP_ONLY_NOTE}{_WORKAROUND_UNSUPPORTED_LAYER}"
            )


def enumerate_linear_layers(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    """model.named_modules() から nn.Linear 層のみを列挙順(=定義順)で返す。"""
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]


def extract_linear_dims(model: nn.Module) -> LinearLayerDims:
    """design_spec.md 8.2: Linear 層列挙と入出力次元抽出。

    1. 非対応レイヤーの検証を行う。
    2. nn.Linear 層を列挙し、2層未満の場合は UnsupportedArchitectureError を送出する。
    3. 先頭 Linear 層の in_features を input_dim、末尾 Linear 層の out_features を output_dim とする。
    """
    validate_supported_architecture(model)

    linear_layers = enumerate_linear_layers(model)
    if len(linear_layers) < 2:
        raise UnsupportedArchitectureError(
            f"Only {len(linear_layers)} nn.Linear layer(s) were detected "
            "(at least 2 layers are required). "
            f"{_MLP_ONLY_NOTE}{_WORKAROUND_TOO_FEW_LINEAR}"
        )

    _, first_layer = linear_layers[0]
    _, last_layer = linear_layers[-1]
    return LinearLayerDims(
        input_dim=first_layer.in_features,
        output_dim=last_layer.out_features,
    )


def resolve_bottleneck_layer(
    model: nn.Module, bottleneck_layer_name: str | None = None
) -> BottleneckLayerInfo:
    """design_spec.md 7.2 / requirements.md F-01: ボトルネック層の自動検出・手動指定。

    非対応アーキテクチャの検証は自動検出・手動指定のどちらでも必ず実行する。
    `bottleneck_layer_name` を指定した場合は自動検出をスキップし、指定層をそのまま採用する。
    """
    validate_supported_architecture(model)

    if bottleneck_layer_name is not None:
        return _resolve_manual_bottleneck_layer(model, bottleneck_layer_name)
    return _resolve_auto_bottleneck_layer(model)


def _resolve_manual_bottleneck_layer(
    model: nn.Module, bottleneck_layer_name: str
) -> BottleneckLayerInfo:
    layer = dict(model.named_modules()).get(bottleneck_layer_name)
    if not isinstance(layer, nn.Linear):
        raise BottleneckDetectionError(
            f"The specified layer '{bottleneck_layer_name}' was not found or is not an nn.Linear."
        )

    info = BottleneckLayerInfo(
        layer_name=bottleneck_layer_name,
        in_features=layer.in_features,
        out_features=layer.out_features,
        excluded_output_layer_name=None,
    )
    _log_bottleneck_layer(info)
    return info


def _resolve_auto_bottleneck_layer(model: nn.Module) -> BottleneckLayerInfo:
    linear_layers = enumerate_linear_layers(model)
    # 最後の1層を出力層候補として除外するため、2層未満では候補が0件になる。
    if len(linear_layers) < 2:
        raise BottleneckDetectionError(
            f"{_BOTTLENECK_NO_CANDIDATES_NOTE}{_MLP_ONLY_NOTE}{_WORKAROUND_TOO_FEW_LINEAR}"
        )

    excluded_output_layer_name, _ = linear_layers[-1]
    candidates = linear_layers[:-1]
    # out_features が同値の場合は、より深い(後に出現した)層を採用する。
    bottleneck_name, bottleneck_layer = candidates[0]
    for name, layer in candidates[1:]:
        if layer.out_features <= bottleneck_layer.out_features:
            bottleneck_name, bottleneck_layer = name, layer

    info = BottleneckLayerInfo(
        layer_name=bottleneck_name,
        in_features=bottleneck_layer.in_features,
        out_features=bottleneck_layer.out_features,
        excluded_output_layer_name=excluded_output_layer_name,
    )
    _log_bottleneck_layer(info)
    return info


def _log_bottleneck_layer(info: BottleneckLayerInfo) -> None:
    _logger.info(
        "Detected bottleneck layer: name='%s', in_features=%d, out_features=%d, "
        "excluded output layer candidate='%s'",
        info.layer_name,
        info.in_features,
        info.out_features,
        info.excluded_output_layer_name,
    )
