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

"""forward hook によるボトルネック中間表現の収集。

design_spec.md 4.1 (hooks.py の責務) / 7.3 中間表現収集 / requirements.md F-02 を参照。
"""

from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from samact_model_adapter.data import extract_input


def _as_tensor(x: object) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    raise TypeError(
        f"model input must be a torch.Tensor or np.ndarray: {type(x)!r}"
    )


def collect_bottleneck_outputs(
    model: nn.Module,
    loader: DataLoader[object],
    layer_name: str,
    input_getter: Callable[[object], object] | None = None,
) -> np.ndarray:
    """design_spec.md 7.3 に従い、`layer_name` の forward hook から中間表現を収集する。

    - `layer_name` の層に forward hook を登録し、出力テンソルを `detach()`, `cpu()`,
      `numpy()` の順で変換して蓄積する。
    - hook handle は `try/finally` により、収集完了後・例外発生時のいずれでも必ず解除する。
    - 収集中は `model.eval()` を呼び出し、処理前の `model.training` を完了後・例外時に復元する。
    - forward は `torch.no_grad()` コンテキスト下で実行する。
    - 戻り値は `np.ndarray` shape `(N, bottleneck_dim)` とし、`bottleneck_dim` は
      hook 収集結果から実測する(`nn.Linear.out_features` からの静的取得は行わない)。
    """
    layer = model.get_submodule(layer_name)

    collected: list[np.ndarray] = []

    def _hook(
        _module: nn.Module, _inputs: tuple[object, ...], output: torch.Tensor
    ) -> None:
        collected.append(output.detach().cpu().numpy())

    was_training = model.training
    handle = layer.register_forward_hook(_hook)
    try:
        model.eval()
        with torch.no_grad():
            for batch in loader:
                x = extract_input(batch, input_getter)
                model(_as_tensor(x))
    finally:
        handle.remove()
        model.train(was_training)

    return np.concatenate(collected, axis=0)
