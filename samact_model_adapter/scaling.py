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

"""MinMax scaling utilities.

See design_spec.md section 10 for the scaling and clipping policy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.preprocessing import MinMaxScaler

FloatArray = NDArray[np.floating[Any]]


def fit_minmax_scaler(values: ArrayLike) -> MinMaxScaler:
    """Fit a MinMaxScaler using the project's fixed feature range."""
    # ガードは不要。MinMaxScalerに任せるため。
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(values)
    return scaler


def transform_with_scaler(
    scaler: MinMaxScaler,
    values: ArrayLike,
    *,
    clip: bool,
) -> FloatArray:
    """Transform values and apply the project's optional clipping policy."""
    # ガードは不要。MinMaxScalerに任せるため。
    transformed: FloatArray = scaler.transform(values)
    if clip:
        return np.clip(transformed, 0.0, 1.0)
    return transformed
