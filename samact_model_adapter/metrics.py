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

"""
MSE / RMSE / MAE / accuracyを算出する。また、モデル間の分類4象限集計、予測不一致率算出を担当する
accuracyはscikit-learnのaccuracy_scoreを用いる

"""

# ============== import ============
from numpy.typing import ArrayLike,NDArray
import numpy as np
from sklearn.metrics import accuracy_score

# ============== error messages ===============
_NUMERIC_VALUE_ERROR = "y_true and y_pred must contain numeric values."
_NAN_ERROR = "y_true and y_pred must not contain NaN values."
_SHAPE_ERROR = "y_true and y_pred must have the same shape."

# ============ validate ============
def validate(*arrays: ArrayLike) -> tuple[NDArray[np.float64], ...]:

    # 確認のみだと冗長になるため、値も返すメソッドにしている
    try:
        y_na = tuple(
            np.asarray(array, dtype=float) for array in arrays
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(_NUMERIC_VALUE_ERROR) from exc

     # This class must be called after to_numeric_array method
    if any(np.isnan(array).any() for array in y_na):
        raise ValueError(_NAN_ERROR)

    # This class must be called after to_numeric_array method

    shape = y_na[0].shape
    if any(array.shape != shape for array in y_na[1:]):
        raise ValueError(_SHAPE_ERROR)

    return y_na

# ============== regression ============
def calc_mse(y_pred:ArrayLike,y_true:ArrayLike) -> float:

    y_true_na, y_pred_na = validate(y_true, y_pred)

    return float(np.mean((y_pred_na - y_true_na) ** 2 ))


def calc_rmse(y_pred:ArrayLike, y_true:ArrayLike) -> float:
    rmse = calc_mse(y_pred, y_true)
    return float(np.sqrt(rmse))


def calc_mae(y_pred:ArrayLike, y_true:ArrayLike) -> float:

    y_true_na, y_pred_na = validate(y_true, y_pred)

    error = y_pred_na - y_true_na
    return float(np.mean(np.abs(error)))

# ============ classification ============
def calc_accuracy(y_pred:ArrayLike, y_true:ArrayLike) -> float:

    y_true_na, y_pred_na = validate(y_true, y_pred)

    return float(accuracy_score(y_true_na, y_pred_na))


def calc_correct_summary(basemodel_pred:ArrayLike,
               samact_pred:ArrayLike,
                    y_true:ArrayLike
                 ) -> dict[str, int]:

    basemodel_pred_na, samact_pred_na, y_true_na = validate(
        basemodel_pred,
          samact_pred,
            y_true)

    # 正解の集合を作成(True or False)し、論理演算で判定
    basemodel_correct = np.asarray(basemodel_pred_na == y_true_na)
    samact_correct = np.asarray(samact_pred_na == y_true_na)
    correct_summary = {
        "both" : int((basemodel_correct & samact_correct).sum()),
        "basemodel_only" : int((basemodel_correct & ~samact_correct).sum()),
        "samact_only" : int((~basemodel_correct & samact_correct).sum()),
        "neither" : int((~basemodel_correct & ~samact_correct).sum())
    }

    return correct_summary

# protectedにしていない理由は、予測一致率が必要になった場合のためです。
# 予測一致率を見たい場合も考えられるため。
def model_agreement_rate(basemodel_pred:ArrayLike, samact_pred:ArrayLike) -> float:

    basemodel_pred_na, samact_pred_na =  validate(basemodel_pred, samact_pred)

    match_basemodel_samact = np.asarray(basemodel_pred_na == samact_pred_na)

    return float(np.mean(match_basemodel_samact))

def model_disagreement_rate(basemodel_pred:ArrayLike, samact_pred:ArrayLike) -> float:
    return 1 - model_agreement_rate(basemodel_pred, samact_pred)
