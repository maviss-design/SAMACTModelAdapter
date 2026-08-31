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

"""SAMACTConfig / LearningPropertyConfig / FitResult の定義と検証。"""

from dataclasses import dataclass

from samact_model_adapter.exceptions import InvalidSAMACTConfigError

_VALID_ENCODER_TYPES = ("rate", "neuron_focus")


@dataclass
class LearningPropertyConfig:
    """PySAMACT の学習パラメータ設定。

    ``SAMACTConfig.learning_property`` に指定すると、SAMACT モデル構築時に
    PySAMACT の ``LearningProperty(eta=..., iota=..., decayPeriod=...)`` に
    変換して使用される。

    Attributes:
        eta: PySAMACT ``LearningProperty`` の ``eta`` に対応する値。
        iota: PySAMACT ``LearningProperty`` の ``iota`` に対応する値。
        decay_period: PySAMACT ``LearningProperty`` の ``decayPeriod`` に
            対応する値。
    """

    eta: int = 7
    iota: int = 4
    decay_period: int = 5


@dataclass
class SAMACTConfig:  # pylint: disable=too-many-instance-attributes
    """``compress_from_pytorch()`` / ``translate_from_pytorch()`` に渡す、
    SAMACT モデルの構築・学習設定。

    Attributes:
        tC: PySAMACT ``Sequential.Compile(tC)`` に渡す値。
        hidden_units: SAMLayer 中間層のユニット数。encoder の ``nUnits`` とは
            区別される。
        epochs: PySAMACT ``Fit()`` に渡す学習エポック数。
        encoder_type: 使用する encoder の種類。``"rate"`` または
            ``"neuron_focus"`` のいずれかを指定する。
        n_neuron_per_exp_var: ``encoder_type="neuron_focus"`` の場合に必須の
            パラメータ。``NeuronFocusEncodeLayer`` の ``nNeuronPerExpVar`` に
            対応する。
        learning_property: PySAMACT の ``LearningProperty`` に変換する設定。
            未指定の場合は PySAMACT の既定値を使用する。
        samlayer_kwargs: PySAMACT ``SAMLayer`` コンストラクタへそのまま渡す
            追加引数。存在しないキーを指定すると
            ``InvalidSAMACTConfigError`` が送出される。``samProperty`` /
            ``weightDist`` を指定しない場合、PySAMACT 自体の既定値では
            なく、より高い学習精度が得られることを確認済みの値
            (``LayerProperty(a=3, p=0.75)`` / ``RandomProperty("kaiming",
            "normal")``)が既定値として使用される。また、デコーダ直前の
            最終 SAMLayer では、``teacherSignalActivate`` を指定しない
            場合に限り PySAMACT の既定値 ``Step()`` の代わりに ``Linear()``
            が使用される。
        clip_scaled_values: ``True`` の場合、scaler の transform 後の値を
            ``[0, 1]`` に clip する。
        torch_seed: 指定した場合、``loader`` / ``val_loader`` が
            ``Dataset`` として渡され、内部で ``DataLoader`` を構築する
            際の shuffle 用 ``torch.Generator`` の seed として使用される。
            ``loader`` / ``val_loader`` に ``DataLoader`` を直接渡した
            場合や、``dataloader_kwargs`` に ``generator`` を指定した
            場合は使用されない。
        n_output_multiplier: 最終 SAMLayer のユニット数に掛ける倍率。1 以上
            の整数を指定する。分類タスクでは無視される。

    Raises:
        InvalidSAMACTConfigError: ``encoder_type`` が不正な場合、
            ``encoder_type="neuron_focus"`` で ``n_neuron_per_exp_var`` が
            未指定の場合、``n_output_multiplier`` が 1 未満の場合に送出される。
    """

    tC: int = 32  # pylint: disable=invalid-name  # PySAMACT Sequential.Compile(tC) に準拠
    hidden_units: int = 64
    epochs: int = 10
    encoder_type: str = "rate"
    n_neuron_per_exp_var: int | None = None
    learning_property: LearningPropertyConfig | None = None
    samlayer_kwargs: dict[str, object] | None = None
    clip_scaled_values: bool = True
    torch_seed: int | None = None
    n_output_multiplier: int = 1

    def __post_init__(self) -> None:
        if self.encoder_type not in _VALID_ENCODER_TYPES:
            raise InvalidSAMACTConfigError(
                f"Invalid value specified for encoder_type: {self.encoder_type!r}. "
                f"Valid values are {_VALID_ENCODER_TYPES}."
            )
        if self.encoder_type == "neuron_focus" and self.n_neuron_per_exp_var is None:
            raise InvalidSAMACTConfigError(
                "n_neuron_per_exp_var is required when encoder_type='neuron_focus'."
            )
        if self.n_output_multiplier < 1:
            raise InvalidSAMACTConfigError(
                "n_output_multiplier must be an integer greater than or equal to 1: "
                f"specified value={self.n_output_multiplier!r}"
            )


@dataclass
class FitResult:
    """学習結果のサマリ。``SAMACTModelAdapter.fit_result`` に保持される。

    Attributes:
        epochs: 学習に使用したエポック数。
        elapsed_seconds: 学習にかかった時間(秒)。
        mse_per_epoch: 特徴量圧縮モードおよびスタンドアロン回帰で使用する、
            エポックごとの学習データに対する MSE のリスト。それ以外の
            場合は ``None``。
        accuracy_per_epoch: スタンドアロン分類で使用する、エポックごとの
            学習データに対する accuracy のリスト。それ以外の場合は
            ``None``。
    """

    epochs: int
    elapsed_seconds: float
    mse_per_epoch: list[float] | None = None
    accuracy_per_epoch: list[float] | None = None
