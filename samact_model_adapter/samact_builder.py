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

"""PySAMACT encoder / decoder / SAMLayer / Sequential ファクトリ。

design_spec.md 11.1 encoder 選択、11.2 SAMLayer、11.3 LearningProperty、
11.4 Compile を参照。

7.5 SAMACT モデル構築(特徴量圧縮モード)・8.3 タスク別 SAMACT 構築(スタンドアロン
モード)が要求する「最終 SAMLayer のユニット数」(bottleneck_dim / output_dim を
config.n_output_multiplier 倍した値)の算出は本 Issue のスコープ外であり、
学習フロー実装 Issue で実装する。
"""

from __future__ import annotations

from samact import (
    LayerProperty,
    LearningProperty,
    Linear,
    MajorityDecodeLayer,
    NeuronFocusDecodeLayer,
    NeuronFocusEncodeLayer,
    RandomProperty,
    RateEncodeLayer,
    SAMLayer,
    Sequential,
)
from samact.ModelBuilder.BuildContract import (
    IBuildableDecoder,
    IBuildableEncoder,
    IBuildableNeuralLayer,
)
from samact.SAMACTFramework.TrainingContract import ILearningProperty

from samact_model_adapter.config import SAMACTConfig
from samact_model_adapter.exceptions import InvalidSAMACTConfigError

# design_spec.md 11.2: SAMLayer コンストラクタのうち nUnits を除くキー。
# nUnits は build_samlayer() の units 引数で決まるため samlayer_kwargs には含められない。
_SAMLAYER_VALID_KWARGS = frozenset(
    {
        "samProperty",
        "gradientActivate",
        "teacherSignalActivate",
        "initW",
        "initTheta",
        "weightDist",
        "thetaDist",
        "seed",
    }
)


def build_encoder(config: SAMACTConfig, input_dim: int) -> IBuildableEncoder:
    """design_spec.md 11.1: config.encoder_type に応じた PySAMACT encoder を構築する。

    encoder_type / n_neuron_per_exp_var の妥当性は SAMACTConfig.__post_init__ が
    構築時に保証済みであるため、ここでは再検証しない(config.py のテストで担保する)。
    """
    if config.encoder_type == "neuron_focus":
        # SAMACTConfig.__post_init__ により neuron_focus では None にならない前提。
        # 型チェッカー向けの絞り込みであり、ユーザー入力の再検証ではない。
        assert config.n_neuron_per_exp_var is not None
        return NeuronFocusEncodeLayer(
            nUnits=input_dim, nNeuronPerExpVar=config.n_neuron_per_exp_var
        )
    return RateEncodeLayer(nUnits=input_dim)


def build_majority_decoder() -> IBuildableDecoder:
    """design_spec.md 8.3(分類): 分類タスク用の decoder を構築する。"""
    return MajorityDecodeLayer()


def build_neuron_focus_decoder(n_objectives: int) -> IBuildableDecoder:
    """design_spec.md 8.3(回帰) / 7.5(特徴量圧縮)を参照。

    スタンドアロン回帰では `n_objectives` に `output_dim` を、
    特徴量圧縮モードでは `bottleneck_dim` を渡す。
    """
    return NeuronFocusDecodeLayer(nObjectives=n_objectives)


def build_samlayer(config: SAMACTConfig, units: int, *, is_final: bool = False) -> SAMLayer:
    """design_spec.md 11.2: SAMLayer を構築する。

    `units` は構築する SAMLayer の `nUnits` を表す。design_spec.md 11.2 の
    `hidden_units`(中間層)と 7.5/8.3 の「最終 SAMLayer のユニット数」の
    どちらを渡すかは呼び出し側の責務とし、本関数はユニット数のみを受け取る。

    `is_final=True` の場合、デコーダ直前の最終 SAMLayer として扱い、
    `samlayer_kwargs` に `teacherSignalActivate` が指定されていなければ
    既定値として `Linear()` を設定する。PySAMACT の既定値である `Step()`
    (スパイク量子化)は連続値を期待するデコーダ(`MajorityDecodeLayer` /
    `NeuronFocusDecodeLayer`)に渡す最終層の教師信号活性化関数としては
    不適切なため。`is_final=False`(既定・中間層)の場合、
    `teacherSignalActivate` は PySAMACT の既定値(`Step()`)のまま使う。

    `samProperty` / `weightDist` は、`samlayer_kwargs` に指定がなければ
    実験的検証により PySAMACT 自体の既定値より高い学習精度が得られることを
    確認済みの値(`LayerProperty(a=3, p=0.75)` / `RandomProperty("kaiming",
    "normal")`)を既定値として使う。中間層・最終層のいずれにも適用される。

    `config.samlayer_kwargs` が指定された場合、SAMLayer コンストラクタ引数として
    渡す。不明なキーが指定された場合、`InvalidSAMACTConfigError` を送出する。
    ユーザーが `samlayer_kwargs` で明示的に指定した値は、上記のいずれの
    既定値注入よりも常に優先される。
    """
    kwargs = dict(config.samlayer_kwargs or {})
    unknown_keys = set(kwargs) - _SAMLAYER_VALID_KWARGS
    if unknown_keys:
        raise InvalidSAMACTConfigError(
            f"Invalid keys specified in samlayer_kwargs: {sorted(unknown_keys)!r}. "
            f"Valid keys are {sorted(_SAMLAYER_VALID_KWARGS)!r}."
        )
    kwargs.setdefault("samProperty", LayerProperty(a=3, p=0.75))
    kwargs.setdefault("weightDist", RandomProperty("kaiming", "normal"))
    if is_final:
        kwargs.setdefault("teacherSignalActivate", Linear())
    return SAMLayer(nUnits=units, **kwargs)


def build_learning_property(config: SAMACTConfig) -> ILearningProperty:
    """design_spec.md 11.3: config.learning_property を PySAMACT の LearningProperty に変換する。

    未指定の場合は PySAMACT の既定値を使用する。
    """
    if config.learning_property is None:
        return LearningProperty()
    return LearningProperty(
        eta=config.learning_property.eta,
        iota=config.learning_property.iota,
        decayPeriod=config.learning_property.decay_period,
    )


def build_sequential(
    config: SAMACTConfig,
    encoder: IBuildableEncoder,
    decoder: IBuildableDecoder,
    layers: list[IBuildableNeuralLayer],
) -> Sequential:
    """design_spec.md 11.4: Sequential を構築し、Sequential.Compile(config.tC) を呼び出す。"""
    sequential = Sequential(encoder=encoder, decoder=decoder, layers=layers)
    sequential.Compile(config.tC)
    return sequential
