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

class SAMACTModelAdapterError(Exception):
    """本パッケージが送出する固有例外の基底クラス。"""


class BottleneckDetectionError(SAMACTModelAdapterError):
    """ボトルネック層の自動検出または検証に失敗した場合に送出される。

    ``bottleneck_layer_name`` 未指定時に出力層候補を除いたボトルネック候補が
    存在しない場合、指定した層名がモデルに存在しない場合、指定した層が
    ``nn.Linear`` ではない場合に送出される。
    """


class UnsupportedArchitectureError(SAMACTModelAdapterError):
    """PyTorch モデルに未対応のアーキテクチャが含まれる場合に送出される。

    ``nn.Conv1d`` / ``nn.Conv2d`` / ``nn.Conv3d`` / ``nn.RNN`` / ``nn.LSTM`` /
    ``nn.GRU`` / ``nn.MultiheadAttention`` / ``nn.Transformer`` /
    ``nn.TransformerEncoder`` / ``nn.TransformerDecoder`` のいずれかを検出した
    場合、またはスタンドアロンモードで ``nn.Linear`` 層が 2 層未満の場合に
    送出される。
    """


class UnsupportedInputShapeError(SAMACTModelAdapterError):
    """入力が 3 次元以上であり、自動 flatten できない場合に送出される。"""


class ArtifactLoadError(SAMACTModelAdapterError):
    """artifact の読み込みに失敗した場合に送出される。

    artifact directory または ``metadata.json`` が存在しない場合、
    ``metadata.json`` を JSON として読み込めない場合、metadata に記載された
    必須ファイルが存在しない場合、``mode`` / ``task`` に応じた必須 metadata
    が不足している場合、metadata と実ファイルの整合性が取れない場合に送出される。
    """


class ArtifactPersistenceError(SAMACTModelAdapterError):
    """artifact の保存に失敗した場合に送出される。

    SAMACT モデル(``.h5``)の保存、``input_scaler`` / ``target_scaler`` の
    pickle 保存、``metadata.json`` の生成・書き込みのいずれかに失敗した
    場合に送出される。保存対象の adapter が ``mode`` / ``task`` に応じた
    必須フィールドを満たしているかどうかは、``SAMACTModelAdapter`` の
    構築時点(``__post_init__``)で ``InvalidSAMACTConfigError`` により
    既に保証されているため、``save_artifact()`` 側では再検証しない。
    """


class UnsupportedDatasetError(SAMACTModelAdapterError):
    """``IterableDataset`` が渡された場合に送出される。

    scaler の fit がデータ全体を 2 回走査する方式であるため、複数回走査
    できない ``IterableDataset`` は未対応である。
    """


class InvalidSAMACTConfigError(SAMACTModelAdapterError):
    """``SAMACTConfig`` の値が不正な場合に送出される。

    ``encoder_type`` が ``"rate"`` / ``"neuron_focus"`` 以外の場合、
    ``encoder_type="neuron_focus"`` で ``n_neuron_per_exp_var`` が未指定の
    場合、``n_output_multiplier`` が 1 未満の場合、``samlayer_kwargs`` に
    PySAMACT ``SAMLayer`` コンストラクタに存在しないキーが指定された場合、
    ``translate_from_pytorch()`` の ``task`` が不正な値の場合に送出される。
    """


class InputDimensionMismatchError(SAMACTModelAdapterError):
    """入力の次元数が adapter の ``input_dim`` と一致しない場合に送出される。"""


class UnsupportedBatchFormatError(SAMACTModelAdapterError):
    """DataLoader / Dataset の batch を解釈できない場合に送出される。

    ``input_getter`` / ``target_getter`` が未指定で、batch を既定形式
    ``(x, y)`` として解釈できない場合、または指定された ``input_getter`` /
    ``target_getter`` の実行に失敗した場合に送出される。
    """
