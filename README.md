# SAMACT Model Adapter

[![PyPI Version](https://img.shields.io/pypi/v/samact_model_adapter.svg)](https://pypi.org/project/samact_model_adapter/)

学習済み PyTorch モデルの入出力(または中間層の出力)を教師データとして、[PySAMACT](https://github.com/maviss-design/SAMACT)(SAMACT Framework)のモデルを再学習し、同じ振る舞いを再現することを目指すアダプタ層です。

***

## 概要 (Overview)

本ライブラリは、以下を目的として開発されています。

* 既存の学習済み PyTorch モデルを、そのまま SAMACT へ載せ替えたときの振る舞いを **PoC(概念実証)レベルで評価する**
* PyTorch モデルの重みやアーキテクチャをそのまま変換するのではなく、**入出力(または中間層出力)の対応関係を教師データとして SAMACT 側を学習し直す**

主に次の 2 つのモードを提供します。

* **standalone モード**: 元モデルと同じ入出力形状になるよう SAMACT モデルを一から学習する(分類 / 回帰)。
* **compression モード**: 既存モデルの中間層(ボトルネック層)の出力を教師信号として、特徴量圧縮用の SAMACT モデルを学習する。

> ⚠️ 本ライブラリが依存する PySAMACT (SAMACT Framework) 自体が PoC 用途のライブラリであり、高速な学習・高性能推論を目的としていません。本ライブラリも同様です。

***

## 特徴 (Features)

* standalone / compression 2 モードによる SAMACT モデルの構築・学習
* 学習済み adapter を artifact directory として保存・復元(`save_artifact()` / `load_from_artifact()`)
* `evaluate()` による元 PyTorch モデルと SAMACT モデルの予測性能比較(accuracy / MSE / MAE 等)
* `DataLoader` / `Dataset` / `Tensor` / `np.ndarray` など柔軟な入力形式に対応

***

## 非対象 (Non-Goals)

以下は明示的にスコープ外です。

* PyTorch モデルの重み・中間構造・活性化関数の SAMACT への変換移植
* `nn.Conv*` / `nn.RNN` / `nn.LSTM` / `nn.GRU` / Attention・Transformer 系レイヤーを含むモデルへの対応(現状 `nn.Linear` ベースの MLP のみ対応)
* GPU 対応や高速化最適化
* 自動微分などの学習アルゴリズム自体の変更

***

## インストール (Installation)

```bash
pip install samact_model_adapter
```

* `torch` / `numpy` / `scikit-learn` / `samact`(PySAMACT)が依存パッケージとして解決されます。
* Python 3.12 以上が必要です。

***

## 想定ユースケース (Use Cases)

* 学習済み PyTorch モデルを SAMACT 対応ハードウェアへ載せ替える前段の PoC 評価
* 既存モデルの中間表現(ボトルネック層出力)を SAMACT でどの程度近似できるかの検証(特徴量圧縮)
* 分類 / 回帰タスクを SAMACT 単体モデルとして再学習した場合の、元モデルとの精度比較

***

## API仕様書 (API Document)

API 仕様書は下記をご覧ください。

https://maviss-design.github.io/samact_model_adapter/

Quickstart ガイドおよび `api` / `adapter` / `config` / `exceptions` 各モジュールのリファレンスを参照できます。

***

## 制限事項 (Limitations)

* 対応するモデルは `nn.Linear` 層を 2 層以上持つ MLP のみです(Conv / RNN / Attention 系のレイヤーを含むモデルは非対応)
* 実行速度は最適化されていません(依存先の PySAMACT 自体が PoC 用途のため)
* artifact のロードには pickle を使用するため、信頼できる artifact のみを読み込んでください

***

## ライセンス (License)

AGPLv3 です。詳細は LICENSE ファイルを参照してください。商用ライセンスに関するお問い合わせは solution-sales@maviss-design.com までご連絡ください。

***

## 注意事項

本ライブラリは **PoC・研究検証用途** を目的としています。
商用品質・量産用途への直接利用は想定していません。

***

# English

# SAMACT Model Adapter

An adapter layer that retrains a [PySAMACT](https://github.com/maviss-design/SAMACT) (SAMACT Framework) model using the input/output (or intermediate-layer output) of a trained PyTorch model as teacher data, aiming to reproduce the same behavior.

***

## Overview

This library is developed for the following purposes:

* To **evaluate, at a PoC (Proof of Concept) level**, how a trained PyTorch model would behave if migrated to SAMACT as-is
* Rather than directly converting the PyTorch model's weights and architecture, it **retrains the SAMACT side using the input/output (or intermediate-layer output) correspondence as teacher data**

It mainly provides two modes:

* **standalone mode**: Build and train a SAMACT model from scratch so that it has the same input/output shape as the original model (classification / regression).
* **compression mode**: Train a SAMACT model for feature compression, using the output of an intermediate layer (bottleneck layer) of an existing model as the teacher signal.

> ⚠️ PySAMACT (SAMACT Framework), which this library depends on, is itself a PoC-oriented library and is not intended for high-speed training or high-performance inference. The same applies to this library.

***

## Features

* Building and training SAMACT models via two modes: standalone / compression
* Saving and restoring a trained adapter as an artifact directory (`save_artifact()` / `load_from_artifact()`)
* Comparing prediction performance between the original PyTorch model and the SAMACT model via `evaluate()` (accuracy / MSE / MAE, etc.)
* Support for flexible input formats: `DataLoader` / `Dataset` / `Tensor` / `np.ndarray`
* Automatic API documentation generation from docstrings (Google Style) and Sphinx autodoc

***

## Non-Goals

The following are explicitly out of scope:

* Converting a PyTorch model's weights, internal structure, or activation functions into SAMACT
* Support for models containing `nn.Conv*` / `nn.RNN` / `nn.LSTM` / `nn.GRU` / Attention / Transformer layers (currently only `nn.Linear`-based MLPs are supported)
* GPU support or performance optimization
* Changes to the learning algorithm itself, such as automatic differentiation

***

## Installation

```bash
pip install samact_model_adapter
```

* Dependencies `torch` / `numpy` / `scikit-learn` / `samact` (PySAMACT) are resolved automatically.
* Python 3.12 or later is required.

***

## Use Cases

* PoC evaluation prior to migrating a trained PyTorch model to SAMACT-compatible hardware
* Verifying how well SAMACT can approximate an existing model's intermediate representation (bottleneck layer output) via feature compression
* Comparing accuracy against the original model when a classification / regression task is retrained as a standalone SAMACT model

***

## API Documentation

Please refer to the following for the API documentation:

https://maviss-design.github.io/samact_model_adapter/

It includes the Quickstart guide and the reference for the `api` / `adapter` / `config` / `exceptions` modules, generated from docstrings (Google Style) via Sphinx autodoc.

> `docs/` also contains internal-only documents other than the API reference (requirements, design specification, etc.); only the API reference above is published.

***

## Limitations

* Only MLPs with at least two `nn.Linear` layers are supported (models containing Conv / RNN / Attention layers are not supported)
* Execution speed is not optimized (since the underlying PySAMACT is itself intended for PoC use)
* Artifact loading uses pickle, so only load artifacts from a trusted source

***

## License

AGPLv3. See the LICENSE file for full details. For commercial licensing options, contact solution-sales@maviss-design.com.

***

## Notes

This library is intended for **PoC and research/validation purposes**.
It is not designed for direct use in production-quality or mass-production applications.

***
