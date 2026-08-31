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

"""Artifact directory の保存・復元。

design_spec.md 14. Artifact 保存・ロード設計 / requirements.md F-06 を参照。
"""

from __future__ import annotations

import json
import pickle
import platform
from datetime import datetime
from importlib.metadata import version as _package_version
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import samact
import sklearn
import torch
from sklearn.preprocessing import MinMaxScaler

from samact_model_adapter.config import LearningPropertyConfig, SAMACTConfig
from samact_model_adapter.exceptions import (
    ArtifactLoadError,
    ArtifactPersistenceError,
    SAMACTModelAdapterError,
)

if TYPE_CHECKING:
    from samact_model_adapter.adapter import SAMACTModelAdapter

MODEL_FILENAME = "samact_model.h5"
INPUT_SCALER_FILENAME = "input_scaler.pkl"
TARGET_SCALER_FILENAME = "target_scaler.pkl"
METADATA_FILENAME = "metadata.json"

_TRUST_NOTICE = "Only load artifacts from a trusted source."
_SAVE_REMEDIATION = "Check write permissions and available disk space."


def _decoder_type_for(mode: str, task: str) -> str:
    """design_spec.md 7.5 / 8.3 に従い decoder 種別を導出する(metadata 記録用)。"""
    if mode == "compression":
        return "neuron_focus"
    if task == "regression":
        return "neuron_focus"
    return "majority"


def _build_metadata(
    adapter: "SAMACTModelAdapter", *, target_scaler_file: str | None
) -> dict[str, object]:
    """design_spec.md 14.3 の metadata 項目を adapter の現在の状態から組み立てる。

    encoder_type == "neuron_focus" の場合、config.py / SAMACTConfig.__post_init__ が
    n_neuron_per_exp_var を必須とするため、14.3 の一覧にはないがロード時の再構築に
    必要な n_neuron_per_exp_var も追加で保存する(14.3 は「少なくとも以下を含める」
    という open-ended な一覧であるため矛盾しない)。
    """
    learning_property = adapter.config.learning_property or LearningPropertyConfig()

    metadata: dict[str, object] = {
        "adapter_version": _package_version("samact_model_adapter"),
        "mode": adapter.mode,
        "task": adapter.task,
        "input_dim": adapter.input_dim,
    }
    if adapter.mode == "compression":
        metadata["bottleneck_layer_name"] = adapter.bottleneck_layer_name
        metadata["bottleneck_dim"] = adapter.bottleneck_dim
    else:
        metadata["output_dim"] = adapter.output_dim

    metadata["encoder_type"] = adapter.config.encoder_type
    if adapter.config.encoder_type == "neuron_focus":
        metadata["n_neuron_per_exp_var"] = adapter.config.n_neuron_per_exp_var
    metadata["decoder_type"] = _decoder_type_for(adapter.mode, adapter.task)
    metadata["n_output_multiplier"] = adapter.config.n_output_multiplier
    metadata["hidden_units"] = adapter.config.hidden_units
    metadata["tC"] = adapter.config.tC
    metadata["learning_property"] = {
        "eta": learning_property.eta,
        "iota": learning_property.iota,
        "decay_period": learning_property.decay_period,
    }
    samlayer_seed = (adapter.config.samlayer_kwargs or {}).get("seed")
    if samlayer_seed is not None:
        metadata["samlayer_seed"] = samlayer_seed
    metadata["learning_mode"] = "online"
    metadata["clip_scaled_values"] = adapter.config.clip_scaled_values
    metadata["samact_model_file"] = MODEL_FILENAME
    metadata["input_scaler_file"] = INPUT_SCALER_FILENAME
    if target_scaler_file is not None:
        metadata["target_scaler_file"] = target_scaler_file
    metadata["created_at"] = datetime.now().astimezone().isoformat()
    metadata["python_version"] = platform.python_version()
    metadata["torch_version"] = torch.__version__
    metadata["numpy_version"] = np.__version__
    metadata["scikit_learn_version"] = sklearn.__version__
    metadata["samact_version"] = samact.__version__
    return metadata


def save_artifact(
    adapter: "SAMACTModelAdapter", path: str | Path, overwrite: bool = False
) -> None:
    """design_spec.md 14.1 / 14.2 に従い artifact directory を保存する。

    保存対象 adapter が mode / task に応じた必須フィールドを満たしていない場合は、
    ここで再検証せず SAMACTModelAdapter.__post_init__ の invariant に委ねる
    (adapter は必ず invariant を満たした状態で存在するため)。
    """
    artifact_dir = Path(path)
    if artifact_dir.exists() and not overwrite:
        raise FileExistsError(
            f"artifact directory already exists: {artifact_dir}. "
            "Pass overwrite=True to overwrite it."
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / MODEL_FILENAME
    try:
        adapter.samact_model.Save(str(model_path))
    except Exception as exc:
        raise ArtifactPersistenceError(
            f"Failed to save the SAMACT model: {model_path}. Details={exc!r}. {_SAVE_REMEDIATION}"
        ) from exc

    input_scaler_path = artifact_dir / INPUT_SCALER_FILENAME
    try:
        with open(input_scaler_path, "wb") as f:
            pickle.dump(adapter.input_scaler, f)
    except Exception as exc:
        raise ArtifactPersistenceError(
            f"Failed to save input_scaler: {input_scaler_path}. Details={exc!r}. "
            f"{_SAVE_REMEDIATION}"
        ) from exc

    target_scaler_file: str | None = None
    if adapter.target_scaler is not None:
        target_scaler_path = artifact_dir / TARGET_SCALER_FILENAME
        try:
            with open(target_scaler_path, "wb") as f:
                pickle.dump(adapter.target_scaler, f)
        except Exception as exc:
            raise ArtifactPersistenceError(
                f"Failed to save target_scaler: {target_scaler_path}. Details={exc!r}. "
                f"{_SAVE_REMEDIATION}"
            ) from exc
        target_scaler_file = TARGET_SCALER_FILENAME

    metadata = _build_metadata(adapter, target_scaler_file=target_scaler_file)
    metadata_path = artifact_dir / METADATA_FILENAME
    try:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        raise ArtifactPersistenceError(
            f"Failed to save metadata.json: {metadata_path}. Details={exc!r}. "
            f"{_SAVE_REMEDIATION}"
        ) from exc


def _require_file(artifact_dir: Path, key: str, filename: object) -> str:
    if not isinstance(filename, str) or not (artifact_dir / filename).is_file():
        raise ArtifactLoadError(
            f"A required file listed in metadata was not found: "
            f"artifact directory path={artifact_dir}, "
            f"missing or inconsistent file={key}={filename!r}. "
            f"{_TRUST_NOTICE}"
        )
    return filename


def _build_config_from_metadata(metadata: dict[str, object]) -> SAMACTConfig:
    config_kwargs: dict[str, object] = {
        "tC": metadata.get("tC"),
        "hidden_units": metadata.get("hidden_units"),
        "encoder_type": metadata.get("encoder_type"),
        "n_output_multiplier": metadata.get("n_output_multiplier"),
        "clip_scaled_values": metadata.get("clip_scaled_values"),
    }
    if metadata.get("encoder_type") == "neuron_focus":
        config_kwargs["n_neuron_per_exp_var"] = metadata.get("n_neuron_per_exp_var")

    learning_property_dict = metadata.get("learning_property")
    if isinstance(learning_property_dict, dict):
        config_kwargs["learning_property"] = LearningPropertyConfig(
            eta=learning_property_dict["eta"],
            iota=learning_property_dict["iota"],
            decay_period=learning_property_dict["decay_period"],
        )

    samlayer_seed = metadata.get("samlayer_seed")
    if samlayer_seed is not None:
        config_kwargs["samlayer_kwargs"] = {"seed": samlayer_seed}

    return SAMACTConfig(**config_kwargs)  # type: ignore[arg-type]


def _read_metadata_json(artifact_dir: Path) -> dict[str, object]:
    metadata_path = artifact_dir / METADATA_FILENAME
    if not metadata_path.is_file():
        raise ArtifactLoadError(
            f"metadata.json was not found: artifact directory path={artifact_dir}, "
            f"missing or inconsistent file={METADATA_FILENAME}. {_TRUST_NOTICE}"
        )
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactLoadError(
            f"Failed to read metadata.json: artifact directory path={artifact_dir}. "
            f"Details={exc!r}. {_TRUST_NOTICE}"
        ) from exc
    if not isinstance(metadata, dict):
        raise ArtifactLoadError(
            f"metadata.json content is invalid: artifact directory path={artifact_dir}, "
            f"metadata={metadata!r}. {_TRUST_NOTICE}"
        )
    return metadata


def _require_input_dim(metadata: dict[str, object], artifact_dir: Path) -> int:
    input_dim = metadata.get("input_dim")
    if not isinstance(input_dim, int):
        raise ArtifactLoadError(
            f"metadata is missing input_dim or its value is invalid: "
            f"artifact directory path={artifact_dir}, input_dim={input_dim!r}. {_TRUST_NOTICE}"
        )
    return input_dim


def _load_samact_model(artifact_dir: Path, model_file: str) -> object:
    try:
        return samact.Pretrained(str(artifact_dir / model_file))
    except Exception as exc:
        raise ArtifactLoadError(
            f"Failed to restore the SAMACT model: artifact directory path={artifact_dir}, "
            f"missing or inconsistent file=samact_model_file={model_file!r}. Details={exc!r}. "
            f"{_TRUST_NOTICE}"
        ) from exc


def _load_pickled_scaler(artifact_dir: Path, key: str, filename: str) -> MinMaxScaler:
    try:
        with open(artifact_dir / filename, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        raise ArtifactLoadError(
            f"Failed to restore {key}: artifact directory path={artifact_dir}, "
            f"missing or inconsistent file={key}={filename!r}. Details={exc!r}. {_TRUST_NOTICE}"
        ) from exc


class _ArtifactFiles(NamedTuple):
    """load_artifact() 内で解決済みの artifact ファイル名一式。"""

    model_file: str
    input_scaler_file: str
    target_scaler_file: str | None


def _reconstruct_adapter(
    artifact_dir: Path,
    metadata: dict[str, object],
    input_dim: int,
    files: _ArtifactFiles,
) -> "SAMACTModelAdapter":
    # SAMACTModelAdapter を遅延importする理由: adapter.py が本モジュールの
    # save_artifact/load_artifact を呼び出すため、モジュールトップレベルでの
    # 相互importを避ける必要がある。
    from samact_model_adapter.adapter import (  # pylint: disable=import-outside-toplevel
        SAMACTModelAdapter,
    )

    samact_model = _load_samact_model(artifact_dir, files.model_file)
    input_scaler = _load_pickled_scaler(artifact_dir, "input_scaler_file", files.input_scaler_file)
    target_scaler = (
        _load_pickled_scaler(artifact_dir, "target_scaler_file", files.target_scaler_file)
        if files.target_scaler_file is not None
        else None
    )

    try:
        config = _build_config_from_metadata(metadata)
        return SAMACTModelAdapter(
            samact_model=samact_model,
            mode=metadata.get("mode"),  # type: ignore[arg-type]
            task=metadata.get("task"),  # type: ignore[arg-type]
            input_dim=input_dim,
            input_scaler=input_scaler,
            target_scaler=target_scaler,
            config=config,
            metadata=metadata,
            output_dim=metadata.get("output_dim"),  # type: ignore[arg-type]
            bottleneck_layer_name=metadata.get("bottleneck_layer_name"),  # type: ignore[arg-type]
            bottleneck_dim=metadata.get("bottleneck_dim"),  # type: ignore[arg-type]
        )
    except SAMACTModelAdapterError as exc:
        raise ArtifactLoadError(
            f"Failed to reconstruct SAMACTModelAdapter from metadata: "
            f"artifact directory path={artifact_dir}. Details={exc!r}. {_TRUST_NOTICE}"
        ) from exc


def load_artifact(path: str | Path) -> "SAMACTModelAdapter":
    """design_spec.md 14.4 に従い artifact directory から SAMACTModelAdapter を復元する。

    mode / task に応じた必須フィールド(target_scaler、bottleneck_layer_name、
    bottleneck_dim、output_dim 等)の充足検証は、ここで再実装せず
    SAMACTConfig / SAMACTModelAdapter の __post_init__ invariant に委ね、
    そこで送出される SAMACTModelAdapterError を ArtifactLoadError へ変換する。
    """
    artifact_dir = Path(path)
    if not artifact_dir.is_dir():
        raise ArtifactLoadError(
            f"artifact directory does not exist: artifact directory path={artifact_dir}. "
            f"{_TRUST_NOTICE}"
        )

    metadata = _read_metadata_json(artifact_dir)
    input_dim = _require_input_dim(metadata, artifact_dir)

    model_file = _require_file(artifact_dir, "samact_model_file", metadata.get("samact_model_file"))
    input_scaler_file = _require_file(
        artifact_dir, "input_scaler_file", metadata.get("input_scaler_file")
    )
    target_scaler_file = metadata.get("target_scaler_file")
    if target_scaler_file is not None:
        target_scaler_file = _require_file(artifact_dir, "target_scaler_file", target_scaler_file)

    files = _ArtifactFiles(
        model_file=model_file,
        input_scaler_file=input_scaler_file,
        target_scaler_file=target_scaler_file,
    )
    return _reconstruct_adapter(artifact_dir, metadata, input_dim, files)
