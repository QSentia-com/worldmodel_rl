from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from zipfile import ZipFile

try:
    import s3fs
except ImportError as exc:  # pragma: no cover
    s3fs = None
    _S3FS_IMPORT_ERROR = exc
else:  # pragma: no cover
    _S3FS_IMPORT_ERROR = None

from .config import LakeFSRuntimeConfig


EXPECTED_SELECTED_MODEL = "v12b_small_rl_guardian"
EXPECTED_MODEL_VERSION = "v12b_exact_same_leg_inverse_plus_small_rl_guardian_v1"
REQUIRED_ARTIFACT_FILES = (
    "artifact_manifest.json",
    "model_metadata.json",
    "live_state.json",
    "deployment_action_mapping.json",
    "live_order_map.csv",
    "selected_decisions_live.csv",
    "small_rl_guardian_actions.csv",
    "small_rl_guardian_summary.csv",
)


def download_lakefs_artifacts(config: LakeFSRuntimeConfig) -> list[Path]:
    if config.skip_download:
        validate_artifacts(config.artifact_dir)
        return sorted(path for path in config.artifact_dir.rglob("*") if path.is_file())

    if "world_rl/" not in config.artifact_object:
        raise RuntimeError(
            "WORLD_MODEL-RL must use the world_rl artifact path; "
            f"refusing artifact object {config.artifact_object!r}."
        )

    filesystem = _s3_filesystem(config)
    if config.clean and config.artifact_dir.exists():
        shutil.rmtree(config.artifact_dir)
    config.artifact_dir.mkdir(parents=True, exist_ok=True)

    object_uri = config.object_uri.rstrip("/")
    if object_uri.lower().endswith(".zip"):
        if not filesystem.exists(object_uri):
            raise RuntimeError(f"lakeFS artifact object does not exist: {object_uri}")
        zip_path = config.artifact_dir / ".artifact_bundle.zip"
        filesystem.get(object_uri, str(zip_path))
        extracted = extract_artifact_zip(zip_path, config.artifact_dir)
        zip_path.unlink(missing_ok=True)
    else:
        extracted = download_artifact_prefix(filesystem, object_uri, config.artifact_dir)

    validate_artifacts(config.artifact_dir)
    return extracted


def extract_artifact_zip(zip_path: Path, artifact_dir: Path) -> list[Path]:
    extracted: list[Path] = []
    with ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"Unsafe path in artifact zip: {member.filename}")
            if member.is_dir():
                continue
            destination = artifact_dir / member_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(destination)
    return extracted


def download_artifact_prefix(filesystem: Any, prefix_uri: str, artifact_dir: Path) -> list[Path]:
    files = [path for path in filesystem.find(prefix_uri) if not str(path).endswith("/")]
    if not files:
        raise RuntimeError(f"lakeFS artifact prefix has no files: {prefix_uri}")

    prefix = prefix_uri.rstrip("/") + "/"
    extracted: list[Path] = []
    for remote in files:
        relative = str(remote)
        if relative.startswith(prefix):
            relative = relative[len(prefix) :]
        else:
            relative = Path(relative).name
        if not relative:
            continue
        destination = artifact_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        filesystem.get(remote, str(destination))
        extracted.append(destination)
    return extracted


def validate_artifacts(artifact_dir: Path) -> None:
    missing = [relative for relative in REQUIRED_ARTIFACT_FILES if not (artifact_dir / relative).exists()]
    if missing:
        missing_list = "\n".join(f"- {relative}" for relative in missing)
        raise RuntimeError(f"Downloaded WORLD_MODEL-RL artifacts are incomplete. Missing:\n{missing_list}")

    manifest = _read_json(artifact_dir / "artifact_manifest.json")
    metadata = _read_json(artifact_dir / "model_metadata.json")
    live_state = _read_json(artifact_dir / "live_state.json")
    mapping = _read_json(artifact_dir / "deployment_action_mapping.json")
    for source_name, payload in {
        "artifact_manifest.json": manifest,
        "model_metadata.json": metadata,
        "live_state.json": live_state,
        "deployment_action_mapping.json": mapping,
    }.items():
        _require_exact_artifact_identity(source_name, payload)


def _require_exact_artifact_identity(source_name: str, payload: dict[str, Any]) -> None:
    selected_model = str(payload.get("selected_model") or "")
    model_version = str(payload.get("model_version") or "")
    if selected_model != EXPECTED_SELECTED_MODEL or model_version != EXPECTED_MODEL_VERSION:
        raise RuntimeError(
            f"{source_name} is not the requested WORLD_MODEL-RL artifact. "
            f"Expected {EXPECTED_SELECTED_MODEL}/{EXPECTED_MODEL_VERSION}, "
            f"got {selected_model}/{model_version}."
        )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return payload


def _s3_filesystem(config: LakeFSRuntimeConfig) -> Any:
    if s3fs is None:
        raise RuntimeError("Install s3fs to download artifacts from lakeFS.") from _S3FS_IMPORT_ERROR
    return s3fs.S3FileSystem(**config.storage_options)
