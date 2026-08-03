from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def require_env(values: dict[str, str | None]) -> None:
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")


@dataclass(frozen=True)
class LakeFSRuntimeConfig:
    endpoint: str
    access_key_id: str
    secret_access_key: str
    repository: str
    artifact_ref: str
    artifact_object: str
    artifact_dir: Path
    clean: bool
    skip_download: bool

    @classmethod
    def from_env(cls) -> "LakeFSRuntimeConfig":
        artifact_ref = os.getenv("QSENTIA_ARTIFACT_REF") or os.getenv("LAKEFS_BRANCH", "")
        artifact_object = os.getenv("QSENTIA_ARTIFACT_OBJECT", "")
        skip_download = bool_env("QSENTIA_SKIP_ARTIFACT_DOWNLOAD", False)
        if not skip_download:
            require_env(
                {
                    "LAKEFS_ENDPOINT": os.getenv("LAKEFS_ENDPOINT"),
                    "LAKEFS_ACCESS_KEY_ID": os.getenv("LAKEFS_ACCESS_KEY_ID"),
                    "LAKEFS_SECRET_ACCESS_KEY": os.getenv("LAKEFS_SECRET_ACCESS_KEY"),
                    "LAKEFS_REPOSITORY": os.getenv("LAKEFS_REPOSITORY"),
                    "QSENTIA_ARTIFACT_REF": artifact_ref,
                    "QSENTIA_ARTIFACT_OBJECT": artifact_object,
                }
            )
        return cls(
            endpoint=os.getenv("LAKEFS_ENDPOINT", "").rstrip("/"),
            access_key_id=os.getenv("LAKEFS_ACCESS_KEY_ID", ""),
            secret_access_key=os.getenv("LAKEFS_SECRET_ACCESS_KEY", ""),
            repository=os.getenv("LAKEFS_REPOSITORY", "qsentia-models"),
            artifact_ref=artifact_ref.strip("/") or "main",
            artifact_object=artifact_object.strip("/"),
            artifact_dir=Path(os.getenv("QSENTIA_ARTIFACT_DIR", "/app/artifacts/current")),
            clean=bool_env("QSENTIA_ARTIFACT_CLEAN", False),
            skip_download=skip_download,
        )

    @property
    def storage_options(self) -> dict:
        return {
            "key": self.access_key_id,
            "secret": self.secret_access_key,
            "client_kwargs": {"endpoint_url": self.endpoint},
        }

    @property
    def object_uri(self) -> str:
        return f"s3://{self.repository}/{self.artifact_ref}/{self.artifact_object}"


@dataclass(frozen=True)
class SignalRuntimeConfig:
    output_path: Path
    append_signal_log: bool
    use_ppo_policy: bool
    require_ppo_policy: bool
    target_gross_exposure: float

    @classmethod
    def from_env(cls) -> "SignalRuntimeConfig":
        explicit = os.getenv("QSENTIA_SIGNAL_OUTPUT", "").strip()
        output_path = Path(explicit) if explicit else Path(os.getenv("QSENTIA_OUTPUT_DIR", "/app/outputs")) / "latest_signal.json"
        return cls(
            output_path=output_path,
            append_signal_log=bool_env("QSENTIA_APPEND_SIGNAL_LOG", True),
            use_ppo_policy=bool_env("QSENTIA_USE_PPO_POLICY", True),
            require_ppo_policy=bool_env("QSENTIA_REQUIRE_PPO_POLICY", True),
            target_gross_exposure=float_env("QSENTIA_TARGET_GROSS_EXPOSURE", 1.0),
        )

