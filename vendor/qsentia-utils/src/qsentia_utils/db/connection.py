from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    database: str
    username: str
    password: str
    port: int = 5432
    sslmode: str = "require"

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.username}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}?sslmode={self.sslmode}"
        )

    @property
    def connection_kwargs(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.database,
            "user": self.username,
            "password": self.password,
            "sslmode": self.sslmode,
        }


def database_config_from_env() -> DatabaseConfig:
    secret = _load_database_secret_from_env()

    host = _env_or_secret("DATABASE_HOST", secret, "host")
    database = _env_or_secret("DATABASE_NAME", secret, "dbname", "database")
    username = _env_or_secret("DATABASE_USER", secret, "username", "user")
    password = _env_or_secret("DATABASE_PASSWORD", secret, "password")
    port = int(os.getenv("DATABASE_PORT") or secret.get("port") or 5432)
    sslmode = os.getenv("DATABASE_SSLMODE", "require")

    missing = [
        name
        for name, value in {
            "DATABASE_HOST": host,
            "DATABASE_NAME": database,
            "DATABASE_USER": username,
            "DATABASE_PASSWORD": password,
        }.items()
        if not value
    ]
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(f"Missing database configuration: {missing_text}")

    return DatabaseConfig(
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
        sslmode=sslmode,
    )


def connect(config: DatabaseConfig | None = None):
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "Database connections require the db extra: pip install 'qsentia-utils[db]'"
        ) from exc

    resolved_config = config or database_config_from_env()
    return psycopg.connect(**resolved_config.connection_kwargs)


@contextmanager
def open_connection(config: DatabaseConfig | None = None) -> Iterator[Any]:
    connection = connect(config)
    try:
        yield connection
    finally:
        connection.close()


def _load_database_secret_from_env() -> dict[str, Any]:
    secret_json = os.getenv("DATABASE_SECRET_JSON")
    if secret_json:
        return json.loads(secret_json)

    secret_arn = os.getenv("DATABASE_SECRET_ARN")
    if not secret_arn:
        return {}

    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "DATABASE_SECRET_ARN requires boto3: pip install 'qsentia-utils[aws]' "
            "or 'qsentia-utils[db]'"
        ) from exc

    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    client = boto3.client("secretsmanager", region_name=region)
    response = client.get_secret_value(SecretId=secret_arn)
    return json.loads(response["SecretString"])


def _env_or_secret(env_name: str, secret: dict[str, Any], *secret_names: str) -> str:
    env_value = os.getenv(env_name)
    if env_value:
        return env_value

    for secret_name in secret_names:
        secret_value = secret.get(secret_name)
        if secret_value:
            return str(secret_value)

    return ""
