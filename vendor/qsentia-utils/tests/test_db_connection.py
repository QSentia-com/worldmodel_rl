import json

from qsentia_utils.db.connection import DatabaseConfig, database_config_from_env


def test_database_config_dsn():
    config = DatabaseConfig(
        host="postgres.local",
        port=5432,
        database="qsentia",
        username="qsentia_user",
        password="secret",
    )

    assert config.dsn == "postgresql://qsentia_user:secret@postgres.local:5432/qsentia?sslmode=require"
    assert config.connection_kwargs["dbname"] == "qsentia"


def test_database_config_from_env_secret_json(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_SECRET_JSON",
        json.dumps(
            {
                "host": "postgres.local",
                "port": 5432,
                "dbname": "qsentia",
                "username": "qsentia_user",
                "password": "secret",
            }
        ),
    )

    config = database_config_from_env()

    assert config.host == "postgres.local"
    assert config.database == "qsentia"
    assert config.username == "qsentia_user"
    assert config.password == "secret"
