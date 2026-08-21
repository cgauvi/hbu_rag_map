"""The four ways this app finds its database, and the order they are tried in.

Worth testing because the order is the contract: a developer with a container
running and AWS credentials in the shell must get the container, and someone
with nothing set must get the SSM lookup rather than a confusing localhost
refusal.
"""

from __future__ import annotations

import pytest

from src.utils import db


def test_database_url_wins_over_everything(monkeypatch):
    monkeypatch.setenv("DATABASE_URL",
                       "postgresql://u:p@localhost:5433/urban_rag?sslmode=disable")
    monkeypatch.setenv("URBAN_RAG_PG_HOST", "should-be-ignored.rds.amazonaws.com")

    resolved = db.resolve()

    assert resolved.host == "localhost"
    assert resolved.port == 5433
    assert resolved.sslmode == "disable"
    assert resolved.source == "DATABASE_URL"


def test_database_url_percent_decodes_credentials(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://urban%40rag:p%40ss%2Fword@host:5432/db"
    )
    resolved = db.resolve()
    assert resolved.user == "urban@rag"
    assert resolved.password == "p@ss/word"


def test_database_url_rejects_a_non_postgres_scheme(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "mysql://u:p@localhost/db")
    with pytest.raises(db.DbError, match="postgresql://"):
        db.resolve()


def test_urban_rag_env_with_a_literal_password(monkeypatch):
    monkeypatch.setenv("URBAN_RAG_PG_HOST", "hbu-dev.rds.amazonaws.com")
    monkeypatch.setenv("URBAN_RAG_PG_PASSWORD", "secret")
    monkeypatch.setenv("URBAN_RAG_PG_SSLMODE", "require")

    resolved = db.resolve()

    assert resolved.host == "hbu-dev.rds.amazonaws.com"
    assert resolved.user == "urban_rag"
    assert resolved.password == "secret"
    assert resolved.sslmode == "require"
    assert not resolved.iam_auth


def test_urban_rag_env_reads_a_secrets_manager_id(monkeypatch):
    monkeypatch.setenv("URBAN_RAG_PG_HOST", "hbu-dev.rds.amazonaws.com")
    monkeypatch.setenv("URBAN_RAG_PG_SECRET_ID", "arn:aws:secretsmanager:...:app")
    monkeypatch.setenv("URBAN_RAG_PG_SSLMODE", "require")
    monkeypatch.setattr(db, "_secret_password", lambda secret_id, region: "from-secrets")

    resolved = db.resolve()

    assert resolved.password == "from-secrets"
    assert "Secrets Manager" in resolved.source


def test_iam_auth_defers_the_password(monkeypatch):
    """A signed token is minted per connection, never at resolution time."""
    monkeypatch.setenv("URBAN_RAG_PG_HOST", "hbu-dev.rds.amazonaws.com")
    monkeypatch.setenv("URBAN_RAG_PG_IAM_AUTH", "1")
    monkeypatch.setenv("URBAN_RAG_PG_SSLMODE", "require")

    resolved = db.resolve()

    assert resolved.iam_auth
    assert resolved.password == ""


def test_a_credential_is_required(monkeypatch):
    monkeypatch.setenv("URBAN_RAG_PG_HOST", "hbu-dev.rds.amazonaws.com")
    monkeypatch.setenv("URBAN_RAG_PG_SSLMODE", "require")
    with pytest.raises(db.DbError, match="no credential"):
        db.resolve()


def test_verify_full_reports_the_missing_ca_bundle(monkeypatch, tmp_path):
    """libpq's own message names a file nobody has heard of; this one does not."""
    monkeypatch.setenv("URBAN_RAG_PG_HOST", "hbu-dev.rds.amazonaws.com")
    monkeypatch.setenv("URBAN_RAG_PG_PASSWORD", "secret")
    monkeypatch.setenv("URBAN_RAG_PG_SSLROOTCERT", str(tmp_path / "absent.crt"))

    with pytest.raises(db.DbError, match="make db-ca"):
        db.resolve()


def test_verify_full_accepts_a_bundle_that_exists(monkeypatch, tmp_path):
    bundle = tmp_path / "root.crt"
    bundle.write_text("-----BEGIN CERTIFICATE-----")
    monkeypatch.setenv("URBAN_RAG_PG_HOST", "hbu-dev.rds.amazonaws.com")
    monkeypatch.setenv("URBAN_RAG_PG_PASSWORD", "secret")
    monkeypatch.setenv("URBAN_RAG_PG_SSLROOTCERT", str(bundle))

    resolved = db.resolve()

    assert resolved.sslmode == "verify-full"
    assert resolved.sslrootcert == str(bundle)


def test_ssm_is_the_fallback_when_nothing_is_set(monkeypatch):
    calls = {}

    def fake_ssm(project, env, region):
        calls.update(project=project, env=env, region=region)
        return db.Connection(host="from-ssm", password="x", source="SSM")

    monkeypatch.setattr(db, "_from_ssm", fake_ssm)

    assert db.resolve().host == "from-ssm"
    assert calls == {"project": "hbu", "env": "dev", "region": db.DEFAULT_REGION}


def test_missing_ssm_parameters_name_the_apply_that_creates_them(monkeypatch):
    class _Paginator:
        def paginate(self, **_kwargs):
            return [{"Parameters": []}]

    class _Client:
        def get_paginator(self, _name):
            return _Paginator()

    monkeypatch.setattr(db, "_boto", lambda service, region: _Client())

    with pytest.raises(db.DbError, match="make apply ENV=dev"):
        db._from_ssm("hbu", "dev", "us-east-1")


def test_url_hides_the_password_by_default():
    connection = db.Connection(host="h", password="hunter2", user="urban_rag")
    assert "hunter2" not in connection.url()
    assert "hunter2" in connection.url(hide_password=False)


def test_kwargs_omit_sslrootcert_when_unset():
    assert "sslrootcert" not in db.Connection(host="h").kwargs()
    assert db.Connection(host="h", sslrootcert="/x").kwargs()["sslrootcert"] == "/x"


@pytest.mark.parametrize(
    "value,expected",
    [("1", True), ("true", True), ("TRUE", True), ("yes", True), ("on", True),
     ("0", False), ("false", False), ("", False), (None, False)],
)
def test_flag_parsing(value, expected):
    assert db._flag(value) is expected


def test_the_pool_opens_its_connections_in_autocommit(monkeypatch):
    """psycopg_pool discards any connection its configure callback leaves in a
    transaction, and two SETs on a non-autocommit connection do exactly that."""
    captured = {}

    class _Pool:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def close(self):
            pass

    import sys
    import types

    module = types.ModuleType("psycopg_pool")
    module.ConnectionPool = _Pool
    monkeypatch.setitem(sys.modules, "psycopg_pool", module)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost/db?sslmode=disable")
    db.close_pool()
    try:
        db.get_pool()
        assert captured["kwargs"]["autocommit"] is True
    finally:
        db.close_pool()
