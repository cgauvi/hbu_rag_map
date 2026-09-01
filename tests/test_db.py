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


def test_secret_password_names_tls_failure_before_iam(monkeypatch):
    class _Client:
        def get_secret_value(self, **_kwargs):
            raise RuntimeError(
                "SSL validation failed for https://secretsmanager.us-east-1.amazonaws.com/ "
                "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"
            )

    monkeypatch.setattr(db, "_boto", lambda service, region: _Client())

    with pytest.raises(db.DbError) as err:
        db._secret_password("arn:aws:secretsmanager:...:app", "us-east-1")

    message = str(err.value)
    assert "before IAM was checked" in message
    assert "AWS_CA_BUNDLE" in message
    assert "needs secretsmanager:GetSecretValue" not in message


def test_secret_password_names_iam_when_not_a_tls_failure(monkeypatch):
    class _Client:
        def get_secret_value(self, **_kwargs):
            raise RuntimeError("AccessDeniedException")

    monkeypatch.setattr(db, "_boto", lambda service, region: _Client())

    with pytest.raises(db.DbError, match="secretsmanager:GetSecretValue"):
        db._secret_password("arn:aws:secretsmanager:...:app", "us-east-1")


def test_boto_can_reuse_a_valid_ssl_cert_file(monkeypatch, tmp_path):
    bundle = tmp_path / "corporate-plus-certifi.pem"
    bundle.write_text("-----BEGIN CERTIFICATE-----")
    monkeypatch.delenv("AWS_CA_BUNDLE", raising=False)
    monkeypatch.delenv("URBAN_RAG_AWS_CA_BUNDLE", raising=False)
    monkeypatch.setenv("SSL_CERT_FILE", str(bundle))

    assert db._aws_ca_bundle() == str(bundle)


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
    monkeypatch.setattr(db, "DEFAULT_CA_BUNDLE", tmp_path / "also-absent.crt")

    with pytest.raises(db.DbError, match="make db-ca"):
        db.resolve()


def test_an_absent_configured_bundle_falls_back_to_the_default(monkeypatch, tmp_path):
    """One .env serves the container and a native run, and the path differs.

    The container carries the bundle at /etc/ssl/certs/rds-global-bundle.pem;
    a native run has it where `make db-ca` writes it. Reading the container's
    path on a host that has the other one is a configuration mismatch, not a
    reason to refuse — both files are Amazon's RDS root bundle.
    """
    default = tmp_path / "root.crt"
    default.write_text("-----BEGIN CERTIFICATE-----")
    monkeypatch.setattr(db, "DEFAULT_CA_BUNDLE", default)
    monkeypatch.setenv("URBAN_RAG_PG_HOST", "hbu-dev.rds.amazonaws.com")
    monkeypatch.setenv("URBAN_RAG_PG_PASSWORD", "secret")
    monkeypatch.setenv(
        "URBAN_RAG_PG_SSLROOTCERT", "/etc/ssl/certs/rds-global-bundle.pem"
    )

    resolved = db.resolve()

    assert resolved.sslrootcert == str(default)


def test_resolve_is_memoised_so_a_pan_costs_no_aws_calls(monkeypatch):
    """Two GetSecretValue per query — one from `_Borrowed`, one from `get_pool`.

    A map pan issues five queries, so the uncached path put ten AWS round
    trips between a user's drag and the first row.
    """
    calls = []
    monkeypatch.setenv("URBAN_RAG_PG_HOST", "hbu-dev.rds.amazonaws.com")
    monkeypatch.setenv("URBAN_RAG_PG_SECRET_ID", "arn:aws:secretsmanager:...:app")
    monkeypatch.setenv("URBAN_RAG_PG_SSLMODE", "require")
    monkeypatch.setattr(
        db, "_secret_password",
        lambda secret_id, region: (calls.append(secret_id), "pw")[1],
    )

    for _ in range(10):
        db.resolve()

    assert len(calls) == 1


def test_a_changed_endpoint_is_not_served_from_the_memo(monkeypatch):
    """The memo must not defeat `get_pool`'s reopen-on-endpoint-change."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@first/db?sslmode=disable")
    assert db.resolve().host == "first"

    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@second/db?sslmode=disable")
    assert db.resolve().host == "second"


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


def test_hostaddr_lets_verify_full_survive_the_tunnel(monkeypatch):
    """`make run-tunnel`'s configuration: endpoint name, loopback address.

    The certificate RDS presents is issued to the endpoint, so `host` has to go
    on naming it for the hostname check while the socket goes to the tunnel.
    Collapsing the two - pointing `host` at 127.0.0.1 - is what used to force
    the `sslmode=require` downgrade.
    """
    monkeypatch.setenv("URBAN_RAG_PG_HOST", "hbu-dev.rds.amazonaws.com")
    monkeypatch.setenv("URBAN_RAG_PG_HOSTADDR", "127.0.0.1")
    monkeypatch.setenv("URBAN_RAG_PG_PORT", "5433")
    monkeypatch.setenv("URBAN_RAG_PG_PASSWORD", "secret")
    monkeypatch.setenv("URBAN_RAG_PG_SSLMODE", "require")

    resolved = db.resolve()

    assert resolved.host == "hbu-dev.rds.amazonaws.com"
    assert resolved.hostaddr == "127.0.0.1"
    assert resolved.port == 5433

    kwargs = resolved.kwargs()
    assert kwargs["host"] == "hbu-dev.rds.amazonaws.com"
    assert kwargs["hostaddr"] == "127.0.0.1"


def test_kwargs_omit_hostaddr_when_unset():
    """The direct path must not pass hostaddr at all, not even as None."""
    assert "hostaddr" not in db.Connection(host="h").kwargs()
    assert db.Connection(host="h", hostaddr="127.0.0.1").kwargs()["hostaddr"] == "127.0.0.1"


def test_blank_hostaddr_is_treated_as_unset(monkeypatch):
    """docker-run-tunnel clears variables with `-e NAME=`, which arrives as ""."""
    monkeypatch.setenv("URBAN_RAG_PG_HOST", "hbu-dev.rds.amazonaws.com")
    monkeypatch.setenv("URBAN_RAG_PG_HOSTADDR", "")
    monkeypatch.setenv("URBAN_RAG_PG_PASSWORD", "secret")
    monkeypatch.setenv("URBAN_RAG_PG_SSLMODE", "require")

    resolved = db.resolve()

    assert resolved.hostaddr is None
    assert "hostaddr" not in resolved.kwargs()


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
