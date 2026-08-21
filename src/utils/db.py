"""
db.py — How this app reaches the Postgres that hbu_infra provisions.

Resolution order, most explicit first:

1. ``DATABASE_URL``            — a full URL. What a local postgis+pgvector
                                 container or an already-open tunnel is
                                 pointed at, and what ``make db-url`` in
                                 hbu_infra prints.
2. ``URBAN_RAG_PG_DSN``        — a full libpq string.
3. ``URBAN_RAG_PG_*``          — the dataplatform's own contract (host, port,
                                 database, user, plus a Secrets Manager id or
                                 an IAM auth flag). ``eval "$(make -s db-app-env
                                 ENV=dev)"`` in hbu_infra sets exactly these.
4. SSM ``/hbu-<env>/db/*``     — nothing set at all: discover the endpoint from
                                 the parameter contract Terraform publishes and
                                 the app-role password from Secrets Manager.

The point of the order is that the same code runs against a container, a
tunnel, and RDS without a branch in the caller — and that no endpoint or
password is ever committed. It mirrors ``hbu_infra/scripts/db.py``, which
resolves the *master* credentials for administration; this resolves the
``urban_rag`` role, because an app that only reads should connect as the role
that only reads.

Credentials are resolved **per connection**, never cached, for the same reason
the dataplatform does it: an RDS IAM auth token is signed for fifteen minutes,
so a pool that cached one would hand out an expired token on its second hour.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

logger = logging.getLogger(__name__)

#: The project/env pair the SSM contract is keyed by. `hbu_infra` writes
#: `/hbu-dev/db/...`; these are the two halves of that prefix.
DEFAULT_PROJECT = os.environ.get("HBU_PROJECT", "hbu")
DEFAULT_ENV = os.environ.get("HBU_ENV", "dev")
DEFAULT_REGION = (
    os.environ.get("URBAN_RAG_PG_REGION")
    or os.environ.get("AWS_REGION")
    or os.environ.get("AWS_DEFAULT_REGION")
    or "us-east-1"
)

#: Where `hbu_infra`'s `make db-ca` puts the RDS root certificate, and where
#: libpq looks by default — so `verify-full` works with nothing else set.
DEFAULT_CA_BUNDLE = Path.home() / ".postgresql" / "root.crt"

#: The role the pipeline writes as and this app reads as.
DEFAULT_USER = "urban_rag"
DEFAULT_DATABASE = "urban_rag"

#: Where the corpus and the geometry live. Overridable because a review copy of
#: the schema is a normal thing to point a UI at.
SCHEMA = os.environ.get("URBAN_RAG_PG_SCHEMA", "rag")


class DbError(RuntimeError):
    """Anything the operator should read rather than see a traceback for."""


# ---------------------------------------------------------------------------
# Connection details
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Connection:
    host: str
    port: int = 5432
    dbname: str = DEFAULT_DATABASE
    user: str = DEFAULT_USER
    password: str = ""
    #: `verify-full` is the only mode that authenticates the server rather than
    #: merely encrypting the link, and it is what the dataplatform defaults to.
    #: A local container speaks no TLS at all and needs `disable`, which is why
    #: this is read from the environment rather than fixed.
    sslmode: str = "verify-full"
    sslrootcert: str | None = None
    #: Set when the password has to be minted per connection (IAM auth) rather
    #: than read once.
    iam_auth: bool = False
    region: str = DEFAULT_REGION
    #: Where the password came from, for the status pane. Never the password.
    source: str = "unknown"

    def url(self, *, hide_password: bool = True) -> str:
        secret = "***" if hide_password else quote(self.password, safe="")
        return (
            f"postgresql://{quote(self.user, safe='')}:{secret}"
            f"@{self.host}:{self.port}/{self.dbname}?sslmode={self.sslmode}"
        )

    def kwargs(self) -> dict:
        params = {
            "host": self.host,
            "port": self.port,
            "dbname": self.dbname,
            "user": self.user,
            "password": self.password,
            "sslmode": self.sslmode,
        }
        if self.sslrootcert:
            params["sslrootcert"] = self.sslrootcert
        return params


# ---------------------------------------------------------------------------
# AWS lookups
# ---------------------------------------------------------------------------


def _boto(service: str, region: str):
    try:
        import boto3  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment problem
        raise DbError(
            "boto3 is not installed, so the endpoint cannot be discovered from "
            "SSM. Either `pip install boto3` or set DATABASE_URL."
        ) from exc
    return boto3.client(service, region_name=region)


def _secret_password(secret_id: str, region: str) -> str:
    """The password half of a Secrets Manager `{username, password}` payload."""
    try:
        payload = _boto("secretsmanager", region).get_secret_value(SecretId=secret_id)
    except Exception as exc:
        raise DbError(
            f"could not read {secret_id}: {exc}\n"
            "  The role running this app needs secretsmanager:GetSecretValue."
        ) from exc
    try:
        return json.loads(payload["SecretString"])["password"]
    except (KeyError, json.JSONDecodeError) as exc:
        raise DbError(
            f"{secret_id} does not look like a database secret — expected JSON "
            'with a "password" key.'
        ) from exc


def _iam_token(connection: Connection) -> str:
    """A 15-minute RDS auth token, signed now.

    Minted per connection on purpose: this is the one credential that expires
    while the process is still running.
    """
    try:
        return _boto("rds", connection.region).generate_db_auth_token(
            DBHostname=connection.host,
            Port=connection.port,
            DBUsername=connection.user,
            Region=connection.region,
        )
    except Exception as exc:
        raise DbError(f"could not sign an RDS IAM auth token: {exc}") from exc


# ---------------------------------------------------------------------------
# The four resolution paths
# ---------------------------------------------------------------------------


def _from_url(url: str, *, source: str) -> Connection:
    parsed = urlparse(url)
    if parsed.scheme not in {"postgres", "postgresql"}:
        raise DbError(f"{source} must be a postgresql:// URL, got {parsed.scheme!r}")
    query = parse_qs(parsed.query)
    return Connection(
        host=parsed.hostname or "localhost",
        port=parsed.port or 5432,
        dbname=(parsed.path or f"/{DEFAULT_DATABASE}").lstrip("/"),
        user=unquote(parsed.username or DEFAULT_USER),
        password=unquote(parsed.password or ""),
        sslmode=query.get("sslmode", ["require"])[0],
        sslrootcert=query.get("sslrootcert", [None])[0],
        source=source,
    )


def _from_dsn(dsn: str) -> Connection:
    """A libpq keyword/value string, parsed by libpq itself when available."""
    try:
        from psycopg.conninfo import conninfo_to_dict  # noqa: PLC0415

        parsed = conninfo_to_dict(dsn)
    except Exception as exc:
        raise DbError(f"URBAN_RAG_PG_DSN is not a valid libpq string: {exc}") from exc
    return Connection(
        host=parsed.get("host", "localhost"),
        port=int(parsed.get("port", 5432)),
        dbname=parsed.get("dbname", DEFAULT_DATABASE),
        user=parsed.get("user", DEFAULT_USER),
        password=parsed.get("password", ""),
        sslmode=parsed.get("sslmode", "verify-full"),
        sslrootcert=parsed.get("sslrootcert"),
        source="URBAN_RAG_PG_DSN",
    )


def _from_urban_rag_env() -> Connection:
    """The dataplatform's `URBAN_RAG_PG_*` contract."""
    region = os.environ.get("URBAN_RAG_PG_REGION", DEFAULT_REGION)
    sslmode = os.environ.get("URBAN_RAG_PG_SSLMODE", "verify-full")
    connection = Connection(
        host=os.environ["URBAN_RAG_PG_HOST"],
        port=int(os.environ.get("URBAN_RAG_PG_PORT", 5432)),
        dbname=os.environ.get("URBAN_RAG_PG_DATABASE", DEFAULT_DATABASE),
        user=os.environ.get("URBAN_RAG_PG_USER", DEFAULT_USER),
        sslmode=sslmode,
        sslrootcert=_ca_bundle(sslmode),
        region=region,
        source="URBAN_RAG_PG_*",
    )

    if _flag(os.environ.get("URBAN_RAG_PG_IAM_AUTH")):
        return replace(connection, iam_auth=True, source="URBAN_RAG_PG_* + IAM auth")

    secret_id = os.environ.get("URBAN_RAG_PG_SECRET_ID")
    if secret_id:
        return replace(
            connection,
            password=_secret_password(secret_id, region),
            source="URBAN_RAG_PG_* + Secrets Manager",
        )

    password = os.environ.get("URBAN_RAG_PG_PASSWORD")
    if password is None:
        raise DbError(
            "URBAN_RAG_PG_HOST is set but no credential is: set one of "
            "URBAN_RAG_PG_SECRET_ID, URBAN_RAG_PG_IAM_AUTH=1, or "
            "URBAN_RAG_PG_PASSWORD."
        )
    return replace(connection, password=password)


def _from_ssm(project: str, env: str, region: str) -> Connection:
    """The `/hbu-<env>/db/*` contract Terraform publishes.

    The one path that needs nothing in the environment: given AWS credentials
    and an env name, the endpoint and the app-role secret are both discoverable.
    """
    prefix = f"/{project}-{env}"
    ssm = _boto("ssm", region)

    values: dict[str, str] = {}
    try:
        paginator = ssm.get_paginator("get_parameters_by_path")
        for page in paginator.paginate(Path=prefix, Recursive=True, WithDecryption=True):
            for param in page["Parameters"]:
                values[param["Name"][len(prefix) + 1 :]] = param["Value"]
    except Exception as exc:
        raise DbError(f"could not read SSM parameters under {prefix}: {exc}") from exc

    if "db/host" not in values:
        raise DbError(
            f"no database parameters under {prefix} in {region}.\n"
            f"  Has `make apply ENV={env}` run in hbu_infra? Is AWS_PROFILE "
            f"pointing at the right account?\n"
            f"  To use a local container instead, set DATABASE_URL."
        )

    sslmode = os.environ.get("URBAN_RAG_PG_SSLMODE", "verify-full")
    connection = Connection(
        host=values["db/host"],
        port=int(values.get("db/port", 5432)),
        dbname=values.get("db/name", DEFAULT_DATABASE),
        user=os.environ.get("URBAN_RAG_PG_USER", DEFAULT_USER),
        sslmode=sslmode,
        sslrootcert=_ca_bundle(sslmode),
        region=region,
        source=f"SSM {prefix}",
    )

    # The app-role secret, not the master one — this app reads, it does not
    # administer. Terraform creates the secret; `make db-bootstrap` fills it.
    app_secret = values.get("db/app_secret_arn")
    if not app_secret:
        raise DbError(
            f"{prefix}/db/app_secret_arn is missing. Run `make db-bootstrap "
            f"ENV={env}` in hbu_infra, which creates the {DEFAULT_USER} role "
            f"and stores its password."
        )
    return replace(
        connection,
        password=_secret_password(app_secret, region),
        source=f"SSM {prefix} + Secrets Manager",
    )


def _ca_bundle(sslmode: str) -> str | None:
    """The root certificate `verify-full` needs, checked before libpq asks.

    libpq's own message for a missing bundle names a file most people have
    never heard of, so this fails with the command that creates it instead.
    """
    if sslmode != "verify-full":
        return None
    override = os.environ.get("URBAN_RAG_PG_SSLROOTCERT") or os.environ.get("PGSSLROOTCERT")
    path = Path(override).expanduser() if override else DEFAULT_CA_BUNDLE
    if not path.exists():
        raise DbError(
            f"sslmode=verify-full needs Amazon's CA bundle, and {path} does not "
            f"exist.\n"
            f"  make db-ca            (in hbu_infra)\n"
            f"  or set URBAN_RAG_PG_SSLMODE=require to encrypt without "
            f"authenticating the server."
        )
    return str(path)


def resolve(env: str | None = None, *, project: str | None = None,
            region: str | None = None) -> Connection:
    """Connection details, by the four-step order documented at the top."""
    url = os.environ.get("DATABASE_URL")
    if url:
        return _from_url(url, source="DATABASE_URL")

    dsn = os.environ.get("URBAN_RAG_PG_DSN")
    if dsn:
        return _from_dsn(dsn)

    if os.environ.get("URBAN_RAG_PG_HOST"):
        return _from_urban_rag_env()

    return _from_ssm(
        project or DEFAULT_PROJECT,
        env or DEFAULT_ENV,
        region or DEFAULT_REGION,
    )


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------

_pool = None
_pool_lock = threading.Lock()
_pool_signature: tuple | None = None

#: Streamlit reruns the whole script on every interaction, and a map that
#: reloads by viewport issues several queries per rerun. Opening a TLS
#: connection to RDS each time would dominate the latency.
POOL_MIN_SIZE = int(os.environ.get("HBU_PG_POOL_MIN", 1))
POOL_MAX_SIZE = int(os.environ.get("HBU_PG_POOL_MAX", 4))

#: Read-only, and short. A UI query that has not answered in this long is a
#: query the user has already given up on.
STATEMENT_TIMEOUT_MS = int(os.environ.get("HBU_PG_STATEMENT_TIMEOUT_MS", 20_000))


def _connection_signature(details: Connection) -> tuple:
    """What has to change for the pool to be rebuilt rather than reused."""
    return (details.host, details.port, details.dbname, details.user, details.sslmode)


def get_pool():
    """The process-wide connection pool, opened on first use.

    Rebuilt when the resolved endpoint changes — which happens when someone
    switches ``HBU_ENV`` or points ``DATABASE_URL`` somewhere else without
    restarting Streamlit.
    """
    global _pool, _pool_signature

    details = resolve()
    signature = _connection_signature(details)

    with _pool_lock:
        if _pool is not None and _pool_signature == signature:
            return _pool
        if _pool is not None:
            logger.info("Database endpoint changed — reopening the pool")
            _pool.close()
            _pool = None

        try:
            from psycopg_pool import ConnectionPool  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment problem
            raise DbError(
                "psycopg-pool is not installed — `pip install 'psycopg[binary]' "
                "psycopg-pool`"
            ) from exc

        def _configure(conn) -> None:
            with conn.cursor() as cur:
                cur.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}")
                cur.execute(f"SET search_path TO {SCHEMA}, public")

        _pool = ConnectionPool(
            # autocommit because this app only reads, and because psycopg_pool
            # discards any connection its configure callback leaves inside a
            # transaction — which two SETs on a non-autocommit connection do.
            kwargs={**details.kwargs(), "autocommit": True},
            min_size=POOL_MIN_SIZE,
            max_size=POOL_MAX_SIZE,
            open=True,
            timeout=15.0,
            configure=_configure,
            # An IAM token expires; recycling connections keeps a long-lived
            # pool from holding one past its fifteen minutes.
            max_lifetime=600.0 if details.iam_auth else 3600.0,
            name="hbu-rag-map",
        )
        _pool_signature = signature
        logger.info("Connection pool opened against %s", details.url())
        return _pool


def close_pool() -> None:
    """Drop the pool. Used by tests and by the sidebar's reconnect button."""
    global _pool, _pool_signature
    with _pool_lock:
        if _pool is not None:
            _pool.close()
        _pool = None
        _pool_signature = None


def _fresh_connection():
    """One connection outside the pool, for the IAM path.

    A signed token cannot be handed to a pool that outlives it, so IAM auth
    opens and closes per query. It is the posture with nothing long-lived to
    leak, and the extra handshake is the price.
    """
    import psycopg  # noqa: PLC0415

    details = resolve()
    kwargs = details.kwargs()
    kwargs["password"] = _iam_token(details)
    conn = psycopg.connect(**kwargs, autocommit=True)
    with conn.cursor() as cur:
        cur.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}")
        cur.execute(f"SET search_path TO {SCHEMA}, public")
    return conn


class _Borrowed:
    """Uniform context manager over `pool.connection()` and a fresh connect."""

    def __init__(self) -> None:
        self._details = resolve()
        self._ctx = None
        self._conn = None

    def __enter__(self):
        if self._details.iam_auth:
            self._conn = _fresh_connection()
            return self._conn
        self._ctx = get_pool().connection()
        return self._ctx.__enter__()

    def __exit__(self, *exc_info):
        if self._ctx is not None:
            return self._ctx.__exit__(*exc_info)
        if self._conn is not None:
            self._conn.close()
        return False


def connection():
    """Borrow a connection. Use as a context manager."""
    return _Borrowed()


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def query(sql: str, params: tuple | dict | None = None) -> list[dict]:
    """Run *sql* and return rows as dicts.

    Dicts rather than tuples because every caller here hands the result to
    folium, to Streamlit, or to an LLM, and all three want names.
    """
    from psycopg.rows import dict_row  # noqa: PLC0415

    with connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        if cur.description is None:
            return []
        return list(cur.fetchall())


def query_one(sql: str, params: tuple | dict | None = None) -> dict | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def scalar(sql: str, params: tuple | dict | None = None):
    row = query_one(sql, params)
    return next(iter(row.values())) if row else None


def ping() -> bool:
    """True when the database answers. Never raises — the sidebar shows this."""
    try:
        return scalar("SELECT 1") == 1
    except Exception as exc:
        logger.warning("Database ping failed: %s", exc)
        return False


def _flag(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"} if value else False
