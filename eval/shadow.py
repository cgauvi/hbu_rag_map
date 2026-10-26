"""A second corpus beside the first: ``rag_eval``, the schema with the other encoder's vectors.

The app reads one schema (``URBAN_RAG_PG_SCHEMA``), and everything it reads
there - the tables, the search functions, the views the functions use - is
owned by hbu_infra. To search the corpus with a second encoder without
touching any of it, this builds a *shadow* of ``rag``:

* a pass-through view for every table, except ``chunks``, which is the same
  rows joined to ``eval.chunk_embeddings`` for the chosen model - so the
  ``embedding`` column is the other encoder's and every other column,
  ``tsv`` included, is the original's;
* ``chunks_meta`` reporting that model, so the app's encoder check passes;
* every view and every function of ``rag``, recreated from the catalog's
  own definition with ``rag.`` rewritten to ``rag_eval.``.

The point of recreating the functions rather than writing new ones is that
the hybrid ranking, the containment search and the radius search are then
*identical* between the two encoders - the SQL is the same text. The only
difference between a run on ``rag`` and a run on ``rag_eval`` is the vector.

What it costs: the dense arm can no longer use ``rag.chunks``' HNSW index
(the planner will not reach through a join), so it scans. At 11k rows of
1024 floats that is tens of milliseconds, and it is exact rather than
approximate - which, if anything, flatters the shadow side slightly.

Drop it with ``drop()``; nothing in ``rag`` is changed by building it.
"""

from __future__ import annotations

import logging
import re

from psycopg import sql

logger = logging.getLogger(__name__)

EMBEDDINGS_TABLE = "eval.chunk_embeddings"
DIMENSION = 1024


def ensure_store(conn) -> None:
    """The table the re-embedded corpus lives in. Idempotent."""
    with conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS eval")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {EMBEDDINGS_TABLE} (
                model        text NOT NULL,
                neighborhood text NOT NULL,
                chunk_id     text NOT NULL,
                embedding    vector({DIMENSION}) NOT NULL,
                tokens       integer,
                embedded_at  timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (model, neighborhood, chunk_id)
            )
            """
        )
    conn.commit()


def coverage(conn, model: str, *, source: str = "rag") -> tuple[int, int]:
    """(chunks embedded with ``model``, chunks in the corpus)."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {EMBEDDINGS_TABLE} WHERE model = %s", (model,))
        have = cur.fetchone()[0]
        cur.execute(f"SELECT count(*) FROM {source}.chunks")
        total = cur.fetchone()[0]
    return int(have), int(total)


def rewrite(definition: str, source: str, target: str) -> str:
    """``rag.chunks`` -> ``rag_eval.chunks`` throughout a definition.

    A word boundary on the left keeps ``urban_rag.`` and ``rag_eval.`` alone;
    the trailing dot keeps a column or alias that happens to be called
    ``rag`` alone too. Applied to view and function bodies alike.
    """
    return re.sub(rf"(?<![\w.]){re.escape(source)}\.", f"{target}.", definition)


def _tables(cur, schema: str) -> list[str]:
    cur.execute(
        "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        " WHERE n.nspname = %s AND c.relkind IN ('r', 'p') ORDER BY 1",
        (schema,),
    )
    return [r[0] for r in cur.fetchall()]


def _views(cur, schema: str) -> list[tuple[str, str]]:
    cur.execute(
        "SELECT c.relname, pg_get_viewdef(c.oid, true) FROM pg_class c "
        "  JOIN pg_namespace n ON n.oid = c.relnamespace "
        " WHERE n.nspname = %s AND c.relkind = 'v' ORDER BY 1",
        (schema,),
    )
    return [(r[0], r[1]) for r in cur.fetchall()]


def _functions(cur, schema: str) -> list[tuple[str, str]]:
    cur.execute(
        "SELECT p.proname, pg_get_functiondef(p.oid) FROM pg_proc p "
        "  JOIN pg_namespace n ON n.oid = p.pronamespace "
        " WHERE n.nspname = %s ORDER BY 1",
        (schema,),
    )
    return [(r[0], r[1]) for r in cur.fetchall()]


def _columns(cur, schema: str, table: str) -> list[str]:
    cur.execute(
        "SELECT column_name FROM information_schema.columns "
        " WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
        (schema, table),
    )
    return [r[0] for r in cur.fetchall()]


def _create_in_order(cur, statements: list[tuple[str, str]], *, what: str) -> None:
    """Run definitions that may depend on one another, retrying until none is left.

    A view over a view, a function calling a function: the catalog lists them
    alphabetically, not topologically, and a SQL-language body is parsed at
    CREATE. Each pass creates what it can; a pass that creates nothing
    reports what it could not.
    """
    pending = list(statements)
    while pending:
        failed: list[tuple[str, str]] = []
        errors: dict[str, str] = {}
        for name, statement in pending:
            cur.execute("SAVEPOINT one")
            try:
                cur.execute(statement)
                cur.execute("RELEASE SAVEPOINT one")
            except Exception as exc:  # noqa: BLE001 - a dependency not yet created
                cur.execute("ROLLBACK TO SAVEPOINT one")
                failed.append((name, statement))
                errors[name] = str(exc).splitlines()[0]
        if len(failed) == len(pending):
            detail = "; ".join(f"{n}: {e}" for n, e in errors.items())
            raise RuntimeError(f"could not create these {what}: {detail}")
        pending = failed


def build(conn, model: str, *, source: str = "rag", target: str = "rag_eval", allow_partial: bool = False) -> dict:
    """Create ``target`` as the shadow of ``source`` carrying ``model``'s vectors."""
    ensure_store(conn)
    have, total = coverage(conn, model, source=source)
    if have == 0:
        raise RuntimeError(
            f"{EMBEDDINGS_TABLE} holds no rows for {model!r}: run `python -m eval reembed` first"
        )
    if have < total and not allow_partial:
        raise RuntimeError(
            f"{model!r} covers {have:,} of {total:,} chunks; finish `reembed` or pass --allow-partial "
            f"(the shadow corpus is then the embedded subset only)"
        )

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {target} CASCADE")
        cur.execute(f"CREATE SCHEMA {target}")

        tables = _tables(cur, source)
        for table in tables:
            if table in ("chunks", "chunks_meta"):
                continue
            cur.execute(f"CREATE VIEW {target}.{table} AS SELECT * FROM {source}.{table}")

        columns = [c for c in _columns(cur, source, "chunks") if c != "embedding"]
        col_list = ", ".join(f"c.{c}" for c in columns)
        # DDL takes no bound parameters, so the model name is a quoted literal.
        cur.execute(
            sql.SQL(f"""
            CREATE VIEW {target}.chunks AS
                SELECT {col_list}, e.embedding
                  FROM {source}.chunks c
                  JOIN {EMBEDDINGS_TABLE} e
                    ON e.neighborhood = c.neighborhood AND e.chunk_id = c.chunk_id
                 WHERE e.model = {{model}}
            """).format(model=sql.Literal(model))
        )
        meta_cols = _columns(cur, source, "chunks_meta")
        key_col, value_col = meta_cols[0], meta_cols[1]
        rest = ", ".join(meta_cols[2:])
        cur.execute(
            sql.SQL(f"""
            CREATE VIEW {target}.chunks_meta AS
                SELECT {key_col},
                       CASE WHEN {key_col} = 'embedding_model' THEN {{model}}
                            WHEN {key_col} = 'dimension' THEN {{dimension}}
                            ELSE {value_col} END AS {value_col}
                       {', ' + rest if rest else ''}
                  FROM {source}.chunks_meta
            """).format(model=sql.Literal(model), dimension=sql.Literal(str(DIMENSION)))
        )

        views = [
            (name, f"CREATE VIEW {target}.{name} AS {rewrite(definition, source, target)}")
            for name, definition in _views(cur, source)
        ]
        _create_in_order(cur, views, what="views")

        functions = [(name, rewrite(definition, source, target)) for name, definition in _functions(cur, source)]
        _create_in_order(cur, functions, what="functions")
    conn.commit()
    report = {
        "schema": target, "model": model, "embedded": have, "chunks": total,
        "tables": len(tables), "views": len(views) + 2, "functions": len(functions),
    }
    logger.info("built %s: %s", target, report)
    return report


def drop(conn, *, target: str = "rag_eval") -> None:
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {target} CASCADE")
    conn.commit()


def status(conn, *, source: str = "rag", target: str = "rag_eval") -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_namespace WHERE nspname = %s", (target,))
        present = cur.fetchone() is not None
        model = None
        if present:
            cur.execute(f"SELECT value FROM {target}.chunks_meta WHERE key = 'embedding_model'")
            row = cur.fetchone()
            model = row[0] if row else None
        cur.execute("SELECT to_regclass(%s) IS NOT NULL", (EMBEDDINGS_TABLE,))
        store = cur.fetchone()[0]
        models: dict[str, int] = {}
        if store:
            cur.execute(f"SELECT model, count(*) FROM {EMBEDDINGS_TABLE} GROUP BY 1")
            models = {m: int(n) for m, n in cur.fetchall()}
        cur.execute(f"SELECT count(*) FROM {source}.chunks")
        total = int(cur.fetchone()[0])
    return {"shadow": target if present else None, "shadow_model": model, "corpus_chunks": total, "embedded": models}
