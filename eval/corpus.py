"""Re-embed the corpus with a second encoder, into ``eval.chunk_embeddings``.

The text embedded is ``rag.chunks.text`` exactly as the dataplatform embedded
it with bge-m3 (``rag/assets.py`` encodes ``frame["text"]`` and nothing
else), so the two encoders see the same strings and a retrieval difference
is the encoder's.

Resumable: every batch is committed as it lands, and a restart picks up the
chunks the model has not yet covered. Voyage's free tier rate-limits an
account with no payment method hard (a few requests a minute); the backoff
below waits that out rather than failing, and ``--batch`` keeps each request
well under the per-request token limit.
"""

from __future__ import annotations

import logging
import random
import time

from eval.shadow import DIMENSION, EMBEDDINGS_TABLE, ensure_store
from eval.spend import SpendMeter

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "voyage-3.5"
MAX_ATTEMPTS = 8


def _pending(conn, model: str, neighborhoods: list[str] | None, *, source: str = "rag") -> list[tuple[str, str, str]]:
    sql = f"""
        SELECT c.neighborhood, c.chunk_id, c.text
          FROM {source}.chunks c
          LEFT JOIN {EMBEDDINGS_TABLE} e
            ON e.model = %s AND e.neighborhood = c.neighborhood AND e.chunk_id = c.chunk_id
         WHERE e.chunk_id IS NULL
    """
    params: list = [model]
    if neighborhoods:
        sql += " AND c.neighborhood = ANY(%s)"
        params.append(neighborhoods)
    sql += " ORDER BY c.neighborhood, c.chunk_id"
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [(r[0], r[1], r[2] or "") for r in cur.fetchall()]


#: A rate limit is a per-minute window: the only wait that helps is the rest
#: of the minute. Retrying sooner spends another request against the same
#: window, and a run of those is what turned a 2-hour job into a 10-hour one.
RATE_LIMIT_WAIT_S = 61.0
RATE_LIMIT_ATTEMPTS = 30


def _embed_batch(client, texts: list[str], model: str):
    delay = 2.0
    attempts = 0
    while True:
        attempts += 1
        try:
            return client.embed(texts, model=model, input_type="document", output_dimension=DIMENSION)
        except Exception as exc:  # noqa: BLE001 - the SDK's error classes vary by version
            name = type(exc).__name__
            if "RateLimit" in name:
                if attempts >= RATE_LIMIT_ATTEMPTS:
                    raise
                logger.warning("voyage rate limit on attempt %d; waiting out the minute", attempts)
                time.sleep(RATE_LIMIT_WAIT_S + random.uniform(0, 3))
                continue
            retryable = any(k in name for k in ("Timeout", "Connection", "ServerError", "APIError", "Service"))
            if not retryable or attempts >= MAX_ATTEMPTS:
                raise
            wait = delay + random.uniform(0, delay / 2)
            logger.warning("voyage %s on attempt %d; waiting %.0fs", name, attempts, wait)
            time.sleep(wait)
            delay = min(delay * 2, 90)


#: What Voyage allows an account with no payment method: 3 requests and
#: 10,000 tokens a minute. One request a minute, under the token cap, is the
#: fastest the free tier goes without the retry churn.
FREE_TIER_TOKENS_PER_REQUEST = 8_000
FREE_TIER_INTERVAL_S = 61.0


def _batches(pending, *, batch: int, max_tokens: int | None, chars_per_token: float):
    """Slices of ``pending``: by count, or by estimated tokens when capped."""
    if not max_tokens:
        for start in range(0, len(pending), batch):
            yield pending[start : start + batch]
        return
    rows: list = []
    estimate = 0.0
    for row in pending:
        cost = max(1.0, len(row[2]) / chars_per_token)
        if rows and (estimate + cost > max_tokens or len(rows) >= batch):
            yield rows
            rows, estimate = [], 0.0
        rows.append(row)
        estimate += cost
    if rows:
        yield rows


def reembed(
    conn,
    *,
    model: str = DEFAULT_MODEL,
    neighborhoods: list[str] | None = None,
    batch: int = 64,
    meter: SpendMeter | None = None,
    limit: int | None = None,
    free_tier: bool = False,
    progress=print,
) -> dict:
    """Embed every chunk ``model`` has not covered yet. Returns counts.

    ``free_tier`` paces the job under Voyage's no-payment-method limits:
    batches sized to ~9K tokens (estimated from characters, corrected from
    each response's real count) and one request a minute. Slow - the whole
    corpus is about seven hours - but it runs unattended and resumes.
    """
    import os

    try:
        import voyageai  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("voyageai is not installed — `uv pip install -e '.[eval]'`") from exc
    key = os.environ.get("VOYAGE_API_KEY", "").strip()
    if not key:
        raise SystemExit("VOYAGE_API_KEY is not set (add it to .env)")
    bundle = os.environ.get("SSL_CERT_FILE", "")
    if bundle and os.path.exists(bundle):
        os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle)
    client = voyageai.Client(api_key=key)

    ensure_store(conn)
    pending = _pending(conn, model, neighborhoods)
    if limit:
        pending = pending[:limit]
    total = len(pending)
    progress(f"{total:,} chunk(s) to embed with {model}" + (f" in {', '.join(neighborhoods)}" if neighborhoods else "")
             + (" (free-tier pacing: ~9K tokens a minute)" if free_tier else ""))
    done = tokens = 0
    chars_per_token = 3.0  # conservative for French; corrected after the first response
    started = time.perf_counter()
    last_request = 0.0
    for rows in _batches(pending, batch=batch, max_tokens=FREE_TIER_TOKENS_PER_REQUEST if free_tier else None,
                         chars_per_token=chars_per_token):
        if free_tier:
            wait = FREE_TIER_INTERVAL_S - (time.perf_counter() - last_request)
            if wait > 0:
                time.sleep(wait)
        last_request = time.perf_counter()
        result = _embed_batch(client, [text for _, _, text in rows], model)
        used = int(getattr(result, "total_tokens", 0) or 0)
        tokens += used
        if used:
            chars_per_token = max(1.5, sum(len(t) for _, _, t in rows) / used)
        vectors = result.embeddings
        with conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {EMBEDDINGS_TABLE} (model, neighborhood, chunk_id, embedding, tokens) "
                f"VALUES (%s, %s, %s, %s::vector, %s) ON CONFLICT DO NOTHING",
                [
                    (model, nb, cid, "[" + ",".join(f"{v:.7f}" for v in vec) + "]", used // max(len(rows), 1))
                    for (nb, cid, _), vec in zip(rows, vectors, strict=True)
                ],
            )
        conn.commit()
        if meter is not None:
            meter.charge(service="voyage", model=model, usage={"input_tokens": used}, batch=len(rows))
        done += len(rows)
        elapsed = time.perf_counter() - started
        rate = done / elapsed if elapsed else 0
        eta = (total - done) / rate if rate else 0
        progress(f"  {done:,}/{total:,} chunks, {tokens:,} tokens, {elapsed:.0f}s elapsed, ~{eta:.0f}s left")
    return {"model": model, "embedded": done, "tokens": tokens, "seconds": round(time.perf_counter() - started, 1)}
