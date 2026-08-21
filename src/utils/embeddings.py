"""
embeddings.py — Turning a question into the vector the corpus was built with.

The corpus is embedded by the dataplatform with ``BAAI/bge-m3`` through
sentence-transformers: 1024 dimensions, multilingual, L2-normalised. A query
has to be embedded by the *same* model or the similarities are meaningless —
and the failure is silent, because a 1024-wide vector from a different encoder
is still a valid argument to ``rag.search_near``.

This app does it through the **HuggingFace Inference API** rather than by
loading sentence-transformers, for one reason: the local path costs 2.2 GB of
weights and a torch install to answer a question a Streamlit process asks a few
times a minute. The token is the same ``HUGGINGFACE_API_TOKEN`` the chat model
uses.

Two checks stand between a query and a wrong answer:

* the width of the returned vector against ``rag.chunks_meta.dimension``
* the model name against ``rag.chunks_meta.embedding_model``

Both are reported as a clear mismatch rather than being allowed through, for
the same reason the dataplatform's ``IndexMismatch`` exists.
"""

from __future__ import annotations

import logging
import math
import os
from functools import lru_cache

logger = logging.getLogger(__name__)

#: Must match ``rag.embeddings.DEFAULT_MODEL`` in hbu_dataplatform. Changing it
#: changes the vector width, which means the corpus has to be rebuilt.
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"

MODEL_ENV = "URBAN_RAG_EMBEDDING_MODEL"
TOKEN_ENV = "HUGGINGFACE_API_TOKEN"

#: The e5 family is trained with asymmetric ``query: ``/``passage: `` prefixes
#: and loses accuracy without them. bge-m3 is trained symmetric and takes none.
#: Mirrors the same rule in the dataplatform's embeddings.py.
_PREFIXED_FAMILIES = ("e5",)


class EmbeddingError(RuntimeError):
    """The query could not be embedded, or was embedded with the wrong model."""


def embedding_model() -> str:
    return os.environ.get(MODEL_ENV, DEFAULT_EMBEDDING_MODEL)


def _token() -> str:
    token = os.environ.get(TOKEN_ENV, "").strip()
    if not token:
        raise EmbeddingError(
            f"{TOKEN_ENV} is not set, so the question cannot be embedded and "
            f"the corpus cannot be searched.\n"
            f"  Copy .env.example to .env and fill in a token from "
            f"https://huggingface.co/settings/tokens"
        )
    return token


@lru_cache(maxsize=1)
def _client():
    try:
        from huggingface_hub import InferenceClient  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment problem
        raise EmbeddingError(
            "huggingface-hub is not installed — `pip install huggingface-hub`"
        ) from exc
    return InferenceClient(api_key=_token())


def _query_prefix(model: str) -> str:
    return "query: " if any(f in model.lower() for f in _PREFIXED_FAMILIES) else ""


def _pool(vectors) -> list[float]:
    """One vector out of whatever the inference endpoint returned.

    The feature-extraction pipeline answers with a sentence embedding for a
    model that carries a sentence-transformers pooling head — which bge-m3 does
    — and with token-level vectors for one that does not. Mean-pooling the
    second shape keeps a differently-configured endpoint from returning a
    nested list into a ``::vector`` cast.
    """
    flat = vectors
    while isinstance(flat, list) and flat and isinstance(flat[0], list):
        if isinstance(flat[0][0], list):
            flat = flat[0]
            continue
        columns = len(flat[0])
        return [sum(row[i] for row in flat) / len(flat) for i in range(columns)]
    return [float(v) for v in flat]


def _normalise(vector: list[float]) -> list[float]:
    """L2-normalise, because the corpus is normalised.

    pgvector's ``<=>`` is cosine distance and normalises internally, so this
    changes no ranking — but it keeps the query vector in the same units as the
    stored ones, which is what makes ``1 - distance`` readable as a similarity.
    """
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        raise EmbeddingError("the embedding endpoint returned a zero vector")
    return [v / norm for v in vector]


@lru_cache(maxsize=256)
def _embed_cached(text: str, model: str) -> tuple[float, ...]:
    """One round trip per distinct (question, model).

    Cached because a Streamlit rerun replays the whole script: panning the map
    after asking a question would otherwise re-embed the same string.
    """
    import numpy as np  # noqa: PLC0415

    try:
        raw = _client().feature_extraction(text, model=model)
    except Exception as exc:
        raise EmbeddingError(
            f"the HuggingFace Inference API could not embed the query with "
            f"{model}: {exc}"
        ) from exc

    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    vector = _normalise(_pool(raw))
    logger.debug("Embedded %d chars with %s → %d dims", len(text), model, len(vector))
    return tuple(vector)


def embed_query(text: str, *, model: str | None = None) -> list[float]:
    """The query vector, ready to hand to ``rag.search_*``."""
    text = (text or "").strip()
    if not text:
        raise EmbeddingError("cannot embed an empty question")
    name = model or embedding_model()
    return list(_embed_cached(_query_prefix(name) + text, name))


def check_against_corpus(dimension: int | None, model: str | None) -> str | None:
    """Warn when the query encoder does not match the one the corpus used.

    Returns a message to show, or None when the two agree — or when the corpus
    has not recorded what it was built with, which is the state of a database
    the dataplatform has not loaded yet.
    """
    configured = embedding_model()
    if model and model != configured:
        return (
            f"The corpus was embedded with **{model}** but this app is "
            f"configured for **{configured}**. Searching across the two "
            f"returns confident nonsense rather than an error — set "
            f"`{MODEL_ENV}={model}`."
        )
    if dimension is None:
        return None
    try:
        width = len(embed_query("dimension probe"))
    except EmbeddingError:
        return None
    if width != dimension:
        return (
            f"The corpus holds {dimension}-wide vectors but **{configured}** "
            f"returns {width}. Retrieval will fail at the `::vector` cast."
        )
    return None
