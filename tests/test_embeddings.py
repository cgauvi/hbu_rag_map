"""Query embedding: the shapes the endpoint can return, and the mismatch guard.

The guard is the point. A 1024-wide vector from the wrong encoder is a
perfectly valid argument to ``rag.search_near`` — it returns rows, ranked, with
plausible similarities, all of them meaningless. Nothing downstream can detect
that, so it has to be caught here.
"""

from __future__ import annotations

import math

import pytest

from src.utils import embeddings


def _clear_client_cache():
    """`_client` is lru_cached in the module but monkeypatched away in most
    tests, where the replacement has no cache to clear."""
    clear = getattr(embeddings._client, "cache_clear", None)
    if clear:
        clear()


@pytest.fixture(autouse=True)
def _clear_cache():
    embeddings._embed_cached.cache_clear()
    _clear_client_cache()
    yield
    embeddings._embed_cached.cache_clear()
    _clear_client_cache()


class _FakeClient:
    """Stands in for huggingface_hub.InferenceClient."""

    def __init__(self, payload, recorder=None):
        self.payload = payload
        self.recorder = recorder if recorder is not None else []

    def feature_extraction(self, text, model=None):
        self.recorder.append((text, model))
        return self.payload


def _install(monkeypatch, payload):
    client = _FakeClient(payload)
    monkeypatch.setattr(embeddings, "_client", lambda: client)
    return client


def test_a_missing_token_says_where_to_get_one():
    with pytest.raises(embeddings.EmbeddingError, match="huggingface.co/settings/tokens"):
        embeddings.embed_query("hauteur maximale")


def test_a_sentence_embedding_is_used_as_is(monkeypatch, hf_token):
    _install(monkeypatch, [3.0, 4.0])
    vector = embeddings.embed_query("hauteur maximale")
    # Normalised: 3-4-5 triangle.
    assert vector == pytest.approx([0.6, 0.8])


def test_token_level_output_is_mean_pooled(monkeypatch, hf_token):
    """An endpoint without a pooling head answers per token, not per sentence."""
    _install(monkeypatch, [[1.0, 0.0], [0.0, 1.0]])
    vector = embeddings.embed_query("hauteur maximale")
    assert vector == pytest.approx([0.7071, 0.7071], abs=1e-4)


def test_a_batch_of_one_is_unwrapped(monkeypatch, hf_token):
    _install(monkeypatch, [[[1.0, 0.0], [0.0, 1.0]]])
    assert embeddings.embed_query("x") == pytest.approx([0.7071, 0.7071], abs=1e-4)


def test_the_result_is_unit_length(monkeypatch, hf_token):
    _install(monkeypatch, [0.3, -0.9, 2.4, 1.0])
    vector = embeddings.embed_query("taux d'implantation")
    assert math.sqrt(sum(v * v for v in vector)) == pytest.approx(1.0)


def test_a_zero_vector_is_an_error(monkeypatch, hf_token):
    _install(monkeypatch, [0.0, 0.0])
    with pytest.raises(embeddings.EmbeddingError, match="zero vector"):
        embeddings.embed_query("x")


def test_an_empty_question_is_refused(hf_token):
    with pytest.raises(embeddings.EmbeddingError, match="empty"):
        embeddings.embed_query("   ")


def test_bge_m3_gets_no_prefix(monkeypatch, hf_token):
    """bge-m3 is trained symmetric; e5 is not. Prefixing the wrong one costs accuracy."""
    client = _install(monkeypatch, [1.0, 0.0])
    embeddings.embed_query("hauteur", model="BAAI/bge-m3")
    assert client.recorder[0][0] == "hauteur"


def test_e5_models_get_the_query_prefix(monkeypatch, hf_token):
    client = _install(monkeypatch, [1.0, 0.0])
    embeddings.embed_query("hauteur", model="intfloat/multilingual-e5-small")
    assert client.recorder[0][0] == "query: hauteur"


def test_repeated_questions_hit_the_cache(monkeypatch, hf_token):
    client = _install(monkeypatch, [1.0, 0.0])
    embeddings.embed_query("same question")
    embeddings.embed_query("same question")
    assert len(client.recorder) == 1


def test_an_endpoint_failure_is_reported_with_the_model(monkeypatch, hf_token):
    class _Broken:
        def feature_extraction(self, *_a, **_k):
            raise RuntimeError("503 Service Unavailable")

    monkeypatch.setattr(embeddings, "_client", lambda: _Broken())
    with pytest.raises(embeddings.EmbeddingError, match="BAAI/bge-m3"):
        embeddings.embed_query("x")


# ---------------------------------------------------------------------------
# The mismatch guard
# ---------------------------------------------------------------------------


def test_a_different_corpus_model_is_flagged(monkeypatch, hf_token):
    monkeypatch.setenv("URBAN_RAG_EMBEDDING_MODEL", "BAAI/bge-m3")
    warning = embeddings.check_against_corpus(384, "intfloat/multilingual-e5-small")
    assert warning is not None
    assert "confident nonsense" in warning


def test_a_matching_model_and_width_passes(monkeypatch, hf_token):
    _install(monkeypatch, [1.0] * 1024)
    monkeypatch.setenv("URBAN_RAG_EMBEDDING_MODEL", "BAAI/bge-m3")
    assert embeddings.check_against_corpus(1024, "BAAI/bge-m3") is None


def test_a_width_mismatch_is_flagged(monkeypatch, hf_token):
    _install(monkeypatch, [1.0] * 384)
    monkeypatch.setenv("URBAN_RAG_EMBEDDING_MODEL", "BAAI/bge-m3")
    warning = embeddings.check_against_corpus(1024, "BAAI/bge-m3")
    assert warning is not None
    assert "1024" in warning and "384" in warning


def test_an_unrecorded_corpus_is_not_an_error():
    """A database the dataplatform has not loaded has no metadata to check."""
    assert embeddings.check_against_corpus(None, None) is None
