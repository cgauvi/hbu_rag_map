"""The provider switches: which chat model and which encoder the environment selects."""

from __future__ import annotations

import sys
import types

import pytest

from src import config
from src.utils import embeddings


def test_the_default_provider_is_huggingface_and_the_label_names_the_repo(monkeypatch):
    monkeypatch.delenv("HBU_LLM_PROVIDER", raising=False)
    assert config.provider() == "huggingface"
    assert config.model_label() == "huggingface:" + config.MODELS[config.DEFAULT_MODEL_ALIAS].repo_id


def test_the_anthropic_provider_needs_a_key(monkeypatch):
    monkeypatch.setenv("HBU_LLM_PROVIDER", "anthropic")
    for name in config.ANTHROPIC_KEY_ENVS:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(config.ConfigurationError, match="ANTHROPIC"):
        config.build_llm()


def test_an_unknown_provider_is_refused(monkeypatch):
    monkeypatch.setenv("HBU_LLM_PROVIDER", "gemini")
    with pytest.raises(config.ConfigurationError, match="gemini"):
        config.build_llm()


def test_the_anthropic_model_is_built_with_the_eval_s_knobs(monkeypatch):
    """What reaches ChatAnthropic: model, cap, effort, a server-side cache breakpoint, callbacks."""
    captured = {}

    class FakeChat:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "langchain_anthropic", types.SimpleNamespace(ChatAnthropic=FakeChat))
    monkeypatch.setenv("HBU_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_KEY", "sk-test")
    monkeypatch.setenv("HBU_ANTHROPIC_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("HBU_LLM_DISABLE_STREAMING", "1")
    sentinel = object()
    config.build_llm(max_new_tokens=777, callbacks=[sentinel])
    assert captured["model"] == "claude-sonnet-5-5"
    assert captured["api_key"] == "sk-test"
    assert captured["max_tokens"] == 777
    assert captured["effort"] == config.DEFAULT_ANTHROPIC_EFFORT
    assert captured["model_kwargs"] == {"cache_control": {"type": "ephemeral"}}
    assert captured["callbacks"] == [sentinel]
    assert captured["disable_streaming"] is True
    assert "temperature" not in captured
    assert config.model_label() == "anthropic:claude-sonnet-5-5"


def test_claude_gets_the_system_prompt_as_a_cached_block_and_the_plan_apart(monkeypatch):
    from src import agent

    monkeypatch.setenv("HBU_LLM_PROVIDER", "anthropic")
    message = agent._system_message("1. find the lot")
    assert isinstance(message.content, list)
    assert message.content[0]["cache_control"] == {"type": "ephemeral"}
    assert message.content[0]["text"] == agent._SYSTEM_PROMPT
    assert "find the lot" in message.content[1]["text"]
    assert "cache_control" not in message.content[1]
    # No plan: one block only, so the cached prefix is the whole prompt.
    assert len(agent._system_message("").content) == 1


def test_huggingface_gets_the_system_prompt_as_one_string(monkeypatch):
    from src import agent

    monkeypatch.delenv("HBU_LLM_PROVIDER", raising=False)
    message = agent._system_message("1. find the lot")
    assert isinstance(message.content, str)
    assert message.content.startswith(agent._SYSTEM_PROMPT)
    assert "find the lot" in message.content


def test_an_answer_made_of_content_blocks_is_read_as_its_text():
    """Claude answers with [thinking, text]; the prose is the text block."""
    from langchain_core.messages import AIMessage

    from src import agent

    blocks = AIMessage(content=[{"type": "thinking", "thinking": ""}, {"type": "text", "text": "Up to 3 storeys."}])
    assert agent._text_of(blocks) == "Up to 3 storeys."
    assert agent._text_of(AIMessage(content="plain")) == "plain"
    assert agent._text_of(AIMessage(content=[{"type": "tool_use", "id": "x", "name": "f", "input": {}}])) == ""


def test_the_default_encoder_is_bge_and_voyage_switches_the_default_model(monkeypatch):
    monkeypatch.delenv("URBAN_RAG_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("URBAN_RAG_EMBEDDING_MODEL", raising=False)
    assert embeddings.embedding_model() == "BAAI/bge-m3"
    monkeypatch.setenv("URBAN_RAG_EMBEDDING_PROVIDER", "voyage")
    assert embeddings.embedding_model() == "voyage-3.5"
    monkeypatch.setenv("URBAN_RAG_EMBEDDING_MODEL", "voyage-3-large")
    assert embeddings.embedding_model() == "voyage-3-large"


def test_a_voyage_query_is_embedded_as_a_query_at_the_corpus_width(monkeypatch):
    calls = {}

    class FakeResult:
        embeddings = [[3.0, 4.0]]
        total_tokens = 7

    class FakeClient:
        def __init__(self, api_key):
            calls["key"] = api_key

        def embed(self, texts, **kwargs):
            calls["texts"] = texts
            calls.update(kwargs)
            return FakeResult()

    monkeypatch.setitem(sys.modules, "voyageai", types.SimpleNamespace(Client=FakeClient))
    monkeypatch.setenv("URBAN_RAG_EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
    embeddings._voyage_client.cache_clear()
    embeddings._embed_cached.cache_clear()
    before = dict(embeddings.USAGE)
    vector = embeddings.embed_query("hauteur maximale")
    assert calls["key"] == "pa-test"
    assert calls["texts"] == ["hauteur maximale"]
    assert calls["input_type"] == "query"
    assert calls["output_dimension"] == embeddings.VOYAGE_DIMENSION
    assert vector == pytest.approx([0.6, 0.8])  # L2-normalised, like the corpus
    assert embeddings.USAGE["calls"] == before["calls"] + 1
    assert embeddings.USAGE["tokens"] == before["tokens"] + 7


def test_a_voyage_query_without_a_key_is_a_clear_error(monkeypatch):
    monkeypatch.setenv("URBAN_RAG_EMBEDDING_PROVIDER", "voyage")
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    embeddings._voyage_client.cache_clear()
    embeddings._embed_cached.cache_clear()
    with pytest.raises(embeddings.EmbeddingError, match="VOYAGE_API_KEY"):
        embeddings.embed_query("anything at all")
