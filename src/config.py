"""
config.py — HuggingFace model catalog for the zoning map assistant.

All models are served through the HuggingFace Inference API and must support
tool/function calling.

The default used to be the strongest *multilingual* model in the catalog, on
the reasoning that the corpus and every zoning grid behind it are in French.
It is now the strongest *reasoning* model instead, because what the assistant
got wrong was rarely the French: it was holding a question that spans the
cadastre, the grid, the roll and the heritage layers together for long enough
to answer it, which is a planning problem rather than a reading one.

That trade is deliberate and it has a cost - gpt-oss reads French less well
than Qwen does. The cost is paid back on the retrieval side rather than here:
the corpus is searched lexically as well as densely, and exact terms (zone
codes, by-law numbers) are lifted out of the question and matched as tokens,
so finding the right passage no longer depends on the chat model being fluent.
Qwen stays in the catalog as the fallback for anyone who disagrees.

Selecting a model
-----------------
Set ``HF_MODEL_ID`` in ``.env`` to either a short alias from ``MODELS`` below
or a full HuggingFace repo ID. Unset, ``DEFAULT_MODEL_ALIAS`` is used.

Note this is the *chat* model only. The query encoder is a separate choice, and
not a free one — it has to match what the corpus was embedded with. See
``src/utils/embeddings.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel


class ConfigurationError(RuntimeError):
    """The app is missing something it needs before it can call the model."""


@dataclass
class ModelConfig:
    """Metadata for a single HuggingFace model."""

    repo_id: str
    description: str
    context_window: int = 32_768
    notes: str = ""


MODELS: dict[str, ModelConfig] = {
    "qwen2.5-72b": ModelConfig(
        repo_id="Qwen/Qwen2.5-72B-Instruct",
        description="Qwen 2.5 72B - strongest French, the gpt-oss fallback",
        context_window=131_072,
        notes="Reads the French regulation text best of the catalog. Was the "
              "default until the work moved to multi-step questions, which it "
              "plans less reliably than gpt-oss.",
    ),
    "gpt-oss-120b": ModelConfig(
        repo_id="openai/gpt-oss-120b",
        description="OpenAI GPT-OSS 120B - strongest reasoning, recommended default",
        context_window=131_072,
        notes="Uses the harmony response format, so tool calls depend on the "
              "chat template being right - check a tool-calling turn after any "
              "langchain-huggingface upgrade. Weaker French than Qwen; the "
              "lexical retrieval arm is what covers that.",
    ),
    "gpt-oss-20b": ModelConfig(
        repo_id="openai/gpt-oss-20b",
        description="OpenAI GPT-OSS 20B — lower latency, 21B params / 3.6B active",
        context_window=131_072,
        notes="Uses harmony response format; weaker on French than the 120B.",
    ),
    "gemma-3-27b": ModelConfig(
        repo_id="google/gemma-3-27b-it",
        description="Google Gemma 3 27B — multimodal, good French",
        context_window=131_072,
        notes="Requires accepting Google's licence on HuggingFace before use.",
    ),
}

DEFAULT_MODEL_ALIAS = "gpt-oss-120b"


def resolve_model(hf_model_id: str | None = None) -> tuple[str, ModelConfig | None]:
    """Resolve an alias or raw repo ID to ``(repo_id, ModelConfig | None)``."""
    raw = hf_model_id or os.environ.get("HF_MODEL_ID", DEFAULT_MODEL_ALIAS)
    if raw in MODELS:
        cfg = MODELS[raw]
        return cfg.repo_id, cfg
    return raw, None


def build_llm(hf_model_id: str | None = None) -> BaseChatModel:
    """A ChatHuggingFace instance for the requested model.

    Raises:
        ConfigurationError: ``HUGGINGFACE_API_TOKEN`` is not set.
    """
    token = os.environ.get("HUGGINGFACE_API_TOKEN", "")
    if not token:
        raise ConfigurationError(
            "HUGGINGFACE_API_TOKEN is not set. "
            "Copy .env.example to .env and fill in your token."
        )

    repo_id, _ = resolve_model(hf_model_id)

    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint  # noqa: PLC0415

    endpoint = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        huggingfacehub_api_token=token,
        # gpt-oss spends part of its budget on reasoning tokens before it emits
        # the answer or the tool call, so the 2048 that sufficed for Qwen can
        # truncate a turn here - and a truncated tool call reads as the
        # "malformed tool call" `src.agent` already has a branch for, which
        # hides the real cause.
        max_new_tokens=4096,
        # Zoning answers are numbers read off a grid. Sampling them is the one
        # way this assistant can be confidently wrong about something checkable.
        temperature=0.1,
    )
    return ChatHuggingFace(llm=endpoint, verbose=False)
