"""
config.py — HuggingFace model catalog for the zoning map assistant.

All models are served through the HuggingFace Inference API and must support
tool/function calling. The corpus and every zoning grid behind it are in
French, so the default is the strongest multilingual instruct model in the
catalog rather than the fastest one — a model that reads "taux d'implantation
maximal" as a phrase and not as three tokens it has seen apart.

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
        description="Qwen 2.5 72B — strongest multilingual quality, recommended default",
        context_window=131_072,
        notes="Reads the French regulation text well; the default for that reason.",
    ),
    "gpt-oss-120b": ModelConfig(
        repo_id="openai/gpt-oss-120b",
        description="OpenAI GPT-OSS 120B — high reasoning, 117B params / 5.1B active",
        context_window=131_072,
        notes="Uses harmony response format; requires a chat template for correct output.",
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

DEFAULT_MODEL_ALIAS = "qwen2.5-72b"


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
        max_new_tokens=2048,
        # Zoning answers are numbers read off a grid. Sampling them is the one
        # way this assistant can be confidently wrong about something checkable.
        temperature=0.1,
    )
    return ChatHuggingFace(llm=endpoint, verbose=False)
