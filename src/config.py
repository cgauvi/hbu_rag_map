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

#: Room for gpt-oss to reason before it answers. See the note at the call site.
DEFAULT_MAX_NEW_TOKENS = 4096

DEFAULT_MODEL_ALIAS = "gpt-oss-120b"

# ---------------------------------------------------------------------------
# Which API serves the chat model
# ---------------------------------------------------------------------------
#
# "huggingface" (the default) is the Inference API and the catalog above.
# "anthropic" is Claude through the Anthropic SDK - the closed orchestrator the
# eval harness (`eval/`) puts beside the open one, on the same tools and the
# same prompt, so that what differs between two runs is the model and nothing
# else. The app itself never sets it; `.env` can.

PROVIDER_ENV = "HBU_LLM_PROVIDER"
ANTHROPIC_MODEL_ENV = "HBU_ANTHROPIC_MODEL"
ANTHROPIC_EFFORT_ENV = "HBU_ANTHROPIC_EFFORT"
#: Either name: the SDK reads the first, this repo's `.env` was written with
#: the second.
ANTHROPIC_KEY_ENVS = ("ANTHROPIC_API_KEY", "ANTHROPIC_KEY")
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
#: Reasoning effort. `medium` is where multi-step tool use pays back on
#: Sonnet 5.5; the harness sweeps it when it wants to.
DEFAULT_ANTHROPIC_EFFORT = "medium"
#: Set to 1 to make every model call a single request rather than a stream.
#: The eval harness does: a provider's token usage is reported reliably on
#: the non-streaming path and the typing effect is nothing to a script.
DISABLE_STREAMING_ENV = "HBU_LLM_DISABLE_STREAMING"


def provider() -> str:
    """``huggingface`` or ``anthropic``, from the environment."""
    return (os.environ.get(PROVIDER_ENV, "") or "huggingface").strip().lower()


def anthropic_model() -> str:
    return os.environ.get(ANTHROPIC_MODEL_ENV, "") or DEFAULT_ANTHROPIC_MODEL


def model_label() -> str:
    """The model the next `build_llm()` would build, as one string.

    What the agent caches on, so a change of provider rebuilds it the same way
    a change of ``HF_MODEL_ID`` always has; and what a report prints.
    """
    if provider() == "anthropic":
        return f"anthropic:{anthropic_model()}"
    repo_id, _ = resolve_model()
    return f"huggingface:{repo_id}"


def _disable_streaming() -> bool:
    return os.environ.get(DISABLE_STREAMING_ENV, "") == "1"


def resolve_model(hf_model_id: str | None = None) -> tuple[str, ModelConfig | None]:
    """Resolve an alias or raw repo ID to ``(repo_id, ModelConfig | None)``."""
    raw = hf_model_id or os.environ.get("HF_MODEL_ID", DEFAULT_MODEL_ALIAS)
    if raw in MODELS:
        cfg = MODELS[raw]
        return cfg.repo_id, cfg
    return raw, None


def build_llm(
    hf_model_id: str | None = None,
    max_new_tokens: int | None = None,
    *,
    callbacks: list | None = None,
) -> BaseChatModel:
    """A chat model for the configured provider.

    ``max_new_tokens`` is overridable for the planner, which writes five short
    lines and should not be given the budget to write an answer instead.
    ``callbacks`` are LangChain callback handlers attached to every call the
    model makes - how the eval harness reads token usage and latency off the
    real agent without wrapping it.

    Raises:
        ConfigurationError: the provider's credential is not set.
    """
    if provider() == "anthropic":
        return _build_anthropic(max_new_tokens, callbacks=callbacks)
    if provider() != "huggingface":
        raise ConfigurationError(
            f"{PROVIDER_ENV}={provider()!r} is not a provider this app knows: "
            f"use 'huggingface' or 'anthropic'."
        )

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
        max_new_tokens=max_new_tokens or DEFAULT_MAX_NEW_TOKENS,
        # Zoning answers are numbers read off a grid. Sampling them is the one
        # way this assistant can be confidently wrong about something checkable.
        temperature=0.1,
    )
    return ChatHuggingFace(
        llm=endpoint,
        verbose=False,
        callbacks=callbacks,
        disable_streaming=_disable_streaming(),
    )


def _build_anthropic(max_new_tokens: int | None, *, callbacks: list | None) -> BaseChatModel:
    """Claude, through the Anthropic SDK that `langchain-anthropic` wraps.

    No temperature: the current models reject a non-default one, and the
    point of the setting on the HuggingFace side - not sampling a storey
    count - is what adaptive thinking at a fixed effort gives here anyway.
    """
    key = next((os.environ.get(name, "").strip() for name in ANTHROPIC_KEY_ENVS
                if os.environ.get(name, "").strip()), "")
    if not key:
        raise ConfigurationError(
            f"{PROVIDER_ENV}=anthropic but none of {', '.join(ANTHROPIC_KEY_ENVS)} is set."
        )
    try:
        from langchain_anthropic import ChatAnthropic  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment problem
        raise ConfigurationError(
            "langchain-anthropic is not installed - `uv pip install langchain-anthropic`"
        ) from exc

    return ChatAnthropic(
        model=anthropic_model(),
        api_key=key,
        # Thinking counts against this on the current models, so the planner's
        # short budget is raised rather than reused: see HBU_PLANNER_MAX_TOKENS.
        max_tokens=max_new_tokens or DEFAULT_MAX_NEW_TOKENS,
        effort=os.environ.get(ANTHROPIC_EFFORT_ENV, "") or DEFAULT_ANTHROPIC_EFFORT,
        # One cache breakpoint, placed by the server at the end of the request.
        # The tool list and the system prompt are the same from one model call
        # to the next inside a turn, and from one turn to the next, so every
        # call after the first reads them from cache rather than paying for
        # them again. Harmless when the prefix is too short to cache.
        model_kwargs={"cache_control": {"type": "ephemeral"}},
        callbacks=callbacks,
        disable_streaming=_disable_streaming(),
    )
