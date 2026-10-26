"""The three builds under comparison, as the environment that selects each.

A configuration is nothing but environment variables: the app already chooses
its chat provider (`src.config`), its encoder (`src.utils.embeddings`) and
its schema (`src.utils.db`) from them. Setting them before the app is imported
is what makes a run of one configuration *be* that build, with no branch in
the app that exists for the benchmark's sake.

Which is also why a configuration runs in its own process (`python -m eval
run --config X`): the schema is read at import time, the agent and the
encoder are cached at module level, and a process that has run one build
would carry pieces of it into the next.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

#: The schema holding the corpus re-embedded with a second encoder: views
#: over ``rag`` with the embedding column swapped. See ``shadow.py``.
SHADOW_SCHEMA = "rag_eval"


@dataclass(frozen=True)
class RagConfig:
    name: str
    label: str
    orchestrator: str  # huggingface | anthropic
    chat_model: str  # an HF alias / repo id, or an Anthropic model id
    embedder: str  # huggingface | voyage
    embedding_model: str
    schema: str  # rag | rag_eval

    @property
    def closed_orchestrator(self) -> bool:
        return self.orchestrator == "anthropic"

    @property
    def closed_embedder(self) -> bool:
        return self.embedder == "voyage"

    def environment(self) -> dict[str, str]:
        env = {
            "HBU_LLM_PROVIDER": self.orchestrator,
            "URBAN_RAG_EMBEDDING_PROVIDER": self.embedder,
            "URBAN_RAG_EMBEDDING_MODEL": self.embedding_model,
            "URBAN_RAG_PG_SCHEMA": self.schema,
            # One request per model call, so usage is reported on every one.
            "HBU_LLM_DISABLE_STREAMING": "1",
            # A script has no spinner to watch; the agent's own wall clock is
            # the ceiling the runner enforces anyway.
            "HBU_AGENT_TURN_BUDGET_S": os.environ.get("HBU_AGENT_TURN_BUDGET_S", "180"),
        }
        if self.closed_orchestrator:
            env["HBU_ANTHROPIC_MODEL"] = self.chat_model
            # Adaptive thinking spends from the same budget as the reply, and
            # the planner's 768 is sized for gpt-oss's reasoning, not Claude's.
            env.setdefault("HBU_PLANNER_MAX_TOKENS", "2048")
        else:
            env["HF_MODEL_ID"] = self.chat_model
        return env


OPEN_MODEL = os.environ.get("HBU_EVAL_OPEN_MODEL", "gpt-oss-120b")
CLAUDE_MODEL = os.environ.get("HBU_EVAL_CLAUDE_MODEL", "claude-sonnet-5-5")

CONFIGS: dict[str, RagConfig] = {
    "open": RagConfig(
        name="open",
        label=f"open: {OPEN_MODEL} + bge-m3",
        orchestrator="huggingface",
        chat_model=OPEN_MODEL,
        embedder="huggingface",
        embedding_model="BAAI/bge-m3",
        schema="rag",
    ),
    "claude-bge": RagConfig(
        name="claude-bge",
        label=f"closed orchestrator, open encoder: {CLAUDE_MODEL} + bge-m3",
        orchestrator="anthropic",
        chat_model=CLAUDE_MODEL,
        embedder="huggingface",
        embedding_model="BAAI/bge-m3",
        schema="rag",
    ),
    "claude-voyage": RagConfig(
        name="claude-voyage",
        label=f"closed: {CLAUDE_MODEL} + voyage-3.5",
        orchestrator="anthropic",
        chat_model=CLAUDE_MODEL,
        embedder="voyage",
        embedding_model="voyage-3.5",
        schema=SHADOW_SCHEMA,
    ),
}

#: The encoder-only comparison needs no chat model; these are the two sides.
EMBEDDERS: dict[str, RagConfig] = {
    "bge": CONFIGS["claude-bge"],
    "voyage": CONFIGS["claude-voyage"],
}


def get(name: str) -> RagConfig:
    try:
        return CONFIGS[name]
    except KeyError:
        raise SystemExit(f"no configuration named {name!r}; one of {', '.join(CONFIGS)}") from None


def apply(config: RagConfig) -> None:
    """Put a configuration into this process's environment.

    Call it before anything under ``src`` is imported - the schema and the
    provider are read at import time, which is the whole reason a
    configuration gets a process of its own.
    """
    for key, value in config.environment().items():
        os.environ[key] = value
