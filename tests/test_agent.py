"""The agent wrapper around the tools: retries, truncation, and citations.

Three things are being checked here, none of which needs a model.

The first is that a tool which keeps failing stops being called, and that a
tool result too long for the context is cut rather than passed on whole — both
are `_wrap_tool`'s job, and both are invisible until a turn burns itself out on
one broken tool.

The second is that a citation number in the answer means a passage that was
actually retrieved. That is the one RAG failure that can be checked mechanically
rather than read for, so it is checked here instead of hoped for in the prompt.

The third is that the prompt still describes the tools that exist. The tool
index used to be hand-written and had already drifted past the tool list.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from pydantic import BaseModel

from src import agent
from src.utils import state


class _Args(BaseModel):
    pass


def _tool(func, name="probe"):
    return StructuredTool.from_function(
        func=func, name=name, description="a probe", args_schema=_Args
    )


# ---------------------------------------------------------------------------
# _wrap_tool
# ---------------------------------------------------------------------------


def test_a_failing_tool_is_given_up_on_rather_than_retried_forever():
    calls = []

    def boom():
        calls.append(1)
        raise RuntimeError("the database is asleep")

    wrapped = agent._wrap_tool(_tool(boom))
    agent._reset_tool_error_counts()

    messages = [wrapped.func() for _ in range(agent._MAX_TOOL_RETRIES + 2)]

    # The tool stops being called once its budget is spent, and the last
    # message tells the model to report rather than try again.
    assert len(calls) == agent._MAX_TOOL_RETRIES
    assert "the database is asleep" in messages[0]
    assert "already failed" in messages[-1]


def test_the_first_failure_says_how_many_tries_are_left():
    def boom():
        raise RuntimeError("no")

    wrapped = agent._wrap_tool(_tool(boom))
    agent._reset_tool_error_counts()

    assert "2 left" in wrapped.func()


def test_a_long_tool_result_is_truncated_and_says_so():
    wrapped = agent._wrap_tool(_tool(lambda: "x" * (agent._MAX_TOOL_OUTPUT_CHARS + 500)))
    agent._reset_tool_error_counts()

    result = wrapped.func()

    assert result.endswith("…(truncated)")
    assert len(result) < agent._MAX_TOOL_OUTPUT_CHARS + 50


def test_a_retrieval_that_fills_its_budget_is_not_truncated():
    """The cap has to clear what one retrieval can return, or widening it does
    nothing and nothing says so."""
    from src.tools import rag_tools

    worst_case = rag_tools.MAX_MATCHES * rag_tools.CHUNK_PREVIEW_CHARS
    assert agent._MAX_TOOL_OUTPUT_CHARS > worst_case


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


def test_an_answer_citing_a_retrieved_passage_is_left_alone():
    state.record_citation({"chunk_id": "c1"}, query="hauteur", scope="lot")

    answer = "The limit is 23 m [1]."

    assert agent._check_citations(answer) == answer


def test_an_answer_citing_a_passage_nobody_retrieved_is_flagged():
    state.record_citation({"chunk_id": "c1"}, query="hauteur", scope="lot")

    flagged = agent._check_citations("The limit is 23 m [1], and parking is free [4].")

    assert "[4]" in flagged
    assert "does not correspond" in flagged
    # The answer itself survives: a bad citation is reported, never edited out,
    # because deleting it would leave the sentence looking sourced.
    assert "parking is free [4]" in flagged


def test_several_invented_citations_are_reported_together():
    state.record_citation({"chunk_id": "c1"})

    flagged = agent._check_citations("a [2] b [3] c [1]")

    assert "[2], [3]" in flagged and "do not correspond" in flagged


def test_an_answer_with_no_citations_is_left_alone():
    state.record_citation({"chunk_id": "c1"})
    answer = "I could not find anything about that in the by-law."

    assert agent._check_citations(answer) == answer


def test_a_four_digit_bracket_is_not_read_as_a_citation():
    """A quoted by-law title carries years. `[2024]` is not a citation."""
    assert agent._check_citations("Règlement [2024] applies.") == "Règlement [2024] applies."


# ---------------------------------------------------------------------------
# History trimming
# ---------------------------------------------------------------------------
#
# The invariant that matters here is pairing: an AIMessage carrying tool_calls
# and the ToolMessages answering it must be kept or dropped together, or the
# next request to the endpoint is malformed. Everything else is quality.


def _turn(question: str, tool: str = "zoning_for_lot", answer: str = "done"):
    """One user turn with a tool call in it, as the graph would record it."""
    call_id = f"call_{abs(hash(question)) % 10000}"
    return [
        HumanMessage(content=question),
        AIMessage(
            content="",
            tool_calls=[{"name": tool, "args": {}, "id": call_id}],
        ),
        ToolMessage(content=f"{tool} says things", tool_call_id=call_id, name=tool),
        AIMessage(content=answer),
    ]


def _conversation(turns: int):
    messages = []
    for i in range(turns):
        messages += _turn(f"question {i}", answer=f"answer {i}.")
    return messages


def _orphans(messages) -> list[str]:
    """Tool results with no call, and calls with no result."""
    answered = {
        m.tool_call_id for m in messages if isinstance(m, ToolMessage)
    }
    requested = {
        call["id"]
        for m in messages
        if isinstance(m, AIMessage)
        for call in (getattr(m, "tool_calls", None) or [])
    }
    return sorted((answered - requested) | (requested - answered))


def test_a_short_conversation_is_left_alone():
    messages = _conversation(2)

    assert agent._trimmed(messages) == messages


def test_trimming_never_orphans_a_tool_result():
    """The one way this can produce an API error rather than a worse answer."""
    kept = agent._trimmed(_conversation(8))

    assert not _orphans(kept), _orphans(kept)


def test_the_recent_turns_survive_whole():
    messages = _conversation(8)

    kept = agent._trimmed(messages)

    # The tail is the last _TAIL_TURNS user turns and everything they pulled in.
    assert kept[1:] == messages[-agent._TAIL_TURNS * 4:]
    assert sum(isinstance(m, HumanMessage) for m in kept) == agent._TAIL_TURNS


def test_the_numbers_from_dropped_turns_survive_as_one_line():
    """What the old 300-char truncation destroyed, and the only thing a
    follow-up question reliably needs out of twenty dropped messages."""
    messages = [
        HumanMessage(content="what about lot 2 170 935"),
        AIMessage(content="Lot 2 170 935 is in zone H04-072 and is assessed at $412,000."),
    ]
    messages += _conversation(5)

    fold = agent._trimmed(messages)[0]

    assert "2 170 935" in fold.content
    assert "H04-072" in fold.content
    assert "$412,000" in fold.content


def test_the_fold_is_one_message_with_no_tool_calls():
    """Folding a whole span into one message is what makes pairing safe by
    construction rather than by bookkeeping."""
    fold = agent._trimmed(_conversation(8))[0]

    assert isinstance(fold, AIMessage)
    assert not getattr(fold, "tool_calls", None)
    assert "Summary of earlier conversation" in fold.content


def test_an_over_budget_conversation_sheds_tool_results_not_the_facts(monkeypatch):
    monkeypatch.setattr(agent, "_HISTORY_BUDGET", 400)
    messages = [
        HumanMessage(content="lot 2 170 935?"),
        AIMessage(content="Zone H04-072."),
    ]
    messages += _conversation(4)
    # A tool result far over the budget on its own.
    messages[-2] = ToolMessage(
        content="x" * 20_000,
        tool_call_id=messages[-3].tool_calls[0]["id"],
        name="zoning_for_lot",
    )

    kept = agent._trimmed(messages)

    assert "…(trimmed)" in str(kept[-2].content)
    assert "2 170 935" in kept[0].content  # the facts line is never shed
    assert not _orphans(kept)


def test_the_hook_shapes_the_call_without_rewriting_state():
    result = agent._trim_hook({"messages": _conversation(8)})

    assert "llm_input_messages" in result
    # Short of the hard-prune threshold the checkpoint is left as it is, so the
    # whole transcript is still there next turn.
    assert "messages" not in result


def test_a_very_long_conversation_prunes_the_checkpoint_too(monkeypatch):
    monkeypatch.setattr(agent, "_HARD_PRUNE_AT", 10)

    result = agent._trim_hook({"messages": _conversation(8)})

    assert result["messages"][0].id == REMOVE_ALL_MESSAGES
    assert result["messages"][1:] == result["llm_input_messages"]


# ---------------------------------------------------------------------------
# The checkpointer
# ---------------------------------------------------------------------------


def test_the_checkpointer_forgets_its_oldest_conversation():
    """It is a process global in a server with no session-close callback, so
    unbounded means a leak that holds whole conversations for ever."""
    saver = agent._BoundedSaver(max_threads=2)

    def _put(thread: str):
        config = {"configurable": {"thread_id": thread, "checkpoint_ns": ""}}
        checkpoint = empty_checkpoint()
        saver.put(config, checkpoint, {"source": "input", "step": 1}, {})

    _put("a")
    _put("b")
    _put("c")

    assert list(saver._recent) == ["b", "c"]
    assert "a" not in saver.storage


def test_a_thread_written_again_is_not_the_oldest():
    saver = agent._BoundedSaver(max_threads=2)

    def _put(thread: str):
        config = {"configurable": {"thread_id": thread, "checkpoint_ns": ""}}
        saver.put(config, empty_checkpoint(), {"source": "input", "step": 1}, {})

    _put("a")
    _put("b")
    _put("a")   # a is now the most recent, so b is the one to go
    _put("c")

    assert list(saver._recent) == ["a", "c"]


# ---------------------------------------------------------------------------
# Turn input
# ---------------------------------------------------------------------------


def test_with_a_thread_only_the_new_message_is_sent(monkeypatch):
    """The rest is in graph state - that is what carrying tool results means."""
    monkeypatch.setattr(agent, "_USE_CHECKPOINT", True)

    payload, config = agent._turn_input("what about it?", [{"role": "user", "content": "hi"}], "t1")

    assert len(payload["messages"]) == 1
    assert config["configurable"]["thread_id"] == "t1"
    assert config["recursion_limit"] == agent._RECURSION_LIMIT


def test_without_a_thread_the_transcript_is_replayed(monkeypatch):
    """Better than inventing a shared thread, which would put two browser
    sessions into one conversation."""
    monkeypatch.setattr(agent, "_USE_CHECKPOINT", True)

    payload, config = agent._turn_input("q", [{"role": "user", "content": "hi"}], None)

    assert len(payload["messages"]) == 2
    assert "configurable" not in config


def test_the_checkpoint_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(agent, "_USE_CHECKPOINT", False)

    payload, config = agent._turn_input("q", [{"role": "user", "content": "hi"}], "t1")

    assert len(payload["messages"]) == 2
    assert "configurable" not in config


# ---------------------------------------------------------------------------
# A turn, end to end
# ---------------------------------------------------------------------------
#
# No network: a scripted model stands in for the endpoint, so the graph itself
# is what is under test - the hook, the checkpointer and the streaming loop
# wired together. This is where the claim "a follow-up can refer back to what
# the last turn found" either holds or does not.


class _ScriptedModel(BaseChatModel):
    """Replays a list of replies and records what it was asked each time."""

    replies: list = []
    seen: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        # The agent binds tools before calling; the script does not care which.
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        reply = self.replies.pop(0)
        return ChatResult(generations=[ChatGeneration(message=reply)])


@pytest.fixture
def scripted(monkeypatch):
    """An agent whose model is a script and whose only tool is harmless."""
    probe = StructuredTool.from_function(
        func=lambda: "the corpus holds 1981 chunks",
        name="probe",
        description="a probe",
        args_schema=_Args,
    )
    monkeypatch.setattr(agent, "ALL_TOOLS", [probe])

    def _build(replies):
        model = _ScriptedModel(replies=list(replies), seen=[])
        monkeypatch.setattr(agent, "build_llm", lambda: model)
        agent.reset_agent()
        return model

    yield _build
    agent.reset_agent()


def test_a_turn_runs_its_tool_and_answers(scripted):
    model = scripted(
        [
            AIMessage(content="", tool_calls=[{"name": "probe", "args": {}, "id": "c1"}]),
            AIMessage(content="The corpus holds 1981 chunks."),
        ]
    )

    events = list(agent.stream_agent("how big is the corpus", thread_id="t-run"))
    kinds = [e["type"] for e in events]

    assert "tool_start" in kinds and "tool_end" in kinds
    assert events[-1] == {"type": "final", "content": "The corpus holds 1981 chunks."}
    assert len(model.seen) == 2


def test_a_second_turn_sees_what_the_first_turn_found(scripted):
    """The point of the checkpointer, stated as a test.

    Before it, `_build_messages` rebuilt the transcript from {role, content}
    dicts that never held tool results, so every follow-up re-derived
    everything and the model could not refer back to a number it had just been
    given.
    """
    model = scripted(
        [
            AIMessage(content="", tool_calls=[{"name": "probe", "args": {}, "id": "c1"}]),
            AIMessage(content="1981 chunks."),
            AIMessage(content="Yes, still 1981."),
        ]
    )

    list(agent.stream_agent("how big is the corpus", thread_id="t-carry"))
    list(agent.stream_agent("are you sure?", thread_id="t-carry"))

    # What the model was asked on the second turn.
    second_turn = model.seen[-1]
    assert any(isinstance(m, ToolMessage) for m in second_turn), [type(m) for m in second_turn]
    assert any("1981 chunks" in str(m.content) for m in second_turn)


def test_two_browser_sessions_do_not_share_a_conversation(scripted):
    model = scripted(
        [
            AIMessage(content="first session"),
            AIMessage(content="second session"),
        ]
    )

    list(agent.stream_agent("hello", thread_id="t-a"))
    list(agent.stream_agent("hello", thread_id="t-b"))

    # The second session's model call carries only its own question.
    assert [m.content for m in model.seen[-1] if isinstance(m, HumanMessage)] == ["hello"]


def test_a_turn_past_its_budget_stops_and_says_so(scripted, monkeypatch):
    """A spent budget ends the turn at the next node boundary.

    With the trim hook in the graph that boundary comes before the model has
    answered, so the honest report is that nothing was finished - not "here is
    what I had", which would be pointing at an empty answer.
    """
    monkeypatch.setattr(agent, "_TURN_BUDGET_S", -1.0)
    scripted([AIMessage(content="an answer that never gets used")])

    final = [e for e in agent.stream_agent("q", thread_id="t-budget") if e["type"] == "final"]

    assert "without finishing" in final[-1]["content"]
    assert "What is above" not in final[-1]["content"]


def test_tokens_are_emitted_as_they_arrive(scripted, monkeypatch):
    monkeypatch.setattr(agent, "_STREAM_TOKENS", True)
    scripted([AIMessage(content="streamed answer")])

    events = list(agent.stream_agent("q", thread_id="t-tok"))
    tokens = "".join(e["content"] for e in events if e["type"] == "token")

    # The scripted model yields its reply in one chunk, so this checks the
    # wiring rather than the chunking: a token event reaches the UI at all.
    assert tokens == "streamed answer"


def test_token_events_can_be_switched_off(scripted, monkeypatch):
    monkeypatch.setattr(agent, "_STREAM_TOKENS", False)
    scripted([AIMessage(content="quiet answer")])

    events = list(agent.stream_agent("q", thread_id="t-quiet"))

    assert not [e for e in events if e["type"] == "token"]
    assert events[-1]["content"] == "quiet answer"


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------


def test_every_tool_the_agent_has_is_named_in_the_prompt():
    """The hand-written index had already lost two tools when this was added."""
    missing = [t.name for t in agent.ALL_TOOLS if t.name not in agent._SYSTEM_PROMPT]

    assert not missing, f"not described in the system prompt: {missing}"
