"""The comparison harness: the same questions through three builds of the assistant.

Three configurations of the zoning assistant, differing only in who answers
and who encodes:

    open           gpt-oss-120b on the HuggingFace Inference API + bge-m3
    claude-bge     Claude (Anthropic API) + bge-m3
    claude-voyage  Claude + voyage-3.5, over a copy of the corpus re-embedded
                   with it (the ``rag_eval`` shadow schema)

and one question set - ``tests/fixtures/agent_eval.yaml`` - whose every row
carries what the database says the answer is, so a run is a number and not an
impression: retrieval recall / MRR / nDCG against the governing document,
whether the answer states the storey count the grid states, whether it cites
what it retrieved, a faithfulness judge on the prose, and beside those the
things a budget feels - latency, tokens, dollars.

Nothing here reimplements the assistant. Every case goes through
``src.agent.stream_agent`` exactly as the chat pane does, with the lot
"clicked" first through ``src.utils.state``; the harness only chooses the
provider through the environment and reads usage off the model calls through
a LangChain callback. See ``eval/README.md`` for how to run it and what each
number means.

    python -m eval --help
"""
