# The comparison harness

The same questions, through three builds of the zoning assistant, scored the
same way. Nothing in here reimplements the assistant: every case goes through
`src.agent.stream_agent` with the lot "clicked" first, exactly as the chat
pane does. The harness chooses the build through the environment and reads
usage off the model calls through a LangChain callback.

| configuration   | orchestrator                              | query + corpus encoder | schema     |
|-----------------|-------------------------------------------|------------------------|------------|
| `open`          | gpt-oss-120b on the HuggingFace Inference API | bge-m3 (HF Inference API) | `rag`      |
| `claude-bge`    | Claude Sonnet 5.5 (Anthropic API)         | bge-m3                 | `rag`      |
| `claude-voyage` | Claude Sonnet 5.5                         | voyage-3.5 (Voyage AI) | `rag_eval` |

`HBU_EVAL_OPEN_MODEL` / `HBU_EVAL_CLAUDE_MODEL` swap the models; the names
stay.

## Running it

The two providers the harness adds are an extra: `uv pip install -e ".[dev,eval]"`
(langchain-anthropic, anthropic, voyageai). The app imports them only when
`HBU_LLM_PROVIDER` / `URBAN_RAG_EMBEDDING_PROVIDER` select them.

```
make eval-cases                         # the 38 questions, with their labels
make eval-run EVAL_CONFIG=open          # HuggingFace credits only
make eval-run EVAL_CONFIG=claude-bge    # Anthropic, under the cap
make eval-compare                       # all three, then eval_out/report.md
make eval-spend                         # what has been spent, per model
```

or `python -m eval --help`. Everything lands in `eval_out/` (gitignored):
`<config>/results.jsonl` (one row per case, written as it completes),
`<config>/traces/<case>_rep0.json` (the whole turn), `<config>/errors.jsonl`
(attempts that produced nothing scorable - a 429, a timeout, the budget -
never in `results.jsonl`, so a re-run picks them up), `spend.jsonl` (every
charge), `report.md`.

Re-running a configuration skips the cases already in `results.jsonl`.
`python -m eval regrade` recomputes the free checks on stored rows when a
check changes, without paying for the turns again.

### The budget

`--budget-usd` (default **$5.00**, `HBU_EVAL_BUDGET_USD`) caps **Anthropic
spend in total** across every run that writes to the same `eval_out/` -
runs, judge calls, retries. The check runs *before* each model call with a
worst-case price for that call (the last input again plus `max_tokens` of
output), so the cap is never crossed; a run that reaches it ends with a
`budget` error row and exit code 2. HuggingFace and Voyage spend is recorded
and reported, not capped - the Inference API does not report what it bills
(it routes to a provider), so those figures are list-price estimates and
carry a `~` in the report.

Measured 2026-10-19 on three pilot cases: Claude Sonnet 5.5 at
**$0.0165 / case** with the system prompt cached (19k tokens of prompt and
tool definitions read from cache on every call; $0.052 without the
breakpoint), ~3 model calls and 6-18 s a turn; gpt-oss-120b ~$0.007 / case
(estimate), 0.5-7 s a turn; the Haiku judge ~$0.0004 / case.

### The closed encoder

`claude-voyage` needs the corpus re-embedded with voyage-3.5 (11,359 chunks,
~4M tokens, about $0.25 at list price; Voyage's 200M free tokens cover it).
An account with **no payment method** is held to 3 requests and 10K tokens a
minute - `--free-tier` paces the job under that (one ~8K-token batch a
minute, and a 429 waits out the rest of the minute rather than retrying into
the same window: ~22 chunks/min, so the whole corpus is ~9 h and VSMPE + CIL,
the two boroughs the question set uses, ~2.7 h). Adding a payment method on the
Voyage dashboard lifts the limits to the standard ones and the same job
takes minutes; the free tokens still apply. Either way it resumes where it
stopped:

```
# VOYAGE_API_KEY=... in .env
make eval-reembed                       # resumable; eval.chunk_embeddings
make eval-reembed EVAL_ARGS="--free-tier --neighborhoods VSMPE,CIL"   # no payment method
make eval-shadow EVAL_ARGS=build        # the rag_eval schema
make eval-run EVAL_CONFIG=claude-voyage
make eval-retrieval EVAL_ARGS="--embedder bge"
make eval-retrieval EVAL_ARGS="--embedder voyage"
```

`rag_eval` is a *shadow* of `rag`: a pass-through view of every table,
`chunks` with its `embedding` column swapped for the Voyage vectors, and
every search function recreated from the catalog with `rag.` rewritten to
`rag_eval.` - so the hybrid ranking, the containment search and the radius
search are byte-for-byte the same SQL under both encoders, and the only
difference between the two runs is the vector. Nothing in `rag` is touched;
`make eval-shadow EVAL_ARGS=drop` removes it.

## The question set

`tests/fixtures/agent_eval.yaml`, **generated** from the database by
`python -m eval cases --write` and frozen: every `expect` is what the
database states (the storey count on the zone polygon or in the parsed grid,
the resolution whose polygon covers the lot, the sheet a zone cites), so the
labels cannot be wrong in a way the data is not. What is hand-written is the
question, the category, and which lots make a fair sample (one zone covering
≥ 90% of the lot, 200-2000 m², deterministic order by a seeded hash).

| category          | n | asked                                                   | scored on                                              |
|-------------------|---|---------------------------------------------------------|--------------------------------------------------------|
| `storeys`         | 8 | storeys min/max (and height, CIL) on the selected lot    | the stated numbers, the zone named                     |
| `uses`            | 5 | permitted uses on the lot                               | the zone and a usage code named                        |
| `coverage`        | 3 | maximum lot coverage                                    | the percentage, the zone                               |
| `ppcmoi`          | 6 | "what zoning change has occurred here" on a lot a projet particulier covers | the resolution retrieved and named     |
| `ppcmoi-negative` | 4 | the same, on a lot no resolution covers                 | no resolution retrieved or asserted                    |
| `combo`           | 2 | storeys and past change in one question                 | both                                                   |
| `corpus`          | 7 | a question with no lot (from `retrieval_eval.yaml`)      | a relevant passage shown; the zone named where one is  |
| `off-corpus`      | 3 | snow removal, library hours, dog licences               | the answer declines                                    |

## What the columns mean

**Retrieval** is scored on the passages the model was actually shown, in the
order it was shown them (the turn's citation ledger), at the document level:
`Doc in top 5`, `MRR`, `nDCG@10`. `Used retrieval` says how often the build
retrieved at all - a storeys question is usually answered off the grid values
without a search, and that is correct, so the retrieval columns only count
turns that retrieved.

**Correct (checks)** is every deterministic check on the case passing: a
value stated within 60 characters of a unit word ("3 storeys", "60 %"), never
anywhere - a storey count also looks like the last digit of a lot number; the
zone code named (with the typographic hyphens models copy in folded away); a
resolution number named; a decline where the corpus is silent; no resolution
retrieved or named where none applies.

**Correct (judge)** and **Faithful** come from `claude-haiku-5-5` reading the
answer against *all* the tool results of the turn and the reference facts,
through a fixed JSON schema (`eval/judge.py`). Faithful is 1 when every
factual claim is in the context, 0.5 when some are, 0 when the answer states
something the context does not hold. The judge never sees which build it is
grading.

**Latency** is wall clock for the whole turn; **tokens** and **cost** are
read off every model call's reported usage (`eval/pricing.py` has the rates
and the HF caveat) and the judge's spend is its own column.

Rates carry a 95% Wilson interval. With 38 cases a difference under about
±15 points is inside the noise; run `--reps 2` before reading one as real.

## A finding the harness surfaced

`rag.search_at_lot` (hbu_infra `sql/004_spatial_search.sql`) matches a
lot's zones to the chunks that cite them with `c.feature_ids ? f.feature_id`.
The `?` operator tests for a JSON *string*; Montreal's PPCMOI resolutions are
indexed with their polygon ids as JSON *numbers* (`[226]`), so the
containment search - the tool the prompt tells the model to prefer - never
returns a resolution, while `search_near` (which goes through
`jsonb_array_elements_text`) does. Every build fails the `ppcmoi` category
at baseline for this reason. The fix is one line
(`EXISTS (SELECT 1 FROM jsonb_array_elements_text(c.feature_ids) fid WHERE fid.value = f.feature_id)`),
and the harness is what tells whether it moved the number.
