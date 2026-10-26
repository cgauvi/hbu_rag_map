"""The encoder on its own: every case's question against the corpus, no chat model.

The agent-path numbers in ``runner.py`` measure the whole build. This
measures one component of it - the encoder and the ranking - by running each
question straight through the search functions the tools call, under each
encoder in turn, and scoring the ranked passages against the same labels.
No model chooses the tool or rephrases the question, so a difference here is
the encoder's and nothing else's.

Three scopes, the three the tools use:

    at_lot   ``rag.search_at_lot``  - containment: the lot's own documents
    near     ``rag.search_near``    - a 30 m radius around the lot's point
    corpus   ``rag.search_corpus``  - the borough (hybrid, lexical arm on)

A lot-anchored case is scored on all three; a corpus case on the last.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from eval import metrics
from eval.configs import RagConfig
from eval.golden import Case
from eval.spend import SpendMeter

K = 10
NEAR_RADIUS_M = 30


def run(config: RagConfig, cases: list[Case], *, out_dir: Path, meter: SpendMeter, progress=print) -> dict:
    """Score every case under ``config``'s encoder. Writes ``<embedder>.jsonl``."""
    from src.utils import embeddings as emb  # noqa: PLC0415
    from src.utils import queries  # noqa: PLC0415

    out = out_dir / f"{config.embedder}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    tokens_before = emb.USAGE["tokens"]
    progress(f"[retrieval:{config.embedder}] {config.embedding_model} over {config.schema}, {len(cases)} case(s)")
    with out.open("w", encoding="utf-8") as f:
        for case in cases:
            expect = case.expect
            if not (expect.get("doc_ids") or expect.get("passages_contain_any")):
                continue
            t0 = time.perf_counter()
            vector = emb.embed_query(case.question)
            embed_s = time.perf_counter() - t0
            scopes: dict[str, dict] = {}
            if case.lot:
                lon, lat = float(case.lot["lon"]), float(case.lot["lat"])
                t = time.perf_counter()
                hits = queries.search_at_lot(vector, lon, lat, match_count=K)
                scopes["at_lot"] = {**metrics.retrieval_scores(hits, expect, total_relevant=expect.get("relevant_chunks")), "search_s": round(time.perf_counter() - t, 4),
                                    "top": [metrics.doc_id_of(h) for h in hits[:5]]}
                t = time.perf_counter()
                hits = queries.search_near(vector, lon, lat, radius_m=NEAR_RADIUS_M, match_count=K)
                scopes["near"] = {**metrics.retrieval_scores(hits, expect, total_relevant=expect.get("relevant_chunks")), "search_s": round(time.perf_counter() - t, 4),
                                  "top": [metrics.doc_id_of(h) for h in hits[:5]]}
            t = time.perf_counter()
            hits = queries.search_corpus(vector, match_count=K, neighborhood=case.neighborhood, query_text=case.question)
            scopes["corpus"] = {**metrics.retrieval_scores(hits, expect, total_relevant=expect.get("relevant_chunks")), "search_s": round(time.perf_counter() - t, 4),
                                "top": [metrics.doc_id_of(h) for h in hits[:5]],
                                "top_similarity": float(hits[0].get("similarity") or 0.0) if hits and hits[0].get("similarity") is not None else None}
            row = {"prompt_id": case.id, "category": case.category, "neighborhood": case.neighborhood, "lang": case.lang,
                   "embedder": config.embedder, "model": config.embedding_model, "embed_s": round(embed_s, 4), "scopes": scopes}
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            progress(f"  {case.id:<40} " + "  ".join(f"{s}: hit@5={v.get('hit@5', 0):.0f} mrr={v.get('mrr', 0):.2f}" for s, v in scopes.items()))
    tokens = emb.USAGE["tokens"] - tokens_before
    if config.embedder == "voyage" and tokens:
        meter.charge(service="voyage", model=config.embedding_model, usage={"input_tokens": tokens}, role="retrieval-eval")
    return summarize(rows)


def summarize(rows: list[dict]) -> dict:
    """Mean of each metric per scope, over the cases that have that scope."""
    out: dict = {"cases": len(rows), "embed_s_median": metrics.median([r["embed_s"] for r in rows])}
    for scope in ("at_lot", "near", "corpus"):
        scored = [r["scopes"][scope] for r in rows if scope in r["scopes"]]
        if not scored:
            continue
        out[scope] = {"n": len(scored)}
        for key in ("hit@1", "hit@3", "hit@5", "hit@10", "mrr", "ndcg@10"):
            out[scope][key] = metrics.mean([s.get(key) for s in scored])
        out[scope]["search_s_median"] = metrics.median([s["search_s"] for s in scored])
    by_cat: dict[str, dict] = {}
    for r in rows:
        best = r["scopes"].get("at_lot") or r["scopes"].get("corpus") or {}
        by_cat.setdefault(r["category"], []).append(best.get("hit@5", 0.0))
    out["hit@5_by_category"] = {k: metrics.mean(v) for k, v in by_cat.items()}
    return out
