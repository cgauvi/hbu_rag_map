#!/usr/bin/env python3
"""
doctor.py — Answer "why is nothing showing up" in one command.

Three repos have to agree before this app draws anything: `hbu_infra` provisions
the database and creates the geometry tables, `hbu_dataplatform` fills
``rag.chunks``, and something has to load ``rag.lots``, ``rag.features`` and
``rag.buildings``. Each can be half-done, and the symptom of any of them is the
same empty map — so this walks the chain in order and reports the first thing
that is not there, with the command that fixes it.

    make check          # or: python scripts/doctor.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

_TTY = sys.stdout.isatty()
OK = "\033[32m✓\033[0m" if _TTY else "[ok]"
NO = "\033[31m✗\033[0m" if _TTY else "[--]"
WARN = "\033[33m!\033[0m" if _TTY else "[!!]"
DIM = "\033[2m" if _TTY else ""
RESET = "\033[0m" if _TTY else ""


def note(text: str) -> None:
    print(f"    {DIM}{text}{RESET}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    problems = 0

    # --- environment ------------------------------------------------------
    print("\nEnvironment")
    if os.environ.get("HUGGINGFACE_API_TOKEN"):
        print(f"  {OK} HUGGINGFACE_API_TOKEN is set")
    else:
        problems += 1
        print(f"  {NO} HUGGINGFACE_API_TOKEN is not set")
        note("The map works without it; the chat and the corpus search do not.")
        note("cp .env.example .env, then fill it in.")

    from src.utils.db import resolve  # noqa: PLC0415
    from src.utils.embeddings import embedding_model  # noqa: PLC0415

    try:
        details = resolve()
        print(f"  {OK} endpoint resolved from {details.source}")
        note(details.url())
    except Exception as exc:  # noqa: BLE001
        print(f"  {NO} cannot resolve an endpoint: {exc}")
        note("Set DATABASE_URL, or run `make db-up` for a local container.")
        return 1

    # --- connectivity -----------------------------------------------------
    print("\nDatabase")
    from src.utils.db import ping  # noqa: PLC0415

    if not ping():
        print(f"  {NO} cannot connect to {details.host}:{details.port}")
        note("If this is RDS: is your IP still in the security group?")
        note("`allow_current_ip = true` allowlists the machine that applied — "
             "re-apply after your ISP hands you a new address.")
        return 1
    print(f"  {OK} connected as {details.user}")

    from src.utils import queries  # noqa: PLC0415

    caps = queries.capabilities()
    checks = [
        ("postgis extension", caps.postgis,
         "make db-init ENV=dev   (in hbu_infra — needs rds_superuser)"),
        ("vector extension", caps.pgvector,
         "make db-init ENV=dev   (in hbu_infra)"),
        (f"{queries.SCHEMA}.lots", caps.lots,
         "created by hbu_infra sql/002_spatial.sql; needs a loader to fill it"),
        (f"{queries.SCHEMA}.features", caps.features,
         "created by hbu_infra sql/002_spatial.sql; needs a loader to fill it"),
        (f"{queries.SCHEMA}.buildings", caps.buildings,
         "created by hbu_infra sql/002_spatial.sql; needs a loader to fill it"),
        (f"{queries.SCHEMA}.building_lots", caps.building_lots,
         "hbu_infra sql/004_building_lots.sql, filled by the dataplatform's "
         "urban_rag.postgis.compute_intersections; without it the Lot pane "
         "computes the overlap per query instead"),
        (f"{queries.SCHEMA}.chunks", caps.chunks,
         "hbu_dataplatform: make publish DATE=... NEIGHBORHOOD=..."),
        (f"{queries.SCHEMA}.search_at_lot()", caps.search_at_lot,
         "hbu_infra sql/003_spatial_search.sql — skipped until rag.chunks exists, "
         "so re-run `make db-init` after the first publish"),
        (f"{queries.SCHEMA}.search_near()", caps.search_near,
         "same as above"),
    ]
    for name, present, fix in checks:
        if present:
            print(f"  {OK} {name}")
        else:
            problems += 1
            print(f"  {NO} {name}")
            note(fix)

    # --- what is actually loaded -----------------------------------------
    if caps.lots or caps.features or caps.buildings:
        print("\nLoaded geometry")
        for table in ("lots", "buildings", "features"):
            if not getattr(caps, table):
                continue
            try:
                rows = queries.query(
                    f"SELECT neighborhood, scrape_date, count(*) AS n "
                    f"FROM {queries.SCHEMA}.{table} "
                    f"GROUP BY 1, 2 ORDER BY 2 DESC, 1 LIMIT 5"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  {WARN} {table}: {exc}")
                continue
            if not rows:
                problems += 1
                print(f"  {WARN} {table}: the table exists but is empty")
                continue
            for row in rows:
                print(f"  {OK} {table}: {row['neighborhood']} {row['scrape_date']} "
                      f"— {row['n']:,} row(s)")

    # --- retrieval --------------------------------------------------------
    if caps.can_retrieve:
        print("\nRetrieval")
        dimension, model = queries.corpus_dimension(), queries.corpus_model()
        print(f"  {OK} corpus: {model or 'unknown model'}, "
              f"{dimension or 'unknown'} dimensions")
        configured = embedding_model()
        if model and model != configured:
            problems += 1
            print(f"  {NO} this app is configured for {configured}")
            note("Querying across two encoders returns confident nonsense "
                 "rather than an error.")
            note(f"Set URBAN_RAG_EMBEDDING_MODEL={model}")
        elif os.environ.get("HUGGINGFACE_API_TOKEN"):
            from src.utils.embeddings import EmbeddingError, embed_query  # noqa: PLC0415

            try:
                width = len(embed_query("hauteur maximale en mètres"))
                if dimension and width != dimension:
                    problems += 1
                    print(f"  {NO} {configured} returns {width} dimensions, "
                          f"corpus holds {dimension}")
                else:
                    print(f"  {OK} {configured} answers, {width} dimensions")
            except EmbeddingError as exc:
                problems += 1
                print(f"  {NO} the encoder did not answer: {exc}")

    print()
    if problems:
        print(f"{WARN} {problems} thing(s) to fix. The app runs with whatever is present.")
    else:
        print(f"{OK} Everything this app reads is in place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
