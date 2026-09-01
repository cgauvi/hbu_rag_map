#!/usr/bin/env python3
"""
doctor.py — Answer "why is nothing showing up" in one command.

Three repos have to agree before this app draws anything: `hbu_infra` provisions
the database and creates the tables, `hbu_dataplatform` fills ``rag.chunks`` and
loads ``rag.lots``, ``rag.features`` and ``rag.buildings``. Each can be
half-done, and the symptom of any of them is the same empty map — so this walks
the chain in order and reports the first thing that is not there, with the
command that fixes it.

Two of the checks are advisory rather than faults. The ``silver`` joins are
tables the pipeline precomputes; without them the Lot pane still answers, by
computing the intersection per click. They are reported so an operator can see
the pipeline has not run over this borough yet, and they do not count toward
the problem total.

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
    #: (name, present, how to fix it, required). `required=False` is a table
    #: whose absence costs speed rather than an answer — see the module
    #: docstring — so it prints as a warning and is not counted.
    checks = [
        ("postgis extension", caps.postgis,
         "make db-init ENV=dev   (in hbu_infra — needs rds_superuser)", True),
        ("PostGIS 3.1+ (ST_AsMVT)", caps.mvt,
         "this PostGIS is too old for vector tiles, so the map falls back to "
         "fetching every shape in the viewport as GeoJSON, capped at "
         f"{queries.DEFAULT_FEATURE_LIMIT} per layer — which does not survive a "
         "whole borough. RDS ships 3.4 on postgres16 and the local container is "
         "built from postgis/postgis:16-3.4; upgrade the instance", False),
        ("vector extension", caps.pgvector,
         "make db-init ENV=dev   (in hbu_infra)", True),
        (f"{queries.SCHEMA}.lots", caps.lots,
         "created by hbu_infra sql/002_spatial.sql; needs a loader to fill it", True),
        (f"{queries.SCHEMA}.features", caps.features,
         "created by hbu_infra sql/002_spatial.sql; needs a loader to fill it", True),
        (f"{queries.SCHEMA}.buildings", caps.buildings,
         "created by hbu_infra sql/002_spatial.sql; needs a loader to fill it", True),
        (f"{queries.SILVER_SCHEMA}.building_lot_intersections", caps.building_lots,
         "hbu_infra sql/004_silver_building_lots.sql, filled by the "
         "dataplatform's building_lot_intersections asset; without it the Lot "
         "pane computes the overlap per click instead", False),
        (f"{queries.SILVER_SCHEMA}.lot_features", caps.lot_features,
         "hbu_infra sql/005_silver_lot_features.sql, filled by the same asset; "
         "without it the zoning a lot falls under is intersected per click", False),
        (f"{queries.GOLD_SCHEMA}.lot_building_massing", caps.massing,
         "hbu_infra sql/022_gold_lot_building_massing.sql, filled by the "
         "dataplatform's lot_building_massing asset (make massing); without it "
         "the Proposed massing layer is disabled and every other layer is "
         "unaffected", False),
        (f"{queries.GOLD_SCHEMA}.lot_highest_best_use", caps.highest_best_use,
         "hbu_infra sql/018_gold_lot_highest_best_use.sql, filled by the "
         "dataplatform's lot_highest_best_use asset (make hbu); without it the "
         "Lot pane still compares floor areas but cannot name the storeys, "
         "height or unit mix behind the proposed side", False),
        (f"{queries.GOLD_SCHEMA}.lot_redevelopment_gap", caps.redevelopment_gap,
         "hbu_infra sql/019_gold_lot_redevelopment_gap.sql, filled by the same "
         "asset run (make hbu); without it the Utilisation layer and the "
         "Capacity pane are both disabled — it is the table that compares what "
         "stands on a lot against what its zoning would hold", False),
        (f"{queries.SCHEMA}.chunks", caps.chunks,
         "hbu_dataplatform: make publish DATE=... NEIGHBORHOOD=...", True),
        (f"{queries.SCHEMA}.search_at_lot()", caps.search_at_lot,
         "hbu_infra sql/003_spatial_search.sql — skipped until rag.chunks exists, "
         "so re-run `make db-init` after the first publish", True),
        (f"{queries.SCHEMA}.search_near()", caps.search_near,
         "same as above", True),
    ]
    for name, present, fix, required in checks:
        if present:
            print(f"  {OK} {name}")
            continue
        if required:
            problems += 1
        print(f"  {NO if required else WARN} {name}")
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
