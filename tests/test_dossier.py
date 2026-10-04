"""The dossier: its registry, the SQL it composes, and the caveats it carries.

Three things are checked here, and the first is the one that matters most.

**Nothing the model says becomes SQL.** It names a column, which is checked
against the registry by exact membership; everything else it supplies is bound
as a parameter. So the test for injection is not "is this escaped properly" —
it is "does a hostile string reach the database at all", and the answer has to
be no, by refusal or by binding, for every input the tools take.

**The registry and the view agree.** They are written in two files, in two
languages, and a column renamed in one and not the other fails at runtime in
front of a user. The integration test compares them against the live view; the
unit tests below pin what can be checked without one.

**The caveats survive the trip through a table.** `lot_efficiency` says in
prose that an unstated floor area is not zero floor area. A result table that
dropped that would be a regression even with every number right.
"""

from __future__ import annotations

import datetime as dt
import os
import time

import pytest
from langchain_core.tools import ToolException
from psycopg import sql

from src.tools import data_tools
from src.utils import dossier, queries

#: Strings that must never become SQL, whatever a tool does with them. Given
#: ids because one of them is 10 000 characters, and pytest writes the whole
#: parameter into PYTEST_CURRENT_TEST — which Windows caps at 32 767.
HOSTILE = [
    pytest.param("'; DROP TABLE rag.lots; --", id="statement-break"),
    pytest.param("1 OR 1=1", id="always-true"),
    pytest.param("lot_number; DELETE FROM gold.lot_profiles", id="second-statement"),
    pytest.param("*", id="star"),
    pytest.param(
        "parcel_assessed_value_cad, (SELECT password FROM pg_shadow)",
        id="smuggled-subselect",
    ),
    pytest.param("%", id="format-char"),
    pytest.param("' UNION SELECT * FROM pg_authid --", id="union"),
    pytest.param("х" * 10_000, id="very-long-homoglyph"),
    pytest.param("lot_numbeʀ", id="latin-lookalike"),
    pytest.param("", id="empty"),
    pytest.param("   ", id="blank"),
]


# ---------------------------------------------------------------------------
# The registry is the allowlist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload", HOSTILE)
def test_a_hostile_column_name_is_refused_by_name(payload):
    with pytest.raises(dossier.UnknownColumn):
        dossier.validated(payload)


def test_a_near_miss_is_answered_with_the_nearest_real_columns():
    """A model corrects from a suggestion far better than from a rule."""
    with pytest.raises(dossier.UnknownColumn) as caught:
        dossier.validated("parcel_assessed_value")

    assert "parcel_assessed_value_cad" in caught.value.suggestions


def test_every_filter_names_a_column_that_exists():
    """A typo here would reach SQL as text, so it is a test rather than care."""
    for name, (column, _) in queries._SITE_FILTERS.items():
        assert column in dossier.COLUMN_NAMES, f"{name} -> {column}"
    for column in queries._USE_COLUMNS.values():
        assert column in dossier.COLUMN_NAMES
    for column in queries.GROUPABLE:
        assert column in dossier.COLUMN_NAMES


def test_the_partition_key_is_not_nameable():
    """The view selects cell_partition; the model has no business ordering by
    it, and absent from the registry means refused."""
    with pytest.raises(dossier.UnknownColumn):
        dossier.validated("cell_partition")


# ---------------------------------------------------------------------------
# What actually reaches the database
# ---------------------------------------------------------------------------


def _rendered(predicates, params) -> str:
    statement = sql.SQL("SELECT lot_number FROM {r} WHERE {w}").format(
        r=queries._dossier_relation(), w=predicates
    )
    return statement.as_string(None)


def test_values_are_bound_and_never_written_into_the_statement():
    predicates, params = queries._site_predicates(
        {"max_assessed_value_cad": 500000, "min_permitted_storeys": 4}, {}
    )
    rendered = _rendered(predicates, params)

    assert "500000" not in rendered
    assert "%(max_assessed_value_cad)s" in rendered
    assert params["max_assessed_value_cad"] == 500000


def test_identifiers_are_quoted_and_come_from_the_registry():
    predicates, _ = queries._site_predicates({"min_permitted_storeys": 4}, {})

    assert '"hbu_floors"' in _rendered(predicates, {})


@pytest.mark.parametrize("payload", HOSTILE[:6])
def test_a_hostile_thesis_is_refused_rather_than_matched(payload):
    with pytest.raises(ValueError):
        queries._site_predicates({}, {}, site_thesis=payload)


@pytest.mark.parametrize("payload", HOSTILE[:6])
def test_a_hostile_use_is_refused_rather_than_matched(payload):
    with pytest.raises(ValueError):
        queries._site_predicates({}, {}, use_permitted=payload)


def test_a_thesis_is_matched_by_membership_not_by_pattern():
    """`LIKE` on a closed set is how a wildcard becomes a wildcard."""
    with pytest.raises(ValueError):
        queries._site_predicates({}, {}, site_thesis="brown%")

    predicates, params = queries._site_predicates({}, {}, site_thesis="brownfield")
    assert params["site_thesis"] == "brownfield"
    assert "brownfield" not in _rendered(predicates, params)


def test_group_by_is_a_closed_set(monkeypatch):
    monkeypatch.setattr(queries, "_run_dossier", lambda *_a, **_k: [])
    with pytest.raises(ValueError, match="cannot be grouped on"):
        queries.summarize_sites(
            neighborhood="VSMPE", scrape_date=dt.date(2026, 9, 1),
            group_by="lot_number; DROP TABLE rag.lots",
        )


def test_a_lot_number_reaches_sql_as_digits_only(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        queries, "_run_dossier",
        lambda statement, params: captured.update(params=params) or [],
    )

    queries.site_dossier(
        "2 170 935'; DROP TABLE rag.lots; --",
        neighborhood="VSMPE", scrape_date=dt.date(2026, 9, 1),
    )

    # Everything that is not a digit is gone before it is even bound.
    assert captured["params"]["digits"] == "2170935"


def test_the_row_cap_cannot_be_raised_past_the_ceiling(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        queries, "_run_dossier",
        lambda statement, params: captured.update(params=params) or [],
    )

    queries.find_sites(
        neighborhood="VSMPE", scrape_date=dt.date(2026, 9, 1), limit=10_000
    )

    assert captured["params"]["limit"] == dossier.MAX_ROWS + 1


def test_a_result_always_carries_the_lot_number(monkeypatch):
    """A row the model cannot name is a row it will invent a name for."""
    captured = {}
    monkeypatch.setattr(
        queries, "_run_dossier",
        lambda statement, params: captured.update(sql=statement.as_string(None)) or [],
    )

    queries.find_sites(
        neighborhood="VSMPE", scrape_date=dt.date(2026, 9, 1),
        columns=["hbu_floors"],
    )

    assert '"lot_number"' in captured["sql"]


# ---------------------------------------------------------------------------
# The caveats
# ---------------------------------------------------------------------------


def test_an_unstated_floor_area_is_never_reported_as_zero():
    notes = dossier.footnotes({"floor_area_unreported": 3})

    assert len(notes) == 1
    assert "NOT KNOWN" in notes[0]
    assert "0%" in notes[0] and "vacant" in notes[0]


def test_a_split_parcel_is_called_out_as_a_piece():
    notes = dossier.footnotes({"split_parcel": 2})

    assert "PIECE" in notes[0]


def test_counts_of_zero_say_nothing():
    assert dossier.footnotes({"split_parcel": 0, "nothing_assessed": 0}) == []


def test_the_footnotes_survive_a_table_too_long_for_the_budget():
    """The agent truncates a long tool result from the end, which is where the
    notes are - so rows are shed instead."""
    table = "\n".join(f"row {i}" for i in range(4_000))
    notes = dossier.footnotes({"floor_area_unreported": 1})

    out = dossier.fit("header", table, notes, limit=4_000)

    assert "NOT KNOWN" in out
    assert "rows dropped" in out


def test_a_null_renders_differently_from_a_zero():
    """For this data that distinction is the whole game."""
    rows = [{"lot_number": "2 170 935", "existing_floor_area_m2": None},
            {"lot_number": "1 740 794", "existing_floor_area_m2": 0}]

    out = dossier.render(rows, ["lot_number", "existing_floor_area_m2"],
                         header="h", truncated=False)

    assert "—" in out
    assert "0" in out


# ---------------------------------------------------------------------------
# describe_data
# ---------------------------------------------------------------------------


def test_every_topic_describes_at_least_one_column():
    for topic in dossier.TOPICS:
        text = dossier.describe(topic)
        assert len(text.splitlines()) > 3, topic


def test_an_unknown_topic_falls_back_rather_than_failing():
    """A model picking a near-miss should get something useful, not an error."""
    assert dossier.describe("money!!") == dossier.describe("overview")
    assert dossier.describe("") == dossier.describe("overview")


def test_the_overview_states_the_grain_and_the_parcel_rule():
    text = dossier.describe("overview")

    assert "ZONE PIECE" in text
    assert "parcel_" in text and "Never add them up" in text


# ---------------------------------------------------------------------------
# The tools report missing data as missing data
# ---------------------------------------------------------------------------


def test_a_missing_dossier_names_the_file_that_creates_it(monkeypatch):
    monkeypatch.setattr(
        queries, "capabilities",
        lambda: queries.Capabilities(postgis=True, pgvector=True, redevelopment_gap=True),
    )
    monkeypatch.setattr(queries, "dossier_loaded", lambda: False)

    with pytest.raises(ToolException, match="034_gold_lot_dossier.sql"):
        data_tools.find_sites.invoke({})


def test_several_loaded_boroughs_and_no_selection_asks_rather_than_guesses(monkeypatch):
    """Guessing between boroughs answers confidently about the wrong city."""
    monkeypatch.setattr(
        queries, "capabilities",
        lambda: queries.Capabilities(postgis=True, pgvector=True, redevelopment_gap=True),
    )
    monkeypatch.setattr(queries, "dossier_loaded", lambda: True)
    monkeypatch.setattr(queries, "neighborhoods", lambda *_a, **_k: ["VSMPE", "CIL"])

    with pytest.raises(ToolException, match="which"):
        data_tools.find_sites.invoke({})


def test_comparing_one_lot_is_refused_with_the_tool_that_fits(monkeypatch):
    monkeypatch.setattr(
        queries, "capabilities",
        lambda: queries.Capabilities(postgis=True, pgvector=True, redevelopment_gap=True),
    )
    monkeypatch.setattr(queries, "dossier_loaded", lambda: True)

    with pytest.raises(ToolException, match="site_dossier"):
        data_tools.compare_sites.invoke({"lot_numbers": "2 170 935"})


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------
#
# The dossier is a materialized view, so "it is there" and "it is right" came
# apart the moment it stopped being a plain one. Only one direction of
# disagreement is a problem, and these pin which.


def _stale_rows(monkeypatch, rows):
    monkeypatch.setattr(queries, "query", lambda *_a, **_k: rows)
    return queries.dossier_stale()


def test_a_dossier_behind_its_gold_tables_is_reported(monkeypatch):
    assert _stale_rows(
        monkeypatch,
        [{"neighborhood": "VSMPE", "gold_date": "2026-10-01",
          "dossier_date": "2026-09-01"}],
    ) == [("VSMPE", "2026-10-01", "2026-09-01")]


def test_a_borough_the_dossier_has_never_seen_is_reported(monkeypatch):
    """A refresh that has never run reads the same as one that is behind."""
    assert _stale_rows(
        monkeypatch,
        [{"neighborhood": "SAG", "gold_date": "2026-09-01", "dossier_date": None}],
    ) == [("SAG", "2026-09-01", None)]


def test_a_borough_dropped_from_gold_is_not_reported(monkeypatch):
    """The dossier holding a borough gold no longer does is not staleness.

    Nothing a refresh would fix, so saying "refresh me" about it would send
    someone to rebuild a view that is already current.
    """
    assert _stale_rows(
        monkeypatch,
        [{"neighborhood": "OLD", "gold_date": None, "dossier_date": "2026-08-01"}],
    ) == []


# ---------------------------------------------------------------------------
# Against the real view
# ---------------------------------------------------------------------------
#
# The registry and the view are written in two files and two languages. A
# column renamed in one and not the other is invisible to every unit test here
# and fails in front of a user, so it is checked against the live view - which
# is also the only place the composed SQL is ever actually parsed by Postgres.


@pytest.fixture
def live(monkeypatch, database_url, real_hf_token):
    """Skip unless the dossier view exists in a reachable database.

    The DSN has to be put back: `_clean_environment` scrubs it from every
    test so the unit suite cannot open a socket, and the session-scoped
    `database_url` fixture is what captured it beforehand. Same arrangement as
    `test_addresses_integration`.
    """
    if not database_url:
        pytest.skip("set DATABASE_URL (or HBU_TEST_DATABASE_URL) to run these")
    monkeypatch.setenv("DATABASE_URL", database_url)
    # Some of these embed for real, and conftest scrubs the token too.
    if real_hf_token:
        monkeypatch.setenv("HUGGINGFACE_API_TOKEN", real_hf_token)
    from src.utils.db import close_pool

    close_pool()

    try:
        if not queries.dossier_loaded():
            pytest.skip("gold.lot_dossier is not in this database (sql/032)")
    except Exception as exc:  # noqa: BLE001 - no database is a skip, not a failure
        pytest.skip(f"no database: {exc}")
    loaded = queries.neighborhoods()
    if not loaded:
        pytest.skip("no borough is loaded")
    code = loaded[0]
    return code, queries.latest_scrape_date(table="lots", neighborhood=code)


@pytest.mark.integration
def test_the_registry_matches_the_view(live):
    """Neither side may gain or lose a column without the other."""
    # pg_attribute, not information_schema.columns: the SQL standard has no
    # materialized view, so information_schema does not list one, and this
    # assertion would pass vacuously with `actual` empty. It did, for one
    # commit, when sql/032 became materialized - every column read as
    # "described but absent" at once, which is at least a loud way to fail.
    actual = {
        row["column_name"]
        for row in queries.query(
            "SELECT a.attname AS column_name"
            "  FROM pg_attribute a"
            "  JOIN pg_class c ON c.oid = a.attrelid"
            "  JOIN pg_namespace n ON n.oid = c.relnamespace"
            " WHERE n.nspname = %s AND c.relname = %s"
            "   AND a.attnum > 0 AND NOT a.attisdropped",
            (queries.GOLD_SCHEMA, queries.DOSSIER_VIEW),
        )
    }
    assert actual, "the dossier reports no columns at all - wrong catalog?"
    described = set(dossier.COLUMN_NAMES)

    # cell_partition is in the view and deliberately not in the registry.
    assert described - actual == set(), f"described but absent: {described - actual}"
    assert actual - described == {"cell_partition"}, (
        f"in the view and undescribed: {sorted(actual - described - {'cell_partition'})}"
    )


@pytest.mark.integration
def test_a_site_search_runs_and_stays_inside_the_timeout(live):
    """R1: a borough-wide read of six joined tables has to fit the budget."""
    code, scrape_date = live
    started = time.monotonic()

    rows, _ = queries.find_sites(
        neighborhood=code,
        scrape_date=scrape_date,
        filters={"min_permitted_storeys": 4, "max_assessed_value_cad": 500_000},
        flags={"only_primary_zone": True, "exclude_heritage": True},
        limit=25,
    )
    elapsed = time.monotonic() - started

    # The statement timeout on the read-only pool is 15 s; this is the margin
    # that says the plan is a hash join over one borough and not a scan of
    # four. Loose on purpose - it is a breakage detector, not a benchmark.
    assert elapsed < 10, f"a borough-wide search took {elapsed:.1f}s"
    for row in rows:
        assert row["lot_number"]


@pytest.mark.integration
def test_the_read_only_connection_refuses_to_write(live):
    """The layer under the design: even if a statement got through, it cannot
    change anything."""
    import psycopg

    from src.utils.db import readonly_connection

    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        with readonly_connection() as conn, conn.cursor() as cur:
            cur.execute("CREATE TEMP TABLE should_not_exist (x int)")


@pytest.mark.integration
def test_an_unqualified_name_cannot_resolve(live):
    """`search_path = ''` is what makes a guessed table name fail cleanly."""
    import psycopg

    from src.utils.db import readonly_connection

    with pytest.raises(psycopg.errors.UndefinedTable):
        with readonly_connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 FROM lot_dossier LIMIT 1")


@pytest.mark.integration
def test_the_caveat_counts_agree_with_the_rows(live):
    """The warnings have to describe the rows that were actually returned."""
    code, scrape_date = live
    rows, _ = queries.find_sites(
        neighborhood=code, scrape_date=scrape_date,
        columns=["lot_number", "num_lot_zones", "floor_area_unreported"],
        limit=25,
    )
    if not rows:
        pytest.skip("no sites in this borough to check")

    counts = queries.dossier_caveats(
        [r["lot_number"] for r in rows], neighborhood=code, scrape_date=scrape_date
    )

    # The aggregate is keyed on lot_number, so it sees every piece of a lot a
    # row named - which is >= what the rows themselves show.
    assert counts["split_parcel"] >= sum(
        1 for r in rows if (r.get("num_lot_zones") or 1) > 1
    )
    assert set(counts) >= {
        "split_parcel", "floor_area_unreported", "nothing_assessed",
        "hbu_parking_waived", "unsolved", "grid_unparsed",
    }


# ---------------------------------------------------------------------------
# Hybrid retrieval
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_the_hybrid_function_is_detected(live):
    """The probe that was wrong, pinned.

    `hybrid_available` asked `to_regclass('rag.search_corpus')`, which is NULL
    for a FUNCTION however present it is — so the app ran the dense-only path
    and an A/B of the two reported, correctly, no difference at all. The eval
    is worthless while this is False, so it is worth its own test.
    """
    queries.hybrid_available.cache_clear()

    assert queries.hybrid_available() is True


@pytest.mark.integration
def test_the_lexical_arm_changes_the_ranking(live):
    """A rare exact term should rank differently once the arm fires."""
    from src.utils.embeddings import embed_query

    if not os.environ.get("HUGGINGFACE_API_TOKEN"):
        pytest.skip("needs the encoder: set HUGGINGFACE_API_TOKEN")
    queries.hybrid_available.cache_clear()
    # Verified to match: "mode d'implantation contigu" matches NOTHING,
    # because websearch_to_tsquery ANDs every term and the three do not
    # co-occur. A lexical arm that finds nothing is correct behaviour, not a
    # bug - so the probe has to be a question the corpus can actually answer.
    question = "implantation contigu"
    embedding = embed_query(question)

    dense = queries.search_corpus(embedding, match_count=10, neighborhood="SSC")
    hybrid = queries.search_corpus(
        embedding, match_count=10, neighborhood="SSC", query_text=question
    )

    assert hybrid, "the hybrid arm returned nothing"
    # Something scored lexically, which is the arm doing its job at all.
    assert any(row.get("lexical_rank") is not None for row in hybrid)
    # And the term is reached sooner than without it.
    def _rank(rows):
        for index, row in enumerate(rows, 1):
            if "contigu" in (row.get("chunk_text") or "").lower():
                return index
        return 99

    assert _rank(hybrid) <= _rank(dense)


@pytest.mark.integration
def test_an_accented_and_unaccented_question_retrieve_alike(live):
    """`to_tsvector('french')` does not fold accents; the index and the query
    both do it with `translate`, and the two have to agree."""
    rows = queries.query(
        "SELECT (SELECT count(*) FROM rag.chunks WHERE tsv @@ rag.corpus_tsquery(%s)) AS a, "
        "       (SELECT count(*) FROM rag.chunks WHERE tsv @@ rag.corpus_tsquery(%s)) AS b",
        ("marges latérales", "marges laterales"),
    )

    assert rows[0]["a"] == rows[0]["b"] > 0


@pytest.mark.integration
def test_an_unindexable_question_yields_no_tsquery_and_no_notice(live):
    """NULL rather than an empty tsquery, so the caller can tell "no lexical
    arm" from "a lexical arm that matched nothing" — and without the NOTICE
    the `''::tsquery` cast emitted on every English question."""
    rows = queries.query("SELECT rag.corpus_tsquery(%s) AS q", ("   ",))

    assert rows[0]["q"] is None
