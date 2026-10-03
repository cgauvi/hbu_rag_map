"""The Address pane's reads: the prefix tsquery, the street search and the doors.

Unit tests, against a captured `queries.query`; the ranking and the timing
are pinned against hbu-dev in tests/test_addresses_integration.py.
"""

from __future__ import annotations

import pytest

from src.utils import queries


@pytest.fixture
def captured(monkeypatch):
    """Capture every SQL call and answer it with what the test put in ``rows``."""
    calls: list[tuple[str, dict | None]] = []
    rows: list[list[dict]] = []

    def fake_query(sql, params=None):
        calls.append((sql, params))
        return rows.pop(0) if rows else []

    monkeypatch.setattr(queries, "query", fake_query)
    monkeypatch.setattr(queries, "_street_directory_probe", None)
    monkeypatch.setattr(queries, "_street_directory_checked", None)
    return calls, rows


# --- street_tsquery ---------------------------------------------------------


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("rue du Cardinal-Rouleau", "cardinal:* & rouleau:*"),
        ("cardinal roule", "cardinal:* & roule:*"),
        ("Rouleau Cardinal", "rouleau:* & cardinal:*"),
        ("boul. St-Michel", "saint:* & michel:*"),
        ("14e Avenue", "14e:* & avenue:*"),
        # The leading type goes, as `street_key` drops it; a trailing one stays.
        ("Avenue François-1er", "francois:* & 1re:*"),
        ("Lajeunesse", "lajeunesse:*"),
    ],
)
def test_the_tsquery_is_every_significant_word_as_a_prefix(typed, expected):
    assert queries.street_tsquery(typed) == expected


@pytest.mark.parametrize("typed", ["rue du", "rue", "", "de la"])
def test_a_street_of_nothing_but_type_and_particles_has_no_tsquery(typed):
    assert queries.street_tsquery(typed) == ""


def test_tsquery_operators_typed_by_accident_do_not_reach_the_query():
    """``& | ! ( ) : *`` are tsquery syntax; a stray one would raise in Postgres."""
    assert queries.street_tsquery("card(inal & rou:leau!") == "cardinal:* & rouleau:*"


# --- search_streets ---------------------------------------------------------


def test_search_streets_reads_the_directory_with_both_arms(captured):
    calls, rows = captured
    rows.append([{"street_name": "Avenue Cardinal-Rouleau"}])
    found = queries.search_streets(
        "128 rue cardinal rouleau"[4:], bounds=(-71.3, 46.7, -71.2, 46.9),
        neighborhood="CIL", municipalities=["quebec"], limit=5,
    )
    assert [r["street_name"] for r in found] == ["Avenue Cardinal-Rouleau"]
    sql, params = calls[-1]
    assert f"{queries.SILVER_SCHEMA}.street_directory" in sql
    assert "to_tsquery('simple', %(tsquery)s)" in sql
    assert "word_similarity(%(core)s, a.street_key)" in sql
    assert params["tsquery"] == "cardinal:* & rouleau:*"
    assert params["core"] == "cardinal rouleau"
    assert params["cutoff"] == queries.STREET_SEARCH_SIMILARITY
    assert params["neighborhood"] == "CIL"
    assert params["municipalities"] == ["quebec"]
    assert (params["west"], params["south"], params["east"], params["north"]) == (
        -71.3, 46.7, -71.2, 46.9
    )
    assert params["limit"] == 5


def test_search_streets_ranks_meaning_before_place():
    """A street every word matched beats one merely in view, and view beats borough."""
    import inspect

    sql = inspect.getsource(queries.search_streets)
    order = sql[sql.index("ORDER BY"):]
    assert order.index("exact_name DESC") < order.index("all_words DESC")
    assert order.index("all_words DESC") < order.index("in_view DESC")
    assert order.index("in_view DESC") < order.index("in_borough DESC")
    assert order.index("in_borough DESC") < order.index("similarity DESC")


def test_search_streets_with_no_viewport_passes_nulls_and_no_borough(captured):
    calls, _rows = captured
    queries.search_streets("lajeunesse")
    _sql, params = calls[-1]
    assert params["west"] is None and params["north"] is None
    assert params["neighborhood"] is None
    assert params["municipalities"] is None


@pytest.mark.parametrize("typed", ["", "rue", "rue du", "   "])
def test_search_streets_refuses_a_street_that_names_nothing(captured, typed):
    calls, _rows = captured
    assert queries.search_streets(typed) == []
    assert calls == []


# --- doors_on_street --------------------------------------------------------


def test_doors_on_street_matches_the_stored_spelling_through_the_index(captured):
    calls, rows = captured
    rows.append([{"civic_number": 801}])
    found = queries.doors_on_street(
        "Avenue Cardinal-Rouleau", neighborhood="CIL", municipality="Québec",
        civic_number=128, civic_suffix=None, limit=6,
    )
    assert found == [{"civic_number": 801}]
    sql, params = calls[-1]
    assert "lower(a.street_name) = lower(%(street)s)" in sql
    assert params["street"] == "Avenue Cardinal-Rouleau"
    assert params["neighborhood"] == "CIL"
    assert params["municipality"] == "Québec"
    assert params["civic"] == 128
    assert params["suffix"] is None
    assert params["limit"] == 6


def test_doors_on_street_orders_the_asked_door_then_prefixes_then_nearest():
    import inspect

    sql = inspect.getsource(queries.doors_on_street)
    order = sql[sql.index("ORDER BY"):]
    assert order.index("exact_number DESC") < order.index("number_prefix DESC")
    assert order.index("number_prefix DESC") < order.index("abs(a.civic_number")
    # A literal percent for the LIKE, doubled for the driver.
    assert "|| '%%'" in sql


def test_doors_on_street_without_a_number_still_runs(captured):
    calls, _rows = captured
    queries.doors_on_street("Rue Lajeunesse", neighborhood="VSMPE")
    _sql, params = calls[-1]
    assert params["civic"] is None
    assert params["municipality"] is None


# --- the directory and its refresh ------------------------------------------


def test_street_directory_reads_the_view_when_present(captured, monkeypatch):
    calls, rows = captured
    monkeypatch.setattr(queries, "scalar", lambda *_a, **_k: True)
    rows.append([{"street_name": "Rue Lajeunesse"}])
    assert queries.street_directory(neighborhood="VSMPE") == [{"street_name": "Rue Lajeunesse"}]
    sql, params = calls[-1]
    assert f"{queries.SILVER_SCHEMA}.street_directory a" in sql
    assert "GROUP BY" not in sql
    assert params["neighborhood"] == "VSMPE"


def test_street_directory_falls_back_to_grouping_the_points(captured, monkeypatch):
    calls, _rows = captured
    monkeypatch.setattr(queries, "scalar", lambda *_a, **_k: False)
    queries.street_directory()
    sql, _params = calls[-1]
    assert f"{queries.SILVER_SCHEMA}.lot_addresses a" in sql
    assert "GROUP BY a.neighborhood, a.municipality, a.street_name" in sql


def test_refresh_runs_only_when_the_table_is_newer_and_once_per_ttl(captured, monkeypatch):
    calls, _rows = captured
    answers = iter([True, True])  # the view exists; it is stale

    monkeypatch.setattr(queries, "scalar", lambda *_a, **_k: next(answers))
    assert queries.refresh_street_directory() is True
    assert any("REFRESH MATERIALIZED VIEW CONCURRENTLY" in sql for sql, _ in calls)
    # Memoised: the next call within the TTL does not even look.
    monkeypatch.setattr(queries, "scalar", lambda *_a, **_k: pytest.fail("looked again"))
    assert queries.refresh_street_directory() is False


def test_refresh_does_nothing_when_the_view_is_current(captured, monkeypatch):
    calls, _rows = captured
    answers = iter([True, False])  # exists; not stale
    monkeypatch.setattr(queries, "scalar", lambda *_a, **_k: next(answers))
    assert queries.refresh_street_directory() is False
    assert calls == []


def test_refresh_does_nothing_on_a_database_without_the_view(captured, monkeypatch):
    calls, _rows = captured
    monkeypatch.setattr(queries, "scalar", lambda *_a, **_k: False)
    assert queries.refresh_street_directory(force=True) is False
    assert calls == []


def test_the_capability_and_its_advisory_listing():
    caps = queries.Capabilities(lots=True, features=True, lot_addresses=True)
    assert f"{queries.SILVER_SCHEMA}.street_directory" in caps.missing()
    assert f"{queries.SILVER_SCHEMA}.street_directory" not in caps.missing(
        include_advisory=False
    )
