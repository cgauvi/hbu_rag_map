"""Address lookup against a real database.

Marked ``integration``. The unit tests in `test_queries.py` pin what
`lots_by_address` sends; these check what it returns about addresses that
exist. The one thing no stub can say is whether the folding written in SQL
(``_STREET_KEY_SQL``) and the folding written in Python (`street_key`) agree,
and they are written twice on purpose - the query has to fold the stored name
where the row is, and the tool has to fold what was typed before it knows
whether to ask at all.

The addresses are read off the table rather than written down, so the tests
hold for whichever borough happens to be loaded.
"""

from __future__ import annotations

import re
import unicodedata

import pytest

from src.utils import places, queries

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _database_url(monkeypatch, database_url):
    if not database_url:
        pytest.skip("set DATABASE_URL (or HBU_TEST_DATABASE_URL) to run these")
    monkeypatch.setenv("DATABASE_URL", database_url)
    from src.utils.db import close_pool

    close_pool()


@pytest.fixture(autouse=True)
def _needs_addresses():
    caps = queries.capabilities()
    if not (caps.lots and caps.lot_addresses):
        pytest.skip("this database has no lots or no lot addresses")


def _one(sql: str, params: dict | None = None) -> dict | None:
    rows = queries.query(sql, params or {})
    return rows[0] if rows else None


@pytest.fixture
def a_door() -> dict:
    """One door on a street printed with its type - accented if the borough has one."""
    row = _one(
        f"""
        SELECT lot_number, neighborhood, civic_number, street_name, civic_address
          FROM {queries.SILVER_SCHEMA}.lot_addresses
         WHERE is_primary_address AND civic_suffix IS NULL
           AND street_name ~ '^(Rue|Avenue|Boulevard) '
         ORDER BY (street_name ~ '[àâäéèêëîïôöùûüç]') DESC, neighborhood, lot_number
         LIMIT 1
        """
    )
    if not row:
        pytest.skip("no primary address on a typed street")
    return row


def test_an_address_as_the_layer_prints_it_finds_its_lot(a_door):
    rows = queries.lots_by_address(
        a_door["street_name"], a_door["civic_number"], neighborhood=a_door["neighborhood"]
    )
    assert rows, a_door
    assert rows[0]["lot_number"] == a_door["lot_number"]
    assert rows[0]["exact_name"] is True
    assert rows[0]["civic_address"] == a_door["civic_address"]
    assert rows[0]["num_civic_addresses"] >= 1
    assert rows[0]["num_lot_addresses"] >= rows[0]["num_addresses"]


def test_the_type_and_the_accents_may_be_left_out(a_door):
    """"LAJEUNESSE" for "Rue Lajeunesse", as a person types it."""
    bare = a_door["street_name"].split(" ", 1)[1]
    typed = "".join(
        c for c in unicodedata.normalize("NFKD", bare) if not unicodedata.combining(c)
    ).upper()

    rows = queries.lots_by_address(
        typed, a_door["civic_number"], neighborhood=a_door["neighborhood"]
    )

    assert rows and rows[0]["lot_number"] == a_door["lot_number"]


def test_the_lot_the_address_names_is_in_the_cadastre(a_door):
    """The tool selects through `lot_by_number`, so the join has to close."""
    lot = queries.lot_by_number(a_door["lot_number"])
    assert lot and lot["neighborhood"] == a_door["neighborhood"]


def test_the_street_alone_spans_the_door(a_door):
    summary = queries.street_summary(a_door["street_name"], neighborhood=a_door["neighborhood"])
    mine = [s for s in summary if s["street_name"] == a_door["street_name"]]
    assert len(mine) == 1
    assert mine[0]["civic_min"] <= a_door["civic_number"] <= mine[0]["civic_max"]
    assert mine[0]["exact_name"] is True


def test_a_street_nobody_named_matches_nothing():
    assert queries.lots_by_address("Zzyzx Qwertyuiop", 1) == []
    assert queries.street_summary("Zzyzx Qwertyuiop") == []


def test_python_and_sql_fold_every_loaded_street_the_same_way():
    """The two foldings are written twice; this is where they are compared."""
    rows = queries.query(
        f"""
        SELECT DISTINCT a.street_name, {queries._STREET_CORE_SQL} AS core
          FROM {queries.SILVER_SCHEMA}.lot_addresses a
        """,
        {"type_prefix": queries.STREET_TYPE_PREFIX_RE},
    )
    assert rows
    mismatched = [
        (r["street_name"], r["core"], queries.street_key(r["street_name"]))
        for r in rows
        if queries.street_key(r["street_name"]) != r["core"]
    ]
    assert not mismatched, mismatched[:10]


# --- the street directory and the Address pane's reads -----------------------


@pytest.fixture
def _needs_directory():
    if not queries.capabilities().street_directory:
        pytest.skip("this database has no silver.street_directory (hbu_infra sql/031)")
    queries.refresh_street_directory(force=True)


def test_the_directory_folds_every_street_as_python_does(_needs_directory):
    """The fold is written a third time in 031's DDL; this is where it is compared."""
    rows = queries.query(
        f"SELECT street_name, street_key FROM {queries.SILVER_SCHEMA}.street_directory"
    )
    assert rows
    strip = re.compile(queries.STREET_TYPE_PREFIX_RE)
    mismatched = [
        (r["street_name"], r["street_key"], queries.street_key(r["street_name"]))
        for r in rows
        if strip.sub("", r["street_key"]).strip() != queries.street_key(r["street_name"])
    ]
    assert not mismatched, mismatched[:10]


def test_the_directory_holds_every_street_of_the_newest_snapshot(_needs_directory):
    """One row per (borough, city, street), and as many as the points group to."""
    counted = _one(
        f"""
        WITH newest AS (
            SELECT neighborhood, max(scrape_date) AS scrape_date
              FROM {queries.SILVER_SCHEMA}.lot_addresses GROUP BY neighborhood
        )
        SELECT count(*) AS n FROM (
            SELECT DISTINCT a.neighborhood, a.municipality, a.street_name
              FROM {queries.SILVER_SCHEMA}.lot_addresses a
              JOIN newest n ON n.neighborhood = a.neighborhood
                           AND n.scrape_date = a.scrape_date
             WHERE a.street_name IS NOT NULL
        ) s
        """
    )
    held = _one(f"SELECT count(*) AS n FROM {queries.SILVER_SCHEMA}.street_directory")
    assert held["n"] == counted["n"]


def test_search_finds_a_street_from_a_prefix_of_its_words(_needs_directory, a_door):
    """"cardinal roule" reaches Cardinal-Rouleau: every word typed, cut short."""
    words = queries.street_tokens(a_door["street_name"])
    typed = " ".join(w[: max(3, len(w) - 2)] for w in words)
    found = queries.search_streets(typed, neighborhood=a_door["neighborhood"])
    assert any(
        r["street_name"] == a_door["street_name"] and r["neighborhood"] == a_door["neighborhood"]
        for r in found
    ), (typed, [r["street_name"] for r in found])
    assert found[0]["all_words"] is True


def test_search_finds_a_misspelt_street_by_trigram(_needs_directory, a_door):
    """One letter dropped from the longest word: no prefix matches, similarity does."""
    words = queries.street_tokens(a_door["street_name"])
    longest = max(range(len(words)), key=lambda i: len(words[i]))
    if len(words[longest]) < 6:
        pytest.skip("street too short to misspell safely")
    word = words[longest]
    words[longest] = word[:2] + word[3:]
    typed = " ".join(words)
    found = queries.search_streets(typed, neighborhood=a_door["neighborhood"])
    assert any(r["street_name"] == a_door["street_name"] for r in found), (
        typed, [r["street_name"] for r in found],
    )


def test_search_ranks_the_street_in_view_first(_needs_directory):
    """Two boroughs printing the same street: the one under the map wins."""
    twice = _one(
        f"""
        SELECT street_name, array_agg(neighborhood ORDER BY neighborhood) AS hoods
          FROM {queries.SILVER_SCHEMA}.street_directory
         GROUP BY street_name HAVING count(DISTINCT neighborhood) > 1
         ORDER BY min(num_civic_addresses) DESC LIMIT 1
        """
    )
    if not twice:
        pytest.skip("no street is printed in two boroughs")
    for hood in twice["hoods"]:
        extent = _one(
            f"""
            SELECT ST_XMin(extent) w, ST_YMin(extent) s, ST_XMax(extent) e, ST_YMax(extent) n
              FROM {queries.SILVER_SCHEMA}.street_directory
             WHERE street_name = %(street)s AND neighborhood = %(hood)s
            """,
            {"street": twice["street_name"], "hood": hood},
        )
        found = queries.search_streets(
            twice["street_name"], bounds=(extent["w"], extent["s"], extent["e"], extent["n"])
        )
        assert found[0]["street_name"] == twice["street_name"]
        assert found[0]["neighborhood"] == hood, (hood, found[:2])
        assert found[0]["in_view"] is True


def test_doors_answer_a_wrong_number_with_the_nearest_ones(_needs_directory, a_door):
    """No question asked: a number off the street lists the doors nearest it."""
    span = _one(
        f"""
        SELECT civic_min, civic_max FROM {queries.SILVER_SCHEMA}.street_directory
         WHERE street_name = %(street)s AND neighborhood = %(hood)s
        """,
        {"street": a_door["street_name"], "hood": a_door["neighborhood"]},
    )
    below = max(1, span["civic_min"] - 500)
    doors = queries.doors_on_street(
        a_door["street_name"], neighborhood=a_door["neighborhood"], civic_number=below,
    )
    assert doors
    assert not any(d["exact_number"] for d in doors)
    numbers = [d["civic_number"] for d in doors]
    assert numbers == sorted(numbers), numbers
    assert numbers[0] == span["civic_min"]


def test_doors_lead_with_the_number_asked_for(_needs_directory, a_door):
    doors = queries.doors_on_street(
        a_door["street_name"], neighborhood=a_door["neighborhood"],
        civic_number=a_door["civic_number"],
    )
    assert doors[0]["exact_number"] is True
    assert doors[0]["civic_number"] == a_door["civic_number"]
    assert doors[0]["lot_number"] == a_door["lot_number"]
    assert doors[0]["lon"] is not None and doors[0]["lat"] is not None


def test_the_pane_reads_keep_up_with_typing(_needs_directory, a_door):
    """Both reads under half a second, so the pane can answer per keystroke."""
    import time

    started = time.perf_counter()
    streets = queries.search_streets(a_door["street_name"], bounds=(-74, 45, -70, 49))
    searched = time.perf_counter() - started
    started = time.perf_counter()
    queries.doors_on_street(
        streets[0]["street_name"], neighborhood=streets[0]["neighborhood"], civic_number=1,
    )
    doors = time.perf_counter() - started
    assert searched < 0.5, searched
    assert doors < 0.5, doors


def test_similar_streets_reads_the_directory_quickly(_needs_directory, a_door):
    """The chat's "did you mean" went from 2-7 s to well under one."""
    import time

    words = queries.street_tokens(a_door["street_name"])
    typed = " ".join(words)[:-1] + "x"
    started = time.perf_counter()
    queries.similar_streets(typed)
    assert time.perf_counter() - started < 1.5


def test_every_loaded_municipality_folds_the_same_way_and_is_known():
    """`places.fold` and ``_MUNICIPALITY_KEY_SQL`` agree, and no city is unmapped."""
    rows = queries.query(
        f"""
        SELECT DISTINCT a.municipality, {queries._MUNICIPALITY_KEY_SQL} AS key
          FROM {queries.SILVER_SCHEMA}.lot_addresses a
         WHERE a.municipality IS NOT NULL
        """
    )
    assert rows
    for row in rows:
        assert places.fold(row["municipality"]) == row["key"], row
        assert places.resolve(row["municipality"])[0].municipality == row["municipality"], row


def test_a_former_town_finds_the_door_in_its_city(a_door):
    city = _one(
        f"""SELECT municipality FROM {queries.SILVER_SCHEMA}.lot_addresses
             WHERE lot_number = %(lot)s AND civic_address = %(address)s LIMIT 1""",
        {"lot": a_door["lot_number"], "address": a_door["civic_address"]},
    )["municipality"]
    alias = next(
        entry.name
        for entries in places.gazetteer().values()
        for entry in entries
        if entry.municipality == city and entry.kind == "former_municipality"
    )

    found = queries.lots_by_address(
        a_door["street_name"], a_door["civic_number"],
        municipalities=[c.key for c in places.resolve(alias)],
    )
    elsewhere = queries.lots_by_address(
        a_door["street_name"], a_door["civic_number"], municipalities=["laval"]
    )

    assert a_door["lot_number"] in {r["lot_number"] for r in found}
    assert elsewhere == []


def test_coverage_names_every_borough_with_addresses():
    coverage = queries.address_coverage()
    assert coverage
    for row in coverage:
        assert row["num_addresses"] >= row["num_lots"] >= 1


def test_a_particle_and_a_word_order_reach_the_same_door(a_door):
    """The containment arm, against whatever street the table happens to hold.

    Read off the table rather than written down: a door's own street name gets
    a particle put in front of its last word and its words reversed, and both
    have to come back with that door's lot. A substring match cannot do either
    - it reads the folded name as one string - which is why the arm exists.
    """
    words = queries.street_tokens(a_door["street_name"])
    if len(words) < 2:
        pytest.skip("this borough's sample street is a single word")

    reversed_words = " ".join(reversed(words))
    with_particle = " ".join(words[:-1] + ["du", words[-1]])

    for typed in (reversed_words, with_particle):
        found = queries.lots_by_address(typed, a_door["civic_number"])
        assert a_door["lot_number"] in {r["lot_number"] for r in found}, (
            f"{typed!r} did not reach {a_door['civic_address']!r}"
        )


def test_containment_does_not_merge_two_streets(a_door):
    """Every word typed must be present, so one shared word is not enough.

    The arm widens recall and must not widen it to any street sharing a saint's
    or a cardinal's name: a word the street does not have is added and the door
    has to disappear.
    """
    words = queries.street_tokens(a_door["street_name"])
    if not words:
        pytest.skip("no significant words in this borough's sample street")

    found = queries.lots_by_address(
        " ".join(words + ["zzzznotastreet"]), a_door["civic_number"]
    )
    assert a_door["lot_number"] not in {r["lot_number"] for r in found}


def test_a_transposed_number_is_only_proposed_when_the_door_is_real(a_door):
    """`doors_with_same_digits` answers off the table, never off arithmetic.

    Every row it returns is a door the layer prints on that street, with the
    same digits as the number asked and not that number itself - so a reading
    it proposes can always be selected once the user confirms it.
    """
    digits = sorted(str(a_door["civic_number"]))
    same = queries.doors_with_same_digits(
        a_door["street_name"], a_door["civic_number"]
    )
    for row in same:
        assert row["civic_number"] != a_door["civic_number"]
        assert sorted(str(row["civic_number"])) == digits
        assert queries.lots_by_address(row["street_name"], row["civic_number"]), (
            f"proposed {row['civic_address']!r} is not findable"
        )


def test_the_street_directory_answers_inside_the_statement_timeout():
    """The regression that killed every "did you mean".

    `street_directory` reads every address point. Correlating the newest-load
    rule ran a subquery per row - 32.96 s over 346,409 points, past the 20 s
    `statement_timeout` the pool sets - so `similar_streets` raised
    `QueryCanceled` rather than proposing a spelling. This asks the real
    database for it and is the only place that can catch it coming back.
    """
    import time

    started = time.monotonic()
    rows = queries.street_directory()
    elapsed = time.monotonic() - started

    assert rows, "expected at least one loaded street"
    assert elapsed < 15, f"street_directory took {elapsed:.1f}s; the pool cancels at 20s"
    assert queries.similar_streets(rows[0]["street_name"], limit=3)
