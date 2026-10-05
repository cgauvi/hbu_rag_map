"""The roll's identifiers, spelled for the cities' own lookups.

`src/utils/roll.py` is three pure functions and a hand-kept table. What they
get wrong is never a crash: it is a matricule cut into the wrong groups, a
lot number pasted with the spaces the city's search box rejects, or a city
with no link. Every test here pins one of those.
"""

from __future__ import annotations

from src.utils import roll


def test_a_matricule_is_written_in_its_seven_groups():
    """What the roll stores as 18 digits, every city writes dashed."""
    assert roll.format_matricule("488562272610000000") == "4885-62-2726-1-000-0000"
    assert roll.format_matricule(" 488562313110000000 ") == "4885-62-3131-1-000-0000"


def test_anything_but_eighteen_digits_is_handed_back_as_typed():
    assert roll.format_matricule("4885-62-2726-1-000-0000") == "4885-62-2726-1-000-0000"
    assert roll.format_matricule("12345") == "12345"
    assert roll.format_matricule("") is None
    assert roll.format_matricule(None) is None


def test_a_lot_number_loses_its_spaces_and_keeps_its_prefix():
    """`5 342 219` is typed `5342219`; a common-parts lot is not on any roll."""
    assert roll.lot_key("5 342 219") == "5342219"
    assert roll.lot_key("5 342 219") == "5342219"
    assert roll.lot_key("PC-9001") == "PC-9001"
    assert roll.lot_key("  ") is None
    assert roll.lot_key(None) is None


def test_the_three_cities_have_a_lookup_and_nobody_else_does():
    quebec = roll.lookup_for("23027")
    assert quebec is not None and quebec.city == "Québec"
    assert quebec.url.startswith("https://www.ville.quebec.qc.ca/")
    assert "matricule" in quebec.accepts
    assert roll.lookup_for("66023").city == "Montréal"
    assert roll.lookup_for("94068").city == "Saguenay"
    assert roll.lookup_for("65005") is None
    assert roll.lookup_for(None) is None
