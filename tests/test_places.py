"""The place an address is written with, read as the municipalities it may mean."""

from __future__ import annotations

import pytest

from src.utils import places


def _readings(place, context=()):
    return [(c.municipality, round(c.score, 2)) for c in places.resolve(place, context)]


@pytest.mark.parametrize(
    ("typed", "city"),
    [
        # Quebec City: its arrondissements, the towns merged into it, quartiers.
        ("Sillery", "Québec"),
        ("SILLERY", "Québec"),
        ("cap rouge", "Québec"),
        ("Ste-Foy", "Québec"),
        ("Sainte-Foy–Sillery–Cap-Rouge", "Québec"),
        ("La Cité-Limoilou", "Québec"),
        ("Val-Bélair", "Québec"),
        ("Ville de Québec", "Québec"),
        ("Quebec City", "Québec"),
        ("québec", "Québec"),
        # Saguenay: the three arrondissements and the towns before them.
        ("JONQUIERE", "Saguenay"),
        ("Chicoutimi", "Saguenay"),
        ("La Baie", "Saguenay"),
        ("Arvida", "Saguenay"),
        # Montreal: boroughs, their halves, and the towns merged into it.
        ("Villeray", "Montréal"),
        ("Parc-Ex", "Montréal"),
        ("St-Léonard", "Montréal"),
        ("Montréal-Nord", "Montréal"),
        ("Le Plateau-Mont-Royal", "Montréal"),
        ("L'Île-Bizard", "Montréal"),
        ("Montréal (Québec) H2R 2H8", "Montréal"),
    ],
)
def test_a_place_with_one_meaning_resolves_to_its_city(typed, city):
    assert _readings(typed) == [(city, 1.0)]


def test_a_name_with_several_meanings_is_ranked_by_its_weights():
    assert _readings("Montcalm") == [("Québec", 0.7), ("Montcalm", 0.3)]
    assert _readings("Mont-Royal") == [("Mont-Royal", 0.7), ("Montréal", 0.3)]
    assert _readings("Plateau") == [("Montréal", 0.8), ("Québec", 0.2)]


def test_a_later_segment_lifts_the_reading_it_agrees_with():
    assert _readings("Montcalm", ["Québec"]) == [("Québec", 0.88), ("Montcalm", 0.13)]
    # A context naming none of the readings changes nothing.
    assert _readings("Montcalm", ["Saguenay"]) == [("Québec", 0.7), ("Montcalm", 0.3)]


def test_a_demerged_town_is_its_own_city_not_montreals():
    assert _readings("Westmount") == [("Westmount", 1.0)]
    assert _readings("L'Ancienne-Lorette") == [("L'Ancienne-Lorette", 1.0)]
    assert places.resolve("Westmount")[0].kind == "municipality"


def test_a_misspelling_reads_as_the_name_it_is_close_to():
    readings = places.resolve("Montcam")
    assert [c.municipality for c in readings] == ["Québec", "Montcalm"]
    assert all(c.fuzzy and c.name == "Montcalm" for c in readings)


def test_an_unknown_place_is_a_municipality_of_its_own():
    (only,) = places.resolve("Ville de Blorpville")
    assert only.kind == "unknown" and only.key == "blorpville"
    assert not places.is_known("Blorpville")
    assert places.resolve(None) == [] and places.resolve(" QC ") == []


def test_every_row_of_the_gazetteer_is_well_formed():
    kinds = {"city", "arrondissement", "former_municipality", "quartier", "municipality"}
    for entries in places.gazetteer().values():
        for entry in entries:
            assert entry.kind in kinds, entry
            assert entry.weight > 0, entry
            assert entry.municipality, entry


def test_every_city_is_its_own_first_reading():
    for city in ("Montréal", "Québec", "Saguenay", "Westmount"):
        assert places.resolve(city)[0].municipality == city


def test_a_reading_describes_itself():
    quartier, town = places.resolve("Montcalm")
    assert quartier.describe() == "Montcalm, a quartier of Québec (La Cité-Limoilou)"
    assert town.describe() == "Montcalm (Laurentides)"


@pytest.mark.parametrize(
    ("typed", "segments"),
    [
        ("7430 Lajeunesse, Montréal (Québec) H2R 2H8", ["Montréal (Québec) H2R 2H8"]),
        ("1234 chemin Saint-Louis, Sillery", ["Sillery"]),
        ("1 rue X, QC, Montcalm, Québec", ["Montcalm", "Québec"]),
        ("7430 Lajeunesse", []),
        ("7430 Lajeunesse, H2R 2H8", []),
        ("", []),
    ],
)
def test_the_places_are_the_named_segments_after_the_street(typed, segments):
    assert places.places_from_address(typed) == segments
