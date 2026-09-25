"""Reading heritage rows: the tiers, the grades that are not grades, the SQL's shape."""

from __future__ import annotations

import re

from src.utils import heritage, queries


def _row(source_table, attributes, **extra):
    return {"source_table": source_table, "feature_id": "1", "attributes": attributes,
            "overlap_m2": 120.0, "pct_of_lot": 40.0, **extra}


def test_a_studied_buildings_grade_sets_its_tier():
    tiers = {
        code: heritage.classify(_row(
            "Patrimoine__BATIMENT_ETUDIE", {"EVALUATION_VALEUR_PATRIMO_NO": float(code)}
        )).tier
        for code in range(1, 7)
    }
    assert tiers == {
        1: heritage.HIGH, 2: heritage.HIGH, 3: heritage.MODERATE, 4: heritage.LOW,
        5: heritage.UNGRADED, 6: heritage.UNGRADED,
    }


def test_presume_and_confirme_are_not_read_as_grades():
    """5 and 6 are not below *faible*; saying "5 of 4" would be the ordinal misreading."""
    p = heritage.classify(_row(
        "Patrimoine__BATIMENT_ETUDIE",
        {"EVALUATION_VALEUR_PATRIMO_NO": 6.0,
         "EVALUATION_VALEUR_PATRIMO_NO_LIBELLE": "confirmé"},
    ))
    assert p.detail == "Interest confirmé, not graded"
    assert "of 4" not in p.detail


def test_a_graded_building_says_where_on_the_scale():
    p = heritage.classify(_row(
        "Patrimoine__BATIMENT_ETUDIE",
        {"EVALUATION_VALEUR_PATRIMO_NO": 2.0, "DENOMINATION_PRINCIPALE": "Terrasse Burroughs",
         "LIEN_FICHE": "https://example.test/fiche=13413"},
    ))
    assert p.detail == "Heritage value: supérieur (2 of 4, 1 highest)"
    assert p.name == "Terrasse Burroughs"
    assert p.url == "https://example.test/fiche=13413"
    assert p.coverage == "Building on the lot (120 m²)"


def test_a_demolished_or_lapsed_designation_drops_to_low():
    demolished = heritage.classify(_row(
        "Patrimoine__BATIMENT_ETUDIE", {"EVALUATION_VALEUR_PATRIMO_NO": 1, "DEMOLI": "O"}
    ))
    assert demolished.tier == heritage.LOW
    assert "demolished" in demolished.detail

    lapsed = heritage.classify(_row(
        "Patrimoine__IMMEUBLE_CLASSE",
        {"STATUT_ACTIF": "I", "STATUT_ACTIF_LIBELLE": "Abrogé"},
    ))
    assert lapsed.tier == heritage.LOW
    assert lapsed.detail == "Abrogé"


def test_the_status_layers_carry_their_own_jurisdiction():
    p = heritage.classify(_row(
        "Patrimoine__DESIGNE_FEDERAL",
        {"JURIDICTION_NO_LIBELLE": "Fédérale / Provinciale", "STATUT_ACTIF": "A"},
    ))
    assert p.jurisdiction == "Federal / Provincial"
    assert p.tier == heritage.LOW


def test_a_classified_immovable_is_ground_covered_not_a_building():
    """Maison Hamel-Bruneau's classified polygon is 7 398 m2: the house and its grounds."""
    p = heritage.classify(_row("Patrimoine__IMMEUBLE_CLASSE", {"STATUT_ACTIF": "A"},
                               overlap_m2=7398.0, pct_of_lot=100.0))
    assert p.coverage == "Covers the whole lot"


def test_within_a_tier_the_classement_leads_the_inventory():
    found = heritage.protections([
        _row("Patrimoine__BATIMENT_ETUDIE", {"EVALUATION_VALEUR_PATRIMO_NO": 1,
                                              "DENOMINATION_PRINCIPALE": "Maison Hamel-Bruneau"}),
        _row("Patrimoine__IMMEUBLE_CLASSE", {"DENOMINATION_PRINCIPALE": "Maison Hamel-Bruneau"}),
    ])
    assert [p.designation for p in found] == [
        "Immeuble patrimonial classé", "Bâtiment d'intérêt patrimonial",
    ]


def test_a_sector_does_not_repeat_its_designation_as_its_name():
    p = heritage.classify(_row(
        "Reglement_urbanisme__VSP_REG_SECTEUR_PATRIMONAL",
        {"NOM_CAT": "Secteur d'intérêt patrimonial",
         "DESCRIPTION": "Secteur d'intérêt patrimonial"},
    ))
    assert p.name is None and p.detail is None


def test_an_area_reports_its_share_of_the_lot():
    whole = heritage.classify(_row("Patrimoine__SITE_DECLARE_CLASSE", {}, pct_of_lot=100.0))
    part = heritage.classify(_row("Patrimoine__AIRE_PROTECTION",
                                  {"RAYON_PROTECTION": 152}, pct_of_lot=37.4))
    assert (whole.tier, whole.coverage) == (heritage.HIGH, "Covers the whole lot")
    assert (part.tier, part.coverage) == (heritage.MODERATE, "Covers 37% of the lot")
    assert part.detail == "152 m around the protected immovable"


def test_montreal_layers_match_on_their_suffix_whatever_the_borough():
    listed = heritage.classify(_row(
        "Reglement_urbanisme__VSP_REG_BATIMENT_PATRIMONIAL",
        {"DESCRIPTION": "Église Saints Cyril and Method", "ADRESSE": "2625, rue Jean-Talon Est",
         "NOM_CAT": "Lieu de culte", "EN_SAVOIR_PLUS": ""},
        overlap_m2=0.0,
    ))
    assert listed.tier == heritage.MODERATE
    assert listed.name == "Église Saints Cyril and Method — 2625, rue Jean-Talon Est"
    assert listed.detail == "Lieu de culte"
    assert listed.url is None
    assert listed.coverage == "Marked on the lot"

    other_borough = heritage.layer_for("Reglement_urbanisme__RPP_REG_SECTEUR_PATRIMONIAL")
    assert other_borough is not None and other_borough.tier == heritage.MODERATE


def test_other_layers_are_not_heritage():
    assert heritage.classify(_row("Reglement_urbanisme__VSP_REG_ZONE", {})) is None
    assert heritage.classify(_row("Zonage__ZONAGE_EN_VIGUEUR", {})) is None


def test_the_pattern_matches_exactly_what_classify_reads():
    pattern = re.compile(heritage.SOURCE_TABLE_PATTERN)
    for slug in [*heritage.QUEBEC_LAYERS, "Reglement_urbanisme__VSP_REG_SECTEUR_PATRIMONAL",
                 "Reglement_urbanisme__VSP_REG_BATIMENT_INTERET_LOCAL"]:
        assert pattern.search(slug), slug
    for slug in ["Reglement_urbanisme__VSP_REG_ZONE", "Reglement_urbanisme__VSP_REG_PIIA",
                 "Zonage__ZONAGE_EN_VIGUEUR", "Patrimoine__ART_PUBLIC"]:
        assert not pattern.search(slug), slug


def test_protections_lead_with_the_most_binding():
    found = heritage.protections([
        _row("Patrimoine__BATIMENT_ETUDIE", {"EVALUATION_VALEUR_PATRIMO_NO": 5}),
        _row("Patrimoine__DESIGNE_FEDERAL", {}),
        _row("Reglement_urbanisme__VSP_REG_ZONE", {}),
        _row("Patrimoine__SITE_DECLARE_CLASSE", {}),
        _row("Patrimoine__AIRE_PROTECTION", {}),
    ])
    assert [p.tier for p in found] == [
        heritage.HIGH, heritage.MODERATE, heritage.LOW, heritage.UNGRADED,
    ]


def test_heritage_for_lot_applies_a_cutoff_per_kind_of_geometry(monkeypatch):
    monkeypatch.setattr(
        queries, "capabilities", lambda: queries.Capabilities(lot_features=True)
    )
    captured = {}
    monkeypatch.setattr(
        queries, "query",
        lambda sql, params=None: captured.update(sql=sql, params=params) or [],
    )
    queries.heritage_for_lot("1 234 567", neighborhood="CIL")

    sql, params = captured["sql"], captured["params"]
    assert "lf.source_table ~ %(pattern)s" in sql
    assert params["pattern"] == heritage.SOURCE_TABLE_PATTERN
    # Points need only fall on the lot; footprints and areas have their own floors.
    assert "WHERE dimension < 2" in sql
    assert params["min_footprint_pct"] == queries.MIN_HERITAGE_FOOTPRINT_PCT
    assert params["min_pct_of_lot"] == queries.MIN_ZONE_PCT_OF_LOT
    assert "Patrimoine__BATIMENT_ETUDIE" in params["footprint_tables"]
    assert "Patrimoine__SITE_DECLARE_CLASSE" not in params["footprint_tables"]
    assert "DISTINCT ON (neighborhood, source_table, feature_id)" in sql


def test_heritage_for_lot_without_the_silver_join_asks_nothing(monkeypatch):
    monkeypatch.setattr(queries, "capabilities", lambda: queries.Capabilities())
    monkeypatch.setattr(queries, "query", lambda *_a, **_k: 1 / 0)
    assert queries.heritage_for_lot("1 234 567") == []
