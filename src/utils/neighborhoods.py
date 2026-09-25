"""
neighborhoods.py — What a partition key like ``VSMPE`` or ``SAG`` is called.

The dataplatform keys every table on a short code (`urban_rag.partitions`):
Montreal's borough abbreviations, Quebec City's three-letter arrondissement
codes, one key for the whole of Saguenay. They are internal. A user reads
"Villeray–Saint-Michel–Parc-Extension, Montréal", never ``VSMPE``, and left
with bare codes the model guesses — it once filed SAG and SSC under Montreal.

So every tool line that names a partition goes through `label`, which keeps
the code in brackets for the tools' ``neighborhood`` argument and puts the
real name first, and `glossary` gives the system prompt the whole table.
Kept by hand in step with the dataplatform's key maps; a key missing here is
printed as itself rather than failing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Neighborhood:
    code: str
    name: str
    city: str
    kind: str  # borough, arrondissement, or city


def _montreal(code: str, name: str) -> Neighborhood:
    return Neighborhood(code, name, "Montréal", "borough")


def _quebec(code: str, name: str) -> Neighborhood:
    return Neighborhood(code, name, "Québec", "arrondissement")


NEIGHBORHOODS: dict[str, Neighborhood] = {
    n.code: n
    for n in (
        _montreal("AC", "Ahuntsic-Cartierville"),
        _montreal("Anjou", "Anjou"),
        _montreal("CDNNDG", "Côte-des-Neiges–Notre-Dame-de-Grâce"),
        _montreal("Lachine", "Lachine"),
        _montreal("LaSalle", "LaSalle"),
        _montreal("PMR", "Le Plateau-Mont-Royal"),
        _montreal("SO", "Le Sud-Ouest"),
        _montreal("MHM", "Mercier–Hochelaga-Maisonneuve"),
        _montreal("MN", "Montréal-Nord"),
        _montreal("Outremont", "Outremont"),
        _montreal("PR", "Pierrefonds-Roxboro"),
        _montreal("RDPPAT", "Rivière-des-Prairies–Pointe-aux-Trembles"),
        _montreal("RPP", "Rosemont–La Petite-Patrie"),
        _montreal("StLeonard", "Saint-Léonard"),
        _montreal("Verdun", "Verdun"),
        _montreal("VM", "Ville-Marie"),
        _montreal("VSMPE", "Villeray–Saint-Michel–Parc-Extension"),
        _quebec("CIL", "La Cité-Limoilou"),
        _quebec("RIV", "Les Rivières"),
        _quebec("SSC", "Sainte-Foy–Sillery–Cap-Rouge"),
        _quebec("CHA", "Charlesbourg"),
        _quebec("BEA", "Beauport"),
        _quebec("HSC", "La Haute-Saint-Charles"),
        Neighborhood("SAG", "Saguenay", "Saguenay", "city"),
    )
}


def name(code: str | None) -> str:
    """"Villeray–Saint-Michel–Parc-Extension, Montréal"; the code itself if unknown."""
    n = NEIGHBORHOODS.get(code or "")
    if n is None:
        return str(code)
    return n.name if n.name == n.city else f"{n.name}, {n.city}"


def label(code: str | None) -> str:
    """`name` with the code in brackets, for a tool line the model may reuse."""
    if code not in NEIGHBORHOODS:
        return str(code)
    return f"{name(code)} [{code}]"


def glossary() -> str:
    """Every known code and what it is, one per line, for the system prompt."""
    return "\n".join(
        f"  {n.code:<10} {n.name}"
        + ("" if n.kind == "city" else f" — {n.kind} of {n.city}")
        + (" (the whole city, one key)" if n.kind == "city" else "")
        for n in NEIGHBORHOODS.values()
    )
