"""`scripts/roll_lookup.py`'s plan: what gets typed where, without a browser.

The script's one job is to type a number into the right box on the city's
page and stop. What it can get wrong offline is the number - a lot with its
spaces left in, a matricule cut into the wrong groups, both given at once -
and `plan()` is where that is decided, so that is what is tested.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "roll_lookup.py"
_SPEC = importlib.util.spec_from_file_location("roll_lookup", _PATH)
roll_lookup = importlib.util.module_from_spec(_SPEC)
sys.modules["roll_lookup"] = roll_lookup
_SPEC.loader.exec_module(roll_lookup)


def test_a_lot_number_goes_in_the_lot_box_without_its_spaces():
    which = roll_lookup.plan(lot="5 342 219")

    assert which.tab == roll_lookup.TAB_LOT
    assert [(s.field_id, s.text) for s in which.steps] == [
        (roll_lookup.LOT_FIELD, "5342219")
    ]
    assert which.label == "lot 5342219"


def test_a_dashed_matricule_goes_one_group_per_box():
    which = roll_lookup.plan(matricule="4885-62-3131-1-000-0000")

    assert which.tab == roll_lookup.TAB_MATRICULE
    assert [s.text for s in which.steps] == ["4885", "62", "3131", "1", "000", "0000"]
    assert [s.field_id for s in which.steps] == list(roll_lookup.MATRICULE_FIELDS)


def test_the_rolls_eighteen_digits_are_cut_the_same_way():
    dashed = roll_lookup.plan(matricule="4885-62-3131-1-000-0000")
    run_together = roll_lookup.plan(matricule="488562313110000000")

    assert run_together.steps == dashed.steps


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"lot": "5342219", "matricule": "488562313110000000"},
        {"lot": "534221"},
        {"lot": "PC-9001"},
        {"matricule": "4885-62-3131"},
        {"matricule": "4885-62-313A-1-000-0000"},
    ],
)
def test_anything_else_is_refused_before_a_browser_opens(kwargs):
    with pytest.raises(ValueError):
        roll_lookup.plan(**kwargs)
