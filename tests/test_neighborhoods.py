"""The partition codes the chat must never show bare."""

from src.utils import neighborhoods


def test_label_puts_the_name_first_and_keeps_the_code():
    assert neighborhoods.label("VSMPE") == (
        "Villeray–Saint-Michel–Parc-Extension, Montréal [VSMPE]"
    )
    assert neighborhoods.label("SSC") == "Sainte-Foy–Sillery–Cap-Rouge, Québec [SSC]"


def test_saguenay_is_a_city_not_a_borough():
    assert neighborhoods.name("SAG") == "Saguenay"
    assert "Montréal" not in neighborhoods.label("SAG")


def test_an_unknown_code_is_printed_as_itself():
    assert neighborhoods.label("XYZ") == "XYZ"
    assert neighborhoods.label(None) == "None"


def test_the_prompt_carries_every_code():
    from src.agent import _SYSTEM_PROMPT

    for code, n in neighborhoods.NEIGHBORHOODS.items():
        assert code in _SYSTEM_PROMPT and n.name in _SYSTEM_PROMPT
