"""The side-channel between the agent's tools and the map."""

from __future__ import annotations

import pytest

from src.utils import state


def test_nothing_pending_by_default():
    assert state.take_map_command() is None


def test_a_request_is_taken_once(monkeypatch):
    state.request_map(zoom=17, note="moved")

    command = state.take_map_command()
    assert command["zoom"] == 17
    assert command["note"] == "moved"

    # A command applied twice would fight the user's next pan.
    assert state.take_map_command() is None


def test_two_requests_in_one_turn_merge():
    """Selecting a lot and toggling a layer must both survive."""
    state.request_map(select_lot="2 170 935")
    state.request_map(layers={"zones": True})

    command = state.take_map_command()
    assert command["select_lot"] == "2 170 935"
    assert command["layers"] == {"zones": True}


def test_dict_fields_merge_rather_than_replace():
    state.request_map(layers={"lots": True})
    state.request_map(layers={"zones": False})
    assert state.take_map_command()["layers"] == {"lots": True, "zones": False}


def test_an_unknown_field_is_a_programming_error():
    with pytest.raises(KeyError):
        state.request_map(centre=[45.5, -73.6])


def test_the_viewport_survives_a_command_being_cleared():
    """Clearing a command must not forget where the map is looking."""
    state.set_viewport((-73.7, 45.5, -73.6, 45.6), 17, (45.55, -73.65))
    state.request_map(zoom=15)
    state.take_map_command()

    assert state.get_viewport() == (-73.7, 45.5, -73.6, 45.6)
    assert state.get_viewport_zoom() == 17


def test_viewport_zoom_falls_back_before_the_map_reports():
    state.set_viewport(None, None, None)
    assert state.get_viewport_zoom(default=16) == 16


def test_selecting_a_lot_marks_the_turn_current():
    state.start_new_turn()
    assert not state.context_is_current_turn()

    state.set_selected_lot("2 170 935", -73.6, 45.5, "VSMPE")

    assert state.context_is_current_turn()
    assert state.get_selected_lot()["lot_number"] == "2 170 935"


def test_a_new_turn_makes_an_older_selection_stale():
    """So a tool can tell 'asked about now' from 'still on screen from before'."""
    state.set_selected_lot("2 170 935", -73.6, 45.5)
    state.start_new_turn()
    assert not state.context_is_current_turn()


def test_rag_results_are_recorded_for_the_pane():
    state.set_rag_result("hauteur", [{"chunk_id": "a"}], scope="lot", lot_number="2 170 935")
    assert state.RagBuffer["scope"] == "lot"
    assert state.RagBuffer["lot_number"] == "2 170 935"

    state.clear_rag_buffer()
    assert state.RagBuffer["hits"] is None
