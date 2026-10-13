"""COIT12204 Assessment 4, Tanjim Ahmed Mahi (12298057)

Tests for AgentState: counters, transitions, history cap, saving and loading."""
import json
import logging

from app.models import AgentResult
from app.state import AgentState


def make_result(action="categorize_expense", **kw):
    fields = dict(action=action, reasoning="test", confidence=0.8)
    fields.update(kw)
    return AgentResult(**fields)


def test_initial_state_is_empty(state):
    assert state.run_count == 0
    assert state.error_count == 0
    assert state.fallback_count == 0
    assert state.last_action is None
    assert state.history == []


def test_record_run_updates_counters_and_last_values(state):
    state.record_run("categorize_expense", make_result(category="food", expense_id=1), tools_called=["lookup_expense"])
    assert state.run_count == 1
    assert state.last_action == "categorize_expense"
    assert state.last_result == "food"
    assert state.error_count == 0
    assert state.fallback_count == 0
    assert state.history[-1]["type"] == "run"
    assert state.history[-1]["tools"] == ["lookup_expense"]


def test_error_and_fallback_are_counted(state):
    state.record_run("monthly_summary", make_result("monthly_summary", summary="x"), fallback_used=True, error="RuntimeError: boom")
    assert state.error_count == 1
    assert state.fallback_count == 1
    assert "boom" in state.last_error


def test_last_result_depends_on_action(state):
    state.record_run("check_unusual_expense", make_result("check_unusual_expense", is_unusual=True))
    assert state.last_result is True
    state.record_run("monthly_summary", make_result("monthly_summary", summary="spent a lot"))
    assert state.last_result == "spent a lot"


def test_record_tool_call(state):
    state.record_tool_call("lookup_expense", {"expense_id": 3}, {"id": 3})
    assert state.last_tool_call == {"name": "lookup_expense", "args": {"expense_id": 3}}
    assert state.history[-1]["type"] == "tool"


def test_history_is_capped(state):
    # the fixture sets history_limit=5
    for i in range(12):
        state.record_tool_call("lookup_expense", {"expense_id": i}, {})
    assert len(state.history) == 5
    # the newest ones are kept
    assert state.history[-1]["args"] == {"expense_id": 11}


def test_transitions_are_logged(state, caplog):
    with caplog.at_level(logging.INFO, logger="expense_agent"):
        state.record_run("categorize_expense", make_result(category="rent"))
    assert "STATE TRANSITION run_count: 0 -> 1" in caplog.text
    assert "STATE TRANSITION last_action: None -> categorize_expense" in caplog.text
    assert "STATE TRANSITION last_result: None -> rent" in caplog.text


def test_unchanged_value_is_not_logged(state, caplog):
    state.record_run("categorize_expense", make_result(category="rent"))
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="expense_agent"):
        state.record_run("categorize_expense", make_result(category="rent"))
    # run_count changed, but last_action and last_result did not
    assert "STATE TRANSITION run_count: 1 -> 2" in caplog.text
    assert "last_action" not in caplog.text
    assert "last_result" not in caplog.text


def test_state_is_saved_and_loaded(state):
    # same order as the agent: tool calls first, then the run (record_run is what saves)
    state.record_tool_call("lookup_expense", {"expense_id": 2}, {})
    state.record_run("categorize_expense", make_result(category="food", expense_id=2))

    # the file exists and has the right content
    with open(state.path) as f:
        saved = json.load(f)
    assert saved["run_count"] == 1

    loaded = AgentState.load(path=state.path, history_limit=5)
    assert loaded.run_count == 1
    assert loaded.last_result == "food"
    assert loaded.last_tool_call["name"] == "lookup_expense"
    assert len(loaded.history) == 2


def test_corrupt_state_file_starts_fresh(tmp_path, caplog):
    path = tmp_path / "bad.json"
    path.write_text("{ this is not json")
    with caplog.at_level(logging.WARNING, logger="expense_agent"):
        loaded = AgentState.load(path=str(path))
    assert loaded.run_count == 0
    assert "corrupt" in caplog.text


def test_reset_clears_everything(state, caplog):
    state.record_run("categorize_expense", make_result(category="food"))
    with caplog.at_level(logging.INFO, logger="expense_agent"):
        state.reset()
    assert state.run_count == 0
    assert state.history == []
    assert "STATE RESET" in caplog.text
