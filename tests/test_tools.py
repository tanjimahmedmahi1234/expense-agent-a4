"""COIT12204 Assessment 4, Tanjim Ahmed Mahi (12298057)

Tests for the three tools. They talk to the store directly, no model involved."""
import logging

from app.tools import TOOLS, lookup_expense, list_recent_expenses, category_totals, set_store


def test_tools_have_names_and_descriptions():
    names = [t.name for t in TOOLS]
    assert names == ["lookup_expense", "list_recent_expenses", "category_totals"]
    for t in TOOLS:
        # the description is what the model reads, so it must not be empty
        assert len(t.description) > 20


def test_lookup_expense_returns_the_expense(store):
    set_store(store)
    result = lookup_expense.invoke({"expense_id": 5})
    assert result["id"] == 5
    assert result["description"] == "Uber to airport"
    assert result["amount"] == 68.5
    assert result["days_ago"] == 5  # seed data puts it 5 days before the frozen date


def test_lookup_expense_missing_id_gives_error_not_exception(store):
    set_store(store)
    result = lookup_expense.invoke({"expense_id": 999})
    assert "error" in result
    assert "999" in result["error"]


def test_list_recent_expenses_window(store):
    set_store(store)
    items = list_recent_expenses.invoke({"days": 7})
    ids = [i["id"] for i in items]
    # ids 1 to 6 are within a week, 7 is 8 days ago
    assert ids == [1, 2, 3, 4, 5, 6]
    # newest first
    assert items[0]["id"] == 1


def test_list_recent_expenses_clamps_days(store):
    set_store(store)
    everything = list_recent_expenses.invoke({"days": 5000})
    assert len(everything) == 14  # clamped to 90 days which still covers all seed data
    tiny = list_recent_expenses.invoke({"days": -3})
    assert all(i["id"] in (1,) for i in tiny) or tiny == []  # clamped to 1 day


def test_category_totals_counts(store):
    set_store(store)
    totals = category_totals.invoke({"days": 30})
    assert totals["count"] == 13
    assert totals["by_category"]["rent"] == 1200.0
    assert totals["uncategorised"] == 5  # ids 4, 5, 6, 9, 10 start with no category
    # 191.70 / 4 = 47.925, and float rounding can go either way on the last cent
    assert abs(totals["average_by_category"]["food"] - 47.925) < 0.01
    assert totals["today"] == "2026-09-15"


def test_tool_calls_are_logged(store, caplog):
    set_store(store)
    with caplog.at_level(logging.INFO, logger="expense_agent"):
        lookup_expense.invoke({"expense_id": 1})
    text = caplog.text
    assert "TOOL CALL lookup_expense(expense_id=1)" in text
    assert "TOOL RESULT lookup_expense" in text
