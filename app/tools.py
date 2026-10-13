"""
COIT12204 Assessment 4, Tanjim Ahmed Mahi (12298057)

The tools the agent can call. They use the @tool decorator from LangChain so the model
can see their names, arguments and docstrings.

The tools return dictionaries and never raise. If something is wrong they return an
"error" key so the model can read it and explain instead of the whole run crashing.
"""
from typing import Optional

from langchain_core.tools import tool

from app.logger import logger
from app.store import ExpenseStore, today

# the store is set from the agent at start up. A global is not great but it is the
# easiest way to let a plain @tool function reach the data
_store: Optional[ExpenseStore] = None


def set_store(store: ExpenseStore):
    global _store
    _store = store


def get_store() -> ExpenseStore:
    if _store is None:
        raise RuntimeError("store not set, call set_store() first")
    return _store


@tool
def lookup_expense(expense_id: int) -> dict:
    """Look up one expense by its id. Returns description, amount, date, merchant,
    category (may be null) and days_ago. Returns an error key if the id does not exist."""
    logger.info("TOOL CALL lookup_expense(expense_id=%s)", expense_id)
    exp = get_store().get(expense_id)
    if exp is None:
        result = {"error": f"Expense {expense_id} does not exist."}
    else:
        result = exp.model_dump(mode="json")
        result["days_ago"] = (today() - exp.date).days
    logger.info("TOOL RESULT lookup_expense -> %s", result)
    return result


@tool
def list_recent_expenses(days: int = 30) -> list:
    """List all expenses from the last N days (1 to 90), newest first.
    Each item has id, description, amount, date, merchant and category."""
    logger.info("TOOL CALL list_recent_expenses(days=%s)", days)
    # clamp so the model can't ask for 10000 days
    if days < 1:
        days = 1
    if days > 90:
        days = 90
    items = [e.model_dump(mode="json") for e in get_store().recent(days)]
    logger.info("TOOL RESULT list_recent_expenses -> %d items", len(items))
    return items


@tool
def category_totals(days: int = 30) -> dict:
    """Total spending per category over the last N days, plus the overall total,
    the number of expenses, the average amount per category, and how many
    expenses still have no category. Use this for summaries and for judging
    whether an amount is unusual for its category."""
    logger.info("TOOL CALL category_totals(days=%s)", days)
    if days < 1:
        days = 1
    if days > 90:
        days = 90

    totals = {}
    counts = {}
    uncategorised = 0
    grand_total = 0.0
    for e in get_store().recent(days):
        cat = e.category or "uncategorised"
        if e.category is None:
            uncategorised += 1
        totals[cat] = round(totals.get(cat, 0) + e.amount, 2)
        counts[cat] = counts.get(cat, 0) + 1
        grand_total += e.amount

    averages = {}
    for cat in totals:
        averages[cat] = round(totals[cat] / counts[cat], 2)

    result = {
        "days": days,
        "today": today().isoformat(),
        "total": round(grand_total, 2),
        "count": sum(counts.values()),
        "by_category": totals,
        "average_by_category": averages,
        "uncategorised": uncategorised,
    }
    logger.info("TOOL RESULT category_totals -> %s", result)
    return result


TOOLS = [lookup_expense, list_recent_expenses, category_totals]
