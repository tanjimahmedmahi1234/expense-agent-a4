"""
COIT12204 Assessment 4, Tanjim Ahmed Mahi (12298057)

Rewrites data/expenses.json with the sample expenses, dated relative to today.
Run it before a demo, otherwise old dates fall out of the 30 day window.

    python -m scripts.seed_expenses
"""
import os

from app.config import EXPENSES_FILE
from app.store import ExpenseStore

if __name__ == "__main__":
    if os.path.exists(EXPENSES_FILE):
        os.replace(EXPENSES_FILE, EXPENSES_FILE + ".old")
    store = ExpenseStore()  # no file, so it writes the seed data with today's dates
    print(f"wrote {len(store.all())} expenses to {EXPENSES_FILE} (old file kept as .old)")
