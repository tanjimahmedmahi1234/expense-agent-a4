"""
COIT12204 Assessment 4, Tanjim Ahmed Mahi (12298057)

A very small "database". Expenses are kept in a list and saved to a JSON file.
I didn't want to set up SQLite for this, a JSON file is enough for the assessment.
"""
import json
import os
from datetime import date, timedelta
from typing import List, Optional

from app.config import EXPENSES_FILE
from app.models import Expense, ExpenseCreate
from app.logger import logger


def today():
    # wrapped in a function so the tests can freeze the date with monkeypatch
    return date.today()


def seed_expenses() -> List[Expense]:
    """Some sample spending so the demo has data. Dates are relative to today."""
    t = today()

    def d(n):
        return t - timedelta(days=n)

    return [
        Expense(id=1, description="Woolworths weekly groceries", amount=86.40, date=d(1), merchant="Woolworths", category="food"),
        Expense(id=2, description="Myki top up", amount=50.00, date=d(2), merchant="PTV", category="transport"),
        Expense(id=3, description="Rent for September", amount=1200.00, date=d(3), merchant="Ray White", category="rent"),
        Expense(id=4, description="Netflix monthly", amount=22.99, date=d(4), merchant="Netflix", category=None),
        Expense(id=5, description="Uber to airport", amount=68.50, date=d(5), merchant="Uber", category=None),
        Expense(id=6, description="Chemist Warehouse vitamins", amount=34.95, date=d(6), merchant="Chemist Warehouse", category=None),
        Expense(id=7, description="Origin electricity bill", amount=210.30, date=d(8), merchant="Origin", category="utilities"),
        Expense(id=8, description="Coffee at campus cafe", amount=5.50, date=d(9), merchant="Cafe", category="food"),
        Expense(id=9, description="JB Hi-Fi headphones", amount=349.00, date=d(10), merchant="JB Hi-Fi", category=None),
        Expense(id=10, description="Textbook for COIT12204", amount=120.00, date=d(12), merchant="Booktopia", category=None),
        Expense(id=11, description="Dinner with friends", amount=95.00, date=d(14), merchant="Restaurant", category="food"),
        Expense(id=12, description="Coffee", amount=4.80, date=d(20), merchant="Cafe", category="food"),
        Expense(id=13, description="Gym membership", amount=59.00, date=d(25), merchant="Anytime Fitness", category="health"),
        Expense(id=14, description="Coles groceries", amount=410.00, date=d(40), merchant="Coles", category="food"),
    ]


class ExpenseStore:
    def __init__(self, path: str = EXPENSES_FILE):
        self.path = path
        self.expenses: List[Expense] = []
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self.expenses = [Expense(**item) for item in raw]
                logger.info("STORE loaded %d expenses from %s", len(self.expenses), self.path)
                return
            except Exception as e:
                logger.warning("STORE could not read %s (%s), using seed data", self.path, e)
        self.expenses = seed_expenses()
        self.save()

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump([e.model_dump(mode="json") for e in self.expenses], f, indent=2)

    def get(self, expense_id: int) -> Optional[Expense]:
        for e in self.expenses:
            if e.id == expense_id:
                return e
        return None

    def all(self) -> List[Expense]:
        return list(self.expenses)

    def add(self, data: ExpenseCreate) -> Expense:
        new_id = max([e.id for e in self.expenses], default=0) + 1
        exp = Expense(
            id=new_id,
            description=data.description,
            amount=data.amount,
            date=data.date or today(),
            merchant=data.merchant,
        )
        self.expenses.append(exp)
        self.save()
        logger.info("STORE added expense %d", new_id)
        return exp

    def set_category(self, expense_id: int, category: str):
        exp = self.get(expense_id)
        if exp:
            exp.category = category
            self.save()

    def recent(self, days: int = 30) -> List[Expense]:
        cutoff = today() - timedelta(days=days)
        out = [e for e in self.expenses if e.date >= cutoff]
        out.sort(key=lambda e: e.date, reverse=True)
        return out
