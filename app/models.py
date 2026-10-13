"""
COIT12204 Assessment 4, Tanjim Ahmed Mahi (12298057)

Pydantic models. Expense is one row of spending. AgentResult is the JSON shape
the agent has to give back no matter which action it ran.
"""
import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.config import CATEGORIES


class Expense(BaseModel):
    id: int
    description: str = Field(min_length=1, max_length=200)
    amount: float = Field(gt=0)
    date: datetime.date
    category: Optional[str] = None  # None means not categorised yet
    merchant: Optional[str] = None


class ExpenseCreate(BaseModel):
    # same as Expense but without the id, the store gives it one.
    # I write datetime.date here because a field called "date" with a type called
    # "date" confused pydantic and the field ended up typed as None.
    description: str = Field(min_length=1, max_length=200)
    amount: float = Field(gt=0)
    date: Optional[datetime.date] = None
    merchant: Optional[str] = None


# ---- request bodies for the API ----

class CategorizeRequest(BaseModel):
    # either give an expense_id that is already saved, or describe a new one
    expense_id: Optional[int] = Field(default=None, ge=1)
    description: Optional[str] = Field(default=None, max_length=300)
    amount: Optional[float] = Field(default=None, gt=0)
    merchant: Optional[str] = Field(default=None, max_length=100)


class CheckUnusualRequest(BaseModel):
    expense_id: int = Field(ge=1)


class SummaryRequest(BaseModel):
    days: int = Field(default=30, ge=1, le=90)


class TeamRequest(BaseModel):
    """A4: one request for the two agent team."""
    action: Literal["categorize_expense", "check_unusual_expense", "monthly_summary"]
    expense_id: Optional[int] = Field(default=None, ge=1)
    description: Optional[str] = Field(default=None, max_length=300)
    amount: Optional[float] = Field(default=None, gt=0)
    merchant: Optional[str] = Field(default=None, max_length=100)
    days: int = Field(default=30, ge=1, le=90)
    # only for the demo and the tests, makes one thing go wrong on purpose
    simulate: Optional[Literal["reviewer_timeout", "invalid_receiver", "bad_message", "force_revise"]] = None


class AgentResult(BaseModel):
    """This is the structured output. The LLM must fill this in as JSON."""

    action: str = Field(
        description="one of: categorize_expense, check_unusual_expense, monthly_summary"
    )
    expense_id: Optional[int] = Field(
        default=None, description="the id of the expense this is about, or null"
    )
    category: Optional[str] = Field(
        default=None, description="one of " + ", ".join(CATEGORIES) + " or null"
    )
    is_unusual: Optional[bool] = Field(
        default=None, description="true if the expense is much bigger than normal for its category"
    )
    summary: Optional[str] = Field(
        default=None, description="a short plain text summary, only for monthly_summary"
    )
    reasoning: str = Field(description="one or two sentences on why")
    confidence: float = Field(description="how sure you are from 0 to 1")
