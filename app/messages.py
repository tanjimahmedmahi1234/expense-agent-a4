"""
COIT12204 Assessment 4, Tanjim Ahmed Mahi (12298057)

The message format the agents use to talk to each other (A4).

Every message has the same envelope (AgentMessage). What goes inside "content"
depends on the intent, and each intent has its own pydantic model below.

The orchestrator calls validate_message() before it routes anything. That checks:
  1. the receiver is an agent that exists
  2. this sender is allowed to send this intent to this receiver
  3. the content matches the model for the intent

I left receiver as a plain string on purpose. If it was a Literal, pydantic would
throw an error when the message is built, and the orchestrator would never get the
chance to log it as an invalid receiver and fall back properly.
"""
import datetime
import uuid
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from app.config import CATEGORIES

AGENT_NAMES = ["orchestrator", "classifier", "reviewer"]

ACTION = Literal["categorize_expense", "check_unusual_expense", "monthly_summary"]


# ---------- content models, one per intent ----------

class TaskRequest(BaseModel):
    """What the user asked for. It travels inside every message so no agent has to guess."""
    action: ACTION
    expense_id: Optional[int] = Field(default=None, ge=1)
    description: Optional[str] = Field(default=None, max_length=300)
    amount: Optional[float] = None
    merchant: Optional[str] = Field(default=None, max_length=100)
    days: int = Field(default=30, ge=1, le=90)


class Proposal(BaseModel):
    """The classifier's answer. Same fields as AgentResult from A3."""
    action: ACTION
    expense_id: Optional[int] = None
    category: Optional[str] = None
    is_unusual: Optional[bool] = None
    summary: Optional[str] = None
    reasoning: str
    confidence: float = Field(ge=0, le=1)


class ReviewResult(BaseModel):
    """What the reviewer decides. The LLM reviewer has to reply in this shape."""
    verdict: Literal["approve", "revise"] = Field(description="approve if the proposal is right, revise if not")
    issues: List[str] = Field(default_factory=list, description="short list of problems, empty if none")
    feedback: str = Field(default="", description="one or two sentences the classifier can act on")
    suggested_category: Optional[str] = Field(default=None, description="the category you think is right, or null")
    suggested_is_unusual: Optional[bool] = Field(default=None, description="for the unusual check only, or null")
    confidence: float = Field(default=0.5, ge=0, le=1, description="how sure you are from 0 to 1")


class ClassifyContent(BaseModel):
    request: TaskRequest


class ReviewContent(BaseModel):
    request: TaskRequest
    proposal: Proposal
    round: int = Field(ge=1)
    classifier_used_fallback: bool = False
    tools_called: List[str] = Field(default_factory=list)


class ReviseContent(BaseModel):
    request: TaskRequest
    previous_proposal: Proposal
    review: ReviewResult
    round: int = Field(ge=1)


class ApprovedContent(BaseModel):
    request: TaskRequest
    proposal: Proposal
    review: ReviewResult
    round: int = Field(ge=1)


# intent -> (allowed senders, allowed receivers, content model)
ROUTES = {
    "classify_request": (["orchestrator"], ["classifier"], ClassifyContent),
    "review_request": (["classifier"], ["reviewer"], ReviewContent),
    "revise_request": (["reviewer"], ["classifier"], ReviseContent),
    "approved": (["reviewer"], ["orchestrator"], ApprovedContent),
}


# ---------- the envelope ----------

class AgentMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    conversation_id: str = ""
    step: int = 0
    sender: str
    receiver: str  # plain str on purpose, see the note at the top
    intent: str
    content: dict
    constraints: dict = Field(default_factory=lambda: {"format": "json"})
    reply_to: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat(timespec="seconds"))

    def short(self):
        """One line for the log."""
        return f"{self.sender} -> {self.receiver} intent={self.intent} id={self.message_id}"


class MessageError(Exception):
    """kind is one of: invalid_receiver, invalid_sender, bad_route, bad_content"""

    def __init__(self, kind, detail):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


def validate_message(msg: AgentMessage):
    """Raise MessageError if the message should not be routed. Returns the parsed content."""
    if msg.receiver not in AGENT_NAMES:
        raise MessageError("invalid_receiver", f"'{msg.receiver}' is not a known agent")
    if msg.sender not in AGENT_NAMES:
        raise MessageError("invalid_sender", f"'{msg.sender}' is not a known agent")
    if msg.intent not in ROUTES:
        raise MessageError("bad_route", f"unknown intent '{msg.intent}'")

    senders, receivers, model = ROUTES[msg.intent]
    if msg.sender not in senders or msg.receiver not in receivers:
        raise MessageError("bad_route", f"{msg.sender} can't send '{msg.intent}' to {msg.receiver}")

    try:
        content = model(**msg.content)
    except ValidationError as e:
        # only the first problem, the full pydantic error is very long
        first = e.errors()[0]
        where = ".".join(str(x) for x in first["loc"])
        raise MessageError("bad_content", f"{where}: {first['msg']}")

    # one extra rule pydantic can't do on its own
    if isinstance(content, (ReviewContent, ApprovedContent)):
        cat = content.proposal.category
        if cat is not None and cat not in CATEGORIES:
            raise MessageError("bad_content", f"proposal.category '{cat}' is not an allowed category")
    return content


def make_message(sender, receiver, intent, content, conversation_id="", reply_to=None, constraints=None):
    """Small helper so every agent builds messages the same way.
    content can be a pydantic model or a dict."""
    if hasattr(content, "model_dump"):
        content = content.model_dump(mode="json")
    msg = AgentMessage(
        sender=sender,
        receiver=receiver,
        intent=intent,
        content=content,
        conversation_id=conversation_id,
        reply_to=reply_to,
    )
    if constraints:
        msg.constraints.update(constraints)
    return msg
