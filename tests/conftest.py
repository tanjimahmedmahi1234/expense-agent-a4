"""
COIT12204 Assessment 4, Tanjim Ahmed Mahi (12298057)

Shared pytest fixtures.

The env vars at the top have to be set BEFORE anything from app is imported, because
config.py reads them at import time. Otherwise the tests would write into the real
logs/agent.log and data/ files.
"""
import os
import tempfile
from datetime import date

_tmp = tempfile.mkdtemp(prefix="expense_agent_tests_")
os.environ["LOG_FILE"] = os.path.join(_tmp, "test.log")
os.environ["EXPENSES_FILE"] = os.path.join(_tmp, "expenses.json")
os.environ["STATE_FILE"] = os.path.join(_tmp, "state.json")
os.environ["TEAM_STATE_FILE"] = os.path.join(_tmp, "team_state.json")
os.environ["CONVERSATIONS_FILE"] = os.path.join(_tmp, "conversations.jsonl")
os.environ["OPENAI_API_KEY"] = ""  # make sure no real key is used in tests

import pytest
from fastapi.testclient import TestClient

import app.agent
import app.reviewer
import app.store
import app.tools
from app.agent import ExpenseAgent
from app.classifier import ClassifierAgent
from app.main import create_app
from app.orchestrator import Orchestrator, TeamState
from app.reviewer import ReviewerAgent
from app.state import AgentState
from app.store import ExpenseStore
from tests.fakes import FakeChatModel


FIXED_TODAY = date(2026, 9, 15)


@pytest.fixture
def fixed_today(monkeypatch):
    """Freeze the date so the seed data and the 'days ago' maths are stable."""
    monkeypatch.setattr(app.store, "today", lambda: FIXED_TODAY)
    monkeypatch.setattr(app.tools, "today", lambda: FIXED_TODAY)
    monkeypatch.setattr(app.agent, "today", lambda: FIXED_TODAY)
    monkeypatch.setattr(app.reviewer, "today", lambda: FIXED_TODAY)
    return FIXED_TODAY


@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch):
    """Safety net: if any test forgets to pass a fake, get_llm blows up instead of calling OpenAI."""

    def _blocked():
        raise RuntimeError("get_llm() was called in a test, that should not happen")

    monkeypatch.setattr(app.agent, "get_llm", _blocked)


@pytest.fixture
def store(tmp_path, fixed_today):
    return ExpenseStore(path=str(tmp_path / "expenses.json"))


@pytest.fixture
def state(tmp_path):
    return AgentState(path=str(tmp_path / "state.json"), history_limit=5)


@pytest.fixture
def make_agent(store, state):
    """Factory: make_agent([responses...]) gives an agent driven by the fake model."""

    def _make(responses=None, raise_error=False, prompt_version="v1"):
        fake = FakeChatModel(responses=list(responses or []), raise_error=raise_error)
        agent = ExpenseAgent(store, state, llm=fake, prompt_version=prompt_version)
        agent.fake = fake  # so tests can inspect what the model was sent
        return agent

    return _make


@pytest.fixture
def make_team(store, make_agent, tmp_path):
    """A4 factory: make_team(classifier_replies, reviewer_replies) gives an orchestrator
    where both agents run on their own fake model. reviewer_replies=None means the
    reviewer has no model at all and only uses its rule checks."""

    def _make(classifier_replies=None, reviewer_replies=None, reviewer_error=False, **kw):
        agent = make_agent(classifier_replies or [])
        if reviewer_replies is None and not reviewer_error:
            reviewer = ReviewerAgent()  # get_llm is blocked in tests, so this is rules only
        else:
            reviewer = ReviewerAgent(llm=FakeChatModel(responses=list(reviewer_replies or []),
                                                       raise_error=reviewer_error))
        orch = Orchestrator(
            ClassifierAgent(agent), reviewer, store,
            state=TeamState(str(tmp_path / "team_state.json")),
            conversations_file=str(tmp_path / "conversations.jsonl"),
            **kw,
        )
        orch.classifier_fake = agent.fake
        orch.reviewer_fake = getattr(reviewer, "llm", None)
        return orch

    return _make


@pytest.fixture
def client(store, state, make_agent, make_team):
    """TestClient with the fake agent plugged in. client.agent.fake is the fake model.
    client.team is the orchestrator, client.team.reviewer_fake its reviewer model."""
    agent = make_agent([])
    team = make_team(reviewer_replies=[])
    # the team and the old A3 endpoints share one classifier, like in the real app
    team.agents["classifier"].agent = agent
    team.classifier_fake = agent.fake
    application = create_app(store=store, state=state, agent=agent, orchestrator=team)
    test_client = TestClient(application, raise_server_exceptions=False)
    test_client.agent = agent
    test_client.team = team
    return test_client
