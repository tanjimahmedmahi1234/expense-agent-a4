"""COIT12204 Assessment 4, Tanjim Ahmed Mahi (12298057)

A4: the reviewer agent on its own (no orchestrator)."""
import logging

from app.messages import ReviewResult, make_message
from app.reviewer import ReviewerAgent, parse_review
from app.tools import set_store
from tests.fakes import FakeChatModel, json_answer, text_answer


def review_msg(action="categorize_expense", expense_id=5, constraints=None, **proposal):
    p = {"action": action, "expense_id": expense_id, "reasoning": "r", "confidence": 0.9}
    p.update(proposal)
    content = {"request": {"action": action, "expense_id": expense_id}, "proposal": p, "round": 1}
    return make_message("classifier", "reviewer", "review_request", content, constraints=constraints)


APPROVE = json_answer(verdict="approve", issues=[], feedback="", confidence=0.9)


def test_approve_goes_to_the_orchestrator(store):
    set_store(store)
    r = ReviewerAgent(llm=FakeChatModel(responses=[APPROVE]))
    reply = r.handle(review_msg(category="transport"))
    assert reply.sender == "reviewer"
    assert reply.receiver == "orchestrator"
    assert reply.intent == "approved"


def test_revise_goes_back_to_the_classifier(store):
    set_store(store)
    revise = json_answer(verdict="revise", issues=["Uber is transport"], feedback="Uber is a ride",
                         suggested_category="transport", confidence=0.8)
    r = ReviewerAgent(llm=FakeChatModel(responses=[revise]))
    reply = r.handle(review_msg(category="entertainment"))
    assert reply.receiver == "classifier"
    assert reply.intent == "revise_request"
    assert reply.content["review"]["suggested_category"] == "transport"


def test_reviewer_gets_its_own_evidence(store):
    set_store(store)
    fake = FakeChatModel(responses=[APPROVE])
    ReviewerAgent(llm=fake).handle(review_msg(category="transport"))
    sent = str(fake.calls[0])
    # it looked the expense up itself, the classifier's message only had the id
    assert "Uber to airport" in sent
    assert "keyword_hint" in sent


def test_hard_rule_overrides_the_model(store, caplog):
    set_store(store)
    # $410 groceries, food average is about $48, so "not unusual" is wrong
    r = ReviewerAgent(llm=FakeChatModel(responses=[APPROVE]))
    with caplog.at_level(logging.WARNING, logger="expense_agent"):
        reply = r.handle(review_msg("check_unusual_expense", 14, category="food", is_unusual=False))
    assert reply.intent == "revise_request"
    assert "more than double" in reply.content["review"]["feedback"]
    assert "REVIEW OVERRIDE" in caplog.text


def test_soft_rule_does_not_override(store):
    set_store(store)
    # keyword rule says transport, proposal says food (Uber Eats style). Model approves, that's fine.
    r = ReviewerAgent(llm=FakeChatModel(responses=[APPROVE]))
    reply = r.handle(review_msg(category="food"))
    assert reply.intent == "approved"


def test_missing_expense_with_high_confidence_is_sent_back(store):
    set_store(store)
    r = ReviewerAgent()  # no model, rules only
    reply = r.handle(review_msg(expense_id=999, category=None, confidence=0.9))
    assert reply.intent == "revise_request"
    assert any("does not exist" in i for i in reply.content["review"]["issues"])


def test_summary_with_wrong_total_is_sent_back(store):
    set_store(store)
    r = ReviewerAgent()
    reply = r.handle(review_msg("monthly_summary", None, summary="You spent $50 this month."))
    assert reply.intent == "revise_request"
    assert "2306.44" in reply.content["review"]["feedback"]


def test_summary_with_right_total_is_approved(store):
    set_store(store)
    r = ReviewerAgent()
    reply = r.handle(review_msg("monthly_summary", None, summary="You spent $2,306.44 and rent was the biggest."))
    assert reply.intent == "approved"


def test_bad_json_twice_falls_back_to_rules(store, caplog):
    set_store(store)
    fake = FakeChatModel(responses=[text_answer("looks fine to me"), text_answer("yes")])
    with caplog.at_level(logging.WARNING, logger="expense_agent"):
        reply = ReviewerAgent(llm=fake).handle(review_msg(category="transport"))
    assert len(fake.calls) == 2
    assert reply.intent == "approved"
    assert "REVIEW FALLBACK" in caplog.text


def test_model_exception_falls_back_to_rules(store):
    set_store(store)
    reply = ReviewerAgent(llm=FakeChatModel(raise_error=True)).handle(review_msg(category="transport"))
    assert reply.intent == "approved"
    assert reply.content["review"]["confidence"] == 0.5


def test_force_revise_simulation(store):
    set_store(store)
    reply = ReviewerAgent().handle(review_msg(category="transport", constraints={"simulate": "force_revise"}))
    assert reply.intent == "revise_request"
    assert "simulate" in reply.content["review"]["feedback"]


def test_made_up_suggestion_is_dropped(store):
    set_store(store)
    revise = json_answer(verdict="revise", issues=["x"], feedback="x", suggested_category="travel", confidence=0.8)
    reply = ReviewerAgent(llm=FakeChatModel(responses=[revise])).handle(review_msg(category="food"))
    assert reply.content["review"]["suggested_category"] is None


def test_no_match_keyword_hint_is_not_sent_to_the_model(store):
    # improvement 1: "other" only means no word matched. The model read it as the answer.
    set_store(store)
    fake = FakeChatModel(responses=[APPROVE])
    msg = make_message("classifier", "reviewer", "review_request", {
        "request": {"action": "categorize_expense", "description": "Officeworks printer ink", "amount": 45},
        "proposal": {"action": "categorize_expense", "category": "education", "reasoning": "r", "confidence": 0.8},
        "round": 1})
    reply = ReviewerAgent(llm=fake).handle(msg)
    human_message = fake.calls[0][-1].content
    evidence = human_message.split("Evidence:")[1].split("Rule checks:")[0]
    assert "keyword_hint" not in evidence
    assert reply.intent == "approved"


def test_pass_checks_are_shown_to_the_model(store):
    # improvement 1: the model sees what the data already confirmed
    set_store(store)
    fake = FakeChatModel(responses=[APPROVE])
    ReviewerAgent(llm=fake).handle(review_msg("check_unusual_expense", 14, category="food", is_unusual=True))
    sent = str(fake.calls[0])
    assert "PASS" in sent
    assert "matches the proposal" in sent
    assert "only used to work out the category average" in sent


def test_parse_review_handles_fences_and_chatter():
    fenced = '```json\n{"verdict": "approve", "issues": [], "feedback": "", "confidence": 0.8}\n```'
    assert parse_review(fenced).verdict == "approve"
    chatty = 'Here you go: {"verdict": "revise", "feedback": "no", "confidence": 0.4} thanks'
    assert parse_review(chatty).verdict == "revise"
    assert parse_review("nope") is None
    assert isinstance(parse_review('{"verdict": "approve"}'), ReviewResult)


def test_reviewer_logs_received_and_sent(store, caplog):
    set_store(store)
    with caplog.at_level(logging.INFO, logger="expense_agent"):
        ReviewerAgent().handle(review_msg(category="transport"))
    assert "AGENT reviewer RECEIVED classifier -> reviewer intent=review_request" in caplog.text
    assert "AGENT reviewer SENT reviewer -> orchestrator intent=approved" in caplog.text
