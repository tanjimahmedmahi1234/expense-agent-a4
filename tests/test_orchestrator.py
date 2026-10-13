"""COIT12204 Assessment 4, Tanjim Ahmed Mahi (12298057)

A4: the orchestrator with both agents on scripted fake models."""
import json
import logging
import time

from tests.fakes import json_answer, text_answer, tool_request

APPROVE = json_answer(verdict="approve", issues=[], feedback="", confidence=0.9)


def cat(category, expense_id=5, confidence=0.9):
    return json_answer(action="categorize_expense", expense_id=expense_id, category=category,
                       reasoning="r", confidence=confidence)


def revise(suggestion):
    return json_answer(verdict="revise", issues=["wrong category"], feedback=f"This should be {suggestion}.",
                       suggested_category=suggestion, confidence=0.8)


def route(conv):
    return [f"{m['sender']}>{m['receiver']}:{m['intent']}" for m in conv["messages"]]


# ---------- normal flow ----------

def test_approved_first_round(make_team, store):
    team = make_team([tool_request("lookup_expense", "a", expense_id=5), cat("transport")], [APPROVE])
    conv = team.run("categorize_expense", expense_id=5)
    assert conv["status"] == "approved"
    assert conv["rounds"] == 1
    assert route(conv) == ["orchestrator>classifier:classify_request",
                           "classifier>reviewer:review_request",
                           "reviewer>orchestrator:approved"]
    assert [m["step"] for m in conv["messages"]] == [1, 2, 3]
    assert conv["saved_to_store"] is True
    assert store.get(5).category == "transport"


def test_messages_are_linked_by_reply_to(make_team):
    team = make_team([cat("transport")], [APPROVE])
    msgs = team.run("categorize_expense", expense_id=5)["messages"]
    assert msgs[1]["reply_to"] == msgs[0]["message_id"]
    assert msgs[2]["reply_to"] == msgs[1]["message_id"]
    assert len({m["conversation_id"] for m in msgs}) == 1


def test_revise_then_approve(make_team, store):
    team = make_team([cat("entertainment"), cat("transport")], [revise("transport"), APPROVE])
    conv = team.run("categorize_expense", expense_id=5)
    assert conv["status"] == "approved_after_revision"
    assert conv["rounds"] == 2
    assert "reviewer>classifier:revise_request" in route(conv)
    assert conv["final"]["category"] == "transport"
    # the classifier's second prompt contains the reviewer's note
    assert "This should be transport." in str(team.classifier_fake.calls[1])
    assert store.get(5).category == "transport"


def test_classifier_does_not_save_before_approval(make_team, store):
    team = make_team([cat("entertainment")] * 3, [revise("transport")] * 3)
    team.run("categorize_expense", expense_id=5, simulate="force_revise")
    assert store.get(5).category is None  # never approved, never saved


# ---------- iteration limits ----------

def test_max_rounds_uses_reviewer_suggestion(make_team, caplog):
    # force_revise is a HARD problem, so the soft one-round limit doesn't stop it
    team = make_team([cat("entertainment")] * 3, [revise("transport")] * 3)
    with caplog.at_level(logging.WARNING, logger="expense_agent"):
        conv = team.run("categorize_expense", expense_id=5, simulate="force_revise")
    assert conv["status"] == "max_rounds"
    assert conv["rounds"] == 3
    assert conv["final"]["category"] == "transport"
    assert conv["final"]["confidence"] <= 0.5
    assert "MAX ROUNDS" in caplog.text
    assert len(team.classifier_fake.calls) == 3


def test_max_rounds_setting_is_respected(make_team):
    team = make_team([cat("entertainment")] * 5, [revise("transport")] * 5, max_rounds=1)
    conv = team.run("categorize_expense", expense_id=5)
    assert conv["status"] == "max_rounds"
    assert len(team.classifier_fake.calls) == 1


def test_max_messages_guard(make_team, caplog):
    team = make_team([cat("entertainment")] * 5, [revise("transport")] * 5, max_rounds=10, max_messages=4)
    with caplog.at_level(logging.WARNING, logger="expense_agent"):
        conv = team.run("categorize_expense", expense_id=5, simulate="force_revise")
    assert conv["stop_reason"] == "max_messages"
    assert len(conv["messages"]) == 4
    assert "MAX MESSAGES" in caplog.text


# ---------- invalid messages and receivers ----------

def test_invalid_receiver_is_caught_and_logged(make_team, caplog):
    team = make_team([cat("transport")], [APPROVE])
    with caplog.at_level(logging.ERROR, logger="expense_agent"):
        conv = team.run("categorize_expense", expense_id=5, simulate="invalid_receiver")
    assert conv["status"] == "unreviewed"
    assert conv["stop_reason"] == "invalid_receiver"
    assert conv["messages"][-1]["receiver"] == "auditor"
    assert conv["messages"][-1]["valid"] is False
    assert conv["final"]["category"] == "transport"
    assert conv["final"]["confidence"] <= 0.5
    assert "ROUTE ERROR" in caplog.text
    assert team.state.route_errors == 1


def test_bad_message_falls_back_to_rules(make_team):
    team = make_team([cat("transport")], [APPROVE])
    conv = team.run("categorize_expense", expense_id=5, simulate="bad_message")
    assert conv["status"] == "fallback"
    assert conv["stop_reason"] == "bad_content"
    assert conv["final"]["confidence"] == 0.3  # A3 rule fallback
    assert conv["final"]["category"] == "transport"  # "uber" keyword


def test_agent_returning_a_wrong_route(make_team):
    team = make_team([cat("transport")], [APPROVE])
    real = team.agents["reviewer"].handle

    def confused(msg):
        reply = real(msg)
        reply.receiver = "reviewer"  # sends to itself
        return reply

    team.agents["reviewer"].handle = confused
    conv = team.run("categorize_expense", expense_id=5)
    assert conv["stop_reason"] == "bad_route"
    assert conv["status"] == "unreviewed"


# ---------- timeouts and crashes ----------

def test_reviewer_timeout(make_team, caplog):
    team = make_team([cat("transport")], [APPROVE])
    start = time.time()
    with caplog.at_level(logging.ERROR, logger="expense_agent"):
        conv = team.run("categorize_expense", expense_id=5, simulate="reviewer_timeout")
    assert time.time() - start < 2.5  # did not wait the full 3 seconds
    assert conv["status"] == "unreviewed"
    assert conv["stop_reason"] == "reviewer_timeout"
    assert "TIMEOUT" in caplog.text
    assert team.state.timeouts == 1


def test_timeout_setting(make_team):
    team = make_team([cat("transport")], [APPROVE], timeout=0.2)
    real = team.agents["classifier"].handle

    def slow(msg):
        time.sleep(0.5)
        return real(msg)  # real handle() sees the cancelled flag and stops

    team.agents["classifier"].handle = slow
    conv = team.run("categorize_expense", expense_id=5)
    assert conv["stop_reason"] == "classifier_timeout"
    assert conv["status"] == "fallback"  # no proposal ever arrived
    time.sleep(0.6)  # let the background thread finish before the next test
    assert team.classifier_fake.calls == []  # it stopped before calling the model


def test_agent_crash_is_handled(make_team, caplog):
    team = make_team([cat("transport")], [APPROVE])

    def boom(msg):
        raise KeyError("boom")

    team.agents["reviewer"].handle = boom
    with caplog.at_level(logging.ERROR, logger="expense_agent"):
        conv = team.run("categorize_expense", expense_id=5)
    assert conv["stop_reason"] == "reviewer_error"
    assert conv["status"] == "unreviewed"
    assert "AGENT ERROR" in caplog.text


def test_classifier_model_down_still_reviewed(make_team):
    # the A3 fallback inside the classifier still produces a proposal, and it still gets reviewed
    team = make_team([], [APPROVE])
    team.classifier_fake.raise_error = True
    conv = team.run("categorize_expense", expense_id=5)
    assert conv["status"] == "approved"
    assert conv["messages"][1]["content"]["classifier_used_fallback"] is True


# ---------- other actions ----------

def test_unusual_hard_rule_forces_a_second_round(make_team):
    wrong = json_answer(action="check_unusual_expense", expense_id=14, category="food", is_unusual=False,
                        reasoning="r", confidence=0.9)
    right = json_answer(action="check_unusual_expense", expense_id=14, category="food", is_unusual=True,
                        reasoning="r", confidence=0.9)
    team = make_team([wrong, right], [APPROVE, APPROVE])
    conv = team.run("check_unusual_expense", expense_id=14)
    assert conv["status"] == "approved_after_revision"
    assert conv["final"]["is_unusual"] is True


def test_summary_reviewed_by_rules_only(make_team):
    good = json_answer(action="monthly_summary", summary="You spent $2306.44, mostly rent.", reasoning="r", confidence=0.9)
    team = make_team([good])  # reviewer has no model
    conv = team.run("monthly_summary", days=30)
    assert conv["status"] == "approved"


# ---------- logging, state and persistence ----------

def test_every_message_is_logged(make_team, caplog):
    team = make_team([cat("transport")], [APPROVE])
    with caplog.at_level(logging.INFO, logger="expense_agent"):
        conv = team.run("categorize_expense", expense_id=5)
    cid = conv["conversation_id"]
    assert f"ORCH START [{cid}]" in caplog.text
    assert f"MSG ROUTED [{cid}] step=1 orchestrator -> classifier" in caplog.text
    assert f"MSG ROUTED [{cid}] step=2 classifier -> reviewer" in caplog.text
    assert f"MSG RECEIVED [{cid}] step=3 reviewer -> orchestrator intent=approved" in caplog.text
    assert f"ORCH END [{cid}] status=approved" in caplog.text
    assert "AGENT classifier RECEIVED" in caplog.text
    assert "AGENT classifier SENT" in caplog.text


def test_soft_disagreement_only_gets_one_round(make_team, caplog):
    # improvement 2: found in the first evaluation, the reviewer kept sending a right answer
    # back over something that wasn't a fact. Now round 2 approves unless there is a HARD problem.
    team = make_team([cat("entertainment")] * 3, [revise("transport")] * 3)
    with caplog.at_level(logging.WARNING, logger="expense_agent"):
        conv = team.run("categorize_expense", expense_id=5)
    assert conv["status"] == "approved_after_revision"
    assert conv["rounds"] == 2
    assert len(team.classifier_fake.calls) == 2
    assert "REVIEW SOFT LIMIT" in caplog.text


def test_team_state_counts_and_logs(make_team, caplog):
    team = make_team([cat("transport"), cat("entertainment"), cat("transport")], [APPROVE, revise("transport"), APPROVE])
    with caplog.at_level(logging.INFO, logger="expense_agent"):
        team.run("categorize_expense", expense_id=5)
        team.run("categorize_expense", expense_id=5)
    assert team.state.conversations == 2
    assert team.state.approved_first_round == 1
    assert team.state.approved_after_revision == 1
    assert "STATE TRANSITION team.conversations: 1 -> 2" in caplog.text


def test_conversation_is_saved_to_disk(make_team):
    team = make_team([cat("transport")], [APPROVE])
    conv = team.run("categorize_expense", expense_id=5)
    with open(team.conversations_file) as f:
        saved = [json.loads(line) for line in f]
    assert saved[-1]["conversation_id"] == conv["conversation_id"]
    assert len(saved[-1]["messages"]) == 3
    assert team.get_conversation(conv["conversation_id"])["status"] == "approved"


def test_reviewer_bad_json_does_not_break_the_team(make_team):
    team = make_team([cat("transport")], [text_answer("ok"), text_answer("ok")])
    conv = team.run("categorize_expense", expense_id=5)
    assert conv["status"] == "approved"  # rules passed
