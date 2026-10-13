"""COIT12204 Assessment 4, Tanjim Ahmed Mahi (12298057)

A4: the message schema and validate_message()."""
import pytest

from app.messages import AgentMessage, MessageError, make_message, validate_message


def good_review_content(category="transport"):
    return {
        "request": {"action": "categorize_expense", "expense_id": 5},
        "proposal": {"action": "categorize_expense", "expense_id": 5, "category": category,
                     "reasoning": "r", "confidence": 0.9},
        "round": 1,
    }


def test_good_message_passes():
    msg = make_message("classifier", "reviewer", "review_request", good_review_content())
    content = validate_message(msg)
    assert content.proposal.category == "transport"
    assert len(msg.message_id) == 8
    assert msg.constraints == {"format": "json"}


def test_invalid_receiver():
    msg = make_message("classifier", "auditor", "review_request", good_review_content())
    with pytest.raises(MessageError) as e:
        validate_message(msg)
    assert e.value.kind == "invalid_receiver"


def test_invalid_sender():
    msg = make_message("hacker", "reviewer", "review_request", good_review_content())
    with pytest.raises(MessageError) as e:
        validate_message(msg)
    assert e.value.kind == "invalid_sender"


def test_unknown_intent():
    msg = make_message("classifier", "reviewer", "delete_everything", {})
    with pytest.raises(MessageError) as e:
        validate_message(msg)
    assert e.value.kind == "bad_route"


def test_wrong_direction_is_a_bad_route():
    # the classifier is not allowed to approve its own work
    msg = make_message("classifier", "orchestrator", "approved", good_review_content())
    with pytest.raises(MessageError) as e:
        validate_message(msg)
    assert e.value.kind == "bad_route"


def test_missing_field_is_bad_content():
    content = good_review_content()
    del content["proposal"]
    msg = make_message("classifier", "reviewer", "review_request", content)
    with pytest.raises(MessageError) as e:
        validate_message(msg)
    assert e.value.kind == "bad_content"
    assert "proposal" in e.value.detail


def test_confidence_out_of_range_is_bad_content():
    content = good_review_content()
    content["proposal"]["confidence"] = 3
    with pytest.raises(MessageError):
        validate_message(make_message("classifier", "reviewer", "review_request", content))


def test_made_up_category_is_bad_content():
    msg = make_message("classifier", "reviewer", "review_request", good_review_content("crypto"))
    with pytest.raises(MessageError) as e:
        validate_message(msg)
    assert "crypto" in e.value.detail


def test_short_line_for_the_log():
    msg = AgentMessage(sender="reviewer", receiver="classifier", intent="revise_request", content={})
    assert msg.short().startswith("reviewer -> classifier intent=revise_request id=")
