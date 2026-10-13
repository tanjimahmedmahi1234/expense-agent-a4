"""
COIT12204 Assessment 4, Tanjim Ahmed Mahi (12298057)

The orchestrator (A4). It owns the conversation between the two agents.

For one user request it:
  - builds the first message (orchestrator -> classifier)
  - validates every message before routing it (see messages.validate_message)
  - routes it to the receiver and waits for the reply, with a timeout
  - counts steps and rounds, and stops at MAX_MESSAGES / MAX_ROUNDS
  - decides the final answer, including what to do when something went wrong
  - logs every message and saves the whole conversation

The pattern is a pipeline with a feedback loop:

  orchestrator -> classifier -> reviewer -> (approved) -> orchestrator
                                    |
                                    +-> (revise) -> classifier -> reviewer -> ...

Agents never call each other directly. They only return a message and the
orchestrator decides if it can be delivered.
"""
import json
import os
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import datetime

from app.agent import InputRejected
from app.config import AGENT_TIMEOUT, CONVERSATIONS_FILE, MAX_MESSAGES, MAX_ROUNDS, TEAM_STATE_FILE
from app.logger import logger
from app.messages import (
    ApprovedContent, ClassifyContent, MessageError, Proposal, ReviseContent, TaskRequest,
    make_message, validate_message,
)

SIMULATIONS = ["reviewer_timeout", "invalid_receiver", "bad_message", "force_revise"]


class TeamState:
    """Counters for the whole team. Same idea as AgentState in A3: every change is logged."""

    COUNTERS = ["conversations", "approved_first_round", "approved_after_revision", "max_rounds_hit",
                "unreviewed", "fallbacks", "timeouts", "route_errors", "messages_routed"]

    def __init__(self, path=TEAM_STATE_FILE):
        self.path = path
        for c in self.COUNTERS:
            setattr(self, c, 0)
        self.last_conversation_id = None
        self.last_status = None

    def _transition(self, name, new_value):
        old = getattr(self, name)
        if old == new_value:
            return
        setattr(self, name, new_value)
        logger.info("STATE TRANSITION team.%s: %s -> %s", name, old, new_value)

    def bump(self, name, by=1):
        self._transition(name, getattr(self, name) + by)

    def set(self, name, value):
        self._transition(name, value)

    def snapshot(self):
        d = {c: getattr(self, c) for c in self.COUNTERS}
        d["last_conversation_id"] = self.last_conversation_id
        d["last_status"] = self.last_status
        return d

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.snapshot(), f, indent=2)
        except Exception as e:
            logger.error("TEAM STATE could not save: %s", e)

    @classmethod
    def load(cls, path=TEAM_STATE_FILE):
        s = cls(path)
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in data.items():
                    if hasattr(s, k):
                        setattr(s, k, v)
            except Exception as e:
                logger.warning("TEAM STATE file %s is corrupt (%s), starting fresh", path, e)
                s = cls(path)
        return s

    def reset(self):
        logger.info("STATE RESET team")
        self.__init__(self.path)
        self.save()


class Orchestrator:
    def __init__(self, classifier, reviewer, store, state=None, max_rounds=MAX_ROUNDS,
                 max_messages=MAX_MESSAGES, timeout=AGENT_TIMEOUT, conversations_file=CONVERSATIONS_FILE):
        self.agents = {"classifier": classifier, "reviewer": reviewer}
        self.classifier = classifier
        self.store = store
        self.state = state if state is not None else TeamState.load()
        self.max_rounds = max_rounds
        self.max_messages = max_messages
        self.timeout = timeout
        self.conversations_file = conversations_file
        self.recent = deque(maxlen=50)
        self._load_recent()

    # ------------------------------------------------------------------

    def run(self, action, expense_id=None, description=None, amount=None, merchant=None,
            days=30, simulate=None):
        start = time.time()
        conv = uuid.uuid4().hex[:6]
        request = TaskRequest(action=action, expense_id=expense_id, description=description,
                              amount=amount, merchant=merchant, days=days)
        logger.info("ORCH START [%s] action=%s expense_id=%s simulate=%s", conv, action, expense_id, simulate)
        self.state.bump("conversations")
        self.state.set("last_conversation_id", conv)

        transcript, errors = [], []
        last_proposal, last_review, last_round = None, None, 0
        stop_reason = None
        tweaked = False

        msg = make_message("orchestrator", "classifier", "classify_request",
                           ClassifyContent(request=request), conversation_id=conv)
        step = 0
        while True:
            step += 1
            msg.step = step
            msg.conversation_id = conv

            if step > self.max_messages:
                logger.warning("MAX MESSAGES [%s] stopped after %d messages", conv, self.max_messages)
                errors.append({"step": step, "kind": "max_messages", "detail": f"more than {self.max_messages} messages"})
                stop_reason = "max_messages"
                break

            # demo hooks, only when the request asks for them
            if simulate and not tweaked and msg.sender == "classifier":
                tweaked = True
                if simulate == "invalid_receiver":
                    logger.warning("SIMULATE [%s] changing receiver to 'auditor'", conv)
                    msg.receiver = "auditor"
                elif simulate == "bad_message":
                    logger.warning("SIMULATE [%s] removing the proposal from the message", conv)
                    msg.content.pop("proposal", None)
                    last_proposal = None
            if msg.receiver == "reviewer":
                if simulate == "force_revise":
                    msg.constraints["simulate"] = "force_revise"
                if simulate == "reviewer_timeout":
                    msg.constraints["simulate_delay"] = 3

            # 1. validate
            try:
                validate_message(msg)
            except MessageError as e:
                logger.error("ROUTE ERROR [%s] step=%d %s | %s", conv, step, msg.short(), e)
                transcript.append(self._record(msg, error=str(e)))
                errors.append({"step": step, "kind": e.kind, "detail": e.detail, "message_id": msg.message_id})
                self.state.bump("route_errors")
                stop_reason = e.kind
                break
            transcript.append(self._record(msg))

            # 2. messages for the orchestrator end the conversation
            if msg.receiver == "orchestrator":
                logger.info("MSG RECEIVED [%s] step=%d %s", conv, step, msg.short())
                c = ApprovedContent(**msg.content)
                last_proposal, last_review, last_round = c.proposal, c.review, c.round
                stop_reason = "approved"
                break

            # 3. don't give the classifier another go if it has used all its rounds
            if msg.intent == "revise_request":
                c = ReviseContent(**msg.content)
                last_review = c.review
                if c.round >= self.max_rounds:
                    logger.warning("MAX ROUNDS [%s] step=%d not delivering %s, already %d rounds",
                                   conv, step, msg.short(), c.round)
                    stop_reason = "max_rounds"
                    break

            self.state.bump("messages_routed")
            logger.info("MSG ROUTED [%s] step=%d %s", conv, step, msg.short())

            # 4. deliver it and wait for the reply
            timeout = 1.0 if (simulate == "reviewer_timeout" and msg.receiver == "reviewer") else self.timeout
            try:
                reply = self._call_with_timeout(self.agents[msg.receiver].handle, msg, timeout)
            except FutureTimeout:
                msg.constraints["cancelled"] = True  # tells the agent thread to stop
                logger.error("TIMEOUT [%s] %s did not answer within %.1fs", conv, msg.receiver, timeout)
                errors.append({"step": step, "kind": "timeout", "detail": f"{msg.receiver} took longer than {timeout}s"})
                self.state.bump("timeouts")
                stop_reason = f"{msg.receiver}_timeout"
                break
            except InputRejected:
                raise
            except Exception as e:
                logger.error("AGENT ERROR [%s] %s raised %s: %s", conv, msg.receiver, type(e).__name__, e)
                errors.append({"step": step, "kind": "agent_error", "detail": f"{msg.receiver}: {type(e).__name__}: {e}"})
                stop_reason = f"{msg.receiver}_error"
                break

            if reply is None:
                errors.append({"step": step, "kind": "no_reply", "detail": f"{msg.receiver} returned nothing"})
                stop_reason = f"{msg.receiver}_error"
                break

            # remember the newest proposal, so there is something to fall back to
            if reply.intent == "review_request":
                try:
                    last_proposal = Proposal(**reply.content["proposal"])
                    last_round = reply.content.get("round", last_round + 1)
                except Exception:
                    pass
            msg = reply

        final, status, saved = self._finish(request, stop_reason, last_proposal, last_review, last_round, errors)
        latency = int((time.time() - start) * 1000)

        self.state.set("last_status", status)
        self.state.save()
        logger.info("ORCH END [%s] status=%s stop=%s rounds=%d messages=%d latency=%dms",
                    conv, status, stop_reason, last_round, len(transcript), latency)

        conversation = {
            "conversation_id": conv,
            "time": datetime.now().isoformat(timespec="seconds"),
            "action": action,
            "request": request.model_dump(mode="json"),
            "status": status,
            "stop_reason": stop_reason,
            "rounds": last_round,
            "final": final.model_dump(mode="json") if final else None,
            "saved_to_store": saved,
            "messages": transcript,
            "errors": errors,
            "simulate": simulate,
            "latency_ms": latency,
        }
        self._save_conversation(conversation)
        return conversation

    # ------------------------------------------------------------------

    def _finish(self, request, stop_reason, proposal, review, rounds, errors):
        """Decide what the user gets back. Returns (final proposal, status, saved?)."""
        if stop_reason == "approved":
            status = "approved" if rounds <= 1 else "approved_after_revision"
            self.state.bump("approved_first_round" if rounds <= 1 else "approved_after_revision")
            final = proposal

        elif stop_reason == "max_rounds" and proposal is not None:
            # the two agents never agreed. Use the reviewer's suggestion if it gave one.
            status = "max_rounds"
            self.state.bump("max_rounds_hit")
            final = proposal.model_copy()
            if review is not None and request.action == "categorize_expense" and review.suggested_category:
                final.category = review.suggested_category
                final.reasoning = f"The agents did not agree after {rounds} rounds, so the reviewer's suggestion is used. " + final.reasoning
            elif review is not None and request.action == "check_unusual_expense" and review.suggested_is_unusual is not None:
                final.is_unusual = review.suggested_is_unusual
                final.reasoning = f"The agents did not agree after {rounds} rounds, so the reviewer's answer is used. " + final.reasoning
            else:
                final.reasoning = f"Not approved by the reviewer after {rounds} rounds. " + final.reasoning
            final.confidence = min(final.confidence, 0.5)

        elif proposal is not None:
            # the classifier answered but the review never happened (timeout, bad route, ...)
            status = "unreviewed"
            self.state.bump("unreviewed")
            final = proposal.model_copy()
            final.reasoning = f"Not reviewed ({stop_reason}). " + final.reasoning
            final.confidence = min(final.confidence, 0.5)

        else:
            # nothing usable from the classifier at all, use the A3 rules
            status = "fallback"
            self.state.bump("fallbacks")
            why = errors[-1]["detail"] if errors else stop_reason
            final = self.classifier.fallback(request, why)

        saved = False
        if status.startswith("approved") and request.action == "categorize_expense" \
                and request.expense_id and final.category:
            self.store.set_category(request.expense_id, final.category)
            saved = True
        return final, status, saved

    def _call_with_timeout(self, fn, msg, timeout):
        # Python can't kill a thread, so after a timeout the old call may still finish in
        # the background. Its reply is just ignored because the conversation has moved on.
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(fn, msg)
        try:
            return future.result(timeout=timeout)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def _record(self, msg, error=None):
        d = msg.model_dump(mode="json")
        d["valid"] = error is None
        d["error"] = error
        return d

    # ---- conversations on disk ----

    def _save_conversation(self, conversation):
        self.recent.append(conversation)
        try:
            os.makedirs(os.path.dirname(self.conversations_file) or ".", exist_ok=True)
            with open(self.conversations_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(conversation, default=str) + "\n")
        except Exception as e:
            logger.error("ORCH could not save conversation: %s", e)

    def _load_recent(self):
        if not os.path.exists(self.conversations_file):
            return
        try:
            with open(self.conversations_file, encoding="utf-8") as f:
                for line in f.readlines()[-50:]:
                    if line.strip():
                        self.recent.append(json.loads(line))
        except Exception as e:
            logger.warning("ORCH could not read old conversations (%s)", e)

    def get_conversation(self, conversation_id):
        for c in self.recent:
            if c["conversation_id"] == conversation_id:
                return c
        return None
