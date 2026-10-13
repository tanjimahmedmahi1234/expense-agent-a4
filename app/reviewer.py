"""
COIT12204 Assessment 4, Tanjim Ahmed Mahi (12298057)

The Reviewer agent (new in A4).

It checks the classifier's proposal before the user sees it. It does three things:

  1. gathers its own evidence with the same tools (it doesn't trust the classifier's copy)
  2. runs simple rule checks. HARD means a fact from the data, SOFT means worth a look
  3. asks the LLM for a verdict in a fixed JSON shape (ReviewResult)

A HARD rule problem always wins over the LLM. If the LLM fails or there is no API
key, the reviewer still works using only the rules.

The reviewer is a plain LangChain chain (prompt | llm), not an AgentExecutor,
because it judges the evidence it is given and doesn't need to decide which tools
to call.
"""
import json
import re
import time

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app import agent as agent_module  # so tests can monkeypatch get_llm
from app.agent import rule_category
from app.base_agent import BaseAgent, check_cancelled
from app.config import CATEGORIES
from app.logger import logger
from app.messages import ApprovedContent, ReviewContent, ReviewResult, ReviseContent, make_message
from app.store import today
from app.tools import category_totals, get_store, lookup_expense

REVIEW_PARSER = PydanticOutputParser(pydantic_object=ReviewResult)

REVIEWER_SYSTEM = (
    "You are the Reviewer in a two agent personal finance app. The Classifier agent made a "
    "proposal and you check it before the user sees it. Today's date is {today}.\n\n"
    "You get the user's request, the Classifier's proposal, evidence fetched from the data, "
    "and the results of some simple rule checks.\n\n"
    "Rules:\n"
    "- Check the proposal against the evidence. Don't redo the whole job.\n"
    "- Approve if the answer is reasonable and the evidence supports it. Wording is not a reason to send it back.\n"
    "- Revise only when something is actually wrong: a category that doesn't fit what the merchant sells, "
    "an unusual decision that contradicts the averages, summary numbers that don't match the evidence, "
    "or high confidence when there is no answer.\n"
    "- A rule check marked HARD is a fact from the data. If there is one, the verdict must be revise.\n"
    "- A rule check marked PASS is a fact from the data that agrees with the proposal. Don't argue with it.\n"
    "- keyword_hint is only a guess from a short word list. It is a hint, not the answer.\n"
    "- For the unusual check, the last 30 days are only used to work out the category average. "
    "An older expense can still be compared with that average.\n"
    "- When you revise, write feedback the Classifier can act on in one or two sentences and fill "
    "suggested_category or suggested_is_unusual if you know the right value.\n"
    "- Categories: food is groceries, cafes, restaurants and food delivery. transport is public transport, "
    "ride share, fuel and parking. utilities is power, gas, water, internet and phone. entertainment is "
    "streaming, music, games, movies. shopping is electronics, clothes and general retail. health is "
    "chemist, doctor, gym. education is textbooks, courses and fees. other only when nothing fits.\n"
    "- Reply with ONLY the JSON object. No markdown fences, no text before or after it.\n\n"
    "{format_instructions}"
)

REVIEWER_HUMAN = (
    "Request:\n{request}\n\nProposal from the Classifier:\n{proposal}\n\n"
    "Evidence:\n{evidence}\n\nRule checks:\n{checks}\n\nThis is review round {round}."
)


def build_reviewer_prompt():
    prompt = ChatPromptTemplate.from_messages([("system", REVIEWER_SYSTEM), ("human", REVIEWER_HUMAN)])
    return prompt.partial(format_instructions=REVIEW_PARSER.get_format_instructions())


def parse_review(text):
    """Same idea as parse_output in agent.py: strip fences, try, then look for { ... }."""
    if text is None:
        return None
    text = str(text).strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return REVIEW_PARSER.parse(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return ReviewResult(**json.loads(text[start:end + 1]))
            except Exception:
                pass
    logger.warning("REVIEW PARSE FAILED %r", text[:120])
    return None


class ReviewerAgent(BaseAgent):
    name = "reviewer"

    def __init__(self, llm=None):
        self.chain = None
        self.prompt = build_reviewer_prompt()
        try:
            self.llm = llm or agent_module.get_llm()
            self.chain = self.prompt | self.llm
        except Exception as e:
            logger.error("REVIEWER could not create the LLM: %s. Rule checks only.", e)

    # ---------------------------------------------------------------

    def process(self, msg):
        content = ReviewContent(**msg.content)

        # test hook for the demo: pretend to be slow so the orchestrator times out
        delay = msg.constraints.get("simulate_delay")
        if delay:
            logger.warning("SIMULATE reviewer sleeping %ss", delay)
            for _ in range(int(float(delay) * 10)):
                check_cancelled(msg)
                time.sleep(0.1)

        evidence = self.gather_evidence(content)
        check_cancelled(msg)  # don't pay for an LLM call nobody will read
        checks = self.rule_checks(content, evidence)
        if msg.constraints.get("simulate") == "force_revise":
            checks.append({"level": "HARD", "check": "simulation",
                           "detail": "forced disagreement for the demo (simulate=force_revise)"})

        review = self.llm_review(content, evidence, checks)
        review = self.combine(content, review, checks)
        logger.info("REVIEW VERDICT round=%d verdict=%s issues=%s", content.round, review.verdict, review.issues)

        if review.verdict == "approve":
            body = ApprovedContent(request=content.request, proposal=content.proposal,
                                   review=review, round=content.round)
            return make_message("reviewer", "orchestrator", "approved", body)
        body = ReviseContent(request=content.request, previous_proposal=content.proposal,
                             review=review, round=content.round)
        return make_message("reviewer", "classifier", "revise_request", body)

    # ---------------------------------------------------------------

    def gather_evidence(self, content):
        req = content.request
        ev = {}
        if req.expense_id is not None:
            ev["expense"] = lookup_expense.invoke({"expense_id": req.expense_id})
        if req.action == "monthly_summary":
            ev["totals"] = category_totals.invoke({"days": req.days})
        elif req.action == "check_unusual_expense":
            ev["totals_30_days"] = category_totals.invoke({"days": 30})
        if req.action == "categorize_expense":
            hint = None
            if "expense" in ev and "error" not in ev["expense"]:
                e = ev["expense"]
                hint = rule_category(e["description"], e.get("merchant"))
            elif req.description:
                hint = rule_category(req.description, req.merchant)
            # improvement 1: "other" from the word list only means no word matched. In the
            # first evaluation the reviewer model read it as the right answer and talked
            # the classifier out of a correct category, so I don't pass it on any more.
            if hint and hint != "other":
                ev["keyword_hint"] = hint
        return ev

    def rule_checks(self, content, ev):
        """Cheap checks straight from the data. Each one is a dict with a level:
        HARD = the data says the proposal is wrong, SOFT = worth a look,
        PASS = the data agrees with the proposal (added after the first evaluation, so the
        model can see what is already confirmed instead of arguing with it)."""
        req, p = content.request, content.proposal
        checks = []

        def add(level, name, detail):
            checks.append({"level": level, "check": name, "detail": detail})

        expense = ev.get("expense")
        missing = expense is not None and "error" in expense

        if req.action == "categorize_expense":
            if missing:
                if p.category is not None:
                    add("HARD", "missing_expense", f"expense {req.expense_id} does not exist but a category was given")
                if p.confidence > 0.2:
                    add("HARD", "confidence", f"expense {req.expense_id} does not exist but confidence is {p.confidence}")
            elif p.category is None:
                add("HARD", "no_answer", "the expense exists but no category was given")
            hint = ev.get("keyword_hint")
            if hint and p.category and hint != p.category:
                add("SOFT", "keyword_hint", f"the keyword rule says '{hint}', the proposal says '{p.category}'")
            elif hint and hint == p.category:
                add("PASS", "keyword_hint", f"the keyword rule also says '{hint}'")

        elif req.action == "check_unusual_expense":
            if missing:
                if p.is_unusual is not None:
                    add("HARD", "missing_expense", f"expense {req.expense_id} does not exist but an answer was given")
            elif expense is not None:
                expected, detail = self._expected_unusual(expense, p)
                if expected is not None and p.is_unusual is not None and expected != p.is_unusual:
                    add("HARD", "unusual_maths", detail)
                elif expected is not None and expected == p.is_unusual:
                    add("PASS", "unusual_maths", detail + ", which matches the proposal")
                elif expected is None and detail:
                    add("SOFT", "unusual_maths", detail)

        else:  # monthly_summary
            totals = ev.get("totals", {})
            total = totals.get("total")
            numbers = [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*\.?\d*", p.summary or "")]
            if total is not None and not any(abs(n - total) < 1.0 for n in numbers):
                add("HARD", "summary_total", f"the summary does not mention the real total ${total:.2f}")
            elif total is not None:
                add("PASS", "summary_total", f"the summary has the right total ${total:.2f}")
            real = {k: v for k, v in totals.get("by_category", {}).items() if k != "uncategorised"}
            if real:
                biggest = max(real, key=real.get)
                if biggest not in (p.summary or "").lower():
                    add("SOFT", "summary_biggest", f"the biggest category is {biggest} but the summary doesn't name it")

        if content.classifier_used_fallback:
            add("SOFT", "fallback", "the classifier used its rule fallback, not the model")
        return checks

    def _expected_unusual(self, expense, p):
        """More than double the average of the other expenses in the same category, last 30 days.
        I work it out two ways (with and without the expense itself). Only if both agree is it HARD."""
        cat = expense.get("category") or p.category or rule_category(expense["description"], expense.get("merchant"))
        store = get_store()
        recent = [e for e in store.recent(30) if e.category == cat]
        others = [e.amount for e in recent if e.id != expense["id"]]
        including = [e.amount for e in recent]
        if len(others) < 2:
            return None, f"not enough {cat} expenses in the last 30 days to judge"
        avg_ex = sum(others) / len(others)
        avg_in = sum(including) / len(including) if including else avg_ex
        exp_ex = expense["amount"] > 2 * avg_ex
        exp_in = expense["amount"] > 2 * avg_in
        detail = (f"amount ${expense['amount']:.2f}, {cat} average in the last 30 days is "
                  f"${avg_ex:.2f} without this expense, so it {'is' if exp_ex else 'is not'} more than double")
        if exp_ex != exp_in:
            return None, detail + " (but the answer changes if the expense itself is in the average)"
        return exp_ex, detail

    def llm_review(self, content, evidence, checks):
        """Ask the model for a ReviewResult. Returns None if there is no model or it fails twice."""
        if self.chain is None:
            return None
        inputs = {
            "today": today().isoformat(),
            "request": content.request.model_dump_json(),
            "proposal": content.proposal.model_dump_json(),
            "evidence": json.dumps(evidence, default=str)[:3000],
            "checks": json.dumps(checks) if checks else "none, all rule checks passed",
            "round": content.round,
        }
        try:
            out = self.chain.invoke(inputs)
            review = parse_review(getattr(out, "content", out))
            if review is None:
                logger.warning("REVIEW RETRY asking the reviewer model again for valid JSON")
                inputs["round"] = f"{content.round}. IMPORTANT: reply with ONLY the JSON object"
                out = self.chain.invoke(inputs)
                review = parse_review(getattr(out, "content", out))
            return review
        except Exception as e:
            logger.error("REVIEW ERROR %s: %s", type(e).__name__, e)
            return None

    def combine(self, content, review, checks):
        """Mix the LLM verdict with the rule checks. HARD rule problems always win."""
        hard = [c for c in checks if c["level"] == "HARD"]

        if review is None:
            logger.warning("REVIEW FALLBACK using rule checks only")
            if hard:
                review = ReviewResult(verdict="revise", issues=[c["detail"] for c in hard],
                                      feedback=" ".join(c["detail"] for c in hard), confidence=0.5)
            else:
                review = ReviewResult(verdict="approve", issues=[], feedback="Rule checks passed.", confidence=0.5)

        elif hard and review.verdict == "approve":
            logger.warning("REVIEW OVERRIDE the model approved but a HARD rule check failed")
            review.verdict = "revise"
            review.issues = [c["detail"] for c in hard] + review.issues
            review.feedback = " ".join(c["detail"] for c in hard)

        # improvement 2: a soft disagreement only gets one round. In the first evaluation
        # the reviewer kept sending right answers back over things that weren't facts
        # (3 rounds on the $95 dinner), so from round 2 only a HARD problem can do that.
        if review.verdict == "revise" and not hard and content.round >= 2:
            logger.warning("REVIEW SOFT LIMIT round=%d no HARD problem, approving with a note", content.round)
            review.verdict = "approve"
            review.feedback = "Approved after one round of soft disagreement: " + (review.feedback or "")

        # don't pass on a suggestion that isn't a real category
        if review.suggested_category is not None and review.suggested_category not in CATEGORIES:
            review.suggested_category = None
        return review
