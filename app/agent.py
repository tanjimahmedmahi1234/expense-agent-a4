"""
COIT12204 Assessment 4, Tanjim Ahmed Mahi (12298057)

The agent. One agent, three actions:

  categorize_expense      pick a category for one expense
  check_unusual_expense   say if an expense is way bigger than normal for its category
  monthly_summary         summarise the last N days of spending

I use create_tool_calling_agent + AgentExecutor from LangChain. The executor does the
tool loop for me. After it finishes I parse the text into AgentResult. If that fails
I try once more, and if it still fails I fall back to simple rules.
"""
import re
import time
import json

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI

from app.config import (
    OPENAI_API_KEY,
    MODEL_NAME,
    PROMPT_VERSION,
    MAX_STEPS,
    MAX_INPUT_CHARS,
    LLM_TIMEOUT,
    CATEGORIES,
)
from app.logger import logger
from app.models import AgentResult
from app.prompts import PARSER, HUMAN_TEMPLATES, build_prompt
from app.state import AgentState
from app.store import ExpenseStore, today
from app.tools import TOOLS, set_store, category_totals


ACTIONS = ["categorize_expense", "check_unusual_expense", "monthly_summary"]


class InputRejected(ValueError):
    """Raised when the user input fails the guardrail checks. The API turns this into a 400."""
    pass


def get_llm():
    # in the tests this function gets monkeypatched to return a fake model
    # max_retries was 1 at first. The evaluation hit the rate limit (429) a lot,
    # so now it retries 3 times before giving up.
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
        api_key=OPENAI_API_KEY,
        timeout=LLM_TIMEOUT,
        max_retries=3,
    )


# ---------- guardrails on the input ----------

BAD_PATTERNS = [
    r"ignore (all |the )?(previous|above|prior) instructions",
    r"disregard (your|the|all) (rules|instructions)",
    r"(reveal|print|show) (your|the) system prompt",
    r"you are now (a|an|the) ",
]


def clean_text(text: str) -> str:
    """Check free text before it goes anywhere near the model."""
    if text is None:
        return ""
    text = str(text).strip()
    # drop control characters, they can mess up the log and the prompt
    text = "".join(ch for ch in text if ch >= " " or ch in "\n\t")
    if len(text) > MAX_INPUT_CHARS:
        logger.warning("GUARDRAIL input too long (%d chars)", len(text))
        raise InputRejected(f"Text is too long, max is {MAX_INPUT_CHARS} characters.")
    for pattern in BAD_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            logger.warning("GUARDRAIL possible prompt injection: %r", text[:60])
            raise InputRejected("Input looks like a prompt injection attempt and was rejected.")
    return text


# ---------- parsing the model output ----------

def parse_output(text):
    """Turn the model's text into an AgentResult. Returns None if it can't."""
    if text is None:
        return None
    # sometimes the model is a list of content blocks instead of a string
    if isinstance(text, list):
        parts = []
        for block in text:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            else:
                parts.append(str(block))
        text = "\n".join(parts)
    text = str(text).strip()

    # strip ```json fences if there are any
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        return PARSER.parse(text)
    except Exception:
        # the model probably talked before the JSON. try to find the first { ... }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                return AgentResult(**json.loads(text[start : end + 1]))
            except Exception:
                pass
    logger.warning("PARSE FAILED could not read JSON from: %r", text[:120])
    return None


# ---------- fallback rules (used when the model fails) ----------

KEYWORDS = {
    "food": ["woolworths", "coles", "aldi", "grocer", "coffee", "cafe", "restaurant", "dinner", "lunch", "uber eats", "menulog", "food"],
    "transport": ["myki", "opal", "uber", "petrol", "fuel", "train", "bus", "parking", "taxi", "ptv"],
    "rent": ["rent", "lease", "bond"],
    "utilities": ["electricity", "gas bill", "water bill", "internet", "nbn", "phone", "telstra", "optus", "origin", "agl"],
    "entertainment": ["netflix", "spotify", "cinema", "movie", "game", "steam", "disney", "concert"],
    "shopping": ["jb hi-fi", "kmart", "amazon", "clothes", "shoes", "headphones", "target", "big w"],
    "health": ["chemist", "pharmacy", "doctor", "gym", "dentist", "medicare", "fitness"],
    "education": ["textbook", "tuition", "course", "udemy", "cqu", "university", "book"],
}


def rule_category(description, merchant=None):
    """Basic keyword matching, used as a backup.
    Fixed after evaluation: "phone" was matching inside "headphones", so now it only
    matches whole words.
    """
    text = (str(description) + " " + str(merchant or "")).lower()
    for cat, words in KEYWORDS.items():
        for w in words:
            if re.search(r"\b" + re.escape(w) + r"\b", text):
                return cat
    return "other"


def rule_is_unusual(store, expense):
    """An expense is unusual if it is more than double the average of the others in its category."""
    cat = expense.category or rule_category(expense.description, expense.merchant)
    # 30 days to match the window the v3 prompt gives the model, so the rule answer
    # and the model answer are judged against the same period.
    others = [e.amount for e in store.recent(30) if e.category == cat and e.id != expense.id]
    if len(others) < 2:
        return False, cat  # not enough data to say
    avg = sum(others) / len(others)
    return expense.amount > 2 * avg, cat


# ---------- the agent ----------

class ExpenseAgent:
    def __init__(self, store: ExpenseStore, state: AgentState = None, llm=None, prompt_version=None):
        self.store = store
        set_store(store)  # so the tools can see the same data
        # if no state is passed in, load the saved one (or start fresh)
        self.state = state if state is not None else AgentState.load()
        self.prompt_version = prompt_version or PROMPT_VERSION
        self.prompt = build_prompt(self.prompt_version)
        self.executor = None

        try:
            self.llm = llm or get_llm()
            agent = create_tool_calling_agent(self.llm, TOOLS, self.prompt)
            self.executor = AgentExecutor(
                agent=agent,
                tools=TOOLS,
                verbose=True,  # prints the thinking steps in the terminal, handy for the video
                max_iterations=MAX_STEPS,
                return_intermediate_steps=True,
                handle_parsing_errors=True,
            )
        except Exception as e:
            # no API key or a bad model name. The agent still works, it just always falls back
            logger.error("AGENT could not create the LLM: %s. Fallback rules will be used.", e)

    # ....................................................................

    def run(self, action, expense_id=None, description=None, amount=None, merchant=None, days=30,
            feedback=None, save=True):
        """Run one action and always return a dict with an AgentResult inside.

        A4: feedback is the reviewer's note when it sends an answer back. save=False lets
        the orchestrator wait for the reviewer to approve before the category is saved.
        """
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action}")

        start_time = time.time()
        logger.info("RUN START action=%s expense_id=%s days=%s", action, expense_id, days)

        # ---- 1. build the human message
        expense = None
        if expense_id is not None:
            expense = self.store.get(expense_id)
            # I only give the model the id so it is forced to call lookup_expense
            expense_block = f"Expense id: {expense_id}. Use the lookup_expense tool to see its details."
        else:
            description = clean_text(description)
            merchant = clean_text(merchant) if merchant else None
            expense_block = f"Description: {description}\nAmount: {amount}\nMerchant: {merchant or 'unknown'}"

        if action == "monthly_summary":
            human = HUMAN_TEMPLATES[action].format(days=days)
        else:
            human = HUMAN_TEMPLATES[action].format(expense_block=expense_block)

        if feedback:
            # the reviewer sent the last answer back, so tell the model what was wrong
            human += (
                "\n\nA reviewer checked your previous answer and sent it back with this note:\n"
                + str(feedback)[:600]
                + "\nLook at the data again and give a corrected answer in the same JSON format."
            )

        # ---- 2. call the executor, parse, retry once, fallback
        result = None
        fallback_used = False
        error = None
        tools_called = []
        raw_output = ""
        warnings = []

        if self.executor is None:
            error = "LLM not available"
        else:
            try:
                raw_output, tools_called = self._invoke(human)
                result = parse_output(raw_output)

                if result is None:
                    logger.warning("RETRY asking the model again for valid JSON")
                    raw_output, more_tools = self._invoke(
                        human + "\n\nIMPORTANT: reply with ONLY the JSON object and nothing else."
                    )
                    tools_called = tools_called + more_tools
                    result = parse_output(raw_output)
            except InputRejected:
                raise
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                logger.error("RUN ERROR %s", error)

        if result is None:
            fallback_used = True
            result = self._fallback(action, expense, expense_id, description, amount, merchant, days, error)
            logger.warning("FALLBACK used for %s", action)

        # ---- 3. checks on the output (guardrails)
        if result.action != action:
            warnings.append(f"model set action to {result.action}, corrected to {action}")
            result.action = action
        if result.category is not None and result.category not in CATEGORIES:
            warnings.append(f"category '{result.category}' is not allowed, changed to other")
            result.category = "other"
        if result.confidence < 0 or result.confidence > 1:
            warnings.append("confidence was outside 0..1, clamped")
            result.confidence = max(0.0, min(1.0, result.confidence))
        if action == "categorize_expense" and result.category is None:
            warnings.append("no category returned")
        if action == "monthly_summary" and not result.summary:
            warnings.append("no summary returned")
        # added after evaluation: the model said the expense does not exist and still
        # gave confidence 0.9. No answer means low confidence.
        no_answer = (action == "categorize_expense" and result.category is None) or \
                    (action == "check_unusual_expense" and result.is_unusual is None)
        if no_answer and result.confidence > 0.2:
            warnings.append(f"confidence {result.confidence} lowered to 0.2 because there is no answer")
            result.confidence = 0.2
        for w in warnings:
            logger.warning("GUARDRAIL %s", w)

        # save the category back to the store if we got one
        if save and action == "categorize_expense" and expense is not None and result.category:
            self.store.set_category(expense.id, result.category)

        # ---- 4. update the state (this is where the STATE TRANSITION lines get logged)
        self.state.record_run(action, result, fallback_used, error, tools_called)

        latency = int((time.time() - start_time) * 1000)
        logger.info(
            "RUN END action=%s fallback=%s tools=%s confidence=%s latency=%dms",
            action, fallback_used, tools_called, result.confidence, latency,
        )
        # False and None are both falsy, so a plain "or" chain printed "summary"
        # for a not-unusual answer and for an expense that does not exist.
        if action == "categorize_expense":
            answer = result.category or "no category"
        elif action == "check_unusual_expense":
            if result.is_unusual is None:
                answer = "could not tell"
            else:
                answer = "unusual" if result.is_unusual else "not unusual"
        else:
            answer = "summary"
        print("Agent finished:", action, "->", answer)

        return {
            "result": result,
            "fallback_used": fallback_used,
            "tools_called": tools_called,
            "warnings": warnings,
            "error": error,
            "raw_output": raw_output if isinstance(raw_output, str) else str(raw_output),
            "prompt_version": self.prompt_version,
            "latency_ms": latency,
            "state": self.state.snapshot(),
        }

    def _invoke(self, human_text):
        """One call to the AgentExecutor. Returns (output text, list of tool names used)."""
        out = self.executor.invoke({"input": human_text, "today": today().isoformat()})
        tools_called = []
        for step in out.get("intermediate_steps", []):
            action_taken = step[0]
            observation = step[1]
            tools_called.append(action_taken.tool)
            self.state.record_tool_call(action_taken.tool, action_taken.tool_input, observation)
        logger.info("MODEL OUTPUT %r", str(out["output"])[:200])
        return out["output"], tools_called

    def _fallback(self, action, expense, expense_id, description, amount, merchant, days, error):
        """Rules only, no model. Confidence is low on purpose so the caller knows."""
        why = f"Fallback rule used because the model step failed ({error or 'invalid output'})."

        if action == "categorize_expense":
            if expense is not None:
                cat = expense.category or rule_category(expense.description, expense.merchant)
            elif expense_id is not None:
                # id was given but the expense does not exist
                return AgentResult(action=action, expense_id=expense_id, category=None,
                                   reasoning=f"Expense {expense_id} does not exist. " + why, confidence=0.0)
            else:
                cat = rule_category(description, merchant)
            return AgentResult(action=action, expense_id=expense_id, category=cat, reasoning=why, confidence=0.3)

        if action == "check_unusual_expense":
            if expense is None:
                return AgentResult(action=action, expense_id=expense_id, is_unusual=None,
                                   reasoning=f"Expense {expense_id} does not exist. " + why, confidence=0.0)
            unusual, cat = rule_is_unusual(self.store, expense)
            return AgentResult(action=action, expense_id=expense_id, category=cat, is_unusual=unusual,
                               reasoning=why, confidence=0.3)

        # monthly_summary
        totals = category_totals.invoke({"days": days})
        by_cat = totals["by_category"]
        # "uncategorised" is a bucket, not a category, so it must not win here
        real = {k: v for k, v in by_cat.items() if k != "uncategorised"}
        biggest = max(real, key=real.get) if real else "nothing"
        summary = (
            f"You spent ${totals['total']:.2f} across {totals['count']} expenses in the last {days} days. "
            f"The biggest category was {biggest}. {totals['uncategorised']} expenses still have no category."
        )
        return AgentResult(action=action, summary=summary, reasoning=why, confidence=0.5)
