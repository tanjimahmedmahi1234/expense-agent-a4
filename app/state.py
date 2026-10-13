"""
COIT12204 Assessment 4, Tanjim Ahmed Mahi (12298057)

Internal state of the agent. The brief asks for an AgentState class with at least three
tracked variables that get updated after every run, plus logging of the transitions.

I track:
  run_count        how many times the agent has run
  error_count      how many runs hit an exception
  fallback_count   how many runs ended up using the rule based fallback
  last_action      the last action that was run
  last_result      the last answer (category / is_unusual / summary)
  last_tool_call   the last tool the model used and its arguments
  history          a list of the last N runs and tool calls
  updated_at       when the state last changed

The state is saved to a JSON file after every run so it survives a restart.
"""
import json
import os
from datetime import datetime

from app.config import STATE_FILE
from app.logger import logger

HISTORY_LIMIT = 50


class AgentState:
    def __init__(self, path=STATE_FILE, history_limit=HISTORY_LIMIT):
        self.path = path
        self.history_limit = history_limit

        self.run_count = 0
        self.error_count = 0
        self.fallback_count = 0
        self.last_action = None
        self.last_result = None
        self.last_tool_call = None
        self.last_error = None
        self.history = []
        self.updated_at = None

    # ---- the one place where a value actually changes ----

    def _transition(self, name, new_value):
        """Set one variable and log the old -> new change. Skips if nothing changed."""
        old_value = getattr(self, name)
        if old_value == new_value:
            return
        setattr(self, name, new_value)
        # cut long values so the log line stays readable
        old_text = str(old_value)[:80]
        new_text = str(new_value)[:80]
        logger.info("STATE TRANSITION %s: %s -> %s", name, old_text, new_text)

    def _add_history(self, entry):
        entry["time"] = datetime.now().isoformat(timespec="seconds")
        self.history.append(entry)
        # keep only the newest entries
        if len(self.history) > self.history_limit:
            self.history = self.history[-self.history_limit :]

    # ---- called by the agent ----

    def record_tool_call(self, name, args, result):
        self._transition("last_tool_call", {"name": name, "args": args})
        self._add_history({"type": "tool", "name": name, "args": args, "result": str(result)[:120]})

    def record_run(self, action, result, fallback_used=False, error=None, tools_called=None):
        """Update everything after one agent run. result is the AgentResult."""
        self._transition("run_count", self.run_count + 1)
        self._transition("last_action", action)

        if error:
            self._transition("error_count", self.error_count + 1)
            self._transition("last_error", str(error)[:200])
        if fallback_used:
            self._transition("fallback_count", self.fallback_count + 1)

        # "last suggestion" depends on the action
        if action == "categorize_expense":
            suggestion = result.category
        elif action == "check_unusual_expense":
            suggestion = result.is_unusual
        else:
            suggestion = result.summary
        self._transition("last_result", suggestion)

        self._add_history(
            {
                "type": "run",
                "action": action,
                "expense_id": result.expense_id,
                "result": suggestion,
                "confidence": result.confidence,
                "fallback": fallback_used,
                "tools": tools_called or [],
                "error": error,
            }
        )
        self.updated_at = datetime.now().isoformat(timespec="seconds")
        self.save()

    def reset(self):
        logger.info("STATE RESET")
        self.__init__(path=self.path, history_limit=self.history_limit)
        self.save()

    # ---- reading ----

    def snapshot(self):
        """Small dict for API responses, without the whole history."""
        return {
            "run_count": self.run_count,
            "error_count": self.error_count,
            "fallback_count": self.fallback_count,
            "last_action": self.last_action,
            "last_result": self.last_result,
            "last_tool_call": self.last_tool_call,
            "last_error": self.last_error,
            "history_length": len(self.history),
            "updated_at": self.updated_at,
        }

    def to_dict(self):
        d = self.snapshot()
        d.pop("history_length")
        d["history"] = self.history
        return d

    # ---- persistence ----

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
        except Exception as e:
            # not worth crashing the app over, just log it
            logger.error("STATE could not save to %s: %s", self.path, e)

    @classmethod
    def load(cls, path=STATE_FILE, history_limit=HISTORY_LIMIT):
        state = cls(path=path, history_limit=history_limit)
        if not os.path.exists(path):
            logger.info("STATE no saved state at %s, starting fresh", path)
            return state
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key in ["run_count", "error_count", "fallback_count", "last_action",
                        "last_result", "last_tool_call", "last_error", "history", "updated_at"]:
                if key in data:
                    setattr(state, key, data[key])
            logger.info("STATE loaded from %s (run_count=%s)", path, state.run_count)
        except Exception as e:
            logger.warning("STATE file %s is corrupt (%s), starting fresh", path, e)
            state = cls(path=path, history_limit=history_limit)
        return state
