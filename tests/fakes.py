"""
COIT12204 Assessment 4, Tanjim Ahmed Mahi (12298057)

A fake chat model so the tests never call the real API.

I give it a list of AIMessages and it hands them back one per call. That way I can
script exactly what "the model" says: a tool call, then JSON, or some rubbish text
to see what the agent does with it.
"""
import json
from typing import Any, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeChatModel(BaseChatModel):
    responses: List[AIMessage] = []
    calls: List[Any] = []  # every message list the agent sent, so tests can look at the prompt
    raise_error: bool = False

    @property
    def _llm_type(self):
        return "fake"

    def bind_tools(self, tools, **kwargs):
        # the real model returns a new runnable, for the fake it is fine to return itself
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(messages)
        if self.raise_error:
            raise RuntimeError("fake API failure")
        if not self.responses:
            raise RuntimeError("fake model ran out of scripted responses")
        msg = self.responses.pop(0)
        return ChatResult(generations=[ChatGeneration(message=msg)])


# little helpers so the tests read nicely

def tool_request(name, call_id, **args):
    """An AIMessage where the model asks to call a tool."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def json_answer(**fields):
    """An AIMessage with a clean JSON answer."""
    return AIMessage(content=json.dumps(fields))


def text_answer(text):
    """An AIMessage with whatever text I want."""
    return AIMessage(content=text)
