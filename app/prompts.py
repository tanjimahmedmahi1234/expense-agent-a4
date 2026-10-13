"""
COIT12204 Assessment 4, Tanjim Ahmed Mahi (12298057)

Prompt templates. I keep the system prompts in a dict keyed by version so I can
compare them in the evaluation later (v1 is the first attempt, v2 and v3 get added
after I see how v1 behaves).
"""
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.config import CATEGORIES
from app.models import AgentResult

# the parser also writes the JSON schema instructions into the prompt for me
PARSER = PydanticOutputParser(pydantic_object=AgentResult)


SYSTEM_PROMPTS = {
    # v1: my first attempt, just says what the job is and gives the schema
    "v1": (
        "You are a helpful personal finance assistant. You help the user categorise "
        "their expenses and understand their spending. Today's date is {today}.\n\n"
        "{format_instructions}"
    ),
    # v2: after the first real runs. The model kept wrapping the JSON in ```json fences
    # and sometimes explained things first, so I tell it to only send the JSON. I also
    # tell it to always use the tools instead of guessing.
    "v2": (
        "You are the reasoning part of a personal finance app. You categorise expenses, "
        "spot unusual spending and summarise spending. Today's date is {today}.\n\n"
        "Rules:\n"
        "- If you are given an expense id, call lookup_expense first. Never guess what an expense is.\n"
        "- For a summary, call category_totals before answering.\n"
        "- Reply with ONLY the JSON object. No markdown fences, no text before or after it.\n\n"
        "{format_instructions}"
    ),
    # v3: v2 plus rules about confidence and categories. In the runs the model gave
    # confidence 0.9 even when the tool said the expense did not exist, and it picked
    # its own time window for the unusual check. So I spell those out.
    "v3": (
        "You are the reasoning part of a personal finance app. You categorise expenses, "
        "spot unusual spending and summarise spending. Today's date is {today}.\n\n"
        "Rules:\n"
        "- If you are given an expense id, call lookup_expense first. Never guess what an expense is.\n"
        "- If a tool returns an error, do not invent data. Set the optional fields to null "
        "and set confidence to 0.0.\n"
        "- For a summary, call category_totals before answering.\n"
        "- To check if an expense is unusual, call category_totals with days=30 and compare the "
        "amount with average_by_category for its category. More than double the average is unusual.\n"
        "- Categories: food is groceries, cafes, restaurants and food delivery. transport is public "
        "transport, ride share, fuel and parking. utilities is power, gas, water, internet and phone. "
        "entertainment is streaming, music, games, movies. shopping is electronics, clothes and "
        "general retail. health is chemist, doctor, gym. education is textbooks, courses and fees. "
        "Use other only when nothing fits.\n"
        "- Confidence: 0.9 or above only when the tool data clearly supports the answer. "
        "0.5 to 0.8 when you had to interpret. Below 0.5 when guessing.\n"
        "- Keep reasoning to one sentence.\n"
        "- Reply with ONLY the JSON object. No markdown fences, no text before or after it.\n\n"
        "{format_instructions}"
    ),
}


# what I send as the human message for each action
HUMAN_TEMPLATES = {
    "categorize_expense": (
        "Categorise this expense into exactly one of these categories: "
        + ", ".join(CATEGORIES)
        + ".\n{expense_block}\n"
        "Set action to categorize_expense and fill in expense_id, category, reasoning and confidence."
    ),
    "check_unusual_expense": (
        "Decide if this expense is unusually large compared to the user's normal spending "
        "in the same category.\n{expense_block}\n"
        "Set action to check_unusual_expense and fill in expense_id, category, is_unusual, reasoning and confidence."
    ),
    "monthly_summary": (
        "Give the user a short summary of their spending over the last {days} days. "
        "Mention the total, the biggest category and anything that looks worth attention.\n"
        "Set action to monthly_summary and fill in summary, reasoning and confidence."
    ),
}


def build_prompt(version: str = "v1") -> ChatPromptTemplate:
    if version not in SYSTEM_PROMPTS:
        raise ValueError(f"unknown prompt version {version}")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPTS[version]),
            ("human", "{input}"),
            # create_tool_calling_agent needs this placeholder, it puts the tool calls
            # and tool results in here between rounds
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    # format_instructions has curly braces in it so it has to go in as a partial,
    # otherwise the template thinks they are variables
    prompt = prompt.partial(format_instructions=PARSER.get_format_instructions())
    return prompt
