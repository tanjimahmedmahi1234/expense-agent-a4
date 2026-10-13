"""
COIT12204 Assessment 4, Tanjim Ahmed Mahi (12298057)

Settings for the expense agent.
I read everything from the .env file so the API key never ends up in the code.
"""
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v3")

# where the data lives
EXPENSES_FILE = os.getenv("EXPENSES_FILE", "data/expenses.json")
STATE_FILE = os.getenv("STATE_FILE", "data/agent_state.json")
LOG_FILE = os.getenv("LOG_FILE", "logs/agent.log")

# limits so the agent can't loop forever or get a huge input
MAX_STEPS = 4
MAX_INPUT_CHARS = 300
LLM_TIMEOUT = 30

# ---- A4: two agent settings ----
# how many times the classifier can try before the orchestrator stops the loop
MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", "3"))
# hard cap on messages in one conversation, in case two agents keep bouncing
MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", "8"))
# seconds one agent gets to answer one message
AGENT_TIMEOUT = float(os.getenv("AGENT_TIMEOUT", "45"))
TEAM_STATE_FILE = os.getenv("TEAM_STATE_FILE", "data/team_state.json")
CONVERSATIONS_FILE = os.getenv("CONVERSATIONS_FILE", "data/conversations.jsonl")

# the categories the agent is allowed to pick from
CATEGORIES = [
    "food",
    "transport",
    "rent",
    "utilities",
    "entertainment",
    "shopping",
    "health",
    "education",
    "other",
]
