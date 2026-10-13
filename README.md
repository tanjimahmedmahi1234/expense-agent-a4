# Expense Agent (COIT12204 Assessments 3 and 4)

Tanjim Ahmed Mahi, Student ID 12298057

A small personal finance app built around LangChain agents.

- **Assessment 4:** a second agent. The A3 agent is now the **Classifier**, a new **Reviewer** checks every answer against the data, and an **Orchestrator** routes the messages between them, validates every message, stops loops, handles timeouts and falls back when something breaks.

Model: `gpt-4o` through `langchain-openai`, LangChain 0.3.

## Setup

You need Python 3.10 or newer and an OpenAI API key.

**Windows:** copy `.env.example` to `.env` and put the key in it. Then double click `run_tests.bat`. The first time it creates `.venv-win` and installs everything.

**Mac:**

```bash
bash setup_mac.sh        # makes .venv, installs, creates .env
# open .env and paste the key in
```

Or by hand on any system:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
cp .env.example .env             # then put your key in .env
```

The `.env` file is in `.gitignore`, so the key never goes into the repository.

## Run it

| What | Windows | Mac | Command |
|---|---|---|---|
| Tests (126, no network) | `run_tests.bat` | `./run_tests.sh` | `python -m pytest` |
| Fresh sample data | `reset_data.bat` | `./reset_data.sh` | `python -m scripts.seed_expenses` |
| Web page and API | `run_server.bat` | `./run_server.sh` | `python -m uvicorn app.main:app --reload` |
| Team evaluation (real model) | `run_team_eval.bat` | `./run_team_eval.sh` | `python -m scripts.evaluate_team --repeats 2 --pause 3` |
| A3 prompt evaluation | `run_eval.bat` | | `python -m scripts.evaluate --versions v1 v2 v3 --repeats 2 --pause 5` |

The web page is at http://127.0.0.1:8000 and the API docs at http://127.0.0.1:8000/docs.

## How the two agents work

```
you -> orchestrator -> classifier -> reviewer -> (approved) -> orchestrator -> you
                                       |
                                       +-> (revise) -> classifier -> reviewer -> ...
```

1. The orchestrator sends a `classify_request` to the classifier.
2. The classifier runs the A3 agent and sends a `review_request` with its proposal to the reviewer.
3. The reviewer looks up its own evidence with the same tools and runs rule checks. A HARD check is a fact from the data and always wins. It then asks the model for a verdict.
4. If it approves, an `approved` message goes to the orchestrator. If not, a `revise_request` with feedback goes back to the classifier.
5. The orchestrator validates and logs every message before routing it. It stops after 3 rounds or 8 messages, gives each agent 45 seconds, and decides the final answer.

Every message has the same envelope:

```json
{
  "message_id": "1a5d8998",
  "conversation_id": "e79085",
  "step": 2,
  "sender": "classifier",
  "receiver": "reviewer",
  "intent": "review_request",
  "content": {"request": {}, "proposal": {}, "round": 1},
  "constraints": {"format": "json"},
  "reply_to": "f2d93ff3",
  "timestamp": "2026-10-09T11:51:43"
}
```

### When something goes wrong

| Problem | What happens | Final status |
|---|---|---|
| Reviewer never agrees | stops after 3 rounds, uses the reviewer's suggestion | `max_rounds` |
| Reviewer too slow | timeout, classifier's answer returned with confidence capped at 0.5 | `unreviewed` |
| Message sent to an agent that doesn't exist | `ROUTE ERROR` in the log | `unreviewed` |
| Broken message (missing field) | `ROUTE ERROR` in the log, A3 keyword rules answer | `fallback` |
| Reviewer model returns junk | one retry, then rule checks only | normal |
| Classifier model down | A3 fallback rules, still reviewed | normal |

The web page has a "simulate failure" menu for the first four, so they can be shown on purpose.

## Results

Real gpt-4o runs on 9 October 2026, 14 cases, 2 repeats each. Full details in `evaluation/`.

| | Before the fixes | After the fixes |
|---|---|---|
| Team accuracy | 24 / 28 (86%) | 28 / 28 (100%) |
| Reviewer broke a right answer | 4 | 0 |
| Reviewer approved a wrong answer | 1 | 0 |
| Right answer sent back | 6 | 2 |
| Runs that hit max rounds | 4 | 0 |
| Mean rounds | 1.36 | 1.07 |
| Mean latency | 4070 ms | 3248 ms |

The single agent alone got 28 / 28 in the first run. The two fixes were better evidence for the reviewer model (no "other" keyword hint, PASS checks, a clearer 30 day rule) and a limit of one round for a soft disagreement.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| POST | `/api/team/run` | two agents. Body `{"action": "categorize_expense", "expense_id": 5}`, optional `"simulate"` |
| GET | `/api/team/conversations` | recent conversations |
| GET | `/api/team/conversations/{id}` | one conversation with every message |
| GET / POST | `/api/team/state`, `/api/team/state/reset` | orchestrator counters |
| POST | `/api/agent/categorize_expense`, `/check_unusual_expense`, `/monthly_summary` | A3 single agent |
| GET | `/api/agent/state`, `/api/agent/history`, `/api/agent/logs` | A3 state and the log |
| GET / POST | `/api/expenses`, `/api/expenses/{id}` | the data |
| GET | `/health` | model, prompt version, limits |

## Project layout

```
app/
  config.py        settings from .env and the limits
  logger.py        one logger, terminal + logs/agent.log
  models.py        Expense, request bodies, AgentResult
  store.py         JSON file store with sample data
  tools.py         lookup_expense, list_recent_expenses, category_totals
  prompts.py       classifier prompts v1 v2 v3
  state.py         AgentState (A3)
  agent.py         the A3 agent (now the classifier's brain)
  messages.py      A4 message envelope, content models, validate_message()
  base_agent.py    A4 handle() with RECEIVED / SENT logging and cancel flag
  classifier.py    A4 classifier agent
  reviewer.py      A4 reviewer agent
  orchestrator.py  A4 routing, limits, timeouts, fallbacks, TeamState
  main.py          FastAPI app
  static/index.html  the web page
scripts/           evaluate.py (A3), evaluate_team.py (A4), seed_expenses.py
tests/             126 tests with fake models
evaluation/        real evaluation results
docs/              reports
CONTEXT.md         everything needed to rebuild the project
```

## AI use declaration

_Tanjim: write your declaration here before submitting, in your own words. Say which AI tools you used and what for. The A4 brief requires it._
