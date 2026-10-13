"""
COIT12204 Assessment 4, Tanjim Ahmed Mahi (12298057)

FastAPI app that wraps the agent.

Run with:  python -m uvicorn app.main:app --reload
Docs at:   http://127.0.0.1:8000/docs

I use a create_app() function instead of a module level app so the tests can pass in
a fake agent. The normal `app` at the bottom is what uvicorn uses.
"""
import os
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse

from app.agent import ExpenseAgent, InputRejected, clean_text
from app.classifier import ClassifierAgent
from app.config import MODEL_NAME, PROMPT_VERSION, LOG_FILE
from app.orchestrator import Orchestrator
from app.reviewer import ReviewerAgent
from app.logger import logger
from app.models import (
    CategorizeRequest,
    CheckUnusualRequest,
    SummaryRequest,
    ExpenseCreate,
    TeamRequest,
)
from app.state import AgentState
from app.store import ExpenseStore

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# the web page polls these every few seconds, logging them would flood the log
QUIET_PATHS = ["/", "/health", "/api/agent/state", "/api/agent/history", "/api/agent/logs", "/api/expenses",
               "/api/team/state", "/api/team/conversations"]


def format_response(request_id, run):
    """Every agent endpoint returns the same shape so the frontend only has to learn one."""
    return {
        "request_id": request_id,
        "action": run["result"].action,
        "result": run["result"].model_dump(mode="json"),
        "fallback_used": run["fallback_used"],
        "tools_called": run["tools_called"],
        "warnings": run["warnings"],
        "prompt_version": run["prompt_version"],
        "latency_ms": run["latency_ms"],
        "state": run["state"],
    }


def create_app(store=None, state=None, agent=None, orchestrator=None):
    app = FastAPI(title="Expense Categoriser Agent", version="0.2")

    # build the real things if the tests didn't give us fakes
    if store is None:
        store = ExpenseStore()
    if state is None:
        state = AgentState.load()
    if agent is None:
        agent = ExpenseAgent(store, state)
    if orchestrator is None:
        # A4: the A3 agent becomes the classifier, plus a new reviewer
        orchestrator = Orchestrator(ClassifierAgent(agent), ReviewerAgent(), store)

    # keep them on the app so tests (and the routes) can reach them
    app.state.store = store
    app.state.agent_state = state
    app.state.agent = agent
    app.state.orchestrator = orchestrator

    # ---------- middleware: request id + one log line in, one out ----------

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = uuid.uuid4().hex[:8]
        request.state.request_id = request_id
        start = time.time()
        quiet = request.method == "GET" and request.url.path in QUIET_PATHS
        if not quiet:
            logger.info("REQUEST %s %s %s", request_id, request.method, request.url.path)
        try:
            response = await call_next(request)
        except Exception as e:
            # a last resort so the client always gets JSON with the request id
            logger.error("UNHANDLED %s %s: %s", request_id, type(e).__name__, e)
            response = JSONResponse(
                status_code=500,
                content={"error": "internal_error", "detail": type(e).__name__, "request_id": request_id},
            )
        ms = int((time.time() - start) * 1000)
        if not quiet:
            logger.info("RESPONSE %s status=%s %dms", request_id, response.status_code, ms)
        response.headers["X-Request-ID"] = request_id
        return response

    # ---------- error handlers, all return the same {error, detail, request_id} shape ----------

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        messages = []
        for err in exc.errors():
            field = ".".join(str(x) for x in err["loc"] if x != "body")
            messages.append(f"{field}: {err['msg']}")
        logger.warning("VALIDATION %s %s", request.state.request_id, messages)
        return JSONResponse(
            status_code=422,
            content={"error": "validation_error", "detail": messages, "request_id": request.state.request_id},
        )

    @app.exception_handler(InputRejected)
    async def input_rejected(request: Request, exc: InputRejected):
        return JSONResponse(
            status_code=400,
            content={"error": "input_rejected", "detail": str(exc), "request_id": request.state.request_id},
        )

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": "http_error", "detail": exc.detail, "request_id": request.state.request_id},
        )

    @app.exception_handler(Exception)
    async def any_error(request: Request, exc: Exception):
        logger.error("UNHANDLED %s %s: %s", request.state.request_id, type(exc).__name__, exc)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "detail": type(exc).__name__, "request_id": request.state.request_id},
        )

    # ---------- agent endpoints ----------

    @app.post("/api/agent/categorize_expense")
    def categorize_expense(body: CategorizeRequest, request: Request):
        # pydantic checks the types, but I still need one of the two ways of describing the expense
        if body.expense_id is None and not body.description:
            raise HTTPException(status_code=422, detail="Give either expense_id or description.")
        if body.expense_id is not None and store.get(body.expense_id) is None:
            # check before calling the model, no point paying for a call that will fail
            raise HTTPException(status_code=404, detail=f"Expense {body.expense_id} not found.")

        run = agent.run(
            "categorize_expense",
            expense_id=body.expense_id,
            description=body.description,
            amount=body.amount,
            merchant=body.merchant,
        )
        return format_response(request.state.request_id, run)

    @app.post("/api/agent/check_unusual_expense")
    def check_unusual_expense(body: CheckUnusualRequest, request: Request):
        if store.get(body.expense_id) is None:
            raise HTTPException(status_code=404, detail=f"Expense {body.expense_id} not found.")
        run = agent.run("check_unusual_expense", expense_id=body.expense_id)
        return format_response(request.state.request_id, run)

    @app.post("/api/agent/monthly_summary")
    def monthly_summary(body: SummaryRequest, request: Request):
        run = agent.run("monthly_summary", days=body.days)
        return format_response(request.state.request_id, run)

    # ---------- A4: two agent team ----------

    @app.post("/api/team/run")
    def team_run(body: TeamRequest, request: Request):
        """Classifier + reviewer, routed by the orchestrator. Returns the whole conversation."""
        if body.action == "categorize_expense" and body.expense_id is None and not body.description:
            raise HTTPException(status_code=422, detail="Give either expense_id or description.")
        if body.action == "check_unusual_expense" and body.expense_id is None:
            raise HTTPException(status_code=422, detail="check_unusual_expense needs an expense_id.")
        if body.expense_id is not None and store.get(body.expense_id) is None:
            raise HTTPException(status_code=404, detail=f"Expense {body.expense_id} not found.")
        # check the text here, before any agent sees it
        description = clean_text(body.description) if body.description else None
        merchant = clean_text(body.merchant) if body.merchant else None

        conversation = orchestrator.run(
            body.action, expense_id=body.expense_id, description=description, amount=body.amount,
            merchant=merchant, days=body.days, simulate=body.simulate,
        )
        conversation["request_id"] = request.state.request_id
        conversation["team_state"] = orchestrator.state.snapshot()
        return conversation

    @app.get("/api/team/conversations")
    def team_conversations(limit: int = 10):
        limit = max(1, min(limit, 50))
        out = []
        for c in list(orchestrator.recent)[-limit:][::-1]:
            out.append({k: c[k] for k in ["conversation_id", "time", "action", "status", "stop_reason",
                                           "rounds", "latency_ms", "simulate"]} | {"messages": len(c["messages"])})
        return out

    @app.get("/api/team/conversations/{conversation_id}")
    def team_conversation(conversation_id: str):
        c = orchestrator.get_conversation(conversation_id)
        if c is None:
            raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found.")
        return c

    @app.get("/api/team/state")
    def team_state():
        return orchestrator.state.snapshot()

    @app.post("/api/team/state/reset")
    def team_state_reset():
        orchestrator.state.reset()
        return orchestrator.state.snapshot()

    # ---------- state endpoints ----------

    @app.get("/api/agent/state")
    def get_state():
        return state.snapshot()

    @app.get("/api/agent/history")
    def get_history(limit: int = 20):
        if limit < 1:
            limit = 1
        if limit > 100:
            limit = 100
        # newest first
        return list(reversed(state.history[-limit:]))

    @app.post("/api/agent/state/reset")
    def reset_state():
        state.reset()
        return state.snapshot()

    @app.get("/api/agent/logs")
    def get_logs(lines: int = 40):
        """The last N lines of the log file, for the web page."""
        if lines < 1:
            lines = 1
        if lines > 500:
            lines = 500
        if not os.path.exists(LOG_FILE):
            return []
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        return [l.rstrip("\n") for l in all_lines[-lines:]]

    # ---------- expenses ----------

    @app.get("/api/expenses")
    def list_expenses():
        return [e.model_dump(mode="json") for e in store.all()]

    @app.get("/api/expenses/{expense_id}")
    def get_expense(expense_id: int):
        exp = store.get(expense_id)
        if exp is None:
            raise HTTPException(status_code=404, detail=f"Expense {expense_id} not found.")
        return exp.model_dump(mode="json")

    @app.post("/api/expenses", status_code=201)
    def add_expense(body: ExpenseCreate):
        exp = store.add(body)
        return exp.model_dump(mode="json")

    # ---------- misc ----------

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": MODEL_NAME,
            "prompt_version": PROMPT_VERSION,
            "llm_ready": agent.executor is not None,
            "reviewer_llm_ready": orchestrator.agents["reviewer"].chain is not None,
            "max_rounds": orchestrator.max_rounds,
            "max_messages": orchestrator.max_messages,
            "agent_timeout": orchestrator.timeout,
            "runs": state.run_count,
        }

    @app.get("/")
    def root():
        # the web page. If it is missing for some reason, fall back to a plain message
        page = os.path.join(STATIC_DIR, "index.html")
        if os.path.exists(page):
            return FileResponse(page)
        return {"message": "Expense agent is running. See /docs for the endpoints."}

    return app


# this is what uvicorn imports
app = create_app()
