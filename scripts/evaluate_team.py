"""
COIT12204 Assessment 4, Tanjim Ahmed Mahi (12298057)

A4 evaluation: is the classifier + reviewer team better than the classifier alone?

    python -m scripts.evaluate_team --repeats 2 --pause 3

For every case it runs:
  single  the A3 agent on its own (classifier only)
  team    the orchestrator with classifier + reviewer

Then it runs each simulated failure once, so the resilience paths are tested with
the real model too.

I picked cases where a keyword rule or a quick guess gets it wrong (Uber Eats,
Bunnings sausage sizzle, Kindle textbook from Amazon) because that's where a second
agent should help. Results go to evaluation/team_results_<time>.json, every team
conversation to evaluation/team_conversations_<time>.jsonl, and a table is added
to evaluation/team_summary.md.
"""
import argparse
import json
import os
import re
import tempfile
import time
from collections import Counter
from datetime import datetime

from app.agent import ExpenseAgent
from app.classifier import ClassifierAgent
from app.orchestrator import Orchestrator, TeamState
from app.reviewer import ReviewerAgent
from app.state import AgentState
from app.store import ExpenseStore

ANY = "any"

CASES = [
    {"id": "T1", "action": "categorize_expense", "args": {"expense_id": 4}, "expect": ["entertainment"], "note": "Netflix"},
    {"id": "T2", "action": "categorize_expense", "args": {"expense_id": 5}, "expect": ["transport"], "note": "Uber to airport"},
    {"id": "T3", "action": "categorize_expense", "args": {"expense_id": 9}, "expect": ["shopping"], "note": "JB Hi-Fi headphones"},
    {"id": "T4", "action": "categorize_expense", "args": {"expense_id": 10}, "expect": ["education"], "note": "textbook"},
    {"id": "T5", "action": "categorize_expense", "args": {"description": "Uber Eats dinner", "amount": 32.0, "merchant": "Uber Eats"},
     "expect": ["food"], "note": "trap: Uber is transport, Uber Eats is food"},
    {"id": "T6", "action": "categorize_expense", "args": {"description": "Bunnings sausage sizzle", "amount": 5.0, "merchant": "Bunnings"},
     "expect": ["food"], "note": "trap: Bunnings is a hardware shop"},
    {"id": "T7", "action": "categorize_expense", "args": {"description": "Kindle textbook for COIT12204", "amount": 59.0, "merchant": "Amazon"},
     "expect": ["education"], "note": "trap: Amazon looks like shopping"},
    {"id": "T8", "action": "categorize_expense", "args": {"description": "Officeworks printer ink for assignments", "amount": 45.0, "merchant": "Officeworks"},
     "expect": ["education", "shopping"], "note": "genuinely ambiguous"},
    {"id": "T9", "action": "categorize_expense", "args": {"description": "Optus phone plan", "amount": 45.0, "merchant": "Optus"},
     "expect": ["utilities"], "note": "phone plan"},
    {"id": "T10", "action": "categorize_expense", "args": {"description": "Shell petrol", "amount": 70.0, "merchant": "Shell"},
     "expect": ["transport"], "note": "fuel"},
    {"id": "T11", "action": "check_unusual_expense", "args": {"expense_id": 14}, "expect": [True], "note": "$410 groceries"},
    {"id": "T12", "action": "check_unusual_expense", "args": {"expense_id": 8}, "expect": [False], "note": "$5.50 coffee"},
    {"id": "T13", "action": "check_unusual_expense", "args": {"expense_id": 11}, "expect": [True, False],
     "note": "borderline: $95 dinner, depends if it's in its own average"},
    {"id": "T14", "action": "monthly_summary", "args": {"days": 30}, "expect": ["summary"], "note": "total and rent"},
]

FAULTS = ["force_revise", "reviewer_timeout", "invalid_receiver", "bad_message"]


def answer_of(action, p):
    if p is None:
        return None
    if action == "categorize_expense":
        return p.get("category")
    if action == "check_unusual_expense":
        return p.get("is_unusual")
    return p.get("summary")


def is_correct(case, answer, store):
    if case["expect"] == ["summary"]:
        total = sum(e.amount for e in store.recent(case["args"]["days"]))
        nums = [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*\.?\d*", answer or "")]
        return any(abs(n - total) < 1.0 for n in nums) and "rent" in (answer or "").lower()
    return answer in case["expect"]


def fresh(tmp, tag):
    """New data for every run, because an approved category is saved and would help the next run."""
    key = f"{tag}_{time.time_ns()}"
    store = ExpenseStore(path=os.path.join(tmp, f"exp_{key}.json"))
    agent = ExpenseAgent(store, AgentState(path=os.path.join(tmp, f"st_{key}.json")))
    return store, agent


def run_single(case, tmp):
    store, agent = fresh(tmp, "single")
    start = time.time()
    run = agent.run(case["action"], save=False, **case["args"])
    p = run["result"].model_dump(mode="json")
    ans = answer_of(case["action"], p)
    return {
        "case": case["id"], "mode": "single", "answer": ans, "first_answer": ans,
        "correct": is_correct(case, ans, store), "status": "fallback" if run["fallback_used"] else "answered",
        "rounds": 1, "messages": 0, "confidence": p["confidence"], "issues": [],
        "latency_ms": int((time.time() - start) * 1000),
    }


def run_team(case, tmp, conv_file, simulate=None):
    store, agent = fresh(tmp, "team")
    orch = Orchestrator(ClassifierAgent(agent), ReviewerAgent(), store,
                        state=TeamState(os.path.join(tmp, f"team_{time.time_ns()}.json")),
                        conversations_file=conv_file)
    conv = orch.run(case["action"], simulate=simulate, **case["args"])
    first = None
    issues = []
    for m in conv["messages"]:
        if m["intent"] == "review_request" and m["valid"] and first is None:
            first = answer_of(case["action"], m["content"].get("proposal"))
        if m["intent"] in ("revise_request", "approved"):
            issues += m["content"]["review"].get("issues", [])
    ans = answer_of(case["action"], conv["final"])
    return {
        "case": case["id"], "mode": "team", "simulate": simulate, "answer": ans, "first_answer": first,
        "correct": is_correct(case, ans, store), "first_correct": is_correct(case, first, store) if first is not None else None,
        "status": conv["status"], "stop_reason": conv["stop_reason"], "rounds": conv["rounds"],
        "messages": len(conv["messages"]), "confidence": conv["final"]["confidence"],
        "issues": issues[:6], "latency_ms": conv["latency_ms"], "conversation_id": conv["conversation_id"],
    }


def summarise(rows, mode):
    rows = [r for r in rows if r["mode"] == mode and not r.get("simulate")]
    n = len(rows) or 1
    by_case = {}
    for r in rows:
        by_case.setdefault(r["case"], []).append(str(r["answer"])[:60])
    s = {
        "mode": mode, "runs": len(rows),
        "accuracy": sum(r["correct"] for r in rows) / n,
        "consistency": sum(1 for a in by_case.values() if len(set(a)) == 1) / (len(by_case) or 1),
        "mean_latency_ms": int(sum(r["latency_ms"] for r in rows) / n),
        "mean_rounds": round(sum(r["rounds"] for r in rows) / n, 2),
        "mean_messages": round(sum(r["messages"] for r in rows) / n, 2),
        "statuses": dict(Counter(r["status"] for r in rows)),
    }
    if mode == "team":
        s["reviewer_fixed"] = sum(1 for r in rows if r["first_correct"] is False and r["correct"])
        s["reviewer_broke"] = sum(1 for r in rows if r["first_correct"] is True and not r["correct"])
        s["reviewer_missed"] = sum(1 for r in rows if r["status"].startswith("approved") and not r["correct"])
        s["needless_revisions"] = sum(1 for r in rows if r["first_correct"] is True and r["rounds"] > 1)
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--pause", type=float, default=3.0, help="seconds between runs, for the rate limit")
    parser.add_argument("--label", default="")
    parser.add_argument("--cases", nargs="*", help="only these case ids, e.g. T5 T6")
    parser.add_argument("--no-single", action="store_true", help="skip the classifier-only baseline")
    parser.add_argument("--no-faults", action="store_true", help="skip the simulated failures")
    args = parser.parse_args()

    os.makedirs("evaluation", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M") + (f"_{args.label}" if args.label else "")
    conv_file = os.path.join("evaluation", f"team_conversations_{stamp}.jsonl")
    tmp = tempfile.mkdtemp(prefix="team_eval_")
    cases = [c for c in CASES if not args.cases or c["id"] in args.cases]
    rows = []

    modes = ["team"] if args.no_single else ["single", "team"]
    for case in cases:
        for mode in modes:
            for i in range(args.repeats):
                row = run_single(case, tmp) if mode == "single" else run_team(case, tmp, conv_file)
                rows.append(row)
                mark = "ok " if row["correct"] else "BAD"
                extra = f"status={row['status']} rounds={row['rounds']} first={str(row['first_answer'])[:20]!r}" if mode == "team" else ""
                print(f"{mark} {case['id']:>3} {mode:6} run{i + 1} answer={str(row['answer'])[:30]!r:34} "
                      f"conf={row['confidence']} {extra} {row['latency_ms']}ms", flush=True)
                time.sleep(args.pause)

    if not args.no_faults:
        print("\n===== simulated failures (real model) =====")
        base = next(c for c in CASES if c["id"] == "T2")
        for fault in FAULTS:
            row = run_team(base, tmp, conv_file, simulate=fault)
            rows.append(row)
            print(f"    T2 {fault:18} status={row['status']} stop={row['stop_reason']} answer={row['answer']!r} "
                  f"conf={row['confidence']} {row['latency_ms']}ms", flush=True)
            time.sleep(args.pause)

    summaries = [summarise(rows, m) for m in modes]
    out = os.path.join("evaluation", f"team_results_{stamp}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summaries": summaries, "cases": cases, "runs": rows}, f, indent=2, default=str)

    lines = [f"\n## Team evaluation {stamp}\n",
             "| Mode | Runs | Accuracy | Consistency | Mean rounds | Mean messages | Mean latency | Statuses |",
             "|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        lines.append(f"| {s['mode']} | {s['runs']} | {s['accuracy']:.0%} | {s['consistency']:.0%} | {s['mean_rounds']} | "
                     f"{s['mean_messages']} | {s['mean_latency_ms']} ms | {s['statuses']} |")
    team = next((s for s in summaries if s["mode"] == "team"), None)
    if team:
        lines.append(f"\nReviewer fixed a wrong first answer: {team['reviewer_fixed']} · broke a right one: {team['reviewer_broke']} · "
                     f"approved a wrong one: {team['reviewer_missed']} · sent back a right one: {team['needless_revisions']}")
    table = "\n".join(lines)
    with open(os.path.join("evaluation", "team_summary.md"), "a", encoding="utf-8") as f:
        f.write(table + "\n")
    print(table)
    print(f"\nsaved {out}\nsaved {conv_file}")


if __name__ == "__main__":
    main()
