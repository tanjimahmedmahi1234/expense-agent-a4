"""
COIT12204 Assessment 4, Tanjim Ahmed Mahi (12298057)

Runs a fixed set of cases against the real model so I can compare prompt versions.

    python -m scripts.evaluate --versions v1 v2 v3 --repeats 2

Each case is run `repeats` times per version. For every run I record whether the
answer parsed, whether it was right, which tools were used, whether the model
wrapped the JSON in fences, whether a retry was needed, the confidence and how long
it took. Results go to evaluation/results_<version>_<time>.json and a table is
appended to evaluation/summary.md.

Every run gets a fresh copy of the sample data, because categorising an expense
writes the category back and that would make the next run too easy.
"""
import argparse
import json
import os
import tempfile
import time
from datetime import datetime

from app.agent import ExpenseAgent
from app.state import AgentState
from app.store import ExpenseStore

# ---- the cases. "expect" is what a correct answer looks like ----
CASES = [
    {"id": "C1", "action": "categorize_expense", "args": {"expense_id": 4}, "expect": {"category": ["entertainment"], "tool": "lookup_expense"}, "note": "Netflix, saved"},
    {"id": "C2", "action": "categorize_expense", "args": {"expense_id": 5}, "expect": {"category": ["transport"], "tool": "lookup_expense"}, "note": "Uber to airport, saved"},
    {"id": "C3", "action": "categorize_expense", "args": {"expense_id": 6}, "expect": {"category": ["health"], "tool": "lookup_expense"}, "note": "Chemist Warehouse vitamins, saved"},
    {"id": "C4", "action": "categorize_expense", "args": {"expense_id": 9}, "expect": {"category": ["shopping"], "tool": "lookup_expense"}, "note": "JB Hi-Fi headphones, saved"},
    {"id": "C5", "action": "categorize_expense", "args": {"expense_id": 10}, "expect": {"category": ["education"], "tool": "lookup_expense"}, "note": "Textbook, saved"},
    {"id": "C6", "action": "categorize_expense", "args": {"description": "Spotify premium", "amount": 12.99, "merchant": "Spotify"}, "expect": {"category": ["entertainment"], "tool": None}, "note": "free text, clear"},
    {"id": "C7", "action": "categorize_expense", "args": {"description": "Bunnings paint and brushes", "amount": 85.0, "merchant": "Bunnings"}, "expect": {"category": ["shopping", "other"], "tool": None}, "note": "free text, ambiguous"},
    {"id": "C8", "action": "check_unusual_expense", "args": {"expense_id": 14}, "expect": {"is_unusual": True, "tool": "category_totals"}, "note": "$410 groceries vs ~$48 average"},
    {"id": "C9", "action": "check_unusual_expense", "args": {"expense_id": 8}, "expect": {"is_unusual": False, "tool": "category_totals"}, "note": "$5.50 coffee, normal"},
    {"id": "C10", "action": "monthly_summary", "args": {"days": 30}, "expect": {"summary_contains": "rent", "tool": "category_totals"}, "note": "summary should name rent as biggest"},
    {"id": "C11", "action": "categorize_expense", "args": {"expense_id": 999}, "expect": {"category": [None], "max_confidence": 0.2, "tool": "lookup_expense"}, "note": "expense does not exist"},
]


def is_correct(case, result):
    exp = case["expect"]
    if "category" in exp and result.category not in exp["category"]:
        return False
    if "is_unusual" in exp and result.is_unusual != exp["is_unusual"]:
        return False
    if "summary_contains" in exp and exp["summary_contains"] not in (result.summary or "").lower():
        return False
    if "max_confidence" in exp and result.confidence > exp["max_confidence"]:
        return False
    return True


def run_one(case, version, tmpdir):
    # fresh data and state every time
    store_path = os.path.join(tmpdir, f"expenses_{version}_{case['id']}_{time.time_ns()}.json")
    state_path = os.path.join(tmpdir, f"state_{version}_{case['id']}_{time.time_ns()}.json")
    store = ExpenseStore(path=store_path)
    state = AgentState(path=state_path)
    agent = ExpenseAgent(store, state, prompt_version=version)

    # count how many times the executor was called (2 means a retry happened)
    calls = {"n": 0}
    original_invoke = agent._invoke

    def counting_invoke(text):
        calls["n"] += 1
        return original_invoke(text)

    agent._invoke = counting_invoke

    start = time.time()
    try:
        run = agent.run(case["action"], **case["args"])
        error = run["error"]
    except Exception as e:
        # should not happen, the agent catches things, but just in case
        return {"case": case["id"], "version": version, "parsed": False, "correct": False, "error": f"{type(e).__name__}: {e}",
                "tools_called": [], "expected_tool_used": False, "fenced": False, "retry": False, "warnings": [],
                "executor_calls": calls["n"], "latency_ms": int((time.time() - start) * 1000), "answer": None, "confidence": None,
                "reasoning": None, "raw_output": None}

    result = run["result"]
    if case["action"] == "categorize_expense":
        answer = result.category
    elif case["action"] == "check_unusual_expense":
        answer = result.is_unusual
    else:
        answer = result.summary

    expected_tool = case["expect"].get("tool")
    expected_tool_used = True if expected_tool is None else expected_tool in run["tools_called"]

    return {
        "case": case["id"],
        "version": version,
        "parsed": not run["fallback_used"],
        "correct": is_correct(case, result) and not run["fallback_used"],
        "tools_called": run["tools_called"],
        "expected_tool_used": expected_tool_used,
        "fenced": (run["raw_output"] or "").strip().startswith("```"),
        "retry": calls["n"] > 1,
        "warnings": run["warnings"],
        "error": error,
        "executor_calls": calls["n"],
        "latency_ms": run["latency_ms"],
        "answer": answer,
        "confidence": result.confidence,
        "reasoning": result.reasoning,
        "raw_output": (run["raw_output"] or "")[:600],
    }


def summarise(rows, version, repeats):
    n = len(rows)
    parsed = sum(r["parsed"] for r in rows)
    correct = sum(r["correct"] for r in rows)
    tool_ok = sum(r["expected_tool_used"] for r in rows)
    fenced = sum(r["fenced"] for r in rows)
    retries = sum(r["retry"] for r in rows)
    warned = sum(1 for r in rows if r["warnings"])
    latency = sum(r["latency_ms"] for r in rows) / n if n else 0

    # consistency: did every repeat of a case give the same answer?
    by_case = {}
    for r in rows:
        by_case.setdefault(r["case"], []).append(str(r["answer"])[:80])
    consistent = sum(1 for answers in by_case.values() if len(set(answers)) == 1)

    return {
        "version": version, "runs": n, "repeats": repeats,
        "parsed_rate": parsed / n, "accuracy": correct / n, "tool_use_rate": tool_ok / n,
        "fenced": fenced, "retries": retries, "warning_runs": warned,
        "consistency": consistent / len(by_case) if by_case else 0, "mean_latency_ms": int(latency),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", nargs="+", default=["v1"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--label", default="")  # e.g. "after_fix" so I can tell runs apart
    # added after the first evaluation hit the OpenAI tokens per minute limit
    parser.add_argument("--pause", type=float, default=4.0, help="seconds to wait between runs")
    args = parser.parse_args()

    os.makedirs("evaluation", exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="eval_")
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    summaries = []

    for version in args.versions:
        rows = []
        print(f"\n===== prompt {version} =====")
        for case in CASES:
            for i in range(args.repeats):
                row = run_one(case, version, tmpdir)
                rows.append(row)
                mark = "ok " if row["correct"] else "BAD"
                print(f"{mark} {case['id']:>3} run{i+1} answer={str(row['answer'])[:40]!r:44} conf={row['confidence']} "
                      f"tools={row['tools_called']} fenced={row['fenced']} retry={row['retry']} {row['latency_ms']}ms", flush=True)
                time.sleep(args.pause)

        summary = summarise(rows, version, args.repeats)
        summaries.append(summary)
        name = f"evaluation/results_{version}{'_' + args.label if args.label else ''}_{stamp}.json"
        with open(name, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "cases": CASES, "runs": rows}, f, indent=2, default=str)
        print(f"saved {name}")

    # a markdown table I can paste into the report
    lines = [f"\n## Evaluation {stamp} {args.label}\n",
             "| Prompt | Runs | Parsed | Accuracy | Expected tool used | Fenced JSON | Retries | Warnings | Consistency | Mean latency |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        lines.append(f"| {s['version']} | {s['runs']} | {s['parsed_rate']:.0%} | {s['accuracy']:.0%} | {s['tool_use_rate']:.0%} | "
                     f"{s['fenced']} | {s['retries']} | {s['warning_runs']} | {s['consistency']:.0%} | {s['mean_latency_ms']} ms |")
    table = "\n".join(lines)
    with open("evaluation/summary.md", "a", encoding="utf-8") as f:
        f.write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
