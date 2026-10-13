"""
COIT12204 Assessment 4, Tanjim Ahmed Mahi (12298057)

Quick way to try the agent from the terminal without the API.
    python run_demo.py
"""
from app.store import ExpenseStore
from app.agent import ExpenseAgent


def show(run):
    r = run["result"]
    print("-" * 60)
    print("action:      ", r.action)
    print("expense_id:  ", r.expense_id)
    print("category:    ", r.category)
    print("is_unusual:  ", r.is_unusual)
    print("summary:     ", r.summary)
    print("reasoning:   ", r.reasoning)
    print("confidence:  ", r.confidence)
    print("fallback:    ", run["fallback_used"], "| tools:", run["tools_called"], "| warnings:", run["warnings"])
    print("-" * 60)


if __name__ == "__main__":
    store = ExpenseStore()
    agent = ExpenseAgent(store)

    # 1. categorise a saved expense (the model should call lookup_expense)
    show(agent.run("categorize_expense", expense_id=5))

    # 2. categorise free text that is not in the store
    show(agent.run("categorize_expense", description="Spotify premium", amount=12.99, merchant="Spotify"))

    # 3. is the $410 Coles shop unusual?
    show(agent.run("check_unusual_expense", expense_id=14))

    # 4. summary of the last 30 days
    show(agent.run("monthly_summary", days=30))

    # 5. an expense id that does not exist, to see how the agent handles a tool error
    show(agent.run("categorize_expense", expense_id=999))

    # what the state looks like after all of that
    print("\nSTATE after the runs:")
    for key, value in agent.state.snapshot().items():
        print(f"  {key:16} {value}")
    print(f"\n(state is saved in {agent.state.path})")
