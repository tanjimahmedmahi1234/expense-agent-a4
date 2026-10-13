"""
COIT12204 Assessment 4, Tanjim Ahmed Mahi (12298057)

The Classifier agent. This is my A3 ExpenseAgent with a message interface around it.

It gets either a classify_request (first try) or a revise_request (the reviewer sent
the last answer back), runs the A3 agent, and always replies to the reviewer with a
review_request that carries its proposal.
"""
from app.base_agent import BaseAgent
from app.messages import ClassifyContent, Proposal, ReviewContent, ReviseContent, make_message


class ClassifierAgent(BaseAgent):
    name = "classifier"

    def __init__(self, expense_agent):
        self.agent = expense_agent

    def process(self, msg):
        if msg.intent == "classify_request":
            content = ClassifyContent(**msg.content)
            feedback = None
            round_no = 1
        elif msg.intent == "revise_request":
            content = ReviseContent(**msg.content)
            feedback = self._feedback_text(content)
            round_no = content.round + 1
        else:
            raise ValueError(f"classifier can't handle intent {msg.intent}")

        req = content.request
        run = self.agent.run(
            req.action,
            expense_id=req.expense_id,
            description=req.description,
            amount=req.amount,
            merchant=req.merchant,
            days=req.days,
            feedback=feedback,
            save=False,  # the orchestrator saves only after the reviewer approves
        )
        proposal = Proposal(**run["result"].model_dump(mode="json"))
        body = ReviewContent(
            request=req,
            proposal=proposal,
            round=round_no,
            classifier_used_fallback=run["fallback_used"],
            tools_called=run["tools_called"],
        )
        return make_message("classifier", "reviewer", "review_request", body)

    def _feedback_text(self, content):
        r = content.review
        prev = content.previous_proposal
        lines = [r.feedback or "The reviewer did not agree with your answer."]
        for issue in r.issues[:4]:
            lines.append("- " + issue)
        lines.append(
            f"Your previous answer was: category={prev.category}, is_unusual={prev.is_unusual}, "
            f"confidence={prev.confidence}."
        )
        return "\n".join(lines)

    def fallback(self, request, error):
        """Used by the orchestrator when the classifier never got a usable answer out."""
        expense = self.agent.store.get(request.expense_id) if request.expense_id else None
        result = self.agent._fallback(
            request.action, expense, request.expense_id, request.description,
            request.amount, request.merchant, request.days, error,
        )
        return Proposal(**result.model_dump(mode="json"))
