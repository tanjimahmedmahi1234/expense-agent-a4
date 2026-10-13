"""
COIT12204 Assessment 4, Tanjim Ahmed Mahi (12298057)

Every agent in the team has a name and a handle() method that takes one AgentMessage
and returns one AgentMessage. handle() logs what came in and what went out, then
process() does the real work. The orchestrator only ever calls handle().
"""
from app.logger import logger


class Cancelled(Exception):
    """Raised inside an agent when the orchestrator has already given up on it."""


def check_cancelled(msg):
    # Python can't kill a thread. So on a timeout the orchestrator sets this flag on the
    # message, and the agent checks it before doing more work.
    if msg.constraints.get("cancelled"):
        raise Cancelled()


class BaseAgent:
    name = "base"

    def handle(self, msg):
        logger.info("AGENT %s RECEIVED %s step=%s", self.name, msg.short(), msg.step)
        try:
            check_cancelled(msg)
            reply = self.process(msg)
            check_cancelled(msg)
        except Cancelled:
            logger.warning("AGENT %s stopped, the orchestrator already timed out on %s", self.name, msg.message_id)
            return None
        reply.conversation_id = msg.conversation_id
        reply.reply_to = msg.message_id
        logger.info("AGENT %s SENT %s", self.name, reply.short())
        return reply

    def process(self, msg):
        raise NotImplementedError
