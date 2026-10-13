"""
COIT12204 Assessment 4, Tanjim Ahmed Mahi (12298057)

One logger for the whole app. It prints to the terminal and also writes to logs/agent.log
so I can show the log file in the demo video.
"""
import logging
import os

from app.config import LOG_FILE

# make sure the logs folder exists otherwise FileHandler crashes
os.makedirs(os.path.dirname(LOG_FILE) or ".", exist_ok=True)

logger = logging.getLogger("expense_agent")
logger.setLevel(logging.INFO)

# only add the handlers once, otherwise every import doubles the log lines
if not logger.handlers:
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
