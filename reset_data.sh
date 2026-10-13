#!/bin/bash
# fresh sample expenses with today's dates
cd "$(dirname "$0")"
source .venv/bin/activate
python -m scripts.seed_expenses
