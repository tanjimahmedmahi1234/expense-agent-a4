#!/bin/bash
# classifier only vs classifier + reviewer, about 10 minutes
cd "$(dirname "$0")"
source .venv/bin/activate
mkdir -p evaluation
python -m scripts.evaluate_team --repeats 2 --pause 3 | tee evaluation/team_eval_output.txt
