#!/bin/bash
# team mode only, after the two reviewer fixes, about 4 minutes
cd "$(dirname "$0")"
source .venv/bin/activate
mkdir -p evaluation
python -m scripts.evaluate_team --repeats 2 --pause 3 --no-single --no-faults --label after_fix | tee evaluation/team_eval_output_after_fix.txt
