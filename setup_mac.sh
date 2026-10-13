#!/bin/bash
# first time setup on a Mac. Run with:  bash setup_mac.sh
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Made a .env file. Open it and paste your OpenAI key in."
fi
