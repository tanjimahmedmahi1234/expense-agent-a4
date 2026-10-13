#!/bin/bash
# starts the app, then open http://127.0.0.1:8000
cd "$(dirname "$0")"
source .venv/bin/activate
python -m uvicorn app.main:app --reload
