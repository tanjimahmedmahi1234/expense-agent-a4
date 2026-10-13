@echo off
cd /d "%~dp0"
if not exist .env (
    echo No .env file found. Copy .env.example to .env and put your OpenAI key in it.
    pause
    exit /b 1
)
if not exist .venv-win\Scripts\python.exe (
    echo Setting up Python for Windows the first time, this takes a minute or two...
    python -m venv .venv-win
    .venv-win\Scripts\python -m pip install -q -r requirements.txt
)
call .venv-win\Scripts\activate.bat
echo Starting the app. Open http://127.0.0.1:8000 in your browser.
echo Press Ctrl+C in this window to stop it.
python -m uvicorn app.main:app --reload
pause
