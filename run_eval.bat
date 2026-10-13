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
if not exist evaluation mkdir evaluation
echo A3 evaluation: 11 cases x 2 repeats for prompts v1 v2 v3. About 8 minutes.
python -m scripts.evaluate --versions v1 v2 v3 --repeats 2 --pause 5
pause
