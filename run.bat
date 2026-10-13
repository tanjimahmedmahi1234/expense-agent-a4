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
echo Running the single agent demo, about 30 seconds...
python run_demo.py > demo_output.txt 2>&1
type demo_output.txt
pause
