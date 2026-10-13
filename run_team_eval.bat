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
echo A4 team evaluation: 14 cases x 2 repeats, single agent vs classifier + reviewer,
echo then 4 simulated failures. About 10 minutes. Do not close this window.
python -m scripts.evaluate_team --repeats 2 --pause 3 > evaluation\team_eval_output.txt 2>&1
type evaluation\team_eval_output.txt | findstr /R "^ok ^BAD ^| ^saved ^Reviewer ^=====" 
findstr /C:"    T2 " evaluation\team_eval_output.txt
echo.
echo Done.
pause
