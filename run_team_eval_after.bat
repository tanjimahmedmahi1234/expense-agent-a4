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
echo Re-running the 14 cases in team mode after the two reviewer fixes. About 4 minutes.
echo Do not close this window.
python -m scripts.evaluate_team --repeats 2 --pause 3 --no-single --no-faults --label after_fix > evaluation\team_eval_output_after_fix.txt 2>&1
type evaluation\team_eval_output_after_fix.txt | findstr /R "^ok ^BAD ^| ^saved ^Reviewer"
echo.
echo Done.
pause
