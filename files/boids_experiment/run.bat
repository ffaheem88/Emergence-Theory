@echo off
REM ============================================================
REM Boids Experiment - One-Click Launcher (Windows)
REM ============================================================
REM Full experiment:  run.bat
REM Quick test:       run.bat quick
REM Custom workers:   run.bat --workers 4
REM ============================================================

cd /d "%~dp0"

REM Create venv if it doesn't exist
if not exist ".venv" (
    echo Setting up Python environment (first time only)...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Python not found. Install Python 3.9+ from https://python.org
        pause
        exit /b 1
    )
)

REM Activate venv and install deps
call .venv\Scripts\activate.bat
pip install -r requirements.txt -q

REM Handle mode as first argument
if "%1"=="quick" (
    python run.py --quick %2 %3 %4 %5
) else if "%1"=="revised" (
    python fcc_revised.py %2 %3 %4 %5
) else (
    python run.py %*
)

pause
