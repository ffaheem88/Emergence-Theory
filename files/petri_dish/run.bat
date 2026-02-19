@echo off
REM ============================================================
REM Petri Dish — LLM Emergence Experiment (Windows)
REM ============================================================
REM Run:          run.bat
REM Custom port:  run.bat --port 9090
REM Custom model: run.bat --model llama3.2:3b
REM ============================================================

cd /d "%~dp0"

REM Check Ollama
where ollama >nul 2>&1
if errorlevel 1 (
    echo ERROR: Ollama not found. Install from https://ollama.com
    pause
    exit /b 1
)

REM GPU detection
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo GPU detected! Tip: Use larger models with --model llama3.2:3b
)

REM Ensure model
echo Checking model...
ollama pull qwen2:0.5b >nul 2>&1

REM Setup venv
if not exist ".venv" (
    echo Setting up Python environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat
pip install -r requirements.txt -q

echo.
echo Starting Petri Dish...
echo Open http://localhost:8080 in your browser
echo Press Ctrl+C to stop
echo.

python server.py %*
pause
