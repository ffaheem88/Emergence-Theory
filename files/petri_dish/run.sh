#!/bin/bash
# ============================================================
# Petri Dish — LLM Emergence Experiment (Linux / macOS)
# ============================================================
# First time:  chmod +x run.sh
# Run:         ./run.sh
# Custom port: ./run.sh --port 9090
# Custom model: ./run.sh --model llama3.2:3b
# ============================================================

set -e
cd "$(dirname "$0")"

# Check Ollama
if ! command -v ollama &>/dev/null; then
    echo "❌ Ollama not found. Install from https://ollama.com"
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
    echo "Starting Ollama..."
    ollama serve &>/dev/null &
    sleep 3
fi

# GPU detection
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "🎮 GPU detected: $GPU_INFO"
    echo "   Tip: Use larger models with --model llama3.2:3b or --model tinyllama"
    DEFAULT_MODEL="qwen2:0.5b"
else
    echo "💻 CPU mode"
    DEFAULT_MODEL="qwen2:0.5b"
fi

# Ensure model is pulled
echo "Checking model..."
ollama pull ${DEFAULT_MODEL} 2>/dev/null || true

# Setup venv
if [ ! -d ".venv" ]; then
    echo "Setting up Python environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install -r requirements.txt -q

echo ""
echo "🧫 Starting Petri Dish..."
echo "   Open http://localhost:8080 in your browser"
echo "   Press Ctrl+C to stop"
echo ""

python server.py "$@"
