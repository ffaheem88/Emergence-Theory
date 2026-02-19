#!/bin/bash
# ============================================================
# Boids Experiment - One-Click Launcher (Linux / macOS)
# ============================================================
# First-time setup:  chmod +x run.sh
# Full experiment:   ./run.sh
# Quick test:        ./run.sh quick
# Custom workers:    ./run.sh --workers 4
# ============================================================

set -e
cd "$(dirname "$0")"

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Setting up Python environment (first time only)..."
    python3 -m venv .venv
fi

# Activate and install deps
source .venv/bin/activate
pip install -r requirements.txt -q

# Handle mode as first argument
case "$1" in
    quick)
        shift
        python run.py --quick "$@"
        ;;
    revised)
        shift
        python fcc_revised.py "$@"
        ;;
    *)
        python run.py "$@"
        ;;
esac
