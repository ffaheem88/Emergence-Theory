# Running the Boids Experiment

## Prerequisites

- Python 3.9 or newer — [download here](https://python.org)
- That's it. Everything else is installed automatically.

## First Time (one step)

**Windows:** No setup needed — just double-click `run.bat`

**Linux / macOS:**
```bash
chmod +x run.sh   # only needed once
./run.sh
```

## Running

| Command | What it does |
|---|---|
| `./run.sh` or `run.bat` | Standard sweep — raw TEΔ across α range (~15-30 min) |
| `./run.sh quick` or `run.bat quick` | Quick test — 7 alpha values, takes ~1 min |
| `./run.sh revised` or `run.bat revised` | **Revised FCC experiment** — macro-state predictability analysis (~30-60 min). This is the real test. |
| `./run.sh --workers 4` | Use 4 CPU cores (default: from config.yaml) |
| `./run.sh revised --runs 12 --boids 500` | Revised with custom params |

## Results

Every run creates a **new timestamped folder** so nothing is ever overwritten:

```
results/
  run_20260218_143022/   ← first run
    sweep_results.csv
    verdict.json
    phase_diagram.png
    plot_data.json
  run_20260219_091500/   ← second run
    ...
  latest/                ← symlink → most recent run (Linux/Mac)
```

On Windows the `latest` symlink may not be created — just open the newest `run_*` folder.

## What to look for

Open `verdict.json` in the latest run folder.

**Standard sweep** (`run.sh`): Measures raw TEΔ — useful as a baseline but may not detect the phase transition.

**Revised experiment** (`run.sh revised`): This is the key test. Measures macro-state predictability:
```json
{
  "verdict": "FCC_SUPPORTED",
  "conditions": {
    "D": { "cohens_d": 1.52, "critical_alpha": 0.04, "p_value": 0.001 }
  }
}
```

- `FCC_SUPPORTED` + Cohen's d > 0.5 → emergence confirmed ✅
- `FCC_FALSIFIED` → no discontinuous jump in macro-state predictability

**Why two modes?** Standard TEΔ measures individual agent-level information transfer. The revised experiment measures whether the *collective* (macro state: flock count, mode, coherence) becomes predictable — which is the actual signature of emergence.

## Sharing results

Push the new run folder back to GitHub:
```bash
git add results/run_XXXXXX/
git commit -m "Robustness run XXXXXX"
git push
```
