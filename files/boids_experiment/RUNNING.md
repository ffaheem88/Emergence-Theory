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
| `./run.sh` or `run.bat` | Full experiment (~15-30 min) |
| `./run.sh quick` or `run.bat quick` | Quick test — 7 alpha values, takes ~1 min |
| `./run.sh --workers 4` | Use 4 CPU cores (default: all) |

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

Open `verdict.json` in the latest run folder. The key field:

```json
{
  "verdict": "FCC_SUPPORTED",
  "critical_alpha": 0.04,
  "effect_size": 1.52,
  ...
}
```

- `FCC_SUPPORTED` + `effect_size > 0.5` → emergence confirmed ✅
- `FCC_NOT_SUPPORTED` → no discontinuous jump found

## Sharing results

Push the new run folder back to GitHub:
```bash
git add results/run_XXXXXX/
git commit -m "Robustness run XXXXXX"
git push
```
