# CLAUDE CODE: Build FCC Falsification Test

## What We're Testing

**FCC Proposition**: Emergence is a phase transition — at a critical constraint strength, there's a discontinuous jump in emergence metrics.

**Prediction**: Sweeping alignment strength (α) in a Boids simulation will show a sharp jump in Transfer Entropy Difference (TEΔ) at some critical value.

**Falsification**: If TEΔ changes gradually with no discontinuity across the full sweep, FCC is wrong.

## Build This

### Structure
```
boids_fcc_test/
├── boids.py         # Boids simulation
├── metrics.py       # TEΔ and S calculations  
├── sweep.py         # Parameter sweep
├── analyze.py       # Jump detection + plotting
├── run.py           # Main entry
├── config.yaml      # Parameters
└── results/         # Output
```

### Boids Model (`boids.py`)
- N=100 agents, periodic boundary, 2D
- Standard rules: alignment, cohesion, separation
- **Key parameter**: `alignment_weight` (α) — this is what we sweep
- Fixed: perception_radius=10, cohesion_weight=1, separation_weight=1.5

### Metrics (`metrics.py`)

**Micro-state**: All positions and velocities (N×4 values)

**Macro-state**: 
- Polarization (velocity alignment, 0-1)
- Centroid (x, y)
- Dispersion (spread)
- Mean speed

**TEΔ** (the key metric):
```
TEΔ = TE(macro_history → macro_future) - TE(micro_history → macro_future)
```
Positive TEΔ = macro predicts better than micro = emergence signal.

Use binned entropy estimation. History length k=3.

**Substrate Robustness (S)**:
```
S = 1 - std(TEΔ across variants) / mean(TEΔ across variants)
```
Variants: different noise levels (0.05, 0.15, 0.2)

### Sweep (`sweep.py`)
- α from 0.0 to 3.0 in 31 steps
- 10 runs per α value
- 500 timesteps per run, discard first 100 (warmup)
- For each run: compute TEΔ, S, mean polarization
- Save to CSV

### Analysis (`analyze.py`)

**Jump detection**:
- Compute second derivative of TEΔ vs α
- Peak in |d²TEΔ/dα²| indicates discontinuity
- Bootstrap test for significance (p < 0.05)
- Cohen's d for effect size (need d > 0.5)

**Plot**:
- Two panels: TEΔ vs α, S vs α
- Mark critical point if detected
- Error bands from replicates

### Main (`run.py`)
```
python run.py --config config.yaml
```

Output:
1. `results/sweep_results.csv`
2. `results/phase_diagram.png`
3. `results/verdict.json` with:
   - verdict: "FCC_SUPPORTED" | "FCC_WEAK" | "FCC_FALSIFIED"
   - critical_alpha, jump_magnitude, p_value, effect_size

## Success Criteria

| Outcome | Condition |
|---------|-----------|
| FCC_SUPPORTED | Discontinuous jump with p < 0.05 AND Cohen's d > 0.5 |
| FCC_WEAK | Jump detected but d < 0.5 |
| FCC_FALSIFIED | No significant discontinuity (p ≥ 0.05) |

## Notes

- Start with visualization: make sure Boids actually flock when α is high
- Validate TE calculation on synthetic data first
- The whole sweep should take ~30-60 minutes on a laptop
- Parallelize if needed (runs are independent)

## Go

Build it, run it, show me the phase diagram.
