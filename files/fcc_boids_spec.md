# FCC Falsification Test: Boids Simulation Specification

## Purpose

Test the core FCC prediction: **a critical F/C ratio produces a discontinuous jump in emergence metrics (TEΔ, S)**.

If the jump exists → FCC has empirical support, proceed to full paper.
If no jump → revise theory before publishing.

---

## System: Boids Flocking Model

### Why Boids?

- Simple, well-understood local rules
- Clear feedback mechanism (agents sense neighbors)
- Tunable constraint parameters
- Observable macro-pattern (flock coherence)
- Fast to simulate, easy to visualize
- Maps cleanly to FCC components

### FCC Mapping

| FCC Component | Boids Implementation |
|---------------|---------------------|
| **Feedback** | Each boid senses neighbors within radius R and adjusts velocity |
| **Constraint** | Interaction radius R, alignment strength α, boundary conditions |
| **Classification** | Macro-state = flock coherence (polarization, clustering) |
| **F/C Ratio** | We vary constraint strength; feedback is fixed by the rules |

---

## Simulation Architecture

```
boids_fcc_test/
├── simulation/
│   ├── boids.py          # Core Boids model
│   ├── metrics.py        # TEΔ, S calculations
│   └── sweep.py          # Parameter sweep orchestration
├── analysis/
│   ├── phase_plot.py     # Visualize TEΔ vs constraint parameter
│   └── statistics.py     # Jump detection, significance tests
├── config.yaml           # All parameters in one place
├── run_experiment.py     # Main entry point
└── results/              # Output data and plots
```

---

## Core Boids Model (`boids.py`)

### Parameters

```python
@dataclass
class BoidsConfig:
    # World
    n_boids: int = 100
    world_size: float = 100.0
    boundary: str = "periodic"  # or "reflective"
    
    # Dynamics
    max_speed: float = 2.0
    dt: float = 0.1
    
    # Feedback (fixed)
    perception_radius: float = 10.0
    
    # Constraint (swept)
    alignment_weight: float = 1.0      # α - PRIMARY SWEEP PARAMETER
    cohesion_weight: float = 1.0
    separation_weight: float = 1.5
    separation_radius: float = 2.0
    
    # Noise
    noise_std: float = 0.1
```

### Update Rules (Standard Boids)

```python
def update(self, boids: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """
    boids: (N, 2) positions
    velocities: (N, 2) velocities
    Returns: (N, 2) new velocities
    """
    new_velocities = []
    
    for i in range(len(boids)):
        neighbors = self.get_neighbors(i, boids)
        
        if len(neighbors) > 0:
            # Alignment: steer toward average heading of neighbors
            alignment = self.alignment_weight * self.align(i, velocities, neighbors)
            
            # Cohesion: steer toward center of mass of neighbors
            cohesion = self.cohesion_weight * self.cohere(i, boids, neighbors)
            
            # Separation: steer away from close neighbors
            separation = self.separation_weight * self.separate(i, boids, neighbors)
            
            delta_v = alignment + cohesion + separation
        else:
            delta_v = np.zeros(2)
        
        # Add noise
        delta_v += np.random.normal(0, self.noise_std, 2)
        
        new_v = velocities[i] + delta_v
        new_v = self.limit_speed(new_v)
        new_velocities.append(new_v)
    
    return np.array(new_velocities)
```

---

## Metrics (`metrics.py`)

### Micro-State

```python
def get_micro_state(boids: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """
    Full micro-description: all positions and velocities.
    Returns: flattened array of shape (N * 4,) - [x1, y1, vx1, vy1, x2, ...]
    """
    return np.concatenate([boids.flatten(), velocities.flatten()])
```

### Macro-State

```python
def get_macro_state(boids: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """
    Compressed macro-description:
    - Polarization: alignment of velocity vectors (0 = chaos, 1 = perfect alignment)
    - Centroid: center of mass position
    - Dispersion: std of distances from centroid
    - Mean speed: average velocity magnitude
    
    Returns: array of shape (5,) - [polarization, cx, cy, dispersion, mean_speed]
    """
    # Polarization (order parameter)
    velocity_norms = np.linalg.norm(velocities, axis=1, keepdims=True)
    velocity_norms = np.where(velocity_norms > 0, velocity_norms, 1)
    unit_velocities = velocities / velocity_norms
    polarization = np.linalg.norm(unit_velocities.mean(axis=0))
    
    # Centroid
    centroid = boids.mean(axis=0)
    
    # Dispersion
    distances = np.linalg.norm(boids - centroid, axis=1)
    dispersion = distances.std()
    
    # Mean speed
    mean_speed = velocity_norms.mean()
    
    return np.array([polarization, centroid[0], centroid[1], dispersion, mean_speed])
```

### Transfer Entropy Calculation (TEΔ)

```python
def compute_transfer_entropy(
    source_history: np.ndarray,
    target_future: np.ndarray,
    target_history: np.ndarray,
    k: int = 3,  # history length
    bins: int = 10
) -> float:
    """
    TE(source → target) = I(target_future; source_history | target_history)
    
    Using binned estimation for simplicity.
    """
    # Discretize
    source_binned = discretize(source_history, bins)
    target_fut_binned = discretize(target_future, bins)
    target_hist_binned = discretize(target_history, bins)
    
    # TE = H(target_future | target_history) - H(target_future | target_history, source_history)
    h_fut_given_hist = conditional_entropy(target_fut_binned, target_hist_binned)
    h_fut_given_both = conditional_entropy(target_fut_binned, 
                                            np.column_stack([target_hist_binned, source_binned]))
    
    return h_fut_given_hist - h_fut_given_both


def compute_TE_delta(
    micro_history: List[np.ndarray],
    macro_history: List[np.ndarray],
    macro_future: np.ndarray,
    k: int = 3
) -> float:
    """
    TEΔ = TE(macro_history → macro_future) - TE(micro_history → macro_future)
    
    Positive TEΔ means macro-history predicts macro-future better than micro-history does.
    This is the core emergence signal.
    """
    # Prepare history arrays
    micro_hist = np.array(micro_history[-k:])
    macro_hist = np.array(macro_history[-k:])
    
    te_macro = compute_transfer_entropy(macro_hist.flatten(), macro_future, macro_hist.flatten())
    te_micro = compute_transfer_entropy(micro_hist.flatten(), macro_future, macro_hist.flatten())
    
    return te_macro - te_micro
```

### Substrate Robustness (S)

```python
def compute_substrate_robustness(
    te_delta_original: float,
    te_delta_variants: List[float]
) -> float:
    """
    S = 1 - (std of TEΔ across variants) / (mean of TEΔ across variants)
    
    High S means the emergence signal is stable across substrate changes.
    
    Variants: different noise levels, perception radii, boundary conditions
    """
    all_te = [te_delta_original] + te_delta_variants
    mean_te = np.mean(all_te)
    std_te = np.std(all_te)
    
    if mean_te == 0:
        return 0.0
    
    return 1 - (std_te / abs(mean_te))
```

---

## Parameter Sweep (`sweep.py`)

### Sweep Design

```python
@dataclass
class SweepConfig:
    # Primary sweep: alignment weight (constraint strength)
    alpha_min: float = 0.0
    alpha_max: float = 3.0
    alpha_steps: int = 31  # gives 0.1 increments
    
    # Simulation per point
    n_runs: int = 10       # statistical replicates
    n_steps: int = 500     # steps per run
    warmup_steps: int = 100  # discard initial transient
    
    # Substrate variants for S calculation
    noise_variants: List[float] = field(default_factory=lambda: [0.05, 0.15, 0.2])
    radius_variants: List[float] = field(default_factory=lambda: [8.0, 12.0])


def run_sweep(boids_config: BoidsConfig, sweep_config: SweepConfig) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
    - alpha: constraint parameter value
    - run: replicate number
    - te_delta: TEΔ for this run
    - te_delta_std: std across variants (for S)
    - substrate_robustness: S value
    - polarization_mean: average polarization (sanity check)
    """
    results = []
    
    alphas = np.linspace(sweep_config.alpha_min, sweep_config.alpha_max, sweep_config.alpha_steps)
    
    for alpha in tqdm(alphas, desc="Sweeping α"):
        for run in range(sweep_config.n_runs):
            # Run original
            config = replace(boids_config, alignment_weight=alpha)
            te_delta, polarization = run_single(config, sweep_config)
            
            # Run variants for S
            variant_te_deltas = []
            for noise in sweep_config.noise_variants:
                var_config = replace(config, noise_std=noise)
                var_te, _ = run_single(var_config, sweep_config)
                variant_te_deltas.append(var_te)
            
            s = compute_substrate_robustness(te_delta, variant_te_deltas)
            
            results.append({
                'alpha': alpha,
                'run': run,
                'te_delta': te_delta,
                'substrate_robustness': s,
                'polarization_mean': polarization
            })
    
    return pd.DataFrame(results)
```

---

## Analysis (`phase_plot.py`, `statistics.py`)

### Jump Detection

```python
def detect_discontinuity(
    alphas: np.ndarray,
    te_deltas: np.ndarray,
    window: int = 3
) -> Tuple[Optional[float], float, float]:
    """
    Detect if there's a discontinuous jump in TEΔ.
    
    Method: look for maximum second derivative (acceleration of change).
    
    Returns:
    - critical_alpha: location of jump (or None if no significant jump)
    - jump_magnitude: size of the jump
    - p_value: significance of jump vs gradual change
    """
    # Smooth
    te_smooth = pd.Series(te_deltas).rolling(window, center=True).mean().values
    
    # First derivative
    d1 = np.gradient(te_smooth, alphas)
    
    # Second derivative
    d2 = np.gradient(d1, alphas)
    
    # Find max |d2|
    max_idx = np.argmax(np.abs(d2[window:-window])) + window
    critical_alpha = alphas[max_idx]
    
    # Jump magnitude: difference in TEΔ from window before to window after
    before = te_smooth[max_idx - window:max_idx].mean()
    after = te_smooth[max_idx:max_idx + window].mean()
    jump_magnitude = after - before
    
    # Significance: bootstrap test
    # H0: change is gradual (linear)
    # H1: change is discontinuous
    p_value = bootstrap_jump_test(alphas, te_deltas, critical_alpha, jump_magnitude)
    
    if p_value < 0.05:
        return critical_alpha, jump_magnitude, p_value
    else:
        return None, jump_magnitude, p_value
```

### Visualization

```python
def plot_phase_diagram(results: pd.DataFrame, output_path: str):
    """
    Two-panel figure:
    1. TEΔ vs α (with error bands and detected critical point)
    2. S vs α (substrate robustness)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Aggregate by alpha
    grouped = results.groupby('alpha').agg({
        'te_delta': ['mean', 'std'],
        'substrate_robustness': ['mean', 'std'],
        'polarization_mean': 'mean'
    })
    
    alphas = grouped.index.values
    te_mean = grouped['te_delta']['mean'].values
    te_std = grouped['te_delta']['std'].values
    s_mean = grouped['substrate_robustness']['mean'].values
    s_std = grouped['substrate_robustness']['std'].values
    
    # Panel 1: TEΔ
    ax1.plot(alphas, te_mean, 'b-', linewidth=2)
    ax1.fill_between(alphas, te_mean - te_std, te_mean + te_std, alpha=0.3)
    ax1.set_xlabel('Constraint Strength (α)', fontsize=12)
    ax1.set_ylabel('TEΔ (Emergence Signal)', fontsize=12)
    ax1.set_title('Transfer Entropy Difference vs Constraint')
    
    # Mark critical point if detected
    critical, magnitude, p = detect_discontinuity(alphas, te_mean)
    if critical is not None:
        ax1.axvline(critical, color='red', linestyle='--', label=f'Critical α = {critical:.2f}')
        ax1.legend()
    
    # Panel 2: S
    ax2.plot(alphas, s_mean, 'g-', linewidth=2)
    ax2.fill_between(alphas, s_mean - s_std, s_mean + s_std, alpha=0.3, color='green')
    ax2.set_xlabel('Constraint Strength (α)', fontsize=12)
    ax2.set_ylabel('Substrate Robustness (S)', fontsize=12)
    ax2.set_title('Substrate Robustness vs Constraint')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
```

---

## Main Entry Point (`run_experiment.py`)

```python
#!/usr/bin/env python3
"""
FCC Falsification Test: Boids Simulation

Tests the prediction that a critical F/C ratio produces 
discontinuous jumps in emergence metrics (TEΔ, S).

Usage:
    python run_experiment.py [--config config.yaml]
"""

import argparse
from pathlib import Path
import yaml
import json

from simulation.boids import BoidsConfig, BoidsFlock
from simulation.sweep import SweepConfig, run_sweep
from analysis.phase_plot import plot_phase_diagram
from analysis.statistics import detect_discontinuity, compute_effect_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--output-dir', default='results')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    boids_config = BoidsConfig(**config.get('boids', {}))
    sweep_config = SweepConfig(**config.get('sweep', {}))
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("FCC FALSIFICATION TEST: BOIDS SIMULATION")
    print("=" * 60)
    print(f"\nSweeping α from {sweep_config.alpha_min} to {sweep_config.alpha_max}")
    print(f"Runs per point: {sweep_config.n_runs}")
    print(f"Steps per run: {sweep_config.n_steps}")
    print()
    
    # Run sweep
    results = run_sweep(boids_config, sweep_config)
    results.to_csv(output_dir / 'sweep_results.csv', index=False)
    
    # Analyze
    grouped = results.groupby('alpha')['te_delta'].mean()
    critical, magnitude, p_value = detect_discontinuity(
        grouped.index.values, 
        grouped.values
    )
    
    effect_size = compute_effect_size(results, critical) if critical else None
    
    # Verdict
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if critical is not None and p_value < 0.05:
        print(f"\n✓ DISCONTINUOUS JUMP DETECTED")
        print(f"  Critical α: {critical:.3f}")
        print(f"  Jump magnitude: {magnitude:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Cohen's d: {effect_size:.3f}")
        
        if effect_size > 0.5:
            verdict = "FCC_SUPPORTED"
            print(f"\n→ FCC PREDICTION SUPPORTED (d > 0.5)")
        else:
            verdict = "FCC_WEAK"
            print(f"\n→ Jump detected but effect size small (d = {effect_size:.3f})")
    else:
        verdict = "FCC_FALSIFIED"
        print(f"\n✗ NO SIGNIFICANT DISCONTINUITY")
        print(f"  p-value: {p_value:.4f}")
        print(f"\n→ FCC PREDICTION NOT SUPPORTED")
    
    # Save verdict
    verdict_data = {
        'verdict': verdict,
        'critical_alpha': critical,
        'jump_magnitude': magnitude,
        'p_value': p_value,
        'effect_size': effect_size,
        'config': config
    }
    
    with open(output_dir / 'verdict.json', 'w') as f:
        json.dump(verdict_data, f, indent=2, default=str)
    
    # Plot
    plot_phase_diagram(results, output_dir / 'phase_diagram.png')
    print(f"\nPlot saved to {output_dir / 'phase_diagram.png'}")
    
    return verdict


if __name__ == '__main__':
    main()
```

---

## Config File (`config.yaml`)

```yaml
boids:
  n_boids: 100
  world_size: 100.0
  boundary: periodic
  max_speed: 2.0
  dt: 0.1
  perception_radius: 10.0
  cohesion_weight: 1.0
  separation_weight: 1.5
  separation_radius: 2.0
  noise_std: 0.1

sweep:
  alpha_min: 0.0
  alpha_max: 3.0
  alpha_steps: 31
  n_runs: 10
  n_steps: 500
  warmup_steps: 100
  noise_variants: [0.05, 0.15, 0.2]
  radius_variants: [8.0, 12.0]
```

---

## Expected Outcomes

### If FCC Is Correct

```
TEΔ
 │
 │                    ╭───────
 │                   ╱
 │                  ╱
 │_________________╱
 │
 └──────────────────────────── α
               ↑
        critical point
```

A clear phase transition: low TEΔ (no emergence) → sharp jump → high TEΔ (emergence).

### If FCC Is Wrong

```
TEΔ
 │
 │                    ───────
 │               ────
 │          ────
 │     ────
 │────
 └──────────────────────────── α
```

Gradual increase, no discontinuity. Or no relationship at all.

---

## Implementation Notes

1. **Start simple**: Get the basic Boids running and visualize before adding metrics.

2. **Validate TEΔ calculation**: Use known test cases (random noise should give TEΔ ≈ 0; perfectly correlated should give high TE).

3. **Watch for artifacts**: Discretization bins affect TE estimates. Try multiple bin sizes.

4. **Parallelization**: The sweep is embarrassingly parallel — each (α, run) pair is independent.

5. **Checkpointing**: Save intermediate results; sweeps can take hours.

---

## Deliverables

After running:

1. `results/sweep_results.csv` — raw data
2. `results/phase_diagram.png` — visualization
3. `results/verdict.json` — binary adjudication with statistics

These become the empirical foundation for (or against) the paper.
