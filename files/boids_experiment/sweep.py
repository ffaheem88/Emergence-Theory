"""
Parameter Sweep for FCC Falsification Test

Sweeps alignment weight (alpha) to detect phase transition in emergence metrics.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field, replace
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os

from boids import BoidsConfig, run_simulation
from metrics import compute_te_delta_from_simulation, compute_substrate_robustness


@dataclass
class SweepConfig:
    """Configuration for parameter sweep"""
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

    # Parallel processing
    n_workers: int = 4

    def to_dict(self):
        return {
            'alpha_min': self.alpha_min,
            'alpha_max': self.alpha_max,
            'alpha_steps': self.alpha_steps,
            'n_runs': self.n_runs,
            'n_steps': self.n_steps,
            'warmup_steps': self.warmup_steps,
            'noise_variants': self.noise_variants,
            'n_workers': self.n_workers
        }


def run_single_simulation(args: Tuple) -> dict:
    """Run a single simulation and compute metrics (for parallel execution)"""
    boids_config_dict, sweep_config_dict, alpha, run_id = args

    # Reconstruct configs
    boids_config = BoidsConfig(**boids_config_dict)
    boids_config.alignment_weight = alpha
    n_steps = sweep_config_dict['n_steps']
    warmup = sweep_config_dict['warmup_steps']
    noise_variants = sweep_config_dict['noise_variants']

    # Run original simulation
    history = run_simulation(boids_config, n_steps)
    te_delta, polarization = compute_te_delta_from_simulation(history, warmup)

    # Run variants for substrate robustness
    variant_te_deltas = []
    for noise in noise_variants:
        variant_config = BoidsConfig(**boids_config_dict)
        variant_config.alignment_weight = alpha
        variant_config.noise_std = noise
        var_history = run_simulation(variant_config, n_steps)
        var_te, _ = compute_te_delta_from_simulation(var_history, warmup)
        variant_te_deltas.append(var_te)

    s = compute_substrate_robustness(te_delta, variant_te_deltas)

    return {
        'alpha': alpha,
        'run': run_id,
        'te_delta': te_delta,
        'substrate_robustness': s,
        'polarization_mean': polarization,
        'variant_te_deltas': variant_te_deltas
    }


def run_sweep(
    boids_config: BoidsConfig,
    sweep_config: SweepConfig,
    progress_callback=None
) -> pd.DataFrame:
    """
    Run the full parameter sweep.

    Returns DataFrame with columns:
    - alpha: constraint parameter value
    - run: replicate number
    - te_delta: TEΔ for this run
    - substrate_robustness: S value
    - polarization_mean: average polarization
    """
    alphas = np.linspace(sweep_config.alpha_min, sweep_config.alpha_max, sweep_config.alpha_steps)

    # Prepare arguments for parallel execution
    args_list = []
    for alpha in alphas:
        for run_id in range(sweep_config.n_runs):
            args_list.append((
                boids_config.to_dict(),
                sweep_config.to_dict(),
                alpha,
                run_id
            ))

    results = []
    total = len(args_list)
    completed = 0

    # Run with parallel processing
    if sweep_config.n_workers > 1:
        with ProcessPoolExecutor(max_workers=sweep_config.n_workers) as executor:
            futures = {executor.submit(run_single_simulation, args): args for args in args_list}

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total, result['alpha'])
                except Exception as e:
                    print(f"Error in simulation: {e}")
                    completed += 1
    else:
        # Sequential execution (for debugging or single-core)
        for args in args_list:
            try:
                result = run_single_simulation(args)
                results.append(result)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total, result['alpha'])
            except Exception as e:
                print(f"Error in simulation: {e}")
                completed += 1

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Sort by alpha and run
    df = df.sort_values(['alpha', 'run']).reset_index(drop=True)

    return df


def run_quick_sweep(
    boids_config: BoidsConfig,
    alpha_values: List[float],
    n_steps: int = 200,
    warmup: int = 50
) -> List[dict]:
    """
    Quick sweep for interactive visualization.
    Returns list of {alpha, te_delta, polarization} dicts.
    """
    results = []

    for alpha in alpha_values:
        config = BoidsConfig(**boids_config.to_dict())
        config.alignment_weight = alpha

        history = run_simulation(config, n_steps)
        te_delta, polarization = compute_te_delta_from_simulation(history, warmup)

        results.append({
            'alpha': alpha,
            'te_delta': te_delta,
            'polarization': polarization
        })

    return results


if __name__ == "__main__":
    import time

    # Test sweep with reduced parameters
    print("Running test sweep...")

    boids_config = BoidsConfig(n_boids=50)
    sweep_config = SweepConfig(
        alpha_min=0.0,
        alpha_max=2.0,
        alpha_steps=5,
        n_runs=2,
        n_steps=200,
        warmup_steps=50,
        n_workers=1
    )

    def progress(done, total, alpha):
        print(f"  Progress: {done}/{total} (α={alpha:.2f})")

    start = time.time()
    results = run_sweep(boids_config, sweep_config, progress)
    elapsed = time.time() - start

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"\nResults summary:")
    print(results.groupby('alpha')[['te_delta', 'polarization_mean']].mean())
