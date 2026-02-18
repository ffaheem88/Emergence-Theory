#!/usr/bin/env python3
"""
FCC Falsification Test: Boids Simulation

Tests the prediction that a critical F/C ratio produces
discontinuous jumps in emergence metrics (TEΔ, S).

Usage:
    python run.py [--config config.yaml] [--quick]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from boids import BoidsConfig
from sweep import SweepConfig, run_sweep, run_quick_sweep
from analyze import analyze_results, plot_phase_diagram, generate_plot_data


def load_config(config_path: str) -> tuple:
    """Load configuration from YAML file"""
    if not HAS_YAML:
        print("PyYAML not installed, using default config")
        return BoidsConfig(), SweepConfig()

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        boids_config = BoidsConfig(**config.get('boids', {}))
        sweep_config = SweepConfig(**config.get('sweep', {}))
        return boids_config, sweep_config
    except FileNotFoundError:
        print(f"Config file {config_path} not found, using defaults")
        return BoidsConfig(), SweepConfig()


def progress_bar(current: int, total: int, alpha: float, width: int = 40):
    """Print progress bar"""
    percent = current / total
    filled = int(width * percent)
    bar = '=' * filled + '-' * (width - filled)
    sys.stdout.write(f'\r[{bar}] {current}/{total} (α={alpha:.2f})')
    sys.stdout.flush()
    if current == total:
        print()


def run_full_experiment(boids_config: BoidsConfig, sweep_config: SweepConfig, output_dir: Path):
    """Run the full parameter sweep experiment"""
    print("=" * 60)
    print("FCC FALSIFICATION TEST: BOIDS SIMULATION")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Boids: {boids_config.n_boids}")
    print(f"  World size: {boids_config.world_size}")
    print(f"  α range: {sweep_config.alpha_min} to {sweep_config.alpha_max}")
    print(f"  α steps: {sweep_config.alpha_steps}")
    print(f"  Runs per α: {sweep_config.n_runs}")
    print(f"  Steps per run: {sweep_config.n_steps}")
    print(f"  Warmup: {sweep_config.warmup_steps}")
    print(f"  Workers: {sweep_config.n_workers}")
    print()

    total_sims = sweep_config.alpha_steps * sweep_config.n_runs
    print(f"Total simulations: {total_sims}")
    print()

    # Run sweep
    print("Running parameter sweep...")
    results = run_sweep(boids_config, sweep_config, progress_bar)

    # Save raw results
    results_path = output_dir / 'sweep_results.csv'
    results.to_csv(results_path, index=False)
    print(f"\nRaw results saved to {results_path}")

    # Analyze
    print("\nAnalyzing results...")
    analysis = analyze_results(results)

    # Generate verdict
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if analysis['critical_alpha'] is not None and analysis['p_value'] < 0.05:
        print(f"\n  DISCONTINUOUS JUMP DETECTED")
        print(f"  Critical α: {analysis['critical_alpha']:.3f}")
        print(f"  Jump magnitude: {analysis['jump_magnitude']:.4f}")
        print(f"  p-value: {analysis['p_value']:.4f}")
        if analysis['effect_size'] is not None:
            print(f"  Cohen's d: {analysis['effect_size']:.3f}")

        if analysis['verdict'] == "FCC_SUPPORTED":
            print(f"\n  --> FCC PREDICTION SUPPORTED (d > 0.5)")
        else:
            print(f"\n  --> Jump detected but effect size small")
    else:
        print(f"\n  NO SIGNIFICANT DISCONTINUITY")
        print(f"  p-value: {analysis['p_value']:.4f}")
        print(f"\n  --> FCC PREDICTION NOT SUPPORTED")

    print(f"\n  VERDICT: {analysis['verdict']}")
    print("=" * 60)

    # Save verdict
    verdict_path = output_dir / 'verdict.json'
    verdict_data = {
        'verdict': analysis['verdict'],
        'critical_alpha': analysis['critical_alpha'],
        'jump_magnitude': analysis['jump_magnitude'],
        'p_value': analysis['p_value'],
        'effect_size': analysis['effect_size'],
        'timestamp': datetime.now().isoformat(),
        'config': {
            'boids': boids_config.to_dict(),
            'sweep': sweep_config.to_dict()
        }
    }
    with open(verdict_path, 'w') as f:
        json.dump(verdict_data, f, indent=2)
    print(f"\nVerdict saved to {verdict_path}")

    # Save plot data for web UI
    plot_data_path = output_dir / 'plot_data.json'
    plot_data = generate_plot_data(results)
    plot_data['analysis'] = {
        'verdict': analysis['verdict'],
        'critical_alpha': analysis['critical_alpha'],
        'p_value': analysis['p_value'],
        'effect_size': analysis['effect_size']
    }
    with open(plot_data_path, 'w') as f:
        json.dump(plot_data, f, indent=2)
    print(f"Plot data saved to {plot_data_path}")

    # Generate matplotlib plot
    plot_path = output_dir / 'phase_diagram.png'
    plot_phase_diagram(results, str(plot_path))

    return analysis


def run_quick_test(boids_config: BoidsConfig, output_dir: Path):
    """Run a quick test for visualization"""
    print("Running quick test (5 alpha values, 1 run each)...")

    alphas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    results = run_quick_sweep(boids_config, alphas, n_steps=200, warmup=50)

    print("\nQuick test results:")
    print(f"{'Alpha':>8} {'TEΔ':>10} {'Polarization':>12}")
    print("-" * 32)
    for r in results:
        print(f"{r['alpha']:>8.2f} {r['te_delta']:>10.4f} {r['polarization']:>12.4f}")

    # Save results
    quick_results_path = output_dir / 'quick_results.json'
    with open(quick_results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nQuick results saved to {quick_results_path}")


def main():
    parser = argparse.ArgumentParser(description='FCC Falsification Test: Boids Simulation')
    parser.add_argument('--config', default='config.yaml', help='Configuration file')
    parser.add_argument('--output-dir', default=None, help='Output directory (default: auto-versioned results/run_YYYYMMDD_HHMMSS/)')
    parser.add_argument('--quick', action='store_true', help='Run quick test only')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    args = parser.parse_args()

    # Load config
    boids_config, sweep_config = load_config(args.config)

    # Override workers if specified
    if args.workers is not None:
        sweep_config.n_workers = args.workers

    # Create output directory — versioned by default
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        run_id = datetime.now().strftime('run_%Y%m%d_%H%M%S')
        output_dir = Path('results') / run_id

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    if args.quick:
        run_quick_test(boids_config, output_dir)
    else:
        run_full_experiment(boids_config, sweep_config, output_dir)

    # Update results/latest symlink to point to this run
    latest_link = Path('results') / 'latest'
    try:
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(output_dir.resolve())
        print(f"results/latest → {output_dir}")
    except Exception as e:
        print(f"(Note: could not update results/latest symlink: {e})")


if __name__ == '__main__':
    main()
