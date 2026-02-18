"""
Analysis for FCC Falsification Test

Detects discontinuous jumps in emergence metrics and generates visualizations.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from scipy import stats
import json


def detect_discontinuity(
    alphas: np.ndarray,
    te_deltas: np.ndarray,
    window: int = 3
) -> Tuple[Optional[float], float, float]:
    """
    Detect if there's a discontinuous jump in TEΔ.

    Method: Look for maximum second derivative (acceleration of change).

    Returns:
    - critical_alpha: location of jump (or None if no significant jump)
    - jump_magnitude: size of the jump
    - p_value: significance of jump vs gradual change
    """
    if len(alphas) < window * 2 + 1:
        return None, 0.0, 1.0

    # Smooth the data
    te_smooth = pd.Series(te_deltas).rolling(window, center=True, min_periods=1).mean().values

    # First derivative
    d1 = np.gradient(te_smooth, alphas)

    # Second derivative
    d2 = np.gradient(d1, alphas)

    # Find max |d2| (excluding edges)
    margin = max(window, 2)
    if len(d2) <= 2 * margin:
        return None, 0.0, 1.0

    search_region = np.abs(d2[margin:-margin])
    max_idx = np.argmax(search_region) + margin
    critical_alpha = alphas[max_idx]

    # Jump magnitude: difference in TEΔ from window before to window after
    before_idx = max(0, max_idx - window)
    after_idx = min(len(te_smooth), max_idx + window)
    before = te_smooth[before_idx:max_idx].mean()
    after = te_smooth[max_idx:after_idx].mean()
    jump_magnitude = after - before

    # Significance test using bootstrap
    p_value = bootstrap_jump_test(alphas, te_deltas, critical_alpha, jump_magnitude)

    if p_value < 0.05:
        return critical_alpha, jump_magnitude, p_value
    else:
        return None, jump_magnitude, p_value


def bootstrap_jump_test(
    alphas: np.ndarray,
    te_deltas: np.ndarray,
    critical_alpha: float,
    observed_jump: float,
    n_bootstrap: int = 1000
) -> float:
    """
    Bootstrap test for jump significance.
    H0: The change is gradual (linear)
    H1: There is a discontinuous jump

    Returns p-value.
    """
    # Fit linear model under H0
    slope, intercept, _, _, _ = stats.linregress(alphas, te_deltas)
    residuals = te_deltas - (slope * alphas + intercept)

    # Bootstrap: resample residuals and measure max local jump
    bootstrap_jumps = []

    for _ in range(n_bootstrap):
        # Resample residuals
        resampled_residuals = np.random.choice(residuals, size=len(residuals), replace=True)
        resampled_te = slope * alphas + intercept + resampled_residuals

        # Measure max local jump in resampled data
        smoothed = pd.Series(resampled_te).rolling(3, center=True, min_periods=1).mean().values
        diffs = np.diff(smoothed)
        max_jump = np.max(np.abs(diffs)) if len(diffs) > 0 else 0
        bootstrap_jumps.append(max_jump)

    # P-value: proportion of bootstrap jumps >= observed jump
    p_value = np.mean(np.array(bootstrap_jumps) >= abs(observed_jump))

    return max(p_value, 1/n_bootstrap)  # Avoid p=0


def compute_effect_size(results: pd.DataFrame, critical_alpha: Optional[float]) -> Optional[float]:
    """
    Compute Cohen's d effect size for the jump.

    Compares TEΔ values before and after the critical point.
    """
    if critical_alpha is None:
        return None

    before = results[results['alpha'] < critical_alpha]['te_delta']
    after = results[results['alpha'] >= critical_alpha]['te_delta']

    if len(before) < 2 or len(after) < 2:
        return None

    # Pooled standard deviation
    n1, n2 = len(before), len(after)
    var1, var2 = before.var(), after.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))

    if pooled_std < 1e-10:
        return 0.0

    # Cohen's d
    d = (after.mean() - before.mean()) / pooled_std

    return d


def analyze_results(results: pd.DataFrame) -> dict:
    """
    Full analysis of sweep results.

    Returns dict with:
    - critical_alpha
    - jump_magnitude
    - p_value
    - effect_size
    - verdict
    """
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

    # Detect discontinuity
    critical, magnitude, p_value = detect_discontinuity(alphas, te_mean)

    # Effect size
    effect_size = compute_effect_size(results, critical)

    # Determine verdict
    if critical is not None and p_value < 0.05:
        if effect_size is not None and effect_size > 0.5:
            verdict = "FCC_SUPPORTED"
        else:
            verdict = "FCC_WEAK"
    else:
        verdict = "FCC_FALSIFIED"

    return {
        'critical_alpha': critical,
        'jump_magnitude': magnitude,
        'p_value': p_value,
        'effect_size': effect_size,
        'verdict': verdict,
        'summary': {
            'alphas': alphas.tolist(),
            'te_delta_mean': te_mean.tolist(),
            'te_delta_std': te_std.tolist(),
            's_mean': s_mean.tolist()
        }
    }


def generate_plot_data(results: pd.DataFrame) -> dict:
    """
    Generate data for plotting (used by web UI).
    """
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
    pol_mean = grouped['polarization_mean'].values

    # Detect critical point
    critical, magnitude, p_value = detect_discontinuity(alphas, te_mean)

    return {
        'alphas': alphas.tolist(),
        'te_delta': {
            'mean': te_mean.tolist(),
            'std': te_std.tolist()
        },
        'substrate_robustness': {
            'mean': s_mean.tolist(),
            'std': s_std.tolist()
        },
        'polarization': pol_mean.tolist(),
        'critical_point': {
            'alpha': critical,
            'magnitude': magnitude,
            'p_value': p_value
        }
    }


try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_phase_diagram(results: pd.DataFrame, output_path: str):
    """
    Generate phase diagram plot.

    Two-panel figure:
    1. TEΔ vs α (with error bands and detected critical point)
    2. S vs α (substrate robustness)
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot generation")
        return

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

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: TEΔ
    ax1 = axes[0]
    ax1.plot(alphas, te_mean, 'b-', linewidth=2, label='TEΔ')
    ax1.fill_between(alphas, te_mean - te_std, te_mean + te_std, alpha=0.3)
    ax1.set_xlabel('Constraint Strength (α)', fontsize=12)
    ax1.set_ylabel('TEΔ (Emergence Signal)', fontsize=12)
    ax1.set_title('Transfer Entropy Difference vs Constraint')
    ax1.grid(True, alpha=0.3)

    # Mark critical point if detected
    critical, magnitude, p_value = detect_discontinuity(alphas, te_mean)
    if critical is not None:
        ax1.axvline(critical, color='red', linestyle='--', linewidth=2,
                    label=f'Critical α = {critical:.2f}\n(p = {p_value:.3f})')
        ax1.legend()

    # Panel 2: S
    ax2 = axes[1]
    ax2.plot(alphas, s_mean, 'g-', linewidth=2, label='S')
    ax2.fill_between(alphas, s_mean - s_std, s_mean + s_std, alpha=0.3, color='green')
    ax2.set_xlabel('Constraint Strength (α)', fontsize=12)
    ax2.set_ylabel('Substrate Robustness (S)', fontsize=12)
    ax2.set_title('Substrate Robustness vs Constraint')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)

    # Panel 3: Polarization
    ax3 = axes[2]
    pol_mean = grouped['polarization_mean'].values
    ax3.plot(alphas, pol_mean, 'm-', linewidth=2, label='Polarization')
    ax3.set_xlabel('Constraint Strength (α)', fontsize=12)
    ax3.set_ylabel('Polarization (Order Parameter)', fontsize=12)
    ax3.set_title('Flock Polarization vs Constraint')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)

    # Create synthetic results with a phase transition
    alphas = np.linspace(0, 3, 31)
    results = []

    for alpha in alphas:
        for run in range(5):
            # Simulate phase transition around alpha=1.5
            if alpha < 1.5:
                te_delta = 0.1 + np.random.normal(0, 0.02)
            else:
                te_delta = 0.3 + np.random.normal(0, 0.03)

            results.append({
                'alpha': alpha,
                'run': run,
                'te_delta': te_delta,
                'substrate_robustness': 0.5 + 0.3 * (alpha > 1.5) + np.random.normal(0, 0.05),
                'polarization_mean': 0.3 + 0.5 * min(alpha / 2, 1) + np.random.normal(0, 0.05)
            })

    df = pd.DataFrame(results)

    # Analyze
    analysis = analyze_results(df)
    print("\nAnalysis Results:")
    print(f"  Critical α: {analysis['critical_alpha']}")
    print(f"  Jump magnitude: {analysis['jump_magnitude']:.4f}")
    print(f"  p-value: {analysis['p_value']:.4f}")
    print(f"  Effect size: {analysis['effect_size']:.3f}")
    print(f"  Verdict: {analysis['verdict']}")

    # Generate plot
    if HAS_MATPLOTLIB:
        plot_phase_diagram(df, 'test_phase_diagram.png')
