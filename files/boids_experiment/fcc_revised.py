"""
Revised FCC Validation Experiment

Addresses reviewer methodological concerns:
1. Mode classification is now INSTANTANEOUS only (no history features)
2. Includes binned-polarization control to rule out discretization artifacts
3. DBSCAN parameter sweep for robustness
4. Proper change-point detection with ruptures
5. Bootstrap CI for Cohen's d

Four conditions compared:
A: Polarization (continuous) - baseline
B: Binned polarization {low, med, high} - discretization control
C: (n_flocks, Mode) - NO Coherence - KEY TEST
D: (n_flocks, Mode, Coherence) - current method
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor
import json
import os
from datetime import datetime

# Import the existing boids simulation
from boids_fast import BoidsConfig, BoidsFastGrid


# ============================================================================
# INSTANTANEOUS MODE CLASSIFICATION (NO HISTORY)
# ============================================================================

def compute_polarization(velocities: np.ndarray) -> float:
    """Compute polarization (alignment) of velocity vectors"""
    if len(velocities) == 0:
        return 0.0
    norms = np.linalg.norm(velocities, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1e-10)
    unit_vels = velocities / norms
    return float(np.linalg.norm(unit_vels.mean(axis=0)))


def detect_flocks_dbscan(positions: np.ndarray, eps: float = 15.0, min_samples: int = 5) -> np.ndarray:
    """
    Detect flocks using DBSCAN clustering.
    Returns: array of flock labels (-1 = noise/not in flock)
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)
    return clustering.labels_


def get_largest_flock(flock_labels: np.ndarray) -> int:
    """Return the label of the largest flock"""
    unique, counts = np.unique(flock_labels[flock_labels >= 0], return_counts=True)
    if len(unique) == 0:
        return -1
    return unique[np.argmax(counts)]


def classify_mode_instantaneous(
    positions: np.ndarray,
    velocities: np.ndarray,
    flock_labels: np.ndarray
) -> str:
    """
    INSTANTANEOUS mode classification - NO HISTORY FEATURES ALLOWED

    Returns: "dispersed", "milling", or "migrating"
    """
    n_boids = len(positions)
    in_clusters = (flock_labels >= 0).sum()

    # If less than 50% in any cluster -> dispersed
    if in_clusters / n_boids < 0.5:
        return "dispersed"

    # Get largest flock polarization (instantaneous)
    largest = get_largest_flock(flock_labels)
    if largest < 0:
        return "dispersed"

    largest_mask = flock_labels == largest
    flock_pol = compute_polarization(velocities[largest_mask])

    # High polarization = migrating, low = milling
    return "migrating" if flock_pol > 0.6 else "milling"


def compute_coherence(velocities: np.ndarray, flock_labels: np.ndarray) -> str:
    """
    Compute coherence level for flocks.
    Returns: "low", "medium", or "high"
    """
    if (flock_labels >= 0).sum() == 0:
        return "low"

    # Get polarization of all clustered boids
    clustered_vels = velocities[flock_labels >= 0]
    pol = compute_polarization(clustered_vels)

    if pol < 0.4:
        return "low"
    elif pol < 0.7:
        return "medium"
    else:
        return "high"


# ============================================================================
# MACRO-STATE REPRESENTATIONS
# ============================================================================

def get_macro_state_A(positions: np.ndarray, velocities: np.ndarray) -> float:
    """Condition A: Continuous polarization"""
    return compute_polarization(velocities)


def get_macro_state_B(positions: np.ndarray, velocities: np.ndarray) -> str:
    """Condition B: Binned polarization {low, med, high}"""
    pol = compute_polarization(velocities)
    if pol < 0.4:
        return "low"
    elif pol < 0.7:
        return "medium"
    else:
        return "high"


def get_macro_state_C(
    positions: np.ndarray,
    velocities: np.ndarray,
    eps: float = 15.0,
    min_samples: int = 5
) -> Tuple[int, str]:
    """Condition C: (n_flocks, Mode) - NO Coherence"""
    flock_labels = detect_flocks_dbscan(positions, eps, min_samples)
    n_flocks = len(np.unique(flock_labels[flock_labels >= 0]))
    mode = classify_mode_instantaneous(positions, velocities, flock_labels)
    return (n_flocks, mode)


def get_macro_state_D(
    positions: np.ndarray,
    velocities: np.ndarray,
    eps: float = 15.0,
    min_samples: int = 5
) -> Tuple[int, str, str]:
    """Condition D: (n_flocks, Mode, Coherence) - current method"""
    flock_labels = detect_flocks_dbscan(positions, eps, min_samples)
    n_flocks = len(np.unique(flock_labels[flock_labels >= 0]))
    mode = classify_mode_instantaneous(positions, velocities, flock_labels)
    coherence = compute_coherence(velocities, flock_labels)
    return (n_flocks, mode, coherence)


# ============================================================================
# MUTUAL INFORMATION COMPUTATION
# ============================================================================

def compute_mutual_information(states_t: List, states_t1: List) -> float:
    """
    Compute I(S_{t+1}; S_t) - mutual information between consecutive states

    Uses plug-in estimator with frequency counts.
    """
    n = len(states_t)
    if n == 0:
        return 0.0

    # Convert to tuples for hashing
    def to_tuple(s):
        if isinstance(s, tuple):
            return s
        return (s,)

    states_t = [to_tuple(s) for s in states_t]
    states_t1 = [to_tuple(s) for s in states_t1]

    # Count joint and marginal frequencies
    from collections import Counter

    joint_counts = Counter(zip(states_t, states_t1))
    marginal_t = Counter(states_t)
    marginal_t1 = Counter(states_t1)

    # Compute MI
    mi = 0.0
    for (st, st1), joint_count in joint_counts.items():
        p_joint = joint_count / n
        p_t = marginal_t[st] / n
        p_t1 = marginal_t1[st1] / n
        if p_joint > 0 and p_t > 0 and p_t1 > 0:
            mi += p_joint * np.log2(p_joint / (p_t * p_t1))

    return max(0.0, mi)  # MI should be non-negative


def compute_predictability_series(
    history: List[dict],
    condition: str,
    eps: float = 15.0,
    min_samples: int = 5
) -> Tuple[List[float], float]:
    """
    Compute predictability (MI) over sliding windows for a given condition.

    Returns: (window_mi_values, mean_mi)
    """
    window_size = 50  # Window for MI estimation

    # Extract macro-states based on condition
    states = []
    for frame in history:
        pos = np.array(frame['positions'])
        vel = np.array(frame['velocities'])

        if condition == 'A':
            states.append(get_macro_state_A(pos, vel))
        elif condition == 'B':
            states.append(get_macro_state_B(pos, vel))
        elif condition == 'C':
            states.append(get_macro_state_C(pos, vel, eps, min_samples))
        elif condition == 'D':
            states.append(get_macro_state_D(pos, vel, eps, min_samples))

    # For condition A (continuous), discretize into bins
    if condition == 'A':
        # Bin polarization into 10 bins
        states_array = np.array(states)
        bins = np.linspace(0, 1, 11)
        states = [f"bin_{np.digitize(s, bins)}" for s in states_array]

    # Compute MI over sliding windows
    mi_values = []
    for i in range(0, len(states) - window_size - 1, window_size // 2):
        window_t = states[i:i+window_size]
        window_t1 = states[i+1:i+window_size+1]
        mi = compute_mutual_information(window_t, window_t1)
        mi_values.append(mi)

    mean_mi = np.mean(mi_values) if mi_values else 0.0
    return mi_values, mean_mi


# ============================================================================
# SIMULATION AND SWEEP
# ============================================================================

def run_single_simulation(
    alpha: float,
    run_id: int,
    n_steps: int = 2000,
    warmup: int = 500,
    eps: float = 15.0,
    min_samples: int = 5,
    n_boids: int = 500
) -> Dict:
    """
    Run a single simulation and compute all four condition metrics.
    """
    config = BoidsConfig(
        n_boids=n_boids,
        world_size=200.0,
        alignment_weight=alpha,
        separation_weight=1.0,
        cohesion_weight=0.0,
        noise_std=0.1,
        perception_radius=15.0
    )

    flock = BoidsFastGrid(config)
    history = []

    # Run simulation
    for step in range(n_steps):
        positions, velocities = flock.step()
        if step >= warmup:
            history.append({
                'step': step,
                'positions': positions.copy(),
                'velocities': velocities.copy()
            })

    # Compute predictability for each condition
    _, mi_A = compute_predictability_series(history, 'A', eps, min_samples)
    _, mi_B = compute_predictability_series(history, 'B', eps, min_samples)
    _, mi_C = compute_predictability_series(history, 'C', eps, min_samples)
    _, mi_D = compute_predictability_series(history, 'D', eps, min_samples)

    # Also compute mean polarization
    polarizations = [compute_polarization(np.array(h['velocities'])) for h in history]
    mean_pol = np.mean(polarizations)

    return {
        'alpha': alpha,
        'run': run_id,
        'eps': eps,
        'min_samples': min_samples,
        'mi_A': mi_A,  # Continuous polarization (binned for MI)
        'mi_B': mi_B,  # Binned polarization
        'mi_C': mi_C,  # (n_flocks, Mode) - KEY
        'mi_D': mi_D,  # (n_flocks, Mode, Coherence)
        'polarization_mean': mean_pol
    }


def run_alpha_sweep(
    alphas: np.ndarray,
    n_runs: int = 20,
    eps: float = 15.0,
    min_samples: int = 5,
    n_workers: int = 4,
    n_boids: int = 500
) -> pd.DataFrame:
    """Run parameter sweep over alpha values."""

    tasks = [(alpha, run) for alpha in alphas for run in range(n_runs)]
    results = []

    print(f"Running {len(tasks)} simulations (n_boids={n_boids}, eps={eps}, min_samples={min_samples})...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(run_single_simulation, alpha, run, 2000, 500, eps, min_samples, n_boids)
            for alpha, run in tasks
        ]

        for i, future in enumerate(futures):
            result = future.result()
            results.append(result)
            if (i + 1) % 20 == 0:
                print(f"  Completed {i+1}/{len(tasks)}")

    return pd.DataFrame(results)


# ============================================================================
# CHANGE-POINT DETECTION
# ============================================================================

def detect_changepoint_ruptures(alphas: np.ndarray, values: np.ndarray, pen: float = 1.0) -> List[float]:
    """
    Detect change-points using ruptures library.
    Returns list of alpha values where changes occur.
    """
    try:
        import ruptures as rpt
    except ImportError:
        print("WARNING: ruptures not installed, using fallback")
        return []

    # Reshape for ruptures
    signal = values.reshape(-1, 1)

    # PELT algorithm with RBF kernel
    algo = rpt.Pelt(model="rbf").fit(signal)
    changepoints = algo.predict(pen=pen)

    # Convert indices to alpha values
    change_alphas = []
    for cp in changepoints[:-1]:  # Last one is always len(signal)
        if 0 <= cp < len(alphas):
            change_alphas.append(alphas[cp])

    return change_alphas


def bootstrap_cohens_d(before: np.ndarray, after: np.ndarray, n_bootstrap: int = 1000) -> Tuple[float, float, float]:
    """
    Compute Cohen's d with bootstrap 95% CI.
    Returns: (d, ci_low, ci_high)
    """
    def cohens_d(b, a):
        n1, n2 = len(b), len(a)
        if n1 < 2 or n2 < 2:
            return 0.0
        var1, var2 = b.var(), a.var()
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
        if pooled_std < 1e-10:
            return 0.0
        return (a.mean() - b.mean()) / pooled_std

    # Point estimate
    d = cohens_d(before, after)

    # Bootstrap
    d_boots = []
    for _ in range(n_bootstrap):
        b_boot = np.random.choice(before, size=len(before), replace=True)
        a_boot = np.random.choice(after, size=len(after), replace=True)
        d_boots.append(cohens_d(b_boot, a_boot))

    d_boots = np.array(d_boots)
    ci_low = np.percentile(d_boots, 2.5)
    ci_high = np.percentile(d_boots, 97.5)

    return d, ci_low, ci_high


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_condition(
    df: pd.DataFrame,
    condition: str,
    mi_column: str
) -> Dict:
    """
    Analyze a single condition: find changepoint, compute effect size.
    """
    # Aggregate by alpha
    grouped = df.groupby('alpha').agg({
        mi_column: ['mean', 'std']
    })

    alphas = grouped.index.values
    mi_mean = grouped[mi_column]['mean'].values
    mi_std = grouped[mi_column]['std'].values

    # Detect changepoint
    changepoints = detect_changepoint_ruptures(alphas, mi_mean, pen=1.0)

    # Find the most significant changepoint (largest jump)
    critical_alpha = None
    max_jump = 0

    for cp in changepoints:
        idx = np.argmin(np.abs(alphas - cp))
        if idx > 0 and idx < len(mi_mean) - 1:
            jump = abs(mi_mean[idx+1] - mi_mean[idx-1])
            if jump > max_jump:
                max_jump = jump
                critical_alpha = cp

    # If no changepoint from ruptures, use max second derivative
    if critical_alpha is None:
        smoothed = pd.Series(mi_mean).rolling(3, center=True, min_periods=1).mean().values
        d2 = np.gradient(np.gradient(smoothed, alphas), alphas)
        margin = 2
        if len(d2) > 2 * margin:
            max_idx = np.argmax(np.abs(d2[margin:-margin])) + margin
            critical_alpha = alphas[max_idx]

    # Compute Cohen's d with bootstrap CI
    if critical_alpha is not None:
        before = df[df['alpha'] < critical_alpha][mi_column].values
        after = df[df['alpha'] >= critical_alpha][mi_column].values
        if len(before) > 2 and len(after) > 2:
            d, ci_low, ci_high = bootstrap_cohens_d(before, after)
        else:
            d, ci_low, ci_high = 0.0, 0.0, 0.0
    else:
        d, ci_low, ci_high = 0.0, 0.0, 0.0

    # P-value via permutation test
    p_value = permutation_test_jump(df, mi_column, critical_alpha)

    return {
        'condition': condition,
        'critical_alpha': critical_alpha,
        'critical_alpha_ci': None,  # Could add bootstrap CI for changepoint
        'jump_magnitude': max_jump,
        'cohens_d': d,
        'cohens_d_ci_low': ci_low,
        'cohens_d_ci_high': ci_high,
        'p_value': p_value,
        'alphas': alphas.tolist(),
        'mi_mean': mi_mean.tolist(),
        'mi_std': mi_std.tolist(),
        'changepoints': changepoints
    }


def permutation_test_jump(df: pd.DataFrame, mi_column: str, critical_alpha: Optional[float], n_perm: int = 1000) -> float:
    """Permutation test for significance of jump at critical point."""
    if critical_alpha is None:
        return 1.0

    before = df[df['alpha'] < critical_alpha][mi_column].values
    after = df[df['alpha'] >= critical_alpha][mi_column].values

    if len(before) < 2 or len(after) < 2:
        return 1.0

    observed_diff = abs(after.mean() - before.mean())

    # Permutation test
    all_values = np.concatenate([before, after])
    n_before = len(before)

    count = 0
    for _ in range(n_perm):
        np.random.shuffle(all_values)
        perm_before = all_values[:n_before]
        perm_after = all_values[n_before:]
        perm_diff = abs(perm_after.mean() - perm_before.mean())
        if perm_diff >= observed_diff:
            count += 1

    return max(count / n_perm, 1 / n_perm)


def run_full_analysis(df: pd.DataFrame) -> Dict:
    """Run full analysis on all four conditions."""

    results = {
        'B': analyze_condition(df, 'B (Binned Pol)', 'mi_B'),
        'C': analyze_condition(df, 'C (F, Mode)', 'mi_C'),
        'D': analyze_condition(df, 'D (F, Mode, Coh)', 'mi_D'),
    }

    # Determine verdict
    c_has_jump = results['C']['cohens_d'] > 0.5 and results['C']['p_value'] < 0.05
    b_has_jump = results['B']['cohens_d'] > 0.5 and results['B']['p_value'] < 0.05

    if c_has_jump and not b_has_jump:
        verdict = "STRUCTURAL_CLASSIFICATION_VALIDATED"
    elif b_has_jump:
        verdict = "DISCRETIZATION_ARTIFACT"
    else:
        verdict = "NO_EFFECT"

    results['verdict'] = verdict

    return results


# ============================================================================
# DBSCAN ROBUSTNESS SWEEP
# ============================================================================

def run_dbscan_sweep(alphas: np.ndarray, n_runs: int = 10, n_workers: int = 4, n_boids: int = 500) -> pd.DataFrame:
    """Run DBSCAN parameter sweep."""

    dbscan_params = [
        (10, 3), (10, 5), (10, 10),
        (15, 3), (15, 5), (15, 10),  # (15, 5) is default
        (20, 3), (20, 5), (20, 10),
    ]

    all_results = []

    for eps, min_samples in dbscan_params:
        print(f"\n=== DBSCAN eps={eps}, min_samples={min_samples} ===")
        df = run_alpha_sweep(alphas, n_runs=n_runs, eps=eps, min_samples=min_samples, n_workers=n_workers, n_boids=n_boids)
        all_results.append(df)

    return pd.concat(all_results, ignore_index=True)


# ============================================================================
# PLOTTING
# ============================================================================

def generate_plots(df: pd.DataFrame, analysis: Dict, output_dir: str):
    """Generate publication-quality plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    # Filter to default DBSCAN params
    df_default = df[(df['eps'] == 15) & (df['min_samples'] == 5)]

    if len(df_default) == 0:
        df_default = df

    # Aggregate by alpha
    grouped = df_default.groupby('alpha').agg({
        'polarization_mean': 'mean',
        'mi_B': ['mean', 'std'],
        'mi_C': ['mean', 'std'],
        'mi_D': ['mean', 'std'],
    })

    alphas = grouped.index.values

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: Continuous Polarization
    ax = axes[0, 0]
    pol_mean = grouped['polarization_mean'].values
    ax.plot(alphas, pol_mean, 'b-', linewidth=2)
    ax.set_xlabel('Alignment Strength (α)', fontsize=12)
    ax.set_ylabel('Polarization', fontsize=12)
    ax.set_title('A: Polarization vs α (Continuous)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Panel B: Binned Polarization Predictability
    ax = axes[0, 1]
    mi_mean = grouped['mi_B']['mean'].values
    mi_std = grouped['mi_B']['std'].values
    ax.plot(alphas, mi_mean, 'g-', linewidth=2)
    ax.fill_between(alphas, mi_mean - mi_std, mi_mean + mi_std, alpha=0.3, color='green')
    if 'B' in analysis and analysis['B']['critical_alpha']:
        ax.axvline(analysis['B']['critical_alpha'], color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Alignment Strength (α)', fontsize=12)
    ax.set_ylabel('I(B_{t+1}; B_t) [bits]', fontsize=12)
    ax.set_title('B: Binned Polarization Predictability', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Panel C: (F, Mode) Predictability - KEY TEST
    ax = axes[1, 0]
    mi_mean = grouped['mi_C']['mean'].values
    mi_std = grouped['mi_C']['std'].values
    ax.plot(alphas, mi_mean, 'orange', linewidth=2)
    ax.fill_between(alphas, mi_mean - mi_std, mi_mean + mi_std, alpha=0.3, color='orange')
    if 'C' in analysis and analysis['C']['critical_alpha']:
        ax.axvline(analysis['C']['critical_alpha'], color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Alignment Strength (α)', fontsize=12)
    ax.set_ylabel('I(C_{t+1}; C_t) [bits]', fontsize=12)
    ax.set_title('C: (n_flocks, Mode) Predictability ← KEY TEST', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Panel D: Full (F, Mode, Coherence) Predictability
    ax = axes[1, 1]
    mi_mean = grouped['mi_D']['mean'].values
    mi_std = grouped['mi_D']['std'].values
    ax.plot(alphas, mi_mean, 'm-', linewidth=2)
    ax.fill_between(alphas, mi_mean - mi_std, mi_mean + mi_std, alpha=0.3, color='magenta')
    if 'D' in analysis and analysis['D']['critical_alpha']:
        ax.axvline(analysis['D']['critical_alpha'], color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Alignment Strength (α)', fontsize=12)
    ax.set_ylabel('I(D_{t+1}; D_t) [bits]', fontsize=12)
    ax.set_title('D: (n_flocks, Mode, Coherence) Predictability', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'four_panel_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Four-panel comparison saved to {output_dir}/four_panel_comparison.png")

    # DBSCAN robustness plot
    if 'eps' in df.columns and df['eps'].nunique() > 1:
        generate_dbscan_robustness_plot(df, output_dir)


def generate_dbscan_robustness_plot(df: pd.DataFrame, output_dir: str):
    """Generate DBSCAN parameter robustness plot."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # For each DBSCAN param combo, find critical alpha and Cohen's d
    results = []

    for (eps, min_samples), group in df.groupby(['eps', 'min_samples']):
        analysis = analyze_condition(group, 'C', 'mi_C')
        results.append({
            'eps': eps,
            'min_samples': min_samples,
            'critical_alpha': analysis['critical_alpha'],
            'cohens_d': analysis['cohens_d']
        })

    results_df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Critical alpha by params
    ax = axes[0]
    for eps in results_df['eps'].unique():
        subset = results_df[results_df['eps'] == eps]
        ax.plot(subset['min_samples'], subset['critical_alpha'], 'o-', label=f'eps={eps}')
    ax.set_xlabel('min_samples')
    ax.set_ylabel('Critical α')
    ax.set_title('Critical Point vs DBSCAN Parameters')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cohen's d by params
    ax = axes[1]
    for eps in results_df['eps'].unique():
        subset = results_df[results_df['eps'] == eps]
        ax.plot(subset['min_samples'], subset['cohens_d'], 's-', label=f'eps={eps}')
    ax.set_xlabel('min_samples')
    ax.set_ylabel("Cohen's d")
    ax.set_title('Effect Size vs DBSCAN Parameters')
    ax.axhline(0.5, color='red', linestyle='--', label='Medium effect threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dbscan_robustness.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"DBSCAN robustness plot saved to {output_dir}/dbscan_robustness.png")


def generate_summary_tables(df: pd.DataFrame, analysis: Dict, output_dir: str):
    """Generate summary tables as CSV and markdown."""

    # Table 1: Summary Statistics
    table1_rows = []
    for cond in ['B', 'C', 'D']:
        if cond in analysis:
            a = analysis[cond]
            table1_rows.append({
                'Condition': cond,
                'Changepoint_alpha': f"{a['critical_alpha']:.3f}" if a['critical_alpha'] else "N/A",
                'Jump_Magnitude': f"{a['jump_magnitude']:.4f}",
                'Cohens_d': f"{a['cohens_d']:.3f}",
                'Cohens_d_95CI': f"[{a['cohens_d_ci_low']:.3f}, {a['cohens_d_ci_high']:.3f}]",
                'p_value': f"{a['p_value']:.4f}"
            })

    table1_df = pd.DataFrame(table1_rows)
    table1_df.to_csv(os.path.join(output_dir, 'summary_statistics.csv'), index=False)

    # Table 2: DBSCAN Robustness
    if 'eps' in df.columns and df['eps'].nunique() > 1:
        table2_rows = []
        for (eps, min_samples), group in df.groupby(['eps', 'min_samples']):
            a = analyze_condition(group, 'C', 'mi_C')
            table2_rows.append({
                'eps': eps,
                'min_samples': min_samples,
                'critical_alpha': f"{a['critical_alpha']:.3f}" if a['critical_alpha'] else "N/A",
                'cohens_d': f"{a['cohens_d']:.3f}"
            })

        table2_df = pd.DataFrame(table2_rows)
        table2_df.to_csv(os.path.join(output_dir, 'dbscan_robustness.csv'), index=False)

    print(f"Summary tables saved to {output_dir}/")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run the full revised FCC experiment."""

    import argparse
    parser = argparse.ArgumentParser(description='Revised FCC Validation Experiment')
    parser.add_argument('--output-dir', default=None, help='Output directory (default: auto-versioned)')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--runs', type=int, default=20, help='Runs per alpha (default: 20)')
    parser.add_argument('--boids', type=int, default=500, help='Number of boids (default: 500)')
    args = parser.parse_args()

    # Try to load config.yaml for defaults
    try:
        import yaml
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        n_boids = args.boids if args.boids != 500 else cfg.get('boids', {}).get('n_boids', 500)
        n_workers = args.workers or cfg.get('sweep', {}).get('n_workers', 4)
    except Exception:
        n_boids = args.boids
        n_workers = args.workers or 4

    # Versioned output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        run_id = datetime.now().strftime('revised_%Y%m%d_%H%M%S')
        output_dir = os.path.join(os.path.dirname(__file__), 'results', run_id)
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("REVISED FCC VALIDATION EXPERIMENT")
    print("="*60)
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"Output: {output_dir}")
    print()

    # Parameters
    alphas = np.arange(0.0, 0.31, 0.01)  # Focus on transition region
    n_runs = args.runs

    print(f"N boids: {n_boids}")
    print(f"Alpha sweep: {alphas[0]:.2f} to {alphas[-1]:.2f}, {len(alphas)} steps")
    print(f"Runs per alpha: {n_runs}")
    print(f"Workers: {n_workers}")
    print()

    # Phase 1: Main sweep with default DBSCAN
    print("PHASE 1: Main Alpha Sweep (default DBSCAN eps=15, min_samples=5)")
    print("-"*60)
    df_main = run_alpha_sweep(alphas, n_runs=n_runs, eps=15.0, min_samples=5, n_workers=n_workers, n_boids=n_boids)

    # Analyze main results
    print("\nAnalyzing results...")
    analysis = run_full_analysis(df_main)

    print("\n" + "="*60)
    print("MAIN RESULTS")
    print("="*60)

    for cond in ['B', 'C', 'D']:
        if cond in analysis:
            a = analysis[cond]
            print(f"\nCondition {cond}: {a['condition']}")
            print(f"  Critical α: {a['critical_alpha']:.3f}" if a['critical_alpha'] else "  Critical α: N/A")
            print(f"  Jump magnitude: {a['jump_magnitude']:.4f}")
            print(f"  Cohen's d: {a['cohens_d']:.3f} [{a['cohens_d_ci_low']:.3f}, {a['cohens_d_ci_high']:.3f}]")
            print(f"  p-value: {a['p_value']:.4f}")

    print(f"\n>>> VERDICT: {analysis['verdict']} <<<")

    # Phase 2: DBSCAN robustness sweep
    print("\n" + "="*60)
    print("PHASE 2: DBSCAN Parameter Robustness Sweep")
    print("="*60)

    df_dbscan = run_dbscan_sweep(alphas, n_runs=n_runs//2, n_workers=n_workers, n_boids=n_boids)

    # Combine results
    df_combined = pd.concat([df_main, df_dbscan], ignore_index=True)

    # Save raw data
    df_combined.to_csv(os.path.join(output_dir, 'fcc_revised_results.csv'), index=False)

    # Generate plots
    print("\nGenerating plots...")
    generate_plots(df_combined, analysis, output_dir)

    # Generate summary tables
    generate_summary_tables(df_combined, analysis, output_dir)

    # Save verdict
    verdict_data = {
        'timestamp': datetime.now().isoformat(),
        'verdict': analysis['verdict'],
        'conditions': {k: v for k, v in analysis.items() if k != 'verdict'},
        'parameters': {
            'alphas': alphas.tolist(),
            'n_runs': n_runs,
            'n_steps': 2000,
            'warmup': 500,
            'n_boids': n_boids
        }
    }

    with open(os.path.join(output_dir, 'verdict.json'), 'w') as f:
        json.dump(verdict_data, f, indent=2, default=str)

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}/")
    print(f"  - fcc_revised_results.csv (raw data)")
    print(f"  - four_panel_comparison.png")
    print(f"  - dbscan_robustness.png")
    print(f"  - summary_statistics.csv")
    print(f"  - dbscan_robustness.csv")
    print(f"  - verdict.json")
    print()
    print(f"FINAL VERDICT: {analysis['verdict']}")

    # Update results/latest symlink
    latest_link = os.path.join(os.path.dirname(__file__), 'results', 'latest')
    try:
        if os.path.exists(latest_link) or os.path.islink(latest_link):
            os.unlink(latest_link)
        os.symlink(os.path.abspath(output_dir), latest_link)
        print(f"results/latest → {output_dir}")
    except Exception as e:
        print(f"(Note: could not update results/latest symlink: {e})")

    return analysis


if __name__ == "__main__":
    main()
