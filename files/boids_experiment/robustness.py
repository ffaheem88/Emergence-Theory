#!/usr/bin/env python3
"""
Robustness checks for FCC paper (Appendix B).
Varies N and noise σ, checks if discontinuity persists.
Uses the same classification-based macro-state approach as the main experiment.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import time
from collections import Counter

# ---- Boids simulation (simplified, cohesion=0 per paper) ----

def run_boids(N, alpha, noise_std, n_steps=600, warmup=200, world_size=100.0,
              perception_radius=10.0, separation_weight=1.0, sep_radius=2.0, max_speed=2.0, dt=0.1):
    """Run boids (vectorized), return post-warmup macro-state time series."""
    positions = np.random.uniform(0, world_size, (N, 2))
    angles = np.random.uniform(0, 2*np.pi, N)
    speeds = np.random.uniform(0.5, 1.0, N) * max_speed
    velocities = np.column_stack([speeds * np.cos(angles), speeds * np.sin(angles)])

    macro_states = []
    half = world_size / 2

    for step in range(n_steps):
        # Vectorized pairwise distances (periodic)
        # diff[i,j] = positions[j] - positions[i]
        diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]  # (N, N, 2)
        diff = np.where(diff > half, diff - world_size, diff)
        diff = np.where(diff < -half, diff + world_size, diff)
        dists = np.linalg.norm(diff, axis=2)  # (N, N)

        # Neighbor masks
        np.fill_diagonal(dists, np.inf)  # exclude self
        neighbor_mask = dists < perception_radius  # (N, N)
        close_mask = dists < sep_radius  # (N, N)

        # Alignment: mean neighbor velocity - own velocity
        n_counts = neighbor_mask.sum(axis=1, keepdims=True).clip(min=1)  # (N, 1)
        # Sum of neighbor velocities
        # neighbor_mask (N,N) x velocities (N,2) -> weighted sum
        avg_vel = (neighbor_mask[:, :, np.newaxis].astype(float) * velocities[np.newaxis, :, :]).sum(axis=1) / n_counts
        alignment = alpha * (avg_vel - velocities) * (neighbor_mask.any(axis=1, keepdims=True)).astype(float)

        # Separation: sum of away-vectors / dist^2 for close neighbors
        # away = -diff (away from neighbor j)
        inv_dist2 = np.where(close_mask & (dists > 0), 1.0 / (dists * dists), 0.0)  # (N, N)
        separation = separation_weight * (-diff * inv_dist2[:, :, np.newaxis]).sum(axis=1)  # (N, 2)

        delta_v = alignment + separation + np.random.normal(0, noise_std, (N, 2))
        velocities = velocities + delta_v * dt

        # Limit speed
        spds = np.linalg.norm(velocities, axis=1, keepdims=True)
        too_fast = spds > max_speed
        velocities = np.where(too_fast, velocities * max_speed / spds, velocities)

        positions = (positions + velocities * dt) % world_size

        if step >= warmup:
            macro_states.append(classify_macro_state(positions, velocities, world_size))

    return macro_states


def classify_macro_state(positions, velocities, world_size):
    """Classify into (n_flocks, mode, coherence) per paper Appendix A."""
    N = len(positions)

    # DBSCAN-like clustering (simple version)
    eps = 15.0
    min_samples = 5
    labels = np.full(N, -1, dtype=int)
    cluster_id = 0
    visited = np.zeros(N, dtype=bool)

    for i in range(N):
        if visited[i]:
            continue
        diff = positions - positions[i]
        diff = np.where(diff > world_size/2, diff - world_size, diff)
        diff = np.where(diff < -world_size/2, diff + world_size, diff)
        dists = np.linalg.norm(diff, axis=1)
        neighbors = np.where(dists < eps)[0]
        if len(neighbors) < min_samples:
            continue
        # BFS expansion
        queue = list(neighbors)
        labels[neighbors] = cluster_id
        visited[i] = True
        qi = 0
        while qi < len(queue):
            j = queue[qi]; qi += 1
            if visited[j]:
                continue
            visited[j] = True
            diff2 = positions - positions[j]
            diff2 = np.where(diff2 > world_size/2, diff2 - world_size, diff2)
            diff2 = np.where(diff2 < -world_size/2, diff2 + world_size, diff2)
            dists2 = np.linalg.norm(diff2, axis=1)
            nb2 = np.where(dists2 < eps)[0]
            if len(nb2) >= min_samples:
                for k in nb2:
                    if labels[k] == -1:
                        labels[k] = cluster_id
                        queue.append(k)
        cluster_id += 1

    n_flocks = cluster_id

    # Polarization
    vnorms = np.linalg.norm(velocities, axis=1, keepdims=True)
    vnorms = np.where(vnorms > 0, vnorms, 1e-10)
    unit_v = velocities / vnorms
    polarization = np.linalg.norm(unit_v.mean(axis=0))

    # Mode
    clustered_frac = np.sum(labels >= 0) / N
    if clustered_frac < 0.5:
        mode = 'dispersed'
    elif polarization > 0.6:
        mode = 'migrating'
    else:
        mode = 'forming'

    # Coherence
    if polarization < 0.3:
        coherence = 'low'
    elif polarization < 0.6:
        coherence = 'medium'
    else:
        coherence = 'high'

    return (n_flocks, mode, coherence)


def compute_predictability(macro_states):
    """Compute I(C_{t+1}; C_t) = H(C_{t+1}) - H(C_{t+1}|C_t)."""
    if len(macro_states) < 2:
        return 0.0

    # Count transitions
    transitions = Counter()
    state_counts = Counter()
    for i in range(len(macro_states) - 1):
        s = macro_states[i]
        s_next = macro_states[i + 1]
        transitions[(s, s_next)] += 1
        state_counts[s] += 1

    total = sum(state_counts.values())
    if total == 0:
        return 0.0

    # H(C_{t+1}) - marginal entropy of next states
    next_counts = Counter()
    for i in range(1, len(macro_states)):
        next_counts[macro_states[i]] += 1
    total_next = sum(next_counts.values())
    h_next = -sum((c/total_next) * np.log2(c/total_next) for c in next_counts.values() if c > 0)

    # H(C_{t+1} | C_t) - conditional
    h_cond = 0.0
    for s, count_s in state_counts.items():
        p_s = count_s / total
        # transitions from s
        trans_from_s = {k[1]: v for k, v in transitions.items() if k[0] == s}
        total_from_s = sum(trans_from_s.values())
        if total_from_s > 0:
            h_s = -sum((c/total_from_s) * np.log2(c/total_from_s) for c in trans_from_s.values() if c > 0)
            h_cond += p_s * h_s

    return max(0, h_next - h_cond)


def unique_states(macro_states):
    return len(set(macro_states))


def run_robustness_sweep(N, noise, alpha_values, n_runs=3):
    """Run sweep for given N and noise."""
    results = []
    for alpha in alpha_values:
        preds = []
        states = []
        for run in range(n_runs):
            ms = run_boids(N, alpha, noise)
            preds.append(compute_predictability(ms))
            states.append(unique_states(ms))
        results.append({
            'alpha': alpha,
            'pred_mean': np.mean(preds),
            'pred_std': np.std(preds),
            'states_mean': np.mean(states),
            'states_std': np.std(states),
        })
        print(f"  N={N}, σ={noise}, α={alpha:.3f}: pred={np.mean(preds):.3f}±{np.std(preds):.3f}, states={np.mean(states):.1f}", flush=True)
    return results


def main():
    # Alpha values focused on critical region (reduced for tractability)
    alpha_values = [0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.2, 0.5]

    # Robustness conditions
    conditions = [
        # Vary N (noise=0.1 fixed)
        (100, 0.1),
        (250, 0.1),
        (500, 0.1),  # baseline
        # Vary noise (N=500 fixed)
        (500, 0.05),
        (500, 0.2),
    ]
    # N=1000 would OOM with vectorized (1000x1000x2), skip for now

    all_results = {}
    t0 = time.time()

    for N, noise in conditions:
        key = f"N={N}_sigma={noise}"
        print(f"\n{'='*50}")
        print(f"Running: {key}", flush=True)
        print(f"{'='*50}", flush=True)
        results = run_robustness_sweep(N, noise, alpha_values, n_runs=5)
        all_results[key] = results

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")

    # Save results
    with open('results/robustness_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Generate robustness figure
    plot_robustness(all_results)


def plot_robustness(all_results):
    """Generate robustness figure with two panels: vary N and vary noise."""
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 12,
        'axes.titlesize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'axes.linewidth': 0.8,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel (a): Vary N
    colors_n = {'100': '#E69F00', '250': '#56B4E9', '500': '#009E73'}
    for N in [100, 250, 500]:
        key = f"N={N}_sigma=0.1"
        data = all_results[key]
        alphas = [r['alpha'] for r in data]
        preds = [r['pred_mean'] for r in data]
        stds = [r['pred_std'] for r in data]
        c = colors_n[str(N)]
        ax1.plot(alphas, preds, '-o', color=c, linewidth=1.8, markersize=4, label=f'N={N}',
                 markeredgecolor='white', markeredgewidth=0.4)
        ax1.fill_between(alphas, np.array(preds)-np.array(stds), np.array(preds)+np.array(stds),
                          alpha=0.15, color=c, linewidth=0)

    ax1.set_xlabel(r'Alignment strength $\alpha$')
    ax1.set_ylabel('Macro-state predictability (bits)')
    ax1.set_title(r'$\bf{(a)}$ Varying population size ($\sigma=0.1$)')
    ax1.legend(loc='upper right', framealpha=0.9, edgecolor='#ccc', fontsize=9)
    ax1.grid(True, alpha=0.15)
    ax1.set_xlim(-0.01, 0.52)
    ax1.set_ylim(-0.05, 2.0)

    # Panel (b): Vary noise
    colors_s = {'0.05': '#CC79A7', '0.1': '#009E73', '0.2': '#0072B2'}
    for noise in [0.05, 0.1, 0.2]:
        key = f"N=500_sigma={noise}"
        data = all_results[key]
        alphas = [r['alpha'] for r in data]
        preds = [r['pred_mean'] for r in data]
        stds = [r['pred_std'] for r in data]
        c = colors_s[str(noise)]
        ax2.plot(alphas, preds, '-s', color=c, linewidth=1.8, markersize=4, label=f'σ={noise}',
                 markeredgecolor='white', markeredgewidth=0.4)
        ax2.fill_between(alphas, np.array(preds)-np.array(stds), np.array(preds)+np.array(stds),
                          alpha=0.15, color=c, linewidth=0)

    ax2.set_xlabel(r'Alignment strength $\alpha$')
    ax2.set_ylabel('Macro-state predictability (bits)')
    ax2.set_title(r'$\bf{(b)}$ Varying noise ($N=500$)')
    ax2.legend(loc='upper right', framealpha=0.9, edgecolor='#ccc', fontsize=9)
    ax2.grid(True, alpha=0.15)
    ax2.set_xlim(-0.01, 0.52)
    ax2.set_ylim(-0.05, 2.0)

    plt.tight_layout(w_pad=2.0)
    plt.savefig('results/robustness.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('results/robustness.pdf', bbox_inches='tight', facecolor='white')
    print("Saved results/robustness.png and results/robustness.pdf")


if __name__ == '__main__':
    main()
