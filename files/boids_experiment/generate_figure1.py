#!/usr/bin/env python3
"""Generate publication-quality Figure 1 for the FCC paper (v3 - with inset)."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import json

with open('results/fcc_phase_diagram_data.json') as f:
    d = json.load(f)

alphas = np.array(d['alphas'])
pred_mean = np.array(d['pred_mean'])
pred_std = np.array(d['pred_std'])
pol_mean = np.array(d['pol_mean'])
pol_std = np.array(d['pol_std'])
states_mean = np.array(d['states_mean'])

# Colorblind-safe palette
c_pred = '#0072B2'
c_pol = '#D55E00'
c_states = '#009E73'
c_crit = '#CC79A7'

crit_lo, crit_hi = 0.04, 0.05
marker_kw = dict(markeredgecolor='white', markeredgewidth=0.5)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.linewidth': 0.8,
})

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2))

# ==================== Panel (a): Predictability ====================
ax1.fill_between(alphas, pred_mean - pred_std, pred_mean + pred_std,
                  alpha=0.2, color=c_pred, linewidth=0)
ax1.plot(alphas, pred_mean, '-o', color=c_pred, linewidth=2, markersize=4.5, zorder=5, **marker_kw)
ax1.axvspan(crit_lo, crit_hi, alpha=0.12, color=c_crit, zorder=0)
ax1.axvline(0.045, color=c_crit, linestyle='--', linewidth=1.3, alpha=0.8)

ax1.set_xlabel(r'Alignment strength $\alpha$')
ax1.set_ylabel('Macro-state predictability (bits)')
ax1.set_title(r'$\bf{(a)}$ Classification-based emergence signal')
ax1.set_xlim(-0.01, 1.02)
ax1.set_ylim(-0.05, 1.6)
ax1.grid(True, alpha=0.15, linewidth=0.5)

from matplotlib.lines import Line2D
ax1.legend(handles=[Line2D([0], [0], color=c_crit, linestyle='--', linewidth=1.3,
           label=r'Critical $\alpha_c \approx 0.045$')],
           loc='upper right', framealpha=0.9, edgecolor='#cccccc', fontsize=9)

# Inset: zoom on critical region
axins = ax1.inset_axes([0.35, 0.55, 0.4, 0.4])  # [x, y, width, height] in axes coords
mask = alphas <= 0.12
axins.fill_between(alphas[mask], (pred_mean - pred_std)[mask], (pred_mean + pred_std)[mask],
                    alpha=0.2, color=c_pred, linewidth=0)
axins.plot(alphas[mask], pred_mean[mask], '-o', color=c_pred, linewidth=2, markersize=5, zorder=5, **marker_kw)
axins.axvspan(crit_lo, crit_hi, alpha=0.15, color=c_crit, zorder=0)
axins.axvline(0.045, color=c_crit, linestyle='--', linewidth=1, alpha=0.7)
axins.set_xlim(-0.005, 0.125)
axins.set_ylim(-0.05, 1.5)
axins.set_xlabel(r'$\alpha$', fontsize=8)
axins.tick_params(labelsize=8)
axins.grid(True, alpha=0.15, linewidth=0.4)
axins.set_title('Critical region', fontsize=8, pad=3)

# Arrow showing the jump magnitude
axins.annotate('', xy=(0.048, 0.78), xytext=(0.048, 0.0),
               arrowprops=dict(arrowstyle='<->', color=c_crit, lw=1.8))
axins.text(0.065, 0.35, r'$\Delta \approx 0.78$' + '\nbits', fontsize=7.5,
           color=c_crit, fontstyle='italic', ha='left',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

ax1.indicate_inset_zoom(axins, edgecolor='#888888', linewidth=0.8)

# ==================== Panel (b): Polarization ====================
ax2.fill_between(alphas, pol_mean - pol_std, pol_mean + pol_std,
                  alpha=0.2, color=c_pol, linewidth=0)
ax2.plot(alphas, pol_mean, '-s', color=c_pol, linewidth=2, markersize=4.5, zorder=5, **marker_kw)
ax2.axvspan(crit_lo, crit_hi, alpha=0.12, color=c_crit, zorder=0)
ax2.axvline(0.045, color=c_crit, linestyle='--', linewidth=1.3, alpha=0.8)

ax2.set_xlabel(r'Alignment strength $\alpha$')
ax2.set_ylabel('Polarization')
ax2.set_title(r'$\bf{(b)}$ Aggregation-based order parameter')
ax2.set_xlim(-0.01, 1.02)
ax2.set_ylim(-0.02, 0.5)
ax2.grid(True, alpha=0.15, linewidth=0.5)

# ==================== Panel (c): Unique States (step plot) ====================
ax3.step(alphas, states_mean, where='mid', color=c_states, linewidth=2, zorder=5)
ax3.plot(alphas, states_mean, 'D', color=c_states, markersize=5, zorder=6, **marker_kw)
ax3.fill_between(alphas, 0, states_mean, alpha=0.1, color=c_states, step='mid', linewidth=0)
ax3.axvspan(crit_lo, crit_hi, alpha=0.12, color=c_crit, zorder=0)
ax3.axvline(0.045, color=c_crit, linestyle='--', linewidth=1.3, alpha=0.8)

ax3.set_xlabel(r'Alignment strength $\alpha$')
ax3.set_ylabel('Unique macro-states')
ax3.set_title(r'$\bf{(c)}$ State space richness')
ax3.set_xlim(-0.01, 1.02)
ax3.set_ylim(0, 10)
ax3.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax3.grid(True, alpha=0.15, linewidth=0.5)

plt.tight_layout(w_pad=1.8)
plt.savefig('results/figure1.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('results/figure1.pdf', bbox_inches='tight', facecolor='white')
print("Saved figure1.png and figure1.pdf")
