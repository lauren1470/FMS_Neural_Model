"""
Sensitivity analysis: mean WDR firing rate across a 7 × 5 NMDA × GABA grid.

Runs N_SEEDS replicates per parameter combination and saves results as CSV/JSON.
Produces a heatmap with healthy, FMS, and intervention operating points marked.

Usage:
    python src/generate_sensitivity_analysis.py
"""

import os
import sys
import json
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Ensure src/ is on the path when running this script directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulations import run_simulation
from synapses import PathologyStates

matplotlib.rcParams['font.family'] = 'Arial'

# --- Configuration ---
NMDA_MULTIPLIERS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
GABA_FACTORS     = [1.0, 0.8, 0.6, 0.4, 0.2]
N_SEEDS          = 5
BASE_SEED        = 42
DURATION_MS      = 1000   # ms per simulation trial

# Project colour palette
BLUE   = '#4472C4'
PINK   = '#E91E63'
PURPLE = '#9B59B6'
WHITE  = '#FFFFFF'

os.makedirs('results/sensitivity', exist_ok=True)
os.makedirs('figures/sensitivity', exist_ok=True)

# --- 1. Grid sweep ---
n_nmda  = len(NMDA_MULTIPLIERS)
n_gaba  = len(GABA_FACTORS)
n_cells = n_nmda * n_gaba

print('=' * 65)
print('  FMS SENSITIVITY ANALYSIS')
print(f'  Grid  : {n_nmda} NMDA levels × {n_gaba} GABA levels = {n_cells} cells')
print(f'  Seeds : {N_SEEDS} per cell  →  {n_cells * N_SEEDS} total runs')
print(f'  Duration per run: {DURATION_MS} ms')
print('=' * 65)

t_start  = time.time()
records  = []
run_idx  = 0

for nmda_mult in NMDA_MULTIPLIERS:
    for gaba_fac in GABA_FACTORS:
        run_idx += 1
        rates = []

        for s in range(N_SEEDS):
            seed  = BASE_SEED + s
            state = PathologyStates.custom(nmda_mult, gaba_fac)
            res   = run_simulation(
                state=state,
                duration_ms=DURATION_MS,
                seed=seed,
                verbose=False,
            )
            rates.append(res['wdr_mean_rate'])

        mean_rate = float(np.mean(rates))
        std_rate  = float(np.std(rates))

        records.append({
            'nmda_multiplier':    nmda_mult,
            'gaba_factor':        gaba_fac,
            'wdr_mean_rate_mean': mean_rate,
            'wdr_mean_rate_std':  std_rate,
            'per_seed_rates':     [float(r) for r in rates],
        })

        elapsed  = time.time() - t_start
        avg_secs = elapsed / run_idx
        remaining = avg_secs * (n_cells - run_idx)
        print(f'  [{run_idx:2d}/{n_cells}]  NMDA={nmda_mult:.1f}×  GABA={gaba_fac:.1f}×  '
              f'WDR = {mean_rate:6.2f} ± {std_rate:.2f} Hz  '
              f'(~{remaining/60:.1f} min remaining)')

total_elapsed = time.time() - t_start
print(f'\nSweep complete in {total_elapsed/60:.1f} min.\n')


# --- 2. Save results ---

# CSV (without per-seed breakdown for readability)
df = pd.DataFrame([
    {k: v for k, v in r.items() if k != 'per_seed_rates'}
    for r in records
])
csv_path = 'results/sensitivity/sensitivity_results.csv'
df.to_csv(csv_path, index=False)
print(f'Saved: {csv_path}')

# JSON (full detail including per-seed rates)
json_data = {
    'config': {
        'nmda_multipliers': NMDA_MULTIPLIERS,
        'gaba_factors':     GABA_FACTORS,
        'n_seeds':          N_SEEDS,
        'base_seed':        BASE_SEED,
        'duration_ms':      DURATION_MS,
    },
    'results': records,
}
json_path = 'results/sensitivity/sensitivity_results.json'
with open(json_path, 'w') as f:
    json.dump(json_data, f, indent=2)
print(f'Saved: {json_path}')


# --- 3. Build heatmap matrix (rows = NMDA, cols = GABA) ---
heat_mean = np.zeros((n_nmda, n_gaba))

for rec in records:
    ri = NMDA_MULTIPLIERS.index(rec['nmda_multiplier'])
    ci = GABA_FACTORS.index(rec['gaba_factor'])
    heat_mean[ri, ci] = rec['wdr_mean_rate_mean']

max_val = float(heat_mean.max())
norm_val = max_val if max_val > 0 else 1.0


# --- 4. Plot heatmap ---

# Custom colormap: BLUE (low / healthy-like) → WHITE → PINK (high / FMS-like)
cmap = mcolors.LinearSegmentedColormap.from_list(
    'fms_sensitivity', [BLUE, WHITE, PINK], N=256
)

fig, ax = plt.subplots(figsize=(7.5, 5.5))
fig.patch.set_facecolor('white')

im = ax.imshow(
    heat_mean, cmap=cmap, aspect='auto', origin='lower',
    vmin=0, vmax=norm_val,
)

# --- Anchor cell positions ---
# Defined here so the annotation loop can reference them.
h_row      = NMDA_MULTIPLIERS.index(1.0)
h_col      = GABA_FACTORS.index(1.0)
f_row      = NMDA_MULTIPLIERS.index(3.0)
f_col      = GABA_FACTORS.index(0.4)
int_row    = NMDA_MULTIPLIERS.index(2.0)
int_col_cell = 1   # nearest cell column to GABA=0.76 (between 0.8 and 0.6)

# Maps (ci, ri) → (marker_style, marker_colour) for the three anchor cells
anchor_markers = {
    (h_col,        h_row):   ('s', BLUE),
    (f_col,        f_row):   ('s', PINK),
    (int_col_cell, int_row): ('^', PURPLE),
}

# --- Cell value annotations ---
# Anchor cells: marker symbol to the left, number to the right.
# All other cells: number centred as normal.
for ri in range(n_nmda):
    for ci in range(n_gaba):
        val = heat_mean[ri, ci]
        if (ci, ri) in anchor_markers:
            marker_style, marker_colour = anchor_markers[(ci, ri)]
            ax.plot(ci - 0.28, ri, marker_style,
                    color=marker_colour, markersize=11,
                    markeredgecolor='white', markeredgewidth=1.5,
                    zorder=6)
            ax.text(ci, ri, f'{val:.1f}',
                    ha='center', va='center',
                    fontsize=8.5, color='#111111', fontweight='bold',
                    zorder=7)
        else:
            ax.text(ci, ri, f'{val:.1f}',
                    ha='center', va='center',
                    fontsize=8.5, color='#111111', fontweight='bold',
                    zorder=7)

# --- Axis ticks and labels ---
ax.set_xticks(range(n_gaba))
ax.set_xticklabels([f'{g:.1f}' for g in GABA_FACTORS], fontsize=10)
ax.set_yticks(range(n_nmda))
ax.set_yticklabels([f'{n:.1f}' for n in NMDA_MULTIPLIERS], fontsize=10)
ax.set_xlabel('GABA-A weight factor (relative to healthy)', fontsize=11, labelpad=6)
ax.set_ylabel('NMDA weight multiplier (relative to healthy)', fontsize=11, labelpad=6)

# --- Legend-only marker entries (not plotted in the axes) ---
ax.plot([], [], 's', color=BLUE,   markersize=14,
        markeredgecolor='white', markeredgewidth=2.0,
        label='Healthy  (NMDA=1.0×, GABA-A=1.0×)')
ax.plot([], [], 's', color=PINK,   markersize=14,
        markeredgecolor='white', markeredgewidth=2.0,
        label='FMS  (NMDA=3.0×, GABA-A=0.4×)')
ax.plot([], [], '^', color=PURPLE, markersize=14,
        markeredgecolor='white', markeredgewidth=2.0,
        label='Intervention  (NMDA=2.0×, GABA-A=0.76×)')

# --- Colorbar ---
cbar = plt.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
cbar.set_label('Mean WDR firing rate (Hz)', fontsize=10)

ax.set_title(
    'Sensitivity Analysis: WDR Firing Rate across NMDA × GABA-A Parameter Space',
    fontsize=11, fontweight='bold', pad=10,
)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.18),
    ncol=3,
    fontsize=9,
    framealpha=0.90,
    edgecolor='#cccccc',
    markerscale=0.75,
)

plt.tight_layout()
png_path = 'figures/sensitivity/sensitivity_heatmap.png'
fig.savefig(png_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved: {png_path}')
