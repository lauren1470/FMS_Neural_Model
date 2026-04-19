import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import json
import os

matplotlib.rcParams['font.family'] = 'Arial'

# --- Project colour palette ---
BLUE        = '#4472C4'
PINK        = '#E91E63'
PURPLE      = '#9B59B6'
BLUE_LIGHT  = '#D6E8F9'
PINK_LIGHT  = '#FDD0E3'
HEADER_BG   = '#2D2D2D'
ROW_ALT     = '#F7F7F7'
WHITE       = '#FFFFFF'

os.makedirs('figures/tables', exist_ok=True)


def draw_table(ax, col_labels, col_widths, col_colours, rows, row_data_colours,
               fontsize=11, header_fontsize=11):
    """
    Draw a styled table on ax.
    col_labels      : list of column header strings
    col_widths      : list of relative widths (must sum to 1.0)
    col_colours     : list of header background colours per column
    rows            : list of row lists (each row is a list of cell strings)
    row_data_colours: list of colours for each data row background
                      (list of lists matching rows x cols, or list of single colour per row)
    """
    n_cols = len(col_labels)
    n_rows = len(rows)
    row_h  = 1.0 / (n_rows + 1)   # +1 for header

    # --- Draw header ---
    x = 0.0
    for i, (label, w, hcol) in enumerate(zip(col_labels, col_widths, col_colours)):
        rect = mpatches.FancyBboxPatch(
            (x, 1 - row_h), w, row_h,
            boxstyle='square,pad=0', linewidth=0.8,
            facecolor=hcol, edgecolor='white', zorder=2
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, 1 - row_h / 2, label,
                ha='center', va='center', fontsize=header_fontsize,
                fontweight='bold', color='white', zorder=3)
        x += w

    # --- Draw data rows ---
    for r, row in enumerate(rows):
        y_top = 1 - row_h - r * row_h
        x = 0.0
        for c, (cell, w) in enumerate(zip(row, col_widths)):
            if isinstance(row_data_colours[r], list):
                bg = row_data_colours[r][c]
            else:
                bg = row_data_colours[r]
            rect = mpatches.FancyBboxPatch(
                (x, y_top - row_h), w, row_h,
                boxstyle='square,pad=0', linewidth=0.8,
                facecolor=bg, edgecolor='white', zorder=2
            )
            ax.add_patch(rect)
            ax.text(x + w / 2, y_top - row_h / 2, cell,
                    ha='center', va='center', fontsize=fontsize,
                    color='#111111', zorder=3)
            x += w

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


# ===========================================================================
# TABLE 1 — Feature Summary Statistics
# ===========================================================================
df = pd.read_csv('data/dataset.csv')
h  = df[df.label == 'healthy']
m  = df[df.label == 'fibromyalgia']

features = [
    ('wdr_mean_rate',       'WDR mean rate',       'Hz'),
    ('gaba_mean_rate',      'GABA mean rate',       'Hz'),
    ('wdr_peak_rate',       'WDR peak rate',        'Hz'),
    ('wdr_windup_ratio',    'WDR wind-up ratio',    'n/a'),
    ('wdr_evoked_response', 'WDR evoked response',  'spikes/neuron'),
    ('wdr_isi_mean',        'WDR ISI mean',         'ms'),
    ('wdr_isi_cv',          'WDR ISI CV',           'n/a'),
    ('wdr_burst_count',     'WDR burst count',      'count'),
    ('wdr_burst_fraction',  'WDR burst fraction',   'n/a'),
    ('wdr_active_fraction', 'WDR active fraction',  'n/a'),
    ('ei_ratio',            'E/I ratio',            'n/a'),
]

rows_t1 = []
row_cols_t1 = []
for i, (col, name, unit) in enumerate(features):
    hm, hs = h[col].mean(), h[col].std()
    mm, ms = m[col].mean(), m[col].std()
    rows_t1.append([name, unit,
                    f'{hm:.2f} \u00b1 {hs:.2f}',
                    f'{mm:.2f} \u00b1 {ms:.2f}'])
    bg = ROW_ALT if i % 2 == 0 else WHITE
    row_cols_t1.append([bg, bg, BLUE_LIGHT, PINK_LIGHT])

fig1, ax1 = plt.subplots(figsize=(7.5, 3.9))
fig1.patch.set_facecolor('white')
draw_table(
    ax1,
    col_labels   = ['Feature', 'Units', 'Healthy (mean \u00b1 SD)', 'FMS (mean \u00b1 SD)'],
    col_widths   = [0.34, 0.16, 0.25, 0.25],
    col_colours  = [HEADER_BG, HEADER_BG, BLUE, PINK],
    rows         = rows_t1,
    row_data_colours = row_cols_t1,
    fontsize=10.5, header_fontsize=11
)
ax1.text(0.0, -0.02, 'n = 500 trials per state.',
         transform=ax1.transAxes, fontsize=9.5,
         color='#444444', style='italic', va='top')
plt.tight_layout(pad=0.3)
fig1.savefig('figures/tables/table_feature_summary.png', dpi=200,
             bbox_inches='tight', facecolor='white')
plt.close(fig1)
print('Saved: figures/tables/table_feature_summary.png')


# ===========================================================================
# TABLE 2 — Combined Classifier Performance (Initial vs Tuned)
# Row colours: RF rows blue-light, SVM rows pink-light
# ===========================================================================
PURPLE_LIGHT = '#EDD5F5'

rows_t2 = [
    ['Random Forest', 'Initial', '1.000', '1.000', '1.000', '1.000', '1.000 \u00b1 0.000'],
    ['Random Forest', 'Tuned',   '1.000', '1.000', '1.000', '1.000', '1.000 \u00b1 0.000'],
    ['SVM-RBF',       'Initial', '1.000', '1.000', '1.000', '1.000', '0.997 \u00b1 0.006'],
    ['SVM-RBF',       'Tuned',   '1.000', '1.000', '1.000', '1.000', '1.000 \u00b1 0.000'],
]
row_cols_t2 = [BLUE_LIGHT, BLUE_LIGHT, PINK_LIGHT, PINK_LIGHT]

fig2, ax2 = plt.subplots(figsize=(7.5, 1.8))
fig2.patch.set_facecolor('white')
draw_table(
    ax2,
    col_labels  = ['Classifier', 'Setting', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV F1 (5-fold)'],
    col_widths  = [0.18, 0.10, 0.12, 0.12, 0.12, 0.12, 0.24],
    col_colours = [HEADER_BG, HEADER_BG, HEADER_BG, HEADER_BG, HEADER_BG, HEADER_BG, HEADER_BG],
    rows        = rows_t2,
    row_data_colours = row_cols_t2,
    fontsize=11, header_fontsize=11
)
ax2.text(0.0, -0.06, 'All metrics macro-averaged. CV F1 computed on training set (n = 700).',
         transform=ax2.transAxes, fontsize=9.5,
         color='#444444', style='italic', va='top')
plt.tight_layout(pad=0.3)
fig2.savefig('figures/tables/table_classifier_performance.png', dpi=200,
             bbox_inches='tight', facecolor='white')
plt.close(fig2)
print('Saved: figures/tables/table_classifier_performance.png')


# ===========================================================================
# TABLE 3 — Feature Importance (Tuned Random Forest)
# WDR features → BLUE_LIGHT rows  |  GABA features → PINK_LIGHT rows
# E/I ratio → ROW_ALT
# ===========================================================================
PURPLE_LIGHT = '#EDD5F5'

with open('results/tuned/classification_results.json') as f:
    tuned_results = json.load(f)

fi = tuned_results['random_forest']['feature_importance']

feature_names_map = {
    'wdr_windup_ratio':    'WDR wind-up ratio',
    'wdr_evoked_response': 'WDR evoked response',
    'wdr_burst_count':     'WDR burst count',
    'wdr_mean_rate':       'WDR mean rate',
    'wdr_peak_rate':       'WDR peak rate',
    'wdr_burst_fraction':  'WDR burst fraction',
    'wdr_total_spikes':    'WDR total spikes',
    'wdr_early_rate':      'WDR early rate',
    'wdr_active_fraction': 'WDR active fraction',
    'ei_ratio':            'E/I ratio',
    'wdr_isi_cv':          'WDR ISI CV',
    'wdr_late_rate':       'WDR late rate',
    'wdr_isi_std':         'WDR ISI std',
    'wdr_isi_mean':        'WDR ISI mean',
    'gaba_mean_rate':      'GABA mean rate',
    'gaba_total_spikes':   'GABA total spikes',
    'gaba_isi_std':        'GABA ISI std',
    'gaba_isi_cv':         'GABA ISI CV',
    'gaba_isi_mean':       'GABA ISI mean',
}

sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)

rows_t3 = []
row_cols_t3 = []
cumulative = 0.0
for rank, (feat, imp) in enumerate(sorted_fi, 1):
    cumulative += imp
    name = feature_names_map.get(feat, feat)
    rows_t3.append([
        str(rank),
        name,
        f'{imp:.4f}',
        f'{imp * 100:.2f}%',
        f'{cumulative * 100:.2f}%'
    ])
    if feat.startswith('gaba'):
        bg = PINK_LIGHT
    elif feat == 'ei_ratio':
        bg = ROW_ALT
    else:
        bg = BLUE_LIGHT
    row_cols_t3.append(bg)

fig3, ax3 = plt.subplots(figsize=(7.5, 6.4))
fig3.patch.set_facecolor('white')
draw_table(
    ax3,
    col_labels        = ['Rank', 'Feature', 'Importance', '% Total', 'Cumulative %'],
    col_widths        = [0.08, 0.34, 0.18, 0.16, 0.24],
    col_colours       = [HEADER_BG, HEADER_BG, HEADER_BG, HEADER_BG, HEADER_BG],
    rows              = rows_t3,
    row_data_colours  = row_cols_t3,
    fontsize=10, header_fontsize=10.5
)
ax3.text(0.0, -0.012,
         'Feature importances from tuned Random Forest (Gini impurity, n_estimators=100).\n'
         'Blue rows = WDR features; Pink rows = GABA features.',
         transform=ax3.transAxes, fontsize=9.5,
         color='#444444', style='italic', va='top')
plt.tight_layout(pad=0.3)
fig3.savefig('figures/tables/table_feature_importance.png', dpi=200,
             bbox_inches='tight', facecolor='white')
plt.close(fig3)
print('Saved: figures/tables/table_feature_importance.png')
