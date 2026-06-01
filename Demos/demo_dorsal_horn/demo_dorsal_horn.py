"""
demo_dorsal_horn.py  -  Animated spinal dorsal horn circuit diagram

Shows a stylised anatomical cross-section of the dorsal horn with:
  - C-fibre (nociceptive) and Abeta (tactile) inputs entering from the left
  - GABA interneurons in Lamina II flashing when they fire
  - WDR projection neurons in Lamina V flashing when they fire
  - Neurons that are currently firing glow brightly; resting neurons are dim
  - Live WDR firing rate output trace on the right
  - Side-by-side: Healthy vs FMS states using real simulation spike data

Usage:
    python demo_dorsal_horn.py           # interactive looping window
    python demo_dorsal_horn.py --save    # saves demo_dorsal_horn.mp4 (or .gif fallback)
"""

import sys
import os
import argparse
from collections import deque
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--save', action='store_true',
                    help='Save to demo_dorsal_horn.mp4/.gif instead of displaying')
args = parser.parse_args()

import matplotlib
if args.save:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon, Ellipse, Rectangle
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from simulations import run_simulation
from synapses import PathologyStates

# ── Colour palette ──────────────────────────────────────────────────────────────
BG      = '#FFF0F4'
BLUE    = '#4472C4'
PINK    = '#E91E63'
GREY    = '#7f8c8d'
TEXT    = '#2d2d3d'
GOLD    = '#8B1A4A'
PURPLE  = '#9B59B6'

# Lamina band fill colours
LAM_COLS = {
    'I':   '#FFD6E0',
    'II':  '#FFAABF',
    'III': '#FFD0DC',
    'IV':  '#FFDDE6',
    'V':   '#FFBBCC',
}

# ── Model constants ─────────────────────────────────────────────────────────────
N_WDR  = 50
N_GABA = 30

# ── Simulation parameters ────────────────────────────────────────────────────────
DURATION_MS = 3000
SEED        = 42

print('Running Healthy simulation ...')
healthy = run_simulation(state=PathologyStates.healthy(),
                         duration_ms=DURATION_MS, seed=SEED, verbose=False)
print('Running FMS simulation ...')
fms = run_simulation(state=PathologyStates.fibromyalgia(),
                     duration_ms=DURATION_MS, seed=SEED, verbose=False)
print('Simulations complete. Building animation ...')

h_wt = np.asarray(healthy['t']);      h_wi = np.asarray(healthy['i'])
h_gt = np.asarray(healthy['t_gaba']); h_gi = np.asarray(healthy['i_gaba'])
f_wt = np.asarray(fms['t']);          f_wi = np.asarray(fms['i'])
f_gt = np.asarray(fms['t_gaba']);     f_gi = np.asarray(fms['i_gaba'])

# ── Animation parameters ─────────────────────────────────────────────────────────
STEP_MS     = 20
FPS         = 25
N_FRAMES    = int(DURATION_MS / STEP_MS)
RATE_WIN_MS = 300
GLOW_MS     = 80     # how long a neuron glows after firing
TRACE_LEN   = 60
MAX_HZ      = 130.0

# ── Horn geometry (normalised coords: x in [-1,1], y in [0,1]) ──────────────────
# Width of horn tapers from 0.75 at base (y=0.12) to 0.30 at tip (y=0.95)
Y_BASE = 0.12
Y_TIP  = 0.95
W_BASE = 0.75
W_TIP  = 0.30

def horn_width(y):
    """Half-width of horn at height y."""
    t = (y - Y_BASE) / (Y_TIP - Y_BASE)
    t = np.clip(t, 0, 1)
    return W_BASE + (W_TIP - W_BASE) * t

# Lamina y-boundaries (ventral=0 to dorsal=1)
LAM_BOUNDS = {
    'I':   (0.83, 0.97),   # superficial tip
    'II':  (0.65, 0.83),   # GABA (substantia gelatinosa)
    'III': (0.49, 0.65),
    'IV':  (0.33, 0.49),
    'V':   (0.12, 0.33),   # WDR neurons
}

# ── Pre-assign fixed neuron positions within their laminae ──────────────────────
def neuron_positions(n, y_lo, y_hi, seed):
    rng = np.random.default_rng(seed)
    y = rng.uniform(y_lo + 0.01, y_hi - 0.01, n)
    # x must stay within the horn at each neuron's y
    x_max = horn_width(y) * 0.88
    x = rng.uniform(-x_max, x_max, n)
    return x, y

wdr_x,  wdr_y  = neuron_positions(N_WDR,  *LAM_BOUNDS['V'],  seed=1)
gaba_x, gaba_y = neuron_positions(N_GABA, *LAM_BOUNDS['II'], seed=2)

# ── Synthetic input fibre spike times (visual only) ─────────────────────────────
# C-fibres at 20 Hz, A-beta at 1.6 Hz — mirroring simulation input rates
rng_in = np.random.default_rng(seed=77)
N_C_SHOW  = 5
N_AB_SHOW = 3

c_spikes  = [np.cumsum(rng_in.exponential(1000/20,  150)) for _ in range(N_C_SHOW)]
ab_spikes = [np.cumsum(rng_in.exponential(1000/1.6, 10))  for _ in range(N_AB_SHOW)]
c_spikes  = [s[s < DURATION_MS] for s in c_spikes]
ab_spikes = [s[s < DURATION_MS] for s in ab_spikes]

# Fixed y-positions for input dots (in horn coord space)
c_dot_y  = np.linspace(0.68, 0.80, N_C_SHOW)
ab_dot_y = np.linspace(0.15, 0.27, N_AB_SHOW)

# ── Figure layout ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 8), facecolor=BG)
gs  = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.55],
                         left=0.01, right=0.98, top=0.91, bottom=0.05, wspace=0.08)

ax_h  = fig.add_subplot(gs[0])
ax_f  = fig.add_subplot(gs[1])
ax_tr = fig.add_subplot(gs[2])

fig.suptitle('Spinal Dorsal Horn Circuit  -  Neural Activity Replay',
             color=TEXT, fontsize=14, fontweight='bold')

# ── Spinal cord overview drawing ─────────────────────────────────────────────────
def draw_spinal_cord_overview(ax):
    """Stylised spinal cord cross-section; right dorsal horn highlighted."""
    ax.set_xlim(-1.30, 1.55)
    ax.set_ylim(-1.38, 1.55)
    ax.set_aspect('equal')
    ax.axis('off')

    # Outer white matter oval
    ax.add_patch(Ellipse((0, 0), width=2.30, height=2.10,
                         facecolor='#EDE8F2', edgecolor='#807090',
                         linewidth=1.5, zorder=1))

    # Grey matter butterfly polygon
    gm = np.array([
        [ 0.18,  0.88],   # right dorsal horn tip
        [ 0.34,  0.40],   # right dorsal shoulder
        [ 0.50,  0.12],   # right body top
        [ 0.56, -0.08],   # right body
        [ 0.58, -0.65],   # right ventral horn tip
        [ 0.40, -0.72],
        [ 0.28, -0.38],
        [ 0.18, -0.20],   # right commissure base
        [-0.18, -0.20],   # left commissure base
        [-0.28, -0.38],
        [-0.40, -0.72],
        [-0.58, -0.65],   # left ventral horn tip
        [-0.56, -0.08],
        [-0.50,  0.12],
        [-0.34,  0.40],
        [-0.18,  0.88],   # left dorsal horn tip
        [-0.12,  0.32],   # top commissure
        [ 0.12,  0.32],
    ])
    ax.add_patch(Polygon(gm, closed=True, facecolor='#C8A8C0',
                         edgecolor='#806090', linewidth=1.0, zorder=2))

    # Highlight right dorsal horn (the region zoomed in below)
    rdh = np.array([[0.18, 0.88], [0.34, 0.40], [0.50, 0.12], [0.12, 0.32]])
    ax.add_patch(Polygon(rdh, closed=True, facecolor='#FF6090',
                         edgecolor='#CC1050', linewidth=1.8, alpha=0.82, zorder=3))

    # Central canal dot
    ax.scatter([0], [0.12], s=14, color='#706080', zorder=4, alpha=0.8)

    # Dashed zoom box around highlighted horn
    ax.add_patch(Rectangle((0.07, 0.07), 0.55, 0.87,
                            linewidth=1.4, linestyle='--',
                            edgecolor='#CC1050', facecolor='none', zorder=5))

    # Title drawn inside the axes at the top so it can't clash with horn titles
    ax.text(0, 1.42, 'Spinal cord', color=TEXT, fontsize=7.5,
            ha='center', va='top', fontweight='bold')
    ax.text(0, 1.28, 'cross-section', color=TEXT, fontsize=6.5,
            ha='center', va='top', style='italic')

    # Dorsal / Ventral direction labels — tucked just inside ylim
    ax.text(0,  1.52, 'Dorsal',  color='#706080', fontsize=6,
            ha='center', va='top', style='italic')
    ax.text(0, -1.32, 'Ventral', color='#706080', fontsize=6,
            ha='center', va='bottom', style='italic')

    # Horn labels — kept well clear of direction labels and the dashed box
    ax.text( 0.24, 0.95, 'Dorsal\nhorn', color='#CC1050',
             fontsize=5.5, ha='center', va='top', fontweight='bold',
             bbox=dict(facecolor='#FFF0F8', edgecolor='none', alpha=0.85, pad=1))
    ax.text(-0.24, 0.95, 'Dorsal\nhorn', color='#907080',
             fontsize=5.5, ha='center', va='top',
             bbox=dict(facecolor='#F8F0F4', edgecolor='none', alpha=0.85, pad=1))
    ax.text( 0.65, -0.58, 'Ventral\nhorn', color='#907080',
             fontsize=5, ha='left',  va='center')
    ax.text(-0.65, -0.58, 'Ventral\nhorn', color='#907080',
             fontsize=5, ha='right', va='center')

    # Zoom callout — sits outside and to the right of the dashed box
    ax.text(0.65, 0.60, 'zoom\n(below)', color='#CC1050',
            fontsize=5.5, ha='left', va='center', style='italic',
            bbox=dict(facecolor='#FFF0F8', edgecolor='#CC1050',
                      alpha=0.9, pad=1.5, linewidth=0.7))


# ── Horn drawing function ─────────────────────────────────────────────────────────
def build_horn_outline():
    """Return (x_poly, y_poly) for the horn outline polygon."""
    # Left edge: bottom to top
    y_left = np.linspace(Y_BASE, Y_TIP, 20)
    x_left = -horn_width(y_left)
    # Top rounded cap
    t_cap  = np.linspace(np.pi, 0, 20)
    x_cap  = np.cos(t_cap) * W_TIP
    y_cap  = Y_TIP + np.sin(t_cap) * 0.06
    # Right edge: top to bottom
    y_right = np.linspace(Y_TIP, Y_BASE, 20)
    x_right = horn_width(y_right)
    # Bottom flat
    x_bot = np.array([W_BASE, -W_BASE])
    y_bot = np.array([Y_BASE,  Y_BASE])
    x_poly = np.concatenate([x_left, x_cap, x_right, x_bot])
    y_poly = np.concatenate([y_left, y_cap, y_right, y_bot])
    return x_poly, y_poly

HORN_X, HORN_Y = build_horn_outline()

def draw_static_horn(ax, title, title_colour):
    """Draw the static background elements of one horn panel."""
    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-0.05, 1.18)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor(BG)
    ax.set_title(title, color=title_colour, fontsize=12,
                 fontweight='bold', pad=6)

    # Base fill
    horn_poly = Polygon(np.column_stack([HORN_X, HORN_Y]),
                        closed=True, facecolor='#FFF5F8',
                        edgecolor='#C07090', linewidth=2.0, zorder=1)
    ax.add_patch(horn_poly)

    # Laminar band fills (trapezoids matching horn taper)
    lam_order = ['I', 'II', 'III', 'IV', 'V']
    lam_labels = {
        'I':   'Lamina I',
        'II':  'Lamina II\n(GABA gate)',
        'III': 'Lamina III',
        'IV':  'Lamina IV',
        'V':   'Lamina V\n(WDR output)',
    }
    for lk in lam_order:
        y1, y2 = LAM_BOUNDS[lk]
        w1, w2 = horn_width(y1), horn_width(y2)
        trap_x = [-w1, w1, w2, -w2]
        trap_y = [ y1,  y1,  y2,  y2]
        band = Polygon(np.column_stack([trap_x, trap_y]),
                       closed=True, facecolor=LAM_COLS[lk],
                       edgecolor='none', alpha=0.65, zorder=2)
        ax.add_patch(band)
        # Lamina boundary line
        ax.plot([-w1, w1], [y1, y1], color='#C090A0',
                lw=0.7, ls='--', zorder=3, alpha=0.7)
        # Label right of horn — background box keeps text readable
        ax.text(W_BASE + 0.08, (y1 + y2) / 2, lam_labels[lk],
                color='#9B5070', fontsize=7, ha='left', va='center', zorder=6,
                bbox=dict(facecolor='#FFF0F4', edgecolor='none',
                          alpha=0.88, pad=1.5))

    # ── Input arrows ─────────────────────────────────────────────────────────────
    # Dots sit at x=-1.58; arrows start from x=-1.52 (clear of dots) and end at
    # the actual horn edge computed by horn_width().  Labels are placed ABOVE their
    # respective dot columns so they never share the same x-band as the dots.

    # C-fibres → Lamina II  (arrow y offset below label centre = 0.74)
    c_y = 0.69
    ax.annotate('', xy=(-horn_width(c_y), c_y), xytext=(-1.52, c_y),
                arrowprops=dict(arrowstyle='->', color=PINK, lw=2.0), zorder=5)
    # Label centred above the C-fibre dot column (dots span y=0.68–0.80)
    ax.text(-1.58, 0.83, 'C-fibres', color=PINK,
            fontsize=7.5, ha='center', va='bottom', fontweight='bold')
    ax.text(-1.58, 0.82, '(pain)', color=PINK,
            fontsize=6.5, ha='center', va='top')

    # A-beta → Lamina V  (arrow y offset below label centre = 0.225)
    ab_y = 0.17
    ax.annotate('', xy=(-horn_width(ab_y), ab_y), xytext=(-1.52, ab_y),
                arrowprops=dict(arrowstyle='->', color=BLUE, lw=2.0), zorder=5)
    # Label centred above the Aβ dot column (dots span y=0.15–0.27)
    ax.text(-1.58, 0.30, 'Aβ-fibres', color=BLUE,
            fontsize=7.5, ha='center', va='bottom', fontweight='bold')
    ax.text(-1.58, 0.29, '(touch)', color=BLUE,
            fontsize=6.5, ha='center', va='top')

    # ── Output arrow ──────────────────────────────────────────────────────────────
    # Label placed ABOVE the arrow (centred) so it stays within the axes bounds
    # and cannot bleed into the trace panel on the right.
    out_y = 0.28
    hw_out = horn_width(out_y)
    ax.annotate('', xy=(1.40, out_y), xytext=(hw_out, out_y),
                arrowprops=dict(arrowstyle='->', color=PURPLE, lw=2.0), zorder=5)
    # Label placed below the Lamina V text in the right-hand label column —
    # same x as all lamina labels, just below the horn base (y=0.12).
    ax.text(W_BASE + 0.08, 0.07, 'Pain signal', color=PURPLE,
            fontsize=7, ha='left', va='top', fontweight='bold', zorder=8,
            bbox=dict(facecolor='#FFF0F4', edgecolor='none', alpha=0.9, pad=1.5))

draw_static_horn(ax_h, 'HEALTHY STATE',      BLUE)
draw_static_horn(ax_f, 'FIBROMYALGIA STATE', PINK)

# ── Neuron scatter objects ────────────────────────────────────────────────────────
# Resting (always shown, dim): positions fixed
# Firing (updated each frame): only neurons currently active

def add_neuron_layer(ax, x, y, colour, s_rest, s_fire, s_glow):
    rest = ax.scatter(x, y, s=s_rest, color='#D8B0BA', alpha=0.45,
                      linewidths=0, zorder=5)
    fire = ax.scatter([], [], s=s_fire, color=colour, alpha=0.95,
                      linewidths=0, zorder=7)
    glow = ax.scatter([], [], s=s_glow, color=colour, alpha=0.18,
                      linewidths=0, zorder=6)
    return rest, fire, glow

h_wdr_rest,  h_wdr_fire,  h_wdr_glow  = add_neuron_layer(ax_h, wdr_x,  wdr_y,  BLUE, 22, 70, 280)
h_gaba_rest, h_gaba_fire, h_gaba_glow = add_neuron_layer(ax_h, gaba_x, gaba_y, GREY, 18, 55, 200)
f_wdr_rest,  f_wdr_fire,  f_wdr_glow  = add_neuron_layer(ax_f, wdr_x,  wdr_y,  PINK, 22, 70, 280)
f_gaba_rest, f_gaba_fire, f_gaba_glow = add_neuron_layer(ax_f, gaba_x, gaba_y, GREY, 18, 55, 200)

# Input fibre dots — pushed left to x=-1.58 so they sit clear of the arrow labels
c_dot_x  = np.full(N_C_SHOW,  -1.58)
ab_dot_x = np.full(N_AB_SHOW, -1.58)

h_c_sc  = ax_h.scatter(c_dot_x,  c_dot_y,  s=35, color=PINK, alpha=0.25, zorder=6)
h_ab_sc = ax_h.scatter(ab_dot_x, ab_dot_y, s=35, color=BLUE, alpha=0.25, zorder=6)
f_c_sc  = ax_f.scatter(c_dot_x,  c_dot_y,  s=35, color=PINK, alpha=0.25, zorder=6)
f_ab_sc = ax_f.scatter(ab_dot_x, ab_dot_y, s=35, color=BLUE, alpha=0.25, zorder=6)

# ── Hz text per panel ─────────────────────────────────────────────────────────────
hz_h_txt = ax_h.text(0, 1.13, '', ha='center', va='top',
                      color=GOLD, fontsize=11, fontweight='bold',
                      fontfamily='monospace', zorder=8)
hz_f_txt = ax_f.text(0, 1.13, '', ha='center', va='top',
                      color=GOLD, fontsize=11, fontweight='bold',
                      fontfamily='monospace', zorder=8)

# ── Output trace panel ───────────────────────────────────────────────────────────
ax_tr.set_facecolor('#FFE0EC')
ax_tr.set_xlim(0, TRACE_LEN)
ax_tr.set_ylim(0, MAX_HZ * 1.1)
ax_tr.set_ylabel('WDR firing rate (Hz)', color=TEXT, fontsize=9)
ax_tr.set_title('WDR Output Signal', color=TEXT, fontsize=11, fontweight='bold')
ax_tr.tick_params(colors=TEXT, labelsize=8)
ax_tr.set_xticks([])
for sp in ax_tr.spines.values():
    sp.set_edgecolor('#C08090')

trace_h_ln, = ax_tr.plot([], [], color=BLUE, lw=2.0, alpha=0.9, label='Healthy')
trace_f_ln, = ax_tr.plot([], [], color=PINK, lw=2.0, alpha=0.9, label='FMS')
ax_tr.legend(fontsize=9, framealpha=0.7, loc='upper left',
             labelcolor=TEXT, edgecolor='#C08090')
trace_h_buf = deque(maxlen=TRACE_LEN)
trace_f_buf = deque(maxlen=TRACE_LEN)

# ── Spinal cord context inset (floated between the two horn panels) ───────────────
# Placed at the centre-top of the figure, straddling the boundary of ax_h / ax_f.
# The inset has a solid background so it cleanly overlays the (empty) horn tips.
ax_sc = fig.add_axes([0.332, 0.62, 0.136, 0.26], facecolor='#FFF5F8')
draw_spinal_cord_overview(ax_sc)
# Thin decorative border
for sp in ax_sc.spines.values():
    sp.set_visible(True)
    sp.set_edgecolor('#C07090')
    sp.set_linewidth(1.5)

# ── Legend ────────────────────────────────────────────────────────────────────────
legend_handles = [
    Line2D([0], [0], marker='o', color=BG, markerfacecolor=BLUE, ms=9,
           label='WDR neuron (firing)'),
    Line2D([0], [0], marker='o', color=BG, markerfacecolor=PINK, ms=9,
           label='WDR neuron (FMS)'),
    Line2D([0], [0], marker='o', color=BG, markerfacecolor=GREY, ms=9,
           label='GABA interneuron (firing)'),
    Line2D([0], [0], marker='o', color=BG, markerfacecolor='#D8B0BA', ms=9,
           label='Neuron (resting)'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=4,
           framealpha=0.6, edgecolor='#C08090', labelcolor=TEXT, fontsize=8.5,
           bbox_to_anchor=(0.5, 0.0))

# ── Helpers ───────────────────────────────────────────────────────────────────────
def rolling_hz(spk_t, t_end, n):
    t0 = max(0.0, t_end - RATE_WIN_MS)
    return int(np.sum((spk_t >= t0) & (spk_t < t_end))) / (n * RATE_WIN_MS * 1e-3)

def firing_positions(spk_t, spk_i, t_current, n_neurons, x_pos, y_pos):
    """Return (x, y) arrays for neurons that fired within GLOW_MS of t_current."""
    t0 = max(0.0, t_current - GLOW_MS)
    recent = (spk_t >= t0) & (spk_t < t_current)
    if not recent.any():
        return np.empty((0, 2))
    fired_ids = np.unique(spk_i[recent])
    # Clamp to valid range
    fired_ids = fired_ids[fired_ids < n_neurons]
    if len(fired_ids) == 0:
        return np.empty((0, 2))
    return np.column_stack([x_pos[fired_ids], y_pos[fired_ids]])

def flash_inputs(spk_list, t_current, sc, base_alpha=0.25, flash_alpha=0.95):
    """Flash an input dot scatter if any fibre fired recently."""
    t0 = max(0.0, t_current - GLOW_MS)
    fired = any(np.any((s >= t0) & (s < t_current)) for s in spk_list)
    sc.set_alpha(flash_alpha if fired else base_alpha)

def update_panel(fire_sc, glow_sc, spk_t, spk_i, t, n, x_pos, y_pos):
    xy = firing_positions(spk_t, spk_i, t, n, x_pos, y_pos)
    fire_sc.set_offsets(xy)
    glow_sc.set_offsets(xy)

# ── Per-frame update ───────────────────────────────────────────────────────────────
def update(frame):
    t = frame * STEP_MS

    # ── Healthy ──
    update_panel(h_wdr_fire,  h_wdr_glow,  h_wt, h_wi, t, N_WDR,  wdr_x,  wdr_y)
    update_panel(h_gaba_fire, h_gaba_glow, h_gt, h_gi, t, N_GABA, gaba_x, gaba_y)
    flash_inputs(c_spikes,  t, h_c_sc)
    flash_inputs(ab_spikes, t, h_ab_sc)
    hz_h = rolling_hz(h_wt, t, N_WDR)
    hz_h_txt.set_text(f'WDR: {hz_h:.1f} Hz')
    trace_h_buf.append(hz_h)
    trace_h_ln.set_data(range(len(trace_h_buf)), list(trace_h_buf))

    # ── FMS ──
    update_panel(f_wdr_fire,  f_wdr_glow,  f_wt, f_wi, t, N_WDR,  wdr_x,  wdr_y)
    update_panel(f_gaba_fire, f_gaba_glow, f_gt, f_gi, t, N_GABA, gaba_x, gaba_y)
    flash_inputs(c_spikes,  t, f_c_sc)
    flash_inputs(ab_spikes, t, f_ab_sc)
    hz_f = rolling_hz(f_wt, t, N_WDR)
    hz_f_txt.set_text(f'WDR: {hz_f:.1f} Hz')
    trace_f_buf.append(hz_f)
    trace_f_ln.set_data(range(len(trace_f_buf)), list(trace_f_buf))

    return []

# ── Build animation ────────────────────────────────────────────────────────────────
anim = animation.FuncAnimation(
    fig, update,
    frames=N_FRAMES,
    interval=1000 / FPS,
    blit=False,
    repeat=True,
)

# ── Display or save ────────────────────────────────────────────────────────────────
if args.save:
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'demo_dorsal_horn.gif')
    writer = animation.PillowWriter(fps=FPS)
    print(f'Saving GIF to {out_path} ...')
    anim.save(out_path, writer=writer, dpi=120,
              savefig_kwargs={'facecolor': BG})
    print('Done.')
else:
    plt.show()
