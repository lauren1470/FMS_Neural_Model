"""
demo_transition.py  -  FMS development animation (single panel)

Starts at a healthy network and progressively develops into full fibromyalgia,
with NMDA and GABA-A meters animating at the bottom as the pathology worsens.
Dot colour shifts from blue (healthy) through to pink (FMS).

Usage:
    python demo_transition.py           # interactive looping window
    python demo_transition.py --save    # saves demo_transition.mp4 (or .gif fallback)
"""

import sys
import os
import argparse
from collections import deque
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--save', action='store_true',
                    help='Save to demo_transition.mp4/.gif instead of displaying')
args = parser.parse_args()

import matplotlib
if args.save:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from simulations import run_simulation
from synapses import PathologyStates

# ── Colour palette ──────────────────────────────────────────────────────────────
BG    = '#FFF0F4'
PANEL = '#FFE0EA'
BLUE  = '#4472C4'
PINK  = '#E91E63'
GREY  = '#7f8c8d'
TEXT  = '#2d2d3d'
GOLD  = '#8B1A4A'

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

BLUE_RGB = hex_to_rgb(BLUE)
PINK_RGB = hex_to_rgb(PINK)
GREY_RGB = hex_to_rgb(GREY)

def lerp_rgb(rgb1, rgb2, t):
    t = max(0.0, min(1.0, t))
    return tuple(a + (b - a) * t for a, b in zip(rgb1, rgb2))

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

def blend_hex(c1, c2, t):
    return rgb_to_hex(lerp_rgb(hex_to_rgb(c1), hex_to_rgb(c2), t))

# ── Disease progression states ──────────────────────────────────────────────────
#   (label, PathologyState, nmda_multiplier, gaba_factor)
STATES_CONFIG = [
    ('Healthy',      PathologyStates.healthy(),              1.0, 1.00),
    ('Early Onset',  PathologyStates.custom(1.5, 0.85),      1.5, 0.85),
    ('Moderate',     PathologyStates.custom(2.0, 0.60),      2.0, 0.60),
    ('Severe',       PathologyStates.custom(2.5, 0.45),      2.5, 0.45),
    ('Fibromyalgia', PathologyStates.fibromyalgia(),         3.0, 0.40),
]
N_STATES    = len(STATES_CONFIG)
DURATION_MS = 2000
SEED        = 42

# Colour for each state: linear interpolation from BLUE to PINK
STATE_RGBS = [lerp_rgb(BLUE_RGB, PINK_RGB, i / (N_STATES - 1))
              for i in range(N_STATES)]

# ── Pre-run all simulations ─────────────────────────────────────────────────────
sims = []
for name, state, nmda, gaba in STATES_CONFIG:
    print(f'Running {name}  (NMDA={nmda}x, GABA={gaba}x) ...')
    res = run_simulation(state=state, duration_ms=DURATION_MS, seed=SEED, verbose=False)
    sims.append({
        'name': name, 'nmda': nmda, 'gaba': gaba,
        'wt': np.asarray(res['t']),      'wi': np.asarray(res['i']),
        'gt': np.asarray(res['t_gaba']), 'gi': np.asarray(res['i_gaba']),
    })
print('All simulations complete. Building transition animation ...')

# ── Animation parameters ────────────────────────────────────────────────────────
WINDOW_MS      = 400
STEP_MS        = 20
FPS            = 25
CROSSFADE_F    = 10    # frames of crossfade between each state
FRAMES_PER_STATE = int((DURATION_MS - WINDOW_MS) / STEP_MS)
RATE_WIN_MS    = 300
TRACE_LEN      = 60

N_WDR       = 50
N_GABA      = 30
GABA_OFFSET = N_WDR + 3
Y_MAX       = GABA_OFFSET + N_GABA + 1
MAX_HZ      = 130.0

# Build frame schedule: list of (state_idx, local_frame, crossfade_fraction)
# crossfade_fraction is None during normal playback, 0-1 during transitions
frame_schedule = []
for si in range(N_STATES):
    for lf in range(FRAMES_PER_STATE):
        frame_schedule.append((si, lf, None))
    if si < N_STATES - 1:
        for cf in range(CROSSFADE_F):
            frame_schedule.append((si, FRAMES_PER_STATE - 1, (cf + 1) / CROSSFADE_F))

N_FRAMES = len(frame_schedule)

# ── Figure + GridSpec ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 9), facecolor=BG)
gs  = gridspec.GridSpec(4, 1, figure=fig,
                         height_ratios=[0.45, 3.2, 1.0, 0.85],
                         hspace=0.12,
                         left=0.10, right=0.95, top=0.93, bottom=0.04)

ax_hdr   = fig.add_subplot(gs[0])   # state name + Hz readout
ax_rast  = fig.add_subplot(gs[1])   # raster
ax_trace = fig.add_subplot(gs[2])   # firing rate trace
ax_mtrs  = fig.add_subplot(gs[3])   # NMDA + GABA meters

fig.suptitle('Central Sensitisation Development in Fibromyalgia',
             color=TEXT, fontsize=14, fontweight='bold')

# ── Header ───────────────────────────────────────────────────────────────────────
ax_hdr.set_facecolor(BG)
ax_hdr.set_xlim(0, 1); ax_hdr.set_ylim(0, 1)
ax_hdr.axis('off')
state_lbl = ax_hdr.text(0.5, 0.85, 'Healthy', ha='center', va='top',
                          color=BLUE, fontsize=20, fontweight='bold')
hz_lbl    = ax_hdr.text(0.5, 0.05, '', ha='center', va='bottom',
                          color=GOLD, fontsize=13, fontweight='bold',
                          fontfamily='monospace')

# ── Raster ───────────────────────────────────────────────────────────────────────
ax_rast.set_facecolor(PANEL)
ax_rast.set_xlim(0, WINDOW_MS)
ax_rast.set_ylim(-1, Y_MAX)
ax_rast.set_xlabel('Time in window (ms)', color=TEXT, fontsize=10)
ax_rast.set_ylabel('Neuron index',        color=TEXT, fontsize=10)
ax_rast.tick_params(colors=TEXT)
for sp in ax_rast.spines.values():
    sp.set_edgecolor('#c08090')
ax_rast.axhline(N_WDR + 0.5, color='#d0a0b0', lw=0.8, ls='--')
ax_rast.text(4, N_WDR * 0.5,                'WDR',  color='#8B4060',
             fontsize=9, va='center', alpha=0.8)
ax_rast.text(4, GABA_OFFSET + N_GABA * 0.5, 'GABA', color='#8B4060',
             fontsize=9, va='center', alpha=0.8)

# Glow scatter layers — fixed colour+alpha per layer, positions updated via set_offsets
# Colour will be updated each frame via set_facecolor on the collection
wdr_outer = ax_rast.scatter([], [], s=160, color=BLUE, alpha=0.09, linewidths=0, zorder=2)
wdr_inner = ax_rast.scatter([], [], s=55,  color=BLUE, alpha=0.25, linewidths=0, zorder=3)
wdr_core  = ax_rast.scatter([], [], s=12,  color=BLUE, alpha=0.90, linewidths=0, zorder=4)
gab_outer = ax_rast.scatter([], [], s=100, color=GREY, alpha=0.07, linewidths=0, zorder=2)
gab_inner = ax_rast.scatter([], [], s=35,  color=GREY, alpha=0.20, linewidths=0, zorder=3)
gab_core  = ax_rast.scatter([], [], s=8,   color=GREY, alpha=0.85, linewidths=0, zorder=4)

# ── Firing rate trace ─────────────────────────────────────────────────────────────
ax_trace.set_facecolor('#FFD5E2')
ax_trace.set_xlim(0, TRACE_LEN)
ax_trace.set_ylim(0, MAX_HZ * 1.1)
ax_trace.set_ylabel('WDR firing rate (Hz)', color=TEXT, fontsize=9)
ax_trace.tick_params(colors=TEXT, labelsize=8)
ax_trace.set_xticks([])
for sp in ax_trace.spines.values():
    sp.set_edgecolor('#c08090')
trace_line, = ax_trace.plot([], [], color=BLUE, lw=1.8)
trace_buf   = deque(maxlen=TRACE_LEN)

# ── NMDA + GABA meters ───────────────────────────────────────────────────────────
ax_mtrs.set_facecolor(BG)
ax_mtrs.set_xlim(0, 10)
ax_mtrs.set_ylim(0, 2.6)
ax_mtrs.axis('off')

BAR_X    = 2.2    # left edge of meter bars
BAR_W    = 6.2    # full bar width (at maximum)
BAR_H    = 0.50   # bar height
NMDA_Y   = 1.8    # y-position of NMDA bar
GABA_Y   = 0.55   # y-position of GABA bar

# Labels
ax_mtrs.text(0, NMDA_Y + BAR_H / 2, 'NMDA', color=TEXT,
             fontsize=11, fontweight='bold', va='center')
ax_mtrs.text(0, GABA_Y + BAR_H / 2, 'GABA-A', color=TEXT,
             fontsize=11, fontweight='bold', va='center')

# Background tracks
ax_mtrs.add_patch(Rectangle((BAR_X, NMDA_Y), BAR_W, BAR_H,
                              facecolor='#F0C0CC', linewidth=0))
ax_mtrs.add_patch(Rectangle((BAR_X, GABA_Y), BAR_W, BAR_H,
                              facecolor='#F0C0CC', linewidth=0))

# Animated fill bars
nmda_fill = ax_mtrs.add_patch(Rectangle((BAR_X, NMDA_Y), 0, BAR_H,
                                          facecolor=BLUE, linewidth=0, zorder=2))
gaba_fill = ax_mtrs.add_patch(Rectangle((BAR_X, GABA_Y), BAR_W, BAR_H,
                                          facecolor=BLUE, linewidth=0, zorder=2))

# Value labels
nmda_val_lbl = ax_mtrs.text(BAR_X + BAR_W + 0.2, NMDA_Y + BAR_H / 2, '',
                              color=TEXT, fontsize=11, fontweight='bold', va='center')
gaba_val_lbl = ax_mtrs.text(BAR_X + BAR_W + 0.2, GABA_Y + BAR_H / 2, '',
                              color=TEXT, fontsize=11, fontweight='bold', va='center')

# Tick marks for NMDA bar (1x, 2x, 3x)
for tick_nmda in [1.0, 2.0, 3.0]:
    frac = (tick_nmda - 1.0) / 2.0
    tx = BAR_X + frac * BAR_W
    ax_mtrs.plot([tx, tx], [NMDA_Y - 0.06, NMDA_Y], color='#555577', lw=0.8)
    ax_mtrs.text(tx, NMDA_Y - 0.10, f'{tick_nmda:.0f}x',
                 color='#9B5070', fontsize=7, ha='center', va='top')

# Tick marks for GABA bar (1.0, 0.6, 0.2)
for tick_gaba in [1.0, 0.6, 0.2]:
    frac = (tick_gaba - 0.4) / 0.6
    tx = BAR_X + frac * BAR_W
    ax_mtrs.plot([tx, tx], [GABA_Y - 0.06, GABA_Y], color='#555577', lw=0.8)
    ax_mtrs.text(tx, GABA_Y - 0.10, f'{tick_gaba:.1f}',
                 color='#9B5070', fontsize=7, ha='center', va='top')

# ── Helpers ───────────────────────────────────────────────────────────────────────
def rolling_hz(spk_t, t_end):
    t0 = max(0.0, t_end - RATE_WIN_MS)
    return int(np.sum((spk_t >= t0) & (spk_t < t_end))) / (N_WDR * RATE_WIN_MS * 1e-3)

def set_glow_colour(outer, inner, core, colour_hex):
    """Update colour of all three glow layers."""
    for sc in (outer, inner, core):
        sc.set_facecolor(colour_hex)

def update_glow(outer, inner, core, spk_t, spk_i, t0, t1, y_offset=0):
    mask = (spk_t >= t0) & (spk_t < t1)
    if mask.any():
        xy = np.column_stack([spk_t[mask] - t0, spk_i[mask] + y_offset])
    else:
        xy = np.empty((0, 2))
    outer.set_offsets(xy)
    inner.set_offsets(xy)
    core.set_offsets(xy)

# ── Per-frame update ───────────────────────────────────────────────────────────────
def update(frame):
    si, lf, cf_t = frame_schedule[frame]
    sim = sims[si]

    # Current WDR colour (blends during crossfade)
    if cf_t is not None and si < N_STATES - 1:
        wdr_rgb = lerp_rgb(STATE_RGBS[si], STATE_RGBS[si + 1], cf_t)
    else:
        wdr_rgb = STATE_RGBS[si]

    t0 = lf * STEP_MS
    t1 = t0 + WINDOW_MS

    # ── Raster ────────────────────────────────────────────────────────────────
    set_glow_colour(wdr_outer, wdr_inner, wdr_core, rgb_to_hex(wdr_rgb))
    update_glow(wdr_outer, wdr_inner, wdr_core, sim['wt'], sim['wi'], t0, t1)
    update_glow(gab_outer, gab_inner, gab_core, sim['gt'], sim['gi'], t0, t1, GABA_OFFSET)

    # ── Hz readout ────────────────────────────────────────────────────────────
    hz = rolling_hz(sim['wt'], t1)
    hz_lbl.set_text(f'WDR firing rate:  {hz:.1f} Hz')

    # ── State label (crossfades) ───────────────────────────────────────────────
    if cf_t is None:
        state_lbl.set_text(sim['name'])
        state_lbl.set_color(rgb_to_hex(STATE_RGBS[si]))
        state_lbl.set_alpha(1.0)
    elif cf_t < 0.5:
        state_lbl.set_text(sim['name'])
        state_lbl.set_alpha(1.0 - cf_t * 2)
    else:
        state_lbl.set_text(STATES_CONFIG[si + 1][0])
        state_lbl.set_color(rgb_to_hex(STATE_RGBS[si + 1]))
        state_lbl.set_alpha((cf_t - 0.5) * 2)

    # ── Firing rate trace ─────────────────────────────────────────────────────
    trace_buf.append(hz)
    trace_line.set_data(range(len(trace_buf)), list(trace_buf))
    trace_line.set_color(rgb_to_hex(wdr_rgb))

    # ── Parameter meters ──────────────────────────────────────────────────────
    if cf_t is not None and si < N_STATES - 1:
        nmda = STATES_CONFIG[si][2] + (STATES_CONFIG[si+1][2] - STATES_CONFIG[si][2]) * cf_t
        gaba = STATES_CONFIG[si][3] + (STATES_CONFIG[si+1][3] - STATES_CONFIG[si][3]) * cf_t
    else:
        nmda = STATES_CONFIG[si][2]
        gaba = STATES_CONFIG[si][3]

    # NMDA: 1.0x -> 3.0x maps to 0 -> full bar
    nmda_frac = (nmda - 1.0) / 2.0
    nmda_fill.set_width(BAR_W * nmda_frac)
    nmda_fill.set_facecolor(rgb_to_hex(lerp_rgb(BLUE_RGB, PINK_RGB, nmda_frac)))
    nmda_val_lbl.set_text(f'{nmda:.1f}x')

    # GABA: 1.0x -> 0.4x maps to full bar -> empty
    gaba_frac = (gaba - 0.4) / 0.6
    gaba_fill.set_width(BAR_W * gaba_frac)
    gaba_fill.set_facecolor(rgb_to_hex(lerp_rgb(PINK_RGB, BLUE_RGB, gaba_frac)))
    gaba_val_lbl.set_text(f'{gaba:.2f}x')

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
    try:
        animation.FFMpegWriter()
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'demo_transition.mp4')
        writer = animation.FFMpegWriter(fps=FPS, bitrate=2500,
                                        extra_args=['-vcodec', 'libx264'])
        print(f'Saving MP4 to {out_path} ...')
        anim.save(out_path, writer=writer, dpi=150,
                  savefig_kwargs={'facecolor': BG})
    except Exception:
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'demo_transition.gif')
        writer = animation.PillowWriter(fps=FPS)
        print(f'ffmpeg not found - saving GIF to {out_path} ...')
        anim.save(out_path, writer=writer, dpi=120,
                  savefig_kwargs={'facecolor': BG})
    print('Done.')
else:
    plt.show()
