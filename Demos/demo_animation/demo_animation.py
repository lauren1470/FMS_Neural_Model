"""
demo_animation.py  -  Enhanced spike raster replay: Healthy | FMS | Intervention

Features:
  1+2. Glow effect on every spike (3 layered scatter objects: outer halo, inner halo, core)
    3. Live scrolling firing-rate trace below each raster
    4. Three panels: Healthy / Fibromyalgia / Intervention
    6. Pain-signal intensity meter on the right of each raster panel
    7. Burst-flash background effect on FMS and Intervention panels

Usage:
    python demo_animation.py           # interactive looping window
    python demo_animation.py --save    # saves demo_animation.mp4 (or .gif fallback)
"""

import sys
import os
import argparse
from collections import deque
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--save', action='store_true',
                    help='Save to demo_animation.mp4/.gif instead of displaying')
args = parser.parse_args()

import matplotlib
if args.save:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from simulations import run_simulation
from synapses import PathologyStates

# ── Colour palette ──────────────────────────────────────────────────────────────
BG     = '#FFF0F4'
PANEL  = '#FFE0EA'
BLUE   = '#4472C4'
PINK   = '#E91E63'
PURPLE = '#9B59B6'
GREY   = '#7f8c8d'
TEXT   = '#2d2d3d'
GOLD   = '#8B1A4A'

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def blend_colour(c1_hex, c2_hex, t):
    r1, g1, b1 = hex_to_rgb(c1_hex)
    r2, g2, b2 = hex_to_rgb(c2_hex)
    t = max(0.0, min(1.0, t))
    return (r1 + (r2-r1)*t, g1 + (g2-g1)*t, b1 + (b2-b1)*t)

def meter_colour(frac):
    if frac < 0.5:
        return blend_colour('#4472C4', '#f39c12', frac * 2)
    else:
        return blend_colour('#f39c12', '#E91E63', (frac - 0.5) * 2)

# ── Simulation parameters ────────────────────────────────────────────────────────
DURATION_MS = 3000
SEED        = 42

# ── Animation parameters ─────────────────────────────────────────────────────────
WINDOW_MS   = 400
STEP_MS     = 20
FPS         = 25
N_FRAMES    = int((DURATION_MS - WINDOW_MS) / STEP_MS)
RATE_WIN_MS = 300
TRACE_LEN   = 50

FLASH_THRESH_F = 110
FLASH_THRESH_I = 70

N_WDR       = 50
N_GABA      = 30
GABA_OFFSET = N_WDR + 3
Y_MAX       = GABA_OFFSET + N_GABA + 1
MAX_HZ      = 130.0

METER_X = WINDOW_MS - 22
METER_W = 16

# ── Run simulations ──────────────────────────────────────────────────────────────
print('Running Healthy simulation ...')
healthy = run_simulation(state=PathologyStates.healthy(),
                         duration_ms=DURATION_MS, seed=SEED, verbose=False)
print('Running FMS simulation ...')
fms = run_simulation(state=PathologyStates.fibromyalgia(),
                     duration_ms=DURATION_MS, seed=SEED, verbose=False)
print('Running Intervention simulation ...')
interv = run_simulation(state=PathologyStates.intervention(),
                        duration_ms=DURATION_MS, seed=SEED, verbose=False)
print(f'All simulations complete. Building {N_FRAMES}-frame animation ...')

def extract(res):
    return (np.asarray(res['t']),      np.asarray(res['i']),
            np.asarray(res['t_gaba']), np.asarray(res['i_gaba']))

h_wt, h_wi, h_gt, h_gi = extract(healthy)
f_wt, f_wi, f_gt, f_gi = extract(fms)
i_wt, i_wi, i_gt, i_gi = extract(interv)

# ── Figure + GridSpec ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 8), facecolor=BG)
gs  = gridspec.GridSpec(2, 3, figure=fig,
                         height_ratios=[3, 1],
                         hspace=0.10, wspace=0.14,
                         left=0.05, right=0.97, top=0.91, bottom=0.09)

ax_rh = fig.add_subplot(gs[0, 0])
ax_rf = fig.add_subplot(gs[0, 1])
ax_ri = fig.add_subplot(gs[0, 2])
ax_th = fig.add_subplot(gs[1, 0])
ax_tf = fig.add_subplot(gs[1, 1])
ax_ti = fig.add_subplot(gs[1, 2])

fig.suptitle('FMS Neural Model  -  Spike Activity Replay',
             color=TEXT, fontsize=15, fontweight='bold')

# ── Axis styling ──────────────────────────────────────────────────────────────────
def style_raster(ax, title, colour):
    ax.set_facecolor(PANEL)
    ax.set_xlim(0, WINDOW_MS)
    ax.set_ylim(-1, Y_MAX)
    ax.set_title(title, color=colour, fontsize=12, fontweight='bold', pad=5)
    ax.set_xlabel('Time in window (ms)', color=TEXT, fontsize=9)
    ax.set_ylabel('Neuron index',        color=TEXT, fontsize=9)
    ax.tick_params(colors=TEXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor('#c08090')
    ax.axhline(N_WDR + 0.5, color='#d0a0b0', lw=0.8, ls='--')
    ax.text(4, N_WDR * 0.5,                'WDR',  color='#8B4060',
            fontsize=8, va='center', alpha=0.8)
    ax.text(4, GABA_OFFSET + N_GABA * 0.5, 'GABA', color='#8B4060',
            fontsize=8, va='center', alpha=0.8)
    ax.text(METER_X + METER_W / 2, Y_MAX + 1.2, 'Pain\nsignal',
            color='#9B5070', fontsize=7, ha='center', va='bottom')

def style_trace(ax):
    ax.set_facecolor('#FFD5E2')
    ax.set_xlim(0, TRACE_LEN)
    ax.set_ylim(0, MAX_HZ * 1.1)
    ax.set_ylabel('WDR (Hz)', color=TEXT, fontsize=8)
    ax.tick_params(colors=TEXT, labelsize=7)
    ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor('#c08090')

style_raster(ax_rh, 'HEALTHY STATE',      BLUE)
style_raster(ax_rf, 'FIBROMYALGIA STATE', PINK)
style_raster(ax_ri, 'INTERVENTION STATE', PURPLE)
style_trace(ax_th)
style_trace(ax_tf)
style_trace(ax_ti)

# ── Glow scatter layers ───────────────────────────────────────────────────────────
# Use fixed colour + alpha per layer — avoids colormap/facecolor conflicts.
# Three sizes create the neon glow look; positions updated each frame via set_offsets.

def make_glow(ax, colour):
    """3-layer glow: outer halo, inner halo, bright core. Returns (outer, inner, core)."""
    outer = ax.scatter([], [], s=160, color=colour, alpha=0.09, linewidths=0, zorder=2)
    inner = ax.scatter([], [], s=55,  color=colour, alpha=0.25, linewidths=0, zorder=3)
    core  = ax.scatter([], [], s=12,  color=colour, alpha=0.90, linewidths=0, zorder=4)
    return outer, inner, core

# Healthy
h_wdr_o, h_wdr_i, h_wdr_c = make_glow(ax_rh, BLUE)
h_gab_o, h_gab_i, h_gab_c = make_glow(ax_rh, GREY)
# FMS
f_wdr_o, f_wdr_i, f_wdr_c = make_glow(ax_rf, PINK)
f_gab_o, f_gab_i, f_gab_c = make_glow(ax_rf, GREY)
# Intervention
i_wdr_o, i_wdr_i, i_wdr_c = make_glow(ax_ri, PURPLE)
i_gab_o, i_gab_i, i_gab_c = make_glow(ax_ri, GREY)

# ── Pain meter rectangles ─────────────────────────────────────────────────────────
def make_meter(ax):
    rect = Rectangle((METER_X, 0), METER_W, 0, linewidth=0, zorder=6)
    ax.add_patch(rect)
    return rect

meter_h = make_meter(ax_rh)
meter_f = make_meter(ax_rf)
meter_i = make_meter(ax_ri)

# ── Rate trace lines ──────────────────────────────────────────────────────────────
trace_h_ln, = ax_th.plot([], [], color=BLUE,   lw=1.5, alpha=0.9)
trace_f_ln, = ax_tf.plot([], [], color=PINK,   lw=1.5, alpha=0.9)
trace_i_ln, = ax_ti.plot([], [], color=PURPLE, lw=1.5, alpha=0.9)
trace_h_buf = deque(maxlen=TRACE_LEN)
trace_f_buf = deque(maxlen=TRACE_LEN)
trace_i_buf = deque(maxlen=TRACE_LEN)

# ── Hz counter text ───────────────────────────────────────────────────────────────
_kw = dict(ha='left', va='top', color=GOLD, fontsize=10,
           fontweight='bold', fontfamily='monospace')
hz_h_txt = ax_rh.text(0.02, 0.97, '', transform=ax_rh.transAxes, **_kw)
hz_f_txt = ax_rf.text(0.02, 0.97, '', transform=ax_rf.transAxes, **_kw)
hz_i_txt = ax_ri.text(0.02, 0.97, '', transform=ax_ri.transAxes, **_kw)
time_txt  = fig.text(0.5, 0.003, '', ha='center', color='#444466', fontsize=8)

# ── Flash state ───────────────────────────────────────────────────────────────────
flash_f = 0.0
flash_i = 0.0

# ── Legend ────────────────────────────────────────────────────────────────────────
legend_handles = [
    Line2D([0], [0], marker='o', color=BG, markerfacecolor=BLUE,   ms=9, label='WDR - Healthy'),
    Line2D([0], [0], marker='o', color=BG, markerfacecolor=PINK,   ms=9, label='WDR - FMS'),
    Line2D([0], [0], marker='o', color=BG, markerfacecolor=PURPLE, ms=9, label='WDR - Intervention'),
    Line2D([0], [0], marker='o', color=BG, markerfacecolor=GREY,   ms=9, label='GABA interneurons'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=4,
           framealpha=0.15, edgecolor='#222244', labelcolor=TEXT, fontsize=9,
           bbox_to_anchor=(0.5, 0.0))

# ── Helpers ───────────────────────────────────────────────────────────────────────
def rolling_hz(spk_t, t_end, n):
    t0 = max(0.0, t_end - RATE_WIN_MS)
    return int(np.sum((spk_t >= t0) & (spk_t < t_end))) / (n * RATE_WIN_MS * 1e-3)

def update_glow(outer, inner, core, spk_t, spk_i, t0, t1, y_offset=0):
    mask = (spk_t >= t0) & (spk_t < t1)
    if mask.any():
        xy = np.column_stack([spk_t[mask] - t0, spk_i[mask] + y_offset])
    else:
        xy = np.empty((0, 2))
    outer.set_offsets(xy)
    inner.set_offsets(xy)
    core.set_offsets(xy)

def update_meter(rect, hz):
    frac = min(1.0, hz / MAX_HZ)
    rect.set_height(frac * Y_MAX)
    rect.set_facecolor(meter_colour(frac))

def update_trace(line, buf, hz):
    buf.append(hz)
    line.set_data(range(len(buf)), list(buf))

# ── Per-frame update ───────────────────────────────────────────────────────────────
def update(frame):
    global flash_f, flash_i
    t0 = frame * STEP_MS
    t1 = t0 + WINDOW_MS

    # Healthy
    update_glow(h_wdr_o, h_wdr_i, h_wdr_c, h_wt, h_wi, t0, t1)
    update_glow(h_gab_o, h_gab_i, h_gab_c, h_gt, h_gi, t0, t1, GABA_OFFSET)
    hz_h = rolling_hz(h_wt, t1, N_WDR)
    hz_h_txt.set_text(f'WDR  {hz_h:5.1f} Hz')
    update_meter(meter_h, hz_h)
    update_trace(trace_h_ln, trace_h_buf, hz_h)

    # FMS
    update_glow(f_wdr_o, f_wdr_i, f_wdr_c, f_wt, f_wi, t0, t1)
    update_glow(f_gab_o, f_gab_i, f_gab_c, f_gt, f_gi, t0, t1, GABA_OFFSET)
    hz_f = rolling_hz(f_wt, t1, N_WDR)
    hz_f_txt.set_text(f'WDR  {hz_f:5.1f} Hz')
    update_meter(meter_f, hz_f)
    update_trace(trace_f_ln, trace_f_buf, hz_f)
    if hz_f > FLASH_THRESH_F:
        flash_f = min(1.0, flash_f + 0.6)
    flash_f *= 0.75
    ax_rf.set_facecolor(blend_colour(PANEL, '#FF8FAF', flash_f))

    # Intervention
    update_glow(i_wdr_o, i_wdr_i, i_wdr_c, i_wt, i_wi, t0, t1)
    update_glow(i_gab_o, i_gab_i, i_gab_c, i_gt, i_gi, t0, t1, GABA_OFFSET)
    hz_i = rolling_hz(i_wt, t1, N_WDR)
    hz_i_txt.set_text(f'WDR  {hz_i:5.1f} Hz')
    update_meter(meter_i, hz_i)
    update_trace(trace_i_ln, trace_i_buf, hz_i)
    if hz_i > FLASH_THRESH_I:
        flash_i = min(1.0, flash_i + 0.4)
    flash_i *= 0.75
    ax_ri.set_facecolor(blend_colour(PANEL, '#D0A0FF', flash_i))

    time_txt.set_text(f't = {t0:.0f} - {t1:.0f} ms')
    return []

# ── Build + run/save ───────────────────────────────────────────────────────────────
anim = animation.FuncAnimation(
    fig, update,
    frames=N_FRAMES,
    interval=1000 / FPS,
    blit=False,
    repeat=True,
)

if args.save:
    try:
        animation.FFMpegWriter()
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'demo_animation.mp4')
        writer = animation.FFMpegWriter(fps=FPS, bitrate=2500,
                                        extra_args=['-vcodec', 'libx264'])
        print(f'Saving MP4 to {out_path} ...')
        anim.save(out_path, writer=writer, dpi=150,
                  savefig_kwargs={'facecolor': BG})
    except Exception:
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'demo_animation.gif')
        writer = animation.PillowWriter(fps=FPS)
        print(f'ffmpeg not found - saving GIF to {out_path} ...')
        anim.save(out_path, writer=writer, dpi=120,
                  savefig_kwargs={'facecolor': BG})
    print('Done.')
else:
    plt.show()
