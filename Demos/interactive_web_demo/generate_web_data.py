"""
generate_web_data.py
Runs 32 Brian2 simulations (16 stimuli × 2 states) and exports spike data
to Demos/interactive_web_demo/web_data.js for use by pain_simulator.html.

Original 4 scenarios: light_touch, moderate_pain, repeated, noxious
Showcase 12 scenarios: arm/leg/torso/head × light/medium/hard

Usage:
    python Demos/interactive_web_demo/generate_web_data.py
"""

import sys, os, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from simulations import run_simulation
from synapses import PathologyStates
from stimulation import build_stimulation_protocol

# ── Horn geometry (must match pain_simulator.html) ────────────────────────────
Y_BASE, Y_TIP = 0.12, 0.95
W_BASE, W_TIP = 0.75, 0.30
N_WDR  = 50
N_GABA = 30

LAM_BOUNDS = {
    'II': (0.65, 0.83),   # GABA  (substantia gelatinosa)
    'V':  (0.12, 0.33),   # WDR
}

def horn_width(y):
    t = np.clip((y - Y_BASE) / (Y_TIP - Y_BASE), 0, 1)
    return W_BASE + (W_TIP - W_BASE) * t

def neuron_positions(n, y_lo, y_hi, seed):
    rng = np.random.default_rng(seed)
    y = rng.uniform(y_lo + 0.01, y_hi - 0.01, n)
    x_max = horn_width(y) * 0.88
    x = rng.uniform(-x_max, x_max, n)
    return x, y

# Pre-compute fixed neuron positions (same seeds as demo_dorsal_horn.py)
wdr_x,  wdr_y  = neuron_positions(N_WDR,  *LAM_BOUNDS['V'],  seed=1)
gaba_x, gaba_y = neuron_positions(N_GABA, *LAM_BOUNDS['II'], seed=2)

# Normalise to [0,1] for easy canvas mapping in JS
# x: [-W_BASE, W_BASE] → [0, 1]
# y: [Y_BASE-0.02, Y_TIP+0.08] → [0, 1]
X_NORM_MIN,  X_NORM_RANGE = -W_BASE,           2 * W_BASE
Y_NORM_MIN,  Y_NORM_RANGE =  Y_BASE - 0.02,    (Y_TIP + 0.08) - (Y_BASE - 0.02)

def norm_x(x): return ((x - X_NORM_MIN) / X_NORM_RANGE).tolist()
def norm_y(y): return ((y - Y_NORM_MIN) / Y_NORM_RANGE).tolist()

# ── Stimulus scenarios ────────────────────────────────────────────────────────
SEED = 42
SCENARIOS = [
    dict(
        id='light_touch',     label='Light Touch',
        protocol='mixed',     duration_ms=2000,
        c_params={'rate_hz': 5},
        ab_params={'rate_hz': 15},
        c_input_hz=5.0,  ab_input_hz=15.0,
    ),
    dict(
        id='moderate_pain',   label='Moderate Pain',
        protocol='constant',  duration_ms=2000,
        c_params={'rate_hz': 20},
        ab_params={'rate_hz': 2},
        c_input_hz=20.0, ab_input_hz=2.0,
    ),
    dict(
        id='repeated',        label='Repeated Stimulation',
        protocol='burst',     duration_ms=3000,
        c_params={'baseline_hz': 2, 'burst_hz': 30,
                  'burst_duration_ms': 100, 'burst_frequency_hz': 1.0},
        ab_params={'rate_hz': 2},
        c_input_hz=30.0, ab_input_hz=2.0,
    ),
    dict(
        id='noxious',         label='Noxious Stimulus',
        protocol='ramp',      duration_ms=2000,
        c_params={'start_hz': 5, 'end_hz': 40},
        ab_params={'rate_hz': 1},
        c_input_hz=40.0, ab_input_hz=1.0,
    ),

    # ── Showcase scenarios: Arm (upper arm) — mixed protocol ──────────────────
    dict(
        id='arm_light',       label='Arm – Light Touch',
        protocol='mixed',     duration_ms=2000,
        c_params={'rate_hz': 8},
        ab_params={'rate_hz': 20},
        c_input_hz=8.0,  ab_input_hz=20.0,
    ),
    dict(
        id='arm_medium',      label='Arm – Medium Pressure',
        protocol='mixed',     duration_ms=2000,
        c_params={'rate_hz': 15},
        ab_params={'rate_hz': 25},
        c_input_hz=15.0, ab_input_hz=25.0,
    ),
    dict(
        id='arm_hard',        label='Arm – Hard Pressure',
        protocol='mixed',     duration_ms=2000,
        c_params={'rate_hz': 25},
        ab_params={'rate_hz': 30},
        c_input_hz=25.0, ab_input_hz=30.0,
    ),

    # ── Showcase scenarios: Leg (thigh) — constant protocol ───────────────────
    dict(
        id='leg_light',       label='Leg – Light Pressure',
        protocol='constant',  duration_ms=2000,
        c_params={'rate_hz': 10},
        ab_params={'rate_hz': 2},
        c_input_hz=10.0, ab_input_hz=2.0,
    ),
    dict(
        id='leg_medium',      label='Leg – Medium Pressure',
        protocol='constant',  duration_ms=2000,
        c_params={'rate_hz': 20},
        ab_params={'rate_hz': 2},
        c_input_hz=20.0, ab_input_hz=2.0,
    ),
    dict(
        id='leg_hard',        label='Leg – Hard Pressure',
        protocol='constant',  duration_ms=2000,
        c_params={'rate_hz': 35},
        ab_params={'rate_hz': 2},
        c_input_hz=35.0, ab_input_hz=2.0,
    ),

    # ── Showcase scenarios: Torso (chest/abdomen) — ramp protocol ─────────────
    dict(
        id='torso_light',     label='Torso – Light Pressure',
        protocol='ramp',      duration_ms=2000,
        c_params={'start_hz': 3, 'end_hz': 15},
        ab_params={'rate_hz': 1},
        c_input_hz=15.0, ab_input_hz=1.0,
    ),
    dict(
        id='torso_medium',    label='Torso – Medium Pressure',
        protocol='ramp',      duration_ms=2000,
        c_params={'start_hz': 5, 'end_hz': 28},
        ab_params={'rate_hz': 1},
        c_input_hz=28.0, ab_input_hz=1.0,
    ),
    dict(
        id='torso_hard',      label='Torso – Hard Pressure',
        protocol='ramp',      duration_ms=2000,
        c_params={'start_hz': 8, 'end_hz': 40},
        ab_params={'rate_hz': 1},
        c_input_hz=40.0, ab_input_hz=1.0,
    ),

    # ── Showcase scenarios: Head/neck — burst protocol ────────────────────────
    dict(
        id='head_light',      label='Head/Neck – Light Pressure',
        protocol='burst',     duration_ms=3000,
        c_params={'baseline_hz': 1, 'burst_hz': 15,
                  'burst_duration_ms': 80, 'burst_frequency_hz': 1.0},
        ab_params={'rate_hz': 1},
        c_input_hz=15.0, ab_input_hz=1.0,
    ),
    dict(
        id='head_medium',     label='Head/Neck – Medium Pressure',
        protocol='burst',     duration_ms=3000,
        c_params={'baseline_hz': 2, 'burst_hz': 25,
                  'burst_duration_ms': 100, 'burst_frequency_hz': 1.0},
        ab_params={'rate_hz': 1},
        c_input_hz=25.0, ab_input_hz=1.0,
    ),
    dict(
        id='head_hard',       label='Head/Neck – Hard Pressure',
        protocol='burst',     duration_ms=3000,
        c_params={'baseline_hz': 2, 'burst_hz': 35,
                  'burst_duration_ms': 120, 'burst_frequency_hz': 1.0},
        ab_params={'rate_hz': 1},
        c_input_hz=35.0, ab_input_hz=1.0,
    ),
]

STATES = [
    ('healthy', PathologyStates.healthy()),
    ('fms',     PathologyStates.fibromyalgia()),
]

# ── Run simulations ───────────────────────────────────────────────────────────
output = {
    'neurons': {
        'wdr':  {'x': norm_x(wdr_x),  'y': norm_y(wdr_y)},
        'gaba': {'x': norm_x(gaba_x), 'y': norm_y(gaba_y)},
    },
    # Store normalisation constants so JS can convert back if needed
    'norm': {
        'x_min': X_NORM_MIN, 'x_range': X_NORM_RANGE,
        'y_min': Y_NORM_MIN, 'y_range': Y_NORM_RANGE,
    },
    'scenarios': {},
}

total = len(SCENARIOS) * len(STATES)
count = 0

for sc in SCENARIOS:
    output['scenarios'][sc['id']] = {
        'label':       sc['label'],
        'duration_ms': sc['duration_ms'],
        'c_hz':        sc['c_input_hz'],
        'ab_hz':       sc['ab_input_hz'],
    }

    for state_id, state in STATES:
        count += 1
        print(f'[{count}/{total}] {sc["label"]} / {state_id} ...')

        # Build stimulation INSIDE the loop so each TimedArray is fresh
        stim = build_stimulation_protocol(
            sc['protocol'],
            duration_ms=sc['duration_ms'],
            c_params=sc.get('c_params'),
            ab_params=sc.get('ab_params'),
        )

        res = run_simulation(
            state=state,
            duration_ms=sc['duration_ms'],
            seed=SEED,
            verbose=False,
            c_rates=stim['c_rates'],
            ab_rates=stim['ab_rates'],
        )

        output['scenarios'][sc['id']][state_id] = {
            'wdr_t':  np.asarray(res['t']).tolist(),
            'wdr_i':  [int(x) for x in res['i']],
            'gaba_t': np.asarray(res['t_gaba']).tolist(),
            'gaba_i': [int(x) for x in res['i_gaba']],
        }

# ── Write web_data.js ─────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web_data.js')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('// Auto-generated by generate_web_data.py -- do not edit by hand\n')
    f.write('const SIM_DATA = ')
    json.dump(output, f, separators=(',', ':'))
    f.write(';\n')

size_kb = os.path.getsize(out_path) // 1024
print(f'\nDone. Wrote {size_kb} KB to {out_path}')
print('Now open pain_simulator.html in a browser.')
