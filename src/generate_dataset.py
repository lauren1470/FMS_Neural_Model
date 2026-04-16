"""
Dataset generation for the FMS spinal dorsal horn ML classifier.

Runs N automated simulation trials across healthy and fibromyalgia states,
extracts spike-train features, and saves the result as a CSV file.

Usage:
    python generate_dataset.py                    # default 1000 trials
    python generate_dataset.py --n_trials 500     # custom count
    python generate_dataset.py --output data.csv  # custom output path
"""

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd

# Ensure src/ is on the path when running this script directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')

from simulations import run_simulation, run_windup_simulation, DEFAULT_DURATION_MS, DEFAULT_DT_MS
from features import extract_features, features_to_row
from stimulation import build_stimulation_protocol


# --- Configuration ---
PROTOCOLS = ['constant', 'ramp', 'burst', 'mixed']
STATES = ['healthy', 'fibromyalgia']

# --- Parameter jitter ranges for dataset diversity ---
C_RATE_RANGE = (10.0, 30.0)      # C-fibre noxious rate (Hz)
AB_RATE_RANGE = (0.5, 5.0)       # Ab-fibre innocuous rate (Hz)
BURST_FREQ_RANGE = (0.5, 2.0)    # Burst frequency (Hz)


def _random_protocol_params(rng):
    """Generate randomised stimulation parameters for one trial."""
    protocol = rng.choice(PROTOCOLS)

    c_rate = rng.uniform(*C_RATE_RANGE)
    ab_rate = rng.uniform(*AB_RATE_RANGE)

    c_params = {}
    ab_params = {'rate_hz': ab_rate}

    if protocol == 'constant':
        c_params['rate_hz'] = c_rate
    elif protocol == 'ramp':
        c_params['start_hz'] = rng.uniform(1.0, 5.0)
        c_params['end_hz'] = c_rate
    elif protocol == 'burst':
        c_params['baseline_hz'] = rng.uniform(1.0, 5.0)
        c_params['burst_hz'] = c_rate
        c_params['burst_duration_ms'] = rng.uniform(30.0, 80.0)
        c_params['burst_frequency_hz'] = rng.uniform(*BURST_FREQ_RANGE)
    elif protocol == 'mixed':
        c_params['rate_hz'] = c_rate
        ab_params['rate_hz'] = rng.uniform(5.0, 15.0)

    return protocol, c_params, ab_params


def generate_dataset(n_trials=1000, duration_ms=DEFAULT_DURATION_MS,
                     output_path='dataset.csv', base_seed=42):
    """
    Generate a balanced dataset of healthy and FMS simulation trials.

    Parameters
    ----------
    n_trials : int
        Total number of trials (split evenly between states).
    duration_ms : float
        Simulation duration per trial in milliseconds.
    output_path : str
        Path to save the output CSV.
    base_seed : int
        Base random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        The generated dataset.
    """
    rng = np.random.default_rng(base_seed)
    trials_per_state = n_trials // 2
    rows = []

    total = trials_per_state * len(STATES)
    start_time = time.time()

    print(f"Generating {total} trials ({trials_per_state} per state)...")
    print(f"Duration: {duration_ms} ms | Protocols: {PROTOCOLS}")
    print(f"Output: {output_path}\n")

    trial_num = 0
    for state in STATES:
        for i in range(trials_per_state):
            trial_num += 1
            seed = int(rng.integers(0, 2**31))

            protocol, c_params, ab_params = _random_protocol_params(rng)

            stim = build_stimulation_protocol(
                protocol, duration_ms, DEFAULT_DT_MS, c_params, ab_params
            )

            try:
                results = run_simulation(
                    state=state,
                    duration_ms=duration_ms,
                    seed=seed,
                    verbose=False,
                    c_rates=stim['c_rates'],
                    ab_rates=stim['ab_rates'],
                )
            except Exception as e:
                print(f"  [SKIP] Trial {trial_num} failed: {e}")
                continue

            # Run dedicated wind-up measurement (discrete 5 Hz C-fibre protocol)
            try:
                windup_data = run_windup_simulation(state=state, seed=seed)
                results['windup_ratio'] = windup_data['windup_ratio']
                results['per_stimulus_counts'] = windup_data['per_stimulus_counts']
            except Exception as e:
                results['windup_ratio'] = 0.0
                results['per_stimulus_counts'] = []

            feats = extract_features(results)
            row = features_to_row(feats, state_label=state, seed=seed,
                                  protocol=protocol)
            rows.append(row)

            # Progress reporting
            if trial_num % 50 == 0 or trial_num == total:
                elapsed = time.time() - start_time
                rate = trial_num / elapsed
                eta = (total - trial_num) / rate if rate > 0 else 0
                print(f"  [{trial_num:4d}/{total}] "
                      f"{state:15s} | {protocol:8s} | "
                      f"WDR={feats['wdr_mean_rate']:6.1f} Hz | "
                      f"GABA={feats['gaba_mean_rate']:6.1f} Hz | "
                      f"ETA: {eta:.0f}s")

    # Build DataFrame and save
    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)

    elapsed = time.time() - start_time
    print(f"\nDataset saved to: {output_path}")
    print(f"  Rows: {len(df)}  |  Features: {len(df.columns) - 3}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/len(df):.2f}s per trial)")
    print(f"\nClass distribution:")
    print(df['label'].value_counts().to_string())

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate FMS neural model training dataset'
    )
    parser.add_argument('--n_trials', type=int, default=1000,
                        help='Total number of trials (default: 1000)')
    parser.add_argument('--duration', type=float, default=DEFAULT_DURATION_MS,
                        help='Simulation duration in ms (default: 1000)')
    parser.add_argument('--output', type=str, default='dataset.csv',
                        help='Output CSV path (default: dataset.csv)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed (default: 42)')
    args = parser.parse_args()

    generate_dataset(
        n_trials=args.n_trials,
        duration_ms=args.duration,
        output_path=args.output,
        base_seed=args.seed,
    )
