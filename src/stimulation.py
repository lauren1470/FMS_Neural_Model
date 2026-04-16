"""
Stimulation protocols for the FMS spinal dorsal horn model.

Provides different input patterns for C-fibres and Ab-fibres to generate
diverse training data for the ML classifier. Each protocol returns a
TimedArray of firing rates that can drive PoissonGroup neurons.

Protocols:
    - constant:  Steady-state firing at a fixed rate
    - ramp:      Linearly increasing rate (demonstrates wind-up buildup)
    - burst:     Periodic bursts at 0.5-2 Hz (classic wind-up frequency)
    - mixed:     Simultaneous noxious + innocuous (tests gate control)
"""

import numpy as np
from brian2 import Hz, ms, TimedArray


def constant_rate(rate_hz, duration_ms, dt_ms=0.1):
    """
    Constant firing rate throughout the simulation.

    Parameters
    ----------
    rate_hz : float
        Firing rate in Hz.
    duration_ms : float
        Total duration in milliseconds.
    dt_ms : float
        Time step in milliseconds.

    Returns
    -------
    TimedArray
        Brian2 TimedArray with constant rate.
    """
    n_steps = int(duration_ms / dt_ms)
    rates = np.full(n_steps, rate_hz) * Hz
    return TimedArray(rates, dt=dt_ms * ms)


def ramp_rate(start_hz, end_hz, duration_ms, dt_ms=0.1):
    """
    Linearly increasing firing rate from start_hz to end_hz.

    Useful for demonstrating progressive wind-up buildup as C-fibre
    input intensifies over time.

    Parameters
    ----------
    start_hz : float
        Initial firing rate in Hz.
    end_hz : float
        Final firing rate in Hz.
    duration_ms : float
        Total duration in milliseconds.
    dt_ms : float
        Time step in milliseconds.

    Returns
    -------
    TimedArray
        Brian2 TimedArray with linearly ramping rate.
    """
    n_steps = int(duration_ms / dt_ms)
    rates = np.linspace(start_hz, end_hz, n_steps) * Hz
    return TimedArray(rates, dt=dt_ms * ms)


def burst_rate(baseline_hz, burst_hz, burst_duration_ms, burst_frequency_hz,
               duration_ms, dt_ms=0.1):
    """
    Periodic bursts of high-frequency firing on a low baseline.

    Models repetitive C-fibre stimulation at 0.5-2 Hz that produces
    the wind-up effect in dorsal horn neurons.

    Parameters
    ----------
    baseline_hz : float
        Background firing rate between bursts.
    burst_hz : float
        Peak firing rate during each burst.
    burst_duration_ms : float
        Duration of each burst in milliseconds.
    burst_frequency_hz : float
        How often bursts occur (e.g. 1.0 for one burst per second).
    duration_ms : float
        Total simulation duration in milliseconds.
    dt_ms : float
        Time step in milliseconds.

    Returns
    -------
    TimedArray
        Brian2 TimedArray with periodic burst pattern.
    """
    n_steps = int(duration_ms / dt_ms)
    rates = np.full(n_steps, baseline_hz)

    if burst_frequency_hz > 0:
        period_ms = 1000.0 / burst_frequency_hz
        burst_steps = int(burst_duration_ms / dt_ms)

        t = 0.0
        while t < duration_ms:
            start_idx = int(t / dt_ms)
            end_idx = min(start_idx + burst_steps, n_steps)
            rates[start_idx:end_idx] = burst_hz
            t += period_ms

    return TimedArray(rates * Hz, dt=dt_ms * ms)


def build_stimulation_protocol(protocol, duration_ms, dt_ms=0.1,
                               c_params=None, ab_params=None):
    """
    Build a complete stimulation protocol returning TimedArrays for
    both C-fibre and Ab-fibre populations.

    Parameters
    ----------
    protocol : str
        One of 'constant', 'ramp', 'burst', 'mixed'.
    duration_ms : float
        Total simulation duration in milliseconds.
    dt_ms : float
        Time step in milliseconds.
    c_params : dict or None
        Override parameters for C-fibre stimulation.
    ab_params : dict or None
        Override parameters for Ab-fibre stimulation.

    Returns
    -------
    dict with keys:
        'c_rates'  : TimedArray for C-fibre PoissonGroup
        'ab_rates' : TimedArray for Ab-fibre PoissonGroup
        'protocol' : str, name of the protocol used
    """
    c_params = c_params or {}
    ab_params = ab_params or {}

    if protocol == 'constant':
        c_rate = c_params.get('rate_hz', 20.0)
        ab_rate = ab_params.get('rate_hz', 1.6)
        c_rates = constant_rate(c_rate, duration_ms, dt_ms)
        ab_rates = constant_rate(ab_rate, duration_ms, dt_ms)

    elif protocol == 'ramp':
        c_start = c_params.get('start_hz', 2.0)
        c_end = c_params.get('end_hz', 40.0)
        ab_rate = ab_params.get('rate_hz', 1.6)
        c_rates = ramp_rate(c_start, c_end, duration_ms, dt_ms)
        ab_rates = constant_rate(ab_rate, duration_ms, dt_ms)

    elif protocol == 'burst':
        c_baseline = c_params.get('baseline_hz', 2.0)
        c_burst = c_params.get('burst_hz', 40.0)
        c_burst_dur = c_params.get('burst_duration_ms', 50.0)
        c_burst_freq = c_params.get('burst_frequency_hz', 1.0)
        ab_rate = ab_params.get('rate_hz', 1.6)
        c_rates = burst_rate(c_baseline, c_burst, c_burst_dur,
                             c_burst_freq, duration_ms, dt_ms)
        ab_rates = constant_rate(ab_rate, duration_ms, dt_ms)

    elif protocol == 'mixed':
        # Both C and Ab active simultaneously — tests gate control
        c_rate = c_params.get('rate_hz', 20.0)
        ab_rate = ab_params.get('rate_hz', 10.0)  # higher Ab to strongly engage gate
        c_rates = constant_rate(c_rate, duration_ms, dt_ms)
        ab_rates = constant_rate(ab_rate, duration_ms, dt_ms)

    else:
        raise ValueError(f"Unknown protocol '{protocol}'. "
                         f"Choose from: constant, ramp, burst, mixed")

    return {
        'c_rates': c_rates,
        'ab_rates': ab_rates,
        'protocol': protocol,
    }


def windup_protocol(n_stimuli=10, stimulus_rate_hz=10.0, stimulus_duration_ms=50.0,
                    isi_ms=150.0, warmup_ms=200.0, dt_ms=0.1):
    """
    Discrete repeated C-fibre stimulation protocol for biologically accurate
    wind-up measurement.

    Uses 5 Hz stimulation to retain sufficient NMDA conductance between stimuli.
    An ISI of 150 ms retains 41% of NMDA conductance (tau_NMDA = 170 ms),
    enabling cumulative NMDA activation across successive stimuli.

    Protocol structure:
        - warmup_ms of silence (allows network to reach equilibrium)
        - n_stimuli bursts of stimulus_duration_ms at stimulus_rate_hz
        - isi_ms of silence between bursts
    Default period: 50 ms burst + 150 ms silence = 200 ms = 5 Hz.

    Parameters
    ----------
    n_stimuli : int
        Number of discrete stimuli.
    stimulus_rate_hz : float
        C-fibre firing rate during each burst (Hz).
    stimulus_duration_ms : float
        Duration of each burst (ms).
    isi_ms : float
        Silent period after each burst (ms).
    warmup_ms : float
        Silent period before the first stimulus (ms).
    dt_ms : float
        Simulation time step (ms).

    Returns
    -------
    tuple : (TimedArray, float, list of float)
        - TimedArray of C-fibre rates
        - total_duration_ms
        - stimulus_onsets_ms: start time of each burst (ms)
    """
    period_ms = stimulus_duration_ms + isi_ms
    total_duration_ms = warmup_ms + n_stimuli * period_ms
    n_steps = int(total_duration_ms / dt_ms)
    rates = np.zeros(n_steps)

    stimulus_onsets_ms = []
    for i in range(n_stimuli):
        onset_ms = warmup_ms + i * period_ms
        onset_idx = int(onset_ms / dt_ms)
        end_idx = min(int((onset_ms + stimulus_duration_ms) / dt_ms), n_steps)
        rates[onset_idx:end_idx] = stimulus_rate_hz
        stimulus_onsets_ms.append(onset_ms)

    return TimedArray(rates * Hz, dt=dt_ms * ms), total_duration_ms, stimulus_onsets_ms


def jitter_parameters(base_params, jitter_fraction=0.1, rng=None):
    """
    Apply random jitter to a parameter dictionary for dataset diversity.

    Each numeric value is multiplied by a factor drawn uniformly from
    [1 - jitter_fraction, 1 + jitter_fraction].

    Parameters
    ----------
    base_params : dict
        Parameter dictionary (e.g. from NeuronParameters or SynapticParameters).
    jitter_fraction : float
        Maximum fractional deviation (0.1 = +/-10%).
    rng : numpy.random.Generator or None
        Random number generator. Uses default if None.

    Returns
    -------
    dict
        New dictionary with jittered values.
    """
    if rng is None:
        rng = np.random.default_rng()

    jittered = {}
    for key, value in base_params.items():
        if isinstance(value, (int, float)):
            factor = rng.uniform(1 - jitter_fraction, 1 + jitter_fraction)
            jittered[key] = value * factor
        else:
            # Brian2 Quantity or non-numeric — try multiplying
            try:
                factor = rng.uniform(1 - jitter_fraction, 1 + jitter_fraction)
                jittered[key] = value * factor
            except (TypeError, ValueError):
                jittered[key] = value
    return jittered
