"""
Spike train feature extraction for the FMS spinal dorsal horn model.

Converts raw simulation output (spike times, neuron indices) into
statistical descriptors suitable for ML classification of healthy
vs. sensitised (FMS) states.

Features extracted:
    - Mean and peak firing rates (WDR and GABA)
    - Inter-spike interval (ISI) statistics: mean, std, CV
    - Wind-up ratio: from dedicated 5 Hz repeated-stimulus simulation
    - Evoked response: mean WDR spikes/neuron/stimulus under repeated noxious stimulation
    - Burst metrics: count, fraction of spikes in bursts
    - Excitation/inhibition balance: WDR rate / GABA rate
"""

import numpy as np
from neurons import NeuronParameters


def _population_isi(spike_times, neuron_indices, n_neurons):
    """
    Compute inter-spike intervals pooled across all neurons in a population.

    Parameters
    ----------
    spike_times : np.ndarray
        Spike times in ms.
    neuron_indices : np.ndarray
        Neuron index for each spike.
    n_neurons : int
        Total number of neurons in the population.

    Returns
    -------
    np.ndarray
        All ISIs in ms (pooled across neurons).
    """
    isis = []
    for n in range(n_neurons):
        mask = neuron_indices == n
        times = np.sort(spike_times[mask])
        if len(times) > 1:
            isis.extend(np.diff(times))
    return np.array(isis) if isis else np.array([])


def _windowed_rate(spike_times, n_neurons, start_ms, end_ms):
    """
    Compute mean firing rate within a time window.

    Parameters
    ----------
    spike_times : np.ndarray
        Spike times in ms.
    n_neurons : int
        Number of neurons.
    start_ms : float
        Window start (ms).
    end_ms : float
        Window end (ms).

    Returns
    -------
    float
        Mean firing rate in Hz within the window.
    """
    window_s = (end_ms - start_ms) / 1000.0
    if window_s <= 0:
        return 0.0
    count = np.sum((spike_times >= start_ms) & (spike_times < end_ms))
    return count / (n_neurons * window_s)


def _count_bursts(spike_times, neuron_indices, n_neurons,
                  max_isi_ms=10.0, min_spikes=3):
    """
    Count burst events across a population.

    A burst is defined as a sequence of >= min_spikes consecutive spikes
    from the same neuron with ISIs all < max_isi_ms.

    Parameters
    ----------
    spike_times : np.ndarray
        Spike times in ms.
    neuron_indices : np.ndarray
        Neuron index for each spike.
    n_neurons : int
        Total number of neurons.
    max_isi_ms : float
        Maximum ISI to be considered within a burst.
    min_spikes : int
        Minimum number of spikes to qualify as a burst.

    Returns
    -------
    tuple (int, int)
        (total burst count, total spikes within bursts)
    """
    total_bursts = 0
    total_burst_spikes = 0

    for n in range(n_neurons):
        mask = neuron_indices == n
        times = np.sort(spike_times[mask])
        if len(times) < min_spikes:
            continue

        isis = np.diff(times)
        in_burst = isis < max_isi_ms

        # Walk through finding consecutive runs of short ISIs
        run_length = 0
        for is_short in in_burst:
            if is_short:
                run_length += 1
            else:
                if run_length >= (min_spikes - 1):
                    total_bursts += 1
                    total_burst_spikes += run_length + 1
                run_length = 0
        # Handle run ending at array boundary
        if run_length >= (min_spikes - 1):
            total_bursts += 1
            total_burst_spikes += run_length + 1

    return total_bursts, total_burst_spikes


def extract_features(results):
    """
    Extract a feature vector from one simulation trial.

    Parameters
    ----------
    results : dict
        Output of run_simulation() containing spike data and metadata.

    Returns
    -------
    dict
        Feature name -> value mapping. All values are scalar floats.
    """
    t_wdr = results['t']
    i_wdr = results['i']
    t_gaba = results['t_gaba']
    i_gaba = results['i_gaba']
    duration_ms = results['duration_ms']
    n_wdr = NeuronParameters.WDR['n_neurons']
    n_gaba = NeuronParameters.GABA['n_neurons']

    features = {}

    # --- Firing rates ---
    features['wdr_mean_rate'] = results['wdr_mean_rate']
    features['gaba_mean_rate'] = results['gaba_mean_rate']

    # Peak instantaneous rate (10 ms bins)
    bin_size_ms = 10.0
    if len(t_wdr) > 0:
        bins = np.arange(0, duration_ms + bin_size_ms, bin_size_ms)
        counts, _ = np.histogram(t_wdr, bins=bins)
        bin_rates = counts / (n_wdr * bin_size_ms / 1000.0)
        features['wdr_peak_rate'] = float(np.max(bin_rates))
    else:
        features['wdr_peak_rate'] = 0.0

    # --- ISI statistics (WDR) ---
    wdr_isis = _population_isi(t_wdr, i_wdr, n_wdr)
    if len(wdr_isis) > 1:
        features['wdr_isi_mean'] = float(np.mean(wdr_isis))
        features['wdr_isi_std'] = float(np.std(wdr_isis))
        features['wdr_isi_cv'] = (features['wdr_isi_std'] / features['wdr_isi_mean']
                                  if features['wdr_isi_mean'] > 0 else 0.0)
    else:
        features['wdr_isi_mean'] = 0.0
        features['wdr_isi_std'] = 0.0
        features['wdr_isi_cv'] = 0.0

    # --- ISI statistics (GABA) ---
    gaba_isis = _population_isi(t_gaba, i_gaba, n_gaba)
    if len(gaba_isis) > 1:
        features['gaba_isi_mean'] = float(np.mean(gaba_isis))
        features['gaba_isi_std'] = float(np.std(gaba_isis))
        features['gaba_isi_cv'] = (features['gaba_isi_std'] / features['gaba_isi_mean']
                                   if features['gaba_isi_mean'] > 0 else 0.0)
    else:
        features['gaba_isi_mean'] = 0.0
        features['gaba_isi_std'] = 0.0
        features['gaba_isi_cv'] = 0.0

    # --- Wind-up ratio and evoked response ---
    if 'windup_ratio' in results:
        features['wdr_windup_ratio'] = float(results['windup_ratio'])
    else:
        features['wdr_windup_ratio'] = 0.0

    if 'per_stimulus_counts' in results and len(results['per_stimulus_counts']) > 0:
        features['wdr_evoked_response'] = float(
            np.mean(results['per_stimulus_counts'])
        )
    else:
        features['wdr_evoked_response'] = 0.0

    # --- Temporal rate dynamics ---
    early_end = duration_ms * 0.2
    late_start = duration_ms * 0.8
    early_rate = _windowed_rate(t_wdr, n_wdr, 0, early_end)
    late_rate = _windowed_rate(t_wdr, n_wdr, late_start, duration_ms)
    features['wdr_early_rate'] = early_rate
    features['wdr_late_rate'] = late_rate

    # --- Burst metrics (WDR) ---
    burst_count, burst_spikes = _count_bursts(t_wdr, i_wdr, n_wdr)
    features['wdr_burst_count'] = float(burst_count)
    total_spikes = len(t_wdr)
    features['wdr_burst_fraction'] = (
        float(burst_spikes) / total_spikes if total_spikes > 0 else 0.0
    )

    # --- Excitation / inhibition balance ---
    if features['gaba_mean_rate'] > 0:
        features['ei_ratio'] = features['wdr_mean_rate'] / features['gaba_mean_rate']
    else:
        features['ei_ratio'] = features['wdr_mean_rate']

    # --- Active neuron fraction ---
    features['wdr_active_fraction'] = float(
        np.sum(results['wdr_spike_count'] > 0) / n_wdr
    )

    # --- Total spike counts ---
    features['wdr_total_spikes'] = float(total_spikes)
    features['gaba_total_spikes'] = float(len(t_gaba))

    return features


def features_to_row(features, state_label, seed=None, protocol=None):
    """
    Convert a feature dict to a flat dict suitable for a DataFrame row.

    Adds metadata columns (label, seed, protocol) alongside features.

    Parameters
    ----------
    features : dict
        Output of extract_features().
    state_label : str
        Class label ('healthy' or 'fibromyalgia').
    seed : int or None
        Random seed used for this trial.
    protocol : str or None
        Stimulation protocol name.

    Returns
    -------
    dict
        Complete row with metadata + features.
    """
    row = {'label': state_label}
    if seed is not None:
        row['seed'] = seed
    if protocol is not None:
        row['protocol'] = protocol
    row.update(features)
    return row
