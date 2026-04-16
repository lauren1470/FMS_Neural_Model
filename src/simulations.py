import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from brian2 import (
    NeuronGroup, Synapses, PoissonGroup, SpikeMonitor, StateMonitor,
    Network, defaultclock, start_scope, ms, Hz, mV, nS, pF, pA, second
)

from neurons import NeuronParameters
from synapses import SynapticParameters, PathologyStates



# --- Simulation defaults ---
DEFAULT_DURATION_MS = 1000   # Simulation length in milliseconds
DEFAULT_DT_MS       = 0.1    # Time step (ms) — 0.1 ms is standard for LIF+NMDA
FIGURES_DIR         = os.path.join("figures", "simulations")



# --- Core simulation function ---

def run_simulation(state: str = 'healthy',
                   duration_ms: float = DEFAULT_DURATION_MS,
                   seed: int = None,
                   verbose: bool = True,
                   c_rates=None,
                   ab_rates=None) -> dict:
    """
    Build and run one trial of the dorsal horn circuit.

    Parameters
    ----------
    state : str
        Pathology state: 'healthy', 'fibromyalgia', or 'intervention'.
    duration_ms : float
        Simulation duration in milliseconds.
    seed : int or None
        Random seed for reproducibility. Pass None for a random seed.
    verbose : bool
        If True, print status messages.
    c_rates : brian2.TimedArray or None
        Optional time-varying C-fibre rates (from stimulation.py).
        If None, uses the constant noxious_rate from NeuronParameters.
    ab_rates : brian2.TimedArray or None
        Optional time-varying Ab-fibre rates (from stimulation.py).
        If None, uses the constant innocuous_rate from NeuronParameters.

    Returns
    -------
    dict with keys:
        'state'         : str, name of the pathology state
        'duration_ms'   : float
        'seed'          : int used
        't'             : np.ndarray, spike times for WDR (ms)
        'i'             : np.ndarray, neuron indices for WDR spikes
        't_gaba'        : np.ndarray, spike times for GABA neurons (ms)
        'i_gaba'        : np.ndarray, neuron indices for GABA spikes
        'wdr_spike_count': np.ndarray, total spikes per WDR neuron
        'wdr_mean_rate' : float, mean WDR firing rate (Hz)
        'gaba_mean_rate': float, mean GABA firing rate (Hz)
        'v_trace'       : np.ndarray, voltage trace of WDR neuron 0 (mV)
        'v_times'       : np.ndarray, time axis for voltage trace (ms)
        'pathology'     : dict, the pathology configuration used
    """

    # --- 0. Setup ---
    if duration_ms <= 0:
        raise ValueError(f"duration_ms must be positive, got {duration_ms}")

    start_scope()  # Clear all Brian2 objects from memory

    if seed is not None:
        np.random.seed(seed)

    defaultclock.dt = DEFAULT_DT_MS * ms
    duration = duration_ms * ms

    # Resolve pathology state
    state_map = {
        'healthy':       PathologyStates.healthy,
        'fibromyalgia':  PathologyStates.fibromyalgia,
        'intervention':  PathologyStates.intervention,
    }
    if state not in state_map:
        raise ValueError(f"Unknown state '{state}'. Choose from: {list(state_map)}")

    pathology = state_map[state]()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  FMS SIMULATION - {pathology['name'].upper()}")
        print(f"  {pathology['description']}")
        print(f"  Duration: {duration_ms} ms  |  dt: {DEFAULT_DT_MS} ms  |  seed: {seed}")
        print(f"{'='*60}")

    # --- 1. Shorthand aliases for parameter dicts ---
    P_WDR   = NeuronParameters.WDR
    P_GABA  = NeuronParameters.GABA
    P_C     = NeuronParameters.C_FIBRE
    P_AB    = NeuronParameters.AB_FIBRE
    P_AMPA  = SynapticParameters.AMPA
    P_NMDA  = SynapticParameters.NMDA
    P_GABAA = SynapticParameters.GABA_A
    P_CONN  = SynapticParameters.CONNECTIVITY
    GAIN    = SynapticParameters.GAIN

    # Select AMPA weight set based on state
    if state == 'healthy':
        ampa_w = SynapticParameters.AMPA_weights_healthy
    else:
        ampa_w = SynapticParameters.AMPA_weights_sensitised

    # --- 2. Afferent fibre input (Poisson spike generators) ---
    if c_rates is not None:
        # Time-varying rates from stimulation protocol
        c_fibres = PoissonGroup(P_C['n_fibres'], rates='c_rate_array(t)')
        c_fibres.namespace['c_rate_array'] = c_rates
        c_rate_label = 'TimedArray'
    else:
        c_fibres = PoissonGroup(P_C['n_fibres'], rates=P_C['noxious_rate'])
        c_rate_label = str(P_C['noxious_rate'])

    if ab_rates is not None:
        ab_fibres = PoissonGroup(P_AB['n_fibres'], rates='ab_rate_array(t)')
        ab_fibres.namespace['ab_rate_array'] = ab_rates
        ab_rate_label = 'TimedArray'
    else:
        ab_fibres = PoissonGroup(P_AB['n_fibres'], rates=P_AB['innocuous_rate'])
        ab_rate_label = str(P_AB['innocuous_rate'])

    if verbose:
        print(f"\n[INPUTS]")
        print(f"  C-fibres  : {P_C['n_fibres']} neurons @ {c_rate_label}")
        print(f"  AB-fibres : {P_AB['n_fibres']} neurons @ {ab_rate_label}")

    # --- 3. Neuron populations ---

    # WDR projection neurons  
    wdr_eqs = NeuronParameters.get_lif_equations('WDR')
    wdr_neurons = NeuronGroup(
        P_WDR['n_neurons'],
        model=wdr_eqs,
        threshold='v > threshold_potential',
        reset='v = reset_potential; w_adapt += b_adapt',
        refractory=P_WDR['refractory_period'],
        method='euler',
        namespace={
            'threshold_potential': P_WDR['threshold_potential'],
            'reset_potential':     P_WDR['reset_potential'],
        }
    )
    # Initialise membrane parameters
    wdr_neurons.membrane_capacitance = P_WDR['membrane_capacitance']
    wdr_neurons.leak_conductance     = P_WDR['leak_conductance']
    wdr_neurons.resting_potential     = P_WDR['resting_potential']
    wdr_neurons.tau_adapt             = P_WDR['tau_adapt']
    wdr_neurons.b_adapt               = P_WDR['b_adapt']
    wdr_neurons.w_adapt               = 0 * pA
    # Initialise voltage to resting potential with small noise
    wdr_neurons.v = P_WDR['resting_potential'] + np.random.uniform(-2, 2, P_WDR['n_neurons']) * mV

    # GABAergic interneurons
    gaba_eqs = NeuronParameters.get_lif_equations('GABA')
    gaba_neurons = NeuronGroup(
        P_GABA['n_neurons'],
        model=gaba_eqs,
        threshold='v > threshold_potential',
        reset='v = reset_potential',
        refractory=P_GABA['refractory_period'],
        method='euler',
        namespace={
            'threshold_potential': P_GABA['threshold_potential'],
            'reset_potential':     P_GABA['reset_potential'],
        }
    )
    gaba_neurons.membrane_capacitance = P_GABA['membrane_capacitance']
    gaba_neurons.leak_conductance     = P_GABA['leak_conductance']
    gaba_neurons.resting_potential     = P_GABA['resting_potential']
    gaba_neurons.v = P_GABA['resting_potential'] + np.random.uniform(-2, 2, P_GABA['n_neurons']) * mV

    if verbose:
        print(f"\n[NEURON POPULATIONS]")
        print(f"  WDR neurons  : {P_WDR['n_neurons']}  (threshold: {P_WDR['threshold_potential']})")
        print(f"  GABA neurons : {P_GABA['n_neurons']} (threshold: {P_GABA['threshold_potential']})")

    # --- 4. Synapses ---

    glu_on_pre  = SynapticParameters.get_glutamate_on_pre()
    ampa_on_pre = SynapticParameters.get_ampa_only_on_pre()
    gaba_on_pre = SynapticParameters.get_gaba_on_pre()

    # --- 4a. C-fibres -> WDR (AMPA + NMDA, excitatory) ---
    syn_c_wdr = Synapses(
        c_fibres, wdr_neurons,
        model=SynapticParameters.get_glutamate_synapse_equations(target_var='I_c_wdr'),
        on_pre=glu_on_pre,
        method='euler'
    )
    syn_c_wdr.connect(p=P_CONN['c_to_wdr'])

    # Conductance weights (GAIN * g_max * dimensionless weight)
    w_ampa_c_wdr = GAIN * P_AMPA['g_max_c_to_wdr'] * ampa_w['c_to_wdr']
    w_nmda_c_wdr = GAIN * P_NMDA['g_max_c_to_wdr'] * pathology['w_nmda_c_to_wdr']

    syn_c_wdr.w_AMPA  = w_ampa_c_wdr
    syn_c_wdr.w_NMDA  = w_nmda_c_wdr
    syn_c_wdr.g_AMPA  = 0 * nS
    syn_c_wdr.g_NMDA  = 0 * nS
    syn_c_wdr.T_AMPA  = P_AMPA['T_decay']
    syn_c_wdr.T_NMDA  = P_NMDA['T_decay']
    syn_c_wdr.E_glu   = P_AMPA['E_glu']
    syn_c_wdr.mg_conc       = P_NMDA['mg_conc']
    syn_c_wdr.mg_factor     = P_NMDA['mg_factor']
    syn_c_wdr.voltage_slope = P_NMDA['voltage_slope']

    # --- 4b. AB-fibres → WDR (AMPA only, weak direct excitation) ---
    syn_ab_wdr = Synapses(
        ab_fibres, wdr_neurons,
        model=SynapticParameters.get_ampa_only_synapse_equations(target_var='I_ab_wdr'),
        on_pre=ampa_on_pre,
        method='euler'
    )
    syn_ab_wdr.connect(p=P_CONN['ab_to_wdr'])
    syn_ab_wdr.w_AMPA = GAIN * P_AMPA['g_max_ab_to_wdr'] * ampa_w['ab_to_wdr']
    syn_ab_wdr.g_AMPA = 0 * nS
    syn_ab_wdr.T_AMPA = P_AMPA['T_decay']
    syn_ab_wdr.E_glu  = P_AMPA['E_glu']

    # --- 4c. AB-fibres -> GABA interneurons (AMPA, strong -- activates the gate) ---
    syn_ab_gaba = Synapses(
        ab_fibres, gaba_neurons,
        model=SynapticParameters.get_ampa_only_synapse_equations(target_var='I_ab_gaba'),
        on_pre=ampa_on_pre,
        method='euler'
    )
    syn_ab_gaba.connect(p=P_CONN['ab_to_gaba'])
    syn_ab_gaba.w_AMPA = P_AMPA['g_max_ab_to_gaba'] * ampa_w['ab_to_gaba']
    syn_ab_gaba.g_AMPA = 0 * nS
    syn_ab_gaba.T_AMPA = P_AMPA['T_decay']
    syn_ab_gaba.E_glu  = P_AMPA['E_glu']

    # --- 4d. C-fibres -> GABA interneurons (AMPA, weaker feedforward inhibition) ---
    syn_c_gaba = Synapses(
        c_fibres, gaba_neurons,
        model=SynapticParameters.get_ampa_only_synapse_equations(target_var='I_c_gaba'),
        on_pre=ampa_on_pre,
        method='euler'
    )
    syn_c_gaba.connect(p=P_CONN['c_to_gaba'])
    syn_c_gaba.w_AMPA = P_AMPA['g_max_c_to_gaba'] * ampa_w['c_to_gaba']
    syn_c_gaba.g_AMPA = 0 * nS
    syn_c_gaba.T_AMPA = P_AMPA['T_decay']
    syn_c_gaba.E_glu  = P_AMPA['E_glu']

    # --- 4e. GABA interneurons -> WDR (GABA_A, THE GATE) ---
    syn_gaba_wdr = Synapses(
        gaba_neurons, wdr_neurons,
        model=SynapticParameters.get_gaba_synapse_equations(target_var='I_gaba_wdr'),
        on_pre=gaba_on_pre,
        method='euler'
    )
    syn_gaba_wdr.connect(p=P_CONN['gaba_to_wdr'])

    w_gaba_effective = GAIN * P_GABAA['g_max_gaba_to_wdr'] * pathology['w_gaba_to_wdr']
    syn_gaba_wdr.w_GABA = w_gaba_effective
    syn_gaba_wdr.g_GABA = 0 * nS
    syn_gaba_wdr.T_GABA = P_GABAA['T_decay']
    syn_gaba_wdr.E_GABA = P_GABAA['E_GABA']

    if verbose:
        print(f"\n[KEY SYNAPTIC WEIGHTS]")
        print(f"  NMDA C->WDR : {w_nmda_c_wdr / nS:.3f} nS  (pathology factor: {pathology['w_nmda_c_to_wdr']}x)")
        print(f"  GABA gate   : {w_gaba_effective / nS:.3f} nS  (pathology factor: {pathology['w_gaba_to_wdr']}x)")
        print(f"  AMPA C->WDR : {w_ampa_c_wdr / nS:.3f} nS")

    # --- 5. Monitors ---
    spike_mon_wdr  = SpikeMonitor(wdr_neurons)
    spike_mon_gaba = SpikeMonitor(gaba_neurons)

    # Record voltage trace from WDR neuron 0
    state_mon_wdr = StateMonitor(wdr_neurons, 'v', record=[0])

    # --- 6. Run the simulation ---
    if verbose:
        print(f"\n[RUNNING SIMULATION ...]")

    net = Network(
        c_fibres, ab_fibres,
        wdr_neurons, gaba_neurons,
        syn_c_wdr, syn_ab_wdr, syn_ab_gaba, syn_c_gaba, syn_gaba_wdr,
        spike_mon_wdr, spike_mon_gaba, state_mon_wdr
    )
    net.run(duration)


    # --- 7. Extract results ---
    t_wdr  = spike_mon_wdr.t / ms
    i_wdr  = spike_mon_wdr.i[:]
    t_gaba = spike_mon_gaba.t / ms
    i_gaba = spike_mon_gaba.i[:]

    # Spike count per WDR neuron
    wdr_spike_count = np.array([np.sum(i_wdr == n) for n in range(P_WDR['n_neurons'])])

    # Mean firing rates
    wdr_mean_rate  = len(t_wdr)  / (P_WDR['n_neurons']  * duration_ms / 1000)
    gaba_mean_rate = len(t_gaba) / (P_GABA['n_neurons'] * duration_ms / 1000)

    if verbose:
        print(f"\n[RESULTS]")
        print(f"  WDR  total spikes : {len(t_wdr)}")
        print(f"  GABA total spikes : {len(t_gaba)}")
        print(f"  WDR  mean rate    : {wdr_mean_rate:.2f} Hz")
        print(f"  GABA mean rate    : {gaba_mean_rate:.2f} Hz")
        print(f"  WDR  active neurons: {np.sum(wdr_spike_count > 0)} / {P_WDR['n_neurons']}")

    return {
        'state':           pathology['name'],
        'duration_ms':     duration_ms,
        'seed':            seed,
        't':               t_wdr,
        'i':               i_wdr,
        't_gaba':          t_gaba,
        'i_gaba':          i_gaba,
        'wdr_spike_count': wdr_spike_count,
        'wdr_mean_rate':   wdr_mean_rate,
        'gaba_mean_rate':  gaba_mean_rate,
        'v_trace':         state_mon_wdr.v[0] / mV,
        'v_times':         state_mon_wdr.t / ms,
        'pathology':       pathology,
    }



# --- Wind-up measurement ---

def run_windup_simulation(state: str = 'healthy', seed: int = None,
                          n_stimuli: int = 10,
                          response_window_ms: float = 200.0) -> dict:
    """
    Run a dedicated wind-up measurement simulation using discrete repeated
    C-fibre stimulation at 5 Hz.

    Wind-up ratio formulation:
        ratio = total_evoked / (n_stimuli x first_stimulus_response)
    where total_evoked is summed WDR spikes/neuron across all stimuli and
    first_stimulus_response is floored at 0.05 to prevent division by zero.
    A ratio > 1 indicates progressive NMDA-mediated amplification (wind-up).

    Parameters
    ----------
    state : str
        Pathology state ('healthy', 'fibromyalgia', or 'intervention').
    seed : int or None
        Random seed.
    n_stimuli : int
        Number of discrete C-fibre bursts (default 10).
    response_window_ms : float
        Time window after each stimulus onset to count WDR spikes (ms).
        Set to 200 ms to capture the primary NMDA-mediated response
        (extends beyond one NMDA time constant: τ_NMDA = 170 ms).

    Returns
    -------
    dict with keys:
        'windup_ratio'         : float, Fieldwalker total/predicted ratio
        'per_stimulus_counts'  : list of float, mean WDR spikes per neuron per stimulus
        'stimulus_onsets_ms'   : list of float, onset time of each stimulus (ms)
    """
    from stimulation import windup_protocol

    c_rates, total_duration_ms, stimulus_onsets_ms = windup_protocol(
        n_stimuli=n_stimuli,
        stimulus_rate_hz=10.0,       # Sub-maximal noxious C-fibre input
        stimulus_duration_ms=50.0,   # Brief burst per stimulus
        isi_ms=150.0,                # 200 ms period = 5 Hz; retains 41% NMDA
        warmup_ms=200.0,             # Network equilibration before first stimulus
    )

    # Run circuit with wind-up protocol (no Ab input — pure C-fibre wind-up test)
    results = run_simulation(
        state=state,
        duration_ms=total_duration_ms,
        seed=seed,
        verbose=False,
        c_rates=c_rates,
        ab_rates=None,
    )

    t_wdr = results['t']
    n_wdr = NeuronParameters.WDR['n_neurons']

    # Count WDR spikes within the response window after each stimulus onset.
    per_stimulus_counts = []
    for onset_ms in stimulus_onsets_ms:
        count = np.sum(
            (t_wdr >= onset_ms) & (t_wdr < onset_ms + response_window_ms)
        )
        per_stimulus_counts.append(count / n_wdr)

    # Wind-up ratio: total_evoked / (n_stimuli x first_stimulus_response).
    # Floor of 0.05 spikes/neuron prevents division by zero; capped at 8.0x.
    FIRST_FLOOR = 0.05   # spikes per neuron
    MAX_RATIO = 8.0
    total_evoked = sum(per_stimulus_counts)
    first_floored = max(per_stimulus_counts[0], FIRST_FLOOR)
    predicted = n_stimuli * first_floored
    windup_ratio = min(total_evoked / predicted, MAX_RATIO)

    return {
        'windup_ratio':        windup_ratio,
        'per_stimulus_counts': per_stimulus_counts,
        'stimulus_onsets_ms':  stimulus_onsets_ms,
    }


# --- Plotting ---

def plot_results(results: dict, save: bool = True) -> None:
    """
    Produce a four-panel diagnostic figure for one simulation trial.

    Panels:
        1. WDR raster plot (spike times vs neuron index)
        2. GABA raster plot
        3. WDR population firing rate (10 ms bins)
        4. Voltage trace of WDR neuron 0
    """
    state_name = results['state']
    colour     = {'Healthy': '#4472C4', 'Fibromyalgia': '#E91E63', 'Intervention': '#9B59B6'}
    c          = colour.get(state_name, '#7E57C2')

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f"Spinal Dorsal Horn Circuit — {state_name}\n"
        f"({results['duration_ms']} ms simulation  |  "
        f"WDR mean rate: {results['wdr_mean_rate']:.1f} Hz  |  "
        f"GABA mean rate: {results['gaba_mean_rate']:.1f} Hz)",
        fontsize=13, fontweight='bold'
    )

    gs = gridspec.GridSpec(4, 1, hspace=0.55, figure=fig)

    # Panel 1: WDR raster 
    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(results['t'], results['i'], s=2, color=c, alpha=0.7, rasterized=True)
    ax1.set_xlim(0, results['duration_ms'])
    ax1.set_ylim(-1, NeuronParameters.WDR['n_neurons'])
    ax1.set_ylabel('WDR neuron #', fontsize=9)
    ax1.set_title('WDR Projection Neuron Raster', fontsize=10)
    ax1.tick_params(labelbottom=False)

    # Panel 2: GABA raster 
    ax2 = fig.add_subplot(gs[1])
    ax2.scatter(results['t_gaba'], results['i_gaba'], s=2, color='#CE93D8', alpha=0.7, rasterized=True)
    ax2.set_xlim(0, results['duration_ms'])
    ax2.set_ylim(-1, NeuronParameters.GABA['n_neurons'])
    ax2.set_ylabel('GABA neuron #', fontsize=9)
    ax2.set_title('GABAergic Interneuron Raster (The Gate)', fontsize=10)
    ax2.tick_params(labelbottom=False)

    # Panel 3: WDR population firing rate 
    ax3 = fig.add_subplot(gs[2])
    bin_size_ms = 10
    bins = np.arange(0, results['duration_ms'] + bin_size_ms, bin_size_ms)
    counts, edges = np.histogram(results['t'], bins=bins)
    # Convert to Hz: divide by (n_neurons × bin duration in seconds)
    rates = counts / (NeuronParameters.WDR['n_neurons'] * bin_size_ms / 1000)
    ax3.bar(edges[:-1], rates, width=bin_size_ms * 0.9, color=c, alpha=0.7, align='edge')
    ax3.set_xlim(0, results['duration_ms'])
    ax3.set_ylabel('WDR rate (Hz)', fontsize=9)
    ax3.set_title('WDR Population Firing Rate (10 ms bins)', fontsize=10)
    ax3.tick_params(labelbottom=False)

    # Panel 4: Voltage trace 
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(results['v_times'], results['v_trace'], color=c, linewidth=0.6, alpha=0.85)
    ax4.axhline(
        y=NeuronParameters.WDR['threshold_potential'] / mV,
        color='#FF80AB', linestyle='--', linewidth=0.8, label='Threshold'
    )
    ax4.set_xlim(0, results['duration_ms'])
    ax4.set_xlabel('Time (ms)', fontsize=9)
    ax4.set_ylabel('Membrane voltage (mV)', fontsize=9)
    ax4.set_title('WDR Neuron 0 — Voltage Trace', fontsize=10)
    ax4.legend(fontsize=8, loc='upper right')

    plt.tight_layout()

    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fname = os.path.join(FIGURES_DIR, f"sim_{state_name.lower()}.png")
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"\n  Figure saved → {fname}")

    plt.show()



# --- Convenience: run all three states and overlay firing rates ---

def compare_states(duration_ms: float = DEFAULT_DURATION_MS, seed: int = 42) -> None:
    """
    Run healthy, FMS, and intervention simulations and produce a
    side-by-side firing rate comparison bar chart.
    """
    states   = ['healthy', 'fibromyalgia', 'intervention']
    colours  = ['#4472C4', '#E91E63', '#9B59B6']
    wdr_rates, gaba_rates, labels = [], [], []

    for s in states:
        r = run_simulation(state=s, duration_ms=duration_ms, seed=seed)
        plot_results(r, save=True)
        wdr_rates.append(r['wdr_mean_rate'])
        gaba_rates.append(r['gaba_mean_rate'])
        labels.append(r['state'])

    # Bar chart comparison
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, wdr_rates,  width, label='WDR (pain signal)',  color=colours, alpha=0.85)
    bars2 = ax.bar(x + width/2, gaba_rates, width, label='GABA (gate)',         color=colours, alpha=0.45, hatch='//')

    ax.set_ylabel('Mean firing rate (Hz)')
    ax.set_title('WDR vs GABA Mean Firing Rates Across Pathology States')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.bar_label(bars1, fmt='%.1f Hz', padding=3, fontsize=8)
    ax.bar_label(bars2, fmt='%.1f Hz', padding=3, fontsize=8)

    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fname = os.path.join(FIGURES_DIR, "comparison_all_states.png")
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"\n  Comparison figure saved → {fname}")
    plt.show()



# --- Entry point ---

if __name__ == '__main__':
    print("\nRunning single healthy simulation first (validation check)...")
    results = run_simulation(state='healthy', duration_ms=1000, seed=0)
    plot_results(results, save=True)

    print("\n\nRunning full three-state comparison...")
    compare_states(duration_ms=1000, seed=42)

    print("\n[DONE] Check the 'figures/' directory for output plots.")