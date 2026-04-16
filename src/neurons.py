from brian2 import pF, mV, Hz, ms, Mohm, nsiemens, pA

class NeuronParameters:
    WDR = {
        # Membrane properties 
        'membrane_capacitance': 35.0 * pF,  
        'resting_potential': -75.0 * mV,      
        'input_resistance': 81.0 * Mohm,
        'leak_conductance': 12.35 * nsiemens,    # 1/input_resistance
        'membrane_time_constant': 2.84 * ms,      # input_resistance * membrane_capacitance

        # Spiking behavior
        'threshold_potential': -35.0 * mV,
        'reset_potential': -78.0 * mV,
        'refractory_period': 3.5 * ms,

        # Spike-frequency adaptation (mAHP)
        'tau_adapt': 150.0 * ms,
        'b_adapt': 80.0 * pA,

        # Population size
        'n_neurons': 50,
    }

    GABA = {
        # Membrane properties
        'membrane_capacitance': 59.0 * pF,                
        'resting_potential': -56.9 * mV, 
        'input_resistance': 407.7 * Mohm,
        'leak_conductance': 2.45 * nsiemens,      
        'membrane_time_constant': 24.05 * ms,

        # Spiking behavior
        'threshold_potential': -35.7 * mV,
        'reset_potential': -59.9 * mV,
        'refractory_period': 1.5 * ms,          

        # Population size
        'n_neurons': 30,
    }

    C_FIBRE = {
        'n_fibres': 100,
        'noxious_rate': 20.0 * Hz,
    }

    AB_FIBRE = {
        'n_fibres': 200,
        'innocuous_rate': 1.6 * Hz,
    }

    @staticmethod
    def get_lif_equations(neuron_type='WDR'):
        """
        Generate Leaky Integrate-and-Fire equations for given neuron type.

        Each synapse pathway writes to its own (summed) variable in the neuron,
        because Brian2 requires a unique summed target per Synapses group.
        The dv/dt equation sums all pathway contributions.

        WDR receives: I_c_wdr (C-fibre AMPA+NMDA), I_ab_wdr (Ab AMPA), I_gaba_wdr (GABA_A)
        GABA receives: I_ab_gaba (Ab AMPA), I_c_gaba (C-fibre AMPA)
        """

        if neuron_type == 'WDR':
            return '''
            dv/dt = (leak_conductance * (resting_potential - v) + I_c_wdr + I_ab_wdr + I_gaba_wdr - w_adapt) / membrane_capacitance : volt (unless refractory)
            dw_adapt/dt = -w_adapt / tau_adapt : amp    # Spike-frequency adaptation (mAHP)
            tau_adapt : second (constant)
            b_adapt : amp (constant)
            membrane_capacitance : farad (constant)
            leak_conductance : siemens (constant)
            resting_potential : volt (constant)
            I_c_wdr : amp       # C-fibre -> WDR (AMPA + NMDA excitation)
            I_ab_wdr : amp      # Ab-fibre -> WDR (AMPA excitation)
            I_gaba_wdr : amp    # GABA -> WDR (inhibition / the gate)
            '''

        elif neuron_type == 'GABA':
            return '''
            dv/dt = (leak_conductance * (resting_potential - v) + I_ab_gaba + I_c_gaba) / membrane_capacitance : volt (unless refractory)
            membrane_capacitance : farad (constant)
            leak_conductance : siemens (constant)
            resting_potential : volt (constant)
            I_ab_gaba : amp     # Ab-fibre -> GABA (AMPA excitation)
            I_c_gaba : amp      # C-fibre -> GABA (AMPA excitation)
            '''

        else:
            raise ValueError(f"Unknown neuron type: {neuron_type}")
