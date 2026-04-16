from brian2 import ms, mV, nS

class SynapticParameters:
    # Global gain applied to WDR-targeting synapses to produce
    # physiologically realistic firing rates while preserving pathway ratios.
    GAIN = 5.5

    # GLUTAMATE RECEPTORS (Excitatory)

    AMPA = {
        # Kinetics
        'T_decay': 6 * ms,  
        
        'E_glu': 0 * mV,
        
        # Maximum conductances 
        'g_max_c_to_wdr': 0.25 * nS,      # C-fibre -> WDR baseline
        'g_max_ab_to_wdr': 0.15 * nS,     # Aβ -> WDR baseline  
        'g_max_ab_to_gaba': 1.2 * nS,     # Aβ -> GABA (strong)
        'g_max_c_to_gaba': 0.4 * nS,      # C -> GABA (weaker)
    }
    
    # Synaptic weights (dimensionless scaling factors)
    AMPA_weights_healthy = {
        'c_to_wdr': 1.0,        # Baseline (effective: 0.25 nS)
        'ab_to_wdr': 0.8,       # Weak - shouldn't drive WDR alone
        'ab_to_gaba': 1.0,      # Strong - activates gate
        'c_to_gaba': 1.0,       # Moderate
    }
    
    AMPA_weights_sensitised = {
        'c_to_wdr': 1.0,
        'ab_to_wdr': 0.8,
        'ab_to_gaba': 1.0,
        'c_to_gaba': 1.0,
    }
    
    NMDA = {
        # Kinetics
        'T_decay': 170 * ms,  # Much slower than AMPA — enables temporal summation
        
        # Reversal potential
        'E_glu': 0 * mV,
        
        # Maximum conductances
        'g_max_c_to_wdr': 0.25 * nS,  # Equal to AMPA conductance
        
        # Mg2+ block parameters 
        'mg_conc': 1.0,     # mM (extracellular Mg2+)
        'mg_factor': 0.28,           # Dimensionless constant
        'voltage_slope': 0.062,      # /mV (how fast block releases)
    }
    
    # NMDA weights
    NMDA_weights_healthy = {
        'c_to_wdr': 1.0,  
    }
    
    NMDA_weights_sensitised = {
        'c_to_wdr': 3.0,  
    }

    # GABA RECEPTORS (Inhibitory)

    GABA_A = {
        # Kinetics
        'T_decay': 20 * ms,  
        
        # Reversal potential
        'E_GABA': -70 * mV,
        
        # Maximum conductances
        'g_max_gaba_to_wdr': 0.8 * nS,  # Strong inhibition when working
    }
    
    # GABA weights
    GABA_weights_healthy = {
        'gaba_to_wdr': 1.0,  # Full strength (gate working)
    }
    
    GABA_weights_sensitised = {
        'gaba_to_wdr': 0.40,  # 60% REDUCTION (gate impaired)
    }

    # CONNECTION PROBABILITIES
    
    CONNECTIVITY = {
        'c_to_wdr': 0.25,        # 25% of C-fibres connect to each WDR
        'c_to_gaba': 0.20,       # 20% of C-fibres connect to each GABA
        'ab_to_wdr': 0.15,       # 15% of Aβ connect to each WDR 
        'ab_to_gaba': 0.40,      # 40% of Aβ connect to each GABA 
        'gaba_to_wdr': 0.50,     # 50% of GABA neurons connect to each WDR
    }

    @staticmethod
    def get_glutamate_synapse_equations(target_var='I_c_wdr'):
        """
        Returns Brian2 equations for dual AMPA + NMDA glutamate transmission.

        Parameters
        ----------
        target_var : str
            Name of the (summed) variable in the postsynaptic neuron group.
            Must be unique per Synapses group.
        """

        return f'''
        # AMPA receptor conductance (fast component)
        dg_AMPA/dt = -g_AMPA / T_AMPA : siemens (clock-driven)
        T_AMPA : second (constant)
        w_AMPA : siemens (constant)

        # NMDA receptor conductance (slow component - enables temporal summation)
        dg_NMDA/dt = -g_NMDA / T_NMDA : siemens (clock-driven)
        T_NMDA : second (constant)
        w_NMDA : siemens (constant)

        # Mg2+ voltage-dependent block of NMDA
        # B = 1 / (1 + [Mg2+] * eta * exp(-gamma * V))
        # At rest (v~-75mV): heavily blocked (~95%)
        # During depolarisation (v~-40mV): partially unblocked
        # This creates the wind-up effect
        B_NMDA = 1 / (1 + mg_conc * mg_factor * exp(-voltage_slope * v_post/mV)) : 1
        mg_conc : 1 (constant)          # Extracellular [Mg2+] in mM
        mg_factor : 1 (constant)        # Dimensionless scaling constant
        voltage_slope : 1 (constant)    # Voltage sensitivity (/mV)

        # Total synaptic current delivered to postsynaptic neuron
        {target_var}_post = (g_AMPA * (E_glu - v_post) +
                          g_NMDA * B_NMDA * (E_glu - v_post)) : amp (summed)

        E_glu : volt (constant)  # Glutamate reversal potential
        '''
    
    @staticmethod
    def get_glutamate_on_pre():
        """Increment AMPA and NMDA conductances on each presynaptic spike."""
        return '''
        g_AMPA += w_AMPA
        g_NMDA += w_NMDA
        '''
    
    @staticmethod
    def get_ampa_only_synapse_equations(target_var='I_ab_wdr'):
        """
        Returns Brian2 equations for AMPA-only glutamate transmission.

        Parameters
        ----------
        target_var : str
            Name of the (summed) variable in the postsynaptic neuron group.

        Used for synapse pathways targeting GABA interneurons and for
        the weak Ab->WDR connection, where NMDA is not modelled.
        """
        return f'''
        dg_AMPA/dt = -g_AMPA / T_AMPA : siemens (clock-driven)
        T_AMPA : second (constant)
        w_AMPA : siemens (constant)
        {target_var}_post = g_AMPA * (E_glu - v_post) : amp (summed)
        E_glu : volt (constant)
        '''

    @staticmethod
    def get_ampa_only_on_pre():
        """Increment AMPA conductance on each presynaptic spike."""
        return 'g_AMPA += w_AMPA'

    @staticmethod
    def get_gaba_synapse_equations(target_var='I_gaba_wdr'):
        """
        Returns Brian2 equations for GABAergic inhibition.

        Parameters
        ----------
        target_var : str
            Name of the (summed) variable in the postsynaptic neuron group.
        """

        return f'''
        # GABA receptor conductance
        dg_GABA/dt = -g_GABA / T_GABA : siemens (clock-driven)
        T_GABA : second (constant)
        w_GABA : siemens (constant)

        # Inhibitory current (negative because E_GABA < v_post)
        {target_var}_post = g_GABA * (E_GABA - v_post) : amp (summed)

        E_GABA : volt (constant)  # GABA reversal potential (hyperpolarizing)
        '''
    
    @staticmethod
    def get_gaba_on_pre():
        """Increment GABA conductance on each presynaptic spike."""
        return '''
        g_GABA += w_GABA
        '''

class PathologyStates:
    """Defines healthy, FMS, and intervention synaptic weight configurations."""
    
    @staticmethod
    def healthy():
        """Normal synaptic weights: full GABA gate, baseline NMDA."""
        return {
            'name': 'Healthy',
            'description': 'Normal gate control, moderate NMDA',
            'w_nmda_c_to_wdr': SynapticParameters.NMDA_weights_healthy['c_to_wdr'],
            'w_gaba_to_wdr': SynapticParameters.GABA_weights_healthy['gaba_to_wdr'],
        }
    
    @staticmethod
    def fibromyalgia():
        """FMS pathology: 3x NMDA weight, 0.4x GABA weight."""
        return {
            'name': 'Fibromyalgia',
            'description': 'Broken gate (disinhibition) + enhanced NMDA (wind-up)',
            'w_nmda_c_to_wdr': SynapticParameters.NMDA_weights_sensitised['c_to_wdr'],
            'w_gaba_to_wdr': SynapticParameters.GABA_weights_sensitised['gaba_to_wdr'],
        }
    
    @staticmethod
    def intervention(nmda_reduction=0.5, gaba_restoration=0.6):
        """
        Interpolate synaptic weights between FMS and healthy states.

        Parameters
        ----------
        nmda_reduction : float (0-1)
            Fractional reduction of NMDA weight from FMS level (1.0 = full reduction to healthy).
        gaba_restoration : float (0-1)
            Fractional restoration of GABA weight toward healthy level (1.0 = full restoration).
        """
        
        fms_nmda = SynapticParameters.NMDA_weights_sensitised['c_to_wdr']
        healthy_nmda = SynapticParameters.NMDA_weights_healthy['c_to_wdr']
        fms_gaba = SynapticParameters.GABA_weights_sensitised['gaba_to_wdr']
        healthy_gaba = SynapticParameters.GABA_weights_healthy['gaba_to_wdr']
        
        # Interpolate between FMS and healthy
        new_nmda = fms_nmda - nmda_reduction * (fms_nmda - healthy_nmda)
        new_gaba = fms_gaba + gaba_restoration * (healthy_gaba - fms_gaba)

        return {
            'name': 'Intervention',
            'description': f'NMDA reduced {nmda_reduction*100:.0f}%, GABA restored {gaba_restoration*100:.0f}%',
            'w_nmda_c_to_wdr': new_nmda,
            'w_gaba_to_wdr': new_gaba,
        }
