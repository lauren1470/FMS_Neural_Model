#!/usr/bin/env python
"""
Test Brian2 installation for FMS neural model project
"""

print("Testing Brian2 installation...\n")

# Test 1: Import Brian2
try:
    import brian2 as b2
    from brian2 import *
    print(f"✓ Brian2 imported successfully")
    print(f"  Version: {b2.__version__}")
except ImportError as e:
    print(f"✗ Failed to import Brian2: {e}")
    exit(1)

# Test 2: Check NumPy version
try:
    import numpy as np
    print(f"✓ NumPy imported successfully")
    print(f"  Version: {np.__version__}")
    
    # Check for ptp issue
    if hasattr(np.ndarray, 'ptp'):
        print(f"  ptp method: Available (NumPy < 2.0)")
    else:
        print(f"  ptp method: Not available (NumPy >= 2.0)")
        print(f"  ⚠ Warning: May need Brian2 2.6+ for NumPy 2.x")
except ImportError as e:
    print(f"✗ Failed to import NumPy: {e}")
    exit(1)

# Test 3: Run simple simulation
try:
    print("\nTesting basic Brian2 functionality...")
    start_scope()
    
    # Simple LIF neuron
    eqs = '''
    dv/dt = (1-v)/(10*ms) : 1
    '''
    
    G = NeuronGroup(10, eqs, threshold='v>0.8', reset='v=0', method='euler')
    G.v = 'rand()'
    
    M = SpikeMonitor(G)
    
    run(100*ms)
    
    print(f"✓ Simulation completed successfully")
    print(f"  Runtime: 100 ms")
    print(f"  Spikes recorded: {M.num_spikes}")
    
except Exception as e:
    print(f"✗ Simulation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Check other dependencies
print("\nChecking other dependencies...")

dependencies = {
    'matplotlib': 'Plotting',
    'scipy': 'Scientific computing',
    'sklearn': 'Machine learning',
    'pandas': 'Data handling'
}

for module, purpose in dependencies.items():
    try:
        __import__(module)
        print(f"✓ {module:15s} - {purpose}")
    except ImportError:
        print(f"✗ {module:15s} - NOT INSTALLED (optional)")


print("INSTALLATION TEST COMPLETE")

print("\nYour environment is ready for the FMS neural model project")