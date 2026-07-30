#!/usr/bin/env python
"""Debug script to trace nucleation execution flow."""

import sys
sys.path.insert(0, r'C:\Users\ericb\Documents\GitHub\PSD_opt\mcpbe\src')

import numpy as np
from mcpbe import MCPBESolver

# Create solver
solver = MCPBESolver(
    dim=1,
    t_vec=np.array([0.0, 0.1, 0.2]),
    verbose=True,
    load_attr=False,
    init=True,
    seed=42,
)

solver.a0 = 100
solver.x = np.full(1, 27e-5)
solver.process_type = "nucleation"
solver.liquid_flow_rate = 0.00001  # 10 µL/s
solver.droplet_diameter = 0.0005  # 500 µm
solver.nucleation_enable = True

print("="*70)
print("INITIALIZING SOLVER")
print("="*70)
solver._initialize_particles()

print(f"\na_tot after init: {solver.a_tot}")
print(f"liquid_volume shape: {solver.liquid_volume.shape}")
print(f"Initial total liquid: {np.sum(solver.liquid_volume):.6e} m³")

print("\n" + "="*70)
print("CALLING solve() - will trace execution")
print("="*70)

# Monkey-patch to trace
original_distribute = solver.distribute_droplets_for_timestep
def traced_distribute(dt):
    print(f"\n>>> distribute_droplets_for_timestep(dt={dt:.6e})")
    print(f"    _accumulated_time BEFORE: {solver._accumulated_time:.6e}")
    print(f"    _droplet_rate: {solver._droplet_rate:.6e}")
    result = original_distribute(dt)
    print(f"    _accumulated_time AFTER: {solver._accumulated_time:.6e}")
    print(f"    Returned: {result} droplets")
    return result

solver.distribute_droplets_for_timestep = traced_distribute

try:
    solver.solve(maxiter=100)  # Only 100 iterations for debugging
    
    print("\n" + "="*70)
    print("FINAL STATE")
    print("="*70)
    print(f"Final particles: {solver.a_tot}")
    print(f"Total liquid: {np.sum(solver.liquid_volume[:solver.a_tot]):.6e} m³")
    print(f"_accumulated_time: {solver._accumulated_time:.6e}")
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
