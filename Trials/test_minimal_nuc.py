#!/usr/bin/env python
"""Minimal test to debug nucleation."""

import sys
sys.path.insert(0, r'C:\Users\ericb\Documents\GitHub\PSD_opt\mcpbe\src')

import numpy as np
from mcpbe import MCPBESolver

print("Creating solver...")
solver = MCPBESolver(
    dim=1,
    t_vec=np.array([0.0, 0.1]),
    verbose=False,
    load_attr=False,
    init=True,
    seed=42,
)

solver.a0 = 100
solver.x = np.full(1, 27e-5)
solver.process_type = "nucleation"
solver.liquid_flow_rate = 0.00001  # 10 µL/s
solver.droplet_diameter = 0.0005

print(f"Before _initialize_particles:")
print(f"  liquid_flow_rate: {solver.liquid_flow_rate}")
print(f"  droplet_diameter: {solver.droplet_diameter}")

solver._initialize_particles()

print(f"\nAfter _initialize_particles:")
print(f"  a_tot: {solver.a_tot}")
print(f"  nucleation_enable: {getattr(solver, 'nucleation_enable', 'NOT SET')}")

# Manually initialize nucleation
if not hasattr(solver, 'nucleation_enable'):
    solver.nucleation_enable = True

print(f"\nCalling initialize_nucleation()...")
solver.initialize_nucleation()

print(f"\nAfter initialize_nucleation():")
print(f"  _droplet_volume: {getattr(solver, '_droplet_volume', 'NOT SET')}")
print(f"  _droplet_rate: {getattr(solver, '_droplet_rate', 'NOT SET')}")
print(f"  _accumulated_time: {getattr(solver, '_accumulated_time', 'NOT SET')}")

# Test distribute_droplets_for_timestep
print(f"\nTesting distribute_droplets_for_timestep(dt=0.01)...")
if hasattr(solver, 'distribute_droplets_for_timestep'):
    result = solver.distribute_droplets_for_timestep(0.01)
    print(f"  Result: {result} droplets distributed")
    print(f"  _accumulated_time after: {solver._accumulated_time}")
    print(f"  Total liquid: {np.sum(solver.liquid_volume[:solver.a_tot]):.6e}")
else:
    print("  ERROR: Method not found!")
