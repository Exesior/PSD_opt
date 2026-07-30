"""Debug nucleation - test if liquid is added correctly."""

import sys
sys.path.insert(0, '.')

from wmcpbe.mcpbe import MCPBESolver
import numpy as np

print("=" * 60)
print("DEBUG: Nucleation Liquid Addition")
print("=" * 60)

# Create solver with init=True (standard usage)
solver = MCPBESolver(
    dim=1, t_total=1.0, t_write=10, verbose=False,
    load_attr=False, init=True, seed=42,
)

solver.c = np.array([1e-3])
solver.x = np.array([1e-6])
solver.PGV = np.array(['mono'])
solver.SIG = np.array([0.0])
solver.a0 = 1000
solver.process_type = 'agglomeration'
solver.COLEVAL = 3
solver.CORR_BETA = 1e-20

print(f"\nAfter initialization:")
print(f"  a_tot: {solver.a_tot}")
print(f"  Has liquid_volume: {hasattr(solver, 'liquid_volume')}")
print(f"  Has saturation: {hasattr(solver, 'saturation')}")
print(f"  Has porosity: {hasattr(solver, 'porosity')}")

# Create nucleation handler
solver.create_nucleation_handler(
    enabled=True, volumenstrom=1e-15, tropfen_durchmesser=1e-6,
    wasserzugabe_start=0.0, wasserzugabe_dauer=1.0,
)

print(f"\nAfter create_nucleation_handler:")
print(f"  _nucleation_dt: {solver.nucleation._nucleation_dt}")
print(f"  config.volumenstrom: {solver.nucleation.config.volumenstrom}")

# Now solve
print("\nCalling solver.solve(maxiter=1000)...")
solver.solve(maxiter=1000)

print(f"\nAfter solve:")
print(f"  Droplets added: {solver.nucleation._droplets_added_total}")
print(f"  Liquid added (stat): {solver.nucleation._liquid_volume_added_total:.6e}")
print(f"  Solver liquid sum: {np.sum(solver.liquid_volume[:solver.a_tot]):.6e}")
print(f"  Mean saturation: {np.mean(solver.saturation[:solver.a_tot]):.4f}" if hasattr(solver, 'saturation') else "")
