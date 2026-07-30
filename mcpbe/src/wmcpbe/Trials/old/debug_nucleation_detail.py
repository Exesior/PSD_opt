"""Debug nucleation step-by-step."""

import sys
sys.path.insert(0, r'C:\Users\ericb\Documents\GitHub\PSD_opt\mcpbe\src')

from wmcpbe.mcpbe import MCPBESolver
import numpy as np

print("=" * 70)
print("DEBUG: Nucleation Step-by-Step")
print("=" * 70)

solver = MCPBESolver(
    dim=1, t_total=1.0, t_write=10, verbose=True,
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

print(f"\nInitial state:")
print(f"  a_tot: {solver.a_tot}")
print(f"  V_flat[-1, 0]: {solver.V_flat[-1, 0]:.6e}")
print(f"  Has liquid_volume: {hasattr(solver, 'liquid_volume')}")
print(f"  Has porosity: {hasattr(solver, 'porosity')}")
print(f"  Has saturation: {hasattr(solver, 'saturation')}")

# Create nucleation handler
solver.create_nucleation_handler(
    enabled=True, volumenstrom=1e-15, tropfen_durchmesser=1e-6,
    wasserzugabe_start=0.0, wasserzugabe_dauer=1.0,
)

# Configure time step
solver.nucleation.configure_time_step(0.1)

print(f"\nNucleation configured:")
print(f"  _nucleation_dt: {solver.nucleation._nucleation_dt}")
print(f"  config.volumenstrom: {solver.nucleation.config.volumenstrom}")
print(f"  config.tropfen_volumen: {solver.nucleation.config.tropfen_volumen:.6e}")

# Manual nucleation step BEFORE solve
print(f"\n=== Manual nucleation step at t=0 ===")
print(f"  Before step:")
print(f"    liquid_volume[0]: {solver.liquid_volume[0]:.6e}")
print(f"    porosity[0]: {solver.porosity[0]}")
print(f"    saturation[0]: {solver.saturation[0]}")

solver.nucleation.step(0.0, 0.1)

print(f"  After step:")
print(f"    Droplets added: {solver.nucleation._droplets_added_total}")
print(f"    Liquid added (stat): {solver.nucleation._liquid_volume_added_total:.6e}")
print(f"    liquid_volume[0]: {solver.liquid_volume[0]:.6e}")
print(f"    porosity[0]: {solver.porosity[0]}")
print(f"    saturation[0]: {solver.saturation[0]}")
print(f"    a_tot: {solver.a_tot}")

# Check first few particles
print(f"\nFirst 5 particles after manual step:")
for i in range(min(5, solver.a_tot)):
    print(f"  [{i}] V_dry={solver.V_flat[-1,i]:.6e}, liq={solver.liquid_volume[i]:.6e}, poro={solver.porosity[i]}, sat={solver.saturation[i]}")

# Now run full solve
print(f"\n=== Running solver.solve() ===")
solver.solve(maxiter=1000)

print(f"\nAfter solve:")
print(f"  Droplets added: {solver.nucleation._droplets_added_total}")
print(f"  Liquid added (stat): {solver.nucleation._liquid_volume_added_total:.6e}")
print(f"  Solver liquid sum: {np.sum(solver.liquid_volume[:solver.a_tot]):.6e}")
print(f"  Mean saturation (all): {np.mean(solver.saturation[:solver.a_tot])}")
print(f"  Mean saturation (non-NaN poro): {np.mean(solver.saturation[~np.isnan(solver.porosity[:solver.a_tot])]) if np.any(~np.isnan(solver.porosity[:solver.a_tot])) else 'N/A'}")

# Check particles with valid porosity
valid_mask = ~np.isnan(solver.porosity[:solver.a_tot])
n_valid = np.sum(valid_mask)
print(f"  Particles with valid porosity: {n_valid} / {solver.a_tot}")
if n_valid > 0:
    print(f"  Mean sat (valid): {np.mean(solver.saturation[valid_mask]):.4f}")
    print(f"  Mean liq (valid): {np.mean(solver.liquid_volume[valid_mask]):.6e}")
