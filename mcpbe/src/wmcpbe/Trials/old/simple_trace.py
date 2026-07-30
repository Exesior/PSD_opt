"""Simple trace to find liquid bug."""

import sys
sys.path.insert(0, r'C:\Users\ericb\Documents\GitHub\PSD_opt\mcpbe\src')

from wmcpbe.mcpbe import MCPBESolver
import numpy as np

print("=" * 80)
print("SIMPLE LIQUID TRACE")
print("=" * 80)

solver = MCPBESolver(
    dim=1, t_total=0.1, t_write=5, verbose=True,
    load_attr=False, init=False, seed=42,  # Manual init!
)

solver.c = np.array([1e-3])
solver.x = np.array([5e-6])
solver.PGV = np.array(['mono'])
solver.SIG = np.array([0.0])
solver.a0 = 50
solver.process_type = 'agglomeration'
solver.COLEVAL = 3
solver.CORR_BETA = 1e-15

# Initialize particles and samplers FIRST
solver._initialize_particles()
solver._initialize_samplers()

# THEN create nucleation handler
solver.create_nucleation_handler(
    enabled=True, volumenstrom=1e-16, tropfen_durchmesser=1e-6,
    wasserzugabe_start=0.0, wasserzugabe_dauer=0.1,
)
solver.nucleation.configure_time_step(0.02)

print(f"\nInitial:")
print(f"  a_tot={solver.a_tot}")
print(f"  V_solid={np.sum(solver.get_V_solid()):.6e}")
print(f"  Liquid={np.sum(solver.liquid_volume[:solver.a_tot]):.6e}")

# Run solve
print(f"\n=== Running solve ===")
solver.solve(maxiter=500)

print(f"\n=== Final State ===")
print(f"  a_tot={solver.a_tot}")
print(f"  V_solid={np.sum(solver.get_V_solid()):.6e}")
print(f"  Liquid in system={np.sum(solver.liquid_volume[:solver.a_tot]):.6e}")
print(f"  Liquid added (stat)={solver.nucleation._liquid_volume_added_total:.6e}")
print(f"  Droplets added={solver.nucleation._droplets_added_total}")
print(f"  Ratio system/stat={np.sum(solver.liquid_volume[:solver.a_tot]) / solver.nucleation._liquid_volume_added_total:.2f}")

# Check particles
print(f"\n=== Particles with liquid ===")
for i in range(min(10, solver.a_tot)):
    liq = solver.liquid_volume[i]
    if liq > 1e-300:
        poro = solver.porosity[i]
        v_dry = solver.V_flat[-1, i]
        print(f"  [{i:2d}] liq={liq:.2e}, dry={v_dry:.2e}, poro={poro}")

# Debug: Check V_flat structure after nucleation
print(f"\n=== V_flat Structure Debug ===")
i = 0  # Check first particle
print(f"V_flat[-1, {i}] = {solver.V_flat[-1, i]:.6e} (V_particle_dry)")
print(f"V_flat[:dim, {i}] = {solver.V_flat[:solver.dim, i]} (solid volumes?)")
print(f"porosity[{i}] = {solver.porosity[i]}")
if not np.isnan(solver.porosity[i]):
    print(f"V_solid calculated = {solver.V_flat[-1, i] * (1 - solver.porosity[i]):.6e}")
    print(f"sum(V_flat[:dim, i]) = {np.sum(solver.V_flat[:solver.dim, i]):.6e}")