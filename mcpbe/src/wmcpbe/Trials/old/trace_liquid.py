"""Trace liquid addition step-by-step to find the bug."""

import sys
sys.path.insert(0, r'C:\Users\ericb\Documents\GitHub\PSD_opt\mcpbe\src')

from wmcpbe.mcpbe import MCPBESolver
import numpy as np

print("=" * 80)
print("LIQUID TRACING DEBUG")
print("=" * 80)

solver = MCPBESolver(
    dim=1, t_total=0.5, t_write=10, verbose=False,
    load_attr=False, init=True, seed=42,
)

solver.c = np.array([1e-3])
solver.x = np.array([5e-6])
solver.PGV = np.array(['mono'])
solver.SIG = np.array([0.0])
solver.a0 = 100
solver.process_type = 'agglomeration'
solver.COLEVAL = 3
solver.CORR_BETA = 1e-15

# Create nucleation handler with SMALL droplet count for tracing
solver.create_nucleation_handler(
    enabled=True, volumenstrom=1e-16, tropfen_durchmesser=1e-6,
    wasserzugabe_start=0.0, wasserzugabe_dauer=0.5,
)
solver.nucleation.configure_time_step(0.05)

print(f"\nInitial:")
print(f"  a_tot: {solver.a_tot}")
print(f"  V_solid total: {np.sum(solver.get_V_solid()):.6e}")
print(f"  Liquid total: {np.sum(solver.liquid_volume[:solver.a_tot]):.6e}")

# Manual steps with tracing
for step in range(5):
    t = step * 0.1
    
    print(f"\n{'='*80}")
    print(f"STEP {step} at t={t:.2f}s")
    print(f"{'='*80}")
    
    # Before step
    liq_before = np.sum(solver.liquid_volume[:solver.a_tot])
    n_particles = solver.a_tot
    n_with_liq = np.sum(solver.liquid_volume[:solver.a_tot] > 0)
    
    print(f"Before: a_tot={n_particles}, liquid={liq_before:.6e}, particles_with_liq={n_with_liq}")
    
    # Step
    solver.nucleation.step(t, 0.1)
    
    # After step
    liq_after = np.sum(solver.liquid_volume[:solver.a_tot])
    n_particles_after = solver.a_tot
    n_with_liq_after = np.sum(solver.liquid_volume[:solver.a_tot] > 0)
    liq_added_stat = solver.nucleation._liquid_volume_added_total
    
    print(f"After:  a_tot={n_particles_after}, liquid={liq_after:.6e}, particles_with_liq={n_with_liq_after}")
    print(f"Stats:  liquid_added_total={liq_added_stat:.6e}")
    print(f"Balance: system={liq_after:.6e}, expected={liq_added_stat:.6e}, ratio={liq_after/liq_added_stat if liq_added_stat > 0 else 0:.2f}")
    
    # Check first few particles with liquid
    print(f"\nFirst 5 particles with liquid:")
    for i in range(min(5, solver.a_tot)):
        if solver.liquid_volume[i] > 0:
            poro = solver.porosity[i]
            sat = solver.saturation[i] if not np.isnan(poro) else 0
            v_dry = solver.V_flat[-1, i]
            v_pore = v_dry * poro if not np.isnan(poro) else 0
            v_liq_int = min(solver.liquid_volume[i], v_pore * sat) if not np.isnan(poro) else 0
            v_liq_ext = solver.liquid_volume[i] - v_liq_int
            print(f"  [{i}] dry={v_dry:.2e}, liq={solver.liquid_volume[i]:.2e}, int={v_liq_int:.2e}, ext={v_liq_ext:.2e}, poro={poro:.2f}, sat={sat:.2f}")

print(f"\n{'='*80}")
print(f"FINAL SUMMARY")
print(f"{'='*80}")
print(f"Liquid in system: {np.sum(solver.liquid_volume[:solver.a_tot]):.6e}")
print(f"Liquid added (stat): {solver.nucleation._liquid_volume_added_total:.6e}")
print(f"Ratio: {np.sum(solver.liquid_volume[:solver.a_tot]) / solver.nucleation._liquid_volume_added_total:.2f}")
