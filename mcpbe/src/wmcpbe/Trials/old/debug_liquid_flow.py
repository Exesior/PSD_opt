"""Debug liquid flow with detailed tracing."""

import sys
sys.path.insert(0, r'C:\Users\ericb\Documents\GitHub\PSD_opt\mcpbe\src')

from wmcpbe.mcpbe import MCPBESolver
import numpy as np

# Monkey-patch _distribute_one_droplet for tracing
original_distribute = None

def patched_distribute(self, v_droplet):
    solver = self.solver
    a_tot = solver.a_tot
    
    # Before
    liq_before = np.sum(solver.liquid_volume[:a_tot])
    n_agg_before = a_tot
    
    # Call original
    result = original_distribute(self, v_droplet)
    
    # After
    a_tot_after = solver.a_tot
    liq_after = np.sum(solver.liquid_volume[:a_tot_after])
    liq_change = liq_after - liq_before
    
    if a_tot_after != n_agg_before:
        print(f"  AGG: particles {n_agg_before} -> {a_tot_after}, liquid {liq_before:.2e} -> {liq_after:.2e} (Δ={liq_change:.2e})")
    else:
        print(f"  ADD: liquid {liq_before:.2e} -> {liq_after:.2e} (Δ={liq_change:.2e}, expected={v_droplet:.2e})")
    
    return result

print("=" * 80)
print("DETAILED LIQUID FLOW DEBUG")
print("=" * 80)

solver = MCPBESolver(
    dim=1, t_total=0.2, t_write=10, verbose=False,
    load_attr=False, init=True, seed=42,
)

solver.c = np.array([1e-3])
solver.x = np.array([5e-6])
solver.PGV = np.array(['mono'])
solver.SIG = np.array([0.0])
solver.a0 = 50  # Small number for easy tracing
solver.process_type = 'agglomeration'
solver.COLEVAL = 3
solver.CORR_BETA = 1e-15

# Create nucleation
solver.create_nucleation_handler(
    enabled=True, volumenstrom=1e-16, tropfen_durchmesser=1e-6,
    wasserzugabe_start=0.0, wasserzugabe_dauer=0.2,
)
solver.nucleation.configure_time_step(0.05)

# Monkey patch
original_distribute = solver.nucleation._distribute_one_droplet
solver.nucleation._distribute_one_droplet = lambda v: patched_distribute(solver.nucleation, v)

print(f"\nInitial: a_tot={solver.a_tot}, liquid={np.sum(solver.liquid_volume[:solver.a_tot]):.6e}")

# Run one step manually
print(f"\n--- Manual step at t=0 ---")
solver.nucleation.step(0.0, 0.05)

print(f"\nAfter step:")
print(f"  a_tot: {solver.a_tot}")
print(f"  Liquid total: {np.sum(solver.liquid_volume[:solver.a_tot]):.6e}")
print(f"  Liquid added (stat): {solver.nucleation._liquid_volume_added_total:.6e}")
print(f"  Droplets added: {solver.nucleation._droplets_added_total}")

# Check individual particles
print(f"\nParticles with liquid:")
for i in range(solver.a_tot):
    if solver.liquid_volume[i] > 1e-300:
        print(f"  [{i:2d}] liq={solver.liquid_volume[i]:.2e}, poro={solver.porosity[i]}, sat={solver.saturation[i] if not np.isnan(solver.porosity[i]) else 0:.2f}")
