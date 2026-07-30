"""
Debug test for nucleation - detailed logging to understand why no liquid is added.
"""

import numpy as np
import sys
sys.path.insert(0, 'C:/Users/ericb/Documents/GitHub/PSD_opt/mcpbe/src')

from wmcpbe.mcpbe import MCPBESolver

print("=" * 70)
print("DEBUG TEST: Large Particles (50 µm)")
print("=" * 70)

solver = MCPBESolver(
    dim=1,
    t_total=2.0,
    t_write=10,
    verbose=False,
    load_attr=False,
    init=False,
    seed=42,
)

# Setup: 50 µm particles
diameter = 5e-5  # 50 µm
solver.c = np.array([1e-3])
solver.x = np.array([diameter])
solver.PGV = np.array(["mono"])
solver.SIG = np.array([0.0])
solver.a0 = 500
solver.process_type = "agglomeration"

solver.COLEVAL = 3
solver.CORR_BETA = 1e-13
solver.G = 1.0
solver.alpha_prim = 1.0

solver._initialize_particles()
solver._initialize_samplers()

n_init = solver.a_tot
solver.liquid_volume[:n_init] = 0.0

print(f"\nInitial state:")
print(f"  Diameter: {diameter*1e6:.1f} µm")
print(f"  n_comp: {n_comp := solver.a_tot}")
print(f"  Vc: {solver.Vc}")
print(f"  Sum(W): {np.sum(solver.W[:n_init]):.2f}")
print(f"  n_phys: {np.sum(solver.W[:n_init]) / solver.Vc:.2f}")

# Calculate particle volume
V_particle = np.pi/6 * diameter**3
print(f"  V_particle (single): {V_particle:.6e} m³")
print(f"  V_solid total: {np.sum(solver.get_V_solid()[:n_init] * solver.W[:n_init]):.6e} m³")

# Create nucleation handler
tropfen_durchmesser = 2e-6  # 2 µm droplet
volumenstrom = 1e-15  # m³/s
duration = 1.0  # s

print(f"\nNucleation config:")
print(f"  Droplet diameter: {tropfen_durchmesser*1e6:.1f} µm")
print(f"  Droplet volume: {np.pi/6 * tropfen_durchmesser**3:.6e} m³")
print(f"  Volumenstrom: {volumenstrom:.6e} m³/s")
print(f"  Duration: {duration:.1f} s")
print(f"  Expected liquid: {volumenstrom * duration:.6e} m³")

handler = solver.create_nucleation_handler(
    enabled=True,
    volumenstrom=volumenstrom,
    tropfen_durchmesser=tropfen_durchmesser,
    wasserzugabe_start=0.0,
    wasserzugabe_dauer=duration,
)

print(f"\nHandler config:")
print(f"  _nucleation_dt: {handler._nucleation_dt}")
print(f"  initialization_type: {handler.config.initialization_type}")
print(f"  wasserzugabe_start: {handler.config.wasserzugabe_start}")
print(f"  wasserzugabe_end: {handler.config.wasserzugabe_end}")
print(f"  tropfen_volumen: {handler.config.tropfen_volumen:.6e}")

# Patch the step method to add logging
original_step = handler.step

def logged_step(current_time, solver_last_dt):
    result = original_step(current_time, solver_last_dt)
    
    # Log after each step
    if current_time <= 0.5 and hasattr(handler, '_last_logged_step') and current_time - handler._last_logged_step >= 0.1:
        print(f"\n[DEBUG t={current_time:.2f}s]")
        print(f"  a_tot: {solver.a_tot}")
        print(f"  liquid_volume_added_total: {handler._liquid_volume_added_total:.6e}")
        print(f"  droplets_added_total: {handler._droplets_added_total:.2f}")
        nuc = ~np.isnan(solver.porosity[:solver.a_tot])
        print(f"  Nucleated particles: {np.sum(nuc)}")
        handler._last_logged_step = current_time
    
    return result

handler._last_logged_step = -999
handler.step = logged_step

# Run simulation
print("\n" + "=" * 70)
print("Running simulation...")
print("=" * 70)

solver.solve(maxiter=20000)

# Final state
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

n_final = solver.a_tot
V_liq_final = np.sum(solver.liquid_volume[:n_final] * solver.W[:n_final])

nuc_stats = handler.get_statistics()
liquid_added = nuc_stats['liquid_volume_added_total']

print(f"n_comp final: {n_final}")
print(f"V_liquid (weighted): {V_liq_final:.6e} m³")
print(f"Liquid added (stat): {liquid_added:.6e} m³")

nuc_mask = ~np.isnan(solver.porosity[:n_final])
print(f"Nucleated particles: {np.sum(nuc_mask)}")

if np.any(nuc_mask):
    print(f"  Mean porosity: {np.mean(solver.porosity[nuc_mask]):.3f}")
    print(f"  Mean saturation: {np.mean(solver.saturation[nuc_mask]):.3f}")
    print(f"  Mean liquid_volume: {np.mean(solver.liquid_volume[nuc_mask]):.6e}")

print("\n" + ("✅ SUCCESS" if liquid_added > 0 else "❌ FAIL: No liquid added"))
