"""Debug test for DSMC-compliant nucleation with weight handling."""

import sys
sys.path.insert(0, r'C:\Users\ericb\Documents\GitHub\PSD_opt\mcpbe\src')

from wmcpbe.mcpbe import MCPBESolver
import numpy as np

print("=" * 80)
print("DSMC NUCLEATION DEBUG TEST")
print("=" * 80)

# Setup solver - need longer time for nucleation to trigger!
solver = MCPBESolver(
    dim=1, t_total=2.0, t_write=10, verbose=True,
    load_attr=False, init=False, seed=42,
)

solver.c = np.array([1e-3])
solver.x = np.array([5e-6])
solver.PGV = np.array(['mono'])
solver.SIG = np.array([0.0])
solver.a0 = 100
solver.process_type = 'agglomeration'
solver.COLEVAL = 3
solver.CORR_BETA = 1e-15

# Initialize particles and samplers
solver._initialize_particles()
solver._initialize_samplers()

# Create nucleation handler - window must be LONGER than dt_agg!
solver.create_nucleation_handler(
    enabled=True, volumenstrom=1e-15, tropfen_durchmesser=1e-6,
    wasserzugabe_start=0.0, wasserzugabe_dauer=3.0,  # Longer than dt_agg!
)
solver.nucleation.configure_time_step(0.1)  # More frequent nucleation steps

print(f"\n=== Initial State ===")
n_initial = solver.a_tot
print(f"  Particles: {n_initial}")
print(f"  Mean W: {np.mean(solver.W[:n_initial]):.2f}")
print(f"  Max W: {np.max(solver.W[:n_initial]):.2f}")

# IMPORTANT: Save initial values BEFORE simulation!
V_solid_initial = np.sum(solver.get_V_solid() * solver.W[:n_initial])
V_liq_initial = np.sum(solver.liquid_volume[:n_initial] * solver.W[:n_initial])

print(f"  V_solid (weighted): {V_solid_initial:.6e} m³")
print(f"  Liquid (weighted): {V_liq_initial:.6e} m³")

# Run simulation
print(f"\n=== Running solve ===")
solver.solve(maxiter=5000)

print(f"\n=== Final State ===")
n_final = solver.a_tot
print(f"  Particles: {n_final}")
print(f"  Mean W: {np.mean(solver.W[:n_final]):.2f}")
print(f"  Max W: {np.max(solver.W[:n_final]):.2f}")

# Conservation checks (WEIGHTED!)
V_solid_final = np.sum(solver.get_V_solid() * solver.W[:n_final])
V_liq_final = np.sum(solver.liquid_volume[:n_final] * solver.W[:n_final])

# Get statistics
nuc_stats = solver.nucleation.get_statistics()
liquid_added = nuc_stats['liquid_volume_added_total']
droplets_added = nuc_stats['droplets_added_total']

print(f"\n=== Conservation Check ===")
print(f"  V_solid initial: {V_solid_initial:.6e} m³")
print(f"  V_solid final:   {V_solid_final:.6e} m³")
print(f"  V_solid change:  {(V_solid_final - V_solid_initial):.6e} m³")
print(f"  V_solid error:   {abs(V_solid_final - V_solid_initial) / V_solid_initial * 100:.2f}%")

# For liquid: compare final to (initial + added)
expected_liquid = V_liq_initial + liquid_added
print(f"\n  Liquid initial (before sim): {V_liq_initial:.6e} m³")
print(f"  Liquid added (statistics):   {liquid_added:.6e} m³")
print(f"  Liquid expected:             {expected_liquid:.6e} m³")
print(f"  Liquid in system (final):    {V_liq_final:.6e} m³")
print(f"  Liquid difference:           {(V_liq_final - expected_liquid):.6e} m³")
print(f"  Liquid error:                {abs(V_liq_final - expected_liquid) / max(expected_liquid, 1e-300) * 100:.2f}%")
print(f"\n  Droplets added: {droplets_added:.0f}")

# Detailed particle analysis
print(f"\n=== Particle Analysis ===")
poro = solver.porosity[:n_final]
liq = solver.liquid_volume[:n_final]
W = solver.W[:n_final]

n_dry = np.sum(np.isnan(poro))
n_wet = n_final - n_dry
print(f"  Dry particles (no poro): {n_dry}")
print(f"  Wet particles (nucleated): {n_wet}")

if n_wet > 0:
    print(f"\n  Wet particles stats:")
    print(f"    Mean W: {np.mean(W[~np.isnan(poro)]):.2f}")
    print(f"    Mean liquid: {np.mean(liq[~np.isnan(poro)]):.6e} m³")
    print(f"    Total liquid (weighted): {np.sum(liq[~np.isnan(poro)] * W[~np.isnan(poro)]):.6e} m³")

# Heterogeneity check
unique_liquids = np.unique(liq[~np.isnan(poro)])
print(f"\n  Unique liquid values in wet particles: {len(unique_liquids)}")
if len(unique_liquids) > 1:
    print(f"    → HETEROGENEITY PRESERVED! ✓")
else:
    print(f"    → All wet particles have same liquid (homogeneous)")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
