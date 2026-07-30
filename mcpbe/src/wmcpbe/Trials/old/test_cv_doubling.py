"""
Test: Ist die Control-Volume-Verdopplung das Problem?

Diese Version deaktiviert CV-Doubling komplett.
"""

import numpy as np
import sys
sys.path.insert(0, 'C:/Users/ericb/Documents/GitHub/PSD_opt/mcpbe/src')

from wmcpbe import MCPBESolver
from wmcpbe.mcpbe_nucleation import NucleationConfig

# =====================================================================
# Physical Parameters (identisch zu test_nucleation_debug.py)
# =====================================================================
particle_diameter = 5e-5  # 50 µm
particle_radius = particle_diameter / 2.0
particle_volume = (4.0 / 3.0) * np.pi * particle_radius ** 3

droplet_diameter = 1e-5  # 10 µm
droplet_radius = droplet_diameter / 2.0
droplet_volume = (4.0 / 3.0) * np.pi * droplet_radius ** 3

volumetric_flow_rate = 1e-11  # m³/s
duration = 2.0  # seconds
agglomeration_coefficient = 1e-8  # niedrige Agglomeration

# =====================================================================
# Expected Results
# =====================================================================
expected_liquid_volume = volumetric_flow_rate * duration
expected_droplet_count = expected_liquid_volume / droplet_volume

print("=" * 80)
print("CV-DOUBLING DEAKTIVIERT TEST")
print("=" * 80)
print(f"Expected liquid: {expected_liquid_volume:.6e} m³")
print(f"Expected droplets: {expected_droplet_count:.2e}")
print()

# =====================================================================
# Solver Setup
# =====================================================================
n_particles_initial = 100
Vc = 1e-2  # m³

solver = MCPBESolver(dim=1, N=n_particles_initial, Vc=Vc, verbose=False)

X0 = np.full(n_particles_initial, particle_diameter)
W0 = np.full(n_particles_initial, 100.0)  # weight per computational particle
solver.init_lmc(X0, W0)

print(f"Initial state:")
print(f"  n_comp: {solver.a_tot}")
print(f"  Vc: {solver.Vc}")
print(f"  Total physical particles: {sum(solver.W[:solver.a_tot]) / solver.Vc:.2e}")
print()

# =====================================================================
# Configure Nucleation
# =====================================================================
nuc_config = NucleationConfig(
    enabled=True,
    t_start=0.0,
    t_end=duration,
    droplet_diameter=droplet_diameter,
    rho_liquid=1000.0,
    rho_solid=2000.0,
    target_wt_percent=None,
)
solver.configure_nucleation(nuc_config)

# =====================================================================
# DEAKTIVIERE CV-DOUBLING!
# =====================================================================
original_double_cv = solver._double_control_volume

def disabled_double_cv():
    print(f"[CV-DOUBLING BLOCKED] Would double from {solver.a_tot} to {solver.a_tot*2}")
    return  # Einfach nichts tun!

solver._double_control_volume = disabled_double_cv

# =====================================================================
# Run Simulation
# =====================================================================
print("Running simulation (CV-doubling disabled)...")
dt_target_agg = 0.1
t_total = 3.0

import time
start_time = time.time()
solver.run_calculate(t_total, dt_target_agg)
elapsed = time.time() - start_time

print(f"Simulation completed in {elapsed:.2f}s")
print()

# =====================================================================
# Results
# =====================================================================
nuc_stats = solver.nucleation.get_statistics()
actual_liquid = nuc_stats['liquid_volume_added_total']
actual_droplets = nuc_stats['droplets_added_total']

# Calculate liquid in system
v_liquid_in_system = 0.0
for k in range(solver.a_tot):
    if solver.W[k] > 0:
        v_liquid_in_system += solver.liquid_volume[k] * solver.W[k] / solver.Vc

error_added = (actual_liquid - expected_liquid_volume) / expected_liquid_volume * 100
error_system = (v_liquid_in_system - expected_liquid_volume) / expected_liquid_volume * 100

print("=" * 80)
print("RESULTS (CV-DOUBLING DISABLED)")
print("=" * 80)
print(f"\nLiquid Addition:")
print(f"  Expected:      {expected_liquid_volume:.6e} m³")
print(f"  Added (stat):  {actual_liquid:.6e} m³  ({error_added:+.4f}%)")
print(f"  In system:     {v_liquid_in_system:.6e} m³  ({error_system:+.4f}%)")

print(f"\nFinal State:")
print(f"  n_comp: {solver.a_tot}")
print(f"  Vc: {solver.Vc}")
print(f"  Total W: {sum(solver.W[:solver.a_tot]):.2f}")

if abs(error_system) < 5:
    print("\n✓ FEHLER VERSCHWUNDEN! Problem WAR CV-Doubling!")
else:
    print(f"\n⚠️  FEHLER BLEIBT: {error_system:+.2f}%")
    print("   Problem ist NICHT CV-Doubling, sondern woanders.")
