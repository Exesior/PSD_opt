#!/usr/bin/env python
"""Deep debug: Prüft OB und WIE OFT Tropfen verteilt werden."""

import sys
sys.path.insert(0, r'C:\Users\ericb\Documents\GitHub\PSD_opt\mcpbe\src')

import numpy as np
from mcpbe import MCPBESolver

print("="*70)
print("DEEP DEBUG: Nucleation Flow")
print("="*70)

# Solver erstellen
solver = MCPBESolver(
    dim=1,
    t_vec=np.array([0.0, 0.1, 0.2]),  # Kurz für Debug
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
solver.nucleation_enable = True

print("\n1. Initialisierung:")
solver._initialize_particles()
solver.initialize_nucleation()

print(f"   a_tot: {solver.a_tot}")
print(f"   _droplet_rate: {solver._droplet_rate:.2f} Tropfen/s")
print(f"   _droplet_volume: {solver._droplet_volume:.3e} m³")
print(f"   Initiale Gesamtflüssigkeit: {np.sum(solver.liquid_volume):.6e} m³")

# Monkey-patch distribute_droplets_for_timestep um Aufrufe zu zählen
original_distribute = solver.distribute_droplets_for_timestep
call_count = 0
total_dt = 0.0

def traced_distribute(dt):
    global call_count, total_dt
    call_count += 1
    total_dt += dt
    print(f"\n   [CALL #{call_count}] distribute_droplets_for_timestep(dt={dt:.6e} s)")
    print(f"      _accumulated_time BEFORE: {solver._accumulated_time:.9f} s")
    print(f"      total_droplets = rate * acc_time = {solver._droplet_rate * solver._accumulated_time:.4f}")
    result = original_distribute(dt)
    print(f"      _accumulated_time AFTER: {solver._accumulated_time:.9f} s")
    print(f"      Verteilt: {result} Tropfen")
    print(f"      _total_droplets_distributed: {getattr(solver, '_total_droplets_distributed', 'NOT SET')}")
    return result

solver.distribute_droplets_for_timestep = traced_distribute

# Monkey-patch _do_one_agg um Agglomerationen zu zählen
agg_count = 0
original_agg = solver._do_one_agg

def traced_agg():
    global agg_count
    agg_count += 1
    if agg_count <= 5 or agg_count % 100 == 0:
        print(f"\n   [AGG #{agg_count}] a_tot before: {solver.a_tot}")
    result = original_agg()
    if agg_count <= 5 or agg_count % 100 == 0:
        print(f"      a_tot after: {solver.a_tot}")
    return result

solver._do_one_agg = traced_agg

print("\n2. Starte solve() mit maxiter=1000:")
print("-"*70)

try:
    solver.solve(maxiter=1000)
except Exception as e:
    print(f"\nERROR during solve: {e}")
    import traceback
    traceback.print_exc()

print("-"*70)
print("\n3. Zusammenfassung:")
print(f"   Loop-Iterationen: {solver._iter_count + 1}")
print(f"   Agglomerationen: {agg_count}")
print(f"   distribute_droplets_for_timestep() Aufrufe: {call_count}")
print(f"   Gesamte dt für Nukleation: {total_dt:.6f} s")
print(f"   _total_droplets_distributed: {getattr(solver, '_total_droplets_distributed', 'NOT SET')}")
print(f"   Finale Partikel: {solver.a_tot}")
print(f"   Finale Gesamtflüssigkeit: {np.sum(solver.liquid_volume[:solver.a_tot]):.6e} m³")

# Erwarte: 10 µL/s × 0.2s = 2 nL = 2e-18 m³
expected_liquid = solver.liquid_flow_rate * 1e-3 * 0.2  # L/s → m³/s × s
print(f"   Erwartete Flüssigkeit (Rate × Zeit): {expected_liquid:.6e} m³")
print(f"   Abweichung: {(np.sum(solver.liquid_volume[:solver.a_tot]) / expected_liquid - 1) * 100:+.1f}%")
