#!/usr/bin/env python
"""
Deep Debug: Woher kommt die zusätzliche Flüssigkeit?

Prüft:
1. Wie oft wird distribute_droplets_for_timestep() aufgerufen?
2. Wie viele Iterationen macht der solve()-Loop?
3. Stimmt dieelapsed_time Logik?
"""

import sys
sys.path.insert(0, r'C:\Users\ericb\Documents\GitHub\PSD_opt\mcpbe\src')

import numpy as np
from mcpbe import MCPBESolver

print("="*70)
print("DEEP DEBUG: Liquid Flow Analysis")
print("="*70)

# Kurzer Test für schnelles Debugging
solver = MCPBESolver(
    dim=1,
    t_vec=np.array([0.0, 0.5, 1.0]),  # Nur 1 Sekunde
    verbose=False,
    load_attr=False,
    init=True,
    seed=42,
)

solver.a0 = 1000
solver.x = np.full(1, 27e-5)
solver.process_type = "nucleation"
solver.liquid_flow_rate = 0.00001  # 10 µL/s
solver.droplet_diameter = 0.0005
solver.nucleation_enable = True

print("\nKonfiguration:")
print(f"  Simulationszeit: 1.0 s")
print(f"  Flüssigkeitsrate: 10 µL/s")
print(f"  Erwartete Tropfen: ~{int(10 * 1.0)}")
print(f"  Erwartete Flüssigkeit: 10 nL")

solver._initialize_particles()
solver.initialize_nucleation()

# Track calls
calls = {
    'distribute': 0,
    'agg': 0,
    'loop_iterations': 0,
    'total_dt_distributed': 0.0,
}

# Monkey-patch distribute_droplets_for_timestep
orig_distribute = solver.distribute_droplets_for_timestep
def traced_distribute(dt):
    calls['distribute'] += 1
    calls['total_dt_distributed'] += dt
    result = orig_distribute(dt)
    if calls['distribute'] <= 3 or calls['distribute'] % 100 == 0:
        print(f"  [DIST #{calls['distribute']}] dt={dt:.6f}s → {result} Tropfen")
    return result
solver.distribute_droplets_for_timestep = traced_distribute

# Monkey-patch _do_one_agg
orig_agg = solver._do_one_agg
def traced_agg():
    calls['agg'] += 1
    return orig_agg()
solver._do_one_agg = traced_agg

# Monkey-patch solve loop to count iterations
orig_solve = solver.solve
def traced_solve(maxiter=int(1e7)):
    count = 0
    while solver.t[-1] <= float(solver.t_vec[-1]) and count < maxiter:
        count += 1
        if count <= 5 or count % 500 == 0:
            print(f"  [LOOP #{count}] t={solver.t[-1]:.6f}s, a_tot={solver.a_tot}")
        # Original logic would go here, but we just count
        try:
            return orig_solve(maxiter=maxiter)
        except Exception as e:
            print(f"ERROR at iteration {count}: {e}")
            raise
    return None

print("\nStarte Simulation...")
try:
    solver.solve(maxiter=int(1e7))
except Exception as e:
    print(f"Simulation failed: {e}")

print("\n" + "="*70)
print("ERGEBNISSE:")
print("="*70)
print(f"Loop-Iterationen: {solver._iter_count + 1}")
print(f"distribute_droplets_for_timestep() Aufrufe: {calls['distribute']}")
print(f"Agglomerationen: {calls['agg']}")
print(f"Summe dt für Nukleation: {calls['total_dt_distributed']:.6f} s")
print(f"_total_droplets_distributed: {getattr(solver, '_total_droplets_distributed', 'N/A')}")
print(f"\nFinale Partikel: {solver.a_tot}")
print(f"Finale Flüssigkeit: {np.sum(solver.liquid_volume[:solver.a_tot])*1e9:.4f} nL")
print(f"Erwartete Flüssigkeit: 10.0 nL")
print(f"Fehler: {(np.sum(solver.liquid_volume[:solver.a_tot])*1e9 / 10.0 - 1)*100:+.1f}%")
