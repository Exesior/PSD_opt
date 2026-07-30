"""
Debug-Script: Direkter Test der Nukleationsfunktionen
"""
import sys
from pathlib import Path
import numpy as np

script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.resolve()
mcpbe_src = project_root / "mcpbe" / "src"
sys.path.insert(0, str(mcpbe_src))

from mcpbe import MCPBESolver

# Konfiguration
dim = 1
initial_particles = 5000
particle_size = 27e-5  # 270 µm
liquid_flow_rate = 0.00001  # 10 µL/s
droplet_diameter = 0.0005  # 0.5 mm
seed = 42

print("="*70)
print("DEBUG: DIREKTER NUKLEATIONSTEST")
print("="*70)

# Solver erstellen OHNE init
solver = MCPBESolver(
    dim=dim,
    t_vec=np.array([0.0, 1.0]),
    verbose=False,
    load_attr=False,
    init=True,
    seed=seed,
)

solver.a0 = float(initial_particles)
solver.x = np.full(dim, particle_size)
solver.process_type = "nucleation"
solver.liquid_flow_rate = liquid_flow_rate
solver.droplet_diameter = droplet_diameter
solver.nucleation_enable = True

solver._initialize_particles()

print(f"\nPartikel initialisiert:")
print(f"  a_tot: {solver.a_tot}")
print(f"  V_flat[0, 0:5]: {solver.V_flat[0, 0:5]*1e9} nL")

# Initialize nucleation
solver.initialize_nucleation()

print(f"\nNach initialize_nucleation():")
print(f"  _droplet_rate: {solver._droplet_rate}")
print(f"  _droplet_volume: {solver._droplet_volume*1e9:.6f} nL")
print(f"  _accumulated_time: {solver._accumulated_time}")

# Test 1: _calculate_droplet_rate prüfen
print("\n" + "-"*50)
print("TEST 1: _calculate_droplet_rate()")
print("-"*50)
solver._calculate_droplet_rate()
print(f"_droplet_rate: {solver._droplet_rate}")
print(f"_droplet_volume: {solver._droplet_volume*1e9:.6f} nL")

# Manuelle Berechnung zum Vergleich
expected_vol = (np.pi / 6.0) * (droplet_diameter ** 3)
expected_rate = (liquid_flow_rate * 1e-3) / expected_vol
print(f"Erwartet: volume={expected_vol*1e9:.6f} nL, rate={expected_rate:.2f}")

# Test 2: distribute_droplets_for_timestep mit verschiedenen dt-Werten
print("\n" + "-"*50)
print("TEST 2: distribute_droplets_for_timestep()")
print("-"*50)

test_dts = [0.001, 0.01, 0.1, 1.0, 1.0/solver._droplet_rate]
for dt in test_dts:
    solver._accumulated_time = 0.0  # Reset
    n_dist = solver.distribute_droplets_for_timestep(dt)
    print(f"dt={dt:.6f}s -> distributed={n_dist}, accumulated_time={solver._accumulated_time:.9f}")
    
    # Manuelle Berechnung
    expected = solver._droplet_rate * dt
    print(f"           erwartet: {expected:.4f} Tropfen (int={int(expected)}, round={int(round(expected))})")

# Test 3: Einzelnen Tropfen verteilen
print("\n" + "-"*50)
print("TEST 3: distribute_droplet() direkt")
print("-"*50)
solver._accumulated_time = 0.0
success = solver.distribute_droplet()
print(f"distribute_droplet() returned: {success}")
print(f"a_tot danach: {solver.a_tot}")
print(f"liquid_volume[0]: {solver.liquid_volume[0]*1e9:.6f} nL")

# Test 4: Volle Simulation mit Debug-Ausgaben
print("\n" + "="*70)
print("TEST 4: VOLLE SIMULATION MIT DEBUG")
print("="*70)

# Neuer Solver
solver2 = MCPBESolver(
    dim=dim,
    t_vec=np.arange(0.0, 5.0 + 0.25, 0.5, dtype=float),
    verbose=True,
    load_attr=False,
    init=True,
    seed=seed,
)

solver2.a0 = float(initial_particles)
solver2.x = np.full(dim, particle_size)
solver2.process_type = "nucleation"
solver2.liquid_flow_rate = liquid_flow_rate
solver2.droplet_diameter = droplet_diameter
solver2.nucleation_enable = True
solver2._initialize_particles()
solver2.initialize_nucleation()

# Füge Debug-Prints hinzu BEFORE solve
print(f"\nVOR solve():")
print(f"  process_type: {solver2.process_type}")
print(f"  nucleation_enable: {solver2.nucleation_enable}")
print(f"  _droplet_rate: {solver2._droplet_rate:.2f}")
print(f"  _droplet_volume: {solver2._droplet_volume*1e9:.6f} nL")

# Patch distribute_droplets_for_timestep für Debug
original_distribute = solver2.distribute_droplets_for_timestep
call_count = [0]

def debug_distribute(dt):
    call_count[0] += 1
    result = original_distribute(dt)
    if call_count[0] <= 5 or call_count[0] % 100 == 0:
        print(f"  [DEBUG] distribute_droplets_for_timestep(dt={dt:.6f}) -> {result} Tropfen")
    return result

solver2.distribute_droplets_for_timestep = debug_distribute

solver2.solve(maxiter=int(1e7))

print(f"\nERGEBNISSE:")
final_particles = solver2.a_tot
total_liquid = np.sum(solver2.liquid_volume[:final_particles])
print(f"Finale Partikel: {final_particles}")
print(f"Gesamt-Liquid: {total_liquid*1e9:.4f} nL")
print(f"_total_droplets_distributed: {solver2._total_droplets_distributed}")
print(f"Aufrufe von distribute_droplets_for_timestep: {call_count[0]}")
