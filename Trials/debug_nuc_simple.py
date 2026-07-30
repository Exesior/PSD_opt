"""
Debug-Script: Einfacher Nukleationstest mit maximaler Ausgabe
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
t_total = 5.0
t_write = 0.5
initial_particles = 5000
particle_size = 27e-5  # 270 µm
liquid_flow_rate = 0.00001  # 10 µL/s
droplet_diameter = 0.0005  # 0.5 mm
seed = 42

print("="*70)
print("DEBUG: NUKLEATION SIMPLE TEST")
print("="*70)

# Erwartete Werte berechnen
droplet_vol = (np.pi / 6.0) * (droplet_diameter ** 3)
liquid_flow_m3 = liquid_flow_rate * 1e-3
droplet_rate = liquid_flow_m3 / droplet_vol
expected_droplets = int(droplet_rate * t_total)
expected_total_liquid = droplet_rate * t_total * droplet_vol

print(f"\nErwartete Werte:")
print(f"  Tropfenvolumen: {droplet_vol*1e9:.4f} nL")
print(f"  Tropfenrate: {droplet_rate:.2f} Tropfen/s")
print(f"  Erwartete Tropfen (5s): {expected_droplets}")
print(f"  Erwartete Gesamtflüssigkeit: {expected_total_liquid*1e9:.4f} nL")

# Solver erstellen
t_vec = np.arange(0.0, t_total + 0.5*t_write, t_write, dtype=float)
solver = MCPBESolver(
    dim=dim,
    t_vec=t_vec,
    verbose=True,
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
solver.initialize_nucleation()

print(f"\nSolver Initialisierung:")
print(f"  a_tot: {solver.a_tot}")
print(f"  _droplet_rate: {solver._droplet_rate:.2f}")
print(f"  _droplet_volume: {solver._droplet_volume*1e9:.4f} nL")
print(f"  _accumulated_time: {solver._accumulated_time}")

# WICHTIG: _calculate_droplet_rate explizit aufrufen
print("\nAufruf von _calculate_droplet_rate()...")
solver._calculate_droplet_rate()
print(f"  Nach Aufruf:")
print(f"    _droplet_rate: {solver._droplet_rate:.2f}")
print(f"    _droplet_volume: {solver._droplet_volume*1e9:.4f} nL")

# Teste distribute_droplets_for_timestep direkt
print("\nDirekter Test von distribute_droplets_for_timestep(dt=1.0):")
n_dist = solver.distribute_droplets_for_timestep(1.0)
print(f"  Verteilte Tropfen: {n_dist}")
print(f"  _accumulated_time danach: {solver._accumulated_time}")
print(f"  _total_droplets_distributed: {solver._total_droplets_distributed}")

# Reset für eigentliche Simulation
solver._accumulated_time = 0.0
solver._total_droplets_distributed = 0

print("\n" + "="*70)
print("START SIMULATION")
print("="*70)

solver.solve(maxiter=int(1e7))

print("\n" + "="*70)
print("ERGEBNISSE")
print("="*70)
final_particles = solver.a_tot
total_liquid = np.sum(solver.liquid_volume[:final_particles])

print(f"Finale Partikel: {final_particles}")
print(f"Gesamt-Liquid: {total_liquid*1e9:.4f} nL")
print(f"_total_droplets_distributed: {solver._total_droplets_distributed}")
print(f"_accumulated_time (Rest): {solver._accumulated_time:.9f} s")

liq_error = abs(total_liquid - expected_total_liquid) / expected_total_liquid * 100
print(f"Massenbilanz-Fehler: {liq_error:.2f}%")
