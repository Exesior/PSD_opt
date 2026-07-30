"""
MINIMALER TEST: Nur Nukleation ohne Solver-Loop Komplexität
"""
import sys
from pathlib import Path
import numpy as np

script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.resolve()
mcpbe_src = project_root / "mcpbe" / "src"
sys.path.insert(0, str(mcpbe_src))

from mcpbe import MCPBESolver

# Konfiguration - EXAKT wie im Haupttest
dim = 1
initial_particles = 5000
particle_size = 27e-5
liquid_flow_rate = 0.00001  # 10 µL/s
droplet_diameter = 0.0005  # 0.5 mm
seed = 42
t_total = 5.0

print("="*70)
print("MINIMALER NUKLEATIONSTEST")
print("="*70)

# Erwartete Werte
droplet_vol = (np.pi / 6.0) * (droplet_diameter ** 3)
liquid_flow_m3 = liquid_flow_rate * 1e-3
droplet_rate = liquid_flow_m3 / droplet_vol
expected_droplets = int(droplet_rate * t_total)
expected_liquid = expected_droplets * droplet_vol

print(f"\nParameter:")
print(f"  liquid_flow_rate: {liquid_flow_rate*1e6} µL/s")
print(f"  droplet_diameter: {droplet_diameter*1e3} mm")
print(f"  droplet_volume: {droplet_vol*1e9:.4f} nL")
print(f"  droplet_rate: {droplet_rate:.2f} Tropfen/s")
print(f"  t_total: {t_total} s")
print(f"\nErwartet nach {t_total}s:")
print(f"  Tropfen: ~{expected_droplets}")
print(f"  Flüssigkeit: {expected_liquid*1e9:.4f} nL")

# Solver erstellen
solver = MCPBESolver(
    dim=dim,
    t_vec=np.arange(0.0, t_total + 0.5, 0.5, dtype=float),
    verbose=True,  # WICHTIG für Debug-Ausgaben!
    load_attr=False,
    init=True,
    seed=seed,
)

solver.a0 = float(initial_particles)
solver.x = np.full(dim, particle_size)
solver.process_type = "nucleation"  # NUR Nukleation!
solver.liquid_flow_rate = liquid_flow_rate
solver.droplet_diameter = droplet_diameter
solver.nucleation_enable = True

solver._initialize_particles()
solver.initialize_nucleation()

print(f"\nNach Initialisierung:")
print(f"  a_tot: {solver.a_tot}")
print(f"  _droplet_rate: {solver._droplet_rate:.2f}")
print(f"  _droplet_volume: {solver._droplet_volume*1e9:.4f} nL")
print(f"  dtd_nucleation (1/rate): {1.0/solver._droplet_rate:.6f} s")

# SIMULATION STARTEN
print("\n" + "="*70)
print("START SIMULATION")
print("="*70 + "\n")

solver.solve(maxiter=int(1e7))

# ERGEBNISSE
print("\n" + "="*70)
print("ERGEBNISSE")
print("="*70)

final_particles = solver.a_tot
total_liquid = np.sum(solver.liquid_volume[:final_particles])

print(f"\nFinale Partikel: {final_particles}")
print(f"Gesamt-Liquid: {total_liquid*1e9:.4f} nL")
print(f"Erwartetes Liquid: {expected_liquid*1e9:.4f} nL")
print(f"Fehler: {abs(total_liquid - expected_liquid)/expected_liquid*100:.2f}%")

print(f"\nTracking-Variablen:")
print(f"  _total_droplets_distributed: {getattr(solver, '_total_droplets_distributed', 'N/A')}")
print(f"  _accumulated_time: {getattr(solver, '_accumulated_time', 'N/A'):.9f} s")
print(f"  MACHINE_TIME: {getattr(solver, 'MACHINE_TIME', 'N/A'):.3f} s")

# Berechne tatsächliche Tropfen aus liquid_volume
actual_droplets = total_liquid / droplet_vol
print(f"\nTatsächlich verteilte Tropfen (aus Liquid): {actual_droplets:.0f}")
print(f"Erwartete Tropfen: {expected_droplets}")
print(f"Faktor: {actual_droplets/expected_droplets:.1f}x zu viel")
