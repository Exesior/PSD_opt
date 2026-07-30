"""
DEBUG: Woher kommt die zusätzliche Flüssigkeit?

Hypothese: Bei Agglomeration wird liquid_volume DOPPELT gezählt:
1. Child bekommt liq_i + liq_j (korrekt)
2. ABER Parents behalten IHRE liquid_volume auch noch!
"""

import sys
sys.path.insert(0, 'C:/Users/ericb/Documents/GitHub/PSD_opt/mcpbe/src')

import numpy as np
from wmcpbe import WMCPBE

# ============================================================================
# Setup (identisch mit test_nucleation_debug.py)
# ============================================================================
d_particle = 50e-6
v_particle = np.pi * d_particle**3 / 6

d_droplet = 10e-6
v_droplet = np.pi * d_droplet**3 / 6

Q_vol = 1e-11  # m³/s
duration = 2.0
Vc = 1e-2  # m³

n_comp_init = 100
W_init = 100.0

# ============================================================================
# Erwartete Werte
# ============================================================================
total_physical_particles = n_comp_init * W_init  # 10,000
liquid_flow_rate = Q_vol  # m³/s
expected_total_liquid = liquid_flow_rate * duration  # 2e-11 m³
expected_droplets = expected_total_liquid / v_droplet

print(f"Erwartete Gesamtflüssigkeit: {expected_total_liquid:.6e} m³")
print(f"Erwartete Tropfen: {expected_droplets:.2f}")
print()

# ============================================================================
# Solver initialisieren
# ============================================================================
solver = WMCPBE(dim=1, N=n_comp_init, Vc=Vc, verbose=False)

X0 = np.full(n_comp_init, d_particle)
W0 = np.full(n_comp_init, W_init)
solver.init_lmc(X0, W0)

# Nucleation konfigurieren
from wmcpbe.mcpbe_nucleation import NucleationConfig
nuc_config = NucleationConfig(
    enabled=True,
    t_start=0.0,
    t_end=duration,
    droplet_diameter=d_droplet,
    rho_liquid=1000.0,
    rho_solid=2000.0,
    target_wt_percent=None,
)
solver.configure_nucleation(nuc_config)

# ============================================================================
# Simulation mit manuellem Tracking
# ============================================================================
print("=" * 80)
print("SIMULATION MIT LIQUID-TRACKING")
print("=" * 80)

# Manuelles Tracking
liquid_added_manual = 0.0
n_droplets_manual = 0

# Hook für Nucleation-Events
original_distribute = solver._distribute_one_droplet_with_dW

def tracked_distribute(droplet_volume, time):
    global liquid_added_manual, n_droplets_manual
    result = original_distribute(droplet_volume, time)
    if result:
        liquid_added_manual += droplet_volume
        n_droplets_manual += 1
    return result

solver._distribute_one_droplet_with_dW = tracked_distribute

# Patch _do_one_agg um liquid_volume vor/nach zu loggen
original_do_agg = solver._do_one_agg
agg_counter = [0]

def tracked_do_agg():
    agg_counter[0] += 1
    
    # Vor Agg: Summe aller liquid_volumes (gewichtet)
    liq_before = sum(solver.liquid_volume[k] * solver.W[k] for k in range(solver.a_tot))
    
    result = original_do_agg()
    
    # Nach Agg: Summe aller liquid_volumes (gewichtet)
    liq_after = sum(solver.liquid_volume[k] * solver.W[k] for k in range(solver.a_tot))
    
    if agg_counter[0] <= 10 or agg_counter[0] % 50 == 0:
        delta = liq_after - liq_before
        if abs(delta) > 1e-20:
            print(f"  Agg #{agg_counter[0]}: Δ(liq*W) = {delta:+.6e} (before={liq_before:.6e}, after={liq_after:.6e})")
    
    return result

solver._do_one_agg = tracked_do_agg

# Simulation laufen
dt_target = 0.1
t_final = 3.0
solver.run_calculate(t_final, dt_target)

# ============================================================================
# Analyse
# ============================================================================
print()
print("=" * 80)
print("ANALYSE")
print("=" * 80)

# Methode 1: Manuell addiert
print(f"\n1. Liquid ADDED (manuell getrackt): {liquid_added_manual:.6e} m³")
print(f"   Droplets distributed: {n_droplets_manual}")

# Methode 2: Im System (Σ liquid_volume[k] × W[k] / Vc)
liq_in_system_weighted = sum(solver.liquid_volume[k] * solver.W[k] / Vc for k in range(solver.a_tot))
print(f"\n2. Liquid IN SYSTEM (Σ liq[k]×W[k]/Vc): {liq_in_system_weighted:.6e} m³")

# Methode 3: Im System (Σ liquid_volume[k]) - falsch aber zur Kontrolle
liq_in_system_unweighted = sum(solver.liquid_volume[k] for k in range(solver.a_tot))
print(f"3. Liquid IN SYSTEM (Σ liq[k]): {liq_in_system_unweighted:.6e} m³")

# Methode 4: Detailanalyse der Partikel
print(f"\n4. DETAILANALYSE:")
print(f"   Active particles: {solver.a_tot}")
print(f"   Total weight: {sum(solver.W[:solver.a_tot]):.2f}")

# Gruppiere nach liquid_volume
liq_vals = solver.liquid_volume[:solver.a_tot]
w_vals = solver.W[:solver.a_tot]

# Partikel OHNE Flüssigkeit
n_zero = sum(1 for lv in liq_vals if lv == 0.0)
print(f"   Particles with liq=0: {n_zero}")

# Partikel MIT Flüssigkeit
n_with_liq = solver.a_tot - n_zero
print(f"   Particles with liq>0: {n_with_liq}")

if n_with_liq > 0:
    liq_nonzero = liq_vals[liq_vals > 0]
    w_with_liq = w_vals[liq_vals > 0]
    
    print(f"   → Avg liquid_volume (per particle): {np.mean(liq_nonzero):.6e}")
    print(f"   → Min/Max liquid_volume: {np.min(liq_nonzero):.6e} / {np.max(liq_nonzero):.6e}")
    print(f"   → Avg weight of particles with liquid: {np.mean(w_with_liq):.2f}")
    
    # Check: Haben alle nucleated particles liquid_volume = v_droplet?
    n_exact = sum(1 for lv in liq_nonzero if abs(lv - v_droplet) < 1e-20)
    n_double = sum(1 for lv in liq_nonzero if abs(lv - 2*v_droplet) < 1e-20)
    n_triple = sum(1 for lv in liq_nonzero if abs(lv - 3*v_droplet) < 1e-20)
    
    print(f"\n   Verteilung der liquid_volume Werte:")
    print(f"     = 1×v_droplet: {n_exact} Partikel")
    print(f"     = 2×v_droplet: {n_double} Partikel")
    print(f"     = 3×v_droplet: {n_triple} Partikel")

# Methode 5: Check ob liquid_volume bei Parent-Reduktion geändert wird
print(f"\n5. AGG-EVENT ANALYSE:")
print(f"   Total agglomeration events: {agg_counter[0]}")

# ============================================================================
# Fazit
# ============================================================================
print()
print("=" * 80)
print("FAZIT")
print("=" * 80)

error_added = (liquid_added_manual - expected_total_liquid) / expected_total_liquid * 100
error_system = (liq_in_system_weighted - expected_total_liquid) / expected_total_liquid * 100

print(f"Error (added): {error_added:+.4f}%")
print(f"Error (in system): {error_system:+.4f}%")

if error_system > 100:
    print("\n⚠️  WARNUNG: Viel zu viel Flüssigkeit im System!")
    print("   Mögliche Ursachen:")
    print("   - Liquid wird bei Agglomeration DUPPLIERT statt kombiniert")
    print("   - Parent-Partikel behalten ihre liquid_volume NACH der Agglomeration")
    print("   - Control Volume Anpassung dupliziert liquid_volume falsch")
