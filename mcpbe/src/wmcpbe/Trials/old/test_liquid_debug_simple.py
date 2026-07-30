"""
Einfacher Debug-Test: Woher kommt die zusätzliche Flüssigkeit?

HYPOTHESE: Nach meiner Korrektur behalten Parent-Partikel ihre liquid_volume!
Wenn Partikel A (liq=1) und B (liq=1) aggregieren:
- Child C bekommt liq = 1+1 = 2 (korrekt)
- A und B werden im Gewicht reduziert, BEHALTEN aber liq=1 (FEHLER!)

Resultat: System hat liq_C + liq_A + liq_B = 2 + 1 + 1 = 4 statt nur 2!
"""

import sys
sys.path.insert(0, 'C:/Users/ericb/Documents/GitHub/PSD_opt/mcpbe/src')

import numpy as np
from wmcpbe import WMCPBE
from wmcpbe.mcpbe_nucleation import NucleationConfig

# Setup
d_particle = 50e-6
v_particle = np.pi * d_particle**3 / 6

d_droplet = 10e-6
v_droplet = np.pi * d_droplet**3 / 6

Q_vol = 1e-11
duration = 2.0
Vc = 1e-2

n_comp_init = 100
W_init = 100.0

expected_total_liquid = Q_vol * duration

print("=" * 80)
print("DEBUG: LIQUID VOLUME HERKUNFT")
print("=" * 80)
print(f"Erwartete Gesamtflüssigkeit: {expected_total_liquid:.6e} m³")
print(f"v_droplet = {v_droplet:.6e} m³")
print()

# Solver
solver = WMCPBE(dim=1, N=n_comp_init, Vc=Vc, verbose=False)

X0 = np.full(n_comp_init, d_particle)
W0 = np.full(n_comp_init, W_init)
solver.init_lmc(X0, W0)

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

# Tracking
liquid_added = 0.0
original_distribute = solver._distribute_one_droplet_with_dW

def track_distribute(dv, t):
    global liquid_added
    result = original_distribute(dv, t)
    if result:
        liquid_added += dv
    return result

solver._distribute_one_droplet_with_dW = track_distribute

# Simulation
dt_target = 0.1
t_final = 3.0
solver.run_calculate(t_final, dt_target)

# Analyse
print("\n" + "=" * 80)
print("ERGEBNISSE")
print("=" * 80)

# 1. Added
print(f"\n1. Liquid ADDED: {liquid_added:.6e} m³ ({liquid_added/expected_total_liquid*100:.2f}% von erwartet)")

# 2. In System (gewichtet)
liq_weighted = sum(solver.liquid_volume[k] * solver.W[k] / Vc for k in range(solver.a_tot))
print(f"2. Liquid IN SYSTEM (Σ liq×W/Vc): {liq_weighted:.6e} m³ ({liq_weighted/expected_total_liquid*100:.2f}%)")

# 3. In System (ungewichtet)
liq_unweighted = sum(solver.liquid_volume[k] for k in range(solver.a_tot))
print(f"3. Liquid IN SYSTEM (Σ liq): {liq_unweighted:.6e} m³")

# 4. Partikel-Analyse
print(f"\n4. PARTIKEL-ANALYSE:")
print(f"   n_comp = {solver.a_tot}")
print(f"   Total W = {sum(solver.W[:solver.a_tot]):.2f}")

liq_arr = solver.liquid_volume[:solver.a_tot]
w_arr = solver.W[:solver.a_tot]

# Wie viele haben welche liquid_volume?
tol = 1e-25
n_zero = sum(1 for lv in liq_arr if lv < tol)
n_single = sum(1 for lv in liq_arr if abs(lv - v_droplet) < tol)
n_double = sum(1 for lv in liq_arr if abs(lv - 2*v_droplet) < tol)
n_triple = sum(1 for lv in liq_arr if abs(lv - 3*v_droplet) < tol)
n_other = solver.a_tot - n_zero - n_single - n_double - n_triple

print(f"   liquid_volume = 0:           {n_zero} Partikel")
print(f"   liquid_volume = 1×v_droplet: {n_single} Partikel")
print(f"   liquid_volume = 2×v_droplet: {n_double} Partikel")
print(f"   liquid_volume = 3×v_droplet: {n_triple} Partikel")
print(f"   liquid_volume = andere:      {n_other} Partikel")

if n_other > 0:
    other_vals = [lv for lv in liq_arr if abs(lv - v_droplet) > tol and abs(lv - 2*v_droplet) > tol and abs(lv - 3*v_droplet) > tol and lv > tol]
    print(f"     → Werte: min={min(other_vals):.6e}, max={max(other_vals):.6e}")

# 5. Check: Σ W für jede Kategorie
w_single = sum(w_arr[i] for i in range(solver.a_tot) if abs(liq_arr[i] - v_droplet) < tol)
w_double = sum(w_arr[i] for i in range(solver.a_tot) if abs(liq_arr[i] - 2*v_droplet) < tol)
w_triple = sum(w_arr[i] for i in range(solver.a_tot) if abs(liq_arr[i] - 3*v_droplet) < tol)

print(f"\n5. GEWICHTETE VERTEILUNG:")
print(f"   Σ W für Partikel mit 1×v_droplet: {w_single:.2f}")
print(f"   Σ W für Partikel mit 2×v_droplet: {w_double:.2f}")
print(f"   Σ W für Partikel mit 3×v_droplet: {w_triple:.2f}")

# Beitrag zum Gesamtsystem
contrib_single = w_single * v_droplet / Vc
contrib_double = w_double * 2 * v_droplet / Vc
contrib_triple = w_triple * 3 * v_droplet / Vc

print(f"\n6. BEITRAG ZUR GESAMTFLÜSSIGKEIT:")
print(f"   Von 1×-Partikeln: {contrib_single:.6e} m³")
print(f"   Von 2×-Partikeln: {contrib_double:.6e} m³")
print(f"   Von 3×-Partikeln: {contrib_triple:.6e} m³")
print(f"   Summe:            {contrib_single + contrib_double + contrib_triple:.6e} m³")

print("\n" + "=" * 80)
error = (liq_weighted - expected_total_liquid) / expected_total_liquid * 100
print(f"ERROR: {error:+.2f}%")

if error > 100:
    print("\n⚠️  PROBLEM: Zu viel Flüssigkeit durch Agglomeration!")
    print("   Wenn zwei Partikel mit je 1×v_droplet aggregieren,")
    print("   hat das Child 2×v_droplet (korrekt).")
    print("   ABER: Parents werden nur im Gewicht reduziert,")
    print("   behalten aber IHRE liquid_volume! → Doppelzählung!")
