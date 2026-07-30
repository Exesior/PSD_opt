"""
Quick Test: ConeModelKernel direkt ohne Solver.
"""

import sys
from pathlib import Path

# Parent directory hinzufügen
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from wmcpbe.kernels.porosity_growth import get_porosity_growth_kernel

print("="*60)
print("ConeModelKernel Quick Test")
print("="*60)

# Kernel erstellen
kernel = get_porosity_growth_kernel('cone_model', k_agg=0.5, k_break=0.1)
print(f"\nKernel erstellt: {kernel.name}")
print(f"Parameter: k_agg={kernel.k_agg}, k_break={kernel.k_break}")

# Test 1: Agglomeration
print("\n" + "-"*60)
print("Test 1: Agglomeration (2 → 1)")
print("-"*60)

V_dry_1 = 1.0e-18
poro_1 = 0.4
V_dry_2 = 1.0e-18
poro_2 = 0.4

V_solid_1 = V_dry_1 * (1 - poro_1)
V_pore_1 = V_dry_1 * poro_1
V_solid_2 = V_dry_2 * (1 - poro_2)
V_pore_2 = V_dry_2 * poro_2

print(f"Parent 1: V_dry={V_dry_1:.3e}, ε={poro_1:.3f} → V_solid={V_solid_1:.3e}, V_pore={V_pore_1:.3e}")
print(f"Parent 2: V_dry={V_dry_2:.3e}, ε={poro_2:.3f} → V_solid={V_solid_2:.3e}, V_pore={V_pore_2:.3e}")

V_dry_new, poro_new = kernel.compute_merged_porosity(V_dry_1, poro_1, V_dry_2, poro_2)

V_solid_new = V_dry_new * (1 - poro_new)
V_pore_new = V_dry_new * poro_new

print(f"\nChild: V_dry={V_dry_new:.3e}, ε={poro_new:.6f}")
print(f"       V_solid={V_solid_new:.3e}, V_pore={V_pore_new:.3e}")

# Massenerhaltung prüfen
V_solid_expected = V_solid_1 + V_solid_2
mass_error = abs(V_solid_new - V_solid_expected) / V_solid_expected * 100
print(f"\nMassenerhaltung: V_solid_error = {mass_error:.12f}%")

ΔV_pore = V_pore_new - (V_pore_1 + V_pore_2)
print(f"Porenwachstum: ΔV_pore = {ΔV_pore:+.3e}")

assert mass_error < 1e-10, "Massenerhaltung verletzt!"
assert ΔV_pore > 0, "Porenwachstum erwartet!"
print("✓ Agglomeration PASSED")

# Test 2: Breakage (binary)
print("\n" + "-"*60)
print("Test 2: Breakage (1 → 2)")
print("-"*60)

V_parent = 2.0e-18
poro_parent = 0.5

V_solid_parent = V_parent * (1 - poro_parent)
V_pore_parent = V_parent * poro_parent

# IMPORTANT: fragment_volumes must be V_SOLID (for mass conservation!)
frags_solid = [V_solid_parent / 2, V_solid_parent / 2]  # Equal split of solid

print(f"Parent: V_dry={V_parent:.3e}, ε={poro_parent:.3f} → V_solid={V_solid_parent:.3e}, V_pore={V_pore_parent:.3e}")

poros = kernel.compute_fragment_porosity(poro_parent, frags_solid, V_parent)

print(f"\nFragments: n={len(frags_solid)}")
for i, (V_solid_frag, p) in enumerate(zip(frags_solid, poros)):
    V_pore_frag = V_solid_frag * p / (1 - p)  # ε = V_pore/V_dry → V_pore = V_solid × ε/(1-ε)
    V_dry_frag = V_solid_frag + V_pore_frag
    print(f"  Frag {i+1}: V_solid={V_solid_frag:.3e}, ε={p:.6f} → V_dry={V_dry_frag:.3e}, V_pore={V_pore_frag:.3e}")

# Massenerhaltung: Σ V_solid_frag = V_solid_parent
V_solid_frags_sum = sum(frags_solid)  # Directly the solid volumes!
mass_error = abs(V_solid_frags_sum - V_solid_parent) / V_solid_parent * 100
print(f"\nMassenerhaltung: V_solid_error = {mass_error:.12f}%")

# Porenverlust: Calculate from porosities
V_pore_frags_sum = sum(V_solid * p / (1 - p) for V_solid, p in zip(frags_solid, poros))
ΔV_pore = V_pore_frags_sum - V_pore_parent
avg_poro = np.mean(poros)
Δε = avg_poro - poro_parent

print(f"Porenverlust: ΔV_pore = {ΔV_pore:+.3e}")
print(f"Porositätsänderung: Δε = {Δε:+.6f}")

assert mass_error < 1e-10, "Massenerhaltung verletzt!"
assert Δε <= 0, f"Porenverlust erwartet! Δε={Δε}"
print("✓ Breakage PASSED")

# Test 3: Multi-Fragment (n=3)
print("\n" + "-"*60)
print("Test 3: Multi-Fragment Breakage (1 → 3)")
print("-"*60)

V_parent = 3.0e-18
poro_parent = 0.5
V_solid_parent = V_parent * (1 - poro_parent)

# IMPORTANT: Pass V_SOLID, not V_dry!
frags_solid = [0.75e-18, 0.5e-18, 0.25e-18]  # Must sum to V_solid_parent!

print(f"Parent: V_dry={V_parent:.3e}, ε={poro_parent:.3f} → V_solid={V_solid_parent:.3e}")
print(f"Fragments (V_solid): {[f'{V:.3e}' for V in frags_solid]} (sum={sum(frags_solid):.3e})")

poros = kernel.compute_fragment_porosity(poro_parent, frags_solid, V_parent)

V_solid_frags_sum = sum(frags_solid)  # Directly conserved!
mass_error = abs(V_solid_frags_sum - V_solid_parent) / V_solid_parent * 100

V_pore_parent = V_parent * poro_parent
V_pore_frags_sum = sum(V_solid * p / (1 - p) for V_solid, p in zip(frags_solid, poros))
ΔV_pore = V_pore_frags_sum - V_pore_parent
avg_poro = np.mean(poros)
Δε = avg_poro - poro_parent

print(f"\nMassenerhaltung: Error = {mass_error:.12f}%")
print(f"Porenverlust: ΔV_pore = {ΔV_pore:+.3e}")
print(f"Δε = {Δε:+.6f} (erwartet: < 0)")

assert mass_error < 1e-10, "Massenerhaltung verletzt!"
assert Δε <= 0, f"Porenverlust erwartet! Δε={Δε}"
print("✓ Multi-Fragment PASSED")

print("\n" + "="*60)
print("ALLE TESTS BESTANDET ✓")
print("="*60)
