"""
Analyze mass conservation in nucleation logic.
Shows the bug in _distribute_liquid_volume() when parent is already porous.
"""
import numpy as np

print("=" * 80)
print("NUCLEATION MASS CONSERVATION ANALYSIS")
print("=" * 80)

# Scenario: Parent particle with porosity gets another droplet via agglomeration
# in _distribute_liquid_volume() -> already_nucleated branch

print("\n=== SCENARIO: Already nucleated particle receives another droplet ===\n")

# Initial state of parent particle
V_dry_parent_before = 1.667e-10  # m³ (example: V_solid=1e-10, poro=0.4 → V_dry = 1e-10/0.6)
poro_parent = 0.4
V_solid_parent = V_dry_parent_before * (1.0 - poro_parent)
V_pore_parent = V_dry_parent_before * poro_parent
W_parent_before = 100.0

print(f"Parent BEFORE nucleation event:")
print(f"  V_dry:    {V_dry_parent_before:.6e} m³")
print(f"  poro:     {poro_parent:.2f}")
print(f"  V_solid:  {V_solid_parent:.6e} m³")
print(f"  V_pore:   {V_pore_parent:.6e} m³")
print(f"  W:        {W_parent_before:.1f}")
print(f"  Mass:     {V_solid_parent * W_parent_before:.6e} (arbitrary units)")

# What happens in _distribute_liquid_volume() line ~1418:
# solver.V_flat[-1, i] = V_solid  ← THIS IS THE BUG!

V_dry_parent_after = V_solid_parent  # ← BUG: Should stay V_dry_parent_before!
W_parent_after = W_parent_before - 10.0  # dW consumed

print(f"\nParent AFTER nucleation (current code):")
print(f"  V_dry:    {V_dry_parent_after:.6e} m³ ← ⚠️  CHANGED!")
print(f"  poro:     {poro_parent:.2f} (unchanged)")
print(f"  V_solid:  {V_dry_parent_after * (1-poro_parent):.6e} m³ ← ⚠️  WRONG!")
print(f"  W:        {W_parent_after:.1f}")
print(f"  Mass:     {V_dry_parent_after * (1-poro_parent) * W_parent_after:.6e}")

# Calculate mass error
mass_before = V_solid_parent * W_parent_before
mass_after = (V_dry_parent_after * (1-poro_parent)) * W_parent_after
mass_error = (mass_after - mass_before) / mass_before

print(f"\nMass change:")
print(f"  Before:   {mass_before:.6e}")
print(f"  After:    {mass_after:.6e}")
print(f"  Error:    {mass_error*100:+.2f}%")

# Correct behavior
print(f"\n=== CORRECT BEHAVIOR ===")
print(f"Parent should keep V_dry unchanged!")
print(f"  V_dry:    {V_dry_parent_before:.6e} m³ ← stays same")
print(f"  poro:     {poro_parent:.2f}")
print(f"  V_solid:  {V_solid_parent:.6e} m³ ← stays same")
print(f"  W:        {W_parent_after:.1f}")
print(f"  Mass:     {V_solid_parent * W_parent_after:.6e}")
print(f"  Error:    {(W_parent_after - W_parent_before)/W_parent_before*100:+.2f}% (only from weight change)")

print("\n" + "=" * 80)
print("CONCLUSION: Line ~1418 must NOT set V_flat[-1,i] = V_solid!")
print("            Parent V_dry must remain unchanged!")
print("=" * 80)
