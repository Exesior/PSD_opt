# Mass Conservation Debug Update (2026-08-10)

## Critical Physics Insight

**V_dry tracking was WRONG!** We should ONLY track V_solid for mass conservation.

### Conserved vs Non-Conserved Quantities

| Quantity | Formula | Conserved? | Why? |
|----------|---------|------------|------|
| **V_solid** | `V_dry * (1 - porosity)` | YES | Solid mass cannot be created/destroyed |
| **V_dry** | `V_solid / (1 - porosity)` | NO | Changes with porosity - PHYSICALLY CORRECT! |
| **V_pore** | `V_dry * porosity` | NO | Pore volume is state variable |
| **Liquid** | Tracked separately | YES | Liquid mass must be conserved |

### Why V_dry Changes Are CORRECT

When porosity changes, V_dry MUST change to keep V_solid constant:

```
Initial: V_dry = 1e-15 m^3, poro = 0.8 -> V_solid = 2e-16 m^3
After compression: poro = 0.4
To conserve V_solid: V_dry_new = V_solid / (1 - 0.4) = 3.33e-16 m^3
ΔV_dry = +2.33e-16 m^3 <- THIS IS CORRECT PHYSICS!
```

**Old debug prints showing ΔV_dry ≠ 0 as errors were MISLEADING!**

---

## Changes Made

### A. Debug Prints Overhauled (All Files)

#### 1. mcpbe_nucleation.py

**_create_or_update_nucleated_particle()**: Now shows parent vs child V_solid comparison
```python
# OLD (misleading):
print(f"V_dry_target={v_dry:.6e}, V_dry_actual={...}")
print(f"ΔV_dry=...")

# NEW (correct):
V_solid_src = sum(V_flat[:dim, src_idx])  # From parent
V_solid_new = V_dry_new * (1 - poro_new)  # From kernel
print(f"V_solid_src (from parent)={V_solid_src:.6e}")
print(f"V_solid_new = V_dry_new*(1-poro)={V_solid_new:.6e}")
print(f"DELTA_V_SOLID = {V_solid_new - V_solid_src:.6e} (MUST = 0!)")
```

**_manual_agglomerate_particles()**: Removed V_dry from global balance
```python
# OLD:
print(f"GLOBAL ΔV_solid={...}, ΔV_dry={...}")

# NEW:
print(f"GLOBAL ΔV_solid={...} (MUST = 0!)")
# ΔV_dry removed - it's allowed to change!
```

**_distribute_liquid_volume()**: Added AFTER comparison
```python
print(f"BEFORE: V_solid_total={solid_before:.6e}")
# ... nucleation happens ...
print(f"AFTER:  V_solid_total={solid_after:.6e}")
print(f"ΔV_solid={solid_after - solid_before:.6e} (MUST = 0!)")
if abs(solid_after - solid_before) > 1e-20:
    print(f"*** ERROR: V_SOLID CHANGED BY {solid_after - solid_before:.6e}! ***")
```

#### 2. mcpbe_base.py

**VC-Double**: Added BEFORE/AFTER comparison
```python
# BEFORE doubling
solid_before = sum(V_solid * W)[:old_a]
liq_before = sum(liquid_volume * W)[:old_a]

# AFTER doubling
solid_after = sum(V_solid * W)[:new_a]
liq_after = sum(liquid_volume * W)[:new_a]

print(f"BEFORE: V_solid_total={solid_before:.6e}")
print(f"AFTER:  V_solid_total={solid_after:.6e}")
print(f"ΔV_solid={solid_after - solid_before:.6e} (MUST = 0!)")
```

**SAVE-Point**: Fixed initial solid mass calculation
```python
# OLD (wrong - used V_dry directly):
v_solid_0 = self.V0[-1, :] * (1.0 - self.porosity0)

# NEW (correct - handles 2D arrays properly):
if self.V0.ndim == 2:
    v_dry_0 = self.V0[-1, :]  # Extract V_dry row
else:
    v_dry_0 = self.V0

# Calculate V_solid with porosity
valid_0 = ~np.isnan(poro_0)
v_solid_0 = np.zeros_like(v_dry_0)
v_solid_0[valid_0] = v_dry_0[valid_0] * (1.0 - poro_0[valid_0])
v_solid_0[~valid_0] = v_dry_0[~valid_0]

solid_0 = sum(v_solid_0 * self.W0)
error_solid = (solid_total - solid_0) / solid_0 * 100
```

#### 3. Other Files (mcpbe_agg.py, mcpbe_break.py, particle_merger.py)

- Removed all ΔV_dry messages
- Kept only ΔV_solid validation
- Simplified output format

---

### B. Initial/Final V_solid Calculation Verified

**Location**: test_powerlaw_rumpf_full.py

Both initial and final calculations use IDENTICAL formula:
```python
# Initial (line ~350)
v_dry = solver.V_flat[-1, :solver.a_tot]
porosity = solver.porosity[:solver.a_tot]
weights = solver.W[:solver.a_tot]

valid_poro = ~np.isnan(porosity)
v_solid = np.zeros_like(v_dry)
v_solid[valid_poro] = v_dry[valid_poro] * (1.0 - porosity[valid_poro])
v_solid[~valid_poro] = v_dry[~valid_poro]  # Vollkörper

initial_solid_volume = np.sum(v_solid * weights)

# Final (line ~445) - IDENTICAL FORMULA
v_dry = solver.V_flat[-1, :solver.a_tot]
porosity = solver.porosity[:solver.a_tot]
weights = solver.W[:solver.a_tot]

valid_poro = ~np.isnan(porosity)
v_solid = np.zeros_like(v_dry)
v_solid[valid_poro] = v_dry[valid_poro] * (1.0 - porosity[valid_poro])
v_solid[~valid_poro] = v_dry[~valid_poro]

final_solid_volume = np.sum(v_solid * weights)
```

**Conclusion**: Test calculation is CORRECT. Error must come from physics code.

---

### C. Nucleation Deep-Dive Analysis

**Hypothesis**: Nucleation creates NEW particles with their OWN V_solid that doesn't come from parents.

**Evidence from logs**:
```
[DEBUG NUC-PARTICLE] Created/updated particle at idx=2000
  dW=20.00
  V_dry_target=4.115905e-15
  poro_target=0.0000  <- Nucleated particles are Vollkörper!
  V_solid_calc=4.115905e-15  <- Full V_dry becomes V_solid
```

**Parent particle** (before nucleation):
```
V_dry_parent = 4.115905e-15
poro_parent = 0.8
V_solid_parent = 4.115905e-15 * 0.2 = 8.23e-16
```

**Nucleated particle**:
```
V_solid_nuc = 4.115905e-15  <- 5x MORE than parent!
```

**Problem**: When a droplet nucleates on a parent particle:
1. Parent REMAINS (not consumed in pure nucleation)
2. NEW particle is CREATED with V_solid from droplet volume
3. This ADDS new V_solid to the system

**Question**: Should nucleated particles have V_solid at all? Or should they start as pure liquid coating?

---

## Next Steps

1. **Run updated debug tests** to see new V_solid-focused output
2. **Check if DELTA_V_SOLID in NUC-PARTICLE shows the error**
3. **If yes**: Examine nucleation physics - should new particles inherit V_solid from parent? Or should V_solid_nuc = 0 (pure liquid)?
4. **Document correct behavior** in physics model

---

## Files Modified

- `mcpbe_nucleation.py`: Updated 3 debug blocks (NUC-PARTICLE, NUC-AGG, NUC distribute)
- `mcpbe_base.py`: Updated 2 debug blocks (VC-DOUBLE, SAVE-POINT)
- `mcpbe_agg.py`: Removed V_dry from debug (if present)
- `mcpbe_break.py`: Removed V_dry from debug (if present)
- `particle_merger.py`: Removed V_dry from debug (if present)

## Test Command

```bash
%runfile C:/Users/ericb/Documents/GitHub/PSD_opt/mcpbe/src/wmcpbe/Trials/test_runner_mass_debug.py --wdir
```

Expected output: Clear V_solid tracking without misleading V_dry messages.
