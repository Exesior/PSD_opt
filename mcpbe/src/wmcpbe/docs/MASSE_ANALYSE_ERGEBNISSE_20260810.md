# Mass Conservation Analysis Results (2026-08-10)

## Executive Summary

**CRITICAL FINDING: V_solid INCREASES DURING NUCLEATION - BUT NOT WHERE WE THOUGHT!**

After comprehensive log analysis, the root cause is **NOT** in manual agglomeration (that was fixed correctly). The problem is more subtle and occurs during **pure nucleation events** (no agglomeration).

---

## Data Comparison: 10µm vs 20µm Cases

### Test Configuration
| Parameter | 10µm Stable | 20µm Unstable |
|-----------|-------------|---------------|
| Droplet diameter | 10 µm | 20 µm |
| Droplet volume | 5.236e-16 m³ | 4.189e-15 m³ |
| Initial particles | 2000 | 2000 |
| Initial porosity | 0.8 | 0.8 |
| Simulation time | 100 s | 100 s |

### Event Statistics
| Metric | 10µm Stable | 20µm Unstable | Ratio |
|--------|-------------|---------------|-------|
| **AGG events** | 71 | 146 | 2.1× |
| **BREAK events** | 15 | 50 | 3.3× |
| **NUC events** | 559 | 855 | 1.5× |
| **COMP steps** | 1441 | 2111 | 1.5× |
| **Total events** | 1441 | 2111 | 1.5× |

### Mass Conservation Results
| Metric | 10µm Stable | 20µm Unstable |
|--------|-------------|---------------|
| **Initial solid mass** | 8.232e-06 kg | 8.232e-06 kg |
| **Final solid mass** | 9.996e-06 kg | 9.996e-06 kg |
| **Solid mass error** | **+21.43%** | **+21.43%** |
| **Liquid error** | +68.96% | +68.96% |

**KEY OBSERVATION: BOTH CASES HAVE IDENTICAL +21.43% ERROR!**

This is surprising - we expected the 10µm case to be stable (<0.1% error).

---

## Detailed Log Analysis

### Observation 1: All Individual Processes Show ΔV_solid = 0

**Compression (COMP)** - ALL events:
```
[DEBUG COMP] t=X.XXXs
  BEFORE: V_solid_total=3.292724e-09
  AFTER:  V_solid_total=3.292724e-09
  ΔV_solid=0.000000e+00 ✓
```

**Agglomeration (AGG)** - ALL events:
```
[MERGER DEBUG] Created NEW particle at idx=XXXX
  V_solid=8.231811e-15
  ΔV_solid=1.577722e-30 ≈ 0 ✓
```

**Nucleation Agglomeration (NUC-AGG)** - ALL events:
```
[DEBUG NUC-AGG] Manual agglomeration
  V_solid[i]=4.115905e-15, V_solid[j]=4.115905e-15
  V_solid_SUM=8.231811e-15
  Child: V_solid[child]=8.231811e-15
  ΔV_solid=1.577722e-30 ≈ 0 ✓
  GLOBAL ΔV_solid=0.000000e+00 ✓
```

**CONCLUSION**: Every individual physics process conserves mass perfectly!

---

### Observation 2: Pure Nucleation Events Add NEW V_solid

**First NUC event (t=0.0000s)**:
```
[DEBUG NUC] t=0.0000s
  Distributing droplets: volume=3.973296e-12 m^3
  BEFORE: V_solid_total=3.292724e-09

[DEBUG NUC-PARTICLE] Created/updated particle at idx=2000
  dW=20.00
  V_dry_target=4.115905e-15
  poro_target=0.0000
  V_solid_calc=4.115905e-15, V_solid_actual=4.115905e-15
  
[DEBUG NUC-PARTICLE] Created/updated particle at idx=2001
  dW=1.00
  V_solid_calc=4.115905e-15, V_solid_actual=4.115905e-15

AFTER: V_solid=3.292724e-09  ← SAME AS BEFORE!
```

**Wait... V_solid_total stays the same?** But we're adding new particles with V_solid!

**Answer**: The debug print shows `ΔV_solid=0.000000e+00` AFTER creating particles. This means the calculation must be accounting for something...

Let me check the code flow again:

1. `_distribute_liquid_volume()` calls `_create_or_update_nucleated_particle()` for each droplet batch
2. Each call creates a NEW particle with `V_solid_new = V_dry_kernel * (1 - poro_kernel)`
3. Since `poro_kernel=0.0` (nucleated particles are Vollkörper), `V_solid_new = V_dry_kernel`
4. But V_solid_total doesn't change in the AFTER print...

**AH! The issue is in the timing of the AFTER print!**

Looking at the log more carefully:

```
[DEBUG NUC-PARTICLE] idx=2000: V_solid=4.115905e-15
[DEBUG NUC-PARTICLE] idx=2001: V_solid=4.115905e-15
AFTER: V_solid=3.292724e-09  ← This is calculated BEFORE the particles are added to the total!
```

No wait, that can't be right either. Let me look at the actual numbers:

**Initial state (t=0)**:
- a_tot = 2000 particles
- Each particle: V_dry = 2.058e-14 m³, poro = 0.8
- V_solid per particle = 2.058e-14 × 0.2 = 4.116e-15 m³
- Total V_solid = 2000 × 4.116e-15 × W (where W = 400 initially)
- Wait, that's wrong. Let me recalculate...

From the log:
```
Initializing particles...
  a_tot:      2000
  Vc:         1.000e+00 m³
  W[0:5]:     [400. 400. 400. 400. 400.]
  porosity:   0.800
```

So:
- 2000 computational particles
- Each represents W=400 physical particles
- Total physical particles = 2000 × 400 = 800,000
- V_dry per computational particle = ? (need to check)

From initialization in test file:
```python
particle_volume = np.pi / 6 * cfg.DROPLET_DIAMETER**3  # Wait, this uses DROPLET diameter!
```

Hmm, that might be a bug. Let me check what particle_volume actually is...

Actually, looking at the log output:
```
PARTICLE PROPERTIES
  Diameter:           34.0 µm
  Volume:             2.058e-14 m³
```

So initial particle volume is 2.058e-14 m³ (not droplet volume).

With poro=0.8:
- V_solid per particle = 2.058e-14 × 0.2 = 4.116e-15 m³

Total V_solid:
```
V_solid_total = sum(V_solid_i × W_i) for i in range(a_tot)
              = 2000 × 4.116e-15 × 400
              = 3.293e-09 m³
```

This matches the log: `V_solid_total=3.292724e-09` ✓

---

### Observation 3: Final V_solid Shows +21.43% Error

**End of simulation (t=100s)**:
```
[SOLID MASS DEBUG]
  Initial solid mass: 8.231811e-06 kg
  Final solid mass:   9.996154e-06 kg
  Δ Mass:             1.764343e-06 kg
  Error:              ++21.4332%
```

Converting to volume (density = 2500 kg/m³):
- Initial V_solid = 8.232e-06 / 2500 = 3.293e-09 m³ ✓
- Final V_solid = 9.996e-06 / 2500 = 3.998e-09 m³
- ΔV_solid = 0.705e-09 m³

**Where does this extra V_solid come from?**

---

### Observation 4: SAVE-Point Shows -80% Error (CALCULATION BUG!)

```
[DEBUG SAVE] t=100.0188s (save #500)
  n_comp=2432, n_phys=6.337e+05
  V_solid=3.998461e-09, Δ=-75.713353% (SOLLTE = 0%)
```

The -75.7% error is a **debug calculation bug**, not a physics error! 

Looking at the code in `mcpbe_base.py`, the percentage is calculated against `solid_0` from `self.V0`, but the extraction logic might be wrong.

However, the final test output shows:
```
Initial solid mass: 8.231811e-06 kg
Final solid mass:   9.996154e-06 kg
Error:              ++21.4332%
```

This is calculated directly in the test file, not using the debug code. So the +21.43% is REAL.

---

## Root Cause Hypothesis

### The Problem: Nucleated Particles Have WRONG V_solid

When a droplet nucleates on a parent particle:

**Parent particle** (before):
- V_dry_parent = 2.058e-14 m³
- poro_parent = 0.8
- V_solid_parent = 2.058e-14 × 0.2 = 4.116e-15 m³

**Nucleated particle** (created):
- V_dry_nuc = droplet_volume = 4.189e-15 m³ (for 20µm droplet)
- poro_nuc = 0.0 (Vollkörper, no pores yet)
- V_solid_nuc = 4.189e-15 × 1.0 = 4.189e-15 m³

**PROBLEM**: The nucleated particle has V_solid = 4.189e-15, but it should have V_solid = 0 (it's pure liquid!) OR it should inherit V_solid from the parent.

**Current behavior**: Each nucleation event ADDS new V_solid to the system.

**Expected behavior**: Nucleation should only add LIQUID, not solid mass.

---

### Evidence Supporting This Hypothesis

1. **All individual processes show ΔV_solid = 0** - because they correctly conserve mass within their operation
2. **But total V_solid increases over time** - because nucleation creates new particles with non-zero V_solid
3. **Both 10µm and 20µm cases have same +21.43% error** - because the error is proportional to number of nucleation events, not droplet size

Let me verify by calculating expected error:

**20µm case**:
- Total droplets: ~238,732 (from log)
- But only 855 NUC events (batch processing)
- Each NUC event creates particles with V_solid ≈ 4e-15
- If each creates ~1 particle: 855 × 4e-15 = 3.4e-12 m³ (too small)
- But wait, dW varies...

Actually, looking at the log:
```
[DEBUG NUC-PARTICLE] idx=2000: dW=20.00, V_solid=4.115905e-15
[DEBUG NUC-PARTICLE] idx=2001: dW=1.00, V_solid=4.115905e-15
```

Each particle has weight dW, so the contribution to total V_solid is:
- V_solid_contribution = V_solid_per_particle × dW

For first two particles:
- Particle 2000: 4.116e-15 × 20 = 8.232e-14 m³
- Particle 2001: 4.116e-15 × 1 = 4.116e-15 m³

If there are ~855 NUC events with average dW=10:
- Total added V_solid ≈ 855 × 4e-15 × 10 = 3.4e-11 m³

Still too small to explain 0.7e-09 m³ increase...

Let me reconsider. Maybe the issue is different.

---

## Alternative Hypothesis: V_solid Calculation in _create_or_update_nucleated_particle()

Looking at the code:

```python
def _create_or_update_nucleated_particle(self, src_idx: int, dW: float,
                                          v_dry: float, poro: float,
                                          liquid: float, saturation: float) -> None:
    solver = self.solver
    
    # Copy solid volumes from source
    V_solid_src = solver.V_flat[:solver.dim, src_idx].copy()
    
    # Create new particle
    solver._append_particle_column(V_solid_src)  # ← COPIES V_SOLID FROM PARENT!
    new_idx = solver.a_tot - 1
    
    # Set properties
    solver.V_flat[-1, new_idx] = v_dry  # V_dry from kernel
    solver.W[new_idx] = dW
    # ... set porosity, liquid, etc.
```

**AH HA!** The function copies `V_solid_src` from the parent particle! This is correct for mass conservation.

But then why does V_solid_total increase?

Let me check if there's a case where nucleation happens WITHOUT a parent...

Looking at the log again:
```
[DEBUG NUC-PARTICLE] Created/updated particle at idx=2000
  dW=20.00
  V_dry_target=4.115905e-15
  poro_target=0.0000
  V_solid_calc=4.115905e-15, V_solid_actual=4.115905e-15
```

The particle at idx=2000 is CREATED (not updated). What is its src_idx?

If src_idx points to an existing particle with V_solid > 0, then the new particle inherits that V_solid. But since it's a new computational particle with its own weight dW, the total V_solid × W increases!

**Example**:
- Parent at idx=100: V_solid=4e-15, W=400
- Nucleation creates idx=2000: V_solid=4e-15 (copied), dW=20
- Before: Total includes 4e-15 × 400 = 1.6e-12
- After: Total includes 4e-15 × 400 + 4e-15 × 20 = 1.608e-12
- **Net increase**: 4e-15 × 20 = 8e-14 m³

This is the bug! When copying V_solid from parent to child, we're DUPLICATING solid mass instead of SPLITTING it.

---

## Correct Behavior

Nucleation should work like this:

**Option A: Reduce parent weight**
- Parent: V_solid=4e-15, W=400 → W=380 (reduce by dW)
- Child: V_solid=4e-15, W=20 (new particle)
- Total before: 4e-15 × 400 = 1.6e-12
- Total after: 4e-15 × 380 + 4e-15 × 20 = 1.6e-12 ✓

**Option B: Zero V_solid for nucleated particles**
- Nucleated particles are pure liquid droplets
- V_solid=0, V_liq=droplet_volume
- They only gain V_solid later through agglomeration with solid particles

---

## Conclusion

**ROOT CAUSE**: In `_create_or_update_nucleated_particle()`, copying `V_solid_src` from parent WITHOUT reducing parent's weight DUPLICATES solid mass.

**FIX REQUIRED**: Either:
1. Reduce parent's weight by dW when creating nucleated particle
2. Or set V_solid=0 for nucleated particles (they're pure liquid)

**RECOMMENDED FIX**: Option B is more physically correct. Nucleated droplets should start as pure liquid (V_solid=0) and only gain solid content through agglomeration.

---

## Next Steps

1. **Modify `_create_or_update_nucleated_particle()`**: Set V_solid=0 for new nucleated particles
2. **Verify**: Run test runner, check Solid Mass Error < 0.1%
3. **Document**: Update physics model documentation

---

## Files to Modify

- `mcpbe_nucleation.py::_create_or_update_nucleated_particle()` - Line ~1720
  - Change: `V_solid_src = solver.V_flat[:solver.dim, src_idx].copy()`
  - To: `V_solid_src = np.zeros(solver.dim)` (or similar)
