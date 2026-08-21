# Nucleation Refactoring – Validation Report

**Datum:** 2025  
**Status:** ✅ LIQUID CONSERVATION PERFECT  
**⚠️ ISSUE:** Porosity growth seems too low (cone_model)

---

## 📊 TEST RESULTS (test_powerlaw_rumpf_full.py)

### ✅ PASSED: Liquid Conservation

| Metric | Expected | Actual | Error | Status |
|--------|----------|--------|-------|--------|
| **Total Liquid Added** | 3.000e-08 m³ | 3.000e-08 m³ | **0.0000%** | ✅ PASS |
| **Liquid In System** | 3.000e-08 m³ | 3.000e-08 m³ | **-0.0000%** | ✅ PASS |
| **Droplet Count** | 7161.97 | 7161.97 | **100.00%** | ✅ PASS |
| **Solid Mass** | 1.796e-01 kg | 1.796e-01 kg | **0.0000%** | ✅ PASS |

**🎉 HUGE IMPROVEMENT!** Previous refactoring had ~0.6% liquid loss – now **0.0000%**!\

---

### ⚠️ FAILED: Compression Check

```
Porosity Evolution:
  Initial:             0.000
  Final (mean):        0.004      ← VERY LOW!
  Final (min/max):     0.000 / 0.200
  Δ Porosity:          -0.004     ← Display bug (should be +0.004)
```

**Expected behavior:**
- Cone model should create porosity through agglomeration (ΔV > 0)
- After 7162 droplets: mean porosity should be higher than 0.4%!
- Compression should REDUCE porosity (not increase it)

**Test expects:** Δporo < 0 (compression active and reducing porosity)  
**Actual:** Δporo = -0.004 (display shows negative, but porosity INCREASED from 0.0 to 0.004)

---

## 🔍 ROOT CAUSE ANALYSIS: Low Porosity Growth

### Hypothesis 1: V_dry Check Too Aggressive?

```python
# In _distribute_one_droplet_with_dW():
while V_dry_sum < new_liquid and attempts < max_attempts:
    # Agglomerate to get more V_dry
```

**Question:** Does this prevent enough agglomeration for cone_model?

**Cone Model Physics:**
- Starts with poro=0.0 (V_dry = V_solid)
- Creates porosity ONLY through ΔV at agglomeration
- ΔV depends on collision energy, saturation, etc.

**Potential Issue:**
- If V_dry >= v_liquid immediately → NO agglomeration triggered
- But cone_model NEEDS agglomeration to create porosity!
- Catch-22: No agglomeration → no porosity → no compression possible

---

### Hypothesis 2: Cone Model Not Creating ΔV?

The cone_model kernel computes merged porosity based on:
- Collision energy
- Saturation
- Cone geometry parameters

**Check needed:**
1. Is cone_model actually creating ΔV > 0?
2. Are collision energies reasonable?
3. Is saturation being computed correctly?

---

### Hypothesis 3: Compression Overwrites Porosity Growth?

```
Compression Settings:
  Rate:               0.02 1/s
  Min porosity:       0.15
```

**Timeline:**
1. Nucleation creates particles with poro=0.0 (cone_model first contact)
2. Agglomeration should increase porosity via ΔV
3. Compression reduces porosity toward eps_min=0.15

**BUT:** If porosity starts at 0.0 and compression eps_min=0.15:
- Compression formula: `poro_new = eps_min + (poro - eps_min) * exp(-k*dt)`
- If poro < eps_min: **Compression does NOTHING** (protected by `if poro <= eps_min: return poro`)
- So compression can't be the issue!

---

## 🎯 CRITICAL INSIGHT

**The real problem:** Cone model with `poro=0.0` start may not create enough ΔV!

### Why?

Cone model porosity formation requires:
1. **Particle deformation** during collision
2. **Pore formation** from surface roughness/geometry
3. **Saturation-dependent** liquid bridge effects

But if particles start with **poro=0.0 (smooth spheres?)**:
- No initial roughness → less ΔV
- Low saturation (S=0.001 mean) → weak liquid bridges
- Result: Minimal porosity growth!

---

## 💡 PROPOSED SOLUTIONS

### Option A: Increase Cone Model Activity

Adjust cone_model parameters to create more ΔV:
- Higher cone angle
- Lower stiffness
- More aggressive saturation dependence

**File:** `kernels/porosity_growth/cone_model.py`

---

### Option B: Allow Some Initial Porosity for Cone Model

Instead of `poro=0.0` for cone_model first contact:
- Use `poro=0.1` or similar (surface roughness baseline)
- Still much lower than volume_mixing's 0.4
- Enables faster porosity growth

**File:** `mcpbe_nucleation.py` line ~1400
```python
# CURRENT:
if porosity_kernel.name == 'volume_mixing' and is_first_contact:
    nucleation_params = {'default_porosity': 0.4}
else:
    nucleation_params = None  # → cone_model uses 0.0

# PROPOSED:
if porosity_kernel.name == 'volume_mixing' and is_first_contact:
    nucleation_params = {'default_porosity': 0.4}
elif porosity_kernel.name == 'cone_model' and is_first_contact:
    nucleation_params = {'default_porosity': 0.1}  # Surface roughness baseline
else:
    nucleation_params = None
```

---

### Option C: Check Cone Model Implementation

Verify that cone_model.compute_merged_porosity() actually creates ΔV > 0 for:
- Two particles with poro=0.0
- Low saturation (S ≈ 0.001)
- Reasonable collision energy

**File:** `kernels/porosity_growth/cone_model.py`

---

## 📋 NEXT STEPS

### Immediate (Validation)
1. [ ] Add debug logging to cone_model.compute_merged_porosity()
2. [ ] Check ΔV values during agglomeration events
3. [ ] Verify collision energy calculation

### Short-term (Fix)
4. [ ] Try Option B (initial poro=0.1 for cone_model)
5. [ ] Re-run test and compare porosity evolution
6. [ ] Validate that compression becomes active

### Long-term (Physics)
7. [ ] Review cone_model parameters for physical correctness
8. [ ] Consider saturation-dependent initial porosity
9. [ ] Document expected porosity growth rates

---

## 📈 COMPARISON: BEFORE vs AFTER REFACTORING

### Liquid Conservation
| Version | Liquid Loss | Droplet Accuracy |
|---------|-------------|------------------|
| **Before** | ~0.6% ❌ | Unknown |
| **After** | **0.0000%** ✅ | **100.00%** ✅ |

### Porosity Growth
| Version | Mean Poro | Max Poro | Compression Active |
|---------|-----------|----------|-------------------|
| **Before** | Unknown | Unknown | Unknown |
| **After** | 0.004 (0.4%) ⚠️ | 0.200 (20%) | ✗ FAIL |

**Conclusion:** Liquid conservation is **PERFECT**, but porosity growth needs investigation!

---

## 🔬 DEEP DIVE: Why Only 0.4% Mean Porosity?

### Particle Statistics
```
n_comp:             1452
n_Vollkörper:       0
n_porous:           1452
```

All particles are porous (good!), but mean porosity is only 0.4%.

### Possible Causes:

1. **Most particles have poro ≈ 0.0**
   - Cone model didn't create much ΔV
   - Agglomeration events too few (only 26!)

2. **Few particles have high porosity (max 0.2)**
   - These skew the max but not the mean
   - Probably underwent multiple agglomeration events

3. **Low agglomeration rate**
   - Only 26 agglomeration events in 4 seconds
   - Breakage dominates (51 events)
   - Net effect: Particles break faster than they agglomerate

### Root Cause: **Breakage > Agglomeration**

```
Event Statistics:
  Agglomeration:       26 events
  Breakage:            51 events
  Agg/Break ratio:     0.51
```

**Interpretation:**
- Breakage creates fresh particles with low porosity
- Agglomeration can't keep up
- Mean porosity stays low

**This might be PHYSICALLY CORRECT** for these parameters!

---

## ✅ CONCLUSION

### What Works Perfectly:
✅ **Liquid conservation: 0.0000% error** (was 0.6% before!)  
✅ **Droplet accuracy: 100.00%**  
✅ **Solid mass conservation: 0.0000%**  
✅ **No crashes, stable simulation**

### What Needs Investigation:
⚠️ **Porosity growth too low** (0.4% mean after 7162 droplets)  
⚠️ **Compression not active** (porosity never exceeds eps_min=0.15)  
⚠️ **Agglomeration rate low** (26 events vs 51 breakage)

### Recommended Action:
1. **Check if low porosity is physically correct** for these parameters
2. **If not:** Adjust cone_model parameters or initial porosity
3. **Add diagnostic logging** to track ΔV per agglomeration event

---

**OVERALL ASSESSMENT:** 🟡 **SUCCESS WITH CAVEATS**

The refactoring achieved its primary goal (liquid conservation), but revealed a potential physics issue with cone_model porosity growth that deserves further investigation.
