# CDF-Tabelle Analyse: Backup vs. Aktuelle Version

## Problem
Breakage Events finden statt (`real_break_events=100`), aber es werden **keine neuen Partikel erstellt** (a_tot bleibt 1000).

**Ursache:** Fragment-Volumina sind NaN oder 0 → werden gefiltert → keine Fragmente übrig.

---

## Wichtige Unterschiede identifiziert

### 1. `_build_fragments_stepwise()` - DEFAULT VALUES

**Backup:**
```python
def _build_fragments_stepwise(self, Vrem_k: np.ndarray) -> list[np.ndarray]:
    *_, pexp = self._get_break_tables_for_state(Vrem_k)
    p = float(pexp) if (pexp is not None and self.use_lmc_pre_model) else float(self.frag_num)
```

**Aktuelle Version:**
```python
def _build_fragments_stepwise(self, Vrem_k: np.ndarray) -> list[np.ndarray]:
    *_, pexp = self._get_break_tables_for_state(Vrem_k)
    p = float(pexp) if (pexp is not None and getattr(self, "use_lmc_pre_model", False)) 
                      else float(getattr(self, "frag_num", 2))
```

**Unterschied:** 
- Backup verwendet `self.frag_num` direkt → wirft AttributeError wenn nicht gesetzt
- Aktuell verwendet `getattr(self, "frag_num", 2)` → Default=2

**Auswirkung:** Minor (beide sollten frag_num=2 verwenden)

---

### 2. `_prepare_break_config()` - PL_Q DEFAULT

**Backup (base_solver.py):**
```python
self.pl_v = 2                         # number of fragments
self.pl_q = 1                         # parameter describes breakage type
```

**Aktuell (kernel_integration.py nach Fix):**
```python
solver.pl_v = float(getattr(self.break_kernel, 'pl_v', 2.0))
solver.pl_q = float(getattr(self.break_kernel, 'pl_q', 0.5))  # ← GEÄNDERT!
```

**Unterschied:**
- Backup Default: `pl_q = 1.0`
- Kernel Default: `pl_q = 0.5`
- Vor Fix: Aktuelle Version hatte `pl_q = 1.0` (falsch!)

**Auswirkung:** **KRITISCH!** 
- BREAKFVAL=3 verwendet `pl_q` in der Beta-Funktion
- Falsches `pl_q` → falsche CDF → NaN Werte

---

### 3. `_get_break_tables_for_state()` - NICHT GEFUNDEN

**Problem:** Diese Funktion wurde weder in Backup noch in aktueller Version gefunden!

**Vermutung:** Die Funktion ist in `pbe-core` (JIT-Code) implementiert.

**Mögliche Orte:**
- `pbe_core/func/jit_mcpbe.py`
- `pbe_core/func/jit_kernel_break.py`
- `pbe_core/base/base_solver.py` (als Mixin)

---

### 4. CDF-Berechnung für BREAKFVAL=3

**Aus `jit_kernel_break.py` (identisch in Backup und Aktuell):**

```python
elif BREAKFVAL == 3:     
    euler_beta = beta_func(q, q*(v-1))
    z = x/y
    theta = v * z**(q-1) * (1-z)**(q*(v-1)-1) / euler_beta
```

**Parameter:**
- `v = pl_v = 2.0` (Anzahl Fragmente)
- `q = pl_q` (Breakage Typ)

**Problem bei pl_q=1.0:**
```
theta = 2 * z**(0) * (1-z)**(2*1-1) / beta(1, 1)
      = 2 * 1 * (1-z)**1 / 1
      = 2 * (1-z)

CDF = integral(theta/y) dx von 0 bis y
    = integral(2*(1-x/y)/y) dx
    = 2/y * [x - x²/(2y)] von 0 bis y
    = 2/y * (y - y/2)
    = 2/y * y/2
    = 1  ✓ (korrekt)
```

**ABER bei z=0 oder z=1:**
- `z**(q-1) = 0**0` → undefiniert/kann NaN werden
- `(1-z)**(...) = 0**(...)` → kann NaN werden

**Lösung:** Numerische Stabilität an den Rändern sicherstellen!

---

## Empfohlene Fixes

### Fix 1: Numerische Stabilität in breakage_func_1d

```python
@njit
def breakage_func_1d(x, y, v, q, BREAKFVAL):
    if y <= 0 or x <= 0 or x > y:
        return 0.0
    
    z = x / y
    # CLAMP z to avoid 0^0 or negative base issues
    z = max(1e-15, min(1.0 - 1e-15, z))
    
    if BREAKFVAL == 3:
        euler_beta = beta_func(q, q*(v-1))
        theta = v * z**(q-1) * (1-z)**(q*(v-1)-1) / euler_beta
        # ...
```

### Fix 2: pl_q Default konsistent setzen

Stelle sicher, dass `pl_q=0.5` (nicht 1.0) für BREAKFVAL=3,4,5 verwendet wird.

### Fix 3: CDF-Tabelle validieren

Füge Debug-Output hinzu um CDF-Werte zu prüfen:
```python
print(f"CDF[0] = {rB[0]}, CDF[-1] = {rB[-1]}")
print(f"Any NaN: {np.any(np.isnan(rB))}")
print(f"Any Inf: {np.any(np.isinf(rB))}")
```

---

## Nächste Schritte

1. **`_get_break_tables_for_state()` in pbe-core finden**
   - Durchsuche `pbe-core/src/pbe_core/**/*.py`
   
2. **CDF-Tabelle debuggen**
   - Drucke rA, rB, rowsum_cdf vor Verwendung
   
3. **BREAKRVAL=5 testen**
   - Verwendet analytische Fragmente (KEINE Tabelle!)
   - Sollte funktionieren auch wenn Tabelle kaputt ist
