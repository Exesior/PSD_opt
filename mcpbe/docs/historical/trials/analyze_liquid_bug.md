# Liquid Volume Bug Analyse

## Das Konzept (korrekt)

```
liquid_volume[k] = Flüssigkeit PRO PHYSIKALISCHEM PARTIKEL (intensive Größe)

Computational Particle k:
  - W[k] = Anzahl repräsentierter physikalischer Partikel
  - liquid_volume[k] = Flüssigkeit JEDES dieser Partikel
  
Gesamtflüssigkeit im System = Σ liquid_volume[k] × W[k] / Vc
```

---

## Beispiel: Agglomeration von 2 Partikeln

### Ausgangszustand (vor Agglomeration)
```
Particle A: W[A]=50, liquid_volume[A]=v_droplet
Particle B: W[B]=50, liquid_volume[B]=v_droplet

System-Flüssigkeit = (v_droplet × 50 + v_droplet × 50) / Vc
                   = 100 × v_droplet / Vc  ✓
```

### Agglomeration: dW=10 wird merged

**SCHRITT 1: Child erzeugen**
```
Child C: W[C]=dW=10, liquid_volume[C]=liquid_volume[A]+liquid_volume[B]
                                     = v_droplet + v_droplet
                                     = 2×v_droplet  ✓ KORREKT
```

**SCHRITT 2: Parents reduzieren**

### ❌ FALSCH (alte Implementierung mit Skalierung):
```python
W[A] = 50 - 10 = 40
liquid_volume[A] = v_droplet × (40/50) = 0.8×v_droplet  ❌ FEHLER!

W[B] = 50 - 10 = 40  
liquid_volume[B] = v_droplet × (40/50) = 0.8×v_droplet  ❌ FEHLER!

System danach:
  Child C:  2×v_droplet × 10 = 20×v_droplet
  Parent A: 0.8×v_droplet × 40 = 32×v_droplet
  Parent B: 0.8×v_droplet × 40 = 32×v_droplet
  SUMME: 84×v_droplet  ❌ SOLLTE 80×v_droplet SEIN!
```

### ✅ RICHTIG (meine Korrektur OHNE Skalierung):
```python
W[A] = 50 - 10 = 40
liquid_volume[A] = v_droplet  ✓ UNVERÄNDERT (intensive Größe!)

W[B] = 50 - 10 = 40
liquid_volume[B] = v_droplet  ✓ UNVERÄNDERT

System danach:
  Child C:  2×v_droplet × 10 = 20×v_droplet
  Parent A: 1×v_droplet × 40 = 40×v_droplet  
  Parent B: 1×v_droplet × 40 = 40×v_droplet
  SUMME: 100×v_droplet  ✓ KORREKT!
```

---

## ABER: Warum zeigt der Test +9827% Fehler?

### Mögliche Ursache 1: Control Volume Verdopplung

Bei `run_calculate()` wird bei Bedarf `_double_control_volume()` aufgerufen:

```python
def _double_control_volume(self):
    # Double Vc
    self.Vc *= 2.0
    
    # Double weights to keep physical particle count constant
    self.W[:self.a_tot] *= 2.0
    
    # ALSO DOUBLE LIQUID_VOLUME???
    lv_active = self.liquid_volume[:self.a_tot].copy()
    lv_dup = np.concatenate((lv_active, lv_active))
    self.liquid_volume[:self.a_tot] = lv_dup
```

**Problem:** Hier wird `liquid_volume` DUPPLIERT aber auch `Vc` verdoppelt!

```
Vorher: Σ liq[k] × W[k] / Vc
Nachher: Σ liq[k] × (2×W[k]) / (2×Vc) = Σ liq[k] × W[k] / Vc  ✓

ABER: lv_dup verdoppelt die ANZAHL der Partikel mit gleichem liq!
→ Σ läuft über 2× soviele Terme!
→ Ergebnis: 2× zu groß! ❌
```

### Mögliche Ursache 2: Nucleation verteilt zu oft

Wenn `_distribute_remaining_liquid()` mehr Tropfen verteilt als erwartet...

### Mögliche Ursache 3: Self-Agglomeration Bug

In `mcpbe_agg.py` Zeile ~280:
```python
if i == j:
    # Self-agglomeration: particle merges with itself
    V_liq_int_contrib = 2.0 * V_liq_int_i_full  # ← Eigenschaften verdoppeln
    V_liq_ext_contrib = 2.0 * V_liq_ext_i_full
```

Aber Parent-Reduktion (Zeile ~330):
```python
if i == j:
    w_rem = w_now - 2.0 * dW
    if w_rem > 0.0:
        self.W[i] = w_rem
        # KEINE Skalierung von liquid_volume!
```

**Das ist korrekt!** Bei Selbst-Agglomeration:
- Ein Partikel "mergt mit sich selbst" → Child hat 2× Eigenschaften
- Parent verliert 2×dW an Gewicht, behält aber gleiche Properties pro Partikel

---

## Diagnose-Plan

1. **Test OHNE Agglomeration** (CORR_BETA=1e-12):
   - Wenn Error ≈ 0%: Problem LIEGT in Agglomeration
   - Wenn Error >> 0%: Problem LIEGT woanders (Nucleation/CV-Doubling)

2. **Test MIT Agglomeration aber OHNE CV-Doubling**:
   - Größeres initiales Vc oder weniger Partikel
   - Wenn Error verschwindet: Problem ist CV-Doubling!

3. **Logging während Simulation**:
   - Nach jedem MC-Event: Σ liq×W/Vc berechnen
   - Plotten vs. Zeit: Wo springt der Wert?
