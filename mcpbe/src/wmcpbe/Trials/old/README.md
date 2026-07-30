# Validation Tests for Continuous Processes Kernels

Dieser Ordner enthält Validierungstests für die neuen Continuous Processes Kernel.

## Tests

### 1. Porosity Compression Kernel
**Datei:** `test_porosity_compression.py`

Validiert den Porosity Compression Kernel gegen die analytische Lösung:
```
ε(t) = ε_min + (ε_0 - ε_min) × exp(-k×t)
```

**Ausführung:**
```bash
python Trials/test_porosity_compression.py
```

**Erwartetes Ergebnis:** Maximale Abweichung < 1e-10

---

### 2. Liquid Internalization Kernel
**Datei:** `test_liquid_internalization.py`

Validiert den Liquid Internalization Kernel durch:
1. Konvergenztest (Vergleich mit feiner Zeitdiskretisierung)
2. Gleichgewichtstest (S → l_total/v_pore)

**ODE:**
```
dl_intern/dt = k × (l_total - l_intern) × (v_pore - l_intern)
```

**Ausführung:**
```bash
python Trials/test_liquid_internalization.py
```

**Erwartetes Ergebnis:** 
- Maximale Abweichung von Referenz < 1e-3
- Gleichgewichtsfehler < 1e-6

---

## Hintergrund

Die Tests validieren die Implementierung nach Braumann et al. (2007):
- **Liquid Internalization:** Kapillar-getriebene Flüssigkeitsaufnahme in Poren
- **Porosity Compression:** Exponentielle Porositätsreduktion unter Scherung

Beide Prozesse sind zeitkontinuierlich und werden via Operator Splitting nach jedem MC-Event angewendet.
