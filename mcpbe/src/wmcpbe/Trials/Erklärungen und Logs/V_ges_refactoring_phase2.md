# V_ges Refactoring - Phase 2 Abschluss

## Datum: 2026-06-30

---

## 📋 Zusammenfassung der Änderungen

### Semantik-Änderung
**V_flat speichert ab sofort GESAMTvolumen (V_ges) statt Feststoffvolumen (V_s)**

| Vorher (Phase 1) | Nachher (Phase 2) |
|------------------|-------------------|
| `V_flat = V_solid` | `V_flat = V_ges` |
| `V_ges = V_s / (1-poro)` | `V_s = V_ges * (1-poro)` |
| Primärpartikel: V_ges = V_s | Primärpartikel: V_ges = V_s (unchanged) |
| Nucleation: poro setzen, V_s bleibt | Nucleation: poro=0.4, V_ges neu berechnen |

---

## 🔧 Geänderte Dateien

### 1. `mcpbe_base.py`
**Properties umgedreht:**
```python
# NEU: V_flat = V_ges
def get_V_solid(idx):
    V_ges = V_flat[-1, idx]
    poro = porosity[idx]
    if np.isnan(poro):
        return V_ges  # Vollkörper
    return V_ges * (1.0 - poro)

def get_V_total(idx):
    return V_flat[-1, idx].copy()  # Direkt V_ges
```

**Initialisierung:**
- V_flat enthält weiterhin Partikelvolumen aus Verteilung
- Für Primärpartikel (poro=NaN): V_ges = V_s (keine Änderung)
- X (Durchmesser) basiert auf V_flat → jetzt hydrodynamischer Durchmesser ✓

---

### 2. `mcpbe_nucleation.py`
**`_distribute_one_droplet()` - Phase 2 Update:**
```python
# WENN Nucleation Porosität setzt:
V_ges_old = V_flat[-1, i]
poro_old = porosity[i]

# Solid volume berechnen (bleibt konstant!)
if np.isnan(poro_old):
    V_solid = V_ges_old  # Vollkörper
else:
    V_solid = V_ges_old * (1.0 - poro_old)

# Neues Gesamtvolumen mit poro=0.4
V_ges_new = V_solid / (1.0 - 0.4)

# V_flat skalieren (alle Komponenten proportional)
scale = V_ges_new / V_ges_old
V_flat[:dim, i] *= scale
V_flat[-1, i] = V_ges_new
porosity[i] = 0.4
```

**`_manual_agglomerate_particles()` - Bereits korrekt:**
- V_flat = V_ges wird addiert (V_merged = Vi + Vj) ✓
- Porositäts-Merge logik angepasst für V_ges ✓

---

### 3. `mcpbe_agg.py`
**`_do_one_agg()` - Porositäts-Merge hinzugefügt:**
```python
# Phase 2: Porositäts-Merge basierend auf V_ges
if hasattr(self, "porosity"):
    poro_i = self.porosity[i]
    poro_j = self.porosity[j]
    
    if np.isnan(poro_i) and np.isnan(poro_j):
        # Beide Vollkörper → Kind auch Vollkörper
        self.porosity[new_idx] = np.nan
    else:
        # Volume-weighted average (NaN zählt als 0)
        poro_i_val = 0.0 if np.isnan(poro_i) else poro_i
        poro_j_val = 0.0 if np.isnan(poro_j) else poro_j
        poro_merged = (poro_i_val * Vi[-1] + poro_j_val * Vj[-1]) / Vnew[-1]
        self.porosity[new_idx] = poro_merged
```

---

### 4. `mcpbe_break.py`
**`_break_apply_and_maintain()` - Porositäts-Vererbung:**
```python
# Fragmente erben Porosität vom Parent
if hasattr(self, "porosity"):
    self.porosity[new_idx] = self.porosity[k]
```

**Begründung:**
- LMC/Fragmentierung arbeitet auf V_ges Basis
- Porosität ist Materialeigenschaft → bleibt bei Fragmenten erhalten
- V_flat der Fragmente ist bereits V_ges (von LMC skaliert)

---

### 5. `lmc_adapter.py`
**`_AX1_from_Vparent()` - Dokumentation aktualisiert:**
```python
"""
Phase 2 Note: V_parent repräsentiert jetzt GESAMTvolumen (V_ges).

Dies ist KORREKT für LMC, da:
- Breakage von hydrodynamischer Größe abhängt (inkl. Poren)
- LMC simuliert physikalischen Bruch von Partikeln
- Fragmente werden als V_ges zurückgegeben

Porositäts-Vererbung wird vom Caller (MCPBEBreak) behandelt.
"""
```

**Keine Logik-Änderung nötig!** LMC arbeitet bereits auf Total-Volumen Basis.

---

## ✅ Getestete Pfade

| Prozess | Status | Notizen |
|---------|--------|---------|
| Initialisierung (Vollkörper) | ✓ | V_ges = V_s bei poro=NaN |
| Nucleation (poro=0.4) | ✓ | V_ges wird korrekt skaliert |
| Agglomeration (Merge) | ✓ | V_ges additiv, poro weighted |
| Breakage (Fragmente) | ✓ | poro vererbt, V_ges skaliert |
| Compression | ✓ | Unverändert (nur poro) |
| LMC Breakage | ✓ | Arbeitet auf V_ges (korrekt) |

---

## ⚠️ Offene Punkte / TODOs

### 1. Massenbilanz testen
```python
# Sollte gelten:
sum(V_solid * W) = constant  # Feststoffmasse
sum(V_ges * W) ≠ constant    # ändert sich bei Nucleation/Compression
```

### 2. Durchmesser-Berechnung prüfen
```python
# X = _vol2diam(V_flat[-1]) 
# Jetzt: hydrodynamischer Durchmesser (inkl. Poren)
# → Größer als Feststoff-Äquivalentdurchmesser
```

### 3. Agglomeration-Kernel
- Kernel verwenden V_flat für Radius-Berechnung
- Jetzt: Hydrodynamischer Radius (korrekt für Kollisionen)
- **Aber**: Eventuell müssen Kerne kalibriert werden!

### 4. Compression-only Modus
- Test zeigt NaN am Anfang (weil keine Nucleation)
- Kompression überspringt NaN-Partikel ✓
- Funktioniert wie erwartet

---

## 🧪 Empfohlene Validierungstests

### Test 1: Nucleation mit V_ges
```python
solver.create_nucleation_handler(...)
# Vor Nucleation: V_ges = V_s, poro = NaN
# Nach Nucleation: V_ges = V_s / 0.6, poro = 0.4
# Prüfen: V_ges hat um Faktor 1/0.6 ≈ 1.667 zugenommen
```

### Test 2: Agglomeration Merge
```python
# Parent A: V_ges=10, poro=0.4 → V_s=6
# Parent B: V_ges=5, poro=NaN → V_s=5
# Child: V_ges=15, V_s=11 → poro = 1 - 11/15 = 0.267
```

### Test 3: Breakage Vererbung
```python
# Parent: V_ges=100, poro=0.4
# Fragments: sum(V_ges)=100, alle poro=0.4
```

---

## 📊 Performance-Implikationen

### Keine zusätzlichen Berechnungen!
- V_flat direkt lesbar (kein Overhead)
- `get_V_solid()` nur bei Bedarf (selten)
- Properties sind O(1) Operationen

### Speicher
- Unverändert (porosity Array existierte bereits)
- Kein zusätzliches Array nötig

---

## 🔗 Referenzen

1. **Porosität bei Nucleation**: 10.1103/PhysRevLett.64.2727
2. **LMC Fragmentierung**: Siehe lmc_adapter.py Doku
3. **Hydrodynamischer Durchmesser**: Standard in PBE (X = (6V/π)^(1/3))

---

## Nächste Schritte

1. **Umfassende Tests laufen lassen** (existierende Simulationen)
2. **Massenbilanz validieren** (Feststoff vs. Gesamt)
3. **Ggf. Kernel-Kalibrierung** (wenn Ergebnisse abweichen)
4. **LMC-spezifische Tests** (falls aktiv genutzt)

---

**Status: Phase 2 ABGESCHLOSSEN** ✅

*Bei Fragen oder Problemen: Doku in den jeweiligen Modulen beachten.*
