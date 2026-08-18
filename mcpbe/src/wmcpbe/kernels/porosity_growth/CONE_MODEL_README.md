# Cone Model Porosity Kernel - Implementierungsdokumentation

## Übersicht

Das **Cone Model** ist ein geometriebasiertes Porositätsmodell, das Porenbildung und Porenverlust aus der Kontaktgeometrie zwischen sphärischen Partikeln ableitet.

---

## Physikalisches Modell

### Agglomeration (2 → 1)

```
Vor der Agglomeration:          Nach der Agglomeration:

   ○                            ╭─────╮
  Partikel i                   ╱       ╲
                              │   i    │◄── Kegelstumpf ──►│   j    │
                               ╲       ╱
                                ╰─────╯
   ○                           Kegelpille
  Partikel j
```

**Formeln:**

| Größe | Formel | Bedeutung |
|-------|--------|-----------|
| `V_cone` | `(1/3)×π×(ri+rj)×(ri² + ri×rj + rj²)` | Volumen des Kegelstumpfs |
| `V_hemi` | `(2/3)×π×(ri³ + rj³)` | Volumen der 2 Halbkugeln |
| `V_pill` | `V_cone + V_hemi` | Gesamtvolumen der Kegelpille |
| `ΔV` | `V_pill - (Vi + Vj)` | Zusätzliches Volumen durch Kontakt |

**Für gleiche Radii (ri = rj = r):**
```
V_old = (8/3)×π×r³
V_pill = (10/3)×π×r³
ΔV = (2/3)×π×r³  (> 0!)
```

### Breakage (1 → n)

```
Parent (Kugel)              Fragmente mit Kegelkontakten:
     ○                         ╭─╮ ╭─╮ ╭─╮
                              ╱   ╲╱   ╲╱   ╲
                             │ 1  │ 2  │ 3  │ ... │ n │
                              ╲   ╱╲   ╱╲   ╱
                               ╰─╯ ╰─╯ ╰─╯
                                 ↑   ↑
                            (n-1) Kegelkontakte
```

**Modell:**
- Bei n Fragmenten entstehen **(n-1) Kegelkontakte** in einer linearen Kette
- Jeder Kontakt repräsentiert neue Oberfläche → **Porenverlust**
- Gesamt-ΔV = Σ ΔV_pair über alle benachbarten Paare

---

## Massenerhaltung (KRITISCH!)

### Invariante: V_solid bleibt IMMER erhalten

```python
# Agglomeration
V_solid_new = V_solid_i + V_solid_j  # ← CONSERVED!
V_pore_new = V_pore_i + V_pore_j + k_agg × ΔV
V_dry_new = V_solid_new + V_pore_new
ε_new = V_pore_new / V_dry_new

# Breakage
V_solid_parent = Σ V_solid_frag_k  # ← CONSERVED!
V_pore_new = V_pore_parent - k_break × ΣΔV
V_pore_frag_k = V_pore_new × (V_solid_frag_k / V_solid_parent)
V_dry_frag_k = V_solid_frag_k + V_pore_frag_k
ε_frag_k = V_pore_frag_k / V_dry_frag_k
```

### Alle Volumina sind INTENSIV

```
W[i]                 = Anzahl PHYSIKALISCHER Partikel (extensiv)
liquid_volume[i]     = Flüssigkeit PRO physikalischem Partikel (intensiv)
V_flat[:dim, i]      = V_solid PRO physikalischem Partikel (intensiv)
V_flat[-1, i]        = V_dry PRO physikalischem Partikel (intensiv)
porosity[i]          = V_pore / V_dry (dimensionslos)
saturation[i]        = V_liq_int / V_pore (dimensionslos)
```

**Konsequenz:** Bei Gewichtsänderung (W) bleiben intensive Größen UNVERÄNDERT!

---

## Fitting-Parameter

### Warum zwei Parameter?

| Parameter | Zweck | Typischer Bereich | Begründung |
|-----------|-------|-------------------|------------|
| `k_agg` | Skaliert ΔV bei Agglomeration | 0.1 - 2.0 | 1 Kegel zwischen 2 Partikeln |
| `k_break` | Skaliert ΔV bei Breakage | 0.01 - 0.5 | (n-1) Kegel in Reihe → stärkere Dämpfung nötig |

**Physikalische Interpretation:**
- `k_agg`: Anteil von ΔV, der zu zugänglichem Porenraum wird
  - k_agg = 1.0: Volles ΔV wird zu Poren
  - k_agg < 1.0: Nur Teil wird zu Poren (Rest = Kompression)
  - k_agg > 1.0: Mehr Poren als geometrisches ΔV (Oberflächenrauheit)
  
- `k_break`: Effektiver Porenverlust pro ΔV
  - Klein wegen (n-1) Kegel in Reihe
  - accounts for pore collapse at fresh fracture surfaces

---

## Verwendung

### Installation

Das Kernel ist bereits registriert in:
```python
# wmcpbe/kernels/porosity_growth/__init__.py
POROSITY_GROWTH_KERNELS = {
    'volume_mixing': VolumeMixingKernel,
    'incomplete_mixing': IncompleteMixingKernel,
    'cone_model': ConeModelKernel,  # ← NEU!
}
```

### Beispiel: Solver mit Cone Model

```python
from wmcpbe import MCPBESolver

# Solver mit Cone Model erstellen
solver = MCPBESolver(
    dim=1,
    t_total=600,
    t_write=10,
    porosity_growth_kernel_name='cone_model',
    porosity_growth_kernel_params={
        'k_agg': 0.5,      # Moderate Porenbildung
        'k_break': 0.1,    # Geringer Porenverlust
    }
)

# Initialisierung wie gewohnt
solver.initialize_particles(...)
solver.solve()
```

### Beispiel: Kernel direkt verwenden

```python
from wmcpbe.kernels.porosity_growth import get_porosity_growth_kernel

kernel = get_porosity_growth_kernel(
    'cone_model',
    k_agg=0.5,
    k_break=0.1
)

# Agglomeration
V_dry_new, poro_new = kernel.compute_merged_porosity(
    v_dry1=1e-18, poro1=0.4,
    v_dry2=2e-18, poro2=0.5
)

# Breakage
poros_frags = kernel.compute_fragment_porosity(
    parent_porosity=0.45,
    fragment_volumes=[0.5e-18, 0.3e-18, 0.2e-18],
    parent_volume=1.0e-18
)
```

---

## Testen

### Test-Suite ausführen

```bash
cd wmcpbe/kernels/porosity_growth
python test_cone_model.py
```

### Getestete Eigenschaften

1. ✓ Radius-Volumen-Konversion (Round-Trip)
2. ✓ Kegelpillen-Volumen (analytische Lösung für gleiche Kugeln)
3. ✓ Massenerhaltung bei Agglomeration (V_solid konstant)
4. ✓ Massenerhaltung bei Breakage (Σ V_solid_frag = V_solid_parent)
5. ✓ Vollkörper-Handling (NaN Porosität)
6. ✓ Fitting-Parameter-Effekte (k_agg, k_break)
7. ✓ Randfälle (nm-Partikel, große Größenverhältnisse, viele Fragmente)

---

## Integration im Solver

### Agglomeration (`mcpbe_agg.py::_do_one_agg`)

**Aktueller Code (Phase 5):**
```python
# Porositäts-Kernel aufrufen
porosity_kernel = self.kernel_manager.porosity_growth_kernel

V_dry_merged, poro_merged = porosity_kernel.compute_merged_porosity(
    v_dry1=float(Vi_dry), poro1=poro_i,
    v_dry2=float(Vj_dry), poro2=poro_j,
    v_liq1=lv_i, v_liq2=lv_j,
    sat1=sat_i, sat2=sat_j,
    solver=self
)

# V_flat aktualisieren (Massenerhaltung!)
self.V_flat[:self.dim, new_idx] = V_solid_merged  # ← UNVERÄNDERT
self.V_flat[-1, new_idx] = V_dry_merged           # ← Aktualisiert!
```

**Mit Cone Model:**
- `compute_merged_porosity()` berechnet V_dry_merged intern korrekt
- V_solid bleibt erhalten (wird nicht modifiziert!)
- Nur V_dry ändert sich durch neues Porenvolumen

### Breakage (`mcpbe_break.py::_break_apply_and_maintain`)

**Aktueller Code (Phase 4):**
```python
# Fragment-Porosität via Kernel
frag_poro = porosity_kernel.compute_fragment_porosity(
    parent_porosity=parent_poro,
    fragment_volume=V_solid_frag,  # ← Einzelnes Fragment
    parent_volume=float(np.sum(self.V_flat[:self.dim, k])),
    solver=self
)
```

**Anpassung für Cone Model (Multi-Fragment):**
```python
# ALLE Fragmente sammeln
fragment_volumes = [float(np.sum(f)) for f in frags]

# Einmaliger Aufruf für ALLE Fragmente (effizient!)
poros_all = porosity_kernel.compute_fragment_porosity(
    parent_porosity=parent_poro,
    fragment_volumes=fragment_volumes,  # ← Liste!
    parent_volume=parent_volume,
    solver=self
)

# Jedem Fragment seine Porosität zuweisen
for idx_frag, frag_poro in enumerate(poros_all):
    self.porosity[new_indices[idx_frag]] = frag_poro
```

---

## Kalibrierung an Experimente

### Empfohlene Vorgehensweise

1. **Startwerte setzen:**
   ```python
   k_agg = 0.5    # Mittlere Porenbildung
   k_break = 0.1  # Geringer Porenverlust
   ```

2. **Agglomeration-only Experiment kalibrieren:**
   - Simulation ohne Breakage
   - k_agg variieren bis PSD matcht
   - Typisch: 0.3 - 1.0

3. **Breakage-only Experiment kalibrieren:**
   - Simulation ohne Agglomeration
   - k_break variieren bis PSD matcht
   - Typisch: 0.05 - 0.3

4. **Vollprozess validieren:**
   - Beide Prozesse aktiv
   - Vorhersagequalität prüfen

### Sensitivität

```
k_agg-Effekt (Agglomeration):
  k_agg ↑ → ε_new ↑ → größere Aggregate → PSD shift zu größeren x

k_break-Effekt (Breakage):
  k_break ↑ → ε_frag ↓ → kleinere Fragmente → PSD shift zu kleineren x
```

---

## Fehlerbehandlung

### Warnungen

```python
# Zu großes k_break
UserWarning: k_break=2.0 is large. This may cause negative pore volumes 
for multi-fragment breakage. Recommended range: 0.01 - 0.5

# Fragment-Volumina summieren nicht zum Parent
UserWarning: Fragment volumes (2.9e-18) don't sum to parent volume 
(3.0e-18). Normalizing fragments.
```

### Guards

- ΔV ≥ 0 immer (geometrisch garantiert)
- V_pore_new ≥ 0 (geclamped)
- 0 ≤ ε < 1 (geclamped auf [0, 0.9999])
- NaN-Porosität wird vererbt (Vollkörper-Status)

---

## Grenzen des Modells

### Annahmen

1. **Sphärische Partikel:** Radien werden aus V_dry als Äquivalentkugeln berechnet
2. **Lineare Kette:** Fragmente sind linear angeordnet (n-1 Kontakte)
3. **Keine Kompression:** ΔV wird direkt zu Poren addiert/subtrahiert
4. **Uniforme Verteilung:** Poren werden proportional zu V_solid verteilt

### Erweiterungen möglich

- **Andere Topologien:** Stern-Graph, Complete Graph statt linearer Kette
- **Größenabhängigkeit:** k_agg(r), k_break(r) statt konstant

---

## Zusammenfassung

| Aspekt | Implementierung |
|--------|-----------------|
| **Geometrie** | Kegelstumpf + 2 Halbkugeln |
| **Agglomeration** | ΔV hinzugefügt (Porenwachstum) |
| **Breakage** | ΔV abgezogen (Porenverlust) |
| **Massenerhalt** | V_solid IMMER konstant |
| **Intensive Größen** | Alle Volumina PRO physikalischem Partikel |
| **Parameter** | k_agg (0.1-2.0), k_break (0.01-0.5) |
| **API** | Backward-compatible (scalar/list input) |

---

## Nächste Schritte

1. **In Solver integrieren:** `_break_apply_and_maintain()` für Multi-Fragment aufrufen
2. **Tests erweitern:** Gegen experimentelle Daten validieren
3. **Dokumentation:** ARCHITEKTUR.md um Cone Model ergänzen
4. **Performance:** Bei vielen Fragmenten ggf. optimieren (JIT?)
