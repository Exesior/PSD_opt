# Nucleation Methods - Vergleich und Dokumentation

## Überblick

Es gibt drei Implementierungen der Nukleation im MC-PBE Solver:

### 1. STANDARD (`MCPBENucleation`)
- **Datei**: `mcpbe_nucleation.py`
- **Solver**: `MCPBESolver` (default)
- **Strategie**: Sampler wird bei JEDEM Tropfen neu gebaut
- **Genauigkeit**: Höchste (korrekteste)
- **Performance**: Langsam (~1x)
- **Use Case**: Referenzimplementierung, kleine Simulationen

**Algorithmus**:
```python
for each droplet:
    1. Build sampler from ALL particles (weighted by solid_volume)
    2. Sample particle i
    3. If i too small:
       - Rebuild sampler
       - Sample particle j
       - Agglomerate i + j → new i
       - Repeat until i can accept droplet
    4. Add droplet to i
```

**Problem**: Nach jeder Agglomeration ändert sich die Partikelliste (swap-pop), daher muss der Sampler neu gebaut werden. Das ist korrekt, aber sehr ineffizient bei vielen Tropfen.

---

### 2. CACHED (`MCPBENucleationSampleOpti`)
- **Datei**: `mcpbe_nucleation_sampleopti.py`
- **Solver**: `MCPBESolverOpti`
- **Strategie**: Sampler wird gecached, nur nach Agglomeration neu bauen
- **Genauigkeit**: Hoch (identisch zu Standard)
- **Performance**: ~100-1000x schneller als Standard
- **Use Case**: Die meisten Anwendungsfälle!

**Algorithmus**:
```python
cache_valid = False
sampler = None

for each droplet:
    if not cache_valid:
        sampler = build_sampler()
        cache_valid = True
    
    1. Sample particle i from cached sampler
    2. If i too small:
       - Sample particle j from SAME sampler
       - Agglomerate i + j → new i
       - cache_valid = False  # Invalidate after agglomeration!
       - Repeat
    3. Add droplet to i
```

**Optimierung**: Der Sampler bleibt gültig solange sich die Partikelliste nicht ändert. Nach Agglomeration (swap-pop) wird er invalidiert und beim nächsten Tropfen neu gebaut.

**Ergebnis**: Bei Fällen OHNE Agglomeration wird der Sampler nur EINMAL gebaut (statt tausende Male)!

---

### 3. DUAL-TIMESTEP (`MCPBENucleationTimestep`) ⭐ NEU
- **Datei**: `mcpbe_nucleation_timestep.py`
- **Solver**: `MCPBESolverTimestep`
- **Strategie**: 2 Sampler pro Zeitschritt + Exclusion-Tracking
- **Genauigkeit**: Mittel-Hoch (kleiner Kompromiss für Geschwindigkeit)
- **Performance**: ~200-1000x schneller als Standard (am schnellsten!)
- **Use Case**: Hohe Tropfenraten, breite Partikelgrößenverteilungen

**Algorithmus**:
```python
for each timestep:
    # Baue BEIDE Sampler (einmalig pro Zeitschritt!)
    i_sampler = build(weights=solid_volume)      # groß bevorzugt
    j_sampler = build(weights=1/solid_volume)    # klein bevorzugt
    used_indices = set()                         # Exclusion-Tracking
    
    for each droplet in timestep:
        1. Wähle i aus i_sampler (NICHT in used_indices)
        2. Addiere i zu used_indices
        3. Wenn i zu klein:
           - Wähle j aus j_sampler (NICHT in used_indices)
           - Addiere j zu used_indices
           - Agglomeriere i + j → neues i
           - Aktualisiere used_indices (nur neues i bleibt)
        4. Füge Tropfen zu i hinzu
```

**Kombinierte Optimierungen**:
1. **Dual-Sampler Strategie**: 
   - Große Partikel können Tropfen oft direkt aufnehmen
   - Kleine Partikel dienen als "Material" zum Vergrößern
   - Reduziert Anzahl benötigter Agglomerationen

2. **Timestep-Caching**:
   - Nur 2 Sampler-Bauten pro Zeitschritt (statt pro Tropfen!)
   - Massive Ersparnis bei vielen Tropfen

3. **Exclusion-Tracking**:
   - Kein Partikel wird doppelt im gleichen Zeitschritt verwendet
   - Verhindert redundante Ziehungen

**Vorteile gegenüber reiner Dual-Methode**:
- Dual baut Sampler pro TROPFEN (2 pro Droplet)
- Dual-Timestep baut Sampler pro ZEITSCHRITT (2 pro Timestep)
- Bei 100 Tropfen/Zeitschritt: 50x weniger Sampler-Rebuilds!

---

## Performance-Vergleich (erwartet)

| Methode | Relative Speed | Genauigkeit | Beste Use Case |
|---------|---------------|-------------|----------------|
| Standard | 1x | ★★★★★ | Referenz, kleine Cases |
| Cached | 100-1000x | ★★★★★ | **Die meisten Fälle!** |
| Dual-Timestep | 200-1000x | ★★★★☆ | **Hohe Tropfenraten + breite PSD** |

---

## Verwendung

### Einfache Nutzung (empfohlen)
```python
# Beste Genauigkeit (Cached-Methode)
from mcpbe import MCPBESolverOpti

solver = MCPBESolverOpti(dim=1, t_vec=t_vec, seed=42)
solver.a0 = 5000
solver.x = [27e-5]
solver.process_type = "nucleation"
solver.liquid_flow_rate = 1e-6  # L/s
solver.droplet_diameter = 5e-4  # m
solver.nucleation_enable = True
solver.solve(maxiter=int(1e7))

# Maximale Geschwindigkeit (Dual-Timestep)
from mcpbe import MCPBESolverTimestep

solver = MCPBESolverTimestep(dim=1, t_vec=t_vec, seed=42)
# ... gleiche Parameter ...
solver.solve()
```

---

## Test-Skripte

### Alle Methoden vergleichen
```bash
python Trials/testcase_nucleation_all_methods.py --method all
```

### Nur spezifische Methode testen
```bash
python Trials/testcase_nucleation_all_methods.py --method cached
python Trials/testcase_nucleation_all_methods.py --method timestep
```

### Opti vs Dual-Timestep Direktvergleich
```bash
python Trials/testcase_smplopt_vs_smpldual.py --mode both
python Trials/testcase_smplopt_vs_smpldual.py --mode opti
python Trials/testcase_smplopt_vs_smpldual.py --mode timestep
```

### Umfassende Testsuite
```bash
python Trials/testcase_nucleation_v1.py --mode all
python Trials/testcase_nucleation_v1.py --mode test1 --opti
python Trials/testcase_nucleation_v1.py --mode test1 --timestep
```

---

## Bekannte Probleme & Lösungen

### Import-Fehler
Wenn Import-Fehler auftreten:
```python
ERROR: Could not import MCPBESolverOpti
```

**Lösung**: Stelle sicher dass der Pfad korrekt gesetzt ist:
```python
import sys
sys.path.insert(0, 'C:/Users/ericb/Documents/GitHub/PSD_opt/mcpbe/src')
from mcpbe import MCPBESolverOpti
```

Oder verwende das Test-Skript das den Pfad automatisch setzt.

### Massenbilanz-Fehler
Wenn die Flüssigkeitsbilanz nicht stimmt (>1% Abweichung):

**Mögliche Ursachen**:
1. Zu wenige Partikel für die Tropfenrate
2. Timestep-Methode mit zu großen Zeitschritten
3. Numerische Probleme bei sehr kleinen Volumina

**Lösungen**:
- Mehr initiale Partikel (`a0` erhöhen)
- Kleinere Zeitschritte (`t_write` reduzieren)
- Zur Cached-Methode wechseln (genauer)

### Performance-Probleme
Wenn die Simulation zu langsam ist:

1. **Prüfe die Methode**: Verwende `MCPBESolverTimestep` oder `MCPBESolverOpti`
2. **Reduziere Zeitschritte**: Größeres `t_write`
3. **Partikelanzahl**: Weniger initiale Partikel wenn möglich

---

## Implementierungs-Details

### Sampler-Caching (Cached-Methode)
```python
def _find_and_agglomerate_for_droplet(self, droplet_vol):
    # Check if sampler is valid
    if not self._nucleation_sampler_valid:
        self._build_nucleation_sampler_opti()
    
    sampler = self._nucleation_sampler
    i = sampler.sample(self._rng)
    
    while not self._can_accept_droplet(i, droplet_vol):
        # Need agglomeration
        j = sampler.sample(self._rng)
        i = self._agglomerate_two_particles(i, j)
        # IMPORTANT: Invalidate sampler after agglomeration!
        self._nucleation_sampler_valid = False
        
        # Rebuild if needed for next iteration
        if not self._nucleation_sampler_valid:
            self._build_nucleation_sampler_opti()
            sampler = self._nucleation_sampler
    
    return i
```

### Dual-Timestep mit Exclusion-Tracking
```python
def _build_dual_timestep_samplers(self):
    # Baue BEIDE Sampler einmalig pro Zeitschritt
    weights_i = [solid_volume[k] for k in range(a_tot)]
    weights_j = [1/solid_volume[k] for k in range(a_tot)]
    
    self._nuc_i_sampler = FenwickSampler(weights_i)  # groß bevorzugt
    self._nuc_j_sampler = FenwickSampler(weights_j)  # klein bevorzugt
    self._nuc_used_indices = set()                   # Exclusion-Set

def _sample_with_exclusion(self, sampler, excluded, max_retries=100):
    # Versuche sampling mit exclusion
    for _ in range(max_retries):
        idx = sampler.sample(self._rng)
        if idx not in excluded:
            return idx
    
    # Fallback: sequentiell
    for k in range(a_tot):
        if k not in excluded:
            return k

def _find_and_agglomerate_for_droplet_timestep(self, droplet_vol):
    # Wähle i aus i_sampler (nicht in used_indices)
    i = self._sample_with_exclusion(self._nuc_i_sampler, self._nuc_used_indices)
    self._nuc_used_indices.add(i)
    
    while not self._can_accept_droplet(i, droplet_vol):
        # Wähle j aus j_sampler (nicht in used_indices)
        j = self._sample_with_exclusion(self._nuc_j_sampler, self._nuc_used_indices)
        self._nuc_used_indices.add(j)
        
        # Agglomeriere und aktualisiere Exclusion-Set
        i = self._agglomerate_two_particles_timestep(i, j)
        # Nach swap-pop: nur neues i bleibt im Set
        self._nuc_used_indices = {i}
    
    return i
```

---

## Empfehlungen

1. **Starte mit `MCPBESolverOpti`** - Beste Balance aus Genauigkeit und Performance
2. **Für maximale Geschwindigkeit: `MCPBESolverTimestep`** - Dual+Timestep kombiniert
3. **Verwende `MCPBESolver` nur für Tests** - Zu langsam für Produktion
4. **Immer Massenbilanz prüfen!** - Liquid sollte = expected_droplets * droplet_volume sein

### Entscheidungshilfe:

| Kriterium | Empfohlene Methode |
|-----------|-------------------|
| Standard-Use-Case | `MCPBESolverOpti` (Cached) |
| Maximale Genauigkeit | `MCPBESolverOpti` (Cached) |
| Maximale Geschwindigkeit | `MCPBESolverTimestep` (Dual-Timestep) |
| Hohe Tropfenrate | `MCPBESolverTimestep` |
| Breite Partikelverteilung | `MCPBESolverTimestep` (Dual-Strategie hilft) |
| Wenige Partikel (<100) | `MCPBESolverOpti` (Timestep hat Exclusion-Probleme) |
| Viele Partikel (>1000) | `MCPBESolverTimestep` |

---

## Historie

**v1.0**: Standard-Methode implementiert
**v2.0**: Cached-Methode hinzugefügt (~100-1000x schneller)
**v2.1**: Dual-S Methode separat implementiert (~50-200x schneller)
**v3.0**: **Dual und Timestep fusioniert** zu Dual-Timestep (~200-1000x schneller)
  - Vorteile beider Methoden kombiniert
  - Weniger Code-Duplikation
  - Bessere Wartbarkeit
