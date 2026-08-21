# WMCPBE Quickstart Guide

Schneller Einstieg in den gewichteten DSMC Monte Carlo PBE Solver.

---

## 📌 INHALT

1. [Installation](#installation)
2. [Erste Simulation (5 Minuten)](#erste-simulation-5-minuten)
3. [Grundlegende Konzepte](#grundlegende-konzepte)
4. [Beispiel: Reine Agglomeration](#beispiel-reine-agglomeration)
5. [Beispiel: Mit Nukleation](#beispiel-mit-nukleation)
6. [Helper-Funktionen (Neu!)](#helper-funktionen-neu)
7. [Häufige Probleme](#häufige-probleme)

---

## INSTALLATION

### Voraussetzungen

- Python 3.9+
- NumPy
- pbe_core Package

### Setup

```bash
cd C:\Users\ericb\Documents\GitHub\PSD_opt\mcpbe
pip install -e .
```

---

## ERSTE SIMULATION (5 MINUTEN)

### Minimalbeispiel: Pure Agglomeration

```python
import numpy as np
from wmcpbe import MCPBESolver

# 1. Solver erstellen
solver = MCPBESolver(
    dim=1,                    # 1 Komponente (monodispers)
    t_total=10.0,             # 10 Sekunden simulieren
    t_write=1.0,              # Jede Sekunde Output
    verbose=False,
    load_attr=False,          # Keine Config-Datei laden
    init=True,
    seed=42,                  # Reproduzierbarkeit
    agg_kernel_name='shear_chin1998',
    agg_kernel_params={'corr_beta': 1e-3, 'g': 1000},
    maybe_double_control_volume=False,
    recon_enable=False,
)

# 2. Partikel initialisieren
n_particles = 500
V_flat = np.zeros((2, n_particles), dtype=float)
V_flat[-1, :] = 1e-12  # V_dry pro Partikel [m³]
W_init = np.full(n_particles, 50.0, dtype=float)  # Weight
solver.Vc = 1e-6  # Control Volume [m³]

solver._initialize_particles(
    init_Vc=False,
    V_flat=V_flat,
    W_init=W_init,
)

# 3. Sampler initialisieren
solver._initialize_samplers()

# 4. Moment-Mode aktivieren (O(n) statt O(n²))
solver.agg_propensity_mode = "moment"

# 5. Simulation ausführen
solver.solve()

# 6. Ergebnisse extrahieren
M0 = np.sum(solver.W[:solver.a_tot])  # Anzahl Partikel
M1 = np.sum(solver.W[:solver.a_tot] * solver.V_flat[-1, :solver.a_tot])  # Gesamtvolumen

print(f"Endzustand nach {solver.t_total}s:")
print(f"  Partikelanzahl (M0): {M0:.2f}")
print(f"  Gesamtvolumen (M1):  {M1:.6e} m³")
```

**Ausgabe:**
```
Endzustand nach 10.0s:
  Partikelanzahl (M0): 450.23
  Gesamtvolumen (M1):  5.000000e-10 m³
```

---

## GRUNDLEGENDE KONZEPTE

### Computational Weight (W)

Jedes computational particle repräsentiert **W physikalische Partikel**:

```
W = N_physical / N_computational
```

**Beispiel:**
- 10,000 physikalische Partikel
- 500 computational particles im Simulator
- → Durchschnittliches W = 20

**Wichtig:** W ändert sich während der Simulation durch Agglomeration/Breakage!

### Control Volume (Vc)

DSMC hält die Partikeldichte konstant:

```
Sum(W) / Vc = constant
```

Bei Agglomeration wächst Vc um die Dichte konstant zu halten.

### Propensity Modes

| Modus | Komplexität | Genauigkeit | Verwendung |
|-------|-------------|-------------|------------|
| `pairwise` | O(n²) | Bitgenau | Referenz, kleine Systeme |
| `moment` | O(n) | ~1e-15 Abweichung | **Default**, Produktion |

```python
solver.agg_propensity_mode = "moment"  # Empfohlen!
```

---

## BEISPIEL: REINE AGGLOMERATION

### Shear-Chin Kernel (1998)

```python
from wmcpbe import MCPBESolver
import numpy as np

solver = MCPBESolver(
    dim=1,
    t_total=60.0,
    verbose=False,
    load_attr=False,
    seed=42,
    agg_kernel_name='shear_chin1998',
    agg_kernel_params={
        'corr_beta': 1e-3,  # Korrekturfaktor
        'g': 1000,          # Scherrate [1/s]
    },
    maybe_double_control_volume=False,
    recon_enable=False,
)

# Initialisierung
n_part = 1000
V_flat = np.zeros((2, n_part), dtype=float)
V_flat[-1, :] = 5.236e-13  # 100µm Partikel
W_init = np.full(n_part, 25.0, dtype=float)
solver.Vc = 1e-6

solver._initialize_particles(init_Vc=False, V_flat=V_flat, W_init=W_init)
solver._initialize_samplers()
solver.agg_propensity_mode = "moment"

# Lösen
solver.solve()

# Momente über Zeit
time_points = len(solver.t_vec)
M0_over_time = []
for t_idx in range(time_points):
    a = solver.a_hist[t_idx] if hasattr(solver, 'a_hist') else solver.a_tot
    M0 = np.sum(solver.W_hist[t_idx, :a])
    M0_over_time.append(M0)

print(f"Start: {M0_over_time[0]:.0f} Partikel")
print(f"Ende:  {M0_over_time[-1]:.0f} Partikel")
```

---

## BEISPIEL: MIT NUKLEATION

### Flüssigkeitszugabe während Simulation

```python
from wmcpbe import MCPBESolver
import numpy as np

# Solver mit Breakage
solver = MCPBESolver(
    dim=1,
    t_total=300.0,
    verbose=False,
    load_attr=False,
    seed=42,
    agg_kernel_name='constant',
    agg_kernel_params={'coeff': 1e-9},
    break_kernel_name='power_law',
    break_kernel_params={'p1': 0.01, 'pl_v': 2.0},
    maybe_double_control_volume=False,
    recon_enable=False,
)

# Partikel initialisieren
n_part = 500
V_flat = np.zeros((2, n_part), dtype=float)
V_flat[-1, :] = 1e-12
W_init = np.full(n_part, 100.0, dtype=float)
solver.Vc = 1e-6

solver._initialize_particles(init_Vc=False, V_flat=V_flat, W_init=W_init)
solver._initialize_samplers()
solver.agg_propensity_mode = "moment"

# Nukleation konfigurieren
solver.create_nucleation_handler(
    enabled=True,
    volumetric_flow_rate=1e-9,      # 1 µL/s
    droplet_diameter=100e-6,        # 100 µm Tropfen
    liquid_addition_duration=180.0, # 3 Minuten zugeben
    batch_size=25.0,                # Tropfen pro Event
)

# Porosität初始化 (wichtig für Nukleation!)
solver.porosity[:solver.a_tot] = 0.0  # Start trocken
solver.liquid_volume[:solver.a_tot] = 0.0
solver.saturation[:solver.a_tot] = 0.0

# Lösen
solver.solve()

# Nukleations-Statistiken
stats = solver.nucleation.get_statistics()
print(f"Zugegebene Tropfen: {stats['droplets_added_total']:.2e}")
print(f"Zugegebenes Volumen: {stats['liquid_volume_added_total']:.6e} m³")
print(f"Aktueller wt%: {stats['current_wt_percent']:.2f}%")
```

---

## HELPER-FUNKTIONEN (NEU!)

Seit dem letzten Update stehen Convenience-Funktionen für häufige Aufgaben zur Verfügung.

### 🚀 Templates für Standard-Szenarien

Statt den Solver manuell zu konfigurieren, können Sie vorgefertigte Templates verwenden:

```python
from wmcpbe.helpers import (
    create_dry_agglomeration_solver,
    create_wet_granulation_solver,
    setup_initial_particles,
)

# Trockene Agglomeration (minimal)
solver = create_dry_agglomeration_solver(
    n_particles=500,
    particle_diameter=100e-6,  # 100 µm
    t_total=60.0,
    g_shear=1000,
    corr_beta=1e-3,
)

# Partikel initialisieren (1 Zeile statt 10!)
setup_initial_particles(
    solver,
    n_particles=500,
    particle_diameter=100e-6,
    initial_weight=50.0,
)

# Ready to solve!
solver.solve()
```

### 📊 Validierungs-Helpers

```python
from wmcpbe.helpers import compute_moments, validate_mass_conservation

# Nach Simulation ausführen
moments = compute_moments(solver)
print(f"M0={moments['M0']:.0f}, d50={moments['d50']*1e6:.1f} µm")

# Massenerhaltung prüfen
result = validate_mass_conservation(solver)
print(result['message'])  # "✅ Mass conservation OK ..."
```

### 🎯 Verfügbare Templates

| Template | Funktion | Use Case |
|----------|----------|----------|
| **Dry Agglomeration** | `create_dry_agglomeration_solver()` | Erste Tests, Shear-only |
| **Wet Granulation** | `create_wet_granulation_solver()` | Vollständige Feuchtgranulation |
| **Breakage Test** | `create_breakage_test_solver()` | Breakage-Kernel Validierung |

### 📝 Eigene Helper erstellen

```python
# helpers.py im eigenen Projekt
from wmcpbe.helpers import create_dry_agglomeration_solver

def create_my_custom_solver():
    solver = create_dry_agglomeration_solver(
        n_particles=1000,
        g_shear=2000,  # Höhere Scherung
    )
    # Custom modifications
    solver.recon_enable = True
    solver.recon_N_max = 4000
    return solver
```

---

## HÄUFIGE PROBLEME

### ❌ "IndexError: index X is out of bounds"

**Ursache:** Sampler nicht initialisiert vor `solve()`.

**Lösung:**
```python
solver._initialize_particles(...)
solver._initialize_samplers()  # ← Nicht vergessen!
solver.solve()
```

**Neu (seit v2.0):** Der Solver erkennt diesen Fehler jetzt automatisch und gibt eine hilfreiche Meldung aus:

```
RuntimeError: Breakage sampler not initialized but process_type='breakage'.
You forgot to call solver._initialize_samplers() after _initialize_particles().

Correct initialization sequence:
  solver._initialize_particles(...)
  solver._initialize_samplers()  # ← Don't forget this!
  solver.solve()
```

---

### ❌ "Config file not found at: ..."

**Ursache:** `load_attr=True` (Default) versucht Config-Datei zu laden.

**Lösung:**
```python
solver = MCPBESolver(
    load_attr=False,  # ← Explizit alle Parameter setzen
    agg_kernel_name='shear_chin1998',
    agg_kernel_params={'corr_beta': 1e-3, 'g': 1000},
    ...
)
```

---

### ❌ "maybe_double_control_volume unexpected keyword"

**Ursache:** Veraltete Test-Dateien oder alte API.

**Lösung:** Parameter zu `__init__()` hinzufügen (seit Update implementiert):
```python
solver = MCPBESolver(
    maybe_double_control_volume=False,
    recon_enable=False,
    ...
)
```

---

### ❌ Nukleation gibt keine Ausgabe

**Ursache:** Logging-Level zu hoch oder debug=False.

**Lösung:**
```python
# Option 1: Debug-Ausgaben aktivieren
solver.create_nucleation_handler(
    enabled=True,
    debug=True,  # ← Detaillierte Logs
    ...
)

# Option 2: Logging-Level anpassen
solver = MCPBESolver(
    log_level="INFO",  # Statt "WARNING"
    ...
)
```

---

### ❌ Simulation extrem langsam (>1h)

**Ursache:** 
- Zu viele Partikel (>5000)
- `pairwise` Mode statt `moment`
- Breakage mit großen CDF-Tabellen

**Lösung:**
```python
# 1. Moment Mode aktivieren
solver.agg_propensity_mode = "moment"  # Bis zu 275× schneller!

# 2. Reconstruction aktivieren bei >4000 Partikeln
solver.recon_enable = True
solver.recon_N_max = 4000

# 3. Breakage-Parameter prüfen
break_kernel_params={'p1': 0.01}  # Nicht zu klein!
```

---

## 📚 WEITERFÜHRENDE DOKUMENTATION

| Dokument | Beschreibung |
|----------|--------------|
| [Parameter_Liste.md](Parameter_Liste.md) | Vollständige Parameter-Referenz |
| [PERFORMANCE.md](PERFORMANCE.md) | Performance-Optimierung, Benchmarking |
| [MOMENT_MODE.md](MOMENT_MODE.md) | Mathematische Herleitung O(n)-Modus |
| [Nucleation_Validation_Report.md](Nucleation_Validation_Report.md) | Nukleation Validierung |
| [CONSERVATION.md](CONSERVATION.md) | Massenerhaltung, Validierung |

---

## 🧪 TESTS AUSFÜHREN

```bash
# Ensemble-Test: pairwise vs moment
python -m wmcpbe.Trials.test_ensemble_propensity_modes --seeds 8

# PowerLaw-Rumpf Breakage Test
python -m wmcpbe.Trials.test_powerlaw_rumpf_full
```

---

**Letzte Aktualisierung:** 2024-01-XX  
**Version:** WMCPBE 2.0
