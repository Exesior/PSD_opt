# Update: Nucleation Debug für Masseerhaltung

## Problem

Die bisherigen Debug-Prints zeigten perfekte Masseerhaltung in jedem einzelnen Prozess (ΔV_solid = 0), aber am Ende der Simulation zeigt sich ein **+21.43% Massefehler** im 20µm-Case.

**Hypothese:** Die Nucleation fügt nicht nur Flüssigkeit hinzu, sondern verändert fälschlicherweise auch V_solid/V_dry durch die manuelle Agglomeration während der V_dry-Sammlung.

## Neue Debug-Prints

### 1. `_distribute_liquid_volume()` - Vor/Nach Verteilung

**Vor der Verteilung:**
```python
[DEBUG NUC] t=0.0000s
  Distributing droplets: volume=3.973296e-12 m^3
  BEFORE: V_solid_total=3.292724e-09, V_dry_total=4.115655e-09, V_liq_total=0.000000e+00
  n_phys=2.000e+03, a_tot=2000
  *** NUR LIQUID SOLLTE SICH AENDERN - V_SOLID MUSS KONSTANT BLEIBEN! ***
```

**Nach der Verteilung:**
```python
  AFTER:  V_solid=3.292724e-09, V_dry=4.115655e-09, V_liq=3.973296e-12
  ΔV_solid=0.000000e+00 (SOLLTE = 0 sein!)
  ΔV_dry=0.000000e+00 (SOLLTE = 0 sein!)  ← NEU!
  ΔV_liq=3.973296e-12 (SOLLTE ≈ added volume)
  n_phys=2.001e+03, a_tot=2001
  Statistics: droplets_added=5.57e+04, volume_added=2.333804e-10
  
  *** FEHLER: V_SOLID HAT SICH GEAENDERT UM 1.23e-15! ***  ← Warnung wenn ≠ 0
```

### 2. `_manual_agglomerate_particles()` - Vor/Nach jeder Agg

**Vor der Agglomeration:**
```python
[DEBUG NUC-AGG] Manual agglomeration for V_dry collection
  Particles: i=2000, j=967
  W[i]=6586.00, W[j]=380.00
  V_dry[i]=1.028976e-14, V_dry[j]=2.057953e-14
  V_solid[i]=8.231811e-15, V_solid[j]=4.115905e-15
  V_solid_SUM=1.234772e-14 (MUSST = Child V_solid sein)
  liq[i]=8.377580e-15, liq[j]=0.000000e+00, SUM=8.377580e-15
  GLOBAL BEFORE: V_solid_total=3.292724e-09, V_dry_total=4.115655e-09
```

**Nach der Agglomeration:**
```python
  Child: idx=967
  W[child]=380.00
  V_dry[child]=3.780110e-14
  V_solid[child]=1.234772e-14
  ΔV_solid=-3.155444e-30 (SOLLTE ≈ 0 sein)
  liq[child]=8.377580e-15
  Δliq=0.000000e+00
  a_tot changed: 2069 (parents removed?)
  
  GLOBAL AFTER: V_solid_total=3.292724e-09, V_dry_total=4.115655e-09
  GLOBAL ΔV_solid=0.000000e+00 (MUSST = 0 sein!)  ← NEU!
  GLOBAL ΔV_dry=0.000000e+00 (MUSST = 0 sein!)    ← NEU!
```

## Erwartete Ergebnisse (korrekte Masseerhaltung)

| Prozess | ΔV_solid | ΔV_dry | ΔV_liq |
|---------|----------|--------|--------|
| **Nucleation (gesamt)** | 0 | 0 | = added_volume |
| **Manual Agg (einzelne)** | ≈ 0 (±1e-30) | ≈ 0 | 0 |

## Fehlerindikatoren

| Symptom | Mögliche Ursache |
|---------|------------------|
| ΔV_solid > 1e-20 bei Nucleation | `_create_or_update_nucleated_particle()` setzt V_flat falsch |
| ΔV_dry > 1e-20 bei Nucleation | Porosity-Growth-Kernel berechnet V_dry_new falsch |
| ΔV_solid > 1e-20 bei Manual Agg | `_manual_agglomerate_particles()` verdoppelt V_solid |
| GLOBAL ΔV_solid ≠ 0 nach Agg | swap-remove Logik entfernt falsches Partikel |
| a_tot ändert sich unerwartet | Eltern werden entfernt, aber Weight-Tracking fehlt |

## Aktivierung

```python
solver.mcpbe_debug_mass = True
solver.mcpbe_debug_nuc = True  # Aktiviert alle Nucleation-Debugs
```

## Auswertung

1. **Suche nach WARNUNGEN** im Log: `*** FEHLER: V_SOLID HAT SICH GEAENDERT ***`
2. **Vergleiche GLOBAL BEFORE/AFTER** Werte in Manual Agg
3. **Prüfe ΔV_dry** - sollte immer 0 sein (außer bei Porosity Growth!)
4. **Achte auf a_tot Änderungen** - unerwartete Changes deuten auf swap-remove Bugs hin

## Dateien geändert

- `mcpbe_nucleation.py`: 
  - `_distribute_liquid_volume()`: V_dry_total + globale Warnungen
  - `_manual_agglomerate_particles()`: Globale Massebilanz vor/nach Agg
