# Kernel Framework – Implementierungskontext

**Datum:** 10. Juli 2026  
**Status:** ✅ Integration abgeschlossen, Validierung ausstehend  
**Autor:** WMCPBE Development Team

---

## 🎯 Ziel der Entwicklung

Das bestehende WMCPBE-System verwendet **hard-coded Kernel-Auswahl** über Integer-Flags (`COLEVAL`, `BREAKRVAL`). Neue physikalische Modelle erforderten Änderungen an Core-Dateien.

**Lösung:** Modulares Kernel-Framework mit:
- Austauschbaren Kerneln zur Laufzeit
- Rückwärtskompatibilität zu bestehenden Configs
- Erweiterbarkeit ohne Core-Änderungen

---

## 📁 Erstellte / Geänderte Dateien

### Neu erstellt (Kernel-Framework)

```
mcpbe/src/wmcpbe/kernels/
├── __init__.py                      # Package-Init mit Base-Klassen exports
├── base.py                          # Abstrakte Basisklassen (5 Klassen)
├── README.md                        # Ausführliche Dokumentation
│
├── aggregation/                     # 5 Kernel
│   ├── __init__.py                  # Factory + Registry
│   ├── shear_chin1998.py            # JIT-basiert (COLEVAL=1)
│   ├── brownian_tsouris1995.py      # JIT-basiert (COLEVAL=2)
│   ├── constant.py                  # COLEVAL=3
│   ├── sum_kernel.py                # COLEVAL=4
│   └── liquid_bridge.py             # NEU: Mit Flüssigkeitsbrücken
│
├── breakage/                        # 2 Kernel
│   ├── __init__.py
│   ├── power_law.py                 # Bestehende Logik
│   └── stress_based.py              # NEU: Weibull-Modell
│
├── porosity_growth/                 # 2 Kernel
│   ├── __init__.py
│   ├── volume_mixing.py             # Bestehende Logik
│   └── incomplete_mixing.py         # NEU: Eingeschlossene Poren
│
├── compression/                     # 2 Kernel
│   ├── __init__.py
│   ├── exponential_decay.py         # Bestehende Logik
│   └── stress_compaction.py         # NEU: Stress-abhängig
│
└── liquid_distribution/             # 3 Kernel
    ├── __init__.py
    ├── uniform_weighted.py          # Bestehende Logik
    ├── surface_weighted.py          # NEU: Oberflächen-gewichtet
    └── saturation_preferential.py   # NEU: Sättigungs-präferentiell

mcpbe/src/wmcpbe/
├── kernel_integration.py            # NEU: KernelManager-Klasse
│
└── Trials/
    ├── __init__.py
    ├── test_kernel_framework.py     # Test-Suite 1: Isolierte Kernel
    ├── verify_kernel_integration.py # Test-Suite 2: Solver-Kontext
    ├── verify_no_fallback_abuse.py  # Test-Suite 3: Echte Features
    ├── test_kernel_integration.py   # Test-Suite 4: Vollständige Integration
    ├── validate_old_vs_new.py       # Test-Suite 5: Regressionstests
    ├── README_TESTS.md              # Test-Übersicht
    └── VALIDATION_REPORT.md         # Validierungsbericht
```

### Geändert (Integration)

| Datei | Geänderte Methoden | Zeilen | Zweck |
|-------|-------------------|--------|-------|
| `mcpbe_base.py` | `__init__()`, `_initialize_kernels()` (neu) | +80 | Kernel-Initialisierung |
| `mcpbe_agg.py` | `_beta()`, `_rebuild_all_propensities()` | +40 | Delegation an Aggregation-Kernel |
| `mcpbe_break.py` | `_break_rate_single()` | +30 | Delegation an Breakage-Kernel |
| `mcpbe_nucleation.py` | `_select_particle_uniform_physical()` | +15 | Delegation an Liquid-Distribution-Kernel |

**Gesamt:** ~520 Zeilen neu, ~50 Zeilen modifiziert

---

## 🔧 Was wurde aus Nucleation ausgelagert?

**Antwort: NICHTS!**

Es wurde **keine Logik ausgelagert**. Die gesamte Nucleation-Logik bleibt in `mcpbe_nucleation.py`.

**Nur hinzugefügt (15 Zeilen):** Optionale Delegation-Schicht für Partikel-Selektion:

```python
def _select_particle_uniform_physical(self) -> int:
    """Select particle with optional kernel delegation."""
    solver = self.solver
    
    # NEU: Kernel Framework Delegation (optional)
    if hasattr(solver, 'kernel_manager') and solver.kernel_manager is not None:
        liq_kernel = solver.kernel_manager.liquid_dist_kernel
        if liq_kernel is not None:
            return solver.kernel_manager.select_liquid_target(
                solver=solver,
                v_droplet=self.config.droplet_volume,
                current_time=getattr(solver, '_elapsed', 0.0)
            )
    
    # ALT: Fallback zu existierender Logik (unverändert!)
    a_tot = solver.a_tot
    if a_tot < 1:
        return -1
    
    if self._weight_sampler is None:
        self._ensure_samplers()
    
    if self._weight_sampler is None or self._weight_sampler.total() <= 0:
        return int(self._rng.integers(0, a_tot))
    
    return int(self._weight_sampler.sample(self._rng))
```

**Bedeutung:**
- ✅ Bestehende Simulationen laufen **unverändert** (uniform_weighted = alte Logik)
- ✅ Neue Features optional verfügbar (`surface_weighted`, `saturation_preferential`)
- ✅ Kein Performance-Overhead ohne Kernel-Konfiguration

---

## ✅ Test-Ergebnisse (Bestanden)

### Test-Suite 1: Kernel-Funktionalität (Isoliert)
**Skript:** `test_kernel_framework.py`  
**Status:** ✅ 5/5 bestanden

| Test | Ergebnis |
|------|----------|
| Module Imports | ✓ PASS |
| Factory Functions | ✓ PASS |
| Kernel Computations | ✓ PASS |
| Parameter Validation | ✓ PASS |
| List Kernels | ✓ PASS |

### Test-Suite 2: Kernel im Solver-Kontext (Mock)
**Skript:** `verify_kernel_integration.py`  
**Status:** ✅ 6/6 bestanden

| Test | Ergebnis |
|------|----------|
| Aggregation + Solver | ✓ PASS |
| Breakage + Solver | ✓ PASS |
| Porosity + Energy | ✓ PASS |
| Compression + Stress | ✓ PASS |
| Liquid Dist + Selection | ✓ PASS |
| State Independence | ✓ PASS |

### Test-Suite 3: Echte Features (keine Fallbacks)
**Skript:** `verify_no_fallback_abuse.py`  
**Status:** ✅ 5/5 bestanden

| Test | Ergebnis |
|------|----------|
| Liquid Bridge Enhancement | ✓ PASS (β=2.907e-17 vs Fallback β=2.700e-17) |
| Incomplete Mixing Pores | ✓ PASS (ΔV=1.5e-19 m³, Δε=0.025) |
| Stress Compaction Size | ✓ PASS |
| Surface Weighted Bias | ✓ PASS (98.9% große Partikel vs 50% uniform) |
| Saturation Preference | ✓ PASS (74.3% trockene vs 25.7% nasse) |

### Test-Suite 4: Vollständige Integration
**Skript:** `test_kernel_integration.py`  
**Status:** ✅ 5/5 bestanden

| Test | Ergebnis |
|------|----------|
| Solver with Kernels | ✓ PASS |
| Backward Compatibility | ✓ PASS (COLEVAL=1 → shear_chin1998) |
| Beta Delegation | ✓ PASS (β=1.000e-15 konstant) |
| Break Rate Delegation | ✓ PASS (S=2.418e-05 1/s) |
| New Kernel Features | ✓ PASS (liquid_bridge, incomplete_mixing) |

### Test-Suite 5: Regressionstests (Alt vs Neu)
**Skript:** `validate_old_vs_new.py`  
**Status:** ⏳ Läuft (zweistufiger Prozess)

---

## 📋 Umfassender Validierungsplan (Phase 1 & 2)

### Test-Struktur

Die finale Validierung läuft in **zwei Phasen**:

#### Phase 1: Basic Sanity Checks (schnell, ~10-15 Minuten)

Drei grundlegende Tests stellen sicher, dass das Framework korrekt integriert ist:

| # | Test | Legacy Config | New Kernel | Erwartung |
|---|------|---------------|------------|-----------|
| 1.1 | Agglomeration Only | COLEVAL=1, BREAKRVAL=1 | shear_chin1998, none | ✅ Q0, Q3, d_mean < 0.1% |
| 1.2 | Breakage Only | COLEVAL=1, BREAKRVAL=1 | none, power_law(BREAKRVAL=1) | ✅ Q0, Q3, d_mean < 0.1% |
| 1.3 | Combined Mix | COLEVAL=1, BREAKRVAL=1 | shear_chin1998 + power_law | ✅ Q0, Q3, d_mean < 0.1% |

**Parameter für alle Tests:**
- `dim=1`, `t_total=5.0s`, `seed=42`
- `CORR_BETA=1e-2`, `G=1000`
- `pl_P1=3e-4`, `pl_P2=1.0`

---

#### Phase 2: Comprehensive Kernel Validation (~30-45 Minuten)

Vollständige Abdeckung aller Legacy-Kernel-Mappings:

##### Aggregation Kernels (COLEVAL 1-4)

| # | COLEVAL | Kernel Name | Formel | Erwartung |
|---|---------|-------------|--------|-----------|
| 2.1 | 1 | `shear_chin1998` | β = CORR_BETA × G × (r1+r2)³ | ✅ < 0.1% Fehler |
| 2.2 | 2 | `brownian_tsouris1995` | β = CORR_BETA × 2kT/(3μ) × (r1+r2)²/(r1×r2) | ✅ < 0.1% Fehler |
| 2.3 | 3 | `constant` | β = CORR_BETA (konstant) | ✅ < 0.1% Fehler |
| 2.4 | 4 | `sum` | β = CORR_BETA × 4π/3 × (r1³+r2³) | ✅ < 0.1% Fehler |

**Test-Parameter:**
- `process_type='agglomeration'`, `t_total=5.0s`, `seed=42`
- `CORR_BETA=1e-2`, `G=1000`

##### Breakage Kernels (BREAKRVAL 1-5)

| # | BREAKRVAL | Rate Formula | Fragment Distribution | Erwartung |
|---|-----------|--------------|----------------------|-----------|
| 2.5 | 1 | S = pl_P1 (konstant) | BREAKFVAL=1: 4 Fragmente | ✅ < 0.1% Fehler |
| 2.6 | 2 | S = pl_P1 × V (linear in V) | BREAKFVAL=2: 2 Fragmente | ✅ < 0.1% Fehler |
| 2.7 | 3 | S = pl_P1 × G × V^pl_P2 | BREAKFVAL=3: pl_v Fragmente | ✅ < 0.1% Fehler |
| 2.8 | 4 | S = pl_P1 × G × V^pl_P2 | BREAKFVAL=4: (pl_v+1)/pl_v | ✅ < 0.1% Fehler |
| 2.9 | 5 | S = pl_P1 × G × V^pl_P2 | BREAKFVAL=5: 2D-Fragmentation | ✅ < 0.1% Fehler |

**Test-Parameter:**
- `process_type='breakage'`, `t_total=5.0s`, `seed=42`
- `pl_P1=3e-4`, `pl_P2=1.0`, `G=1000`
- `pl_v=2.0`, `pl_q=0.5`

---

### Erfolgskriterien

Ein Test gilt als **bestanden**, wenn ALLE folgenden Bedingungen erfüllt sind:

| Kriterium | Toleranz | Begründung |
|-----------|----------|------------|
| **Q0 (Anzahldichte)** | < 0.1% rel. Fehler | Monte-Carlo-Statistik |
| **Q3 (Volumendichte)** | < 0.1% rel. Fehler | Massenerhaltung kritisch |
| **d_mean (Mittlerer Durchmesser)** | < 0.1% rel. Fehler | PSD-Form muss stimmen |
| **Partikelanzahl am Ende** | Identisch | Direkter Vergleich |
| **Simulationszeit** | ±20% akzeptabel | Algorithmische Unterschiede |

**Achtung:** Größere Abweichungen deuten auf hin:
- > 1% Fehler: **Kritischer Bug** (sofort untersuchen!)
- 0.1-1% Fehler: **Numerische Toleranz** (RNG, Floating-Point)
- < 0.1% Fehler: **Perfekt** ✅

---

### Bekannte Bugs (bereits gefixt)

Folgende Probleme wurden während der Entwicklung identifiziert und behoben:

| Bug | Symptom | Root Cause | Fix | Status |
|-----|---------|------------|-----|--------|
| Self-Agglomeration fehlt | Nur 1 Event statt ~688 | `i==j` wurde übersprungen in `_rebuild_all_propensities()` | `(W[i]-1)*beta_ii` hinzufügen | ✅ Gefixt |
| Test-Parameter falsch | corr_beta=1e-12 statt 1.0 | Copy-Paste-Fehler in `validate_old_vs_new.py` | Auf 1.0 korrigiert | ✅ Gefixt |
| Step 7 Akzeptanztest | 0% Akzeptanz bei corr_beta=1 | `beta_max` viel zu groß geschätzt | Komplette Entfernung (nicht im Original) | ✅ Gefixt |

---

### Expected Results (Erwartete Ergebnisse)

Nach erfolgreicher Validierung sollte das Ergebnis so aussehen:

```
======================================================================
 OVERALL SUMMARY
======================================================================

  [PASS] Agglomeration Only (COLEVAL=1)
  [PASS] Breakage Only (BREAKRVAL=1)
  [PASS] Combined Mix (COLEVAL=1, BREAKRVAL=1)
  [PASS] All Agg Kernels (COLEVAL 1-4)
  [PASS] All Break Kernels (BREAKRVAL 1-5)

  Total: 5/5 validations passed

  ALL VALIDATIONS PASSED!
  The new kernel framework is verified to produce equivalent results.
  Safe to use for production simulations.
```

**Bei Fehlschlag:**
1. Fehlermeldung dokumentieren
2. Diskrepanz analysieren (welcher Kernel? welche Formel?)
3. Fix implementieren
4. Test wiederholen

---

## 🚀 Nächste Schritte

### Sofort (Option A – Validierung)

```bash
cd C:\Users\ericb\Documents\GitHub\PSD_opt\mcpbe\src\wmcpbe
python Trials/validate_old_vs_new.py
```

**Erwartung:** Alle Tests bestehen mit < 0.1% Abweichung

### Nach erfolgreicher Validierung

1. ✅ Kernel Framework für Produktion freigeben
2. ✅ Dokumentation finalisieren
3. ⏳ Bestehende Cases migrieren (optional)
4. ⏳ Legacy-Code entfernen (erst nach vollständiger Validierung!)

---

## 💡 Verwendung im Code

### Beispiel 1: Explizite Kernel-Namen (neu)

```python
from wmcpbe.mcpbe import MCPBESolver

solver = MCPBESolver(
    dim=1,
    t_total=600,
    agg_kernel_name='liquid_bridge',
    agg_kernel_params={
        'corr_beta': 1e-3,
        'g': 1000,
        'optimal_saturation': 0.5,
    },
    porosity_growth_kernel_name='incomplete_mixing',
    porosity_growth_kernel_params={
        'trapped_pore_fraction': 0.15,
    },
)
solver.solve()
```

### Beispiel 2: Legacy COLEVAL/BREAKRVAL (rückwärtskompatibel)

```python
from wmcpbe.mcpbe import MCPBESolver

solver = MCPBESolver(
    dim=1,
    t_total=600,
    load_attr=True,  # Lädt Config mit COLEVAL=1, BREAKRVAL=1
)
# Funktioniert exakt wie vorher!
solver.solve()
```

### Beispiel 3: Kernel direkt verwenden

```python
from wmcpbe.kernels.aggregation import get_aggregation_kernel

kernel = get_aggregation_kernel('liquid_bridge', 
                                 corr_beta=1e-3, 
                                 g=1000,
                                 optimal_saturation=0.5)

beta = kernel.compute_beta(r1=1e-6, r2=2e-6)
print(f"Kollisionsfrequenz: {beta:.3e} m³/s")
```

---

## 📊 Neue physikalische Modelle

| Kernel | Physik | Anwendung | Status |
|--------|--------|-----------|--------|
| `liquid_bridge` | Kapillarkräfte zwischen nassen Partikeln | Feuchtgranulation | ✅ Verfügbar |
| `incomplete_mixing` | Eingeschlossene Poren bei unvollständiger Koaleszenz | Poröse Granulate | ✅ Verfügbar |
| `stress_compaction` | Größenabhängige Kompression durch lokalen Stress | Hochscher-Mischer | ✅ Verfügbar |
| `surface_weighted` | Größere Partikel treffen häufiger Tropfen | Sprühgranulation | ✅ Verfügbar |
| `saturation_preferential` | Trockene Partikel benetzen bevorzugt | Ungleichmäßige Benetzung | ✅ Verfügbar |

---

## 🎓 Eigene Kernel erstellen

### Schritt-für-Schritt

1. **Datei erstellen** im richtigen Ordner (z.B. `kernels/aggregation/my_kernel.py`)

```python
from ..base import AggregationKernel

class MyKernel(AggregationKernel):
    @property
    def name(self) -> str:
        return 'my_kernel'
    
    def get_default_params(self) -> dict:
        return {'param1': 1.0}
    
    def compute_beta(self, r1, r2, ...):
        # Eigene Implementierung
        return beta_value
```

2. **In Factory registrieren** (`kernels/aggregation/__init__.py`)

```python
from .my_kernel import MyKernel

AGG_KERNELS['my_kernel'] = MyKernel
```

3. **Verwenden**

```python
solver = MCPBESolver(agg_kernel_name='my_kernel')
```

---

## 📞 Kontakt & Support

Bei Fragen zum Kernel-Framework:

1. **Dokumentation lesen:** `kernels/README.md`
2. **Tests studieren:** `Trials/test_kernel_framework.py`
3. **Beispiele ansehen:** Kernel-Implementierungen in `kernels/*/`

---

## 🔗 Wichtige Dateien im Überblick

| Datei | Zweck |
|-------|-------|
| `kernels/README.md` | Hauptdokumentation des Frameworks |
| `kernel_integration.py` | KernelManager-Klasse (Integration) |
| `Trials/VALIDATION_REPORT.md` | Zusammenfassung aller Änderungen |
| `Trials/README_TESTS.md` | Übersicht aller Test-Skripte |
| `Trials/IMPLEMENTATION_CONTEXT.md` | Diese Datei – Kontext für spätere Sitzungen |

---

## 🔄 Kontext wiederherstellen

Um diesen Kontext in einer späteren Sitzung wiederherzustellen:

```python
# Diese Datei lesen
with open('Trials/IMPLEMENTATION_CONTEXT.md', 'r') as f:
    context = f.read()
    print(context)
```

Oder einfach den Assistenten fragen: "Zeige mir den Kernel Framework Kontext"

---

**Ende des Dokuments**
