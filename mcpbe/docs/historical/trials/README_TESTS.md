# Test-Skripte im Trials-Ordner

## Kernel Framework Tests (neu erstellt)

| Skript | Zweck | Status |
|--------|-------|--------|
| `test_kernel_framework.py` | Isolierte Kernel-Funktionalität | ✅ 5/5 bestanden |
| `verify_kernel_integration.py` | Kernel im Solver-Kontext (Mock) | ✅ 6/6 bestanden |
| `verify_no_fallback_abuse.py` | Verifiziert echte Features | ✅ 5/5 bestanden |
| `test_kernel_integration.py` | Vollständige Solver-Integration | ✅ 5/5 bestanden |

## Bestehende Validation-Skripte

Hier sollten bestehende Validation-Skripte liegen wie:
- `simple_validation.py`
- `validate_aggregation.py`
- `validate_breakage.py`
- etc.

**Falls nicht vorhanden:** Bitte Pfad zu bestehenden Validation-Skripten angeben!

---

## Nächster Schritt: Option A - Validierung

### Vorgehen

1. **Bestehendes Case finden** – Ein funktionierendes Simulation-Case mit bekanntem Ergebnis
2. **Mit alter Architektur laufen** – Referenzergebnis speichern
3. **Mit neuer Architektur laufen** – Gleiche Parameter, aber über Kernel-Framework
4. **Vergleichen** – Ergebnisse müssen identisch sein (bis auf numerische Toleranz)

### Erwartete Übereinstimmung

| Größe | Toleranz | Grund |
|-------|----------|-------|
| Partikelgrößenverteilung | < 0.1% | Deterministische Physik |
| Momente (Q0, Q3) | < 0.1% | Gleiche Formeln |
| Porosität | < 0.001 | Volume mixing = alte Logik |
| Rechenzeit | ±10% | Kernel-Overhead minimal |

---

## Beispiel: Validierungsskript erstellen

```python
"""
Validation: Old vs New Kernel Architecture.

Compares results from legacy COLEVAL-based simulation
with new kernel framework using equivalent settings.
"""

import numpy as np
from wmcpbe.mcpbe import MCPBESolver

def run_legacy_case():
    """Run case with old COLEVAL/BREAKRVAL approach."""
    solver = MCPBESolver(
        dim=1,
        t_total=100,
        t_write=10,
        load_attr=True,  # Load config with COLEVAL=1, BREAKRVAL=1
    )
    # ... configure and solve
    solver.solve()
    return solver.X_save, solver.W_save

def run_new_case():
    """Run equivalent case with new kernel framework."""
    solver = MCPBESolver(
        dim=1,
        t_total=100,
        t_write=10,
        load_attr=False,  # Skip config
        agg_kernel_name='shear_chin1998',  # Equivalent to COLEVAL=1
        agg_kernel_params={'corr_beta': 1e-3, 'g': 1000},
        break_kernel_name='power_law',     # Equivalent to BREAKRVAL=1
        break_kernel_params={'p1': 3e-2, 'p2': 1.0},
    )
    # ... configure same physics as legacy case
    solver.solve()
    return solver.X_save, solver.W_save

def compare_results(legacy, new):
    """Compare results from both approaches."""
    X_legacy, W_legacy = legacy
    X_new, W_new = new
    
    # Compare moments
    Q0_legacy = np.sum(W_legacy)
    Q0_new = np.sum(W_new)
    
    rel_error_Q0 = abs(Q0_new - Q0_legacy) / Q0_legacy
    
    print(f"Q0 relative error: {rel_error_Q0*100:.4f}%")
    
    assert rel_error_Q0 < 0.001, "Q0 mismatch > 0.1%"
    print("✅ Validation passed!")

if __name__ == "__main__":
    legacy = run_legacy_case()
    new = run_new_case()
    compare_results(legacy, new)
```

---

## Bitte um Input

**Frage:** Wo liegen bestehende Validation-Skripte oder Cases?

Optionen:
1. In `mcpbe/src/wmcpbe/Trials/`?
2. In separatem `validation/` Ordner?
3. Als Jupyter Notebooks?
4. Noch nicht erstellt?

→ Bitte Pfad angeben oder Bestätigung geben, dass wir ein neues Validation-Skript erstellen sollen!
