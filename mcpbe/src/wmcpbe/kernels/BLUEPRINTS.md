# Kernel Blueprints: Implementierungs-Leitfaden

## 📋 Übersicht

Dieses Verzeichnis enthält **Blueprint-Dateien** für jede Kernel-Kategorie. Blueprints sind vollständige, kommentierte Vorlagen, die als Ausgangspunkt für die Implementierung eigener Kernel dienen.

## 🎯 Zweck von Blueprints

Blueprints bieten:

1. **Struktur**: Klare Definition aller erforderlichen Methoden
2. **Dokumentation**: Ausführliche Kommentare zu jedem Parameter und Rückgabewert
3. **Physik-Hintergrund**: Erklärung der zugrundeliegenden physikalischen Modelle
4. **Beispiele**: Mehrere Implementierungsbeispiele pro Methode
5. **Tests**: Integrierte Test-Sektionen zum Überprüfen der Implementierung
6. **Best Practices**: Empfehlungen für numerische Stabilität und Performance

## 📁 Verfügbare Blueprints

| Kernel-Typ | Datei | Beschreibung |
|------------|-------|--------------|
| **Aggregation** | `aggregation/blueprint.py` | Kollisionsfrequenz β(i,j) [m³/s] |
| **Breakage** | `breakage/blueprint.py` | Bruchrate S(V) [1/s] |
| **Porosity Growth** | `porosity_growth/blueprint.py` | Porositätsentwicklung bei Ereignissen |
| **Compression** | `compression/blueprint.py` | Porositätsreduktion über Zeit [1/s] |
| **Liquid Distribution** | `liquid_distribution/blueprint.py` | Partikel-Selektion für Flüssigkeitszugabe |

## 🚀 Schnellstart: Eigenen Kernel implementieren

### Schritt 1: Blueprint kopieren

```bash
# Beispiel: Neuer Agglomerations-Kernel
cd mcpbe/src/wmcpbe/kernels/aggregation
cp blueprint.py my_custom_kernel.py
```

### Schritt 2: Klasse umbenennen

```python
# In my_custom_kernel.py

# ALT (aus Blueprint):
class ExampleAggregationKernel(AggregationKernel):

# NEU (dein Kernel):
class MyCustomKernel(AggregationKernel):
```

### Schritt 3: Kernel-Name anpassen

```python
@property
def name(self) -> str:
    return 'my_custom_kernel'  # Muss mit Registry-Key übereinstimmen!
```

### Schritt 4: Parameter definieren

```python
def get_default_params(self) -> dict:
    return {
        'param1': 0.1,              # Dein Parameter 1
        'param2': 1000.0,           # Dein Parameter 2
        # ... weitere Parameter
    }
```

### Schritt 5: Physik implementieren

```python
def compute_beta(self, r1, r2, particle1_idx, particle2_idx, solver):
    # DEINE PHYSIK HIER
    
    # Beispiel: Shear-basiert mit Korrektur
    beta_base = (4.0 / 3.0) * self.g * (r1 + r2)**3
    beta = self.corr_beta * beta_base
    
    # Zusätzliche Effekte...
    # if solver is not None:
    #     ...
    
    return beta
```

### Schritt 6: In Registry eintragen

```python
# In aggregation/__init__.py

from .my_custom_kernel import MyCustomKernel

AGGREGATION_KERNELS['my_custom_kernel'] = MyCustomKernel
```

### Schritt 7: Testen

**Option A: Direkt ausführen (mit Mock-Basisklasse)**

Blueprints enthalten einen Fallback für direkte Ausführung:

```bash
python blueprint.py
```

Die Blueprints verwenden diesen Import-Mechanismus:

```python
try:
    from ..base import BreakageKernel  # Package-Kontext
except ImportError:
    class BreakageKernel:  # Mock für direkte Ausführung
        pass
```

⚠️ **Einschränkung**: Bei direkter Ausführung werden nur Syntax und Logik getestet,
nicht die Integration mit dem Solver.

**Option B: Im Package-Kontext ausführen (empfohlen)**

```bash
# Vom Projekt-Root aus mit -m Flagge
cd C:/Users/ericb/Documents/GitHub/PSD_opt/mcpbe/src
python -m wmcpbe.kernels.breakage.blueprint
```

**Option C: Im Solver verwenden (vollständiger Test)**

```python
from wmcpbe.kernels.breakage import get_breakage_kernel

kernel = get_breakage_kernel('my_custom_kernel', param1=0.2)
rate = kernel.compute_rate(v_particle=1e-18)
print(f"S = {rate:.3e} 1/s")
```

## 📖 Blueprint-Struktur

Jede Blueprint-Datei folgt diesem Aufbau:

```
================================================================================
HEADER
================================================================================
- Zweck des Kernel-Typs
- Physikalischer Hintergrund
- Implementierungs-Guide (Schritt-für-Schritt)

================================================================================
KLASSEN-DEFINITION
================================================================================

class ExampleKernel(BaseKernel):
    """
    Ausführliche Docstring mit:
    - Physikalische Modellbeschreibung
    - Parameter-Erklärung
    - Anwendungsszenarien
    - Literaturangaben
    - Verwendungsbeispiel
    """
    
    # --- REQUIRED: Kernel Name ---
    @property
    def name(self) -> str:
        """Return kernel name (must match registry key)."""
        return 'example_kernel'
    
    # --- REQUIRED: Default Parameters ---
    def get_default_params(self) -> dict:
        """Define default parameters with units and ranges."""
        return {
            'param1': value,  # Comment: units, range, default justification
            'param2': value,
        }
    
    # --- REQUIRED: Constructor ---
    def __init__(self, **params):
        """Initialize kernel, validate params, cache values."""
        defaults = self.get_default_params()
        defaults.update(params)
        self.params = self.validate_params(defaults)
        # Cache frequently used values
        self.param1 = float(self.params['param1'])
    
    # --- REQUIRED: Parameter Validation ---
    def validate_params(self, params: dict) -> dict:
        """Validate input parameters, raise ValueError if invalid."""
        if params['param1'] <= 0:
            raise ValueError("param1 must be positive")
        return params
    
    # --- REQUIRED: Core Computation Method ---
    def compute_xxx(self, ...):
        """
        Main computation method.
        
        Detailed docstring with:
        - Parameter descriptions (type, range, purpose)
        - Return value description
        - Important notes
        - Accessing solver state
        - Multiple example implementations
        
        Args:
            param1: Description...
            param2: Description...
        
        Returns:
            result: Description...
        """
        # STEP 1: [Preparation]
        # ...
        
        # STEP 2: [Core Physics]
        # TODO: Add your physics here!
        # Examples provided as comments
        
        # STEP 3: [Edge Cases]
        # ...
        
        return result
    
    # --- OPTIONAL: Helper Methods ---
    def _helper_method(self, ...):
        """Utility functions for complex calculations."""
        pass


================================================================================
TESTING SECTION
================================================================================

if __name__ == '__main__':
    """Quick test of the kernel implementation."""
    # Test creation
    kernel = ExampleKernel()
    
    # Test computation
    result = kernel.compute_xxx(...)
    
    # Test edge cases
    # ...
    
    print("✓ All tests passed!")
```

## 🔧 Kernel-spezifische Details

### Aggregation Kernel

**Methode:** `compute_beta(r1, r2, particle1_idx, particle2_idx, solver)`

**Rückgabe:** `float` (Kollisionsfrequenz [m³/s])

**Typische Modelle:**
- Shear-induced (Smoluchowski)
- Brownian diffusion
- Differential settling
- Liquid bridge enhancement

**Besonderheit:** Kann auf Solver-Zustand zugreifen (Sättigung, Porosität)

---

### Breakage Kernel

**Methode:** `compute_rate(v_particle, particle_idx, solver)`

**Rückgabe:** `float` (Bruchrate [1/s])

**Typische Modelle:**
- Power-law (legacy-kompatibel)
- Stress-based (Weibull)
- Energy-based

**Besonderheit:** Größe-abhängige Rate (größere Partikel brechen leichter)

---

### Porosity Growth Kernel

**Methoden:** 
- `compute_merged_porosity(...)` → `(v_dry, porosity)`
- `compute_fragment_porosity(...)` → `porosity`
- `compute_nucleation_porosity(...)` → `(v_dry, porosity)`

**Rückgabe:** Tuple oder float (Porosität dimensionless)

**Typische Modelle:**
- Volume mixing (additiv)
- Incomplete mixing (trapped pores)
- Energy-dependent

**Besonderheit:** 
- DREI Methoden für konsistente Physik!
- Unterscheidung Vollkörper (NaN) vs. porös ([0,1])
- V_flat-Semantik beachten (V_solid vs. V_dry)

---

### Compression Kernel

**Methode:** `compute_porosity_decay(porosity, dt, v_particle, local_stress, saturation, solver)`

**Rückgabe:** `float` (neue Porosität)

**Typische Modelle:**
- Exponential decay
- Stress-compaction
- Consolidation theory

**Besonderheit:** 
- Zeit-basiert (nicht ereignis-basiert!)
- Nur Porositäts-REDUKTION
- Operative Splitting im Solver

---

### Liquid Distribution Kernel

**Methode:** `select_target_particle(solver, v_droplet, current_time, rng)`

**Rückgabe:** `int` (Partikel-Index)

**Typische Modelle:**
- Uniform weighted
- Surface weighted
- Saturation preferential

**Besonderheit:**
- Gewichtetes Random Sampling (keine Rate!)
- IMMER Solver-Zugriff erforderlich
- Selektiert EINEN Partikel pro Aufruf

## 📝 Best Practices

### 1. Parameter-Validierung

```python
def validate_params(self, params: dict) -> dict:
    # Immer prüfen!
    if params['param1'] <= 0:
        raise ValueError("param1 must be positive, got {params['param1']}")
    
    # Bereichsprüfungen
    if not (0 <= params['param2'] <= 1):
        raise ValueError("param2 must be in [0, 1]")
    
    return params
```

### 2. Numerische Stabilität

```python
# NaN behandeln
if np.isnan(porosity):
    return np.nan  # oder Fallback

# Division durch Null vermeiden
denominator = max(value, 1e-30)

# Inf/NaN abfangen
if not np.isfinite(result):
    result = fallback_value
```

### 3. Performance

```python
# Häufig genutzte Werte cachen (in __init__)
self.param1 = float(self.params['param1'])

# Dictionary-Lookups vermeiden (in hot path)
# SCHLECHT:
beta = self.params['param1'] * r**2

# GUT:
beta = self.param1 * r**2
```

### 4. Solver-Zugriff

```python
# Immer auf None prüfen!
if solver is not None and particle_idx is not None:
    # Sicher zugreifen
    poro = solver.porosity[particle_idx]
    
    # NaN prüfen
    if not np.isnan(poro):
        # Verwenden
        pass

# Attribute mit getattr (fallback-safe)
g = getattr(solver, 'G', 1000.0)  # Default: 1000 1/s
```

### 5. Dokumentation

```python
def compute_xxx(self, param1, param2, ...):
    """
    Kurze Beschreibung (1 Zeile).
    
    Ausführliche Erklärung (optional):
    - Physikalischer Hintergrund
    - Mathematische Formeln
    - Randbedingungen
    
    Args:
        param1: Typ, Bereich, Zweck, Einheit
                Beispiel: 1e-6 (typischer Wert)
        param2: ...
    
    Returns:
        result: Typ, Bereich, physikalische Bedeutung
                Beispiel: 1e-15 m³/s (typisch)
    
    Note:
    - Wichtige Hinweise
    - Bekannte Einschränkungen
    - Performance-Considerations
    
    Example:
        >>> kernel = MyKernel(param1=0.1)
        >>> result = kernel.compute_xxx(1e-6, 2e-6)
        >>> print(result)
        1.23e-15
    """
```

## 🧪 Testing

### Unit Tests (direkt in Blueprint)

```python
if __name__ == '__main__':
    print("Testing kernel...")
    
    # Creation
    kernel = MyKernel()
    
    # Basic computation
    result = kernel.compute_xxx(...)
    assert result > 0, "Result must be positive"
    
    # Edge cases
    result_nan = kernel.compute_xxx(np.nan, ...)
    assert np.isnan(result_nan), "NaN input should give NaN output"
    
    print("✓ All tests passed!")
```

### Integration Tests (im Solver)

```python
def test_my_kernel_integration():
    solver = MCPBESolver(
        agg_kernel_name='my_custom_kernel',
        agg_kernel_params={'param1': 0.1},
    )
    
    # Run simulation
    solver.solve()
    
    # Check results
    assert solver.mass_error < 1e-6, "Mass conservation violated"
```

## 📚 Weiterführende Ressourcen

- **Kern-Dokumentation**: `README.md` (Framework-Übersicht)
- **Kernel-Übersicht**: `KERNEL_UEBERSICHT.md` (alle verfügbaren Kernel)
- **Base-Klassen**: `base.py` (abstrakte Schnittstellen)
- **Beispiele**: `*/volume_mixing.py`, `*/shear_chin1998.py` (konkrete Implementierungen)

## ❓ FAQ

**Q: Muss ich alle Methoden implementieren?**  
A: Ja, alle mit `REQUIRED` markierten Methoden müssen vorhanden sein. Optionale Helper-Methoden können weggelassen werden.

**Q: Was tun bei NaN-Werten?**  
A: Immer explizit prüfen! Für Porosität: `if np.isnan(porosity): ...`. Für andere Werte: `np.nan_to_num()` oder Fallback.

**Q: Wie teste ich meinen Kernel?**  
A: 1) Direktes Ausführen (`python my_kernel.py`), 2) Im Solver verwenden, 3) Massenbilanz prüfen.

**Q: Kann ich mehrere Kernel kombinieren?**  
A: Pro Kategorie nur EIN Kernel aktiv (z.B. EIN Aggregations-Kernel). Aber verschiedene Kategorien können kombiniert werden (z.B. `shear_chin1998` + `volume_mixing` + `exponential_decay`).

**Q: Wo melde ich Bugs?**  
A: Issue-Tracker im Repository. Bitte Minimalbeispiel und Fehlermeldung beifügen.

---

*Stand: Juli 2026*  
*WMCPBE Development Team*
