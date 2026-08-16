# WMCPBE Kernel Framework

Modulare Physik-Kernel für den gewichteten DSMC Monte Carlo PBE Solver.

---

## 📌 INHALT

1. [Übersicht](#übersicht)
2. [Kernel-Architektur](#kernel-architektur)
3. [Verfügbare Kernel](#verfügbare-kernel)
4. [Kernel erstellen](#kernel-erstellen)
5. [Konfigurationsbeispiele](#konfigurationsbeispiele)
6. [Physik-Hintergrund](#physik-hintergrund)

---

## ÜBERSICHT

Das Kernel-Framework ermöglicht modulare Physik-Modelle für:

| Prozesstyp | Beschreibung | Kernel-Klassen |
|------------|--------------|----------------|
| **Aggregation** | Partikelkollision & Koaleszenz | `AggregationKernel` |
| **Breakage** | Partikelzerfall durch Scherung/Stress | `BreakageKernel` |
| **Porosity Growth** | Porositätsbildung bei Benetzung | `PorosityGrowthKernel` |
| **Compression** | Porositätsreduktion durch Kompression | `CompressionKernel` |
| **Liquid Distribution** | Tropfenverteilung auf Partikel | `LiquidDistributionKernel` |
| **Liquid Internalization** | Kapillare Flüssigkeitsaufnahme | `LiquidInternalizationKernel` |
| **Agg. Acceptance** | Kollisionsakzeptanz (Stokes) | `AggAcceptanceKernel` |

---

## KERNEL-ARCHITEKTUR

### Basisklassen

```python
from wmcpbe.kernels.base import AggregationKernel, BreakageKernel

class MyCustomKernel(AggregationKernel):
    """Benutzerdefinierter Agglomerations-Kernel."""
    
    name = "my_custom"  # Eindeutiger Name
    
    def __init__(self, params: dict):
        super().__init__(params)
        self.my_param = params.get('my_param', 1.0)
    
    def compute_rate(self, i: int, j: int, solver) -> float:
        """Berechne Kollisionsrate β(i,j)."""
        # Implementierung hier
        return beta_ij
```

### Kernel-Registrierung

Kernel werden automatisch registriert via `get_*_kernel()` Funktionen:

```python
from wmcpbe.kernels.aggregation import get_aggregation_kernel

# Kernel erstellen
kernel = get_aggregation_kernel(
    'shear_chin1998',
    params={'corr_beta': 1e-3, 'g': 1000}
)
```

### Verwendung im Solver

```python
solver = MCPBESolver(
    agg_kernel_name='shear_chin1998',
    agg_kernel_params={'corr_beta': 1e-3, 'g': 1000},
    break_kernel_name='power_law',
    break_kernel_params={'p1': 0.01, 'p2': 1.0, 'breakrval': 4},
)
```

---

## VERFÜGBARE KERNEL

### 1. AGGREGATION KERNELS

#### shear_chin1998

**Beschreibung:** Shear-induzierte Agglomeration nach Chin et al. (1998).

**Formel:**
```
β(i,j) = corr_beta × G × (r_i + r_j)³
```

**Parameter:**
| Parameter | Einheit | Beschreibung | Typisch |
|-----------|---------|--------------|---------|
| `corr_beta` | - | Korrekturfaktor (Kalibrierung) | 1e-4 bis 1e-2 |
| `g` | 1/s | Scherrate | 100 bis 5000 |

**Beispiel:**
```python
agg_kernel_name='shear_chin1998',
agg_kernel_params={
    'corr_beta': 1e-3,
    'g': 1000,
}
```

**Referenz:** Chin, W.C., et al. (1998). "Agglomeration in high-shear mixers."

---

#### brownian_tsouris1995

**Beschreibung:** Brownsche Bewegung nach Tsouris & Tavlarides (1995).

**Formel:**
```
β(i,j) = (2k_BT/3μ) × (1/V_i + 1/V_j) × (V_i^(1/3) + V_j^(1/3))
```

**Parameter:** Keine (nur temperaturabhängig)

**Beispiel:**
```python
agg_kernel_name='brownian_tsouris1995',
agg_kernel_params={}
```

**Anwendung:** Feine Partikel (<10µm) in ruhenden Systemen.

---

#### constant

**Beschreibung:** Konstanter Kernel (einfachster Fall).

**Formel:**
```
β(i,j) = coeff
```

**Parameter:**
| Parameter | Einheit | Beschreibung | Typisch |
|-----------|---------|--------------|---------|
| `coeff` | m³/s | Konstante Rate | 1e-12 bis 1e-8 |

**Beispiel:**
```python
agg_kernel_name='constant',
agg_kernel_params={'coeff': 1e-9}
```

**Anwendung:** Testfälle, theoretische Studien.

---

#### sum_kernel

**Beschreibung:** Additiver Kernel (Volumen-summiert).

**Formel:**
```
β(i,j) = coeff × (V_i + V_j)
```

**Parameter:**
| Parameter | Einheit | Beschreibung | Typisch |
|-----------|---------|--------------|---------|
| `coeff` | 1/s | Ratenkonstante | 0.01 bis 1.0 |

**Beispiel:**
```python
agg_kernel_name='sum_kernel',
agg_kernel_params={'coeff': 0.1}
```

---

#### liquid_bridge

**Beschreibung:** Flüssigkeitsbrücken-induzierte Agglomeration.

**Formel:** Berücksichtigt optimale Sättigung für maximale Brückenkraft.

**Parameter:**
| Parameter | Einheit | Beschreibung | Typisch |
|-----------|---------|--------------|---------|
| `corr_beta` | - | Korrekturfaktor | 1e-4 bis 1e-2 |
| `g` | 1/s | Scherrate | 100 bis 5000 |
| `optimal_saturation` | - | Optimale Sättigung | 0.3 bis 0.7 |

**Beispiel:**
```python
agg_kernel_name='liquid_bridge',
agg_kernel_params={
    'corr_beta': 1e-3,
    'g': 1000,
    'optimal_saturation': 0.5,
}
```

**Anwendung:** Feuchtgranulation mit bindemittelhaltigen Tropfen.

---

### 2. BREAKAGE KERNELS

#### power_law

**Beschreibung:** Power-Law Breakage (volumenbasiert).

**Formel:**
```
Γ(V) = p1 × G × V^p2
```

**Parameter:**
| Parameter | Einheit | Beschreibung | Typisch |
|-----------|---------|--------------|---------|
| `p1` | 1/s·m⁻³ᵅ | Pre-factor | 1e-4 bis 1e-1 |
| `p2` | - | Shear-Exponent | 1.0 bis 3.0 |
| `g` | 1/s | Scherrate | 100 bis 5000 |
| `p2` | - | Volumen-Exponent | 0.0 bis 2.0 |

**Beispiel:**
```python
break_kernel_name='power_law',
break_kernel_params={
    'p1': 0.01,
    'g': 1000,
}
```

---

#### stress_based

**Beschreibung:** Stress-basiertes Breakage (energiegetrieben).

**Formel:**
```
Γ(V) = f(σ_mech / σ_strength)
```

**Parameter:**
| Parameter | Einheit | Beschreibung | Typisch |
|-----------|---------|--------------|---------|
| `stress_coeff` | - | Spannungs-Koeffizient | 0.1 bis 10 |
| `g` | 1/s | Scherrate | 100 bis 5000 |

**Beispiel:**
```python
break_kernel_name='stress_based',
break_kernel_params={
    'stress_coeff': 1.0,
    'g': 1000,
}
```

---

#### powerlaw_rumpf

**Beschreibung:** PowerLaw-Rumpf mit porositäts-/sättigungsabhängiger Festigkeit.

**Formel:**
```
Γ(V) = p1 × G × V^p2 / sigma(poro, sat)

f_strength = k × (1 - α × sat)^γ / (1 + δ × poro)
```

**Parameter:**
| Parameter | Einheit | Beschreibung | Typisch |
|-----------|---------|--------------|---------|
| `p1` | 1/s·Pa·m⁻³ᵅ | Pre-factor | 0.1 bis 10 |
| `p2` | - | Shear-Exponent | 0.5 bis 2.0 |
| `g` | 1/s | Scherrate | 100 bis 5000 |
| `breakrval` | - | Modellvariante (1-4) | 4 (empfohlen) |
| `p2` | - | Volumen-Exponent | 0.5 bis 2.0 |
| `k` | - | Rumpf k (trocken) | 2.2 bis 2.8 |
| `alpha` | - | Rumpf α (nass) | 1.0 bis 1.33 |
| `gamma` | N/m | Oberflächenspannung | 0.03 bis 0.07 |
| `delta` | rad | Kontaktwinkel | 0.0 bis 0.5 |
| `x_s` | m | Referenzgröße | None (auto) |

**Beispiel:**
```python
break_kernel_name='powerlaw_rumpf',
break_kernel_params={
    'p1': 0.5,
    'p2': 1.0,
    'g': 1000,
    'breakrval': 4,
    'k': 2.5,
    'alpha': 1.0,
    'gamma': 0.036,
    'delta': 0.0,
}
```

**Anwendung:** Feuchtgranulation mit porösen Partikeln.

**Referenz:** Rumpf, K.E. (1990). "Granulation processes."

---

### 3. POROSITY GROWTH KERNELS

#### volume_mixing

**Beschreibung:** Volumen-Mischung bei Erstbenetzung.

**Logik:**
- Erster Tropfen → poro = 0.4 (empirisch)
- Weitere Tropfen → Porosität aus Volumenbilanz

**Parameter:** Keine

**Beispiel:**
```python
porosity_growth_kernel_name='volume_mixing',
porosity_growth_kernel_params={}
```

**Anwendung:** Erste Benetzung trockener Pulver.

---

#### incomplete_mixing

**Beschreibung:** Unvollständige Mischung (eingeschlossene Poren).

**Logik:**
- Ein Teil der Poren wird bei Agglomeration eingeschlossen
- Restporosität aus Trapped-Pore-Fraktion

**Parameter:**
| Parameter | Einheit | Beschreibung | Typisch |
|-----------|---------|--------------|---------|
| `trapped_pore_fraction` | - | Eingeschlossene Poren | 0.1 bis 0.3 |

**Beispiel:**
```python
porosity_growth_kernel_name='incomplete_mixing',
porosity_growth_kernel_params={'trapped_pore_fraction': 0.15}
```

---

#### cone_model

**Beschreibung:** Kegelmodell für Porenwachstum.

**Logik:**
- Porosität wächst mit ΔV bei Agglomeration
- Geometrisches Kegelmodell

**Parameter:** Keine (Default-Werte intern)

**Beispiel:**
```python
porosity_growth_kernel_name='cone_model',
porosity_growth_kernel_params={}
```

**Anwendung:** Kontinuierliches Porositätswachstum.

---

### 4. COMPRESSION KERNELS

#### exponential_decay

**Beschreibung:** Exponentielle Porositätsabnahme über Zeit.

**Formel:**
```
poro(t) = poro_min + (poro_0 - poro_min) × exp(-rate × t)
```

**Parameter:**
| Parameter | Einheit | Beschreibung | Typisch |
|-----------|---------|--------------|---------|
| `rate` | 1/s | Zerfallsrate | 0.01 bis 0.1 |
| `min_porosity` | - | Minimalporosität | 0.1 bis 0.3 |

**Beispiel:**
```python
compression_kernel_name='exponential_decay',
compression_kernel_params={
    'rate': 0.02,
    'min_porosity': 0.15,
}
```

---

#### stress_compaction

**Beschreibung:** Spannungsgetriebene Kompaktion.

**Formel:**
```
dp/dt = f(σ_applied, material_props)
```

**Parameter:**
| Parameter | Einheit | Beschreibung | Typisch |
|-----------|---------|--------------|---------|
| `compaction_coeff` | 1/Pa·s | Kompaktionskoeffizient | 1e-6 bis 1e-4 |

**Beispiel:**
```python
compression_kernel_name='stress_compaction',
compression_kernel_params={'compaction_coeff': 1e-5}
```

---

### 5. LIQUID DISTRIBUTION KERNELS

#### uniform_weighted

**Beschreibung:** Uniforme Verteilung gewichtet nach W.

**Logik:** Alle Partikel haben gleiche Kollisionswahrscheinlichkeit.

**Parameter:** Keine

**Beispiel:**
```python
liquid_dist_kernel_name='uniform_weighted',
liquid_dist_kernel_params={}
```

**Default:** Wird automatisch verwendet wenn nicht spezifiziert.

---

#### surface_weighted

**Beschreibung:** Oberflächen-gewichtete Verteilung.

**Logik:** Größere Partikel erhalten mehr Tropfen (proportional zu Oberfläche).

**Parameter:** Keine

**Beispiel:**
```python
liquid_dist_kernel_name='surface_weighted',
liquid_dist_kernel_params={}
```

**Anwendung:** Wenn Tropfen bevorzugt große Partikel treffen.

---

#### saturation_preferential

**Beschreibung:** Bevorzugte Verteilung auf trockene Partikel.

**Logik:** Partikel mit niedriger Sättigung werden bevorzugt.

**Parameter:** Keine

**Beispiel:**
```python
liquid_dist_kernel_name='saturation_preferential',
liquid_dist_kernel_params={}
```

**Anwendung:** Homogenisierung der Flüssigkeitsverteilung.

---

### 6. LIQUID INTERNALIZATION KERNELS

#### liquid_internalization

**Beschreibung:** Kapillar-getriebene interne Flüssigkeitsaufnahme.

**Formel:**
```
dV_int/dt = k_intern × (V_pore - V_int)
```

**Parameter:**
| Parameter | Einheit | Beschreibung | Typisch |
|-----------|---------|--------------|---------|
| `k_intern` | 1/(m³·s) | Internalisierungsrate | 1e6 bis 1e10 |

**Beispiel:**
```python
liquid_internalization_kernel_name='liquid_internalization',
liquid_internalization_kernel_params={'k_intern': 1e8}
```

**Anwendung:** Langsame Porenfüllung während Simulation.

---

### 7. AGGLOMERATION ACCEPTANCE KERNELS

#### stokes_krit

**Beschreibung:** Stokes-Kriterium für Kollisionsakzeptanz (Braumann et al. 2007).

**Formel:**
```
St = (4/3) × ρ_part × U_coll × r / (9 × μ_binder)
E_coag = 1 wenn St < St_crit, sonst 0
```

**Parameter:**
| Parameter | Einheit | Beschreibung | Typisch |
|-----------|---------|--------------|---------|
| `U_coll` | m/s | Kollisionsgeschwindigkeit | 0.1 bis 2.0 |
| `binder_viscosity` | Pa·s | Binder-Viskosität | 0.01 bis 10 |
| `rho_solid` | kg/m³ | Feststoffdichte | 1000 bis 5000 |
| `rho_liquid` | kg/m³ | Flüssigkeitsdichte | 800 bis 1500 |
| `h_a` | m | Minimale Filmdicke | 1e-9 bis 1e-6 |

**Beispiel:**
```python
agg_acceptance_kernel_name='stokes_krit',
agg_acceptance_kernel_params={
    'U_coll': 0.5,
    'binder_viscosity': 0.1,
    'rho_solid': 2500,
    'rho_liquid': 1000,
    'h_a': 1e-7,
}
```

**Anwendung:** Nassgranulation mit viskosem Bindemittel.

**Referenz:** Braumann, A.P., et al. (2007). "Population balance modelling of granulation."

---

### 8. LIQ. INTERNALIZATION DURING AGGLOMERATION

#### braumann_2007

**Beschreibung:** Poren-Einschluss bei Agglomeration (Braumann et al. 2007).

**Logik:**
- Bei Kontakt zweier Partikel wird externe Flüssigkeit eingeschlossen
- Interne Flüssigkeit = min(V_liq_ext, V_pore_available)

**Parameter:** Keine

**Beispiel:**
```python
liq_internalisation_agglomeration_kernel_name='braumann_2007',
liq_internalisation_agglomeration_kernel_params={}
```

**Anwendung:** Realistische Porenfüllung bei Agglomeration.

---

## KONFIGURATIONSPRAXIS

### Beispiel 1: Trockene Agglomeration

```python
solver = MCPBESolver(
    dim=1,
    t_total=60.0,
    seed=42,
    load_attr=False,
    
    # Nur Agglomeration (Shear)
    agg_kernel_name='shear_chin1998',
    agg_kernel_params={'corr_beta': 1e-3, 'g': 1000},
    
    maybe_double_control_volume=False,
    recon_enable=False,
)
```

---

### Beispiel 2: Feuchtgranulation (Vollsuite)

```python
solver = MCPBESolver(
    dim=1,
    t_total=300.0,
    seed=42,
    load_attr=False,
    
    # Agglomeration mit Flüssigkeitsbrücken
    agg_kernel_name='liquid_bridge',
    agg_kernel_params={
        'corr_beta': 1e-3,
        'g': 1000,
        'optimal_saturation': 0.5,
    },
    
    # Stokes-Akzeptanzkriterium
    agg_acceptance_kernel_name='stokes_krit',
    agg_acceptance_kernel_params={
        'U_coll': 0.5,
        'binder_viscosity': 0.1,
        'rho_solid': 2500,
        'rho_liquid': 1000,
        'h_a': 1e-7,
    },
    
    # Breakage mit Rumpf-Stärke
    break_kernel_name='powerlaw_rumpf',
    break_kernel_params={
        'p1': 0.5,
        'p2': 1.0,
        'g': 1000,
        'breakrval': 4,
            'k': 2.5,
        'alpha': 1.0,
        'gamma': 0.036,
        'delta': 0.0,
    },
    
    # Porositätswachstum
    porosity_growth_kernel_name='cone_model',
    porosity_growth_kernel_params={},
    
    # Kompression
    compression_kernel_name='exponential_decay',
    compression_kernel_params={
        'rate': 0.02,
        'min_porosity': 0.15,
    },
    
    # Kontinuierliche Internalisierung
    liquid_internalization_kernel_name='liquid_internalization',
    liquid_internalization_kernel_params={'k_intern': 1e8},
    
    # Event-basierte Internalisierung
    liq_internalisation_agglomeration_kernel_name='braumann_2007',
    liq_internalisation_agglomeration_kernel_params={},
    
    maybe_double_control_volume=False,
    recon_enable=False,
)

# Nukleation hinzufügen
solver.create_nucleation_handler(
    enabled=True,
    volumetric_flow_rate=1e-9,
    droplet_diameter=100e-6,
    liquid_addition_duration=180.0,
    batch_size=25.0,
)
```

---

### Beispiel 3: Eigener Kernel

```python
# Datei: wmcpbe/kernels/aggregation/my_custom.py

from wmcpbe.kernels.base import AggregationKernel

class MyCustomKernel(AggregationKernel):
    """Benutzerdefinierter Kernel für spezielle Physik."""
    
    name = "my_custom"
    
    def __init__(self, params: dict):
        super().__init__(params)
        self.custom_param = params.get('custom_param', 1.0)
    
    def compute_rate(self, i: int, j: int, solver) -> float:
        """Berechne β(i,j) basierend auf benutzerdefinierter Formel."""
        r_i = solver.X[i] * 0.5
        r_j = solver.X[j] * 0.5
        
        # Benutzerdefinierte Physik hier
        beta = self.custom_param * (r_i + r_j) ** 2
        
        return float(beta)

# Registrierung in wmcpbe/kernels/aggregation/__init__.py
from .my_custom import MyCustomKernel

AGG_KERNELS['my_custom'] = MyCustomKernel
```

---

## PHYSIK-HINTERGRUND

### Agglomeration

**Mechanismen:**
1. **Shear-induziert:** Partikel kollidieren durch Geschwindigkeitsgradienten
2. **Brownsch:** Thermische Bewegung dominiert bei kleinen Partikeln
3. **Flüssigkeitsbrücken:** Kapillarkräfte beschleunigen Koaleszenz

**Kritische Parameter:**
- Scherrate G [1/s]: Höhere Scherung → mehr Kollisionen
- Korrekturfaktor corr_beta: Kalibrierung an Experimente
- Optimale Sättigung: Max. Brückenkräfte bei S ≈ 0.3-0.7

---

### Breakage

**Mechanismen:**
1. **Attrition:** Oberflächenabrieb (geringe Energie)
2. **Fragmentation:** Zerfall in mehrere Teile (hohe Energie)
3. **Erosion:** Kontinuierlicher Materialverlust

**Kritische Parameter:**
- Mechanische Spannung σ_mech ∝ G²
- Partikelfestigkeit σ_strength(poro, sat)
- Kritische Partikelgröße V_crit wo Γ(V) maximal

**PowerLaw-Rumpf Besonderheit:**
- Stärke nimmt mit Sättigung ab (α > 1)
- Stärke nimmt mit Porosität ab
- Oberflächenspannung γ erhöht nasse Festigkeit

---

### Porositätswachstum

**Mechanismen:**
1. **Erstbenetzung:** Sofortige Porositätsbildung (volume_mixing)
2. **Schichtwachstum:** Porosität wächst mit jeder Schicht (cone_model)
3. **Poreneinschluss:** Bei Agglomeration eingeschlossene Poren

**Kritische Parameter:**
- Initiale Porosität ε_0: Typisch 0.0 (trocken) bis 0.4 (erste Benetzung)
- Maximale Porosität ε_max: Theoretisch ~0.74 (FCC-Packung)
- Trapped-Pore-Fraktion: 10-30% bei realen Granulaten

---

### Liquid Internalization

**Mechanismen:**
1. **Kapillar:** spontane Porenfüllung (Washburn-Gleichung)
2. **Viskos:** verzögert durch hohe Viskosität
3. **Agglomeration:** Einschluss bei Partikelkontakt

**Zeitskalen:**
- Schnell: k_intern > 1e9 (sofortige Füllung)
- Mittel: k_intern ≈ 1e6-1e8 (Sekunden bis Minuten)
- Langsam: k_intern < 1e4 (Stunden)

---

## FEHLERBEHANDLUNG

### Validation bei Kernel-Erstellung

```python
from wmcpbe.kernels.aggregation import get_aggregation_kernel

try:
    kernel = get_aggregation_kernel('shear_chin1998', {'corr_beta': -1e-3})
except ValueError as e:
    print(f"Ungültiger Parameter: {e}")
```

### Typische Fehler

| Fehler | Ursache | Lösung |
|--------|---------|--------|
| `Unknown kernel 'xyz'` | Tippfehler oder nicht registriert | Kernel-Name prüfen |
| `Missing required param` | Parameter vergessen | docs/Parameter_Liste.md konsultieren |
| `NaN in rate calculation` | Ungültige Werte (negativ, NaN) | Input-Validation vor compute_rate() |

---

## PERFORMANCE-TIPPS

1. **JIT-Kernel bevorzugen:**
   - `jit_kernels.py` enthält hochoptimierte Versionen
   - Bis zu 100× schneller als Python-Loops

2. **CDF-Caching bei Breakage:**
   - Erstes Mal langsam (CDF wird aufgebaut)
   - Folgeaufrufe nutzen Cache

3. **Moment Mode für Agglomeration:**
   ```python
   solver.agg_propensity_mode = "moment"  # O(n) statt O(n²)
   ```

---

## 📚 REFERENZEN

### Wissenschaftliche Arbeiten

1. **Chin et al. (1998):** "Agglomeration in high-shear mixers"
2. **Tsouris & Tavlarides (1995):** "Breakage and coalescence models"
3. **Braumann et al. (2007):** "Population balance modelling of granulation"
4. **Rumpf (1990):** "Granulation processes"

### Interne Dokumente

- [Parameter_Liste.md](../../docs/Parameter_Liste.md) - Vollständige Parameter
- [QUICKSTART.md](../../docs/QUICKSTART.md) - Erster Einstieg
- [PERFORMANCE.md](../../docs/PERFORMANCE.md) - Benchmarks

---

**Letzte Aktualisierung:** 2024-01-XX  
**Maintainer:** WMCPBE Development Team
