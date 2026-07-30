# WMCPBE Kernel Framework

## Übersicht

Das Kernel-Framework ermöglicht eine flexible, erweiterbare Architektur für physikalische Modelle im WMCPBE-Solver. Jeder physikalische Prozess (Agglomeration, Bruch, Porositätswachstum, Kompression, Flüssigkeitsverteilung) wird durch austauschbare Kernel implementiert.

## Motivation

**Bisherige Architektur:**
- Kernel-Auswahl über Integer-Flags (`COLEVAL`, `BREAKRVAL`, etc.)
- Feste Implementierung in JIT-Dateien (`jit_kernel_agg.py`, `jit_kernel_break.py`)
- Schwer erweiterbar: Neue Kernel erfordern Änderungen an Core-Dateien
- Porositätslogik fest in `_do_one_agg()` codiert

**Neue Architektur:**
- Kernel-Auswahl über String-Namen und Parameter-Dicts
- Jede Kernel-Implementierung in separater Klasse
- Einfache Erweiterung: Neue Klasse + Factory-Eintrag
- **PorosityGrowthKernel mit 3 konsistenten Methoden für Agg/Break/Nuc**

## Struktur

```
kernels/
├── base.py                          # Abstrakte Basisklassen
│   ├── KernelBase
│   ├── AggregationKernel
│   ├── BreakageKernel
│   ├── PorosityGrowthKernel         # 3 Methoden: Agg/Break/Nuc
│   ├── CompressionKernel
│   └── LiquidDistributionKernel
│
├── aggregation/                     # Agglomerationskerne
│   ├── shear_chin1998.py            # Shear-induziert (Chin 1998)
│   ├── brownian_tsouris1995.py      # Brown'sche Diffusion
│   ├── constant.py                  # Konstanter Kernel
│   ├── sum_kernel.py                # Summenkernel
│   └── liquid_bridge.py             # Mit Flüssigkeitsbrücken (NEU)
│
├── breakage/                        # Bruchkerne
│   ├── power_law.py                 # Power-Law (bestehend)
│   └── stress_based.py              # Stress-basiert (NEU)
│
├── porosity_growth/                 # Porositätswachstum (Ereignis-basiert)
│   ├── volume_mixing.py             # Volumenaddition (Standard)
│   └── incomplete_mixing.py         # Eingeschlossene Poren (Erweitert)
│
├── compression/                     # Kompression (Zeit-basiert)
│   ├── exponential_decay.py         # Exponentieller Zerfall (bestehend)
│   └── stress_compaction.py         # Stress-abhängig (NEU)
│
└── liquid_distribution/             # Flüssigkeitsverteilung
    ├── uniform_weighted.py          # Gleichverteilt (bestehend)
    ├── surface_weighted.py          # Oberflächen-gewichtet (NEU)
    └── saturation_preferential.py   # Sättigungs-präferentiell (NEU)
```

---

### PorosityGrowthKernel mit einheitlicher Physik

### Design-Prinzip: EIN Kernel = DREI Operationen

Jeder `PorosityGrowthKernel` implementiert **drei konsistente Methoden**, die alle auf demselben physikalischen Modell basieren:

```python
class PorosityGrowthKernel(KernelBase):
    
    def compute_merged_porosity(self, ...) -> tuple[float, float]:
        """
        AGGLOMERATION: Zwei Partikel mergen
        
        Physikalische Annahme: 
        - volume_mixing: Volumina additiv
        - incomplete_mixing: +Trapped Pores durch unvollständige Koaleszenz
        """
        pass
    
    def compute_fragment_porosity(self, ...) -> float:
        """
        BREAKAGE: Fragment-Porosität aus Parent
        
        Physikalische Annahme:
        - volume_mixing: Fragmente erben Parent-Porosität (gleichmäßige Aufteilung)
        - incomplete_mixing: Porenkollaps durch mechanischen Stress
        """
        pass
    
    def compute_nucleation_porosity(self, ...) -> tuple[float, float]:
        """
        NUCLEATION: Porosität neu gebildeter Partikel
        
        Wichtig: Default-Porosität (0.4) ermöglicht Porositätsentstehung für Standartfall.
        Ohne Default würde volume_mixing nie Porosität aus Vollkörpern erzeugen.
        """
        pass
```

### Warum drei Methoden?

| Grund 			| Erklärung 								|
|-------------------------------|-----------------------------------------------------------------------|
| **Konsistenz** 		| Alle Operationen basieren auf DEMSELBEN physikalischen Modell 	|
| **Wartbarkeit** 		| Änderungen am Modell nur an einer Stelle 				|
| **Erweiterbarkeit** 		| Neue Modelle (z.B. `stress_dependent_mixing`) einfach implementierbar |
| **Physikalische Korrektheit** | Agglomeration erzeugt Poren → Breakage reduziert sie (spiegelbildlich)|

---

## Verwendung

### Beispiel 1: PorosityGrowthKernel verwenden

```python
from wmcpbe.kernels.porosity_growth import get_porosity_growth_kernel

# VolumeMixingKernel (Standard, additive Poren)
kernel = get_porosity_growth_kernel('volume_mixing')

# Agglomeration
v_dry_merged, poro_merged = kernel.compute_merged_porosity(
    v_dry1=1e-18, poro1=0.4,
    v_dry2=2e-18, poro2=0.5
)

# Breakage
poro_frag = kernel.compute_fragment_porosity(
    parent_porosity=0.4,
    fragment_volume=5e-19,
    parent_volume=1e-18
)

# Nucleation
v_dry_nuc, poro_nuc = kernel.compute_nucleation_porosity(
    v_solid=1e-18,
    v_liquid=1e-19
)
print(f"Nucleation: poro={poro_nuc:.2f} (Default: 0.4)")
```

### Beispiel 2: IncompleteMixingKernel (erweitertes Modell)

```python
from wmcpbe.kernels.porosity_growth import get_porosity_growth_kernel

kernel = get_porosity_growth_kernel('incomplete_mixing',
    trapped_pore_fraction=0.1,   # 10% zusätzliche Poren bei Agg
    collapse_factor=0.1          # 10% Porenkollaps bei Break
)

# Agglomeration: Mehr Poren durch eingeschlossene Volumina
v_dry, poro = kernel.compute_merged_porosity(...)
# → poro höher als bei volume_mixing

# Breakage: Fragmente dichter (Porenkollaps)
poro_frag = kernel.compute_fragment_porosity(...)
# → poro_frag < parent_porosity
```

---

## PorosityGrowthKernel im Detail

### 1. `volume_mixing` (Standard)

**Physikalische Annahme:** Poren und Feststoff sind ideal mischbar, keine eingeschlossenen Volumina.

| Operation 	    | Logik 		| Code 				   |
|-------------------|-------------------|----------------------------------|
| **Agglomeration** | Volumina additiv 	| `V_pore_new = V_pore₁ + V_pore₂` |
| **Breakage** 	    | Vererbung 	| `poro_frag = poro_parent` 	   |
| **Nucleation**    | Default 0.4 	| `poro_new = 0.4` 		   |

**Warum 0.4 in `volume_mixing`?**
- Mit volume_mixing ist Porosität NUR additiv (`V_pore_new = V_pore₁ + V_pore₂`)
- Bei Vollkörpern (poro=NaN) würde **nie** Porosität entstehen!
- **Hardcodierter Default 0.4** in `VolumeMixingKernel.compute_nucleation_porosity()` ermöglicht Porositätsbildung
- Andere Kernel (z.B. `incomplete_mixing`) können eigene Defaults haben

```python
# AGG:
V_solid_merged = V_solid_1 + V_solid_2
V_pore_merged  = V_pore_1 + V_pore_2
poro_merged    = V_pore_merged / (V_solid_merged + V_pore_merged)

# BREAK:
poro_frag = poro_parent  # Einfach vererben

# NUC:
poro_new = 0.4  # Ermöglicht Porositätsbildung!
v_dry = v_solid / (1.0 - 0.4)
```

### 2. `incomplete_mixing` (Erweitert)

**Physikalische Annahme:** Beim Zusammenkleben entstehen eingeschlossene Poren durch unvollständige Koaleszenz.

| Operation | Logik | Code |
|-----------|-------|------|
| **Agglomeration** | +Trapped Pores | `V_pore_new = V_pore₁ + V_pore₂ + ΔV_pore` |
| **Breakage** | Porenkollaps | `poro_frag = poro_parent × (1 - collapse_factor)` |
| **Nucleation** | Default 0.4 | `poro_new = 0.4` |

**Physikalische Mechanismen:**
1. **Contact porosity**: Poren an Partikel-Partikel-Grenzfläche
2. **Surface roughness**: Mikro-Voids durch nicht-glatte Oberflächen
3. **Viscous limitation**: Unzureichende Zeit für vollständige Restrukturierung

```python
# AGG:
V_pore_trapped = f_trap × min(V_dry_1, V_dry_2)
V_pore_merged  = V_pore_1 + V_pore_2 + V_pore_trapped
poro_merged    = V_pore_merged / (V_solid_merged + V_pore_merged)

# BREAK:
collapse_factor = 0.1  # 10% Porenreduktion
poro_frag = poro_parent * (1.0 - collapse_factor)

# NUC:
poro_new = 0.4  # Konsistent mit volume_mixing
```

---

## Kernel-Kategorien

### 1. Aggregation (`aggregation/`)

Berechnet Kollisionsfrequenz β(i,j) zwischen zwei Partikeln.

**Schnittstelle:**
```python
beta = kernel.compute_beta(r1, r2, particle1_idx, particle2_idx, solver)
```

**Verfügbare Kernel:**
| Name 			| Beschreibung 			     | Parameter 				|
|-----------------------|------------------------------------|------------------------------------------|
| `shear_chin1998` 	| Shear-induziert (Chin et al. 1998) | `corr_beta`, `g` 			|
| `brownian_tsouris1995`| Brown'sche Diffusion 		     | `corr_beta`, `temperature`, `viscosity` 	|
| `constant` 		| Konstante Frequenz 		     | `corr_beta` 				|
| `sum` 		| Proportional zu V₁+V₂ 	     | `corr_beta` 				|
| `liquid_bridge` 	| Mit Flüssigkeitsbrücken 	     | `corr_beta`, `g`, `optimal_saturation`,`liquid_enhancement` |

### 2. Breakage (`breakage/`)

Berechnet Bruchrate S(V) für ein Partikel.

**Schnittstelle:**
```python
rate = kernel.compute_rate(v_particle, particle_idx, solver)
```

**Verfügbare Kernel:**
| Name 		| Beschreibung 		   | Parameter 							|
|---------------|--------------------------|------------------------------------------------------------|
| `power_law` 	| Power-Law Rate 	   | `p1`, `p2`, `g`, `breakrval` 				|
| `stress_based`| Stress-basiert (Weibull) | `critical_stress`, `weibull_modulus`, `porosity_weakening` |

### 3. Porosity Growth (`porosity_growth/`) 

Berechnet Porosität bei diskreten Ereignissen (Agg/Break/Nuc).

**Schnittstelle:**
```python
# Agglomeration
v_dry, poro = kernel.compute_merged_porosity(v_dry1, poro1, v_dry2, poro2, ...)

# Breakage
poro_frag = kernel.compute_fragment_porosity(parent_porosity, fragment_volume, parent_volume, ...)

# Nucleation
v_dry, poro = kernel.compute_nucleation_porosity(v_solid, v_liquid, nucleation_params, ...)
```

**Verfügbare Kernel:**
| Name 		     | Agg 		| Break 	| Nuc 	     | Parameter |
|--------------------|------------------|---------------|------------|-----------|
| `volume_mixing`    | Additiv 		| Vererbung 	| 0.4 Default| - |
| `incomplete_mixing`| +Trapped Pores 	| Porenverlust  | siehe Agg  | `trapped_pore_fraction` `revealed_factor` |

### 4. Compression (`compression/`)

Berechnet Porositätsreduktion über Zeit dt (Zeit-basiert).

**Schnittstelle:**
```python
poro_new = kernel.compute_porosity_decay(porosity, dt, v_particle, local_stress, saturation, solver)
```

**Verfügbare Kernel:**
| Name | Beschreibung | Parameter |
|------|--------------|-----------|
| `exponential_decay` | Exponentieller Zerfall | `rate`, `min_porosity` |
| `stress_compaction` | Stress-abhängig | `compaction_rate`, `yield_stress`, `g` |

### 5. Liquid Distribution (`liquid_distribution/`)

Selektiert Partikel für Flüssigkeitszugabe während Nucleation.

**Schnittstelle:**
```python
idx = kernel.select_target_particle(solver, v_droplet, current_time)
```

**Verfügbare Kernel:**
| Name | Beschreibung | Parameter |
|------|--------------|-----------|
| `uniform_weighted` | Gewichts-proportional | - |
| `surface_weighted` | Oberflächen-gewichtet | - |
| `saturation_preferential` | Bevorzugt ungesättigte Partikel | `saturation_bias` |

---

## Eigene Kernel implementieren

### Schritt 1: Basisklasse erweitern

```python
from wmcpbe.kernels.base import PorosityGrowthKernel

class MyCustomPorosityKernel(PorosityGrowthKernel):
    
    @property
    def name(self) -> str:
        return 'my_custom_porosity'
    
    def get_default_params(self) -> dict:
        return {
            'param1': 0.1,
            'param2': 0.5,
        }
    
    def __init__(self, **params):
        defaults = self.get_default_params()
        defaults.update(params)
        self.params = self.validate_params(defaults)
    
    def validate_params(self, params: dict) -> dict:
        if not (0 <= params['param1'] <= 1):
            raise ValueError("param1 must be in [0, 1]")
        return params
    
    # AGGLOMERATION
    def compute_merged_porosity(self, v_dry1, poro1, v_dry2, poro2, ...):
        V_solid_1 = v_dry1 * (1.0 - poro1) if not np.isnan(poro1) else v_dry1
        V_solid_2 = v_dry2 * (1.0 - poro2) if not np.isnan(poro2) else v_dry2
        V_solid_merged = V_solid_1 + V_solid_2
        
        # Eigene Porenlogik hier
        V_pore_merged = ... 
        
        v_dry_merged = V_solid_merged + V_pore_merged
        poro_merged = V_pore_merged / v_dry_merged if v_dry_merged > 0 else np.nan
        
        return v_dry_merged, poro_merged
    
    # BREAKAGE
    def compute_fragment_porosity(self, parent_porosity, fragment_volume, parent_volume, ...):
        # Eigene Breakage-Logik hier
        return parent_porosity  # Oder modifizieren
    
    # NUCLEATION
    def compute_nucleation_porosity(self, v_solid, v_liquid, nucleation_params, ...):
        # Eigene Nucleation-Logik hier
        default_porosity = nucleation_params.get('default_porosity', 0.4)
        v_dry = v_solid / (1.0 - default_porosity)
        return v_dry, default_porosity
```

### Schritt 2: Factory registrieren

In `kernels/porosity_growth/__init__.py`:

```python
from .my_custom_porosity import MyCustomPorosityKernel

POROSITY_GROWTH_KERNELS['my_custom_porosity'] = MyCustomPorosityKernel
```

Fertig! Der Kernel ist jetzt verfügbar via:

```python
kernel = get_porosity_growth_kernel('my_custom_porosity', param1=0.2)
```

---

## Testing

Test-Skript ausführen:

```bash
cd C:\Users\ericb\Documents\GitHub\PSD_opt\mcpbe\src\wmcpbe
python Trials/test_kernel_framework.py
```

Das Test-Skript überprüft:
1. Alle Module können importiert werden
2. Factory-Funktionen erstellen korrekte Instanzen
3. Berechnungsmethoden liefern plausible Ergebnisse
4. Parameter-Validierung funktioniert

---

## Migration von bestehendem Code

### Alte COLEVAL-Werte → neue Kernel

| COLEVAL | Alter Code | Neuer Kernel |
|---------|------------|--------------|
| 1 | Shear (Chin) | `shear_chin1998` |
| 2 | Brownian | `brownian_tsouris1995` |
| 3 | Constant | `constant` |
| 4 | Sum | `sum` |

### Alte BREAKRVAL-Werte → neue Kernel

| BREAKRVAL | Alter Code | Neuer Kernel |
|-----------|------------|--------------|
| 1,2,4,5 | Power-Law Varianten | `power_law` |

### Porositätslogik migrieren

**Alt (inline in `_do_one_agg()`):**
```python
# Hardcoded logic
if np.isnan(poro_i):
    V_solid_i = Vi_dry
else:
    V_solid_i = Vi_dry * (1.0 - poro_i)
# ...
```

**Neu (über Kernel):**
```python
kernel = self.kernel_manager.porosity_growth_kernel
v_dry_merged, poro_merged = kernel.compute_merged_porosity(
    v_dry1=Vi_dry, poro1=poro_i,
    v_dry2=Vj_dry, poro2=poro_j,
    collision_energy=E_coll,
    solver=self
)
```

---

## Nächste Schritte

1. **Integration in MCPBESolver**: Kernel-Bindung in `__init__()` und `_initialize_kernels()`
2. **Bestehende Logik migrieren**: `_do_one_agg()`, `_break_apply_and_maintain()`, `_create_nucleated_particle_copy()` auf Kernel delegieren
3. **Validation**: Bestehende Test-Cases mit neuer Architektur ausführen (Regressionstests)
4. **Alte Logik entfernen**: Nach erfolgreicher Validierung inline-Porositätslogik löschen

---

## Literatur / Referenzen

Für die Implementierung der Kernel wurden folgende Quellen verwendet:

- **Shear-induced agglomeration**: Chin et al., "Shear-induced flocculation in stirred tanks", 1998
- **Brownian diffusion**: Tsouris et al., "Brownian diffusion as controlling mechanism", 1995
- **Liquid bridge models**: Iveson et al., "Non-inertial droplet-particle collisions", 2001
- **Weibull breakage**: Tardos et al., "Stress-induced compaction", 1997
- **Porosity consolidation**: 10.1103/PhysRevLett.64.2727
- **Incomplete mixing**: Liu et al., "Wet granulation: mechanism and modeling", 2009

---

## Autor & Kontakt

- Implementierung: WMCPBE Development Team
- Framework-Design: Basierend auf Kernel Registry Pattern
- PorosityGrowthKernel erweitert: Juli 2026 (einheitliche Physik für Agg/Break/Nuc)

---

## Changelog

### Juli 2026: PorosityGrowthKernel Refactoring

**Änderungen:**
- ✅ Basisklasse um `compute_fragment_porosity()` erweitert
- ✅ `VolumeMixingKernel`: compute_nucleation_porosity() gibt Default 0.4 zurück
- ✅ `IncompleteMixingKernel`: compute_fragment_porosity() mit Porenkollaps
- ✅ Dokumentation aktualisiert (KERNEL_UEBERSICHT.md, README.md)

**Motivation:**
- Einheitliche Physik über Agglomeration, Breakage, Nucleation
- Weniger Code-Duplikation
- Einfachere Erweiterung um neue Porositätsmodelle

**Migration:**
- Keine Breaking Changes (rückwärtskompatibel)
- Default-Kernel bleibt `volume_mixing`
- Bestehende Simulationen laufen unverändert
