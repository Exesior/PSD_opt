# Kernel-Übersicht: WMCPBE Solver

**Stand:** Juli 2026

## 📑 Inhaltsverzeichnis

1. [Agglomeration-Kernels](#agglomeration-kernels)
2. [Breakage-Kernels](#breakage-kernels)
3. [Porosity Growth-Kernels](#porosity-growth-kernels)
4. [Compression-Kernels (REMOVED)](#compression-kernels-removed)
5. [Liquid Distribution-Kernels](#liquid-distribution-kernels)
6. [Agglomeration Acceptance-Kernels](#agglomeration-acceptance-kernels)
7. [Continuous Processes-Kernels](#continuous-processes-kernels)
8. [Syntax für Flüssigkeit & Porosität](#syntax-für-flüssigkeit--porosität)
9. [Alle Kernel im Überblick](#alle-kernel-im-überblick)

---

## Agglomeration-Kernels

| Kernel | Formel | Parameter | Flüssigkeit? | Porosität? |
|--------|--------|-----------|--------------|------------|
| **shear_chin1998** | `β = CORR_BETA × G × (r1+r2)³` | `corr_beta`, `g` | ❌ | ❌ |
| **brownian_tsouris1995** | `β = 2kT(r1+r2)² / (3μ r1 r2)` | `corr_beta`, `temperature`, `viscosity` | ❌ | ❌ |
| **constant** | `β = const` | `beta` | ❌ | ❌ |
| **sum** | `β = r1 + r2` | - | ❌ | ❌ |
| **liquid_bridge** ⭐ | `β = β_shear × f_bridge × f_capillary` | `corr_beta`, `g`, `optimal_saturation`, `saturation_width`, `liquid_enhancement`, `surface_tension`, `contact_angle` | ✅ **JA!** | ✅ **JA!** |

### Details

#### `shear_chin1998` (Standard)
```python
kernel = get_aggregation_kernel('shear_chin1998', corr_beta=1e-3, g=1000)
```
- **Anwendung**: High-shear Granulatoren, Rührwerke
- **Physik**: Turbulente Scherung dominiert Kollisionen
- **Einschränkung**: Ignoriert Flüssigkeitseffekte

#### `liquid_bridge` (Erweitert) ⭐
```python
kernel = get_aggregation_kernel('liquid_bridge',
    corr_beta=1e-3,
    g=1000,
    optimal_saturation=0.5,      # Max. Brückenbildung bei 50% Sättigung
    saturation_width=0.2,        # Breite des Optimums
    liquid_enhancement=2.0,      # Stärke der Kapillar-Verstärkung
    surface_tension=0.072,       # N/m (Wasser)
    contact_angle=0.0            # rad (perfekte Benetzung)
)
```
- **Anwendung**: Feuchte Granulation, Spray-Agglomeration
- **Physik**: 
  - **Brückenbildung**: Maximum bei intermediärer Sättigung (Gauß-Funktion)
  - **Kapillarkräfte**: Logarithmische Verstärkung mit externer Flüssigkeit
- **Zugriff auf Solver-Zustand**:
  ```python
  s1 = solver.saturation[particle1_idx]           # Interne Sättigung
  v_liq_ext1 = solver.get_V_liquid_external(idx)  # Externe Flüssigkeit
  ```

---

## Breakage-Kernels

| Kernel | Formel | Parameter | Flüssigkeit? | Porosität? |
|--------|--------|-----------|--------------|------------|
| **power_law** (Standard) | `S = P1 × G × V^P2` | `p1`, `p2`, `g`, `breakrval` | ❌ | ❌ |
| **powerlaw_rumpf** ⭐ | `S = (P1 × G × V^P2) / σ` | `p1`, `p2`, `g`, `breakrval`, `k`, `alpha`, `gamma`, `delta`, `x_s` | ✅ **JA!** | ✅ **JA!** |

> **Achtung, geänderte Bedeutung von `P2`.** Bis 15.08.2026 rechneten beide Kernel
> `S = P1 × G^P2 × V^(pl_v/3)` — `P2` war der *Scher*-Exponent und ein separates `pl_v`
> der Volumen-Exponent. Das war eine Verwechslung mit `BREAKFVAL` (dort heißen die
> Parameter der *Fragmentverteilung* tatsächlich `pl_v`/`pl_q`). Jetzt gilt wieder die
> Referenzformel: **`P2` ist der Volumen-Exponent, `G` geht linear ein.** `pl_v`/`pl_q`
> werden von den Ratenkerneln **abgelehnt** (`ValueError` mit Migrationshinweis);
> die Fragmentverteilung steuerst du über `solver.break_frag_v` / `break_frag_q`.
> Alte `P1`-Kalibrierungen sind dadurch ungültig, s. `docs/Breakage_Kernel_Formula_Divergenz.md`.

**Hinweis:** `stress_based` wurde entfernt (nicht verwendet).

### Details

#### `power_law` (Legacy-kompatibel)
```python
kernel = get_breakage_kernel('power_law',
    p1=3e-2,          # = pl_P1 in Legacy
    p2=1.0,           # = pl_P2 in Legacy (VOLUMEN-Exponent)
    g=1000,           # = G in Legacy
    breakrval=1,      # Modell-Variante (1-4)
)
```
- **BREAKRVAL-Varianten** (identisch zu `pbe_core.func.jit_kernel_break.calc_break_rate_1d`):
  - `1`: Konstant (`S = P1`)
  - `2`: Linear im Volumen (`S = P1 × V`)
  - `3`: Potenzgesetz Pandy & Spielmann (`S = P1 × G × V^P2`)
  - `4`: identisch zu `3` in 1D
- **`breakrval=5` existiert nicht** und wird abgelehnt — der Zweig war eine
  Fehlübernahme aus dem 5-wertigen `BREAKFVAL`-Schalter.

#### `stress_based` (Physikalisch fundiert) ⭐
```python
kernel = get_breakage_kernel('stress_based',
    critical_stress=1e6,      # Pa (Partikelstärke)
    weibull_modulus=2.0,      # Streuungsparameter
    g=1000,                   # Scherrate
    stress_size_exp=1.0,      # σ ~ d^exp
    porosity_weakening=2.0    # σ_eff = σ_crit × (1-poro)^exp
)
```
- **Anwendung**: Spröde Materialien, poröse Granulate
- **Physik**:
  - Weibull-Statistik für Bruchwahrscheinlichkeit
  - Größere Partikel brechen leichter (höhere Spannung)
  - Porosität reduziert effektive Festigkeit
- **Zugriff auf Solver-Zustand**:
  ```python
  poro = solver.porosity[particle_idx]
  sigma_eff = sigma_crit * (1.0 - poro) ** poro_weakening
  ```

#### `powerlaw_rumpf` (Rumpf-Theorie mit Sättigung) ⭐⭐
```python
kernel = get_breakage_kernel('powerlaw_rumpf',
    # PowerLaw Basis
    p1=3e-2,          # Pre-factor
    p2=1.0,           # Shear exponent
    g=1000,           # Scherrate [1/s]
    breakrval=4,      # Modell-Variante (1-5)
    pl_v=2.0,         # Volumen-Exponent (für BREAKRVAL=4)
    
    # Rumpf Stärke-Modell
    k=2.5,            # Fittingparameter trocken [2.2-2.8]
    alpha=1.15,       # Fittingparameter nass [1.0-1.33]
    gamma=0.072,      # Oberflächenspannung [N/m] (Wasser)
    delta=0.0,        # Benetzungswinkel [rad] (0 = perfekte Benetzung)
    x_s=10e-6         # Sauterdurchmesser [m] (oder None für Berechnung aus X0)
)
```
- **Formel**: `S(V) = P1 × (1/σ) × G^P2 × V^alpha`
- **Bruchfestigkeit σ (3 Sättigungs-Regime nach Rumpf)**:
  - **Trocken (S < 0.3)**: `σ = (1-poro)/poro × k × γ / x_s`
  - **Übergang (0.3 ≤ S ≤ 0.8)**: Lineare Interpolation
  - **Nass (S > 0.8)**: `σ = 6×α × (1-poro)/poro × γ×cos(δ) / x_s × S`
- **Anwendung**: Feuchte Granulation, kapillar-brücken-dominierte Festigkeit
- **Physik**:
  - Basierend auf Rumpf's Theorie für Zugfestigkeit von Agglomeraten
  - Höhere Sättigung → höhere Festigkeit → niedrigere Bruchrate
  - Porosität reduziert Festigkeit (`(1-poro)/poro` Faktor)
  - Nass: Zusätzliche Verstärkung durch Flüssigkeitsfilm
- **Besonderheiten**:
  - **Vollkörper (poro=NaN)**: PowerLaw ohne σ-Korrektur
  - **x_s**: Wird automatisch aus `solver.X0` berechnet (Sauter-Durchmesser)
  - **Bei Verteilungen**: `x_s = Σ(d³) / Σ(d²)` (Volumen-Oberfläche-Mittel)
  - ⚠️ **Achtung**: Annahme monodisperser Startpartikel für x_s nicht immer gerechtfertigt!
- **Zugriff auf Solver-Zustand**:
  ```python
  poro = solver.porosity[particle_idx]
  sat = solver.saturation[particle_idx]
  x_s = self._compute_sauter_diameter(solver)  # Aus X0
  ```

---

## Porosity Growth-Kernels

| Kernel | Agglomeration | Breakage | Nucleation | Parameter |
|--------|---------------|----------|------------|-----------|
| **volume_mixing** (Standard) | Volumina additiv | Vererbung | Default: 0.4 | - |
| **incomplete_mixing** ⭐ | +Trapped Pores | Porenkollaps | Default: 0.4 | `trapped_pore_fraction`, `collapse_factor` |

### 🔹 WICHTIG: Einheitliche Physik über alle Operationen

Jeder PorosityGrowthKernel implementiert **DREI konsistente Methoden**:

```python
class PorosityGrowthKernel:
    
    def compute_merged_porosity(self, ...) -> tuple[float, float]:
        """AGGLOMERATION: Zwei Partikel mergen"""
        pass
    
    def compute_fragment_porosity(self, ...) -> float:
        """BREAKAGE: Fragment-Porosität aus Parent"""
        pass
    
    def compute_nucleation_porosity(self, ...) -> tuple[float, float]:
        """NUCLEATION: Porosität neu gebildeter Partikel"""
        pass
```

**Alle drei basieren auf DEMSELBEN physikalischen Modell!**

---

### Details

#### `volume_mixing` (Einfach, Standard)

**Physikalische Annahme:** Poren und Feststoff sind ideal mischbar, keine eingeschlossenen Volumina.

```python
kernel = get_porosity_growth_kernel('volume_mixing')
```

| Operation | Logik | Mathematik |
|-----------|-------|------------|
| **Agglomeration** | Volumina addieren sich linear | `V_solid_new = V_solid₁ + V_solid₂`<br>`V_pore_new = V_pore₁ + V_pore₂`<br>`poro_new = V_pore_new / V_dry_new` |
| **Breakage** | Fragmente erben Parent-Porosität | `poro_frag = poro_parent` |
| **Nucleation** | **Hardcodiert: 0.4** (im Kernel!) | `poro_new = 0.4`<br>`v_dry = v_solid / (1 - 0.4)` |

**Warum 0.4 in `volume_mixing`?**
- Mit volume_mixing ist Porosität NUR additiv (`V_pore_new = V_pore₁ + V_pore₂`)
- Bei Vollkörpern (poro=NaN) würde **nie** Porosität entstehen!
- **Hardcodierter Default 0.4** in `VolumeMixingKernel.compute_nucleation_porosity()`
- Andere Kernel können eigene Defaults haben (z.B. `incomplete_mixing`: konfigurierbar)

**Code-Beispiel:**
```python
# AGG:
V_solid_merged = V_solid_1 + V_solid_2
V_pore_merged  = V_pore_1 + V_pore_2
poro_merged    = V_pore_merged / (V_solid_merged + V_pore_merged)

# BREAK:
poro_frag = poro_parent  # Einfach vererben

# NUC:
poro_new = 0.4  # Ermöglicht Porositätsbildung aus Vollkörpern!
v_dry = v_solid / (1.0 - 0.4)
```

---

#### `incomplete_mixing` (Erweitert) ⭐

**Physikalische Annahme:** Beim Zusammenkleben entstehen eingeschlossene Poren durch unvollständige Koaleszenz.

```python
kernel = get_porosity_growth_kernel('incomplete_mixing',
    trapped_pore_fraction=0.1,   # 10% zusätzliche Poren
    collapse_factor=0.1          # 10% Porenkollaps beim Bruch
)
```

| Operation | Logik | Mathematik |
|-----------|-------|------------|
| **Agglomeration** | +Trapped Pores durch eingeschlossene Volumina | `V_pore_new = V_pore₁ + V_pore₂ + ΔV_pore`<br>`ΔV_pore = f_trap × min(V_dry₁, V_dry₂)` |
| **Breakage** | Porenkollaps durch mechanischen Stress | `poro_frag = poro_parent × (1 - collapse_factor)` |
| **Nucleation** | Default: 0.4 (konsistent mit volume_mixing) | `poro_new = 0.4` |

**Physikalische Mechanismen:**
1. **Contact porosity**: Poren an Partikel-Partikel-Grenzfläche
2. **Surface roughness**: Mikro-Voids durch nicht-glatte Oberflächen
3. **Viscous limitation**: Unzureichende Zeit für vollständige Restrukturierung

**Energieabhängigkeit (optional):**
- Höhere Kollisionsenergie → bessere Koaleszenz → weniger trapped pores
- Höhere Bruchenergie → mehr Porenkollaps → dichtere Fragmente

**Code-Beispiel:**
```python
# AGG:
V_solid_merged = V_solid_1 + V_solid_2
V_pore_base    = V_pore_1 + V_pore_2
V_pore_trapped = f_trap × min(V_dry_1, V_dry_2)
V_pore_merged  = V_pore_base + V_pore_trapped
poro_merged    = V_pore_merged / (V_solid_merged + V_pore_merged)

# BREAK:
collapse_factor = 0.1  # 10% Porenreduktion
poro_frag = poro_parent * (1.0 - collapse_factor)

# NUC:
poro_new = 0.4  # Konsistent mit volume_mixing
```

---

## Compression-Kernels (REMOVED) ⚠️

**Das Modul `kernels/compression/` wurde entfernt!**

Verwenden Sie stattdessen **ContinuousProcessesHandler** mit den Kerneln aus `kernels/continuous_processes/`.

### Migration Guide

**ALT (nicht mehr verfügbar):**
```python
solver = MCPBESolver(
    compression_kernel_name='exponential_decay',
    compression_kernel_params={'rate': 0.02, 'min_porosity': 0.3},
)
```

**NEU (empfohlen):**
```python
solver = MCPBESolver(...)  # Ohne compression_kernel_name

# ContinuousProcessesHandler erstellen
solver.create_continuous_processes_handler(
    enabled=True,
    compression_rate=0.02,    # 1/s
    min_porosity=0.3,         # Minimale Porosität
    k_int=1e12,               # Optional: Liquid Internalization
)
```

### Warum wurde das Modul entfernt?

1. **Redundanz**: Die Funktionalität ist in `continuous_processes` enthalten
2. **Konsistenz**: Alle zeitkontinuierlichen Prozesse in einem Modul
3. **Erweiterbarkeit**: Einfacher neue Prozesse hinzufügen (Operator Splitting)

### Verfügbare Ersatz-Kernel

| Prozess | Kernel | Modul |
|---------|--------|-------|
| Porositätskompression | `porosity_compression` | `continuous_processes` |
| Flüssigkeitsaufnahme | `liquid_internalization` | `continuous_processes` |
| Poren-Einschluss (Agg) | `liq_internalisation_agglomeration` | `continuous_processes` |

---

## Liquid Distribution-Kernels

| Kernel | Auswahlregel | Parameter |
|--------|--------------|-----------|
| **uniform_weighted** (Standard) | `P(i) ∝ W[i]` | - |
| **surface_weighted** ⭐ | `P(i) ∝ Oberfläche[i]` | - |
| **saturation_preferential** ⭐ | `P(i) ∝ (1 - S[i])` | `saturation_bias` |

### Details

#### `uniform_weighted` (Standard)
```python
kernel = get_liquid_distribution_kernel('uniform_weighted')
```
- **Regel**: Partikel mit höherem Weight wahrscheinlicher
- **Physik**: Größere Partikel treffen öfter Tropfen

#### `surface_weighted` (Oberflächen-basiert) ⭐
```python
kernel = get_liquid_distribution_kernel('surface_weighted')
```
- **Regel**: `P(i) ∝ r[i]²` (Oberfläche)
- **Physik**: Größere Oberfläche → höhere Trefferwahrscheinlichkeit

#### `saturation_preferential` (Sättigungs-präferentiell) ⭐
```python
kernel = get_liquid_distribution_kernel('saturation_preferential',
    saturation_bias=2.0  # Stärke der Präferenz für ungesättigte Partikel
)
```
- **Regel**: `P(i) ∝ (1 - S[i])^bias`
- **Physik**: Ungesättigte Partikel nehmen Flüssigkeit preferenziell auf
- **Zugriff auf Solver-Zustand**:
  ```python
  sat = solver.saturation[target_idx]
  prob = (1.0 - sat) ** bias
  ```

---

## Syntax für Flüssigkeit & Porosität

### 🔹 Porosität (`solver.porosity`)

```python
# Zugriff während Kernel-Berechnung
def compute_beta(self, r1, r2, particle1_idx=None, particle2_idx=None, solver=None):
    if solver is not None and particle1_idx is not None:
        poro1 = solver.porosity[particle1_idx]
        
        # Unterscheidung: Vollkörper vs. porös
        if np.isnan(poro1):
            # Vollkörper: keine Poren
            pass
        else:
            # Poröses Partikel: poro ∈ [0, 1]
            effective_radius = r1 * (1.0 + poro1)  # Beispiel
```

**Spezialfall Vollkörper**: `porosity = np.nan`

---

### 🔹 Flüssigkeit (`solver.liquid_volume`, `solver.saturation`)

```python
# Interne Flüssigkeit (in Poren)
liq_vol = solver.liquid_volume[particle_idx]  # [m³]
sat = solver.saturation[particle_idx]         # [0, 1] oder np.nan

# Externe Flüssigkeit (auf Oberfläche)
v_liq_ext = solver.get_V_liquid_external(particle_idx)  # [m³]

# Unterscheidung: Nass vs. Trocken
if hasattr(solver, 'saturation'):
    sat = solver.saturation[idx]
    if np.isnan(sat):
        # Trocken / Vollkörper
        pass
    elif sat < 1.0:
        # Ungesättigt
        pass
    else:
        # Gesättigt
        pass
```

---

---

## Agglomeration Acceptance-Kernels

Diese Kernel entscheiden, ob eine Kollision zur Agglomeration führt oder ob die Partikel abprallen.

| Kernel | Kriterium | Parameter | Flüssigkeit? | Porosität? |
|--------|-----------|-----------|--------------|------------|
| **stokes_krit** (Standard) | `St < St_crit` → akzeptiert | `U_coll`, `binder_viscosity`, `rho_solid`, `rho_liquid`, `h_a` | ✅ | ❌ |
| **fittable** | `rng() < u_acc` | `u_acc` | ❌ | ❌ |

### Details

#### `stokes_krit` (Braumann et al. 2007)
```python
kernel = get_agglomeration_acceptance_kernel('stokes_krit',
    U_coll=1.0,           # m/s (Kollisionsgeschwindigkeit)
    binder_viscosity=0.1, # Pa·s (Binder-Viskosität)
    rho_solid=2500.0,     # kg/m³ (Feststoffdichte)
    rho_liquid=1000.0,    # kg/m³ (Flüssigkeitsdichte)
    h_a=500e-9            # m (minimale Filmdicke)
)
```
- **Physik**: Stokes-Zahl vergleicht Trägheit mit viskoser Dämpfung
- **Anwendung**: Nassgranulation mit viskosem Bindemittel
- **Referenz**: Braumann, A.P., et al. (2007)

#### `fittable` (Einfach)
```python
kernel = get_agglomeration_acceptance_kernel('fittable',
    u_acc=0.8  # Akzeptanzwahrscheinlichkeit [0-1]
)
```
- **Physik**: Konstante Akzeptanzrate (fitting parameter)
- **Anwendung**: Vereinfachte Modelle, Kalibrierung

---

## Continuous Processes-Kernels

Diese Kernel modellieren zeitkontinuierliche Prozesse (via Operator Splitting).

| Kernel | Prozess | Formel | Parameter |
|--------|---------|--------|-----------|
| **porosity_compression** | Porositätskompression | `p(t) = p_min + (p₀-p_min) × exp(-rate×t)` | `rate`, `min_porosity` |
| **liquid_internalization** | Flüssigkeitsaufnahme | `dl_intern/dt = k × l_ex × (v_pore - l_intern)` | `k_int` |
| **liq_internalisation_agglomeration** | Poren-Einschluss bei Agg | Braumann 2007 Formel | - |

### Details

#### `porosity_compression`
```python
from wmcpbe.kernels.continuous_processes import get_continuous_kernel

kernel = get_continuous_kernel('porosity_compression',
    rate=0.02,          # 1/s
    min_porosity=0.3    # Minimale Restporosität
)
```
- **Anwendung**: Zeitabhängige Verdichtung unter Scherung

#### `liquid_internalization`
```python
kernel = get_continuous_kernel('liquid_internalization',
    k_int=1e12  # 1/(m³·s) (Internalisierungsrate)
)
```
- **Physik**: Kapillar-getriebene Porenfüllung (Braumann et al. 2007)
- **Zeitskala**: 
  - Schnell: k_int > 1e9 (sofortige Füllung)
  - Mittel: k_int ≈ 1e6-1e8 (Sekunden bis Minuten)

#### `liq_internalisation_agglomeration`
```python
kernel = get_continuous_kernel('liq_internalisation_agglomeration')
```
- **Physik**: Einschluss externer Flüssigkeit bei Partikelkontakt
- **Formel**: Siehe Braumann et al. (2007), Eq. 15

---

## 🔹 Kombination in Kernels

**Beispiel: Liquid Bridge mit Porosität**

```python
class LiquidBridgeKernel(AggregationKernel):
    def compute_beta(self, r1, r2, particle1_idx, particle2_idx, solver):
        # Basis-Scherung
        beta_shear = self.corr_beta * self.g * (r1 + r2)**3
        
        # Prüfe ob Solver Flüssigkeits-Zustand hat
        if not hasattr(solver, 'saturation'):
            return beta_shear  # Graceful degradation
        
        # Sättigung lesen
        s1 = solver.saturation[particle1_idx]
        s2 = solver.saturation[particle2_idx]
        
        # Vollkörper? → Keine Brückenbildung
        if np.isnan(s1) or np.isnan(s2):
            return beta_shear
        
        # Brückenfaktor (Maximum bei 50% Sättigung)
        s_eff = (s1 + s2) / 2.0
        bridge_factor = exp(-((s_eff - 0.5)**2) / (2 * 0.2**2))
        
        # Externe Flüssigkeit
        v_liq_ext1 = solver.get_V_liquid_external(particle1_idx)
        v_liq_ext2 = solver.get_V_liquid_external(particle2_idx)
        liquid_factor = 1.0 + alpha * log1p((v_liq_ext1 + v_liq_ext2) / v_ref)
        
        return beta_shear * bridge_factor * liquid_factor
```

---

## 📊 Zusammenfassung

| Kernel-Typ | Unterstützt Flüssigkeit? | Unterstützt Porosität? |
|------------|-------------------------|------------------------|
| **Agglomeration** | ✅ (liquid_bridge) | ✅ (liquid_bridge) |
| **Breakage** | ✅ (powerlaw_rumpf) | ✅ (stress_based, powerlaw_rumpf) |
| **Porosity Growth** | ✅ (incomplete_mixing) | ✅ (alle!) |
| **Compression** | ❌ | ✅ | ⚠️ **REMOVED** - Use `continuous_processes` |
| **Liquid Distribution** | ✅ (alle!) | ✅ (saturation_preferential) |
| **Agg. Acceptance** | ✅ (stokes_krit) | ❌ |
| **Continuous Processes** | ✅ (liquid_internalization) | ✅ (porosity_compression) |

---

## 🎯 Empfehlungen

| Anwendungsszenario | Empfohlene Kernel-Kombination |
|--------------------|-------------------------------|
| **Trockene Agglomeration** | `shear_chin1998` + `volume_mixing` + `exponential_decay` |
| **Feuchte Granulation** | `liquid_bridge` + `volume_mixing` + `saturation_preferential` |
| **Spröde Materialien** | `stress_based` + `volume_mixing` |
| **High-Shear mit Bruch** | `shear_chin1998` + `power_law(breakrval=4)` |
| **Nasse Sprühgranulation** | `liquid_bridge` + `incomplete_mixing` + `saturation_preferential` |
| **Porositätsentwicklung studieren** | `incomplete_mixing` (zeigt Porenentstehung/-kollaps) |

---

## 📚 Porosity Growth Kernel Vergleich

| Eigenschaft | `volume_mixing` | `incomplete_mixing` |
|-------------|-----------------|---------------------|
| **Agglomeration** | Additiv | +Trapped Pores |
| **Breakage** | Vererbung | Porenkollaps |
| **Nucleation** | 0.4 Default | 0.4 Default |
| **Parameter** | Keine | `trapped_pore_fraction`, `collapse_factor` |
| **Porositätsentstehung** | Nur aus vorhandenen Poren | Auch aus Vollkörpern (+ΔV_pore) |
| **Physikalische Komplexität** | Niedrig | Mittel |
| **Rückwärtskompatibilität** | ✅ Vollständig | ⚠️ Neue Parameter |

---

## ⚠️ Wichtige Hinweise

### 1. V_flat Semantik

```
V_flat Struktur für dim=1:
┌─────────────────────────────────────────────────────────────┐
│ V_flat[0, :]  =  V_solid  (Feststoffvolumen, MASSEERHALTUNG!) │
│ V_flat[1, :]  =  V_dry    (Gesamtvolumen = V_solid + V_pore)  │
├─────────────────────────────────────────────────────────────┤
│ Relation: V_solid = V_dry × (1 - porosity)                   │
│          V_dry    = V_solid / (1 - porosity)                 │
└─────────────────────────────────────────────────────────────┘
```

### 2. Porosity NaN Handling

- **Vollkörper**: `porosity = np.nan` (nicht 0!)
- **Porös**: `porosity ∈ [0, 1]`
- **Prüfung**: `if np.isnan(porosity): ...`

### 3. Nucleation Default-Porosität

- **volume_mixing**: `0.4` (sonst nie Porositätsbildung!)
- **incomplete_mixing**: `0.4` (konsistent, aber kann auch NaN sein)

### 4. Breakage Fragment-Porosität

- **volume_mixing**: `poro_frag = poro_parent` (Vererbung)
- **incomplete_mixing**: `poro_frag = poro_parent × (1 - collapse_factor)` (Kollaps)

---

## Alle Kernel im Überblick

### Komplette Kernel-Liste nach Kategorie

#### 1. Aggregation (5 Kernel)
| Name | Beschreibung | Flüssigkeit | Porosität |
|------|--------------|-------------|----------|
| `shear_chin1998` | Scherungsgetrieben (Chin 1998) | ❌ | ❌ |
| `brownian_tsouris1995` | Brownsche Bewegung | ❌ | ❌ |
| `constant` | Konstante Rate | ❌ | ❌ |
| `sum` | Summenkernel | ❌ | ❌ |
| `liquid_bridge` ⭐ | Mit Flüssigkeitsbrücken | ✅ | ✅ |

#### 2. Breakage (3 Kernel)
| Name | Beschreibung | Flüssigkeit | Porosität |
|------|--------------|-------------|----------|
| `power_law` | Potenzgesetz (Legacy) | ❌ | ❌ |
| `stress_based` ⭐ | Weibull-Spannungsmodell | ❌ | ✅ |
| `powerlaw_rumpf` ⭐⭐ | Rumpf-Theorie mit Sättigung | ✅ | ✅ |

#### 3. Porosity Growth (3 Kernel)
| Name | Beschreibung | Agg | Break | Nuc |
|------|--------------|-----|-------|-----|
| `volume_mixing` | Volumina additiv | + | Vererbung | 0.4 |
| `incomplete_mixing` ⭐ | +Trapped Pores | +ΔV | Kollaps | 0.4 |
| `cone_model` | Kegelmodell | Geometrisch | - | 0.4 |

#### 4. Compression (❌ ENTFERNT)
| Name | Beschreibung | Status |
|------|--------------|--------|
| `exponential_decay` | Exponentieller Zerfall | ❌ REMOVED |
| `stress_compaction` | Spannungsgetrieben | ❌ REMOVED |

⚠️ **Achtung**: Modul `kernels/compression/` wurde entfernt! 
Verwenden Sie `kernels/continuous_processes/` mit `porosity_compression` Kernel.

#### 5. Liquid Distribution (3 Kernel)
| Name | Auswahlregel | Bias |
|------|--------------|------|
| `uniform_weighted` | `P ∝ W` | - |
| `surface_weighted` | `P ∝ r²` | - |
| `saturation_preferential` ⭐ | `P ∝ (1-S)^bias` | konfigurierbar |

#### 6. Agglomeration Acceptance (2 Kernel)
| Name | Kriterium | Physik |
|------|-----------|--------|
| `stokes_krit` | Stokes-Zahl | Braumann et al. 2007 |
| `fittable` | Konstante Wahrscheinlichkeit | Fitting-Parameter |

#### 7. Continuous Processes (3 Kernel)
| Name | Prozess | Parameter |
|------|---------|----------|
| `porosity_compression` | Porositätsabbau | `rate`, `min_porosity` |
| `liquid_internalization` | Kapillare Aufnahme | `k_int` |
| `liq_internalisation_agglomeration` | Poren-Einschluss | - |

---

## 🎯 Empfehlungen für Anwendungsszenarien

| Szenario | Empfohlene Kernel-Kombination |
|----------|-------------------------------|
| **Trockene Agglomeration** | `shear_chin1998` + `volume_mixing` + `create_continuous_processes_handler(compression_enabled=True, compression_rate=0.02)` |
| **Feuchte Granulation (einfach)** | `liquid_bridge` + `volume_mixing` + `uniform_weighted` |
| **Feuchte Granulation (vollständig)** | `liquid_bridge` + `powerlaw_rumpf` + `incomplete_mixing` + `saturation_preferential` + `stokes_krit` + `liquid_internalization` |
| **Spröde Materialien** | `stress_based` + `volume_mixing` |
| **High-Shear mit Bruch** | `shear_chin1998` + `power_law(breakrval=4)` |
| **Nasse Sprühgranulation** | `liquid_bridge` + `incomplete_mixing` + `saturation_preferential` + `liq_internalisation_agglomeration` |
| **Porositätsentwicklung studieren** | `incomplete_mixing` (zeigt Porenentstehung/-kollaps) |
| **Validierung/Tests** | `constant` + `volume_mixing` (analytische Lösungen) |

---

## 🔧 Eigene Porosity Growth Kernel implementieren

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
    
    def compute_merged_porosity(self, v_dry1, poro1, v_dry2, poro2, ...):
        # Eigene Agglomeration-Logik hier
        V_solid_merged = ...
        V_pore_merged = ...
        poro_merged = V_pore_merged / (V_solid_merged + V_pore_merged)
        return V_solid_merged + V_pore_merged, poro_merged
    
    def compute_fragment_porosity(self, parent_porosity, ...):
        # Eigene Breakage-Logik hier
        return parent_porosity * 0.9  # Beispiel: 10% Kollaps
    
    def compute_nucleation_porosity(self, v_solid, v_liquid, ...):
        # Eigene Nucleation-Logik hier
        poro = 0.4  # Default
        return v_solid / (1 - poro), poro
```

---

*Stand: Juli 2026*
