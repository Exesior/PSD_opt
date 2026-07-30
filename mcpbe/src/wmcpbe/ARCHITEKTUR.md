# WMCPBE Architektur-Dokumentation

## Übersicht

WMCPBE (Weighted Monte Carlo Population Balance Equation) ist ein modulares Framework zur Simulation von Partikelpopulationsdynamiken mittels DSMC (Direct Simulation Monte Carlo). Das System simuliert Agglomeration, Breakage, Nukleation und Kompression von porösen Partikeln.

---

## 1. Gesamtsystem-Architektur

### 1.1 Klassenhierarchie (MRO - Method Resolution Order)

```
MCPBESolver(MCPBEPost, MCPBEBreak, MCPBEAgg, MCPBEBase, ReconstructionMixin)
│
├── MCPBEPost          → Post-Processing (Momente, PSD, Statistiken)
├── MCPBEBreak         → Breakage-Physik (Zerfallsraten, Fragmentverteilung)
├── MCPBEAgg           → Agglomeration-Physik (Kollisionsfrequenzen)
├── MCPBEBase          → Kern-Solver (Partikelzustand, Control Volume, Hauptschleife)
└── ReconstructionMixin → Resampling/Rebinning (CAM, RS, 2PM, QMX)

Composition Pattern (Handler):
├── NucleationHandler  → Flüssigkeitszugabe während der Simulation
└── CompressionHandler → Porositätsreduktion über Zeit
```

### 1.2 Dateistruktur

```
wmcpbe/
├── mcpbe.py                  # Hauptklasse MCPBESolver (Factory für Handler)
├── mcpbe_base.py             # Basis-Solver mit DSMC-Kern
├── mcpbe_agg.py              # Agglomeration-Mixin
├── mcpbe_break.py            # Breakage-Mixin
├── mcpbe_post.py             # Post-Processing-Mixin
├── mcpbe_nucleation.py       # NucleationHandler + Config
├── mcpbe_compression.py      # CompressionHandler + Config
├── reconstruction_mixin.py   # Rekonstruktionsmethoden
├── fenwick_new.py            # Fenwick-Tree Sampler (O(log n))
├── kernel_integration.py     # KernelManager (zentrale Delegation)
└── kernels/                  # Modulare Physik-Kernels
    ├── base.py               # Abstrakte Basisklassen
    ├── aggregation/          # β(i,j) Kollisionsfrequenzen
    │   ├── shear_chin1998.py
    │   ├── liquid_bridge.py
    │   └── ...
    ├── breakage/             # S(V) Zerfallsraten
    │   ├── power_law.py
    │   └── stress_based.py
    ├── porosity_growth/      # Porositätsbildung bei Events
    │   └── volume_mixing.py
    ├── compression/          # Porositätsabbau über Zeit
    │   └── exponential_decay.py
    └── liquid_distribution/  # Partikelselektion für Nukleation
        └── uniform_weighted.py
```

---

## 2. Der DSMC-Algorithmus (Direct Simulation Monte Carlo)

### 2.1 Grundprinzip

DSMC verwendet **gewichtete Partikel** zur effizienten Simulation:

```
Computational Weight (W):
    W = N_physical / N_computational
    
Beispiel: 10.000 physikalische Partikel mit 500 computational particles
    → Durchschnittliches W = 20
    → Jedes computational particle repräsentiert 20 physikalische Partikel
```

### 2.2 Control Volume Management

Während der Agglomeration reduziert sich die Partikelanzahl. DSMC verdoppelt dann das Control Volume Vc:

```python
Wenn a_tot < a_ref / 2:  # Partikelanzahl unter 50%
    Vc *= 2.0
    Partikel duplizieren (W bleibt gleich)
    
Invariant: Sum(W) / Vc = konstant (= initiale Partikeldichte)
```

### 2.3 Hauptschleife (in `mcpbe_base.py::solve()`)

```python
while current_time <= t_total:
    # 1. Event-Typ bestimmen (Agg/Break/Mix)
    if process_type == "agglomeration":
        _do_one_agg()
    elif process_type == "breakage":
        _do_one_break()
    elif process_type == "mix":
        # Stochastische Entscheidung basierend auf Raten
        if random() < agg_rate / total_rate:
            _do_one_agg()
        else:
            _do_one_break()
    
    # 2. Zeit fortschreiben (propensity-basiert)
    current_time += dt_event
    
    # 3. Handler anwenden (nach MC-Event)
    if compression.enabled:
        compression.step(dt_event)
    if nucleation.enabled:
        nucleation.step(dt_event)
    
    # 4. Control Volume prüfen
    _maybe_double_control_volume()
    
    # 5. Snapshots speichern
    save_state_at_output_times()
```

---

## 3. Kernel-Framework (DETAILLIERTE ERKLÄRUNG)

### 3.1 Motivation für Kernel-Architektur

**Vorher (Legacy):**
- Physik in Solver fest codiert (z.B. `CORR_BETA`, `G`, `pl_P1` als Solver-Attribute)
- Keine Flexibilität für alternative Modelle
- Schwer erweiterbar

**Nachher (Kernel-Framework):**
- Modulare Physik-Modelle als austauschbare Komponenten
- Einheitliche Schnittstellen für alle Kernel-Typen
- Einfache Erweiterung durch neue Kernel-Klassen

### 3.2 Kernel-Hierarchie

```
KernelBase (ABC)
│
├── AggregationKernel       → compute_beta(r1, r2, ...)
├── BreakageKernel          → compute_rate(v_particle, ...)
├── PorosityGrowthKernel    → compute_merged_porosity(...)
├── CompressionKernel       → compute_porosity_decay(...)
└── LiquidDistributionKernel → select_target_particle(...)
```

### 3.3 KernelBase - Die abstrakte Basisklasse

**Datei:** `kernels/base.py`

```python
class KernelBase(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Eindeutiger Identifier, z.B. 'shear_chin1998'"""
        pass
    
    @property
    @abstractmethod
    def category(self) -> str:
        """Kategorie: 'aggregation', 'breakage', etc."""
        pass
    
    @abstractmethod
    def __init__(self, **params):
        self.params = dict(params)
    
    def get_default_params(self) -> Dict[str, Any]:
        """Defaults für diesen Kernel"""
        return {}
    
    def validate_params(self, params: dict) -> dict:
        """Parameter-Validierung"""
        return params
```

**Was KernelBase tut:**
1. Definiert einheitliche Schnittstelle für alle Kernel
2. Bietet Standard-Implementierungen für `__repr__`, `validate_params`
3. Erzwingt Implementation aller abstrakten Methoden

---

### 3.4 KernelManager - Zentrale Delegation

**Datei:** `kernel_integration.py`

Der `KernelManager` ist das Herzstück der Kernel-Integration:

```python
class KernelManager:
    def __init__(
        self,
        agg_kernel_name: Optional[str] = None,
        agg_kernel_params: Optional[dict] = None,
        break_kernel_name: Optional[str] = None,
        break_kernel_params: Optional[dict] = None,
        # ... weitere Kernel-Typen
    ):
        # Speichert Konfiguration
        self.agg_kernel_name = agg_kernel_name
        self.agg_kernel_params = agg_kernel_params or {}
        # ...
        
        # Kernel-Instanzen (werden später initialisiert)
        self.agg_kernel = None
        self.break_kernel = None
        # ...
    
    def initialize_kernels(self, solver) -> None:
        """
        Initialisiert ALLE Kernel basierend auf Konfiguration.
        
        Priorität:
        1. Expliziter Kernel-Name (z.B. agg_kernel_name='liquid_bridge')
        2. Legacy COLEVAL/BREAKRVAL Mapping
        3. Default-Kernel
        """
        from .kernels.aggregation import get_aggregation_kernel
        from .kernels.breakage import get_breakage_kernel
        # ...
        
        # Aggregation Kernel
        if self.agg_kernel_name:
            self.agg_kernel = get_aggregation_kernel(
                self.agg_kernel_name,
                **self.agg_kernel_params
            )
            # Rückwärtskompatibilität: Setze Solver-Attribute
            solver.CORR_BETA = self.agg_kernel_params['corr_beta']
            solver.G = self.agg_kernel_params['g']
        else:
            # Default: shear_chin1998
            self.agg_kernel = get_aggregation_kernel('shear_chin1998', corr_beta=1e-3, g=1000)
        
        # ... analog für andere Kernel-Typen
    
    # ==========================================
    # Delegationsmethoden
    # ==========================================
    
    def compute_beta(self, r1, r2, particle1_idx=None, particle2_idx=None, solver=None):
        """Delegiert β-Berechnung an Aggregation-Kernel"""
        if self.agg_kernel is None:
            raise RuntimeError("Aggregation kernel not initialized")
        return self.agg_kernel.compute_beta(r1, r2, particle1_idx, particle2_idx, solver)
    
    def compute_break_rate(self, v_particle, particle_idx=None, solver=None):
        """Delegiert Breakage-Rate an Breakage-Kernel"""
        return self.break_kernel.compute_rate(v_particle, particle_idx, solver)
    
    # ... weitere Delegationsmethoden
```

**Was KernelManager tut:**
1. **Initialisierung**: Erstellt Kernel-Instanzen aus Konfiguration
2. **Rückwärtskompatibilität**: Setzt legacy Solver-Attribute für alten Code
3. **Delegation**: Leitet Physik-Berechnungen an richtige Kernel weiter
4. **Zentralisierung**: Single Point of Truth für alle Kernel

---

### 3.5 Beispiel: Aggregation-Kernel Implementierung

**Datei:** `kernels/aggregation/shear_chin1998.py`

```python
@njit(cache=True)
def _beta_shear_jit(corr_beta: float, g: float, r1: float, r2: float) -> float:
    """JIT-compilierte Shear-Kernel für maximale Performance"""
    return corr_beta * g * (r1 + r2)**3


class ShearChinKernel(AggregationKernel):
    """
    Shear-induced agglomeration nach Chin et al. (1998).
    
    Formel: β = CORR_BETA × G × (r1 + r2)³
    """
    
    @property
    def name(self) -> str:
        return 'shear_chin1998'
    
    def get_default_params(self) -> dict:
        return {
            'corr_beta': 1e-3,  # Kollisionseffizienz
            'g': 1.0,           # Scherrate [1/s]
        }
    
    def __init__(self, **params):
        defaults = self.get_default_params()
        defaults.update(params)
        self.params = self.validate_params(defaults)
        
        # Cache für Performance
        self.corr_beta = float(self.params['corr_beta'])
        self.g = float(self.params['g'])
    
    def validate_params(self, params: dict) -> dict:
        if params['corr_beta'] <= 0:
            raise ValueError("corr_beta must be positive")
        if params['g'] < 0:
            raise ValueError("g must be non-negative")
        return params
    
    def compute_beta(self, r1, r2, particle1_idx=None, particle2_idx=None, solver=None):
        """
        Berechne Kollisionsfrequenz β(i,j).
        
        Args:
            r1, r2: Partikelradien [m]
            particle1_idx, particle2_idx: Indices im Solver (optional)
            solver: Referenz für Zugriff auf Zustandsarrays (optional)
        
        Returns:
            beta: Kollisionsfrequenz [m³/s]
        """
        if r1 <= 0 or r2 <= 0:
            return 0.0
        
        # Verwende JIT-compilierte Funktion
        beta = _beta_shear_jit(self.corr_beta, self.g, r1, r2)
        return max(0.0, float(beta))
```

---

### 3.6 Erweitertes Beispiel: Liquid Bridge Kernel

**Datei:** `kernels/aggregation/liquid_bridge.py`

Dieser Kernel zeigt die Stärke des Frameworks: **Erweiterbare Physik ohne Änderung am Core-Solver**.

```python
class LiquidBridgeKernel(AggregationKernel):
    """
    Agglomeration mit Flüssigbrücken-Verstärkung.
    
    β = β_shear × f_bridge × f_capillary
    
    where:
    - f_bridge = exp(-((s_eff - s_opt)²) / (2σ_s²))  (Optimum bei ~50% Sättigung)
    - f_capillary = 1 + α × log(1 + V_liq_ext / V_ref)
    """
    
    @property
    def name(self) -> str:
        return 'liquid_bridge'
    
    def get_default_params(self) -> dict:
        return {
            'corr_beta': 1e-3,
            'g': 1.0,
            'optimal_saturation': 0.5,      # Maximale Brückenbildung
            'saturation_width': 0.2,        # Breite des Optimums
            'liquid_enhancement': 1.0,      # Stärke der Verstärkung
            'surface_tension': 0.072,       # N/m (Wasser)
            'contact_angle': 0.0,           # rad (perfekte Benetzung)
        }
    
    def compute_beta(self, r1, r2, particle1_idx=None, particle2_idx=None, solver=None):
        # Basis: Shear-Kollision
        beta_shear = self.corr_beta * self.g * (r1 + r2)**3
        
        # Ohne Solver-Referenz: Return Basis (graceful degradation)
        if solver is None or particle1_idx is None:
            return float(beta_shear)
        
        # Hole Sättigung aus Solver
        s1 = solver.saturation[particle1_idx]
        s2 = solver.saturation[particle2_idx]
        
        if np.isnan(s1) or np.isnan(s2):  # Vollkörper
            return float(beta_shear)
        
        # Bridge Formation Factor (Gaussian um optimal_saturation)
        s_eff = (s1 + s2) / 2.0
        bridge_factor = np.exp(-((s_eff - self.s_opt)**2) / (2 * self.sigma_s**2))
        
        # Capillary Enhancement (logarithmisch mit externer Flüssigkeit)
        v_liq_ext1 = solver.get_V_liquid_external(particle1_idx)
        v_liq_ext2 = solver.get_V_liquid_external(particle2_idx)
        liquid_factor = 1.0 + self.alpha_liq * np.log1p((v_liq_ext1 + v_liq_ext2) / v_ref)
        
        return max(0.0, float(beta_shear * bridge_factor * liquid_factor))
```

**Warum das wichtig ist:**
- Der Kernel kann auf **Partikeleigenschaften** zugreifen (Sättigung, externe Flüssigkeit)
- **Keine Änderung** am Solver nötig für neue Physik
- **Graceful degradation**: Funktioniert auch ohne Solver-Referenz (trockene Simulationen)

---

### 3.7 Kernel-Integration im Solver

**Datei:** `mcpbe_base.py`

```python
class MCPBEBase(MCPBETimeHelper, BaseSolver):
    def __init__(
        self,
        dim: int = 2,
        # ... andere Parameter
        # Kernel-Framework Parameter (neu, optional)
        agg_kernel_name: Optional[str] = None,
        agg_kernel_params: Optional[dict] = None,
        break_kernel_name: Optional[str] = None,
        break_kernel_params: Optional[dict] = None,
        # ...
    ):
        # ... andere Initialisierung
        
        # 1. Kernel Manager erstellen
        self.kernel_manager = KernelManager(
            agg_kernel_name=agg_kernel_name,
            agg_kernel_params=agg_kernel_params,
            break_kernel_name=break_kernel_name,
            break_kernel_params=break_kernel_params,
            # ...
        )
        
        # 2. Alle Kernel initialisieren
        self.kernel_manager.initialize_kernels(self)
        
        # 3. Partikel initialisieren
        self._initialize_particles()
```

**Nutzung im Agglomeration-Mixin (`mcpbe_agg.py`):**

```python
class MCPBEAgg:
    def _beta(self, i: int, j: int) -> float:
        """Pair kernel β(i,j) via Kernel-Framework"""
        if self.kernel_manager is None or self.kernel_manager.agg_kernel is None:
            raise RuntimeError("Aggregation kernel not initialized")
        
        r1 = float(self.X[i] * 0.5)
        r2 = float(self.X[j] * 0.5)
        
        return self.kernel_manager.compute_beta(
            r1, r2,
            particle1_idx=i,
            particle2_idx=j,
            solver=self
        )
    
    def _rebuild_all_propensities(self):
        """Berechne Propensities r_i = Σ_j W[j] × β(i,j)"""
        a = self.a_tot
        R = (self.X[:a] * 0.5).astype(np.float64)
        W = self.W[:a].astype(np.float64)
        
        kernel_name = getattr(self.kernel_manager.agg_kernel, 'name', '')
        
        # JIT-optimierte Pfade für Standard-Kernel
        if kernel_name == 'shear_chin1998':
            r = np.zeros(a, dtype=float)
            rebuild_r_array_shear(
                float(self.kernel_manager.agg_kernel.corr_beta),
                float(self.kernel_manager.agg_kernel.g),
                R, W, r
            )
        elif kernel_name == 'liquid_bridge':
            # Fallback für benutzerdefinierte Kernel (Python-Loop)
            r = np.zeros(a, dtype=float)
            for i_idx in range(a):
                ri = 0.0
                r1 = float(R[i_idx])
                for j_idx in range(a):
                    r2 = float(R[j_idx])
                    beta_ij = self.kernel_manager.compute_beta(
                        r1, r2,
                        particle1_idx=i_idx,
                        particle2_idx=j_idx,
                        solver=self
                    )
                    if i_idx == j_idx:
                        ri += max(0.0, float(W[j_idx] - 1)) * beta_ij
                    else:
                        ri += float(W[j_idx]) * beta_ij
                r[i_idx] = ri
        
        # Normalize mit delta
        self._r_agg[:a] = r / delta
```

---

## 4. Kernel-Typen im Detail

### 4.1 Aggregation Kernels (`compute_beta`)

| Kernel | Formel | Anwendung |
|--------|--------|-----------|
| `shear_chin1998` | β = k × G × (r1+r2)³ | Standard für High-Shear Mixer |
| `brownian_tsouris1995` | β = f(T, η, r1, r2) | Brown'sche Bewegung |
| `constant` | β = konst. | Validierung |
| `sum` | β = k × (r1 + r2) | Analytische Lösungen |
| `liquid_bridge` | β = β_shear × f(S, V_liq) | Feuchte Granulation |

### 4.2 Breakage Kernels (`compute_rate`)

| Kernel | Formel | Anwendung |
|--------|--------|-----------|
| `power_law` | S(V) = P1 × G^P2 × V^α | Klassisches Modell |
| `stress_based` | S(V) = f(σ_local, σ_strength) | Stress-basiert (Weibull) |

### 4.3 Porosity Growth Kernels (`compute_merged_porosity`)

| Kernel | Modell | Anwendung |
|--------|--------|-----------|
| `volume_mixing` | V_pore,new = V_pore,1 + V_pore,2 | Volumen-additiv |
| `incomplete_mixing` | Mit trapped pore fraction | Realistischere Porenbildung |

### 4.4 Compression Kernels (`compute_porosity_decay`)

| Kernel | Formel | Anwendung |
|--------|--------|-----------|
| `exponential_decay` | ε(t) = ε_min + (ε₀-ε_min)×exp(-kt) | Standardmodell |
| `stress_dependent` | dε/dt = f(σ, ε, S) | Stress-/Sättigungs-abhängig |

### 4.5 Liquid Distribution Kernels (`select_target_particle`)

| Kernel | Strategie | Anwendung |
|--------|-----------|-----------|
| `uniform_weighted` | P(i) ∝ W[i] | Standard (alle Partikel gleich) |
| `surface_weighted` | P(i) ∝ Oberfläche[i] | Größere Partikel bevorzugt |
| `saturation_preferential` | P(i) ∝ f(1-S[i]) | Ungesättigte Partikel bevorzugt |

---

## 5. Datenfluss im Kernel-Framework

```
┌─────────────────────────────────────────────────────────────┐
│                     MCPBESolver.solve()                      │
│                   (Hauptsimulationsschleife)                 │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
┌──────────────┐              ┌──────────────────┐
│ _do_one_agg()│              │ _do_one_break()  │
│ (Agg-Mixin)  │              │ (Break-Mixin)    │
└──────┬───────┘              └────────┬─────────┘
       │                               │
       │ 1. Partner selektieren        │ 1. Partikel selektieren
       │    (_pick_partner_kernel)     │    (_break_sampler.sample)
       │                               │
       ▼                               ▼
┌──────────────────────┐     ┌────────────────────┐
│ kernel_manager.      │     │ kernel_manager.    │
│ compute_beta()       │     │ compute_break_rate()│
└──────┬───────────────┘     └───────┬────────────┘
       │                             │
       │ 2. Delegation an Kernel     │ 2. Delegation an Kernel
       │                             │
       ▼                             ▼
┌──────────────────────┐     ┌────────────────────┐
│ AggregationKernel    │     │ BreakageKernel     │
│ .compute_beta()      │     │ .compute_rate()    │
│ (z.B. ShearChinKernel)│    │ (z.B. PowerLaw)    │
└──────┬───────────────┘     └───────┬────────────┘
       │                             │
       │ 3. Physik berechnen         │ 3. Physik berechnen
       │    (β = k×G×(r1+r2)³)       │    (S = P1×V^α)
       │                             │
       ▼                             ▼
┌──────────────────────┐     ┌────────────────────┐
│ Beta-Wert [m³/s]     │     │ Rate-Wert [1/s]    │
└──────────────────────┘     └────────────────────┘
```

---

## 6. Fenwick Tree Sampler

**Datei:** `fenwick_new.py`

Der Fenwick Tree ermöglicht **effizientes gewichtetes Sampling** in O(log n):

```python
class FenwickSampler:
    """
    Fenwick Tree (Binary Indexed Tree) für dynamische Gewichte.
    
    Operationen:
    - sample(rng): Wähle Index mit P(i) = W[i] / ΣW  → O(log n)
    - update(idx, new_weight): Ändere Gewicht  → O(log n)
    - append(weight): Füge neues Element  → O(log n)
    - remove(idx): Entferne Element (swap-with-last) → O(log n)
    """
    
    def __init__(self, weights: np.ndarray):
        self._w = weights.copy()
        self._tree = np.zeros(len(weights) + 1, dtype=float)
        nb_fenwick_build(self._tree, self._n, self._w)  # JIT
    
    def sample(self, rng) -> int:
        """Sample Index proportional zu Gewicht"""
        t = self.total()
        u = rng.random() * t
        return self.prefix_sum_search(u)
    
    def prefix_sum_search(self, s: float) -> int:
        """Binary Lifting auf Fenwick Tree"""
        i = 0
        bit = 1 << (self._n.bit_length() - 1)
        while bit:
            nxt = i + bit
            if nxt <= self._n and self._tree[nxt] <= s:
                s -= self._tree[nxt]
                i = nxt
            bit >>= 1
        return i
```

**Anwendung im Solver:**
```python
# Agglomeration: r_i = Σ_j W[j] × β(i,j)
self._agg_sampler = FenwickSampler(self._r_agg[:self.a_tot])

# Sample first particle
i = self._agg_sampler.sample(self._rng)

# Breakage: propensity_i = W[i] × S_i
self._break_sampler = FenwickSampler(self._break_rate[:self.a_tot])
i = self._break_sampler.sample(self._rng)
```

---

## 7. Handler (Composition Pattern)

### 7.1 NucleationHandler

**Datei:** `mcpbe_nucleation.py`

```python
class NucleationHandler:
    """
    Verteilt Flüssigkeitstropfen auf Partikel während der Simulation.
    
    Physikalischer Modus:
    - Tropfen kollidieren zufällig mit Partikeln
    - Alle Partikel haben gleiche Kollisionswahrscheinlichkeit
    - DSMC-Scaling: physical_droplets = (Vc_ref/Vc) × W_selected
    """
    
    def step(self, current_time: float, dt_event: float):
        """Verteile Tropfen basierend auf volumetrischem Fluss"""
        if not self.config.enabled:
            return
        
        # Berechne Anzahl Tropfen-Events für dieses Zeitintervall
        n_droplets = self.volumetric_flow_rate * dt_event / self.droplet_volume
        
        for _ in range(n_droplets):
            # Wähle Partikel proportional zu W (uniform für PHYSICAL particles)
            target_idx = self.solver._select_particle_uniform_physical()
            
            # Addiere Flüssigkeit
            self.solver.liquid_volume[target_idx] += self.droplet_volume
```

### 7.2 CompressionHandler

**Datei:** `mcpbe_compression.py`

```python
class CompressionHandler:
    """
    Reduziert Porosität über Zeit (exponentieller Zerfall).
    
    Modell: ε(t) = ε_min + (ε₀ - ε_min) × exp(-k × t)
    """
    
    def step(self, current_time: float, dt_event: float):
        """Wende Kompression auf alle porösen Partikel an"""
        if not self.config.enabled:
            return
        
        for i in range(self.solver.a_tot):
            poro = self.solver.porosity[i]
            
            if np.isnan(poro):  # Vollkörper
                continue
            
            if poro > self.config.min_porosity:
                # Kernel-basierte Berechnung
                new_poro = self.compression_kernel(i, dt_event)
                self.solver.porosity[i] = new_poro
                
                # V_dry anpassen (V_solid bleibt konstant!)
                V_solid = self.solver.V_flat[-1, i] * (1.0 - poro)
                V_dry_new = V_solid / (1.0 - new_poro)
                self.solver.V_flat[-1, i] = V_dry_new
```

---

## 8. Zusammenfassung: Was macht die Kernel-Base-Funktion?

### KernelBase (abstrakte Basisklasse)

1. **Definiert Schnittstelle**: Jede Kernel-Klasse MUSS implementieren:
   - `name`: Eindeutiger Identifier
   - `category`: Kernel-Typ ('aggregation', 'breakage', etc.)
   - Kategorie-spezifische Methode (z.B. `compute_beta` für Aggregation)

2. **Bietet Standard-Funktionalität**:
   - `get_default_params()`: Defaults pro Kernel
   - `validate_params()`: Parameter-Validierung
   - `__repr__()`: String-Repräsentation

3. **Ermöglicht Polymorphismus**:
   - KernelManager kann JEDEN Kernel behandeln
   - Neue Kernel benötigen KEINE Änderungen am Solver
   - Einheitlicher Aufruf: `kernel_manager.compute_beta(...)`

### KernelManager (Integration)

1. **Initialisierung**: Erstellt Kernel-Instanzen aus Konfiguration
2. **Delegation**: Leitet Aufrufe an richtigen Kernel weiter
3. **Rückwärtskompatibilität**: Setzt legacy Solver-Attribute
4. **Zentralisierung**: Single Point of Truth für Physik-Modelle

### Vorteile dieser Architektur

| Aspekt | Vorher (Legacy) | Nachher (Kernel) |
|--------|-----------------|------------------|
| Erweiterbarkeit | Änderung am Solver nötig | Neue Kernel-Klasse |
| Testbarkeit | Schwer zu isolieren | Kernel einzeln testbar |
| Flexibilität | Feste Modelle | Austauschbare Modelle |
| Wartbarkeit | Monolithisch | Modular |
| Dokumentation | Implizit im Code | Explizite Kernel-Klassen |

---

## 9. Beispiel: Eigenen Kernel hinzufügen

```python
# Datei: kernels/aggregation/my_custom_kernel.py

from ..base import AggregationKernel

class MyCustomKernel(AggregationKernel):
    @property
    def name(self) -> str:
        return 'my_custom'
    
    def get_default_params(self) -> dict:
        return {'factor': 1.0}
    
    def compute_beta(self, r1, r2, particle1_idx=None, particle2_idx=None, solver=None):
        # Eigene Physik hier implementieren
        return self.params['factor'] * (r1 * r2)**2

# Registrierung in kernels/aggregation/__init__.py
AGG_KERNELS['my_custom'] = MyCustomKernel

# Verwendung im Solver
solver = MCPBESolver(
    agg_kernel_name='my_custom',
    agg_kernel_params={'factor': 2.0}
)
```

---

## 10. Glossar

| Begriff | Bedeutung |
|---------|-----------|
| **DSMC** | Direct Simulation Monte Carlo - stochastischer Algorithmus |
| **W (Weight)** | Computational Weight = N_physical / N_computational |
| **Vc** | Control Volume - Simulationsvolumen |
| **Propensity** | Ereignisrate r_i = Σ_j W[j] × β(i,j) |
| **Fenwick Tree** | Datenstruktur für O(log n) weighted sampling |
| **Vollkörper** | Partikel ohne Porosität (porosity = NaN) |
| **Saturation** | S = V_liquid_internal / V_pore ∈ [0, 1] |
| **Kernel** | Modulbares Physik-Modell mit einheitlicher Schnittstelle |
