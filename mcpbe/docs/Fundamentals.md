# WMCPBE Projekt - Fundamentals

## Limits

Wenn du unsicher bist Frage immer die nutzende Person. Treffe vorallem keine Annahmen ohne den Nutzer

### Tools

Bash, PowerShell und Python sind nicht verfügbar. Nur read, write und edit sind nutzbar. Wenn Tests ausgeführt werden sollen, oder Code stellen nicht gefunden werden frage die nutzende Person mit eine Ortsangabe zu suchen.

### Änderungen

Es werden am Core-Code und an den Kernels keine Änderungen ohne explizite Zustimmung der Nutzenden Person gemacht. Frage bei 

## Physik

Alle größen V_dry, Poro, Saturation, V_liquid sind partikelintesiv, verändern sich also bei er Vererbung nicht. Der unterschied zwischn n_phys und n_commp liegt darin, dass ein n_comp W n_phys darstellt. Deswegen sind Änderungen bei z.B. der Agglomeration nur im Child und im W der Parents sichtbar. 

## 2. Software-Architektur

### 2.1 Hauptklassen-Hierarchie

```
MCPBESolver (Hauptsolver)
├── MCPBEBase (Basis-Klasse via Mehrfachvererbung)
│   ├── Partikelzustand (V_flat, W, X, liquid_volume, porosity, saturation)
│   ├── Control Volume Management
│   ├── Hauptsolve-Loop mit Event-driven Time Stepping
│   └── KernelManager Integration
├── MCPBEAgg (Agglomeration-Mixin)
│   ├── Beta-Kernel-Berechnung (Partikelkollisionsrate)
│   ├── Partner-Selection (Fenwick Sampler)
│   └── Agglomerations-Event-Durchführung
├── MCPBEBreak (Breakage-Mixin)
│   ├── Breakage-Rate-Berechnung (MLP/JIT/Kernel)
│   ├── Fragment-Verteilung (CDF-Tabellen, LMC-Adapter)
│   └── Multi-Fragment-Event-Durchführung
├── MCPBEPost (Post-Processing-Mixin)
│   ├── Momentenberechnung (µ(i,j,t))
│   └── PSD-CDF-Berechnung über Zeit
├── ReconstructionMixin (Rekonstruktion/Resampling)
│   ├── CAM (Cell Average Method)
│   ├── RS (Resampling mit C-1 Korrektur)
│   ├── 2PM (Two-Point-Methode)
│   └── QMX (Quantile MiX - hybride Methode)
└── Handler (Komposition, keine Vererbung)
    ├── NucleationHandler (Flüssigkeitszugabe)
    ├── CompressionHandler (Porositätskompression, legacy)
    └── ContinuousProcessesHandler (Kontinuierliche Prozesse)
```

### 2.2 Method Resolution Order (MRO)

```python
MCPBESolver → MCPBEPost → MCPBEBreak → MCPBEAgg → MCPBEBase → ReconstructionMixin
```

**Warum diese Reihenfolge?**
- Post-Processing muss zuletzt kommen für Zugriff auf alle States
- Breakage vor Agglomeration für korrekte Event-Reihenfolge im Mix-Modus
- Base enthält fundamentale Initialisierung

---

## 3. Dateistruktur

```
wmcpbe/
├── mcpbe.py                      # Hauptsolver-Klasse (MCPBESolver)
├── mcpbe_base.py                 # Basis-Solver mit DSMC-Logik
├── mcpbe_agg.py                  # Agglomeration-Mixin
├── mcpbe_break.py                # Breakage-Mixin
├── mcpbe_post.py                 # Post-Processing-Mixin
├── reconstruction_mixin.py       # Rekonstruktionsmethoden (CAM, RS, 2PM, QMX)
├── mcpbe_nucleation.py           # NucleationHandler + Config
├── mcpbe_compression.py          # CompressionHandler + Config (legacy)
├── mcpbe_continuous_processes.py # ContinuousProcessesHandler + Config
├── kernel_integration.py         # KernelManager für modulare Physik
├── fenwick_new.py                # FenwickSampler für gewichtetes Sampling
├── mcpbe_time_helper.py          # Zeitmanagement-Helfer
├── kernels/                      # Modulare Physik-Kernels
│   ├── aggregation/              # Aggregationskernel (shear, brownian, constant, etc.)
│   ├── breakage/                 # Breakage-Kernel (power_law, etc.)
│   ├── porosity_growth/          # Porositätswachstum bei Agglomeration
│   ├── compression/              # Porositätskompression (legacy)
│   ├── liquid_distribution/      # Flüssigkeitsverteilung
│   ├── agglomeration_acceptance/ # Kollisionsakzeptanz (Stokes-Kriterium)
│   └── continuous_processes/     # Kontinuierliche Prozesse
├── lmc_adapter.py                # LMC-Adapter für Fragmentverteilung
└── mlp_breakage_adapter.py       # MLP-Modell für Breakage-Raten
```

---

## 4. Kernkomponenten im Detail

### 4.1 MCPBEBase - Core Solver

**Verantwortlichkeiten:**
- Partikelarrays verwalten: `V_flat`, `W`, `X`, `liquid_volume`, `porosity`, `saturation`
- Control Volume Management mit dynamischem Verdoppeln
- Hauptsolve-Loop mit Event-driven Time Stepping
- Capacity Management für Partikelarrays (dynamisches Wachstum)
- KernelManager initialisieren

**Wichtige Methoden:**

| Methode | Beschreibung |
|---------|--------------|
| `_initialize_particles()` | Initialisiert Partikel aus Verteilung oder CDF |
| `_initialize_samplers()` | Baut FenwickSampler für Agg/Break |
| `_maybe_double_control_volume()` | DSMC: Vc verdoppeln wenn Partikel < 50% |
| `_ensure_capacity_for()` | Dynamisches Array-Wachstum |

**State Arrays (alle Länge = _cap):**

| Array | Dimension | Beschreibung |
|-------|-----------|--------------|
| `V_flat[dim+1, _cap]` | Volumina | dim Komponenten + total |
| `W[_cap]` | Gewichte | Computational Weights |
| `X[_cap]` | Durchmesser | Aus V_total berechnet |
| `liquid_volume[_cap]` | Flüssigkeit | Pro Partikel [m³] |
| `porosity[_cap]` | Porosität | NaN für Vollkörper |
| `saturation[_cap]` | Sättigung | 0-1, nur für poröse Partikel |

### 4.2 MCPBEAgg - Agglomeration

**Physikalisches Modell:**
- Partikel kollidieren mit Rate β(i,j) abhängig von Radien r_i, r_j
- Kollisionseffizienz α(i,j) für 2D-Systeme (4 Komponentenkombinationen)
- Effektive Rate: `β_eff = β × α`

**Algorithmus:**
1. Wähle Partikel i proportional zu Propensität r_i = Σⱼ Wⱼ×β(i,j)
2. Wähle Partner j proportional zu Wⱼ (gewichtetes Sampling)
3. Berechne β(i,j) via Kernel
4. Akzeptanzprüfung (SIZEEVAL, Stokes-Kriterium via Kernel)
5. Erzeuge neues Partikel mit dW Gewicht
6. Reduziere Eltern-Gewichte um dW
7. Rebuild Propensities (O(n²) via JIT oder Kernel)

**Wichtige Methoden:**

| Methode | Beschreibung |
|---------|--------------|
| `_beta(i, j)` | Berechnet β(i,j) via Kernel |
| `_rebuild_all_propensities()` | Neuberechne alle r_i (parallel via JIT) |
| `_do_one_agg()` | Führt ein Agglomerationsereignis durch |
| `_pick_partner_kernel()` | Wählt Partner j gewichtet nach W |

**JIT-optimierte Kernel:**
- `rebuild_r_array_shear()`: Scherungsgetriebene Agglomeration
- `rebuild_r_array_brownian()`: Brownsche Bewegung
- `rebuild_r_array_constant()`: Konstanter Kernel
- `rebuild_r_array_sum()`: Summenkernel

### 4.3 MCPBEBreak - Breakage

**Physikalisches Modell:**
- Breakage-Rate S(V) abhängig von Partikelvolumen
- Fragmentverteilung aus CDF-Tabellen oder LMC (Lattice Monte Carlo)

**Fragmentverteilungs-Methoden:**

| Methode | Beschreibung |
|---------|--------------|
| **Precomputed Tables** | Tabelle/Rank/Copula/Flow-Adapter |
| **Live LMC** | Echtzeit-Fragmentgenerierung |
| **Analytisch** | Stepwise Splitting aus Beta-Funktion |

**Algorithmus:**
1. Wähle Partikel k proportional zu Propensität W_k × S_k
2. Bestimme Fragmentanzahl n (stochastisches Runden von E[n])
3. Generiere n Fragmente via CDF-Sampling
4. Erzeuge neue Partikel für Fragmente (jeweils Gewicht dW)
5. Reduziere Elterngewicht um dW
6. Update Breakage/Agg Samplers

**Wichtige Methoden:**

| Methode | Beschreibung |
|---------|--------------|
| `_calc_break_rates_full()` | Berechnet alle Breakage-Raten |
| `_break_rate_single(i)` | Einzelne Breakage-Rate |
| `_build_break_function()` | Baut CDF-Tabellen für Fragmente |
| `_produce_one_frag_from_remaining()` | Sampling eines Fragments |
| `_do_one_break()` | Führt ein Breakage-Ereignis durch |

**LMC-Adapter:**
- `LMCTableAdapter`: Marginaltabellen
- `LMCRankAdapter`: Rank-basierte Tabellen
- `LMCCopulaAdapter`: Copula-basierte Korrelationen
- `LMCFlowAdapter`: Flow-basierte Modelle
- `LMCLiveAdapter`: Live-LMC-Simulator

### 4.4 NucleationHandler - Flüssigkeitszugabe

**Physikalisches Modell:**
- Flüssigkeit wird als diskrete Tropfen (Droplets) zugegeben
- Alle Partikel haben gleiche Kollisionswahrscheinlichkeit (uniform sampling by weight W)
- DSMC-Skalierung: `physical_droplets = (Vc_ref / Vc) × W_selected`

**Duration-Spezifikation (mutually exclusive):**

| Option | Beschreibung |
|--------|--------------|
| **Option A** | Direkte Duration via `liquid_addition_duration` [s] |
| **Option B** | Berechnet aus `target_wt_percent + solid_mass_in_mixer + densities` |

Formel Option B: `duration = (solid_mass × wt%/100 / rho_liquid) / flow_rate`

**DSMC-Skalierung (kritisch!):**

```
Computational Weight (W):
  W = N_physical / N_computational
  Beispiel: W=50 bedeutet 50 physikalische Partikel pro computational particle

Droplet distribution:
  droplet_comp = 1 (ein computational event)
  droplet_phys = (Vc_ref / Vc) × W_selected
  
Beispiel: Vc_ref=1e-6 m³, Vc=2e-6 m³, W=50
  → Physical droplets = (1e-6/2e-6) × 50 = 25 droplets
```

**Wichtige Methoden:**

| Methode | Beschreibung |
|---------|--------------|
| `step(current_time, dt_event)` | Ein Nucleation-Schritt |
| `_distribute_liquid_volume()` | Verteilt Flüssigkeit als Droplets |
| `_select_particle_uniform_physical()` | Wählt Partikel uniform nach phys. Count |
| `_ensure_samplers()` | Baut FenwickSampler für W-gewichtetes Sampling |

**Statistiken:**
- `_droplets_added_total`: Gesamte physikalische Droplets (weighted sum)
- `_liquid_volume_added_total`: Gesamtes Flüssigkeitsvolumen [m³]
- `_liquid_remainder`: Akkumulierter Rest für nächste Iteration (Massenerhaltung!)

### 4.5 ContinuousProcessesHandler - Kontinuierliche Prozesse

**Prozesse:**

| Prozess | Beschreibung | Gleichung |
|---------|--------------|-----------|
| **Liquid Internalization** | Braumann et al. 2007 | `dl_intern/dt = k × l_ex × (v_pore - l_intern)` |
| **Porosity Compression** | Exponentieller Zerfall | `ε(t) = ε_min + (ε_0 - ε_min) × exp(-rate × t)` |

**Prozessreihenfolge pro Zeitschritt:**
1. Liquid Internalization updaten (Sättigung ändern)
2. Porosity Compression anwenden (Porenvolumen reduzieren)
3. Überschüssige Flüssigkeit externalisieren wenn S > 1

**Wichtige Methoden:**

| Methode | Beschreibung |
|---------|--------------|
| `step(current_time, dt_event)` | Ein Schritt kontinuierlicher Prozesse |
| `_apply_internalization(dt)` | Wendet Internalisierung an |
| `_apply_compression(dt)` | Wendet Kompression an |

### 4.6 ReconstructionMixin - Rekonstruktion/Resampling

**Zweck:**
- Reduziere Partikelanzahl wenn zu groß (> recon_N_max)
- Erhalte statistische Momente (M0, M1, M2)
- Vermeide numerische Diffusion der PSD

**Methoden:**

#### CAM (Cell Average Method)
- Baue Grid in Komponenten-Volumenraum
- Berechne Mittelwert pro Zelle (gewichtet nach W)
- Verteile Zellmasse auf benachbarte Pivot-Punkte (linear/bilinear)

#### RS (Resampling mit C-1 Korrektur)
- Bucket Partikel nach CAM-Zellen
- Resample pro Zelle nach M0-Proportion
- Two-weight correction für M1-Erhaltung (1D/2D)

#### 2PM (Two-Point-Methode)
- Pro Zelle: zwei Repräsentanten bei x₁ = mean - σ, x₂ = mean + σ
- Gewichte w₁ = w₂ = M0/2
- Erhält M0, M1, M2 exakt

#### QMX (Quantile MiX - hybrid)
- Partitioniere Partikel nach Vtot-Quantilen (small/mid/tail)
- Wende unterschiedliche Methoden pro Partition an:
  - Small: 2PM (gut für kleine Partikel)
  - Mid: RS (effizient, gute Genauigkeit)
  - Tail: CAM (erhält Tail-Struktur)

**Wichtige Methoden:**

| Methode | Beschreibung |
|---------|--------------|
| `maybe_reconstruct(iter_count)` | Prüft ob Rekonstruktion nötig |
| `reconstruct(method="CAM")` | Führt Rekonstruktion durch |
| `_kernel_cam()/RS()/2pm()/qmx()` | Kernel-spezifische Logik |

---

## 5. Kernel Framework (modulare Physik)

### 5.1 KernelManager

**Zweck:**
- Zentrale Verwaltung aller Physik-Kernels
- Ermöglicht Austausch von Physikmodellen ohne Codeänderung
- Delegiert Berechnungen an spezifische Kernel-Implementierungen

**Kernel-Typen:**

| Kernel | Beschreibung |
|--------|--------------|
| `agg_kernel` | Aggregationsrate β(r1, r2) |
| `break_kernel` | Breakage-Rate S(V) |
| `porosity_growth_kernel` | Porosität bei Agglomeration |
| `compression_kernel` | Porositätskompression (legacy) |
| `liquid_dist_kernel` | Partikelauswahl für Flüssigkeit |
| `agglomeration_acceptance_kernel` | Kollisionsakzeptanz (Stokes) |
| `porosity_compression_kernel` | Kontinuierliche Kompression |
| `liquid_internalization_kernel` | Kontinuierliche Internalisierung |
| `liq_internalisation_agglomeration_kernel` | Internalisierung bei Agglomeration |

### 5.2 Kernel-Initialisierung

```python
solver = MCPBESolver(
    agg_kernel_name='shear_chin1998',
    agg_kernel_params={'corr_beta': 1e-3, 'g': 1000},
    break_kernel_name='power_law',
    break_kernel_params={'p1': 3e-2, 'pl_v': 2.0},
    porosity_growth_kernel_name='volume_mixing',
    liquid_internalization_kernel_name='braumann2007',
    liquid_internalization_kernel_params={'k_int': 1e12},
)
```

### 5.3 Verfügbare Kernel (Beispiele)

**Aggregation:**
- `shear_chin1998`: Scherungsgetrieben (Chin et al. 1998)
- `brownian_tsouris1995`: Brownsche Bewegung
- `constant`: Konstanter Kernel
- `sum`: Summenkernel (r1 + r2)
- `liquid_bridge`: Flüssigkeitsbrücken-modelliert

**Breakage:**
- `power_law`: Potenzgesetz-basiert
- `mlp_model`: MLP-neuronales Netz

**Porosity Growth:**
- `volume_mixing`: Einfache Volumenmittelung
- `incomplete_mixing`: Unvollständige Mischung
- `cone_model`: Kegelmodell für Porenkollaps

---

## 6. FenwickSampler - Effizientes gewichtetes Sampling

**Zweck:**
- O(log n) Sampling nach Gewichten W
- O(log n) Updates bei Gewichtsänderung
- Kritisch für Performance bei vielen Partikeln (>10.000)

**API:**

```python
sampler = FenwickSampler(weights)
idx = sampler.sample(rng)           # Sample Index nach weights
sampler.update(idx, new_weight)     # Update einzelnes Gewicht O(log n)
sampler.append(weight)              # Neues Element hinzufügen
sampler.remove(idx)                 # Element entfernen (swap-with-last)
total = sampler.total()             # Summe aller Gewichte
```

**Implementierung:**
- Binary Indexed Tree (Fenwick Tree)
- Numba-JIT für Performance
- Prefix-Sum-Suche für Sampling

---

## 7. Wichtige physikalische Invarianten

### 7.1 Massenerhaltung

**Bei Agglomeration:**
```python
# V_flat[:dim] = V_solid (CONSTANT!)
V_solid_merged = V_solid_i + V_solid_j  # Masse erhalten!

# V_flat[-1] = V_dry (ändert sich mit Porosität)
V_dry_merged = V_solid_merged / (1 - porosity_merged)
```

**Bei Breakage:**
```python
# Fragment-Volumina summieren zu Parent-Volumen
sum(V_frag) = V_parent

# liquid_volume wird proportional verteilt
liquid_frag = liquid_parent × (V_frag / V_parent)
```

**Bei Nucleation:**
```python
# Total liquid added = flow_rate × duration
# Keine systematischen Verluste durch _liquid_remainder
```

### 7.2 Intensive vs. extensive Eigenschaften

| Eigenschaft | Typ | Verhalten bei Vc-Verdopplung |
|-------------|-----|------------------------------|
| `W` | Extensiv | Wird verdoppelt (×2) |
| `V_flat` | Intensiv | Bleibt gleich (pro physical particle) |
| `liquid_volume` | Intensiv | Bleibt gleich (pro physical particle) |
| `porosity` | Intensiv | Bleibt gleich |
| `saturation` | Intensiv | Bleibt gleich |
| `X` | Intensiv | Bleibt gleich |

**Vc-Verdopplung Code:**
```python
# W wird verdoppelt (repräsentiert mehr physical particles)
W_new = np.concatenate((W_active, W_active))

# Intensive properties werden KOPIERT (nicht skaliert!)
liquid_volume[:old_a] = lv_active
liquid_volume[old_a:a_tot] = lv_active  # Gleiche Werte!
```

---

## 8. Wichtige Konstanten und Default-Werte

```python
# Agglomeration
CORR_BETA = 1e-3          # Agglomerationsrate
G = 1000                  # Scherrate [1/s]
alpha_prim = 1.0          # Kollisionseffizienz (1D)
alpha_prim = [1,1,1,1]    # Kollisionseffizienz (2D, 4 Kombinationen)

# Breakage
pl_P1 = 3e-2              # Breakage-Rate Parameter
pl_v = 2.0                # Breakage-Funktion Exponent
pl_q = 1.0                # Breakage-Funktion Exponent

# Nucleation
batch_size = 25.0         # Computational weight per Droplet-Event
consistency_tol = 0.001   # 0.1% Toleranz für Parameter-Checks

# Compression
compression_rate = 0.02   # 1/s
min_porosity = 0.3        # Minimale Porosität

# Liquid Internalization
k_int = 1e12              # 1/(m³·s)

# Reconstruction
recon_N_max = 4000        # Trigger bei >4000 Partikeln
recon_bins = 500          # Bins pro Dimension für CAM
recon_RS_target = 2000    # Zielpartikelanzahl nach RS

# DSMC
maybe_double_control_volume = False  # Vc-Verdopplung deaktivierbar
```

---

## 9. Glossar

| Begriff | Bedeutung |
|---------|-----------|
| DSMC | Direct Simulation Monte Carlo |
| PBE | Population Balance Equation |
| PSD | Particle Size Distribution |
| Vc | Control Volume |
| W | Computational Weight (N_physical / N_computational) |
| n_comp | Anzahl computational particles (Simulator) |
| n_phys | Anzahl physical particles (real, = Σ(W)) |
| V_solid | Feststoffvolumen (ohne Poren) |
| V_dry | Gesamtvolumen (mit Poren) |
| V_pore | Porenvolumen |
| Intensive Eigenschaft | Pro physical particle (z.B. porosity, liquid_volume) |
| Extensive Eigenschaft | Skaliert mit W (z.B. Gesamtmasse = W × V_solid) |
| Saturation (S) | V_liq_intern / V_pore |
| LMC | Lattice Monte Carlo (Fragmentverteilung) |
| CAM | Cell Average Method (Rekonstruktion) |
| RS | Resampling (Rekonstruktion) |
| 2PM | Two-Point Method (Rekonstruktion) |
| QMX | Quantile MiX (hybride Rekonstruktion) |
| Fenwick Tree | Binary Indexed Tree für effizientes Sampling |

---

## 10. Projektstruktur

```
PSD_opt/
├── .agents/
│   └── informations.txt          # Detaillierte Architekturdokumentation
├── mcpbe/                         # Hauptpackage
│   ├── pyproject.toml
│   ├── README.md
│   ├── docs/
│   │   └── Fundamentals.md       # Diese Datei
│   ├── src/
│   │   ├── mcpbe/                # Standard MC-PBE Solver
│   │   ├── wmcpbe/               # Weighted MC-PBE Solver (Hauptfokus)
│   │   └── wmcpbe_granulation/   # Granulation-spezifische Erweiterungen
│   └── tests/
├── pbe-core/                      # Gemeinsame Basis (separates Package)
│   ├── base/
│   │   └── base_solver.py        # BaseSolver Klasse
│   ├── func/
│   │   └── jit_mcpbe.py          # JIT-Helferfunktionen
│   └── kernels/                   # Gemeinsame Kernel-Basen
└── simulations/                   # Simulationsdaten, Ergebnisse
    ├── config/
    ├── results/
    └── scripts/
```

---

## 11. Ressourcen

- **Projekt-Root**: `C:\Users\ericb\Documents\GitHub\PSD_opt\`
- **Hauptcode**: `C:\Users\ericb\Documents\GitHub\PSD_opt\mcpbe\src\wmcpbe\`
- **Tests**: `C:\Users\ericb\Documents\GitHub\PSD_opt\mcpbe\src\wmcpbe\Trials`
- **Kernels**: `C:\Users\ericb\Documents\GitHub\PSD_opt\mcpbe\src\wmcpbe\kernels`
- **Architektur-Docs**: `C:\Users\ericb\Documents\GitHub\PSD_opt\.agents\informations.txt`

---

*Letzte Aktualisierung: 2025*  
*Version: 1.0*
