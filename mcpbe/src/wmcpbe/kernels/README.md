# WMCPBE Kernel Framework

Modulare Physik-Kernel für den gewichteten DSMC-Monte-Carlo-PBE-Solver.

> **Stand 20.08.2026.** Dieses Dokument ist der *einzige* Kernel-Katalog. Die
> frühere `KERNEL_UEBERSICHT.md` war eine zweite, parallele Aufzählung derselben
> Kernel und ist hier aufgegangen — zwei Kataloge bedeuteten in der Praxis, dass
> beim nächsten Kernel beide vergessen wurden.

---

## Inhalt

1. [Die Grundregel: Kernel sind eigenständig](#die-grundregel-kernel-sind-eigenständig)
2. [Kernel-Architektur](#kernel-architektur)
3. [Schnellübersicht](#schnellübersicht)
4. [Verfügbare Kernel](#verfügbare-kernel)
5. [Empfohlene Konfiguration](#empfohlene-konfiguration)
6. [Eigenen Kernel schreiben](#eigenen-kernel-schreiben)
7. [Zustandsgrößen: Porosität und Flüssigkeit](#zustandsgrößen-porosität-und-flüssigkeit)
8. [Fehlerbehandlung](#fehlerbehandlung)
9. [Performance](#performance)
10. [Referenzen](#referenzen)

---

## Die Grundregel: Kernel sind eigenständig

**Kein Kernel darf von einem anderen Kernel abhängen.**

Ein Kernel soll einzeln weitergegeben werden können. Erlaubt sind darum nur:

* Standardbibliothek, `numpy`, `numba`
* `kernels/base.py` — die abstrakte Basisklasse
* `kernels/mixer_speed.py` — die gemeinsamen Mischer-Konstanten

**Nicht** erlaubt: ein Import aus einem Nachbar-Kernelmodul, Vererbung von einem
anderen Kernel, oder ein geteiltes privates Hilfsmodul.

Das kostet Duplikation, und zwar bewusst. Die BREAKRVAL-Basisrate steht wörtlich
identisch in allen drei Bruchkerneln; `powerlaw_rumpf_dynamic` hat eine eigene
Kopie des Rumpf-Festigkeitsmodells; `stokes_dynamik` eine eigene Kopie der
Stokes-Logik. Früher gab es dafür ein geteiltes `_base_rate.py` und Vererbung —
beides ist entfernt.

Damit die Kopien nicht auseinanderlaufen, prüft
`Trials/test_dry_mixer_kernels.py` das dreifach:

| Abschnitt | Was geprüft wird |
|---|---|
| 9 | statisch: AST-Prüfung auf Importe, Vererbung nur aus `kernels.base` |
| 10 | numerisch: alle Kopien liefern **bitgenau** dasselbe |
| 11 | dynamisch: jeder Kernel läuft in einem leeren Paket, allein mit `base.py` + `mixer_speed.py` |

Wer einen neuen Kernel hinzufügt, trägt ihn dort in die Listen ein.

---

## Kernel-Architektur

### Basisklassen

Alle in `kernels/base.py`. Jede Kategorie hat ihre eigene abstrakte Methode:

| Basisklasse | Pflichtmethode | Rückgabe |
|---|---|---|
| `AggregationKernel` | `compute_beta(r1, r2, ...)` | Kollisionsfrequenz β [m³/s] |
| `BreakageKernel` | `compute_rate(v_particle, ...)` | Bruchrate S [1/s] |
| `PorosityGrowthKernel` | `compute_merged_porosity(...)` | `(v_dry, porosity)` |
| `CompressionKernel` | `compute(porosity, dt, ...)` | neue Porosität |
| `LiquidInternalizationKernel` | `compute(saturation, v_pore, l_total, dt, ...)` | neue Sättigung |
| `LiquidDistributionKernel` | `select_target_particle(solver, v_droplet, ...)` | Partikelindex |
| `AggAcceptanceKernel` | `accept_collision(r1, r2, v_dry1, v_dry2, ...)` | `bool` |

Ein minimaler Kernel:

```python
from ..base import AggregationKernel


class MyKernel(AggregationKernel):

    @property
    def name(self) -> str:
        return 'my_kernel'                      # muss zum Registry-Schlüssel passen

    def get_default_params(self) -> dict:
        return {'corr_beta': 1e-3}

    def __init__(self, **params):               # **params, nicht params: dict
        defaults = self.get_default_params()
        defaults.update(params)
        self.params = self.validate_params(defaults)
        self.corr_beta = float(self.params['corr_beta'])

    def validate_params(self, params: dict) -> dict:
        if params['corr_beta'] <= 0:
            raise ValueError(f"corr_beta must be positive, got {params['corr_beta']}")
        return params

    def compute_beta(self, r1, r2, particle1_idx=None, particle2_idx=None,
                     solver=None) -> float:
        if r1 <= 0 or r2 <= 0:
            return 0.0
        return self.corr_beta * (r1 + r2) ** 3
```

### Registrierung

Kernel werden **nicht** automatisch gefunden. Eintragen in die Registry der
Kategorie, z. B. `kernels/aggregation/__init__.py`:

```python
from .my_kernel import MyKernel

AGG_KERNELS = {
    ...
    'my_kernel': MyKernel,
}
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

## Schnellübersicht

Alle Defaults unten sind aus dem Code ausgelesen, nicht abgeschrieben.

**β [m³/s] und S [1/s] sind konzentrationsbezogene Größen**, keine reinen
Ereignisraten. β·n beschreibt Kollisionen pro Volumen und Zeit — n ist die
Partikelkonzentration, die hier über `n_phys = ΣW/Vc` aus `a0`, `W` und dem
Kontrollvolumen `Vc` folgt (§1.3 der Projekt-README). `Vc` legt damit fest,
in welcher Konzentration deine Partikel überhaupt erzeugt werden. Wichtig:
`Vc` taucht in der Propensity-Berechnung selbst **nicht** als 1/Vc-Faktor auf —
die Kollisionsrate hängt in diesem Code direkt an `W`, nicht an `W/Vc`. Ein
anderes `Vc` bei gleichem `a0`/`W` ändert also `n_phys`, aber nicht automatisch
die simulierte Rate; `corr_beta`/`P1` müssen für das gewählte `Vc` mitkalibriert
sein. (Für `S` gilt das nicht: Bruch ist eine Eigenschaft eines einzelnen
Partikels, keine Paarwechselwirkung, daher ohne Konzentrationsabhängigkeit.)

### Aggregation — `compute_beta` → β [m³/s]

| Name | Formel | Parameter (Default) |
|---|---|---|
| `shear_chin1998` | `corr_beta · G · (r₁+r₂)³` | `corr_beta`=1e-3, `g`=1.0 |
| `brownian_tsouris1995` | `corr_beta · 2kT(r₁+r₂)² / (3μ r₁r₂)` | `corr_beta`=1.0, `temperature`=293.0, `viscosity`=1e-3 |
| `constant` | `corr_beta` | `corr_beta`=1e-10 |
| `sum` | `corr_beta · (V₁+V₂)` | `corr_beta`=1e-9 |
| `eke_darelius2005` | `corr_beta · n^c · (r₁+r₂)² · √(1/r₁³+1/r₂³)` | `corr_beta`=1e-11, `n_mixer`=20.0, `c_mixer`=0.0995 |
| `etm_darelius2005` | `corr_beta · n^c · (r₁+r₂)² · √(1/r₁⁶+1/r₂⁶)` | `corr_beta`=1e-16, `n_mixer`=20.0, `c_mixer`=0.0995 |

### Breakage — `compute_rate` → S [1/s]

| Name | Formel | Parameter (Default) |
|---|---|---|
| `power_law` | BREAKRVAL-Schalter, 3/4: `P1·G·V^P2` | `p1`=3e-2, `p2`=1.0, `g`=1000.0, `breakrval`=1 |
| `powerlaw_rumpf` | `P1·G·V^P2 / σ(ε, S)` | + `k`=2.5, `alpha`=1.15, `gamma`=0.072, `delta`=0.0, `x_s`=None, `poro_max`=0.9999 |
| `powerlaw_rumpf_dynamic` | `P1·n^c·V^P2 / σ(ε, S)` | wie oben, aber `n_mixer`=20.0, `c_mixer`=0.6699 **statt** `g` |

### Porosity Growth

| Name | Parameter (Default) |
|---|---|
| `volume_mixing` | – |
| `incomplete_mixing` | `trapped_pore_fraction`=0.1, `nucleation_porosity`=0.0 |
| `cone_model` | `k_agg`=1.0, `k_break`=1.0 |

### Agglomerations-Akzeptanz — `accept_collision` → `bool`

| Name | Parameter (Default) |
|---|---|
| `stokes_krit` | `U_coll`=1.0, `binder_viscosity`=5, `rho_solid`=2500.0, `rho_liquid`=1000.0, `h_a`=5e-7, `debug`=False |
| `stokes_dynamik` | `U_coll_ref`=0.0794, `n_mixer`=20.0, `c_vel`=0.2852, + die vier oben |
| `fittable` | `u_acc`=1.0 |

### Liquid Distribution

| Name | Parameter (Default) |
|---|---|
| `uniform_weighted` | – |

### Continuous Processes

| Name | Parameter (Default) |
|---|---|
| `porosity_compression` | `rate`=0.02, `min_porosity`=0.3 |
| `liquid_internalization` | `k_int`=1e12 |
| `liq_internalisation_agglomeration` | – |

---

## Verfügbare Kernel

### Aggregation

#### `shear_chin1998`

Scherinduzierte Agglomeration (Chin et al. 1998), Momentenform vorhanden.

```
β(i,j) = corr_beta · G · (r_i + r_j)³
```

| Parameter | Einheit | Bedeutung | typisch |
|---|---|---|---|
| `corr_beta` | – | Kollisionseffizienz / Kalibrierung | 1e-4 … 1e-2 |
| `g` | 1/s | Scherrate | 100 … 5000 |

Setzt eine kontinuierliche **Flüssigphase** voraus — G ist deren
Geschwindigkeitsgradient. Für einen Mischer mit Gasphase siehe `eke_darelius2005`.

#### `brownian_tsouris1995`

Brownsche Diffusion (Tsouris et al. 1995). Relevant für Partikel < 1 µm.

```
β(i,j) = corr_beta · 2kT(r_i+r_j)² / (3μ r_i r_j)
```

| Parameter | Einheit | Bedeutung |
|---|---|---|
| `corr_beta` | – | Kollisionseffizienz |
| `temperature` | K | Temperatur |
| `viscosity` | Pa·s | dynamische Viskosität der kontinuierlichen Phase |

#### `constant` / `sum`

Validierungskernel mit analytisch lösbarer Smoluchowski-Gleichung.
`β = corr_beta` bzw. `β = corr_beta · (V_i + V_j)`. Beide nehmen **nur**
`corr_beta`.

#### `eke_darelius2005` — Equipartition of Kinetic Energy

Für einen Mischer, dessen **Kontinuum ein Gas** ist: es gibt keinen
Flüssigkeits-Geschwindigkeitsgradienten G. Angetrieben wird stattdessen über die
Mischerdrehzahl.

```
β(i,j) = corr_beta · n_mixer^c_mixer · (r_i + r_j)² · √(1/r_i³ + 1/r_j³)
```

Der Wurzelterm ist ∝ √(1/m_i + 1/m_j): die fluktuierende kinetische Energie ist
gleich auf alle Granulate verteilt, ein leichtes bewegt sich also schneller.

> **Das heißt nicht, dass der Prozess trocken ist.** Binder wird weiterhin
> zugegeben, Granulate haben Flüssigkeitsbrücken und Sättigung. Trocken ist nur
> die Phase, durch die sich die Granulate bewegen.

| Parameter | Bedeutung |
|---|---|
| `corr_beta` [m^(5/2)/s] | Fit-Vorfaktor, schluckt DEM-Vorfaktor und Einheitenkonvention |
| `n_mixer` | Mischergeschwindigkeit, siehe [mixer_speed.py](mixer_speed.py) |
| `c_mixer` | Exponent, Default `C_FREQ` = 0.0995 |

**Größenabhängigkeit:** für gleich große Partikel β ∝ √r — gegenüber dem
Scherkernel (β ∝ r³) also drastisch schwächer. Deutlich weniger Runaway-Wachstum,
engere Verteilung. Kleine Partner werden stark bevorzugt (β ∝ r_klein^(−3/2)).

**Keine Momentenform:** die Wurzel einer *Summe* ist nicht separierbar. Der
Kernel bekommt den kompilierten O(n²)-Pfad, nicht den O(n log n)-Pfad.

#### `etm_darelius2005` — Equipartition of Translational Momentum

Gleiches Bild, andere Annahme: jedes Granulat bekommt denselben zufälligen
**Impuls** statt derselben Energie, also v ∝ 1/m.

```
β(i,j) = corr_beta · n_mixer^c_mixer · (r_i + r_j)² · √(1/r_i⁶ + 1/r_j⁶)
```

`corr_beta` hat hier die Einheit [m⁴/s], **nicht** die [m^(5/2)/s] von EKE — die
beiden Vorfaktoren sind nicht austauschbar. β ∝ 1/r für gleich große Partikel,
also noch stärkere Bevorzugung der Feinanteile. Darelius fand ETM bei *hohen*,
EKE bei *niedrigen* Rührerdrehzahlen besser.

---

### Breakage

#### `power_law`

Klassischer BREAKRVAL-Schalter, deckungsgleich mit
`pbe_core.func.jit_kernel_break.calc_break_rate_1d`:

| BREAKRVAL | Formel |
|---|---|
| 1 | `S = P1` |
| 2 | `S = P1 · V` |
| 3, 4 | `S = P1 · G · V^P2` |

`P2` ist der **Volumen**exponent, G geht linear ein. `pl_v` / `pl_q` gehören zur
Bruch**funktion** (Fragmentgrößenverteilung) und werden hier abgelehnt.

#### `powerlaw_rumpf`

Wie oben, geteilt durch die Granulatfestigkeit nach Rumpf:

```
S(V, ε, S_sat) = S_base(V) / σ(ε, S_sat)
```

σ in drei Sättigungsregimen (trocken < 0.3, Übergang, nass > 0.8):

```
trocken:  σ = (1-ε)/ε · k · γ / x_s
nass:     σ = 6α · (1-ε)/ε · γ·cos(δ) / x_s · S
```

Sonderfälle: ε = NaN (Vollkörper) oder ε ≤ 0 → Rate 0 (keine Poren, kein
Versagensmechanismus — das ist der Grenzwert der Formel, nicht ein Abbruch).
ε > `poro_max` wird geklemmt. Sättigung NaN wird als trocken behandelt.

#### `powerlaw_rumpf_dynamic`

Identische Physik, aber **drehzahlgetrieben statt schergetrieben**:

```
S_base = P1 · n_mixer^c_mixer · V^P2
```

`c_mixer` = `C_BREAK` = 0.6699 = Stoßfrequenz × Stoßenergie (`C_FREQ + 2·C_VEL`).
G war eine *Belastungsfrequenz*; die Übersetzung besteht aus wie oft belastet wird
(n^0.0995) mal wie hart (v_rel² ~ n^0.5704).

**Kein Massenterm** — und das ist ein Ergebnis, keine Auslassung. Unter EKE
trägt jedes Granulat dieselbe Energie E₀, also v′ ∝ m^(−1/2), und die Stoßenergie
eines Paares ist

```
E_stoß = ½·μ·v_rel² = ½ · m_i m_j/(m_i+m_j) · 2E₀(1/m_i + 1/m_j) = E₀
```

Die Masse kürzt sich exakt weg. Ein massenabhängiger Term würde der Annahme
widersprechen, auf der der Agglomerationskernel steht. Eine massenabhängige
Variante bräuchte eine andere Hypothese (gleicher Impuls → E ∝ 1/m, gleiche
Geschwindigkeit → E ∝ m) und gehört dann in einen **eigenen** Kernel.

Der Kernel nimmt **kein** `g`. Wer eine Scherrate übergibt, bekommt einen Fehler.

---

### Porosity Growth

| Name | Modell |
|---|---|
| `volume_mixing` | volumengewichtete Mischung der Elternporositäten |
| `incomplete_mixing` | zusätzlich eingeschlossene Kontaktporen (`trapped_pore_fraction`) |
| `cone_model` | Kegelmodell für Kontaktzone, mit `k_agg` / `k_break` als Fitfaktoren |

`cone_model` ist der einzige, der auch bei Bruch Porenvolumen *vernichtet* — was
die Gegenoperation `compute_externalization` in
`liq_internalisation_agglomeration` überhaupt erst auslöst.

---

### Agglomerations-Akzeptanz

Läuft **nach** der Partnerwahl, **vor** der Ausführung. Entscheidet, ob die
Kollision zur Agglomeration führt oder abprallt — effektiv ein α von 1 oder 0.

#### `stokes_krit`

Stokes-Kriterium nach Braumann et al. (2007):

```
St      = m_harm · U / (3π η R_harm²)          (Gl. 8)
St_crit = (1 + 1/e_coag) · ln(h / h_a)          (Gl. 10)
```

Agglomeration bei `St < St_crit`. Ohne Flüssigkeitsfilm (`h = 0`) wird immer
abgelehnt — trockene Partikel agglomerieren nicht.

#### `stokes_dynamik`

Gleiche Physik, aber die Stoßgeschwindigkeit hängt an der Mischerdrehzahl:

```
U_coll(n) = U_coll_ref · n_mixer^c_vel
```

Direkte Potenzform, kein Referenzpunkt — anders als eine frühere Version dieses
Kernels, die `U_coll_ref · (n_mixer/n_ref)^c_vel` rechnete. Der Referenzpunkt
`n_ref` erwies sich als nicht identifizierbar: aus einem einzigen Kalibrierlauf
lassen sich `U_coll_ref` und `n_ref` nicht trennen (`U_coll_ref · (n/n_ref)^c
= (U_coll_ref · n_ref^-c) · n^c`), und `c_vel = 0` leistete als parameterfreie
Abschaltung ohnehin schon dasselbe.

**`U_coll_ref` ist hier bewusst der DEM-Fit-Vorfaktor 1:1** (0.0794 aus
`y = 0.0794·x^0.2852`), nicht nur der Exponent wie bei den übrigen Kerneln.
Explizit unphysikalisch — `stokes_krit`s `U_coll` war ohnehin nur eine geratene
Größenordnung, keine Kalibrierung. Konsequenz: `n_mixer` muss für diesen Kernel
auf derselben Skala wie die DEM-Studie gelesen werden (m/s, ~5–40), weil kein
freier Vorfaktor mehr übrig ist, der eine Einheitenabweichung auffangen könnte.

Bei `n_mixer == 1.0` ist der Kernel **bitgenau identisch** zu `stokes_krit` mit
`U_coll = U_coll_ref` (`1^c == 1` für jedes `c`) — ein mathematischer, kein
physikalischer Ankerpunkt.

#### `fittable`

Reiner Fit-Parameter: akzeptiert mit Wahrscheinlichkeit `u_acc`. `u_acc = 0`
lehnt jede Kollision ab — nützlich in Tests, um Prozesse zu isolieren.

---

### Continuous Processes

Laufen per Operator-Splitting **nach** jedem MC-Ereignis.

| Name | Wirkung |
|---|---|
| `porosity_compression` | ε → `min_porosity`, exponentiell mit `rate`; V_solid bleibt erhalten, V_pore schrumpft |
| `liquid_internalization` | kapillares Einziehen: verschiebt Flüssigkeit von außen nach innen, `l_total` bleibt erhalten |
| `liq_internalisation_agglomeration` | ereignisbasiert: Flüssigkeit, die beim Verschmelzen in Kontaktporen eingeschlossen wird |

> **Achtung, das war ein echter Fehler:** die Kompression schreibt `porosity`
> **und** `V_flat[-1]`, ändert also den Durchmesser. Da jeder
> Agglomerationskernel eine Funktion der Radien ist, veralten dadurch die
> gespeicherten Propensities. Seit 20.08.2026 löst der Solver nach
> `continuous_processes.step()` einen Rebuild aus. Siehe
> `../../../docs/historical/Nucleation_Propensity_Blockade.md`, Abschnitt 8.

---

### Mischerdrehzahl: `mixer_speed.py`

Vier Kernel werden von der Mischerdrehzahl angetrieben: `eke_darelius2005`,
`etm_darelius2005`, `stokes_dynamik`, `powerlaw_rumpf_dynamic`. Sie lesen alle
**dieselbe** physikalische Größe — es gibt einen Mischer, der sich mit einer
Drehzahl dreht.

| Konstante | Wert | Bedeutung |
|---|---|---|
| `N_MIXER_DEFAULT` | 20.0 | Mitte des DEM-Bereichs 5–40 m/s |
| `C_FREQ` | 0.0995 | Kollisionsfrequenz ~ n^C_FREQ |
| `C_VEL` | 0.2852 | mittlere Relativgeschwindigkeit ~ n^C_VEL |
| `C_BREAK` | 0.6699 | `C_FREQ + 2·C_VEL`, Frequenz × Stoßenergie |

Übertragen wurden nur die **Exponenten**; die absoluten DEM-Zahlen beschreiben
eine andere Maschine und ein anderes Pulver. `n_mixer` ist deshalb nur bis auf
einen konstanten Faktor definiert — ein Einheitenwechsel skaliert jeden Kernel um
eine Konstante, die die Fitparameter schlucken.

`assert_consistent_mixer_speed` bricht mit einem **Fehler** ab, wenn zwei aktive
Kernel unterschiedliche `n_mixer` tragen. Nicht als Warnung: zwei Drehzahlen für
einen Mischer haben keine sinnvolle Deutung, und der Lauf sähe plausibel aus.

---

## Empfohlene Konfiguration

**Die aktuell empfohlene Konfiguration ist die aus
`Trials/test_powerlaw_rumpf_full.py`** — scher- und nicht drehzahlgetrieben:

```python
solver = MCPBESolver(
    dim=1,
    agg_kernel_name='shear_chin1998',
    agg_kernel_params={'corr_beta': 4.0, 'g': 1000.0},

    agg_acceptance_kernel_name='stokes_krit',
    agg_acceptance_kernel_params={
        'U_coll': 0.5, 'binder_viscosity': 0.1,
        'rho_solid': 500.0, 'rho_liquid': 1000.0, 'h_a': 500e-9,
    },

    break_kernel_name='powerlaw_rumpf',
    break_kernel_params={
        'p1': 4e13, 'p2': 1.0, 'g': 1000.0, 'breakrval': 4,
        'k': 2.5, 'alpha': 1.0, 'gamma': 0.072, 'delta': 0.0,
        'x_s': None,                       # aus X0 berechnen
    },

    porosity_growth_kernel_name='cone_model',
    porosity_growth_kernel_params={},

    porosity_compression_kernel_name='porosity_compression',
    porosity_compression_kernel_params={'rate': 0.02, 'min_porosity': 0.2},

    liquid_internalization_kernel_name='liquid_internalization',
    liquid_internalization_kernel_params={'k_int': 1e12},

    liq_internalisation_agglomeration_kernel_name='liq_internalisation_agglomeration',
    liq_internalisation_agglomeration_kernel_params={},
)
solver.process_type = "mix"                # Agglomeration + Bruch
solver.agg_propensity_mode = "moment"
solver.agg_dW_min, solver.agg_dW_max = 1.0, 20.0
solver.break_dW_max = 50.0
```

`g` muss in Agglomerations- und Bruchkernel **denselben** Wert haben: beide
beschreiben denselben Mischer. Weichen sie ab, warnt `kernel_integration`.

### Der drehzahlgetriebene Zweig ist in Entwicklung

`eke_darelius2005`, `etm_darelius2005`, `stokes_dynamik` und
`powerlaw_rumpf_dynamic` sind implementiert, getestet und lauffähig, aber **noch
nicht kalibriert und noch nicht gegen Messdaten validiert**. Sie sind derzeit
keine Empfehlung, sondern ein Entwicklungszweig.

Was konkret offen ist:

* **`corr_beta` und `p1` müssen neu gefunden werden.** In den geschertriebenen
  Kerneln steckt G ≈ 1000 im Vorfaktor; ersetzt wird es durch `n^c` in der
  Größenordnung 1–10. Die bisherigen Fitwerte sind damit um Größenordnungen
  daneben — mit den Kernel-Defaults liegt die Gesamtpropensity bei ~5e-3 1/s,
  also ~0.01 Ereignisse in 2 s.
* **Nicht gegen Messdaten geprüft.** Es gibt ein experimentelles Datenset, das
  am Ende einmal über alle Prozesse gefittet wird. Bis dahin ist unbekannt, ob
  EKE oder ETM das System besser beschreibt.
* **Konsistenz Agglomeration ↔ Bruch.** Ein Lauf sollte entweder durchgehend
  schergetrieben oder durchgehend drehzahlgetrieben sein. Der Querabgleich
  erzwingt nur, dass die *Drehzahlen* übereinstimmen — nicht, dass man nicht
  `shear_chin1998` mit `powerlaw_rumpf_dynamic` mischt.

Ein drehzahlgetriebener Lauf sieht so aus (Beispiel, **nicht** kalibriert):

```python
agg_kernel_name='eke_darelius2005',
agg_kernel_params={'corr_beta': 1e-11, 'n_mixer': 20.0},

agg_acceptance_kernel_name='stokes_dynamik',
agg_acceptance_kernel_params={
    'U_coll_ref': 0.0794, 'n_mixer': 20.0,       # DEM-Fit 1:1, s.o.
    'binder_viscosity': 0.1, 'rho_solid': 500.0,
    'rho_liquid': 1000.0, 'h_a': 500e-9,
},

break_kernel_name='powerlaw_rumpf_dynamic',
break_kernel_params={
    'p1': 4e13, 'p2': 1.0, 'n_mixer': 20.0, 'breakrval': 4,
    'k': 2.5, 'alpha': 1.0, 'gamma': 0.072, 'delta': 0.0, 'x_s': None,
},
```

`n_mixer` muss überall gleich sein, sonst bricht der Solver beim Aufbau ab.

---

## Eigenen Kernel schreiben

1. `blueprint.py` der Kategorie kopieren — sie sind Vorlagen, keine Kernel, und
   in keiner Registry eingetragen. Siehe [BLUEPRINTS.md](BLUEPRINTS.md).
2. Klasse umbenennen, `name` setzen.
3. `get_default_params`, `__init__`, `validate_params` und die Pflichtmethode
   der Kategorie implementieren.
4. **Die Grundregel einhalten:** nichts aus einem Nachbar-Kernel importieren,
   nicht von einem anderen Kernel erben. Gebraucht wird etwas aus einem anderen
   Kernel? Dann kopieren, nicht importieren.
5. In die Registry der Kategorie eintragen.
6. In `Trials/test_dry_mixer_kernels.py` in die Listen der Abschnitte 9 und 11
   aufnehmen, damit Eigenständigkeit und Weitergabe-Tauglichkeit geprüft werden.
7. Hat der Kernel einen kompilierten Batch-Pfad verdient, siehe
   `aggregation/jit_kernels.py`: reicht `beta = f(p0, r_i, r_j)`, genügen ein
   Eintrag in `KERNEL_IDS`, ein Zweig in `_beta_of` und einer in `kernel_p0`.
   Ist er zusätzlich *separierbar*, kommt er in `MOMENT_KERNELS` und braucht
   eine Zerlegung in `separable_tables`.

---

## Zustandsgrößen: Porosität und Flüssigkeit

### Volumensemantik

```
V_flat[:dim, i]   Feststoffvolumen je Komponente   (erhalten bei Agglomeration)
V_flat[-1, i]     Trockenvolumen = V_solid + V_pore
V_solid = V_dry · (1 - porosity)
```

`porosity` ist **NaN** für Vollkörper — dann gilt `V_solid == V_dry`. NaN ist
kein Fehlerwert, sondern bedeutet „hat keine Poren". Jede Rechnung mit der
Porosität muss das abfangen:

```python
poro = solver.porosity[i]
if np.isnan(poro):
    v_solid = v_dry              # Vollkörper
else:
    v_solid = v_dry * (1.0 - poro)
```

### Flüssigkeit

```
solver.liquid_volume[i]   Gesamtflüssigkeit des Partikels [m³]
solver.saturation[i]      S = V_liq_intern / V_pore, in [0, 1]
V_pore     = V_dry · porosity
V_liq_int  = V_pore · saturation
V_liq_ext  = liquid_volume - V_liq_int        (äußerer Film)
```

Für Vollkörper ist die gesamte Flüssigkeit extern. `solver.get_V_liquid_external()`
kapselt das.

### Intensiv vs. extensiv

`liquid_volume`, `porosity`, `saturation` und `V_flat` sind **pro physikalischem
Partikel** gespeichert (intensiv). `W` zählt, wie viele physikalische Partikel
ein Rechenpartikel vertritt (extensiv). Wird `W` eines Elternteils reduziert,
dürfen seine intensiven Größen sich **nicht** ändern.

---

## Fehlerbehandlung

### Unbekannte Parameternamen werden abgelehnt

Seit 17.08.2026 prüft `reject_unknown_params` jeden übergebenen Namen gegen
`get_default_params()` plus `get_optional_params()`:

```python
get_continuous_kernel('liquid_internalization', k_intern=1e12)
# ValueError: Unknown parameter(s) for kernel 'liquid_internalization':
# 'k_intern' (did you mean 'k_int'?). Accepted parameters: ['k_int']. ...
```

Der Grund: ein Tippfehler überschrieb vorher nichts, sondern legte einen
zusätzlichen Dict-Eintrag an, den niemand liest — der Kernel lief still auf
seinem Default weiter. Das war nicht hypothetisch: mehrere Trial-Skripte
übergaben `k_intern` statt `k_int` und liefen sämtlich auf dem Default.

### Typische Fehler

| Meldung | Ursache |
|---|---|
| `Unknown aggregation kernel 'x'` | Name nicht in der Registry — Tippfehler oder Eintrag vergessen |
| `Unknown parameter(s) ... did you mean` | Parametername falsch geschrieben |
| `Conflicting mixer speeds across kernels` | zwei Kernel mit verschiedenem `n_mixer` |
| `k must be in range [2.2, 2.8]` | Rumpf-Fitparameter außerhalb des physikalisch belegten Bereichs |
| `delta must be in [0, π/2)` | cos(δ) = 0 gäbe null Festigkeit und unendliche Bruchrate |
| `breakrval=5 is not a valid ...` | gab es in der Referenzimplementierung nie |
| `'pl_v' is not a breakage RATE parameter` | gehört zur Bruch*funktion*, nicht zur Rate |

---

## Performance

Die Propensity-Neuberechnung dominiert die Laufzeit — sie läuft einmal je
akzeptiertem Ereignis über alle aktiven Partikel.

| Kernel | pairwise | moment |
|---|---|---|
| `constant`, `sum`, `shear_chin1998`, `brownian_tsouris1995` | O(n²) | **O(n log n)** |
| `eke_darelius2005`, `etm_darelius2005` | O(n²) | – nicht separierbar |

`solver.agg_propensity_mode = "moment"` wählt die Momentenform, wo es sie gibt,
und fällt sonst still auf den kompilierten pairwise-Pfad zurück.

Ein Kernel *ohne* Eintrag in `KERNEL_IDS` landet im generischen Python-Pfad —
O(n²) Python-Aufrufe je Ereignis. Der Solver warnt dann einmalig. Auch 2D-Setups
laufen dort, weil α dann vom Komponentengemisch des Paares abhängt.

---

## Referenzen

### Wissenschaftliche Arbeiten

* Chin, W.C. et al. (1998) — scherinduzierte Flockung in Rührbehältern
* Tsouris, C. et al. (1995) — Brownsche Diffusion als kontrollierender Mechanismus
* Smoluchowski, M. (1917) — Koagulationskinetik kolloider Lösungen
* Hounslow, M.J. (1998) — *The population balance as a tool for understanding
  particle rate processes*, KONA 16, 179–193 (EKE- und ETM-Kernel)
* Darelius, A. et al. (2005) — *High shear wet granulation modelling — a
  mechanistic approach using population balances*, Powder Technology 160, 209–218
* Braumann, A. et al. (2007) — Granulation mit heterogener Binderverteilung
  (Stokes-Kriterium)
* Rumpf, H. (1979) — *Grundlagen der Agglomeration*
* Iveson, S.M. et al. (2001) — Sättigungsregime, Powder Technology
* Ji & Rhein — gewichtetes DSMC, Bias-Korrektur

### Interne Dokumente

| Datei | Inhalt |
|---|---|
| [BLUEPRINTS.md](BLUEPRINTS.md) | Vorlagen für neue Kernel |
| [mixer_speed.py](mixer_speed.py) | Herkunft der DEM-Exponenten |
| [porosity_growth/CONE_MODEL_README.md](porosity_growth/CONE_MODEL_README.md) | Kegelmodell im Detail |
| [COMPRESSION_MIGRATION.md](COMPRESSION_MIGRATION.md) | Umstieg vom entfernten Compression-Modul |
| `../../../docs/README.md` | Solver-Grundlagen |
| `../../../docs/README.md` | Gesamtüberblick |
| `../../../docs/historical/Bias_Correction_und_Gewichtsdisziplin.md` | Herleitung der Propensity-Korrektur |
| `../../../docs/historical/Nucleation_Propensity_Blockade.md` | Propensity-Aktualität, beide Ursachen |
