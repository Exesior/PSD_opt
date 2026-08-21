# PSD_opt

Populationsbilanz-Solver für Partikelprozesse — Agglomeration, Bruch,
Nassgranulation — und die Werkzeuge, um ihre Modellparameter an Messdaten zu
fitten.

Das Repository ist ein **Monorepo aus eigenständigen Python-Paketen**. Jedes hat
sein eigenes `pyproject.toml` und kann einzeln installiert werden.

---

## Die Pakete

| Paket | Inhalt |
|---|---|
| **`mcpbe/`** | Monte-Carlo-Solver. Enthält `wmcpbe` — den **gewichteten** Solver für Nassgranulation, um den es in diesem Dokument geht — sowie den älteren ungewichteten `mcpbe` und die vereinfachte Variante `wmcpbe_granulation`. `src/historical_versions/` archiviert frühere Entwicklungsstände (nicht Teil des aktuellen Codes, siehe Kommentar am Kopf jeder Datei dort). |
| `dpbe/` | Diskreter Solver (Finite Volumen, Cell Average Technique). Schwesterprojekt zu `mcpbe`: dieselben Kernel, anderer Lösungsweg. |
| `qmom/` | Momentenmethoden (QMOM, HyQMOM, GQMOM, CQMOM). |
| `pbe-core/` | Gemeinsame Basis aller Solver: `BaseSolver`, JIT-Kernel, Plotter, mathematische Hilfsfunktionen. |
| `pbe-optimizer/` | Parameteroptimierung auf Ray Tune. Fittet Kernel-Parameter gegen gemessene Partikelgrößenverteilungen. |
| `lmc/` | Lattice-Monte-Carlo: simuliert Bindungsbruch in einzelnen Aggregaten. Liefert Fragmentverteilungen für den Bruchprozess. |
| `agggenerator/` | Erzeugt und verwaltet Aggregat-Pools für `lmc`. |
| `breakage-rate-model/` | ML-Modelle (MLP, ANN) für Bruchraten als Alternative zu geschlossenen Formeln. |
| `scripts/` | Ausführbare Beispiele, Validierungs- und Auswerteskripte. |

> Der Rest dieses Dokuments beschreibt **`wmcpbe`**. Für die anderen Pakete
> siehe deren jeweilige `README.md`.

---

## Installation und Ausführen

Es gibt kein Paket im Repository-Wurzelverzeichnis — jedes Paket wird einzeln
installiert:

```bash
pip install -e pbe-core
pip install -e mcpbe
```

Ein Lauf wird aus `mcpbe/src` heraus gestartet:

```bash
python -m wmcpbe.Trials.test_powerlaw_rumpf_full
```

Bei Sonderzeichen in der Ausgabe zusätzlich `PYTHONIOENCODING=utf-8` setzen,
sonst brechen `print()`-Aufrufe auf einer cp1252-Konsole ab.

---

# Teil 1 — Das große Bild

## 1.1 Was simuliert wird

`wmcpbe` ist ein **gewichteter Monte-Carlo-Löser für Populationsbilanz-
gleichungen**, zugeschnitten auf **Nassgranulation**:

```
   trockene poröse Primärpartikel
              │
              │  Nucleation: Flüssigkeitstropfen werden zugegeben
              ▼
   benetzte Partikel  ──────────┐
              │                  │
              │ Agglomeration    │ kontinuierliche Prozesse:
              │ (Kollision +     │  · Flüssigkeit wandert von
              │  Haftung)        │    außen in die Poren
              ▼                  │  · Poren werden unter Scherung
        Granulate                │    zusammengedrückt
              │                  │
              │ Breakage         │
              │ (Bruch)          │
              ▼                  │
        Fragmente  ◄─────────────┘
```

Gesucht ist die **Partikelgrößenverteilung (PSD)** über der Zeit, die gegen
experimentelle Daten gefittet werden soll.

## 1.2 Warum Monte Carlo und warum gewichtet

Eine Populationsbilanz ist eine Integro-Differentialgleichung über einer
kontinuierlichen Größenkoordinate. Statt sie zu diskretisieren — das macht das
Schwesterprojekt `dpbe` — verfolgt der Monte-Carlo-Ansatz **einzelne Partikel**
und würfelt Ereignisse aus. Vorteil: ein mehrdimensionaler Zustand (Volumen
*und* Porosität *und* Sättigung *und* Flüssigkeit) kostet nicht exponentiell
mehr Speicher.

Nachteil: ein echter Granulator enthält ~10¹² Partikel. Deshalb **gewichtet**:

> Jedes **Rechenpartikel** steht stellvertretend für `W` **physische Partikel**.

`W` ist damit die einzige *extensive* Größe im Zustand. Alles andere — Volumen,
Porosität, Sättigung, Flüssigkeitsmenge — ist **intensiv**, also „pro physischem
Partikel" definiert, und ändert sich beim Kopieren oder beim Reduzieren von `W`
nicht.

## 1.3 Das Kontrollvolumen `Vc`

`Vc` verbindet die simulierte Stichprobe mit der realen Anzahldichte:

```
n_phys = Σ(W) / Vc        physikalische Anzahldichte [1/m³]
n_comp = a_tot            Anzahl Rechenpartikel
```

Agglomeration verbraucht Rechenpartikel. Fällt ihre Zahl unter die Hälfte der
Kapazität, **verdoppelt** `_maybe_double_control_volume()` das Kontrollvolumen
und dupliziert die Population. `n_phys` bleibt dabei konstant — es wird
lediglich ein größerer Ausschnitt desselben Systems betrachtet.

## 1.4 Der Ereigniszyklus

Der Hauptloop steht in `mcpbe_base.py::solve()`. Pro Durchlauf:

```
 ┌─ 1. Zeitschritt bestimmen ──────────────────────────────────┐
 │    dt aus der Gesamt-Propensity (Gillespie-artig)           │
 └─────────────────────────────────────────────────────────────┘
 ┌─ 2. EIN Monte-Carlo-Ereignis ausführen ─────────────────────┐
 │    process_type = "agglomeration" → _do_one_agg()           │
 │                 = "breakage"      → _do_one_break()         │
 │                 = "mix"           → Würfel zwischen beiden  │
 └─────────────────────────────────────────────────────────────┘
 ┌─ 3. Snapshot speichern, falls ein t_vec-Punkt erreicht ist ─┐
 └─────────────────────────────────────────────────────────────┘
 ┌─ 4. Kontinuierliche Prozesse (Operator Splitting) ──────────┐
 │    continuous_processes.step(t, dt)   Internalisierung +    │
 │                                        Kompression          │
 │    nucleation.step(t, dt)              Tropfenzugabe        │
 └─────────────────────────────────────────────────────────────┘
 ┌─ 5. Aufräumen ──────────────────────────────────────────────┐
 │    ggf. Vc verdoppeln, ggf. Rekonstruktion (n_comp senken)  │
 └─────────────────────────────────────────────────────────────┘
```

**Warum Operator Splitting?** Nucleation und Kompression laufen auf einer
*eigenen* Zeitskala — sie sind keine gewürfelten Ereignisse, sondern
kontinuierliche Prozesse. Sie werden deshalb *nach* dem MC-Ereignis mit dem
gerade vergangenen `dt` angewendet. Dadurch kann eine Porositäts- oder
Flüssigkeitsänderung die Zeitwahl des *aktuellen* Ereignisses nicht mehr
beeinflussen; die Ereignisstatistik bleibt sauber.

## 1.5 Vererbung vs. Komposition

Der Solver wird in `mcpbe.py` zusammengesetzt:

```python
class MCPBESolver(MCPBEPost, MCPBEBreak, MCPBEAgg, MCPBEBase, ReconstructionMixin):
```

Der tatsächliche MRO (Method Resolution Order) ist:

```
MCPBESolver → MCPBEPost → MCPBEBreak → MCPBEAgg → MCPBEBase
            → MCPBETimeHelper → BaseSolver → ReconstructionMixin → object
```

`MCPBETimeHelper` und `BaseSolver` (aus `pbe-core`) kommen über `MCPBEBase`
herein. Zu beachten: **`ReconstructionMixin` löst zuletzt auf**, also *nach*
`BaseSolver` — bei einer Namenskollision gewinnt `BaseSolver`.

Daneben stehen zwei **Handler**, die keine Mixins sind:

| | Mixins | Handler |
|---|---|---|
| Beispiele | `MCPBEAgg`, `MCPBEBreak`, `MCPBEPost`, `ReconstructionMixin` | `NucleationHandler`, `ContinuousProcessesHandler` |
| Verhältnis zum Solver | Teil der Klasse (Vererbung) | eigenes Objekt mit `self.solver`-Rückreferenz |
| Aufruf | **innerhalb** der Ereignisschleife, wenn das Ereignis gewürfelt wurde | **nach** jedem Ereignis über `.step(t, dt)` |
| Zeitskala | Ereigniszeit | eigene, kontinuierliche |

Angelegt werden sie über `solver.create_nucleation_handler(...)` und
`solver.create_continuous_processes_handler(...)`.

## 1.6 Datei-Landkarte (`mcpbe/src/wmcpbe/`)

| Datei | Rolle |
|---|---|
| `mcpbe.py` | Zusammenbau der `MCPBESolver`-Klasse, Handler-Factories |
| `mcpbe_base.py` | Partikelarrays, Kontrollvolumen, **Haupt-Solve-Loop**, Kapazitätsverwaltung |
| `mcpbe_agg.py` | Agglomerations-Physik: Paarauswahl, Paketgröße, Verschmelzung, Propensity-Rebuild |
| `mcpbe_break.py` | Bruch-Physik: Rate, Fragmentanzahl, Fragmentverteilung |
| `mcpbe_post.py` | Auswertung: Momente, PSD, Statistik |
| `mcpbe_nucleation.py` | `NucleationHandler` — Flüssigkeitszugabe |
| `mcpbe_continuous_processes.py` | `ContinuousProcessesHandler` — Internalisierung + Kompression |
| `mcpbe_time_helper.py` | Ereignisuhr: Zeitschritt und Paketgrößen |
| `reconstruction_mixin.py` | Partikelzahl senken (CAM, RS, 2PM, QMX) |
| `particle_merger.py` | Dedup neu erzeugter Partikel |
| `kernel_integration.py` | `KernelManager` — hält und verteilt die Physik-Kernel |
| `fenwick_new.py` | `FenwickSampler` — gewichtetes Ziehen in O(log n) |
| `lmc_adapter.py` | Fragmentverteilungen aus Lattice Monte Carlo |
| `mlp_breakage_adapter.py` | Bruchraten aus einem neuronalen Netz |
| `kernels/` | Die austauschbaren Physik-Modelle, siehe Teil 4 |
| `framework/` | Wiederholungen mit unabhängigen Seeds, siehe Teil 6 |
| `Trials/` | Ausführbare Studien und Testskripte |

---

# Teil 2 — Das Datenmodell

Alles, was der Solver über die Population weiß, steckt in wenigen NumPy-Arrays.
Sie sind auf `_cap` vorallokiert; gültig sind die Spalten `[0, a_tot)`.

## 2.1 `V_flat` — das Volumen-Array

Form `(dim+1, _cap)`. Für den Standardfall `dim = 1`:

```
V_flat[ 0, idx]  = V_solid   ← ERHALTUNGSGRÖSSE
V_flat[-1, idx]  = V_dry     ← V_solid + V_pore, KEINE Erhaltungsgröße
```

Bei `dim > 1` enthalten die Zeilen `0 … dim-1` die Feststoffvolumina der
einzelnen Komponenten, die letzte Zeile weiterhin `V_dry`.

**Warum diese Trennung?** Bei Nassgranulation passieren zwei Dinge gleichzeitig,
die man nicht vermischen darf:

- Der **Feststoff** wird nur umverteilt, nie erzeugt oder vernichtet.
- Das **Gesamtvolumen** ändert sich sehr wohl — Poren wachsen bei Agglomeration
  und schrumpfen unter Kompression.

Nur `V_solid` ist deshalb die Bilanzgröße. `V_dry` ist die Geometrie, aus der
der Durchmesser folgt.

## 2.2 Die begleitenden Arrays

| Array | Typ | Bedeutung |
|---|---|---|
| `W[idx]` | **extensiv** | Wie viele physische Partikel `idx` repräsentiert |
| `X[idx]` | intensiv | Durchmesser, aus `V_dry` abgeleitet |
| `porosity[idx]` | intensiv | ε = V_pore / V_dry |
| `liquid_volume[idx]` | intensiv | Gesamte Flüssigkeit **pro physischem Partikel** [m³] |
| `saturation[idx]` | intensiv | S = V_liq_intern / V_pore, ∈ [0,1] |

Dazu die Rechenhilfen `_r_agg`, `_break_rate`, `_delta_agg`, `_delta_break`.
Sie sind **indexgleich** mit den Zustandsarrays und werden genauso verwaltet.

**Gesamtmasse eines Zustands = Σ (V_solid[idx] · W[idx])** über alle aktiven
Indizes.

## 2.3 Die drei Flüssigkeitsbegriffe

Erfahrungsgemäß die häufigste Verwechslungsquelle:

```
liquid_volume  =  V_liq_intern            +  V_liq_extern
                  └─ in den Poren ─┘         └─ Film auf der Oberfläche ─┘
                  = V_pore · S               = liquid_volume − V_liq_intern
```

- **Intern** ist Flüssigkeit, die in die Poren eingedrungen ist. Sie macht das
  Granulat *fester* (Kapillarkräfte im Rumpf-Modell).
- **Extern** ist Oberflächenfilm. Er bildet beim Zusammenstoß die
  **Flüssigkeitsbrücke** — nur damit kann eine Kollision überhaupt haften
  (Stokes-Kriterium).

Gespeichert werden nur `liquid_volume` und `saturation`; die Aufteilung wird bei
Bedarf über `get_V_liquid_internal()` / `get_V_liquid_external()` berechnet. So
kann sie nicht inkonsistent auseinanderlaufen.

## 2.4 Sonderfall „Vollkörper"

Historisch wurden porenfreie Partikel mit `porosity = NaN` markiert. Moderne
Kernel liefern stattdessen `0.0`. **Beide werden unterstützt**, weil gespeicherte
Läufe und ältere Konfigurationen sonst brechen würden.

Jede Stelle, die aus Porosität etwas ableitet, behandelt das explizit, z. B. in
`get_V_solid`:

```python
V_solid = np.where(np.isnan(poro), V_dry, V_dry * (1.0 - poro))
```

Ohne diesen Zweig würde ein einziges NaN in jede darauf aufbauende Massensumme
durchschlagen. `V_solid = V_dry` ist für einen porenfreien Körper korrekt, weil
er keinen Porenraum hat.

## 2.5 Wachsen und Löschen von Partikeln

**`_append_particle_column(frag_vols)`** hängt ein Partikel an Index `a_tot` an.
Reicht die Kapazität nicht, wächst sie vorher über `_ensure_capacity_for` um
einen Faktor zwischen 1,1 und 2,0 — früh in der Simulation aggressiver, weil
dort die meisten Umschichtungen passieren.

**`_remove_particle_column(j)`** löscht per **Swap-with-last**: das letzte
Partikel wird nach `j` kopiert, `a_tot` sinkt um eins, der freigewordene Slot
wird geleert. Das ist O(1) statt O(n) — aber es bedeutet:

> **Jeder Index ≥ j kann sich durch eine Löschung verschieben.**

Zwei Schutzmechanismen:

1. Die kanonische Liste `_PARTICLE_ARRAY_NAMES` aller indexgleichen Arrays. Wer
   ein neues Array einträgt, bekommt Swap und Slot-Freigabe automatisch — man
   kann es nicht mehr vergessen.
2. `notify_index_swap(old, new)` benachrichtigt den `ParticleMerger`, **bevor**
   die Daten verschoben werden.

Code, der einen frisch erzeugten Kindindex über mehrere Elternlöschungen hinweg
halten muss, verfolgt ihn aktiv nach (Muster `child_current_idx`). Wo möglich
wird stattdessen **absteigend** gelöscht (`sorted({i, j}, reverse=True)`), dann
kann eine Löschung den offenen Index gar nicht erst verschieben.

---

# Teil 3 — Die Prozesse

## 3.1 Agglomeration (`mcpbe_agg.py`)

Ablauf pro Ereignis (`_do_one_agg`):

1. **`_select_pair(a)`** — zweistufiger Sampler. Partikel `i` wird über den
   Fenwick-Sampler proportional zur bias-korrigierten Propensity
   `R_i* = W_i · Σ_j W_j·β(i,j) / min(Δ_i, Δ_j)` gezogen, Partner `j` dann
   bedingt proportional zu `W_j·β(i,j)/Δ_ij`. Anschließend das physikalische
   Akzeptanzkriterium (`agglomeration_acceptance_kernel`, z. B. Stokes). Bei
   Ablehnung ist das Ereignis ein No-Op.
2. **`_compute_agg_dW(...)`** — Paketgröße `dW`: wie viele physische Kollisionen
   dieses Ereignis repräsentiert. **Selbstkollision (`i == j`) ist erlaubt** und
   wird gekappt auf `δ_ii = min(δ_i, W_i/2)`, weil ein solches Ereignis zwei
   physische Partikel aus demselben Pool verbraucht.
3. **`_merge_pair(i, j, dW)`** — erzeugt das Kindpartikel über
   `ParticleMerger.find_or_create`. `V_solid_child = V_solid_i + V_solid_j`,
   exakt und unabhängig vom Porositätskernel. `V_dry` und Porosität des Kindes
   kommen vom `porosity_growth_kernel`.
4. **`_consume_parent_weight(i, j, dW)`** — `W[i] -= dW`, `W[j] -= dW` (bzw.
   `W[i] -= 2·dW` bei `i == j`). Partikel mit `W <= 0` werden entfernt.
5. **`_refresh_samplers_after_agg()`** — Propensities neu berechnen.

**Bias-Korrektur (Ji & Rhein, Gl. 33/36–41):** Weil das ausgeführte Paket durch
das verfügbare Gewicht *beider* Partner begrenzt ist, muss die
Ziehwahrscheinlichkeit durch genau diese Grenze geteilt werden, und die
Teilrate trägt einen `W_i`-Vorfaktor. Herleitung:
[`Bias_Correction_und_Gewichtsdisziplin.md`](historical/Bias_Correction_und_Gewichtsdisziplin.md).

**`agg_propensity_mode`** steuert, wie `R_i*` berechnet wird: `"pairwise"`
(O(n²), bitgenau) oder `"moment"` (O(n), geschlossene Form für die separablen
Kernel, Abweichung ~1e-15). Für Produktionsläufe `"moment"`.

## 3.2 Breakage (`mcpbe_break.py`)

`_do_one_break()` wählt Partikel `k` proportional zu `W_k · S_k`, wobei `S` die
Bruchrate aus dem `break_kernel` ist. Die Fragmentanzahl wird stochastisch
gerundet, die Fragmentvolumina aus CDF-Tabellen, über den LMC-Adapter oder
analytisch gezogen; jedes Fragment geht durch `ParticleMerger.find_or_create`,
das Elterngewicht sinkt um `dW`.

> **Zwei Parameter heißen `pl_v`.** Der eine ist der Volumenexponent der
> **Rate** (im Bruchkernel), der andere der Exponent der
> **Fragmentgrößenverteilung** (Solver-Attribut, kanonisch `break_frag_v`). Sie
> sind unabhängig — den einen zu setzen, wenn der andere gemeint war, fällt
> nicht auf.

## 3.3 Nucleation (`mcpbe_nucleation.py`)

Fügt Flüssigkeit als diskrete Tropfen zu; `.step()` läuft nach jedem
MC-Ereignis.

- **`_distribute_liquid_volume(v)`** akkumuliert Volumen in einem Restspeicher.
  Ohne den ginge durch die Ganzzahl-Diskretisierung der Tropfenzahl
  systematisch Flüssigkeit verloren.
- **`_distribute_one_droplet_with_dW(...)`** wählt ein Partikel gewichtet nach
  `W`. Kann das Partikel die Flüssigkeit nicht aufnehmen (`V_dry < v_liquid`),
  wird **manuell agglomeriert**, bis genug `V_dry` gesammelt ist. Danach:
  Porosität über `porosity_growth_kernel.compute_nucleation_porosity(...)`,
  Sättigung berechnen, und entweder in ein ähnliches bestehendes Partikel
  mergen oder ein neues anlegen.

**DSMC-Skalierung:** `effective_dW = dW · (Vc_ref/Vc)` — die physikalische
Tropfenzahl bleibt korrekt, auch wenn sich `Vc` durch Verdopplung ändert.

> Die Nucleation hat eine **eigene** „Finde-ähnliches-Partikel"-Implementierung
> (`_find_similar_particle`), unabhängig vom `ParticleMerger`. Verbesserungen an
> einer der beiden übertragen sich nicht automatisch auf die andere.

## 3.4 Kontinuierliche Prozesse (`mcpbe_continuous_processes.py`)

Pro `.step(t, dt)`:

1. **Liquid Internalization** — kapillargetriebenes Eindringen von externer in
   interne Flüssigkeit, nach Braumann et al. (2007):
   `dl_intern/dt = k_int · l_extern · (V_pore − l_intern)`.
2. **Porosity Compression** — exponentieller Porositätsabfall
   `ε(t) = ε_min + (ε_0 − ε_min)·exp(−k·t)`. Ändert `V_dry` und `porosity`, hält
   `V_solid` exakt konstant. Übersteigt die Sättigung dabei 1, wird
   überschüssige Flüssigkeit externalisiert.

## 3.5 `ParticleMerger` (`particle_merger.py`)

Verhindert, dass redundante Kindpartikel entstehen. Jedes neu erzeugte Partikel
wird gegen den Bestand geprüft; passt eines, wird **nur `W` erhöht** statt ein
neues `n_comp` anzulegen. Alle intensiven Eigenschaften bleiben unverändert —
deshalb ist das Verfahren masseerhaltend per Konstruktion.

Das Matching ist **toleranzbasiert**, nicht exakt: `tol_rel` (Default `1e-6`)
relativ auf V_dry/Porosität/Sättigung, `tol_abs_liquid` absolut auf
`liquid_volume`. Ein Hash-Index (log-gebinnt für Volumina, linear für Porosität)
macht den Lookup O(1); er wird inkrementell gepflegt und nur nach einer
Rekonstruktion komplett neu aufgebaut.

## 3.6 Rekonstruktion (`reconstruction_mixin.py`)

Bruch erzeugt mehr Partikel, als Agglomeration verbraucht. Überschreitet `a_tot`
die Schwelle `recon_N_max` (Default 4000), wird die Population durch eine
kleinere mit derselben Verteilung ersetzt. Verfahren: **CAM** (Cell Average,
deterministisch), **RS** (Resampling mit exakter M1-Korrektur), **2PM/4PM**
(Momentenreproduktion je Zelle), **QMX** (Quantile Mix — teilt in
small/mid/tail und wendet je Bereich ein anderes Verfahren an).

> Rekonstruktion überträgt `liquid_volume`, `porosity` und `saturation`
> **nicht**. Auf einer nassen Population würde sie deshalb die Massenbilanz
> verletzen; `_assert_reconstruction_is_safe()` verweigert das und lässt sich
> nur mit `recon_allow_granulation_state = True` übergehen. Für Nassläufe
> `recon_enable = False` lassen.

---

# Teil 4 — Das Kernel-Framework

## 4.1 Warum Kernel

Die Physik der Nassgranulation ist nicht abschließend geklärt — für jeden
Teilprozess gibt es konkurrierende Modelle. Statt sie mit `if`-Kaskaden in den
Solver zu schreiben, sind sie als **austauschbare Objekte** hinter festen
Schnittstellen gekapselt. `KernelManager` (`kernel_integration.py`) hält acht
Steckplätze und reicht Berechnungen an den jeweils konfigurierten Kernel weiter.

Alle sind optional **außer der Aggregation** — ohne Kollisionsmodell gibt es
keine Simulation, deshalb wirft das Fehlen dort einen `ValueError` mit einer
Liste der verfügbaren Kernel.

## 4.2 Die acht Steckplätze

| Steckplatz | Beantwortet die Frage | Implementierungen |
|---|---|---|
| `agg_kernel` | Wie oft stoßen zwei Partikel zusammen? β(r₁,r₂) | `shear_chin1998`, `brownian_tsouris1995`, `constant`, `sum`, `eke_darelius2005`, `etm_darelius2005` |
| `agglomeration_acceptance_kernel` | Bleibt der Stoß haften? | `stokes_krit`, `stokes_dynamik`, `fittable` |
| `break_kernel` | Wie oft bricht ein Partikel? S(V) | `power_law`, `powerlaw_rumpf`, `powerlaw_rumpf_dynamic` |
| `porosity_growth_kernel` | Wie entwickelt sich der Porenraum? | `volume_mixing`, `cone_model`, `incomplete_mixing` |
| `liquid_dist_kernel` | Welches Partikel trifft der Tropfen? | `uniform_weighted` |
| `porosity_compression_kernel` | Wie schnell kollabieren Poren? | `porosity_compression` |
| `liquid_internalization_kernel` | Wie schnell zieht Flüssigkeit in die Poren? | `liquid_internalization` |
| `liq_internalisation_agglomeration_kernel` | Wie viel Film wird beim Stoß eingeschlossen? | `liq_internalisation_agglomeration` |

Jede Kategorie hat eine `blueprint.py` mit der abstrakten Basisklasse und einer
Anleitung, wie ein eigener Kernel hinzugefügt wird. Die gemeinsamen
Basisklassen stehen in `kernels/base.py`.

## 4.3 Parametervalidierung

Kernel werden über Name + Parameter-Dict konfiguriert:

```python
MCPBESolver(
    agg_kernel_name='shear_chin1998',
    agg_kernel_params={'corr_beta': 1e-3, 'g': 1000},
    ...
)
```

Jede `get_*_kernel(...)`-Factory ruft `kernels/base.py::reject_unknown_params`
und wirft einen `ValueError`, wenn ein übergebener Name von diesem Kernel gar
nicht gelesen wird — **mit Vorschlag des vermutlich gemeinten Namens**.

Der Grund: `get_default_params()` wird mit den Aufrufer-Werten aktualisiert. Ein
unbekannter Schlüssel überschreibt dabei nichts, er landet nur zusätzlich im
Dict — der Kernel liefe still auf seinem Default weiter. Ein Tippfehler wäre
damit unsichtbar. Kernel mit absichtlich default-losen Parametern deklarieren
diese über `get_optional_params()`.

## 4.4 Eigentümerschaft geteilter Größen

Manche Parameter beschreiben den **Prozess**, nicht ein einzelnes Modell. Die
Scherrate `G` ist das Paradebeispiel: sie taucht im Agglomerations- und im
Bruchkernel auf, beschreibt aber denselben Mischer.

Die Regel: **der Aggregationskernel besitzt `solver.G`.** Der Bruchkernel
rechnet mit seinem eigenen `g`, überschreibt `solver.G` aber nicht. Weicht sein
`g` ab, gibt es eine `UserWarning` statt einer stillen Änderung.

Das ist wichtig, weil `solver.G` auch von der Nucleation gelesen wird (für die
Kollisionsgeschwindigkeit). Ohne diese Regel folgte die Nucleation
stillschweigend dem Bruchmodell.

---

# Teil 5 — Wiederkehrende Muster

Wer diese Muster einmal verstanden hat, liest den Rest des Codes deutlich
schneller.

**Intensiv bleibt intensiv.** Immer wenn `W` reduziert wird, bleiben
`liquid_volume`, `porosity`, `saturation` und `V_flat` **unverändert**. Die
übriggebliebenen physischen Partikel sind dieselben wie vorher — es sind nur
weniger.

**Feststoff wird addiert, nie modelliert.** Bei jedem Ereignis, das Partikel
zusammenführt oder teilt, wird `V_solid` direkt summiert bzw. aufgeteilt. Kernel
dürfen nur den *Porenanteil* bestimmen. Damit kann kein Modell die Massenbilanz
verletzen.

**Erst rechnen, dann schreiben.** Zusammengeführte Eigenschaften werden
vollständig in lokalen Variablen berechnet, bevor irgendetwas in die Arrays
geschrieben wird. So gibt es keinen Zeitpunkt, an dem der Zustand halb alt und
halb neu ist — wichtig, weil der `ParticleMerger` dazwischen auf den Bestand
zugreift.

**Absteigend löschen.** Müssen mehrere Partikel entfernt werden, geschieht das
in absteigender Indexreihenfolge. Dann kann Swap-with-last die noch offenen
Indizes nicht verschieben.

**Grenzfälle sind der Grenzwert, nicht ein Fallback.** Wo eine Formel an einem
Rand divergiert, gibt der Code den *mathematischen Grenzwert* zurück — nicht
einen „neutralen" Ersatzwert. Beispiel: `ε → 0` im Rumpf-Modell ergibt Bruchrate
`0`, nicht die unkorrigierte Grundrate. Ein Ersatzwert wäre ein stiller
Regimewechsel mitten in der Parameterlandschaft.

**Laut scheitern statt still weiterlaufen.** NaN/inf in Bruchraten werfen einen
`RuntimeError` mit Diagnose. Ein widersprüchlicher Snapshot-Zustand wirft. Ein
unbekannter Kernel-Parameter wirft. Die Leitlinie: ein Fehler, der die
Simulation weiterlaufen lässt, kostet Tage; einer, der sie anhält, Minuten.

---

# Teil 6 — Wiederholungen und Statistik (`framework/`)

Ein einzelner Monte-Carlo-Lauf ist eine Stichprobe, kein Ergebnis. `framework/`
rechnet dieselbe Konfiguration mit **unabhängigen Seeds** mehrfach und wertet
sie statistisch aus.

| Modul | Rolle |
|---|---|
| `builder.py` | Beschreibt einen Lauf als reines `dict` und baut daraus den Solver |
| `metrics.py` | Reduziert einen fertig gerechneten Solver auf kompakte Zeitreihen |
| `runner.py` | Führt N Wiederholungen aus, seriell oder auf mehreren Prozessen |
| `statistics.py` | Mittelwert, Streuung, Standardfehler, Konvergenzprüfung |
| `run_reference_repeats.py` | Ausführbares Skript: eine Konfiguration, zehn Seeds |

Drei Entwurfsentscheidungen, die den Unterschied machen:

- **Seeds über `SeedSequence.spawn(N)`**, nicht `seed = base + i`. Benachbarte
  Seeds können korrelierte Zufallsströme erzeugen; nur mit `spawn` bedeutet ein
  Standardfehler das, was er zu bedeuten vorgibt.
- **Der Worker bekommt das Rezept, nicht den Solver.** Drei Unterobjekte des
  Solvers halten eine Rückreferenz auf ihn; ein `deepcopy` erzeugte Kopien, die
  auf einen verwaisten Solver zeigen. Details im Docstring von `builder.py`.
- **Reduktion im Worker.** Zurück wandern wenige Kilobyte statt der kompletten
  Partikel-Historie.

Aufruf aus `mcpbe/src`:

```bash
python -m wmcpbe.framework.run_reference_repeats --repeats 10 --workers 6
```

---

# Anhang

## Typische Initialisierungsreihenfolge

```python
solver = MCPBESolver(dim=1, t_vec=..., load_attr=False, <kernel-konfiguration>)
solver.process_type = "mix"
solver.agg_dW_max = 20.0                       # VOR _initialize_samplers()
solver._initialize_particles(init_Vc=False, V_flat=V_flat, W_init=W_init)
solver.porosity[:solver.a_tot] = ...           # intensive Startwerte
solver._initialize_samplers()                  # NACH den Partikeln!
solver.create_nucleation_handler(...)
solver.create_continuous_processes_handler(...)
solver.solve()
```

`_initialize_samplers()` muss **nach** `_initialize_particles()` kommen — es
baut die Propensities aus dem Partikelzustand auf. `_validate_solve_readiness()`
prüft das zu Beginn von `solve()` und erklärt im Fehlerfall die richtige
Reihenfolge.

`V_flat` muss beide Zeilen gefüllt haben: `V_flat[0] = V_solid` **und**
`V_flat[-1] = V_dry`. `framework/builder.py` zeigt das korrekt.

## Debug-Schalter

`solver.mcpbe_debug_mass` schaltet detaillierte Massenbilanz-Ausgaben frei;
ergänzend `mcpbe_debug_agg` / `_break` / `_merger` / `_nuc` / `_comp` / `_vc`.
`_debug_max_events` (Default 20) begrenzt die Ausgabemenge.

Für Untersuchungen über einen **kompletten** Lauf ist ein externes
Monkeypatch-Skript oft geeigneter: `sys.path` auf `mcpbe/src` setzen, die
Zielmethode durch einen Wrapper ersetzen, der vor und nach dem Originalaufruf
`Σ(V_solid·W)` misst. So lässt sich jede Abweichung kategorienweise
(AGG/BREAK/NUC/COMP/VC) zuordnen, ohne eine Repo-Datei zu ändern.

## Glossar

| Begriff | Bedeutung |
|---|---|
| DSMC | Direct Simulation Monte Carlo |
| PBE | Population Balance Equation |
| PSD | Particle Size Distribution |
| `Vc` | Kontrollvolumen [m³] |
| `W` | Rechengewicht = physische Partikel je Rechenpartikel |
| `n_comp` / `n_phys` | Zahl der Rechen- bzw. physischen Partikel |
| `a_tot` | Zahl aktiver Rechenpartikel (Indexgrenze aller Arrays) |
| `_cap` | Kapazität der vorallokierten Arrays (≥ `a_tot`) |
| `V_solid` | Feststoffvolumen — **Erhaltungsgröße** |
| `V_dry` | `V_solid + V_pore` — keine Erhaltungsgröße |
| `dW` | Paketgröße: wie viele physische Partikel ein Ereignis bewegt |
| `δ_i` | `min(dW_const, W_i)` — maximal ausführbares Paket für Partikel `i` |
| ε | Porosität = `V_pore / V_dry` |
| S | Sättigung = `V_liq_intern / V_pore` |
| Vollkörper | porenfreies Partikel (`ε = 0`, historisch `NaN`) — bewusst als deutscher Fachbegriff auch im Code, siehe unten |
| β(i,j) | Kollisionsfrequenz eines Paares |
| S(V) | Bruchrate eines Partikels |
| `R_i*` | bias-korrigierte Agglomerations-Propensity von Partikel `i` |

### Sprache im Code

Docstrings und Kommentare sind **englisch**, diese Dokumentation ist deutsch.

Eine bewusste Ausnahme: **„Vollkörper"** bleibt auch im Code stehen. Der
Begriff ist nicht mit „non-porous" austauschbar, weil der Code zwei Zustände
unterscheidet, die beide „ohne Poren" heißen:

| Kodierung | Bezeichnung im Code | Bedeutung |
|---|---|---|
| `porosity = NaN` | Vollkörper | Alt-Kodierung, wird weiterhin unterstützt |
| `porosity = 0.0` | poreless | moderne Kodierung, was neue Kernel liefern |

Beide führen auf `V_solid = V_dry`, aber nur die erste muss überall per
`np.isnan()` abgefangen werden. Eine Übersetzung beider auf „non-porous"
würde diese Unterscheidung einebnen.

## Weiterführende Dokumente

Dieses Dokument beschreibt den **Ist-Zustand**. Wie er zustande kam — Herleitungen,
Audits, Fehleranalysen — steht in
[`historical/`](historical/). Die wichtigsten:

| Dokument | Inhalt |
|---|---|
| [`Bias_Correction_und_Gewichtsdisziplin.md`](historical/Bias_Correction_und_Gewichtsdisziplin.md) | Herleitung der Propensity-Korrektur und der Gewichtsdisziplin |
| [`Audit_2026-08-17.md`](historical/Audit_2026-08-17.md) | Systematisches Audit auf schlafende Probleme, 25 Befunde |
| [`MOMENT_MODE.md`](historical/MOMENT_MODE.md) | Herleitung des O(n)-Momentenmodus |
| [`Kompression_und_Parametervalidierung.md`](historical/Kompression_und_Parametervalidierung.md) | Warum die Parametervalidierung eingeführt wurde |
| [`Nucleation_Propensity_Blockade.md`](historical/Nucleation_Propensity_Blockade.md) | Propensity-Aktualität nach Nucleation |
| [`Parallelisierung_Wiederholungen.md`](historical/Parallelisierung_Wiederholungen.md) | Entwurf des Wiederholungs-Frameworks |

Offene Befunde aus dem Dokumentationsdurchgang:
[`historical/Befunde_2026-08-20.md`](historical/Befunde_2026-08-20.md).
