# WMCPBE – Überblick über Aufbau und Funktionsweise

Diese Datei erklärt, **wie der Code aufgebaut ist und warum**. Sie geht stufenweise
vor: erst die großen Zusammenhänge, dann die Bausteine (Klassen, Handler, Kernel),
zuletzt die nicht-trivialen Funktionen im Detail. Sonderfälle sind jeweils dort
erklärt, wo sie behandelt werden.

Gedacht als Einstieg für jemanden, der den Code noch nicht kennt – oder für dich
selbst in ein paar Monaten.

---

# Stufe 1 – Das große Bild

## 1.1 Was simuliert wird

WMCPBE ist ein **gewichteter Monte-Carlo-Löser für Populationsbilanzgleichungen**
(weighted Monte Carlo Population Balance Equations), zugeschnitten auf
**Nassgranulation**.

Der physikalische Ablauf, den er abbildet:

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

Die gesuchte Größe ist die **Partikelgrößenverteilung (PSD)** über der Zeit, die
gegen experimentelle Daten gefittet werden soll.

## 1.2 Warum Monte Carlo und warum gewichtet

Eine Populationsbilanz ist eine Integro-Differentialgleichung über einer
kontinuierlichen Größenkoordinate. Statt sie zu diskretisieren (das macht das
Schwesterprojekt `dpbe`), verfolgt der MC-Ansatz **einzelne Partikel** und würfelt
Ereignisse aus. Vorteil: mehrdimensionale Zustände (Volumen *und* Porosität *und*
Sättigung *und* Flüssigkeit) kosten nicht exponentiell mehr Speicher.

Nachteil: Ein echter Granulator enthält ~10¹² Partikel. Deshalb **gewichtet**:

> Jedes **Rechenpartikel** (computational particle) steht stellvertretend für
> `W` **physische Partikel**.

`W` ist damit die einzige *extensive* Größe im Zustand. Alles andere (Volumen,
Porosität, Sättigung, Flüssigkeitsmenge) ist **intensiv**, also „pro physischem
Partikel" definiert und ändert sich beim Kopieren oder beim Reduzieren von `W`
nicht. Dieses Prinzip zieht sich durch den gesamten Code und ist der Schlüssel zum
Verständnis fast jeder Zeile, die `W` anfasst.

**Merkregel:**
`Gesamte Feststoffmasse = Σ_i (V_solid_i · W_i)` – und genau diese Summe muss über
die ganze Simulation konstant bleiben.

## 1.3 Das Kontrollvolumen `Vc`

DSMC (Direct Simulation Monte Carlo) simuliert nicht den ganzen Apparat, sondern
ein repräsentatives **Kontrollvolumen** `Vc`. Daraus folgt die Partikeldichte:

```
n_phys = Σ W_i / Vc        [Partikel pro m³]
```

`Vc` ist damit **keine numerische Stellschraube, sondern eine physikalische
Größe**: sie legt zusammen mit `Σ W` die Konzentration fest, und die Konzentration
steuert die Agglomerationsrate (∝ 1/Vc).

Wenn Agglomeration die Partikelzahl stark reduziert, wird die Statistik dünn.
Dagegen gibt es `_maybe_double_control_volume`: `Vc` wird verdoppelt und der
gesamte Partikelbestand dupliziert. `n_phys` bleibt konstant, aber es gibt wieder
doppelt so viele Rechenpartikel. (Details in 3.6.)

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
 │                                        Kompression           │
 │    nucleation.step(t, dt)              Tropfenzugabe        │
 └─────────────────────────────────────────────────────────────┘
 ┌─ 5. Aufräumen ──────────────────────────────────────────────┐
 │    ggf. Vc verdoppeln, ggf. Rekonstruktion (n_comp senken)  │
 └─────────────────────────────────────────────────────────────┘
```

**Warum Operator Splitting?** Nucleation und Kompression laufen auf einer
*eigenen* Zeitskala – sie sind keine gewürfelten Ereignisse, sondern
kontinuierliche Prozesse. Sie werden deshalb *nach* dem MC-Ereignis mit dem
gerade vergangenen `dt` angewendet. Dadurch kann eine Porositäts- oder
Flüssigkeitsänderung die Zeitwahl des *aktuellen* Ereignisses nicht mehr
beeinflussen – die Ereignisstatistik bleibt sauber.

## 1.5 Vererbung vs. Komposition – die zentrale Designentscheidung

```python
# mcpbe.py
class MCPBESolver(MCPBEPost, MCPBEBreak, MCPBEAgg, MCPBEBase, ReconstructionMixin):
```

Es gibt **zwei verschiedene Arten**, wie Physik an den Solver angebunden ist:

| | **Mixins** (Vererbung) | **Handler** (Komposition) |
|---|---|---|
| Wer | `MCPBEAgg`, `MCPBEBreak`, `MCPBEPost`, `ReconstructionMixin` | `NucleationHandler`, `ContinuousProcessesHandler` |
| Zugriff | direkt auf `self.V_flat`, `self.W`, … | über `self.solver.…` |
| Aufruf | *innerhalb* der Ereignisschleife, wenn das Ereignis gewürfelt wurde | *nach* jedem Ereignis über `.step(t, dt)` |
| Warum so | sind untrennbar Teil des MC-Ereignisses und brauchen den kompletten Zustand | laufen auf eigener Zeitskala, sind optional zuschaltbar, haben eigenen internen Zustand (Statistik, Restvolumen) |

Die MRO ist **Post → Break → Agg → Base → Reconstruction**. `MCPBEPost` steht
vorn, weil Auswertung auf alles darunter zugreifen können muss; `MCPBEBase` steht
hinten, weil es die Grundlage liefert, die alle anderen benutzen.

Die Handler werden über Factory-Methoden erzeugt:
`solver.create_nucleation_handler(...)` und
`solver.create_continuous_processes_handler(...)`. Beide warnen, wenn sie einen
bestehenden Handler ersetzen – sonst würde man stillschweigend die gesammelte
Statistik verlieren.

## 1.6 Datei-Landkarte

| Datei | Rolle | Zeilen |
|---|---|---|
| `mcpbe.py` | Setzt `MCPBESolver` zusammen, Handler-Factories | 206 |
| `mcpbe_base.py` | Partikelzustand, Kontrollvolumen, **Hauptschleife**, Array-Verwaltung | 3579 |
| `mcpbe_agg.py` | Agglomerationsphysik und -ereignis | 1186 |
| `mcpbe_break.py` | Breakage-Raten, Fragmenterzeugung, Bruchereignis | 1306 |
| `mcpbe_nucleation.py` | `NucleationHandler` – Flüssigkeitszugabe | 2721 |
| `mcpbe_continuous_processes.py` | Internalisierung + Kompression | 448 |
| `particle_merger.py` | Dedup neu erzeugter Partikel (senkt `n_comp`) | 898 |
| `reconstruction_mixin.py` | Partikelzahl-Reduktion (CAM/RS/2PM/QMX) | 1532 |
| `mcpbe_post.py` | Momente, PSD/CDF, Auswertecontainer | 1412 |
| `kernel_integration.py` | `KernelManager` – hält und verteilt alle Physikmodelle | 533 |
| `kernels/` | Die austauschbaren Physikmodelle | ~5000 |
| `fenwick_new.py` | `FenwickSampler` – O(log n) gewichtetes Ziehen | 216 |
| `mcpbe_time_helper.py` | Zeitschritt- und Paketgrößen-Logik | 251 |
| `helpers.py` | Fertige Solver-Konfigurationen, Massecheck | 739 |
| `lmc_adapter.py` | Lattice-MC-Fragmentverteilungen (optional) | 2390 |
| `mlp_breakage_adapter.py` | Neuronales Netz für Bruchraten (optional) | 220 |

---

# Stufe 2 – Das Datenmodell

Alles, was der Solver über die Population weiß, steckt in wenigen NumPy-Arrays.
Sie sind alle auf `_cap` (Kapazität) vorallokiert; gültig sind die Spalten
`[0, a_tot)`.

## 2.1 `V_flat` – das Volumen-Array

Form `(dim+1, _cap)`. Für den Standardfall `dim = 1`:

```
V_flat[ 0, idx]  = V_solid   ← ERHALTUNGSGRÖSSE
V_flat[-1, idx]  = V_dry     ← V_solid + V_pore, KEINE Erhaltungsgröße
```

Bei `dim > 1` enthalten die Zeilen `0 … dim-1` die Feststoffvolumina der
einzelnen Komponenten, die letzte Zeile weiterhin `V_dry`.

**Warum diese Trennung?** Weil bei Nassgranulation zwei Dinge gleichzeitig
passieren, die man nicht vermischen darf:

- Der **Feststoff** wird nur umverteilt, nie erzeugt oder vernichtet.
- Das **Gesamtvolumen** des Partikels ändert sich sehr wohl – Poren wachsen bei
  Agglomeration und schrumpfen unter Kompression.

Nur `V_solid` ist deshalb die Bilanzgröße. `V_dry` ist die Geometrie, aus der der
Durchmesser folgt.

## 2.2 Die begleitenden Arrays

| Array | Typ | Bedeutung |
|---|---|---|
| `W[idx]` | **extensiv** | Wie viele physische Partikel `idx` repräsentiert |
| `X[idx]` | intensiv | Durchmesser, aus `V_dry` abgeleitet |
| `porosity[idx]` | intensiv | ε = V_pore / V_dry |
| `liquid_volume[idx]` | intensiv | Gesamte Flüssigkeit **pro physischem Partikel** [m³] |
| `saturation[idx]` | intensiv | S = V_liq_intern / V_pore, ∈ [0,1] |

Zusätzlich die Rechenhilfen `_r_agg` (Agglomerations-Propensity), `_break_rate`,
`_delta_agg`, `_delta_break` (Paketgrößen). Sie sind **indexgleich** mit den
Zustandsarrays und werden deshalb genauso mitverwaltet.

## 2.3 Die drei Flüssigkeitsbegriffe

Das ist erfahrungsgemäß die häufigste Verwechslungsquelle:

```
liquid_volume  =  V_liq_intern            +  V_liq_extern
                  └─ in den Poren ─┘         └─ Film auf der Oberfläche ─┘
                  = V_pore · S               = liquid_volume − V_liq_intern
```

- **Intern** ist Flüssigkeit, die in die Poren eingedrungen ist. Sie macht das
  Granulat *fester* (Kapillarkräfte im Rumpf-Modell).
- **Extern** ist Oberflächenfilm. Er ist es, der beim Zusammenstoß die
  **Flüssigkeitsbrücke** bildet – nur damit kann eine Kollision überhaupt haften
  (Stokes-Kriterium).

Gespeichert wird nur `liquid_volume` und `saturation`; die Aufteilung wird bei
Bedarf über `get_V_liquid_internal()` / `get_V_liquid_external()` (in
`mcpbe_base.py`) berechnet. So kann sie nie inkonsistent auseinanderlaufen.

## 2.4 Sonderfall „Vollkörper"

Historisch wurden porenfreie Partikel mit `porosity = NaN` markiert. Moderne
Kernel liefern stattdessen `0.0`. **Beide werden bis heute unterstützt**, weil
gespeicherte Läufe und ältere Konfigurationen sonst brechen würden.

Der Code behandelt das konsequent an jeder Stelle, die aus Porosität etwas
ableitet, z. B. in `get_V_solid`:

```python
V_solid = np.where(np.isnan(poro), V_dry, V_dry * (1.0 - poro))
```

Ohne diesen Zweig würde ein einziges NaN in jede Massensumme durchschlagen, die
darauf aufbaut. `V_solid = V_dry` ist für einen porenfreien Körper die korrekte
Auslegung, weil er keinen Porenraum hat.

## 2.5 Wachsen und Löschen von Partikeln

Zwei Operationen in `mcpbe_base.py` verwalten die Spalten:

**`_append_particle_column(frag_vols)`** hängt ein Partikel an Index `a_tot` an
und erhöht `a_tot`. Reicht die Kapazität nicht, wächst sie vorher über
`_ensure_capacity_for` um einen Faktor zwischen 1,1 und 2,0 – **früh in der
Simulation aggressiver**, weil dort die meisten Umschichtungen passieren
(`_growth_factor`).

**`_remove_particle_column(j)`** löscht per **Swap-with-last**: das letzte
Partikel wird nach `j` kopiert, `a_tot` sinkt um eins, der freigewordene Slot wird
geleert. Das ist O(1) statt O(n) – aber es bedeutet:

> **Jeder Index ≥ j kann sich durch eine Löschung verschieben.**

Deshalb gibt es zwei Schutzmechanismen:

1. Eine kanonische Liste `_PARTICLE_ARRAY_NAMES` aller indexgleichen Arrays. Wer
   ein neues Array einträgt, bekommt Swap und Slot-Freigabe automatisch – man kann
   es nicht mehr vergessen.
2. `notify_index_swap(old, new)` benachrichtigt den `ParticleMerger`, **bevor** die
   Daten verschoben werden, damit dessen Index nicht auf einen Slot zeigt, der
   gleich von einem fremden Partikel wiederverwendet wird.

Code, der einen frisch erzeugten Kindindex über mehrere Elternlöschungen hinweg
festhalten muss, verfolgt ihn aktiv nach (Muster `child_current_idx` in
`mcpbe_nucleation.py::_manual_agglomerate_particles`). Wo möglich wird stattdessen
**absteigend** gelöscht (`sorted({i, j}, reverse=True)` in
`_consume_parent_weight`), dann kann eine Löschung den noch offenen Index gar
nicht erst verschieben.

---

# Stufe 3 – Die Bausteine im Einzelnen

## 3.1 `MCPBEBase` – Zustand und Hauptschleife

Zuständig für: Initialisierung, Kapazität, Kontrollvolumen, Volumen-Hilfsfunktionen
und `solve()`.

### Drei Wege, Partikel zu initialisieren

`_initialize_particles()` kennt drei sich ausschließende Modi:

1. **Aus `c`/`x`/`PGV`/`SIG`** (`init_Vc=True`): Konzentration und Durchmesser
   werden vorgegeben, `Vc` folgt aus `a0/n0`. Verteilungstypen: `mono`, `norm`,
   `weibull`.
2. **Direkt über `V_flat` + `W_init`** (`init_Vc=False`): der übliche Weg in den
   Trial-Skripten, weil man Porosität und Gewicht exakt kontrollieren will.
3. **Aus einer experimentellen CDF** (`init_cdf`): `_build_init_from_cdf` invertiert
   die gemessene Summenverteilung an Mittelpunktsquantilen und erzeugt daraus
   Repräsentanten. Unterstützt Zahl- (Q0) und Volumenbasis (Q3) – bei Q3 werden die
   Gewichte so gewählt, dass **jeder Repräsentant denselben Volumenanteil trägt**.
   Das ist der Weg, um direkt gegen Messdaten zu starten.

Optional lässt sich die Startpopulation über `_compress_init_by_quantile` auf
`V_eff_init` Repräsentanten eindampfen – ebenfalls wahlweise Q0 oder Q3.

### `solve()` – was dort nicht offensichtlich ist

**Snapshot-Vorhersage.** Der Loop speichert pro Ausgabezeitpunkt einen Zustand
*vor* und *nach* dem Ereignis. Eine Kopie des Vorzustands kostet O(n·(dim+1))
Speicherbandbreite. Deshalb wird **vorher berechnet**, ob dieses Ereignis
überhaupt einen Snapshot auslöst (`will_save`), und nur dann kopiert. Da der
relevante Timer den Zeitstempel des kommenden Ereignisses bereits enthält, ist die
Vorhersage exakt – und es gibt eine `RuntimeError`-Zusicherung, die laut scheitert,
falls sie es je nicht wäre.

**Handler-Auflösung außerhalb der Schleife.** `nucleation` und
`continuous_processes` werden einmal vor der Schleife per `getattr` aufgelöst.
Handler können während `solve()` nicht angehängt werden, also spart das drei
Attributsuchen pro Ereignis.

**Propensity-Auffrischung nach Nucleation.** Nucleation legt Partikel an und
verschiebt Gewicht. Ein frisch angehängter Slot hat `_r_agg = 0` und wäre für die
Paarziehung unerreichbar. Deshalb meldet der Handler über
`consume_population_changed()`, ob er tatsächlich etwas verändert hat – und nur
dann wird einmal pro Ereignis (nicht pro Tropfen!) aufgefrischt. Das hält die
Kosten bei O(n) je Ereignis statt O(n) je Tropfen.

**Abbruchbedingungen.** `maxiter`, `t_vec[-1]`, optional `max_particles`
(wichtig bei bruchdominierten Läufen, sonst kann die Partikelzahl explodieren) und
ein externes `cancel_flag` für GUI-/Optimizer-Abbrüche.

### `solve_repeats()` – Ensembles

Führt N Realisierungen mit unterschiedlichen Seeds aus, seriell oder parallel
(`ProcessPoolExecutor`). Optional werden PSDs direkt mitgemittelt, wahlweise als
`Q(x)` (Summenverteilung auf festem Durchmessergitter) oder `x(Q)` (Durchmesser an
festen Quantilen). Bei sehr großen Ensembles können die Ergebnisse über `memmap`
auf Platte geschrieben werden, statt den RAM zu füllen.

## 3.2 `MCPBEAgg` – Agglomeration

### Die Grundidee der gewichteten Agglomeration

Ein Agglomerationsereignis lässt **nicht** zwei Rechenpartikel verschmelzen.
Stattdessen:

```
  vorher:  i (W=600)          j (W=400)
             │                  │
             └──── dW = 50 ─────┘        dW = "Paketgröße"
                     │
  nachher: i (W=550)   j (W=350)   Kind (W=50)
```

Es kollidieren `dW` physische Partikel-Paare gleichzeitig. Die Eltern verlieren je
`dW` an Gewicht, das Kind bekommt `dW`. Die intensiven Eigenschaften der Eltern
bleiben **unverändert** – die übriggebliebenen physischen Partikel sind ja
dieselben wie vorher, es sind nur weniger.

### Bias-Korrektur nach Ji & Rhein

Naiv würde man `i` proportional zu `Σ_j W_j β(i,j)` ziehen. Das ist verzerrt,
weil die tatsächlich ausführbare Paketgröße durch **beide** Partner begrenzt ist:
`dW_ij = min(δ_i, δ_j)` mit `δ_i = min(dW_const, W_i)`. Wer nur ein kleines Paket
ausführen kann, muss dafür entsprechend häufiger gezogen werden.

Die implementierte, korrigierte Propensity lautet deshalb (Gl. 36/37):

```
R_i* = W_i · [  Σ_{j≠i} W_j · β(i,j) / min(δ_i, δ_j)
              + (W_i − 1) · β(i,i) / min(δ_i, W_i/2)  ]
```

Zwei Dinge sind hier wichtig und leicht zu übersehen:

- Der **`W_i`-Vorfaktor** vorne.
- Die Division steht **innerhalb** der Summe (paarweise), nicht davor.

Die Partnerziehung ist dazu konsistent: `j` wird proportional zu
`W_j·β(i,j)/min(δ_i,δ_j)` gezogen (Gl. 40), und die Normierung dafür ist genau
`R_i*/W_i` – also exakt die Summe, die der Propensity-Aufbau für dieses `i`
akkumuliert hat. Dadurch können die beiden Stufen per Konstruktion nicht
auseinanderlaufen.

### Selbstkollision `i == j`

Zwei physische Partikel **desselben** Rechenpartikels stoßen zusammen. Das ist
ausdrücklich erlaubt und braucht überall dieselbe Sonderbehandlung:

- Es gibt `W_i − 1` mögliche Partner (nicht `W_i`) – daher der Term `(W_i−1)` oben.
- Ein Ereignis verbraucht **zwei** physische Partikel aus demselben Vorrat, also
  wird `dW` auf `δ_ii = min(δ_i, W_i/2)` **gekappt** statt das Ereignis abzulehnen.
- Der Verbrauch ist dann `W[i] -= 2·dW`.

Die Kappung (statt Ablehnung) hat einen numerischen Grund: `W_i/2` ist in IEEE-754
exakt (nur der Exponent sinkt), also gilt `2·(W_i/2) == W_i` bitgenau. Das letzte
Ereignis räumt das Partikel damit **exakt auf 0.0** ab – es bleiben keine
Restgewichte von ~1e-15 übrig. Genau deshalb genügt `W > 0.0` als Löschkriterium
und es braucht keine Epsilon-Schwelle.

### Ablauf eines Ereignisses (`_do_one_agg`)

```
1. _select_pair(a)
     ├─ Fenwick-Sampler zieht i ∝ R_i*
     ├─ Partnerziehung j ∝ W_j·β(i,j)/dW_ij     (eigene Zufallszahl)
     ├─ SIZEEVAL-Größenakzeptanz                (eigene Zufallszahl)
     └─ Akzeptanzkernel, z. B. Stokes           (physikalisch)
2. _compute_agg_dW(i, j, …)   → Paketgröße
3. _merge_pair(i, j, dW)      → Kind erzeugen
4. _consume_parent_weight     → W_i, W_j reduzieren, ggf. löschen
5. _refresh_samplers_after_agg → Propensities neu
```

Jede Ablehnung führt zu einem **No-Op-Ereignis**: die Zeit schreitet fort, es
passiert nichts. Das ist statistisch korrekt (Rejection Sampling) und der Grund,
warum die Ereigniszahl höher sein kann als die Zahl tatsächlicher Agglomerationen.

**Warum drei getrennte Zufallszahlen?** Früher wurde die Variable der
Partnerauswahl für den Größentest wiederverwendet. Das korreliert die beiden
Entscheidungen und verzerrt die Statistik. Der zweistufige Sampler macht diese
Wiederverwendung ohnehin unmöglich, und die Akzeptanz zieht ausdrücklich frisch.

### `_merge_pair` – wo die Erhaltung sichergestellt wird

```python
V_solid_merged = V_solid_i + V_solid_j        # EXAKT, unabhängig vom Kernel
V_dry_merged, poro_merged = porosity_growth_kernel.compute_merged_porosity(...)
```

Der Feststoff wird **direkt addiert** und nicht dem Kernel überlassen. Der Kernel
bestimmt nur, wie viel *Porenraum* dazukommt. Damit kann kein Porositätsmodell –
egal wie exotisch – die Massenbilanz verletzen.

Die Flüssigkeit wird in vier Schritten zusammengeführt:

1. Beide Eltern in intern/extern zerlegen.
2. Der Kernel `liq_internalisation_agglomeration` sagt, wie viel Oberflächenfilm
   in den **neu entstehenden Kontaktporen** eingeschlossen wird (Braumann 2007).
3. Kann nicht mehr internalisiert werden, als extern vorhanden ist → Kappung.
4. Passt die interne Flüssigkeit nicht ins neue Porenvolumen (S > 1), wird der
   Überschuss **externalisiert** und S auf 1 gesetzt.

Bei `i == j` werden die Beiträge verdoppelt statt addiert – es sind ja zwei
physische Partikel derselben Sorte.

### Propensity-Aufbau: `pairwise` vs. `moment`

`_rebuild_all_propensities` dominiert die Laufzeit (typisch > 90 %). Es gibt zwei
Modi:

| Modus | Kosten | Wann |
|---|---|---|
| `"pairwise"` | O(n²) | exakt, für beliebige Kernel; Referenz |
| `"moment"` | **O(n)** | für *separierbare* Kernel (shear, brownian, sum, constant) |

**Warum funktioniert der Momentenmodus?** Ein separierbarer Kernel lässt sich als
`β(i,j) = Σ_k f_k(r_i)·g_k(r_j)` schreiben. Dann ist

```
Σ_j W_j β(i,j) = Σ_k f_k(r_i) · [ Σ_j W_j g_k(r_j) ]
                                  └──── einmal für alle i ────┘
```

Die inneren Summen sind **Momente** der Population und werden einmal berechnet.
Aus O(n²) wird O(n) – bei n = 4000 rund 275× schneller.

Der Dispatch in `_compute_raw_propensities` wählt automatisch:
kompilierte Momentenform → kompilierte O(n²)-Form → kompilierte Sonderform für
`liquid_bridge` (nicht separierbar wegen des Kreuzterms `s_i·s_j`) → generischer
Python-Fallback. Der Fallback warnt **einmal**, damit die Kosten sichtbar sind und
nicht als mysteriöse Langsamkeit erscheinen.

Die kompilierte Schiene gilt nur für `dim == 1`, weil dort die Kollisionseffizienz
`alpha` eine einzige Zahl ist und in den Vorfaktor `p0` gefaltet werden kann. In 2D
hängt `alpha` von der Komponentenmischung des Paares ab (`_alpha_ccm`), also läuft
der generische Weg.

## 3.3 `MCPBEBreak` – Breakage

### Rate und Propensity

```
Propensity_i = W_i · S_i / δ_i        mit δ_i = min(break_dW_const, W_i)
```

`S_i` ist die Einzelpartikel-Bruchrate aus dem Kernel. Die Division durch `δ_i`
ist wieder die Paketkorrektur.

Es gibt drei Quellen für `S_i`, in dieser Priorität:
1. **Kernel-Framework** (`break_kernel`) – der Normalfall.
2. **MLP-Modell** (`mlp_breakage_adapter`) – ein neuronales Netz.
3. Nichts konfiguriert → Rate 0, Breakage ist aus.

Batch- und Einzelpfad verhalten sich in Fall 3 bewusst **gleich** (beide liefern
0). Früher warf der Einzelpfad einen Fehler – dieselbe Konfiguration war also auf
einem Weg legal und auf dem anderen ein Absturz, der erst beim ersten
inkrementellen Update auftrat.

`_assert_finite_rates` wirft bei NaN/inf **laut** einen `RuntimeError` mit
Diagnose (Partikelindex, V_dry, W, Porosität). Der Grund: die Kernel behandeln NaN
wie ein nicht-positives Volumen und lieferten sonst still `0.0` – das Partikel
würde einfach nie brechen, und man sucht die Ursache tagelang.

### Fragmenterzeugung

`_break_build_fragments` kennt vier Quellen, in dieser Reihenfolge:

1. **Live-LMC** – ein Lattice-Monte-Carlo läuft zur Laufzeit mit. Am genauesten,
   am teuersten. Ist das Partikel zu klein für das Gitter, gibt es einen
   definierten Fallback (gleichmäßiger Split in `NO_FRAG` Teile) oder das Partikel
   wird als unbrechbar markiert.
2. **One-Shot-Adapter** (Rank / Copula / Flow) – vorberechnete
   Verteilungen, die alle Fragmente auf einmal ziehen.
3. **Marginaltabellen** (`LMCTableAdapter`).
4. **Analytische CDF** aus `BREAKFVAL` – die Standardschiene.

Die erwartete Fragmentzahl `p` ist meist keine ganze Zahl. Sie wird deshalb
**stochastisch gerundet**: `n = ceil(p)` mit Wahrscheinlichkeit `p − floor(p)`,
sonst `floor(p)`. Im Mittel kommt genau `p` heraus.

### Wie Fragmente Eigenschaften erben

```
V_solid_frag       ← aus der Fragmentverteilung (Summe = V_solid_parent)
frag_poro          ← porosity_growth_kernel.compute_fragment_porosity(...)
V_dry_frag         = V_solid_frag / (1 − frag_poro)
V_intern, V_extern ← anteilig am Volumenanteil des Fragments
```

Anschließend läuft die **umgekehrte Braumann-Rechnung**: Beim Bruch entstehen neue
Oberflächen, das gesamte Porenvolumen der Fragmente ist kleiner als das des
Elternteils. Die dadurch „herausgedrückte" interne Flüssigkeit wird extern
gemacht (`compute_liquid_externalization_breakage`) und proportional auf die
Fragmente verteilt. Die **Summe** der Fragment-Flüssigkeiten bleibt dabei exakt
gleich der Eltern-Flüssigkeit – es verschieben sich nur die Töpfe intern/extern.

Ein Sonderfall wird explizit gekappt: `porosity`, `saturation` und
`liquid_volume` sind unabhängig setzbare Größen und nicht garantiert konsistent
(z. B. nach manuellem Testaufbau). Deshalb wird die interne Flüssigkeit auf den
tatsächlich gespeicherten Wert begrenzt:
`V_intern_p = min(V_pore·S, liquid_volume)`.

## 3.4 `NucleationHandler` – Flüssigkeitszugabe

Der mit Abstand umfangreichste Handler (2721 Zeilen), weil er als einziger
**Partikel erzeugen, verschmelzen und löschen** muss, ohne ein reguläres
MC-Ereignis zu sein.

### Zeitfenster und Menge

Die Zugabedauer kann auf zwei Wegen angegeben werden:

- direkt über `liquid_addition_duration`, oder
- berechnet aus `target_wt_percent` + `solid_mass_in_mixer` + Dichten.

Werden **beide** angegeben, müssen sie auf 0,1 % übereinstimmen – sonst gibt es
einen Fehler. Das verhindert stillschweigend widersprüchliche Konfigurationen.

### `_distribute_liquid_volume` – warum ein Restspeicher

Pro Ereignis kommt ein Volumen `Q̇·dt` dazu. Das entspricht meist keiner ganzen
Tropfenzahl. Würde man abrunden, ginge systematisch Flüssigkeit verloren.

Deshalb: der Rest wird in `_liquid_remainder` **akkumuliert** und beim nächsten
Mal mitgezählt. Ganze Tropfen werden verteilt, ein verbleibender Rest am Ende des
Fensters als „Mini-Tropfen". `finalize_after_solve` sorgt zusätzlich dafür, dass
auch dann die volle Menge ankommt, wenn während des Fensters gar keine
MC-Ereignisse stattfanden.

### `_distribute_one_droplet_with_dW` – der Kern

```
1. Zielpartikel i wählen
      → über liquid_dist_kernel (uniform / oberflächengewichtet /
        sättigungspräferenziell) oder gewichtet über W
2. Passt der Tropfen auf das Partikel?
      V_dry[i] < benötigtes Volumen  →  ZWANGSAGGLOMERATION:
          Partner suchen, manuell agglomerieren, bis genug V_dry da ist
3. Neue Geometrie bestimmen
      porosity_growth_kernel.compute_nucleation_porosity(v_solid, v_liquid, …)
4. Ähnliches Partikel suchen?
      ja  → Gewicht dorthin addieren
      nein → neues Partikel anlegen
5. Quellpartikel um dW entlasten
```

**DSMC-Skalierung.** Die physikalische Tropfenzahl muss unabhängig davon stimmen,
wie oft `Vc` inzwischen verdoppelt wurde. Deshalb:
`effective_dW = dW · (Vc_ref / Vc)`.

**Warum eine eigene Agglomerationsroutine?** `_manual_agglomerate_particles` ist
absichtlich **nicht** `mcpbe_agg::_merge_pair`. Innerhalb der Tropfenschleife darf
kein Propensity-Rebuild laufen – das wäre O(n) *pro Tropfen* statt pro Ereignis.
Der Preis dafür ist, dass diese Routine dieselben Regeln nachbilden muss
(insbesondere die `i == j`-Kappung auf `min(δ_i, W_i/2)`). Nachgeholt wird der
Rebuild einmal pro MC-Ereignis über `consume_population_changed()` (siehe 3.1).

**Index-Nachverfolgung.** Weil die Zwangsagglomeration Eltern löschen kann und
dabei Swap-with-last greift, wird der Index des frisch erzeugten Kindes über
`child_current_idx` aktiv mitgeführt.

## 3.5 `ContinuousProcessesHandler` – Internalisierung und Kompression

Zwei Prozesse, in fester Reihenfolge pro `step(t, dt)`:

**1. Liquid Internalization** (Braumann et al. 2007)

```
dl_intern/dt = k_int · l_extern · (V_pore − l_intern)
```

Kapillargetriebenes Eindringen. Die Rate ist proportional zum verfügbaren
Oberflächenfilm **und** zum noch freien Porenraum – sie geht also gegen null,
wenn entweder nichts mehr außen ist oder die Poren voll sind. Integriert wird
explizit (Euler) mit Kappung auf `[0, min(l_total, V_pore)]`.
`liquid_volume` bleibt unangetastet – es verschiebt sich nur `saturation`.

**2. Porosity Compression**

```
ε(t+dt) = ε_min + (ε(t) − ε_min) · exp(−k·dt)
```

Analytische Lösung des exponentiellen Zerfalls, also stabil für beliebiges `dt`.
Kompression ist **ausschließlich reduktiv**: Partikel mit `ε ≤ ε_min` bleiben
unverändert (das schließt porenfreie Partikel automatisch mit ein, weil
`ε_min ≥ 0`).

Danach wird das Volumen so nachgezogen, dass **`V_solid` exakt konstant bleibt**:

```
V_solid   = V_dry_alt · (1 − ε_alt)          ← festgehalten
V_pore_neu = V_solid · ε_neu / (1 − ε_neu)
V_dry_neu = V_solid + V_pore_neu
```

Schrumpfen die Poren, steigt die Sättigung. Übersteigt sie 1, wird der Überschuss
extern – `liquid_volume` (die Gesamtmenge) bleibt dabei per Konstruktion erhalten.

Beide Schritte sind **vektorisiert**. Sie laufen einmal pro MC-Ereignis über die
gesamte Population; eine Python-Schleife darüber dominierte früher die Laufzeit.

## 3.6 Kontrollvolumen-Verdopplung

`_maybe_double_control_volume` greift, wenn die Partikelzahl unter 50 % eines
Referenzwerts fällt:

```
Vc      → 2·Vc
Spalten → dupliziert (jedes Partikel zweimal)
W       → KOPIERT, nicht halbiert
```

Warum kopiert? Weil das doppelte Kontrollvolumen auch doppelt so viele physische
Partikel enthält. `n_phys = ΣW/Vc` bleibt damit konstant, während `n_comp` sich
verdoppelt – genau das war das Ziel.

Die **intensiven** Eigenschaften werden schlicht mitkopiert: Die duplizierten
Rechenpartikel repräsentieren physische Partikel mit denselben Eigenschaften.

Der Referenzwert für den Trigger ist `_cv_a_ref` (die Partikelzahl nach der
Initialisierung), **nicht** ein fester Startwert – sonst würde bei aktivierter
`V_eff_init`-Kompression sofort nach dem Start fälschlich verdoppelt.

Zum Schluss werden die Sampler neu aufgebaut, und `W0` (die Referenz für
Massechecks) wird verdoppelt, damit die Bilanz weiter aufgeht.

## 3.7 `ParticleMerger` – Dedup neuer Partikel

**Das Problem:** Bricht ein Partikel in vier Fragmente, entstehen vier neue
Rechenpartikel. Brechen 1000 identische Partikel, entstehen 4000 – obwohl es
physikalisch nur vier *verschiedene* Sorten gibt. `n_comp` explodiert, und da der
Propensity-Aufbau mit n skaliert, wird die Simulation unnötig langsam.

**Die Lösung:** Bevor ein neues Partikel angelegt wird, wird geprüft, ob es schon
ein hinreichend ähnliches gibt. Wenn ja, wird **nur `W` erhöht**:

```python
if match:
    solver.W[match_idx] += weight_to_add   # sonst NICHTS
```

Das ist exakt massenerhaltend, weil alle anderen Eigenschaften intensiv sind – ein
zusätzliches physisches Partikel derselben Sorte ändert sie nicht.

### Warum die Toleranzen unterschiedlich normiert sind

| Größe | Toleranz | Begründung |
|---|---|---|
| `V_dry` | **relativ** (`tol_rel`, 1e-6) | erstreckt sich über viele Größenordnungen |
| `liquid_volume` | relativ, optional gedeckelt, mit absolutem Boden | s. u. |
| `porosity`, `saturation` | **absolut** (`tol_abs_frac`, 1e-6) | dimensionslose Brüche in [0,1] |

Der Punkt bei Porosität/Sättigung: eine relative Toleranz ist bei 0 undefiniert –
und genau dort sitzen die porenfreien Partikel. Sie würde außerdem 0,0001 gegen
0,0002 als 100 % Abweichung werten, aber 0,80 gegen 0,799992 als identisch. Für
eine beschränkte Größe ist absolut die richtige Norm.

Bei der Flüssigkeit gibt es zusätzlich einen **Deckel gegen eine Referenzskala**
(in der Praxis das Tropfenvolumen). Sonst wächst die zulässige absolute Differenz
mit der bereits vorhandenen Flüssigkeitsmenge: späte Merges würden immer größere
Unterschiede schlucken, während die Statistik weiterhin ganze Tropfen bucht. Der
Deckel kann die Toleranz nur verschärfen, nie lockern.

### Hash-Index

Statt linear zu scannen, gruppiert ein Dictionary die Partikel nach *gebinnten*
Eigenschaften:

```python
key = (round(log10(V_dry), 8), round(log10(liquid), 8), round(poro, 4), round(sat, 4))
```

Volumina logarithmisch (viele Größenordnungen), Brüche linear (beschränkt).
Da Binning nur eine Näherung ist, wird jeder Kandidat aus dem Bucket **noch einmal
exakt** gegen die Toleranzen geprüft (`_matches_exact`). Der Hash ist damit reine
Beschleunigung und kann für sich genommen kein falsches Ergebnis erzeugen.

Es gibt einen `force_linear_scan`-Schalter: Der Hash binnt die Flüssigkeit
logarithmisch, ein Treffer innerhalb einer *absoluten* Toleranz kann also im
Nachbar-Bin liegen und übersehen werden. Der lineare Scan liefert außerdem
immer den **kleinsten** passenden Index, was Läufe reproduzierbar hält.

## 3.8 `ReconstructionMixin` – Partikelzahl senken

Wenn `a_tot` über `recon_N_max` (typisch 4000) steigt, wird die Population auf
eine kleinere, **momententreue** Repräsentation abgebildet. Vier Verfahren:

| Methode | Idee |
|---|---|
| **CAM** (Cell Average) | Gitter über den Volumenraum; jede Zelle wird auf ihre Stützstellen so verteilt, dass Zahl und Volumen der Zelle erhalten bleiben |
| **RS** (Resampling) | gewichtetes systematisches Resampling mit anschließender Korrektur zweier Repräsentanten, damit das erste Moment exakt stimmt |
| **2PM** (Two-Point) | ersetzt eine Gruppe durch zwei Punkte, die M0, M1 und M2 exakt reproduzieren |
| **QMX** | Hybrid: Quantile plus Mischung der obigen |

Gemeinsames Prinzip: **Momente sind die Erhaltungsgröße**, nicht die einzelnen
Partikel. `_assert_reconstruction_is_safe` prüft vorher die Voraussetzungen,
`_post_reconstruct_safety` danach das Ergebnis. Anschließend werden `X` aus
`V_dry` neu berechnet, die Sampler neu aufgebaut und der Hash-Index des Mergers
neu erzeugt – die alten Indizes sind nach einer Rekonstruktion bedeutungslos.

Optional lassen sich Partikel über `_recon_select_protected` von der Reduktion
ausnehmen (z. B. die größten, deren Auflösung man nicht verlieren will).

## 3.9 `FenwickSampler` – gewichtetes Ziehen in O(log n)

Ein **Binary Indexed Tree** (Fenwick-Baum), Numba-kompiliert.

Das Problem: aus n Partikeln eines proportional zu `r_i` ziehen. Naiv braucht das
eine kumulierte Summe – O(n) pro Ziehung und O(n) pro Gewichtsänderung.

Der Fenwick-Baum speichert Teilsummen in einer Baumstruktur:

| Operation | Kosten |
|---|---|
| `sample(rng)` | O(log n) |
| `update(idx, w)` | O(log n) |
| `total()` | O(1) |
| `append` / `remove` | O(log n), amortisiert |

`remove(idx)` benutzt dieselbe **Swap-with-last**-Semantik wie
`_remove_particle_column` – dadurch bleiben Sampler und Zustandsarrays ohne
Zusatzaufwand indexgleich.

## 3.10 `MCPBEPost` – Auswertung

Zwei Ebenen:

**`ParticlePropertyContainer`** – ein Auswerte-Objekt, das alle Snapshots als
rechteckige `(T, N_max)`-Arrays bereitstellt (mit NaN aufgefüllt, wo weniger
Partikel aktiv waren). Bietet `filter`, `time_series`, `scatter_data` und
`plot_correlation`, um Zusammenhänge wie „Porosität über Durchmesser zum
Zeitpunkt t" direkt untersuchen zu können.

**Momente und PSD** – `calc_moments_over_time` liefert µ(i,j,t),
`compute_psd_cdf_over_time` die Summenverteilungen. Wahlweise zahl- (Q0) oder
volumenbasiert (Q3), und wahlweise als `Q(x)` oder invertiert als `x(Q)` (für
d10/d50/d90). Die Inversion `_invert_cdf_monotone` behandelt Plateaus und
nicht-strenge Monotonie explizit, weil eine MC-CDF nie glatt ist.

Für Ensembles mittelt `aggregate_psd_repeats` über Realisierungen. Die
Durchmesser werden hier durchgängig aus `V_dry` neu berechnet.

---

# Stufe 4 – Das Kernel-Framework

## 4.1 Warum überhaupt Kernel

Die Physik der Nassgranulation ist nicht abschließend geklärt – für jeden
Teilprozess gibt es konkurrierende Modelle. Statt sie mit `if`-Kaskaden in den
Solver zu schreiben, sind sie als **austauschbare Objekte** hinter festen
Schnittstellen gekapselt.

`KernelManager` (`kernel_integration.py`) hält acht Steckplätze und reicht
Berechnungen an den jeweils konfigurierten Kernel weiter. Alle sind optional
außer der Aggregation – ohne Kollisionsmodell gibt es keine Simulation, deshalb
wirft das Fehlen dort einen `ValueError` mit einer Liste der verfügbaren Kernel.

## 4.2 Die acht Steckplätze

| Steckplatz | Beantwortet die Frage | Implementierungen |
|---|---|---|
| `agg_kernel` | Wie oft stoßen zwei Partikel zusammen? β(r₁,r₂) | `shear_chin1998`, `brownian_tsouris1995`, `constant`, `sum_kernel`, `liquid_bridge` |
| `agglomeration_acceptance_kernel` | Bleibt der Stoß haften? | `stokes_krit`, `fittable` |
| `break_kernel` | Wie oft bricht ein Partikel? S(V) | `power_law`, `powerlaw_rumpf` |
| `porosity_growth_kernel` | Wie entwickelt sich der Porenraum? | `volume_mixing`, `cone_model`, `incomplete_mixing` |
| `liquid_dist_kernel` | Welches Partikel trifft der Tropfen? | `uniform_weighted`, `surface_weighted`, `saturation_preferential` |
| `porosity_compression_kernel` | Wie schnell kollabieren Poren? | `porosity_compression` |
| `liquid_internalization_kernel` | Wie schnell zieht Flüssigkeit in die Poren? | `liquid_internalization` |
| `liq_internalisation_agglomeration_kernel` | Wie viel Film wird beim Stoß eingeschlossen / beim Bruch freigesetzt? | `liq_internalisation_agglomeration` |

Jede Kategorie hat eine `blueprint.py` mit der abstrakten Basisklasse. Die
gemeinsamen Basisklassen stehen in `kernels/base.py`.

## 4.3 Parametervalidierung

Kernel werden über Name + Parameter-Dict konfiguriert:

```python
MCPBESolver(
    agg_kernel_name='shear_chin1998',
    agg_kernel_params={'corr_beta': 1e-3, 'g': 1000},
    ...
)
```

Jede `get_*_kernel(...)`-Factory ruft `kernels/base.py::reject_unknown_params` und
wirft einen `ValueError`, wenn ein übergebener Name von diesem Kernel gar nicht
gelesen wird – **mit Vorschlag des vermutlich gemeinten Namens**.

Der Grund: `get_default_params()` wird mit den Aufrufer-Werten aktualisiert. Ein
unbekannter Schlüssel überschreibt dabei nichts, er landet nur zusätzlich im Dict –
der Kernel läuft still auf seinem Default weiter. Ein Tippfehler war damit
unsichtbar.

Kernel mit absichtlich default-losen Parametern deklarieren diese über
`get_optional_params()` (derzeit nur `cone_model` mit `default_porosity` und
`liquid_split_ratio`).

## 4.4 Eigentümerschaft geteilter Größen

Manche Parameter beschreiben den **Prozess**, nicht ein einzelnes Modell. Die
Scherrate `G` ist das Paradebeispiel: sie taucht im Agglomerations- und im
Bruchkernel auf, beschreibt aber denselben Mischer.

Die Regel im Code: **der Aggregationskernel besitzt `solver.G`.** Der Bruchkernel
rechnet mit seinem eigenen `g`, überschreibt `solver.G` aber nicht. Weicht sein
`g` ab, gibt es eine `UserWarning` statt einer stillen Änderung.

Warum das wichtig ist: `solver.G` wird auch von der Nucleation gelesen (für die
Kollisionsgeschwindigkeit). Ohne diese Regel folgte die Nucleation
stillschweigend dem Bruchmodell.

Ähnlich getrennt sind zwei Parameter, die historisch denselben Namen tragen:

| Name | Bedeutung | Wo |
|---|---|---|
| `pl_v` im **Bruchkernel** | Volumenexponent der **Rate** | Kernel-Parameter |
| `break_frag_v` (Fallback `pl_v`) | Exponent der **Fragmentgrößenverteilung** | Solver-Attribut |

Änderte man früher den Ratenexponenten, änderte sich stillschweigend auch die
Fragmentanzahl. Heute sind es getrennte Größen mit getrennten Namen.

## 4.5 Die wichtigen Kernel im Detail

### `shear_chin1998` (Aggregation)

Scherinduzierte Kollision im gerührten Behälter:
`β ∝ corr_beta · G · (r₁+r₂)³`. Separierbar → Momentenmodus verfügbar.

### `liquid_bridge` (Aggregation)

Bezieht die Sättigung ein: die Kollisionsrate ist um eine Gauß-Glocke um eine
optimale Sättigung `s_opt` moduliert. Zu trocken → keine Brücke; zu nass → das
Granulat wird weich. **Nicht separierbar**, weil der Ausdruck einen Kreuzterm
`s_i·s_j` enthält – dafür gibt es einen eigenen kompilierten O(n²)-Pfad.

### `stokes_krit` (Akzeptanz, Braumann 2007)

Entscheidet, ob eine Kollision haftet, über die **Stokes-Zahl**:

```
St      = m_harm · U_coll / (3π · η · R_harm²)
St_crit = (1 + 1/e_coag) · ln(h / h_a)
haftet  ⟺  St < St_crit
```

Anschaulich: `St` misst Trägheit gegen viskose Dämpfung im Flüssigkeitsfilm. Ist
die Trägheit klein genug, wird die Stoßenergie im Film dissipiert und die Partikel
bleiben zusammen.

Die Bestandteile:
- `e_coag = √(e₁·e₂)` mit `e_i = m_solid/m_total` – ein effektiver
  Restitutionskoeffizient. Viel Flüssigkeit → kleines `e` → mehr Dissipation →
  größeres `St_crit` → haftet leichter.
- `h` ist die Filmdicke aus dem **externen** Flüssigkeitsvolumen, berechnet als
  Differenz der Kugelradien von nassem und trockenem Volumen.
- `h_a` ist der Abstand der dichtesten Annäherung (Rauigkeitsmaß).

Zwei Sonderfälle: Sind **beide** Partikel trocken (`h₁+h₂ ≤ 0`), gibt es keine
Brücke → Ablehnung. Ist nur einer trocken, ist das erlaubt – deshalb wird
bewusst das **arithmetische** Mittel `(h₁+h₂)/2` verwendet, das mit `h = 0` auf
einer Seite umgehen kann (ein harmonisches Mittel wäre dort null).
Ist der Film dünner als `h_a`, wird `ln(h/h_a) ≤ 0` und `St_crit` nicht-positiv →
Ablehnung, ohne dass gerechnet werden muss.

### `powerlaw_rumpf` (Breakage)

Kombiniert eine Potenzgesetz-Grundrate mit der **Rumpf'schen Festigkeitstheorie**:

```
S(V) = S_base(V) / σ(ε, S_sat)
```

`σ` ist die Zugfestigkeit des Granulats. Sie steigt mit sinkender Porosität
(dichter gepackt = fester) und hängt stark von der Sättigung ab:

- **trocken** (S < Schwelle): Festigkeit aus Feststoffbrücken, Faktor `k`
- **nass** (S > Schwelle): Kapillarkraft,
  `σ_wet = 6·α·((1−ε)/ε)·(γ·cos δ / x_s)·S`
- dazwischen: linear überblendet, damit `σ(S)` stetig bleibt

`x_s` ist der Sauterdurchmesser der Primärpartikel (bei Bedarf aus `X0` berechnet).

Zwei Ränder werden ausdrücklich behandelt:
- **ε = 0 oder NaN** (porenfrei): `σ ∝ (1−ε)/ε` divergiert, also `S → 0`. Der
  Kernel gibt `0.0` zurück – das **ist** der Grenzwert der Formel. Ein
  porenfreier Körper hat keinen Poren-/Kapillar-Versagensmechanismus.
- **ε über `poro_max`**: wird **geklemmt**, nicht abgebrochen. So bleibt `σ`
  endlich und `S(ε)` stetig, statt genau dort einen Sprung zu haben, wo das
  Granulat am schwächsten ist.

### `cone_model` (Porositätswachstum)

Geometrisches Modell: Stoßen zwei Kugeln zusammen, entsteht an der Kontaktstelle
ein zusätzlicher Hohlraum, dessen Volumen sich über eine **Kegelstumpf-Formel**
aus der Kontaktgeometrie ergibt.

```
V_solid_neu = V_solid_1 + V_solid_2                (exakt)
V_pore_neu  = V_pore_1 + V_pore_2 + ΔV_geometrisch
V_dry_neu   = V_solid_neu + V_pore_neu
```

Beim Bruch läuft das rückwärts: neue Bruchflächen reduzieren das Porenvolumen um
`k_break · ΣΔV`.

Für die Nucleation startet `cone_model` neue Partikel mit `poro = 0.0`: Porosität
soll erst durch die *nachfolgende* Agglomeration entstehen, nicht schon beim
Benetzen. (Über `default_porosity` lässt sich das umstellen.)
`volume_mixing` – das einfachere Modell – setzt hier `0.4`, damit überhaupt
Porenraum vorhanden ist.

### `volume_mixing` (Porositätswachstum)

Rein additiv: Porenvolumina werden summiert, es gibt keinen Porenkollaps.
Der konservative Standard und der Default-Fallback, wenn kein Kernel konfiguriert
ist.

### `liq_internalisation_agglomeration`

Bedient **beide Richtungen** aus demselben Steckplatz:
`compute_internalization` beim Stoß (Film wird in Kontaktporen eingeschlossen),
`compute_externalization` beim Bruch (verlorener Porenraum drückt Flüssigkeit
heraus). Analog dazu bedient `porosity_growth_kernel` sowohl Merge als auch
Fragment. Ist kein Kernel konfiguriert, liefern beide `0.0` – also schlicht kein
Transfer, was ältere Konfigurationen unverändert weiterlaufen lässt.

---

# Stufe 5 – Wiederkehrende Muster

Diese Muster tauchen an vielen Stellen auf. Wer sie einmal verstanden hat, liest
den Rest des Codes deutlich schneller.

### Intensiv bleibt intensiv

Immer wenn `W` reduziert wird, bleiben `liquid_volume`, `porosity`, `saturation`
und `V_flat` **unverändert**. Die übriggebliebenen physischen Partikel sind
dieselben wie vorher – es sind nur weniger.

### Feststoff wird addiert, nie modelliert

Bei jedem Ereignis, das Partikel zusammenführt oder teilt, wird `V_solid`
**direkt** summiert bzw. aufgeteilt. Kernel dürfen nur den *Porenanteil*
bestimmen. Damit kann kein Modell die Massenbilanz verletzen.

### Erst rechnen, dann schreiben

Zusammengeführte Eigenschaften werden vollständig in lokalen Variablen berechnet,
bevor irgendetwas in die Arrays geschrieben wird. So gibt es keinen Zeitpunkt, an
dem der Zustand halb alt und halb neu ist – wichtig, weil der `ParticleMerger`
dazwischen auf den Bestand zugreift.

### Absteigend löschen

Müssen mehrere Partikel entfernt werden, geschieht das in absteigender
Indexreihenfolge (`sorted({i, j}, reverse=True)`). Dann kann Swap-with-last die
noch offenen Indizes nicht verschieben.

### Grenzfälle sind der Grenzwert, nicht ein Fallback

Wo eine Formel an einem Rand divergiert, gibt der Code den **mathematischen
Grenzwert** zurück – nicht einen „neutralen" Ersatzwert. Beispiel: `ε → 0` im
Rumpf-Modell ergibt Bruchrate `0`, nicht die unkorrigierte Grundrate. Ein
Ersatzwert wäre ein stiller Regimewechsel mitten in der Parameterlandschaft und
würde die Physik an dieser Stelle umkehren.

### Laut scheitern statt still weiterlaufen

NaN/inf in Bruchraten werfen einen `RuntimeError` mit Diagnose. Ein widersprüchlicher
Snapshot-Zustand wirft. Ein unbekannter Kernel-Parameter wirft. Die Leitlinie:
Ein Fehler, der die Simulation weiterlaufen lässt, kostet Tage; einer, der sie
anhält, kostet Minuten.

---

# Anhang – Praktisches

### Ausführen

```bash
"C:/Users/ericb/anaconda3/envs/Masterarbeitv4/python.exe" -m wmcpbe.Trials.test_powerlaw_rumpf_full
```

Aus dem Verzeichnis `mcpbe/src`. Für saubere Unicode-Ausgabe zusätzlich
`PYTHONIOENCODING=utf-8` setzen, sonst brechen `print()`-Aufrufe mit Sonderzeichen
auf einer cp1252-Konsole ab.

### Typische Initialisierungsreihenfolge

```python
solver = MCPBESolver(dim=1, t_vec=..., load_attr=False, <kernel-konfiguration>)
solver.process_type = "mix"
solver._initialize_particles(init_Vc=False, V_flat=V_flat, W_init=W_init)
solver.porosity[:solver.a_tot] = ...          # intensive Startwerte
solver._initialize_samplers()                  # NACH den Partikeln!
solver.create_nucleation_handler(...)
solver.create_continuous_processes_handler(...)
solver.solve()
```

`_initialize_samplers()` muss **nach** `_initialize_particles()` kommen – es baut
die Propensities aus dem Partikelzustand auf. `_validate_solve_readiness()` prüft
das zu Beginn von `solve()` und erklärt im Fehlerfall die richtige Reihenfolge.

### Debug-Schalter

`solver.mcpbe_debug_mass` schaltet detaillierte Massenbilanz-Ausgaben an vielen
Stellen frei; ergänzend `mcpbe_debug_agg/_break/_merger/_nuc/_comp/_vc`.
`_debug_max_events` (Default 20) begrenzt die Ausgabemenge.

Für Untersuchungen über einen **kompletten** Lauf ist ein externes
Monkeypatch-Skript oft geeigneter: `sys.path` auf `mcpbe/src` setzen, die
Zielmethode durch einen Wrapper ersetzen, der vor und nach dem Originalaufruf
`Σ(V_solid·W)` misst. So lässt sich jede Abweichung kategorienweise
(AGG/BREAK/NUC/COMP/VC) zuordnen, ohne eine einzige Repo-Datei zu ändern.

### Nützliche Helfer

`helpers.py` enthält fertige Konfigurationen (`create_dry_agglomeration_solver`,
`create_wet_granulation_solver`, `create_breakage_test_solver`) sowie
`validate_mass_conservation(solver)` für einen schnellen Bilanzcheck.

---

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
| `V_solid` | Feststoffvolumen – **Erhaltungsgröße** |
| `V_dry` | `V_solid + V_pore` – keine Erhaltungsgröße |
| `dW` | Paketgröße: wie viele physische Partikel ein Ereignis bewegt |
| `δ_i` | `min(dW_const, W_i)` – maximal ausführbares Paket für Partikel i |
| ε | Porosität = `V_pore / V_dry` |
| S | Sättigung = `V_liq_intern / V_pore` |
| Vollkörper | porenfreies Partikel (`ε = 0`, historisch `NaN`) |
| β(i,j) | Kollisionsfrequenz eines Paares |
| S(V) | Bruchrate eines Partikels |
| `R_i*` | bias-korrigierte Agglomerations-Propensity von Partikel i |

---

*Diese Datei beschreibt den Aufbau des Codes. Für den Verlauf einzelner Bugfixes
siehe die Git-Historie und die thematischen Dokumente in diesem Verzeichnis.*
