# WMCPBE Refactoring — Befundregister

**Umfang:** `mcpbe/src/wmcpbe/` (Weighted-DSMC-Granulationssolver)
**Referenz-Commit (Baseline):** `7a35310`
**Umgebung:** Windows 11, Python 3.11.9, numpy 2.3.5, numba 0.63.1, scipy 1.17.1

Dieses Dokument listet jeden gefundenen Defekt bzw. jede technische Schuld auf,
mit Nachweis (Messung), Ursache und der gewählten Lösung. Reine
Performance-Arbeit ist in [`PERFORMANCE.md`](PERFORMANCE.md) beschrieben, der
O(n)-Moment-Modus im Detail in [`MOMENT_MODE.md`](MOMENT_MODE.md), die
Massenerhaltungs-Verifikation in [`CONSERVATION.md`](CONSERVATION.md).

## Legende

| Feld | Bedeutung |
|---|---|
| **Schweregrad** | `kritisch` = falsche Physik/Massenverlust · `hoch` = falsche Ergebnisse unter bestimmten Konfigurationen · `mittel` = Robustheit/Wartbarkeit · `niedrig` = Kosmetik |
| **Ergebniswirkung** | Ändert der Fix numerische Resultate? |

---

## Übersicht

| ID | Titel | Schweregrad | Ergebniswirkung |
|---|---|---|---|
| [F-01](#f-01) | 2D-Agglomeration schreibt Feststoffvolumen skalar in alle Komponenten | kritisch | ja (nur `dim=2`) |
| [F-02](#f-02) | Rekonstruktion verwirft Flüssig-/Porositätszustand | kritisch | nein (jetzt Fehler statt stiller Fehlrechnung) |
| [F-03](#f-03) | Nucleation-Resttropfen zerstört die Flüssigkeit des Elternpartikels | kritisch | ja (nur Granulation) |
| [F-04](#f-04) | Agglomerations-Propensity ist O(n²) pro Ereignis | hoch (Performance) | nein |
| [F-05](#f-05) | `liquid_bridge`-Kernel fällt auf n² Python-Aufrufe zurück | hoch (Performance) | nein |
| [F-06](#f-06) | Kontinuierliche Prozesse als Python-Schleife pro Ereignis | hoch (Performance) | nein |
| [F-07](#f-07) | „Links“-Snapshots mischen Vor- und Nach-Ereignis-Zustand | mittel | nein (dokumentiert) |
| [F-08](#f-08) | `u_sel` wird doppelt verwendet (Partnerwahl + Akzeptanz) | mittel | nein (dokumentiert) |
| [F-09](#f-09) | Reine 2D-Agglomeration bricht an Breakage-Parameter ab | mittel | nein |
| [F-10](#f-10) | Toter Code-Zweig mit garantiertem `NameError` | mittel | nein |
| [F-11](#f-11) | `BREAKFVAL` inkonsistent zwischen Fragmentzahl und CDF-Tabelle | mittel | nein (dokumentiert) |
| [F-12](#f-12) | Unbedingte Debug-Ausgaben in Bibliothekscode | mittel | nein |
| [F-13](#f-13) | `_find_similar_particle`: Toleranz an falscher Größe gemessen | mittel | nein (Opt-in) |
| [F-14](#f-14) | Freigegebene Array-Slots behalten Altwerte | niedrig | nein |
| [F-15](#f-15) | Toter Code: 3 Paketkopien, 4 Module, 80 Debug-Skripte | niedrig | nein |
| [F-16](#f-16) | `except Exception` verschluckt Kernel-Fehler | niedrig | nein |

---

<a name="f-01"></a>
## F-01 — 2D-Agglomeration schreibt das Feststoffvolumen skalar in alle Komponenten

**Schweregrad:** kritisch · **Datei:** `src/wmcpbe/mcpbe_agg.py` (vorher Zeile 703)

### Problem

Beim Verschmelzen zweier Partikel wurde das gemeinsame Feststoffvolumen als
**Skalar** in *alle* Komponentenspalten geschrieben:

```python
V_solid_merged = V_solid_i + V_solid_j        # skalar
self.V_flat[:self.dim, new_idx] = V_solid_merged   # broadcast über dim!
```

Für `dim=1` ist das korrekt. Für `dim=2` bekommt jede der beiden Komponenten
den **vollen** Wert, d. h. `sum(V_flat[:dim])` ist doppelt so groß wie das
physikalische Feststoffvolumen. Zusätzlich geht die Zusammensetzung
(Komponentenverhältnis) des Kindpartikels vollständig verloren — sie wird immer
50/50, unabhängig von den Eltern.

Damit widersprechen sich die beiden Repräsentationen desselben Zustands:

* `V_flat[:dim]` — Feststoffvolumen pro Komponente
* `V_flat[-1] * (1 - porosity)` — Feststoffvolumen aus Trockenvolumen und Porosität

### Nachweis

Szenario `agg_shear_2d` (dim=2, 500 Startpartikel, 300 Ereignisse), gemessen mit
dem Originalcode aus `7a35310`:

```
before: sum(W*Vsolid_from_dry)=2.617994e-16   sum(W*sum(V_flat[:dim]))=2.617994e-16
after : sum(W*Vsolid_from_dry)=2.617994e-16   sum(W*sum(V_flat[:dim]))=4.188790e-16
rel. Abweichung Komponenten-Route : 6.000e-01     <-- +60 %
```

Der Trockenvolumen-Weg bleibt erhalten, der Komponenten-Weg wächst um 60 %.
Nachgelagerte Auswertungen, die auf `V_flat[:dim]` beruhen — insbesondere
`calc_moments_over_time()` für `dim=2` sowie jede Breakage in 2D
(`Vrem_k = [V_flat[0,k], V_flat[1,k]]`) — arbeiten daher mit der falschen Masse.

### Lösung

Das gemeinsame Feststoffvolumen wird **proportional zu den Komponentenvolumina
der Eltern** verteilt. Für `dim=1` bleibt der Code bitgenau identisch:

```python
if dim == 1:
    self.V_flat[0, new_idx] = V_solid_merged
else:
    comp = Vi_comp + Vj_comp
    comp_total = float(np.sum(comp))
    if comp_total > 0.0:
        self.V_flat[:dim, new_idx] = comp * (V_solid_merged / comp_total)
    else:
        self.V_flat[:dim, new_idx] = V_solid_merged / dim
```

### Verifikation

Nach dem Fix stimmen beide Wege exakt überein (Abweichung `0.000e+00`), und die
Feststoffmasse driftet über den ganzen Lauf um `1.9e-16` (Maschinengenauigkeit).
Regressionstest: `tests/test_conservation.py::test_volume_representations_agree`.

### Auswirkung auf Ergebnisse

`dim=2`-Läufe ändern sich. Im Benchmark-Szenario endet der Lauf mit 286 statt
349 Rechenpartikeln, weil bei korrekten (kleineren) Volumina andere Kollisionen
akzeptiert werden. **Alle bestehenden 2D-Ergebnisse mit Agglomeration sind zu
verwerfen.** `dim=1` ist bitgenau unverändert.

---

<a name="f-02"></a>
## F-02 — Rekonstruktion verwirft den intensiven Granulationszustand

**Schweregrad:** kritisch · **Datei:** `src/wmcpbe/reconstruction_mixin.py`

### Problem

`_recon_replace_active_particles()` baut die Partikelpopulation neu auf, fasst
dabei aber nur drei Arrays an:

```python
self.V_flat[:, :] = 0.0
self.W[:] = 0.0
self.X[:] = 0.0
...
self.V_flat[-1, j] = float(np.sum(vcomp))     # V_dry := V_solid
```

`liquid_volume`, `porosity` und `saturation` werden **überhaupt nicht
angefasst**. Nach der Rekonstruktion beschreiben diese Arrays daher die
*vorherigen* Partikel an denselben Indizes — sie gehören zu völlig anderen
Partikeln. Zusätzlich wird `V_dry` auf das Feststoffvolumen gesetzt, wodurch das
Porenvolumen verschwindet, während `porosity` weiter einen Wert > 0 meldet.

Folge: `V_solid = V_dry * (1 - porosity)` ist nach der Rekonstruktion falsch,
und `sum(W * liquid_volume)` ist eine Zufallszahl.

### Nachweis

Granulationsszenario mit erzwungener Rekonstruktion (`recon_N_max=200`,
Methode `RS`), ein einziger Rekonstruktionsschritt:

```
recon_enable=False  recon_count=0   Feststoff-Drift = 3.5e-16   (erhalten)
recon_enable=True   recon_count=1   Feststoff-Drift = 4.4e-03   (0.44 % verloren)
```

Der Fehler akkumuliert mit jedem weiteren Rekonstruktionsschritt.

### Lösung

Eine physikalisch korrekte Übertragung von Flüssigkeit und Porosität durch
CAM/RS/2PM/QMX ist eine **Modellierungsentscheidung** (wie mittelt man Sättigung
über zusammengefasste Partikel?) und keine reine Refactoring-Aufgabe. Statt eine
Annahme zu treffen, schlägt die Operation jetzt **laut fehl**:

```python
def _assert_reconstruction_is_safe(self) -> None:
    ...
    raise NotImplementedError(
        "Reconstruction does not carry the intensive granulation state ..."
    )
```

* Trockene Läufe (keine Flüssigkeit, keine Porosität) sind nicht betroffen und
  laufen unverändert.
* `solver.recon_allow_granulation_state = True` erlaubt das alte Verhalten
  bewusst und dokumentiert.

### Offener Punkt

Für nasse Läufe mit Rekonstruktion muss festgelegt werden, wie
`liquid_volume`/`porosity`/`saturation` auf die neuen Repräsentanten abgebildet
werden. Vorschlag: pro Zelle W-gewichtete Mittelung der intensiven Größen und
anschließende Neuberechnung von `V_dry = V_solid / (1 - porosity)`. **Das ist mit
Eric abzustimmen.**

---

<a name="f-03"></a>
## F-03 — Nucleation-Resttropfen zerstört die Flüssigkeit des Elternpartikels

**Schweregrad:** kritisch · **Datei:** `src/wmcpbe/mcpbe_nucleation.py`

### Problem

`_distribute_liquid_volume()` verteilt ganze Tropfen in einer Schleife und
danach den akkumulierten Rest (`_liquid_remainder`) als *einen* physikalischen
Tropfen. In diesem Restpfad wurde das Kindpartikel mit **nur** dem Rest angelegt:

```python
self._create_nucleated_particle_copy(i, dW_for_one_physical, self._liquid_remainder)
```

`_create_nucleated_particle_copy(src, dW, liquid)` setzt
`liquid_volume[child] = liquid` **absolut**. Hatte das Elternpartikel bereits
Flüssigkeit (`liquid_volume[i] > 0`), so gehen `dW * liquid_volume[i]` verloren:
das Kind übernimmt `dW` physikalische Partikel vom Elternteil, bekommt aber nur
den Resttropfen statt „Elternflüssigkeit + Resttropfen“.

Im regulären Tropfenpfad ist es korrekt gemacht
(`new_liquid = current_liquid + v_droplet`) — nur der Restpfad war falsch.

### Nachweis

Zuordnung der Flüssigkeitsbilanz pro Prozess (Szenario `granulation_1d`,
Originalcode):

```
Δ durch Agglomerationsereignisse : +0.000000e+00      (erhält exakt)
Δ durch kontinuierliche Prozesse : +0.000000e+00      (erhält exakt)
Δ durch Nucleation (Zustand)     : +5.088607e-19
Nucleation meldet als zugegeben  : +5.109551e-19
Tropfenpfad allein (Zustand)     : +9.985029e-19  == erwartet dW*v_drop  (exakt)
------------------------------------------------------------------
Gemeldet gesamt                  :  1.000000e-18
Tatsächlich im Zustand           :  9.942404e-19
Fehlend                          :  5.76e-21  = 0.576 %
```

Agglomeration, Breakage und die kontinuierlichen Prozesse erhalten die
Flüssigkeit exakt. Der gesamte Verlust entsteht im Nucleation-Restpfad.

### Lösung

```python
parent_liquid = float(solver.liquid_volume[i])
self._create_nucleated_particle_copy(
    i, dW_for_one_physical, parent_liquid + self._liquid_remainder
)
```

### Verifikation

| | vorher | nachher |
|---|---|---|
| relative Abweichung `sum(W*liquid)` gegen zugegebene Menge | `5.76e-03` | `2.02e-14` |

Regressionstest: `tests/test_conservation.py` (`test_liquid_conserved_by_*`) und
der Standalone-Report `python -m tests.test_conservation`.

### Auswirkung auf Ergebnisse

Granulationsläufe ändern sich: es ist jetzt exakt die angeforderte
Flüssigkeitsmenge im System (`1.0e-18 m³` statt `9.94e-19 m³`). Der Fehler wuchs
mit der Laufzeit, lange Läufe waren stärker betroffen.

---

<a name="f-04"></a>
## F-04 — Agglomerations-Propensity ist O(n²) pro Ereignis

**Schweregrad:** hoch (Performance) · **Dateien:** `kernels/aggregation/jit_kernels.py`, `mcpbe_agg.py`

### Problem

Nach **jedem** akzeptierten Ereignis wird `_rebuild_all_propensities()`
aufgerufen, das `r_i = Σ_j W_j β(i,j)` mit einer expliziten Doppelschleife
auswertet — O(n²) pro Ereignis.

### Nachweis

`cProfile`, Szenario `agg_shear_1d_large` (n ≈ 1900, 149 Ereignisse), nach
JIT-Warmup:

```
ncalls  tottime  percall  filename:lineno(function)
   145    0.549    0.004  jit_kernels.py:54(rebuild_r_array_shear)     <-- 88 % der Laufzeit
   150    0.012    0.000  mcpbe_agg.py:480(_do_one_agg)
```

### Lösung

Alle vier eingebauten Kernel sind in `(r_i, r_j)` **separabel**, die innere
Summe lässt sich also exakt auf wenige gewichtete Momente der Population
zusammenziehen. Für den Scherkernel:

```
(r_i + r_j)³ = r_i³ + 3r_i²r_j + 3r_i r_j² + r_j³

Σ_j W_j β(i,j) = c·g·( r_i³·S0 + 3r_i²·S1 + 3r_i·S2 + S3 )
mit S0 = ΣW_j, S1 = ΣW_j r_j, S2 = ΣW_j r_j², S3 = ΣW_j r_j³
```

Der Selbstkollisions-Term wird anschließend exakt korrigiert
(`β_ii = 8·c·g·r_i³`). Analog für Brownian
(`(r_i+r_j)²/(r_i r_j) = r_i/r_j + 2 + r_j/r_i`) und Sum
(`β = c(v_i+v_j)`). Der Constant-Kernel war bereits geschlossen.

Das ist **algebraisch exakt**, keine Näherung — Herleitung für alle Kernel in
[`MOMENT_MODE.md`](MOMENT_MODE.md). Umgesetzt sind zwei Modi:

| `agg_propensity_mode` | Verfahren | Kosten | Bitgenau zum Original |
|---|---|---|---|
| `"pairwise"` (Default) | Doppelschleife, über `prange` parallelisiert | O(n²) | **ja** |
| `"moment"` | geschlossene Momentenform | O(n) | nein (~1e-15 relativ) |

Da `fastmath` ausgeschaltet ist, darf LLVM die innere Gleitkomma-Reduktion nicht
umordnen; die parallelisierte Variante liefert daher **bitgleiche** Ergebnisse
wie die ursprüngliche serielle. Das wurde gegen wortgleiche Kopien der
Originalfunktionen geprüft:

```
prange pairwise == serial reference, bit-for-bit: True
```

Zusätzlich wird unterhalb von `PARALLEL_MIN_N = 800` auf serielle Zwillinge
umgeschaltet, weil der Thread-Dispatch bei kleinen Populationen teurer ist als
die Arbeit selbst.

### Messung

Reine Kernel-Zeit, Scherkernel:

| n | pairwise | moment | Faktor | max. rel. Abw. | max. ULP |
|---|---|---|---|---|---|
| 250 | 0.038 ms | 0.0026 ms | 14× | 1.3e-15 | 8 |
| 500 | 0.054 ms | 0.0020 ms | 27× | 2.0e-15 | 13 |
| 1000 | 0.173 ms | 0.0031 ms | 55× | 3.5e-15 | 24 |
| 2000 | 0.667 ms | 0.0066 ms | 100× | 5.9e-15 | 41 |
| 4000 | 2.969 ms | 0.0108 ms | **275×** | 7.2e-15 | 63 |

Die Abweichung von wenigen Dutzend ULP ist genau das, was eine andere
Summationsreihenfolge über n Terme erwarten lässt.

Ensemble-Vergleich über 8 Seeds (`agg_shear_1d`): die Momente M0/M1/M2 stimmen
in beiden Modi überein, `|Δ| / MC-Streuung = 0.000`.

---

<a name="f-05"></a>
## F-05 — `liquid_bridge`-Kernel fiel auf n² Python-Aufrufe zurück

**Schweregrad:** hoch (Performance) · **Dateien:** `kernels/aggregation/jit_kernels.py`, `mcpbe_agg.py`

### Problem

`_rebuild_all_propensities()` hatte JIT-Batchfunktionen nur für `shear`,
`brownian`, `constant` und `sum`. Jeder andere Kernel — insbesondere
`liquid_bridge`, der eigentliche Granulationskernel — landete in einer
Python-Doppelschleife mit `n²` Aufrufen von `kernel_manager.compute_beta()`.

### Nachweis

Szenario mit nur 150 Partikeln und 7 Ereignissen: **533 ms pro Ereignis**
(3.74 s für 7 Ereignisse). Bei 500 Partikeln wären das ~6 s/Ereignis; ein Lauf
mit einigen tausend Ereignissen wäre praktisch nicht durchführbar.

### Lösung

`rebuild_r_array_liquid_bridge()` als JIT-Batchfunktion, die
`LiquidBridgeKernel.compute_beta()` Term für Term nachbildet (Brückenfaktor,
Kapillarfaktor, NaN-Behandlung für Vollkörper). Die Komplexität bleibt O(n²) —
der Kernel ist nicht separabel —, aber sie läuft kompiliert statt interpretiert.
Kernel ohne Batch-Implementierung geben jetzt eine einmalige Warnung aus, statt
still langsam zu sein.

### Messung

**1053× schneller**, bitgenau identisches Ergebnis (0.755 ms/Ereignis statt
700 ms/Ereignis bei n=150).

---

<a name="f-06"></a>
## F-06 — Kontinuierliche Prozesse als Python-Schleife pro Ereignis

**Schweregrad:** hoch (Performance) · **Datei:** `src/wmcpbe/mcpbe_continuous_processes.py`

### Problem

`_apply_internalization()` und `_apply_compression()` iterieren nach **jedem**
MC-Ereignis in reinem Python über alle aktiven Partikel — O(n)
Interpreter-Overhead pro Ereignis. Dazu kam `_find_similar_particle()` in der
Nucleation, das pro Tropfen linear über alle Partikel suchte.

### Nachweis

`cProfile`, Szenario `granulation_1d` (400 Partikel, 299 Ereignisse):

```
ncalls  tottime  filename:lineno(function)
   300    0.770   mcpbe_continuous_processes.py:260(_apply_compression)
  1514    0.388   mcpbe_nucleation.py:1082(_find_similar_particle)
   300    0.369   mcpbe_continuous_processes.py:165(_apply_internalization)
107686    0.194   mcpbe_continuous_processes.py:221(_internalization_fallback)
```

Zusammen ~1.7 s von 2.2 s Gesamtlaufzeit.

### Lösung

* `LiquidInternalizationKernel.compute_array()` und
  `PorosityCompressionKernel.compute_array()` als vektorisierte Zwillinge der
  Skalarmethoden (gleiche Terme, gleiche Reihenfolge).
* Beide Handler-Schleifen durch maskierte numpy-Operationen ersetzt. Die
  `continue`-Verzweigungen des Originals sind 1:1 als Bool-Masken abgebildet,
  inklusive der Feinheit, dass die Porosität auch dann aktualisiert wird, wenn
  die anschließende Volumenbuchhaltung übersprungen wird.
* `_find_similar_particle()` als vektorisierter Maskenscan. Die Vergleiche sind
  bewusst als `~(|Δ| > tol)` und nicht als `|Δ| <= tol` geschrieben — die beiden
  unterscheiden sich bei NaN, und die ursprünglichen `continue`-Wächter nutzten
  die `>`-Form.

### Messung

Granulation **12.9× schneller**, Fingerprint bitgenau identisch.

---

<a name="f-07"></a>
## F-07 — „Links“-Snapshots mischen Vor- und Nach-Ereignis-Zustand

**Schweregrad:** mittel · **Datei:** `src/wmcpbe/mcpbe_base.py` (`solve()`)

### Problem

Die `*_save_left`-Snapshots sollen den Zustand **vor** dem Ereignis festhalten.
Das gilt für `V_save_left` und `W_save_left` (echte Vorher-Kopien), aber
`liquid_volume_save_left`, `porosity_save_left` und `saturation_save_left`
werden aus den **aktuellen** (Nachher-)Arrays gefüllt:

```python
self.V_save_left.append(V_prev_active.copy())                          # vorher
self.liquid_volume_save_left.append(self.liquid_volume[:self.a_tot].copy())  # nachher!
```

### Lösung

Unverändert gelassen und im Code kommentiert, weil eine Korrektur die
gespeicherten Arrays und damit jede aufgezeichnete Auswertung ändern würde.
Die betroffenen Felder werden im aktuellen Post-Processing nicht ausgewertet.
**Zu klären, ob die Left-Snapshots für Flüssigkeit/Porosität überhaupt gebraucht
werden** — wenn nein, ersatzlos streichen.

---

<a name="f-08"></a>
## F-08 — `u_sel` wird für Partnerwahl *und* Akzeptanz verwendet

**Schweregrad:** mittel · **Datei:** `src/wmcpbe/mcpbe_agg.py`

### Problem

In `_pick_partner_kernel()` und `pick_partner_constant_jit()` wird dieselbe
Zufallszahl `u_sel` zweimal benutzt:

```python
j = searchsorted(cumsum_W, u_sel * total_W)   # Partnerwahl
...
if u_sel > size_factor:                        # SIZEEVAL-Akzeptanz
    return -1, 0.0
```

Damit sind Partnerwahl und Größenakzeptanz **korreliert**: ein großes `u_sel`
wählt systematisch Partner am oberen Ende der Gewichtsverteilung *und* führt
gleichzeitig eher zur Ablehnung. Der resultierende Kollisionskernel entspricht
nicht mehr dem beabsichtigten `β·α·f_size`.

### Lösung

Bewusst **unverändert** gelassen — ein zweiter Zufallsziehungsschritt würde den
RNG-Strom verschieben und jede aufgezeichnete Trajektorie ändern. Das Verhalten
ist jetzt an beiden Stellen dokumentiert.

**Empfehlung:** eigenes `u_acc = rng.random()` für den Akzeptanztest einführen.
Das ist statistisch korrekt, ändert aber alle Ergebnisse — Entscheidung liegt
bei Eric.

---

<a name="f-09"></a>
## F-09 — Reine 2D-Agglomeration bricht an einem Breakage-Parameter ab

**Schweregrad:** mittel · **Datei:** `src/wmcpbe/mcpbe_base.py` (`_compute_frag_num`)

### Problem

`_initialize_particles()` ruft unbedingt `_compute_frag_num()` auf, das den
Fragmentzahl-Parameter `BREAKFVAL` auswertet. Der `BaseSolver`-Default ist
`BREAKFVAL = 3`, und für `dim=2` wirft die Funktion:

```
ValueError: BREAKFVAL=3 (product function) not implemented for 2D!
```

Ein reiner Agglomerationslauf in 2D — bei dem nie ein Fragment entsteht —
scheitert also an einem Parameter, den er gar nicht benutzt.

### Lösung

In den Benchmark-Szenarien explizit gesetzt und dokumentiert.
**Empfehlung:** `_compute_frag_num()` nur aufrufen, wenn
`process_type in ("breakage", "mix")`, bzw. bei reiner Agglomeration `frag_num`
auf `NaN` setzen und erst beim ersten Breakage-Ereignis validieren. Nicht
umgesetzt, da es die Initialisierungsreihenfolge berührt.

---

<a name="f-10"></a>
## F-10 — Toter Code-Zweig mit garantiertem `NameError`

**Schweregrad:** mittel · **Datei:** `src/wmcpbe/mcpbe_agg.py` (vorher Zeile 539)

### Problem

`_do_one_agg()` hatte einen „Legacy-JIT-Pfad“, der `nb_pick_partner_weighted()`
aufrief. Diese Funktion war **nirgends importiert**. Der Zweig war nur deshalb
harmlos, weil die Bedingung (`kernel_manager is None`) nie eintreten kann — der
Konstruktor verlangt einen Aggregationskernel. Jede Änderung an dieser Bedingung
hätte einen `NameError` zur Laufzeit ausgelöst.

### Lösung

Zweig entfernt. Ebenso entfernt: der ungenutzte `delta`-Parameter von
`_pick_partner_kernel()` (samt der O(n)-Kopie, die pro Ereignis dafür angelegt
wurde) sowie die toten Funktionen `rebuild_r_agg_*` und
`rebuild_r_array_general` in `jit_kernels.py`.

---

<a name="f-11"></a>
## F-11 — `BREAKFVAL` inkonsistent zwischen Fragmentzahl und CDF-Tabelle

**Schweregrad:** mittel · **Datei:** `src/wmcpbe/mcpbe_break.py`

### Problem

`_compute_frag_num()` bestimmt die *erwartete Fragmentanzahl* aus
`self.BREAKFVAL`. `_build_break_function()` baut die *Fragmentgrößenverteilung*
aber mit einem fest verdrahteten Wert:

```python
def _build_break_function(self, num_points: int = 1000):
    # BREAKFVAL=1 is the default (single breakage mode)
    BREAKFVAL = 1
```

Bei `BREAKFVAL != 1` passen Fragmentanzahl und Fragmentgrößenverteilung also
nicht zusammen.

### Lösung

Nur dokumentiert — welcher der beiden Werte der beabsichtigte ist, ist eine
Modellfrage. Die Benchmark-Szenarien setzen `BREAKFVAL` explizit, damit sie nicht
unbemerkt vom Default abhängen. **Zu klären mit Eric.**

---

<a name="f-12"></a>
## F-12 — Unbedingte Debug-Ausgaben in Bibliothekscode

**Schweregrad:** mittel · **Datei:** `src/wmcpbe/mcpbe_nucleation.py`

### Problem

30 `print()`-Aufrufe schrieben bei jedem Lauf ungefragt nach stdout
(`[NUCLEATION DEBUG]`, `[FINALIZE DEBUG]`, `[LOOP DEBUG]`, …). Das verschmutzt
die Ausgabe jeder Produktionsrechnung und jeder Parameterstudie und kostet in
engen Schleifen messbar Zeit.

### Lösung

Neues Feld `NucleationConfig.debug` (Default `False`) und ein `_log()`-Helfer;
alle Ausgaben laufen darüber. `debug=True` stellt das alte Verhalten her.

---

<a name="f-13"></a>
## F-13 — `_find_similar_particle`: Toleranz an der falschen Größe gemessen

**Schweregrad:** mittel · **Datei:** `src/wmcpbe/mcpbe_nucleation.py`

### Problem

Wenn ein Tropfen auf ein bereits benetztes Partikel trifft, sucht die Nucleation
ein existierendes Partikel mit demselben Nachher-Zustand und schiebt nur das
Gewicht um — statt ein neues Rechenpartikel anzulegen. Die Flüssigkeitstoleranz
ist dabei **relativ zur Gesamtflüssigkeit** des Partikels:

```python
liq_tol = tol * liquid_target          # tol = 1e-5
```

Gebucht wird aber ein **ganzer Tropfen**. Trägt ein Partikel bereits N Tropfen,
so darf der Treffer um `tol · N` Tropfen danebenliegen — der relative Fehler auf
das gebuchte Inkrement wächst also linear mit der Laufzeit.

### Lösung

Neue Option `NucleationConfig.liquid_match_scale`:

* `"total"` (Default, altes Verhalten) — Toleranz relativ zur Gesamtflüssigkeit
* `"droplet"` — Toleranz relativ zum Tropfenvolumen; der Merge ist dann auf
  Tropfenskala massenneutral, dafür sinkt die Trefferquote (mehr Rechenpartikel)

Zusätzlich ist die bisher hart kodierte Toleranz als
`NucleationConfig.similarity_tol` konfigurierbar.

Im untersuchten Szenario wird dieser Pfad nie erreicht (kein Merge-Treffer), der
Beitrag zum gemessenen Flüssigkeitsverlust war also null — der eigentliche
Verlust stammt aus [F-03](#f-03). Die Option ist trotzdem vorhanden, weil der
Defekt in längeren Läufen mit mehr Merge-Treffern relevant wird.

---

<a name="f-14"></a>
## F-14 — Freigegebene Array-Slots behielten Altwerte

**Schweregrad:** niedrig · **Datei:** `src/wmcpbe/mcpbe_base.py`

### Problem

`_remove_particle_column()` löschte beim Freigeben eines Slots nur
`V_flat`, `X`, `W` und die Propensity-Arrays. `liquid_volume`, `porosity` und
`saturation` behielten die Werte des entfernten Partikels. Legt ein späterer
`_append_particle_column()` ein Partikel in diesen Slot und vergisst der Aufrufer,
eine dieser Größen zu setzen, erbt das neue Partikel stillschweigend fremde
Flüssigkeit oder Porosität.

Zusätzlich war die Auffüllung inkonsistent: `_initialize_particles()` legt
`porosity` mit `NaN` an (Sentinel „Vollkörper“), `_ensure_capacity_for()` füllte
neue Slots dagegen mit `0.0` — also „Porosität exakt null“ statt „keine
Porosität“.

### Lösung

Alle per-Partikel-Arrays sind in `_PARTICLE_ARRAY_NAMES` zentral registriert und
werden gemeinsam getauscht, freigegeben und vergrößert. `_RELEASED_SLOT_FILL`
legt den Freigabewert fest — für `porosity` `NaN`, sonst `0.0`. Ein neu
hinzugefügtes Array wird dadurch automatisch überall korrekt mitgeführt; genau
diese Buchhaltung war die Quelle mehrerer Massenerhaltungsfehler.

---

<a name="f-15"></a>
## F-15 — Toter Code

**Schweregrad:** niedrig

### Problem

| Pfad | Umfang | Status |
|---|---|---|
| `src/wmcpbe_backup/` | 502 KB, 15 Module | vollständige Kopie von `wmcpbe` |
| `src/wmcpbe_recon_debug/` | 534 KB, 16 Module | zweite vollständige Kopie |
| `src/wmcpbe/Trials/old/` | 754 KB, 80 Skripte | Ad-hoc-Debugskripte einer Fehlersuche |
| `src/wmcpbe/fenwick.py` | 98 Zeilen | nirgends importiert |
| `src/wmcpbe/mcpbe_jit.py` | 847 Zeilen | nirgends importiert |
| `src/wmcpbe/mcpbe_new_stru.py` | 921 Zeilen | nirgends importiert, ruft `FenwickSampler(..., capacity=…)` auf — Signatur existiert nicht, wäre also ohnehin defekt |
| `src/wmcpbe/mcpbe_old_stru.py` | 808 Zeilen | nirgends importiert |

Drei nahezu identische Kopien desselben Solvers sind eine reale Fehlerquelle:
ein Bugfix in einer Kopie ist in den anderen nicht wirksam, und beim Lesen ist
nicht erkennbar, welche Version tatsächlich läuft.

### Lösung

Alle oben genannten Pfade entfernt (~1.8 MB, ~104 Dateien). Nachweis, dass keine
Importe darauf zeigen, per `grep` über `src/`, `scripts/` und `Trials/`.
Der Verlauf bleibt in Git erhalten; nichts geht unwiederbringlich verloren.

---

<a name="f-16"></a>
## F-16 — `except Exception` verschluckt Kernel-Fehler

**Schweregrad:** niedrig · **Datei:** `src/wmcpbe/mcpbe_agg.py`

### Problem

```python
try:
    beta_ij = self.kernel_manager.compute_beta(...)
except Exception:
    beta_ij = 0.0
```

Ein fehlerhafter Kernel führte damit stillschweigend dazu, dass jede Kollision
abgelehnt wird — die Simulation läuft scheinbar normal weiter und liefert
einfach keine Agglomeration.

### Lösung

Verhalten beibehalten (Ergebnisstabilität), aber der Fehler wird jetzt einmalig
als `RuntimeWarning` mit Ausnahmetyp und Meldung ausgegeben (`_warn_once()`).

---

## Nicht behoben / offene Fragen an Eric

1. **F-02** — Wie sollen `liquid_volume`, `porosity` und `saturation` durch die
   Rekonstruktion (CAM/RS/2PM/QMX) übertragen werden?
2. **F-08** — Soll die Korrelation zwischen Partnerwahl und
   SIZEEVAL-Akzeptanz aufgelöst werden? (Ändert alle Ergebnisse.)
3. **F-11** — Welcher `BREAKFVAL` ist maßgeblich: der für die Fragmentanzahl
   oder der hart kodierte für die CDF-Tabelle?
4. **F-13** — Soll `liquid_match_scale="droplet"` zum Default werden?
5. **F-04** — Soll `agg_propensity_mode="moment"` zum Default werden? Das ist
   der verbleibende große Performance-Hebel (siehe `PERFORMANCE.md`).
