# Parallelisierung von Monte-Carlo-Wiederholungen

**Stand:** 18.08.2026
**Referenz-Implementierung:** `upstream/dev_monorepo` = Commit `02dfac6` (10.08.2026),
identisch mit github.com/pdhs-group/PSD_opt/tree/dev_monorepo
**Neuer Code:** `mcpbe/src/wmcpbe/Trials/framework/` (rein additiv, aendert keine bestehende Datei)

Ein einzelner Monte-Carlo-Lauf ist eine Stichprobe, kein Ergebnis. Dieses
Dokument haelt fest, wie die Wiederholungslogik des Betreuers aufgebaut ist,
was beim Uebertragen auf `dev_eric` nicht funktioniert hat und wie das geloest
wurde.

---

## 1. Die Vorlage: drei Ebenen

Die Loesung im Branch `dev_monorepo` ist kein einzelnes Feature, sondern ein
Sandwich aus drei Schichten. Jede kennt nur die darunter.

| Ebene | Datei | Aufgabe |
|---|---|---|
| 3 — Kampagne | `scripts/pbe_validation/new/WMCPBE_sensitivity_analysis.py` | Sobol-Sampling ueber Parameter, verteilt Konfigurationen auf Prozesse, **wiederaufsetzbar** |
| 2 — Rezept | `scripts/pbe_validation/new/validation.py` | Dataclasses beschreiben einen Lauf als Daten; `ValidationRunner` baut daraus den Solver |
| 1 — Statistik | `mcpbe/src/wmcpbe/mcpbe_base.py::solve_repeats` | N Wiederholungen desselben Setups mit verschiedenen Seeds |

Analogie: Ebene 1 ist „wuerfle 40 mal", Ebene 2 ist „hier ist das Rezept, nach
dem gewuerfelt wird", Ebene 3 ist „koche 512 Rezepte durch und finde heraus,
welche Zutat das Ergebnis bestimmt".

### Die vier Ideen, die uebernommen wurden

**a) Unabhaengige Seeds.** Nicht `seed = base + i`, sondern
`np.random.SeedSequence(base).spawn(N)`. Benachbarte Ganzzahl-Seeds koennen
korrelierte Zufallsstroeme erzeugen; `SeedSequence` garantiert Unabhaengigkeit.
Das ist die Voraussetzung dafuer, dass ein Standardfehler ueberhaupt bedeutet,
was er zu bedeuten vorgibt.

**b) Reduktion im Worker.** Ein fertiger Solver traegt die komplette
Partikel-Historie (bei 2000 Partikeln und 501 Zeitpunkten dreistellige
Megabyte). Alles, was zwischen Prozessen wandert, muss serialisiert werden —
schickt man den Solver zurueck, frisst die Serialisierung den Zeitgewinn wieder
auf. Der Betreuer wertet die PSD deshalb schon im Worker auf einem festen
Gitter aus und schickt nur ein `(T, M)`-Array zurueck.

**c) Backpressure.** Es liegen nie mehr als `workers` Auftraege gleichzeitig in
der Warteschlange (`wait(..., FIRST_COMPLETED)`, dann nachruecken). Bei 512
Laeufen und 8 Kernen bleibt der Speicherbedarf konstant statt mit N zu wachsen.

**d) Wiederaufsetzbarkeit.** Jeder fertige Lauf schreibt sofort seine eigene
`.npz` — atomar (`tmp_path.replace(npz_path)`), also nie eine halbe Datei. Ein
SQLite-Index merkt sich fertige `task_id`s; beim Neustart wird nur der Rest
gerechnet. Ein Absturz in Stunde 39 kostet damit einen Lauf, nicht die Kampagne.

---

## 2. Warum die Vorlage nicht 1:1 uebertragbar war

`dev_eric` hat bereits eine `solve_repeats` — eine **aeltere Version**,
mitgekommen beim Fork vor drei Monaten und nie benutzt, waehrend der Solver um
Kernel-Framework, Nucleation, Merger und Continuous Processes gewachsen ist.
Drei Befunde, alle reproduziert (Messprotokolle in Abschnitt 5):

### B-P1 — Der Parallelpfad stuerzt sofort ab

```
workers=2: RuntimeError: [parallel] worker 0 failed:
           No aggregation kernel specified. Aggregation kernel is required.
```

Der Worker ruft `cls(dim=dim, init=False, load_attr=False)` auf. Bei `dev_eric`
laeuft dabei der `KernelManager` los und verlangt einen Aggregationskernel — der
aber erst im naechsten Schritt (`__dict__.update`) ankaeme. Der Konstruktor
stirbt, bevor der Zustand da ist.

### B-P2 — Drei Objekte zeigen nach dem Kopieren auf einen verwaisten Solver

`solve_repeats` macht `copy.deepcopy(self.__dict__)`. Drei Unterobjekte halten
eine Rueckreferenz `self.solver`:

| Objekt | Ort |
|---|---|
| `_particle_merger` | `particle_merger.py:130` |
| `continuous_processes` | `mcpbe_continuous_processes.py:95` |
| `nucleation` | `mcpbe_nucleation.py:373` |

Gemessen:

```
vor  deepcopy: .solver is s -> True  True  True
nach deepcopy: .solver is s -> False False False
```

Der Merger im Worker wuerde seine Updates in eine **Geisterkopie** des Solvers
schreiben, die niemand ausliest. Kein Absturz — stiller Datenmuell. Verschaerft
durch den Guard in `mcpbe_base.py:1320`
(`if getattr(self, "_particle_merger", None) is not None: return`): der
veraltete Merger wird nicht einmal neu gebaut.

### B-P3 — Der Zufallsstrom der Nucleation friert ein

`NucleationHandler.__init__` macht `self._rng = solver._rng`
(`mcpbe_nucleation.py:404`). Nach dem `deepcopy` zeigt `nucleation._rng` auf
eine Kopie des alten Stroms. Der Worker setzt zwar `obj._rng` neu — die
Nucleation merkt davon nichts. Alle N Wiederholungen wuerden dieselben Tropfen
an dieselben Partikel haengen.

Das ist die gefaehrlichste der drei: sie sieht auf keinem Plot verdaechtig aus,
sie macht nur die Varianz kuenstlich zu klein.

### Nebenbefund

In `solve_repeats` liegt ab `mcpbe_base.py:2934` ein zweiter, kompletter
Parallelblock, der noch die alte 3-Tupel-Rueckgabekonvention erwartet. Er ist
unerreichbar (beim Betreuer ebenso). Beim Aufraeumen ersatzlos streichbar.

### Was sauber war

Die Seed-Spawn-Logik funktioniert, und der Solver-Zustand ist picklebar —
auch mit Logger, Numba-Samplern und Kernel-Manager. Der Transportweg war also
frei; nur die Rekonstruktion am Zielort stimmte nicht.

---

## 3. Die Entscheidung: Rezept statt Zustand

Der `deepcopy`-Ansatz passt zum Solver des Betreuers, weil dieser vollstaendig
durch flache Attribute konfiguriert wird (`solver.G = 1000`, `solver.a0 = ...`).
`dev_eric` nicht: hier gibt es Fabrikmethoden (`create_nucleation_handler`),
Kernel-Objekte und Nach-Init-Mutationen der Arrays
(`solver.porosity[:a_tot] = 0.8`). Ein solcher Solver laesst sich nicht als
Zustandsdump verschicken.

Uebertragen wird deshalb ein **Rezept**: ein flaches `dict` aus Zahlen und
Strings plus ein Seed. Der Worker baut den Solver selbst
(`framework/builder.py::build_reference_solver`). Damit:

- entstehen Merger, Nucleation und Continuous Processes **im Worker** und zeigen
  automatisch auf den richtigen Solver — B-P2 und B-P3 loesen sich auf, ohne
  dass eine Zeile an diesen Modulen geaendert werden muss;
- bekommt der Konstruktor die Kernel-Argumente, die er verlangt — B-P1 geloest;
- ist das Payload winzig (ein `dict`) statt ein komplettes Partikel-Array.

Das Muster existierte in `dev_eric` bereits: `Trials/stufe3_ensemble.py:47`
baut jeden Lauf mit einer `build(seed, t_end)`-Funktion neu auf.

Der Preis: die Initialisierung laeuft N-mal statt einmal. Bei ~1 s Init gegen
~44 s Rechenzeit je Lauf ist das irrelevant.

---

## 4. Aufbau des neuen Codes

```
mcpbe/src/wmcpbe/framework/          <- wiederverwendbares Paket
├── __init__.py                 Pfad-Bootstrap + Thread-Begrenzung
├── builder.py                  Rezept -> fertiger Solver
├── metrics.py                  fertiger Solver -> kompakte Kennzahlen
├── runner.py                   N Wiederholungen, seriell oder parallel
├── statistics.py               Mittelwert / Streuung / Konvergenzpruefung
└── run_reference_repeats.py    ausfuehrbares Skript: eine Config, N Seeds

mcpbe/src/wmcpbe/Trials/             <- ausfuehrbare Studien
└── test_sensitivity_repeats.py  Sensitivitaetsstudie mit Wiederholungen
```

Beides ist **rein additiv**: keine bestehende Datei wurde geaendert.
Rueckgaengig machen = die beiden Pfade loeschen.

### Aufruf

Alles aus `mcpbe/src/wmcpbe`:

```
python -m framework.run_reference_repeats --repeats 10 --workers 6
python -m Trials.test_sensitivity_repeats --dry-run
python -m Trials.test_sensitivity_repeats --workers 8
```

Kurzer Rauchtest mit Aequivalenzpruefung (`t_total = 4 s`, 4 Wiederholungen):

```
python -m framework.run_reference_repeats --smoke
```

**Wichtig zum Importpfad.** Das Paket liegt in `wmcpbe/`, laesst sich aber
nicht ueber `python -m wmcpbe.framework....` starten: dabei wird zuerst
`wmcpbe/__init__.py` ausgefuehrt, das `pbe_core` nachzieht -- und der eigene
Pfad-Bootstrap kommt zu spaet. Deshalb wird es als **oberste Ebene** aus dem
Verzeichnis `wmcpbe/` heraus importiert (`import framework`). Skripte in
`Trials/`, die `from wmcpbe.framework import ...` schreiben, muessen den
`sys.path` vorher selbst setzen -- `test_sensitivity_repeats.py` macht das in
den ersten Zeilen und dient als Vorlage.

### Ausgabe

Nach `Trials/Results/repeats/`:

| Datei | Inhalt |
|---|---|
| `timeseries_per_seed.csv` | Langformat, eine Zeile je (Seed, Zeitpunkt) — die Datei fuer die Auswertung |
| `timeseries_summary.csv` | Breitformat je Zeitpunkt mit `_mean` / `_std` / `_sem` — direkt plotbar |
| `scalars_per_seed.csv` | eine Zahl je Lauf (Endwerte, Bilanzfehler, Laufzeit) |
| `scalars_summary.csv` | deren Statistik |
| `repeats/*.npz` | Rohdaten je Lauf, Grundlage der Wiederaufnahme |
| `params.json`, `report.txt` | verwendete Konfiguration und Pruefbericht |

### Zwei Windows-Fallstricke, die eingebaut sind

- **`if __name__ == "__main__":` ist Pflicht.** Windows startet Worker mit
  `spawn`, jeder importiert das Hauptmodul neu. Ohne Guard startet die Kampagne
  rekursiv sich selbst.
- **Thread-Oversubscription.** Numba (`fenwick_new.py`) und BLAS starten je
  eigene Threads; 6 Prozesse mit je 8 Threads auf 8 Kernen sind langsamer als
  ein serieller Lauf. `pin_threads()` setzt `OMP_NUM_THREADS` und Verwandte auf
  1, im Elternprozess (die Worker erben die Umgebung).

---

## 5. Pruefungen und was sie bedeuten

Der Bericht laeuft nach jedem Durchlauf automatisch mit.

**Seeds wirken.** Sind alle Wiederholungen identisch, wirkt der Seed nicht —
typisch fuer B-P3. Dann waere jede Statistik wertlos, obwohl alles fehlerfrei
durchlaeuft.

**Feststoff- und Fluessigkeitsbilanz.** Masse darf nicht vom Zufall abhaengen.
Streut ein Bilanzfehler ueber die Seeds, ist das ein starkes Signal fuer einen
zustandsabhaengigen Fehler — genau die Sorte, die in einem Einzellauf wie
Rauschen aussieht (vgl. `Entscheidungsgrundlage_Seed_Wiederholungen.md`).

**Konvergenz.** Sind die Laeufe unabhaengig, faellt der Standardfehler wie
`1/sqrt(N)`, im doppelt logarithmischen Bild also mit Steigung -0.5. Kommt
deutlich weniger heraus, sind die Laeufe korreliert und alle Fehlerbalken waeren
zu klein. Bei zehn Wiederholungen ist die Steigung selbst verrauscht — alles
zwischen etwa -0.8 und -0.2 ist unauffaellig, ein Wert nahe 0 dagegen nicht.

**Aequivalenz seriell/parallel** (`--check`). Bei gleichem Seed muss ein Lauf im
Worker-Prozess bitgenau dasselbe ergeben wie im Hauptprozess. Weicht etwas ab,
ist Zustand ueber die Prozessgrenze gewandert, der dort nicht hingehoert.
Ergebnis beim Rauchtest: **bitgenau identisch**.

**Treue zum Originalskript.** `builder.py` haelt eine Kopie der Konfiguration
aus `test_powerlaw_rumpf_full.py`. Beide wurden mit Seed 205 und `t_total=100`
gegeneinander laufen gelassen:

| | `test_powerlaw_rumpf_full.py` | `framework/builder.py` |
|---|---|---|
| n_comp am Ende | 2999 | 2999 |
| real agg | 7740 | 7740 |
| real break | 10760 | 10760 |
| Rechenzeit | 46,6 s | 44,0 s |

Ereignis fuer Ereignis identisch. Wird `TestConfig` geaendert, muss dieser
Vergleich wiederholt werden (siehe Abschnitt 7, Doppelung der Config).

### Ergebnis des ersten Zehn-Seed-Laufs (18.08.2026)

```
[OK] Seeds wirken:        10/10 Laeufe unterscheidbar
[OK] Feststoffbilanz:     groesster |Fehler| = 5.4e-16  (Streuung ueber Seeds 8.5e-17)
[OK] Fluessigkeitsbilanz: groesster |Fehler| = 5.1e-13
[OK] Dosierung:           groesster |Fehler| = 5.1e-13
[OK] Konvergenz n_phys:   Steigung log(sem) ueber log(N) = -0.51 (erwartet -0.50)
```

Laufzeit 225 s fuer zehn Laeufe auf 6 Prozessen (seriell waeren es ~1114 s
gewesen, Faktor 4,9). Relativer Standardfehler der Endwerte: Durchmesser
0,00 %, Rechenpartikel 0,31 %, Brueche 1,17 %, Agglomerationen 1,35 %,
Saettigung 3,29 %, Porositaet 2,58 %.

Die Porositaet ist damit die am schlechtesten bestimmte Groesse. Soll sie auf
1 % genau werden, braucht es wegen `1/sqrt(N)` etwa 67 statt 10 Wiederholungen.

Nicht streuen darf die Tropfenzahl (716197,24 +/- 0,00): die Dosierung ist
deterministisch (`Q*t / v_Tropfen`). Dass alles andere streut, zeigt, dass das
keine fehlende Seed-Wirkung ist.

### Zwei vorbestehende Eigenheiten des Originalskripts

Beim Abgleich aufgefallen, **nicht** durch diese Arbeit verursacht und hier
nicht geaendert:

- `test_powerlaw_rumpf_full.py` legt nur `.../src/wmcpbe` auf den Importpfad,
  nicht `.../src` und nicht `pbe-core/src`. Der direkte Aufruf scheitert
  deshalb mit `ModuleNotFoundError`; es laeuft nur mit passend gesetztem
  `PYTHONPATH` (in Spyder ueber die Projektpfade). Das neue Framework bringt
  seinen eigenen Bootstrap mit und laeuft in beiden Varianten.
- Die Ergebnisausgabe enthaelt ein `Δ` (Delta). Wird stdout auf Windows in
  eine Datei umgeleitet, bricht das Skript dort mit `UnicodeEncodeError`
  (cp1252) ab -- **nach** der vollstaendigen Rechnung, es ist rein kosmetisch.

---

## 6. Zwei Fallen beim Auswerten, die beim Bauen aufgefallen sind

### Snapshot-Zeit ist nicht Endzeit

Die Solve-Schleife laeuft nach dem letzten Speicherzeitpunkt noch bis zum
naechsten Ereignis weiter. Der Snapshot bei `t_vec[-1]` liegt also etwas
**frueher** als der Live-Endzustand. Mischt man beide — Snapshot-Fluessigkeit
gegen den Live-Zaehler `liquid_volume_added_total` — sieht eine perfekte Bilanz
wie ein Verlust von einigen Prozent aus:

```
Snapshot bei t=4.0 : 1.1807e-09 m^3
Live am Laufende   : 1.2310e-09 m^3   (= exakt der zugefuehrten Menge)
```

Die Bilanz-Kennzahlen in `metrics.py` werden deshalb **komplett aus dem
Live-Endzustand** gerechnet. Die Zeitreihen bleiben Snapshots — dort ist das
richtig. Nach der Korrektur: Fluessigkeitsbilanz 3e-13, Feststoffbilanz 4e-16.

### Ungewichtete Mittelwerte

`Trials/test_sensitivity.py` mittelt Porositaet und Saettigung ungewichtet ueber
die Rechenpartikel. Physikalisch richtig ist das Mittel mit dem DSMC-Gewicht
`W`: ein Rechenpartikel mit W=600 steht fuer 600 reale Partikel und darf nicht
so zaehlen wie eines mit W=1. `metrics.py` schreibt beide Varianten — `*_mean`
(ungewichtet, fuer die Vergleichbarkeit mit vorhandenen CSVs) und `*_wmean`
(gewichtet). **Fuer physikalische Aussagen die `*_wmean`-Spalten verwenden.**

---

## 7. Die Sensitivitaetsstudie (`Trials/test_sensitivity_repeats.py`)

Sechs Suiten, 34 Zellen, je 5 Seeds. Abweichend von der Referenzconfig:
`t_total = 300 s` (die Kompression hat bei `rate = 0.02 1/s` eine Zeitkonstante
von 50 s und ist bei 100 s noch nicht ausgelaufen) und `control_volume = 0.5`
(V_c wirkt ueber `n_phys = sum(W)/V_c` auf die Kollisionsrate; die
Gesamt-Ereignisrate skaliert etwa mit `1/V_c`).

| Suite | Parameter | Zellen |
|---|---|---|
| A | `k_int` | 1e10 … 1e14, 5 Werte |
| B | `compression_rate` | 0 / 0.005 / 0.02 / 0.05 / 0.1 |
| C | Merger-Toleranz | 1e-8 … 1e-2, plus 1 Lauf ohne Hash-Index |
| D | Tropfendurchmesser | 15 / 20 / 25 / 32 / 40 µm |
| E | `batch_size` | 5 / 10 / 20 / 50 / 100 |
| F | Fluessigkeitsmenge | 4 Mengen x 2 Modi |

Suite F stellt dieselbe Gesamtmenge auf zwei Wegen ein -- `flow` (Dauer fest,
Flussrate variabel) und `duration` (Flussrate fest, Dauer variabel) -- damit
sichtbar wird, ob nur die Menge zaehlt oder auch die Zugabegeschwindigkeit.

### Drei Eigenschaften, die Rechenzeit sparen bzw. Fehler vermeiden

**Dubletten werden nur einmal gerechnet.** Alle Zellen teilen sich einen
`.npz`-Pool, und der Dateiname enthaelt den Fingerabdruck der Parameter. Die
Referenzzelle kommt in jeder Suite vor und Suite F enthaelt sie doppelt --
gerechnet wird sie einmal. Von 34 Zellen bleiben 28 eigenstaendige, also 140
statt 170 Laeufe.

**Gleiche Seeds ueber alle Zellen.** Ein Unterschied zwischen zwei Zellen
stammt damit vom Parameter, nicht vom Seed-Rauschen (gepaarter Vergleich).

**Aufwaermlauf vor der Messung.** Die Sampler sind mit `@njit(cache=True)`
uebersetzt; der allererste Lauf auf einer Maschine kompiliert sie. Ohne
Aufwaermen traegt die zuerst gerechnete Zelle die komplette Kompilierzeit.
Gemessen im Wiring-Test: `C_tol_1e-08` kostete **10,7 s** gegen 2,3 s fuer alle
folgenden Zellen -- Faktor 4,6, rein durch die Uebersetzung. Fuer Suite C, in
der die Rechenzeit die eigentliche Messgroesse ist, waere das ein handfester
Fehler gewesen. Mit Aufwaermlauf faellt die Zelle auf 2,4 s und liegt damit im
Feld.

Zusaetzlich unterscheidet `metrics.py` jetzt zwei Zeiten: `wall_time_s`
(verstrichene Uhrzeit, enthaelt bei zu vielen Workern Wartezeit) und
`cpu_time_s` (tatsaechlich verbrauchte Rechenzeit, unter Konkurrenz
belastbar). Fuer den saubersten Vergleich Suite C einzeln mit `--workers 1`
laufen lassen.

### Der Fingerabdruck ist streng -- absichtlich

Er deckt **alle** Parameter ab. Schon das Hinzufuegen eines neuen Schluessels
zu `REFERENCE_PARAMS` macht vorhandene Ergebnisse formal ungueltig, auch wenn
der neue Wert genau dem bisherigen Verhalten entspricht. Genau das ist beim
Nachruesten der Merger-Parameter passiert: die zehn Referenzlaeufe wurden nicht
mehr wiederverwendet.

Das ist die richtige Reaktion -- lieber einmal zu viel rechnen als zwei
Konfigurationen stillschweigend vermischen. Damit es nicht wie ein Cache-Fehler
aussieht, meldet `runner.py` es jetzt ausdruecklich:

```
[repeats] HINWEIS: im Pool liegen Ergebnisse mit anderem Parameter-Fingerabdruck
          (38d0ad6cb3 (10 Dateien)), aktuell ist '638b817472'. Es wird deshalb
          komplett neu gerechnet. Ursache ist eine geaenderte oder neu
          hinzugefuegte Parameterangabe -- nicht ein Fehler im Cache.
```

**Konsequenz fuer eine laufende Kampagne:** waehrend sie laeuft (oder zwischen
zwei Teilstarts) nichts an `REFERENCE_PARAMS` aendern, sonst faengt alles von
vorn an.

---

## 8. Was bewusst offen bleibt

- **Sobol / SALib** (Ebene 3 des Betreuers) ist zurueckgestellt. Er nutzt es,
  weil er wenige, billige Parameter hat. Hier kostet ein Punkt ~44 s und es gibt
  ueber 20 Parameter — Sobol braeuchte Tausende Laeufe. Erst die
  Wiederholungsstatistik, dann die Frage, ob sich Sobol lohnt.
- **`test_sensitivity.py`** ist noch nicht auf das Framework umgestellt. Der
  Umbau der vier Suiten (A–D) auf N Wiederholungen je Zelle ist der naechste
  Schritt.
- **Die alte `solve_repeats`** im Solver bleibt unangetastet und ungenutzt. Sie
  kann geloescht werden, sobald das neue Framework produktiv ist.
- **`test_powerlaw_rumpf_full.py`** wurde nicht refaktoriert. `builder.py`
  enthaelt eine Kopie der Konfiguration, keine gemeinsame Quelle. Das ist
  Absicht (nichts Bestehendes anfassen), aber eine Doppelung: aendert sich
  `TestConfig`, muss `REFERENCE_PARAMS` nachgezogen werden.
