# Nucleation-Partikel waren für Agglomeration und Bruch unsichtbar

**Stand 17.08.2026.** Von der Nucleation angelegte Partikel bekamen die Propensity 0
und konnten vom Sampler nie gezogen werden. Bei `INITIAL_POROSITY = 0` entstand daraus
eine Selbstblockade, die den Lauf dauerhaft im Anfangszustand einfror. Gemessen belegt
und behoben.

Ausgangspunkt war die Frage, warum 34-µm-Partikel mit poro = 0 nicht agglomerieren,
obwohl die Flüssigkeit außen anliegt.

---

## 1. Was der Verdacht war und warum er falsch war

Naheliegend wäre das Stokes-Kriterium. Statische Auswertung des echten Kernels über 72
Kombinationen aus Dichte, Binderviskosität, Stoßgeschwindigkeit und Flüssigkeitsanteil:
53 davon akzeptieren. Für 34 µm gilt

```
St      ≈ 2.7e-4          (m_harm · U / 3π η R_harm²)
St_krit ≈ 0.25 … 22       ((1 + 1/e) · ln(h/h_a))
```

St liegt drei bis fünf Größenordnungen unter der Schwelle. Nötig ist lediglich ein
externer Flüssigkeitsanteil von `f > 3·h_a/r = 8.8e-5` — ein einziger Tropfen genügt
weit.

Auch die Kollisionshäufigkeit reicht: β = 7.86e-10 m³/s bei n = 8e5 m⁻³ ergibt 251
Kollisionen/s, rund 45.000 über 180 s.

**Der Widerspruch:** im Diagnoselauf wurde trotzdem jede Kollision abgelehnt — mit dem
Grund „beide Partner trocken", bei 99 % benetztem Gewicht.

---

## 2. Die Ursache

`mcpbe_base.py::_append_particle_column` hängt ein neues Partikel so in den Sampler:

```python
if self._agg_sampler is not None and self._r_agg is not None:
    self._agg_sampler.append(float(self._r_agg[idx]))
```

Übernommen wird der Wert, der zufällig schon in `_r_agg[idx]` steht — bei einem frischen
Slot 0. Die Propensity des neuen Partikels wird an dieser Stelle **nicht berechnet**.

Nachgerechnet wird sie ausschließlich in `_rebuild_all_propensities()`, und das läuft an
vier Stellen:

| Auslöser | Ort |
|---|---|
| Initialisierung | `mcpbe_base.py::_initialize_samplers` |
| **erfolgreiches** Agglomerations-Ereignis | `mcpbe_agg.py::_refresh_samplers_after_agg` |
| Breakage-Ereignis | `mcpbe_break.py` |
| Vc-Verdopplung | `mcpbe_base.py::_maybe_double_control_volume` |

Die Nucleation ruft keine davon — obwohl sie Partikel anlegt und Gewichte in großem
Umfang verschiebt.

### Die Selbstblockade

Bei `INITIAL_POROSITY = 0` schließt sich der Kreis:

```
poro = 0  ->  "porenfrei bricht nicht"  ->  kein Breakage-Ereignis  ->  kein Refresh
kein Refresh  ->  benetzte Partikel haben Propensity 0  ->  unziehbar
           ->  jedes gezogene Paar besteht aus den alten, trockenen Partikeln
           ->  Stokes lehnt ab ("beide trocken")  ->  kein Agg-Ereignis
           ->  kein Refresh
```

Der Zustand friert bei t = 0 ein und kann sich nicht selbst befreien. Es braucht
genau ein akzeptiertes Ereignis, um die Kette zu durchbrechen — und das Stokes-Kriterium
verhindert das erste.

### Warum die Verwechslung naheliegt

Die Nucleation **hat** eine sorgfältige Sampler-Pflege: `_ensure_samplers`,
`_record_weight_change`, `_apply_pending_weight_updates`, inkrementelle O(log n)-Updates.
Dieser Sampler (`_weight_sampler`) dient aber ausschließlich der **Zielauswahl für
Tropfen**. Über `_r_agg` und `_break_rate` sagt er nichts. Zwei Sampler mit ähnlichem
Namen und verschiedener Aufgabe.

### Historie: bewusst ausgelassen, aber nie kompensiert

Der Refresh ist **nicht verloren gegangen, es gab ihn nie**:

```
git log -S "_refresh_samplers_after_agg" -- mcpbe_base.py mcpbe_nucleation.py  -> 0 Commits
git log -S "_rebuild_all_propensities"   -- mcpbe_nucleation.py                -> 0 Commits
```

`wmcpbe_granulation` ruft die Nucleation gar nicht aus der Solve-Schleife auf, dort
konnte das Problem nicht entstehen.

Es war auch kein Versehen, sondern eine dokumentierte Entscheidung. `Fundamentals.md`
begründet `_manual_agglomerate_particles` genau damit: eine eigene Kopie der
Agglomerationslogik, „weil hier ohne Propensity-Rebuild gearbeitet werden muss (würde
bei jedem Tropfen O(n²) kosten)".

**Diese Begründung ist richtig — der Fehler liegt eine Ebene höher.** Innerhalb der
Tropfenschleife darf nicht neu gebaut werden. Nur hat es danach auch niemand auf
Ereignisebene nachgeholt. Aus „nicht pro Tropfen" wurde stillschweigend „nie". Die
Korrektur setzt genau dort an: einmal pro MC-Ereignis, nicht pro Tropfen — die
Performance-Überlegung bleibt gewahrt.

---

## 3. Messung

Identischer Seed, 500 Partikel, 34 µm, poro = 0, 2 s. **A** = Code wie er war,
**B** = zusätzlich `_refresh_samplers_after_agg()` nach jedem Nucleations-Schritt.

| t [s] | W benetzt | Propensity benetzt (A) | Propensity benetzt (B) |
|---|---|---|---|
| 0.00 | 0 | 0.000e+00 | 0.000e+00 |
| 0.13 | 47.002 | **0.000e+00** | 7.39 |
| 0.32 | 98.924 | **0.000e+00** | 15.56 |
| 0.70 | 154.987 | **0.000e+00** | 24.37 |

| | real_agg | Stokes akzeptiert | „beide trocken" |
|---|---|---|---|
| A | **0** | 0 von 26 | 26 |
| B | **27** | 27 von 32 | 5 |

Zusätzlich blieb in A die Propensity der **alten** Partikel bei 31.44 stehen, während
ihr Gewicht von 200.000 auf 45.000 fiel: rund Faktor 4 zu hoch.

### Parameter-Sweep, 14 Fälle

Vor dem Fix ist `real_agg` in **13 von 14 Fällen exakt null** — kein Parameter hilft:

| Fall | real_agg (A) | Kollisionen | real_agg (B) |
|---|---|---|---|
| Referenz | 0 | 26 | 27 |
| poro = 0.7 | 0 | 19 | 6 |
| corr_beta 4 → 100 | 0 | 646 | 688 |
| G 5000 → 20000 | 0 | 104 | 106 |
| Vc 1 → 1e-3 | **0** | **21.812** | 3.891 |
| Gewicht 400 → 4000 | 0 | 3.147 | 977 |
| Viskosität 0.7 → 0.07 | 0 | 26 | 27 |
| U_coll 0.01 → 0.5 | 0 | 26 | 27 |
| h_a 500 pm → 50 nm | 0 | 26 | 27 |
| Flüssigkeit ×10 / ×0.1 | 0 / 0 | 3 / 32 | 9 / 8 |
| Tropfen 15 → 30 µm | 0 | 32 | 13 |
| rho 2500 → 500 | 0 | 26 | 27 |
| **ohne Stokes-Kriterium** | **32** | – | 32 |

Drei Ablesungen:

1. **21.812 Kollisionen, null Agglomerationen** — mit Gewalt geht es nicht.
2. Die Stokes-Parameter (η, U_coll, h_a, ρ) liefern **bit-identische** Zahlen. In allen
   28 Läufen wurde **kein einziges Mal** über den St-Vergleich abgelehnt (Zähler `St>=`
   durchgehend 0); 100 % der Ablehnungen lauten „beide trocken". Sie wirken auf eine
   Entscheidung, die nie fällt.
3. **Ohne Stokes-Kriterium agglomeriert auch A.** Damit ist die Kette bewiesen: das erste
   akzeptierte Ereignis löst `_refresh_samplers_after_agg` aus, ab da heilt sich das
   System selbst.

---

## 4. Die Korrektur

`mcpbe_nucleation.py` bekommt ein Flag `_population_changed`, gesetzt in
`_record_weight_change` — dem einen Punkt, durch den jede Gewichtsänderung des Handlers
läuft (Tropfenübertrag, Elternabbau, Kind der Zwangs-Agglomeration) — plus
`consume_population_changed()`.

`mcpbe_base.py::solve` frischt direkt nach `nucleation.step(...)` auf, nach
`process_type` gestaffelt wie beim Vc-Doubling. **Breakage ist mitgefixt**: `_break_rate`
war für neue Partikel genauso 0, sie konnten also auch nie brechen.

Der Rebuild ist O(n) und hängt deshalb an einer tatsächlichen Änderung, nicht am
Ereignis: `nucleation.step` kehrt meist früh zurück, und nach Ende der Flüssigkeitszugabe
passiert dort gar nichts mehr.

### Verifikation

- Solver **ohne** Patch: `real_agg` 0 → **27**, Propensity wächst mit dem Gewicht mit.
- Sweep gegen den gefixten Code: A und B in **13 von 14 Fällen zahlengleich** — der
  zusätzliche Refresh in B ist wirkungslos geworden, die Änderung also idempotent.
- Im dichten Fall ist der Fix **20 % schneller** als der Monkeypatch (0.472 s statt
  0.392 s Simulationszeit in 45 s), weil das Flag die überflüssigen Rebuilds unterdrückt.
- `mcpbe/tests`: **38 passed**, keine Regression.

---

## 5. Tragweite

**Betroffen ist jeder Lauf mit Nucleation, nicht nur die mit Agglomeration.** Die
MC-Uhr bezieht ihre Zeitschritte aus den Propensity-Summen; waren die eingefroren, war
die Zuordnung von Ereignissen zur Simulationszeit verschoben (gemessen Faktor ~4).
Zusammen mit der wirkungslosen Kompression
([`Kompression_und_Parametervalidierung.md`](Kompression_und_Parametervalidierung.md))
müssen die Sensitivity-Suiten in zwei Punkten neu gerechnet werden.

**Der Mechanismus ist frei, die Menge bleibt klein.** Bei den Referenzparametern sind es
27 Ereignisse in 2 s bei 200.000 physikalischen Partikeln — x50 bleibt bei 34 µm. Für
sichtbares Wachstum muss die Agglomerationsrate in die Größenordnung „Partikelzahl pro
Prozesszeit" kommen. Stärkster Hebel ist die Teilchenzahldichte: `CONTROL_VOLUME = 1` m³
ergibt einen Feststoff-Volumenanteil von 1.6e-8, reale Mischer liegen bei 1e-3 … 0.5.
`Vc = 1e-3` liefert 10.589 agg/s statt 13.

---

## 6. Dieselbe Fehlerklasse eine Ebene tiefer: der Nucleations-Sampler

Der Handler hält einen eigenen Fenwick-Sampler (`_weight_sampler`) für die
**Zielauswahl der Tropfen** — gewichtet nach W, damit alle *physikalischen* Partikel
gleich wahrscheinlich getroffen werden. Aktualisiert wurde er nur **einmal pro
Nucleations-Schritt**, vor der Verteilschleife:

```python
_ensure_samplers()                      # EINMAL
while v_distributed < v_target:
    _distribute_one_droplet_with_dW()   # N Tropfen, Gewichte aendern sich laufend
```

`_record_weight_change` legte die Änderungen nur in `_pending_weight_updates` ab;
angewandt wurden sie erst beim **nächsten** Schritt. Ab dem zweiten Tropfen eines
Schrittes zog der Sampler also mit veralteten Gewichten, und die innerhalb desselben
Schrittes neu entstandenen Partikel waren für die restlichen Tropfen gar nicht
auswählbar.

Das verzerrt genau das, was `batch_size` leisten soll. `batch_size` ist das **W des
Tropfens**, analog zum W eines Partikels; bei `batch_size = 1` soll ein Event genau
einem physikalischen Tropfen auf ein Partikel entsprechen. Das gilt nur, wenn zwischen
zwei Tropfen tatsächlich neu gezogen wird.

**Korrektur:** `_ensure_samplers()` läuft jetzt nach jedem Tropfen, in beiden
Verteilschleifen (`_distribute_liquid_volume` und `_distribute_remaining_liquid`). Die
Methode entscheidet selbst zwischen vollständigem Rebuild (Partikelzahl geändert) und
inkrementellem O(log n)-Update.

**Kostet nichts — im Gegenteil.** Erwartet war ein Aufschlag, gemessen wurde das
Gegenteil:

| Fall | nur Propensity-Fix | + Sampler pro Tropfen |
|---|---|---|
| Referenz | 27 agg, 5.3 s | 27 agg, **4.6 s** |
| corr_beta → 100 | 688 agg, 11.8 s | 687 agg, **8.5 s** |
| Gewicht → 4000 | 977 agg, 19.9 s | 995 agg, **16.2 s** |
| Vc = 1e-3 (45 s Limit) | 4998 agg bei t = 0.472 | **6109 agg bei t = 0.553** |

Plausible Erklärung: ein aktueller Sampler zieht seltener ins Leere. Vorher konnte er
Partikel treffen, deren Gewicht innerhalb desselben Schrittes längst aufgebraucht war —
jeder solche Griff war eine verworfene Schleifeniteration. Für Läufe mit 10⁴ Partikeln
und 10⁵ Tropfen steht die Messung noch aus; hier standen 500 bis 5.000 Partikel im Feld.

Die Ergebnisse verschieben sich leicht (`poro = 0.7`: 6 → 8 Agglomerationen; x50 bei
zehnfacher Flüssigkeit 58.14 → 53.97 µm), weil sich die Ziehverteilung tatsächlich
geändert hat.

---

## 7. Warum es niemand bemerkt hat

Es existiert ein Test `test_sampler_totals_match_arrays`, der genau diese Konsistenz
prüft — und der war vorher wie nachher grün. Zwei Gründe, beide lehrreich:

1. Er läuft **ohne aktive Nucleation**. Der Wächter stand an der falschen Tür.
2. Selbst mit Nucleation hätte er nicht angeschlagen: Länge und Summe bleiben
   konsistent, wenn ein Partikel mit Propensity **0** angehängt wird — Null addiert
   sich sauber. Eine Summenprüfung kann ein unsichtbares Partikel nicht von einem
   nicht vorhandenen unterscheiden.

### Der Regressionstest (`tests/test_sampling_and_batch_paths.py`)

Entscheidend war die Frage, ob der neue Test den Defekt überhaupt fängt. Erster Versuch:
**nein**. Das Szenario `granulation_1d` lässt Agglomerations-Ereignisse zu, und jedes
akzeptierte Ereignis ruft `_refresh_samplers_after_agg` — der Lauf heilt sich selbst,
der Test blieb mit wieder eingebautem Bug grün.

Sichtbar wird der Defekt erst, wenn **nichts** akzeptiert wird. Der Test hängt dafür
`fittable` mit `u_acc = 0.0` an: jede Kollision wird abgelehnt, die Nucleation bleibt
der einzige Prozess, der die Population verändert. Das entspricht genau der produktiven
Lage bei `INITIAL_POROSITY = 0` (porenfrei bricht nicht, Stokes lehnt als „beide
trocken" ab).

Gegenprobe mit per Monkeypatch wieder eingebauten Defekten:

| Defekt | Ergebnis |
|---|---|
| Kompressions-No-op (verkettete Zuweisung) | **erkannt** — 5 Tests fallen |
| kein Propensity-Refresh nach Nucleation | **erkannt** — 2 Tests fallen |

**Lehre für Regressionstests dieser Art: ein Test, der beim Wiedereinbau des Bugs grün
bleibt, ist kein Test.** Die Gegenprobe gehört dazu.

---

## 8. Zweite Ursache derselben Klasse: die kontinuierlichen Prozesse

**Stand 20.08.2026.** Die Korrektur aus Abschnitt 4 hat die Nucleations-Lücke geschlossen,
nicht aber eine zweite mit identischer Struktur. `tests/test_sampling_and_batch_paths.py::
test_nucleation_keeps_agg_propensities_current` blieb deshalb rot, mit 5,545e-03 relativer
Abweichung.

### Der Mechanismus

`mcpbe_base.py` ruft bei **jedem** MC-Ereignis `continuous_processes.step()`. Deren
Kompressionszweig erhält V_solid und schrumpft das Porenvolumen — er schreibt also
`porosity` **und** `V_flat[-1]`, womit sich der Trockendurchmesser `X` ändert.

Jeder Agglomerationskernel dieses Pakets ist eine Funktion der Radien. Ändert sich `X`,
ändert sich jedes β(i,j) — `_r_agg` ist veraltet, und weil `_agg_sampler` daraus gebaut
wird, sind Ereigniszeit **und** Partnerziehung verzerrt. `powerlaw_rumpf` liest zusätzlich
Porosität und Sättigung für sein Festigkeitsmodell, weshalb auch `_break_rate` veraltet —
dort sogar stärker.

Ausgelöst wurde danach nichts: der einzige Rebuild in dieser Region hing an
`nucleation.consume_population_changed()`.

### Warum es so lange unauffällig blieb

Die Nucleation **maskiert** den Defekt. Solange Flüssigkeit zugegeben wird, feuert sie
fast jedes Ereignis, und ihr Rebuild repariert den Kompressionsschaden nebenbei mit.
Gemessen an `granulation_1d` (Zugabefenster 2 s, Lauf bis 4,13 s):

| t [s] | rel. Abweichung | Rebuilds |
|---|---|---|
| 0,52 | 5,7e-04 | 1 |
| 2,01 | 5,3e-04 | 9 ← letzter Rebuild |
| 2,40 | 1,03e-03 | 9 |
| 3,17 | 3,06e-03 | 9 |
| 4,13 | **5,55e-03** | 9 |

Bis zum Ende des Zugabefensters bleibt der Fehler auf dem Stand eines Ereignisses.
Danach rebuildet nichts mehr und die Ereignisfehler summieren sich linear.

### Nachweis

Setzt man nach dem Lauf `X`, `V_flat` und `porosity` auf den Stand des letzten Rebuilds
zurück und rechnet die Propensities neu, reproduzieren sich die gespeicherten Werte
**bitgenau** (Abweichung 0,000e+00). Die Geometrieänderung ist damit die vollständige
und einzige Ursache. Ausschlussexperiment, konsistent dazu:

| Variante | max. Abweichung |
|---|---|
| alles an | 5,545e-03 |
| ohne Kompression | 0 |
| ohne Internalisierung | 5,545e-03 |
| ohne Nucleation | 0 |

(„ohne Nucleation" ist 0, weil ohne sie keine porösen Partikel entstehen — die
Kompression hat dann nichts zu komprimieren.)

Mit `process_type="mix"` und `powerlaw_rumpf` zeigt sich die zweite Hälfte:

```
Agglomeration _r_agg      max_dev=5.49e-03   betroffen 209/209
Breakage    _break_rate   max_dev=2.49e-02   betroffen  22/209
```

Nur 22 von 209, weil der Rest Vollkörper ist (Rate 0) — dafür fünfmal so stark.

### Die Korrektur

Beide Auslöser laufen jetzt in ein gemeinsames `state_changed` und lösen **einen**
Rebuild pro Ereignis aus, nach Kompression *und* Nucleation. Vorher hätte ein Rebuild
je Block in jedem Ereignis, in dem beide feuern, die Arbeit doppelt gemacht.

Bewusst **ohne** Gate auf „hat die Kompression überhaupt etwas geändert". Der Handler
wüsste es (er rechnet `active = ~isnan(poro) & (poro > min_poro)` für seinen eigenen
Early-Return), aber gemessen spart das Gate:

| Kernel | Rebuilds heute | Kosten je Rebuild | bedingungslos zusätzlich |
|---|---|---|---|
| `shear_chin1998` (Momentenform) | 28 bei 19 Ereignissen | 0,09 ms | ±0 % |
| `eke_darelius2005` (nur pairwise) | 205 bei 300 Ereignissen | 0,43 ms | +4 % |

Beim Shear-Kernel gibt es ohnehin mehr Rebuilds als Ereignisse. 4 % im ungünstigsten
Fall rechtfertigen keinen zweiten Mechanismus, der mit der Physik synchron gehalten
werden muss. Ein Flag **je geänderter Größe** wäre noch schlechter: es müsste wissen,
welche Kernel die Sättigung lesen, und würde beim ersten Agglomerationskernel, der das
tut, still falsch werden — `liquid_bridge` war genau so einer.

### Gegenprobe

Nach der Lehre aus Abschnitt 7 gegen den ungefixten Stand geprüft. Der neue Test
`test_compression_keeps_break_rates_current` fällt dort mit 2,486e-02 und ist mit dem
Fix grün — er fängt den Defekt also wirklich.
