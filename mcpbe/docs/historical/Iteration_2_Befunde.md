# Iteration 2 — Systematische Suche nach NaN-Quellen und Konstantenfehlern

**Stand 16.08.2026.** Reine Analyse. **Es wurde nichts geändert.** Alle Aussagen sind
gemessen, nicht gelesen — die Messskripte laufen als Monkeypatch bzw. als direkter
Randwert-Test gegen die Kernel-APIs, kein Repo-File wird angefasst.

Auftrag: das gesamte Paket `wmcpbe` erneut durchgehen, mit Schwerpunkt auf Kernel,
JIT-Kernel, Nucleation und Konstanten — und besonders auf Stellen, an denen NaN
entsteht oder verwendet wird.

---

## 0. Methodik und Abgrenzung

**NaN ist in diesem Codebase kein reiner Unfall.** `porosity = NaN` ist ein
bewusster Sentinel für „Vollkörper". Ich klassifiziere deshalb jede Fundstelle:

| Klasse | Bedeutung |
|---|---|
| **S** — Sentinel | absichtlich, dokumentiert, wird an der Verbrauchsstelle abgefragt |
| **P** — Post-Processing | „kein Messwert" in Auswertungs-Arrays, verlässt die Physik nie |
| **U** — unbeabsichtigt | entsteht aus Rechnung/Clamp, wird nicht abgefangen |
| **V** — Verzerrung | NaN wird abgefangen, aber auf einen **falschen** Wert abgebildet |

Die Klasse **V** ist die gefährlichste: kein NaN im Ergebnis, kein Crash, aber ein
stillschweigend falscher Zahlenwert.

Bewertungsmaßstab der Randwert-Tests:

```
NaN-OUT   Ergebnis ist NaN -> pflanzt sich in die Partikelarrays fort
MASS      V_dry*(1-eps) weicht von der Summe der V_solid ab
RANGE     Ergebnis ausserhalb des dokumentierten Wertebereichs
RAISE     Exception
```

---

## 1. Gemessene Befunde — Porositäts-Kernel

Getestet: `compute_merged_porosity` mit 8 Randfällen, `compute_fragment_porosity`
mit 5, `compute_nucleation_porosity` mit 2 — je Kernel.

### I2-01 — `incomplete_mixing.compute_nucleation_porosity` liefert **immer** NaN — **hoch**

```
[NaN-OUT] incomplete_mixing/normal:      V_dry=1e-18, eps=nan
[NaN-OUT] incomplete_mixing/v_solid=0:   V_dry=0.0,   eps=nan
```

Nicht ein Randfall — der **Normalfall**. Jede Nucleation mit diesem Kernel schreibt
NaN in `solver.porosity`. Klasse **U**.

Zum Vergleich: `volume_mixing` liefert (1.667e-18, 0.4), `cone_model` (1.0e-18, 0.0).

`incomplete_mixing` ist in der Produktionskonfiguration nicht ausgewählt — der
Befund ist damit latent, aber der Kernel ist über `porosity_growth_kernel_name`
jederzeit erreichbar.

### I2-02 — `incomplete_mixing.compute_merged_porosity`: zwei Vollkörper → NaN — **hoch**

```
[NaN-OUT] incomplete_mixing/beide NaN (Vollkoerper): V_dry=2e-18, eps=nan
```

Klasse **U**. `cone_model` liefert hier seit der letzten Runde 0.2 mit exakt
erhaltener Masse; `volume_mixing` liefert einen gültigen Wert.

### I2-03 — Fragment-Porosität wird zu NaN — **mittel**

```
[NaN-OUT] volume_mixing/parent NaN:      1/1 Fragmente NaN
[NaN-OUT] incomplete_mixing/parent NaN:  1/1 Fragmente NaN
```

Beide reichen einen NaN-Parent unverändert an die Fragmente durch. Klasse **S**
laut Docstring (`blueprint.py:575`: „Solid spheres produce solid fragments"), in der
Wirkung aber **U**: `mcpbe_break.py:705-706` fängt es zwar ab (`NaN -> 0.0`), womit
die Absicht des Sentinels genau verloren geht. `cone_model` ist sauber.

### I2-04 — `volume_mixing` erzeugt Fragment-ε = 1.0, das stromabwärts durch Null teilt — **hoch**

```
[RANGE]  volume_mixing/parent 1.0: eps ausserhalb [0,1): [1.0]
[RAISE]  mcpbe_break:707: eps=1.0: float division by zero
```

Die Kette ist geschlossen und nachgerechnet:

1. `volume_mixing.compute_merged_porosity` klemmt mit `min(1.0, ...)` — ε = 1.0 ist
   also ein **zulässiges Ergebnis** (anders als bei `cone_model`, das auf 0.9999
   deckelt).
2. `compute_fragment_porosity` reicht den Parent-Wert unverändert weiter → 1.0.
3. `mcpbe_break.py:707` rechnet `V_dry_frag = V_solid_frag / (1.0 - frag_poro)`
   → `ZeroDivisionError`, der Lauf bricht ab.

Gemessene Folgekette:

```
eps=0.0    -> V_dry = 1.000000e-18
eps=0.9    -> V_dry = 1.000000e-17
eps=0.9999 -> V_dry = 1.000000e-14
eps=1.0    -> float division by zero
```

Erreichbarkeit: ε = 1.0 kann `volume_mixing` nur produzieren, wenn bereits ein
Elternteil ε = 1.0 trägt (die Formel ist ein gewichtetes Mittel). Eintrittspunkte
sind `helpers.py:490` und direkt gesetzte Testzustände. Also latent — aber es ist
ein harter Abbruch, kein stiller Fehler.

### I2-05 — Ungültige Porositäts-Eingänge erzeugen bzw. vernichten Masse — **mittel**

```
[MASS] volume_mixing/eps > 1:     V_dry*(1-eps)=1.000e-19 vs soll 6.000e-19  (Faktor 0.167)
[MASS] volume_mixing/eps < 0:     V_dry*(1-eps)=1.800e-18 vs soll 1.600e-18  (Faktor 1.125)
[MASS] incomplete_mixing/eps > 1: identisch
[MASS] incomplete_mixing/eps < 0: identisch
```

Bei ε = 1.5 gehen 83 % der Masse verloren, bei ε = −0.2 entstehen 12,5 % aus dem
Nichts. `cone_model` ist in **allen acht** Fällen sauber — es rechnet V_pore aus
V_solid zurück statt ε allein zu klemmen (Korrektur der letzten Runde).

Ungültige Eingänge sollten es nie geben; genau deshalb wäre ein `raise` hier besser
als ein stiller Clamp.

---

## 2. Gemessene Befunde — Bruch-Kernel

### I2-06 — `V = inf` erzeugt `rate = inf` — **mittel**

```
[RANGE] power_law/V=inf/*:       rate ist inf   (5 von 5 Porositaets-Varianten)
[RANGE] powerlaw_rumpf/V=inf/*:  rate ist inf   (3 von 5)
```

Eine unendliche Rate vergiftet die Fenwick-Summe und macht die Ereignisauswahl
undefiniert. Klasse **U**.

Bei `powerlaw_rumpf` fangen die neuen Guards zwei der fünf Fälle ab
(`poro = NaN` und `poro = 0` liefern jetzt 0.0). Der Batch-Pfad wird seit der
letzten Runde durch `mcpbe_break._assert_finite_rates` laut abgefangen — **der
Einzelpfad `_break_rate_single` hat diesen Schutz nicht.**

`V = NaN` liefert in beiden Kerneln korrekt 0.0 (Guard aus der letzten Runde).

---

## 3. Gemessene Befunde — kontinuierliche Prozesse

### I2-07 — `liquid_internalization`: NaN-Sättigung wird zu **voll gesättigt** — **hoch**

```
[ok] liquid_internalization/sat=NaN:  Rueckgabe 1.0
[ok] liquid_internalization/v_pore=NaN: Rueckgabe 0.0
[ok] liquid_internalization/l_total=NaN: Rueckgabe 0.0
```

Dazu die Laufzeitwarnung aus dem Kernel selbst:

```
liquid_internalization.py:173: RuntimeWarning: invalid value encountered
in scalar divide -> l_intern_new = numerator / denominator
```

Klasse **V** — die gefährlichste Sorte. Eine NaN-Sättigung wird nicht als „unbekannt"
oder „trocken" behandelt, sondern als **S = 1.0**, also vollständig gesättigt. Das ist
genau das Gegenteil der Konvention an anderer Stelle: `_compute_sigma_jit` in
`powerlaw_rumpf` behandelt NaN-Sättigung ausdrücklich als **trocken (0.0)**.

Zwei Stellen im selben Rechenweg widersprechen sich damit in der Bedeutung von NaN.
Da S über die Rumpf-Festigkeit direkt in die Bruchrate eingeht, ist das kein
kosmetischer Unterschied: „voll gesättigt" und „trocken" trennen im nassen Regime
ein σ-Verhältnis von 6α/k.

### I2-08 — `porosity_compression` reicht NaN durch — **niedrig, Klasse S**

```
[NaN-OUT] porosity_compression/poro=NaN: porosity_new ist NaN
```

Dokumentiert und beabsichtigt („Vollkörper cannot be compressed", Z. 107-109). Der
Kernel *erzeugt* kein NaN, er reicht es durch. Der Aufrufer
(`mcpbe_continuous_processes._apply_compression`) maskiert NaN korrekt über
`active = ~np.isnan(poro_old) & ...`. Kein Handlungsbedarf, nur der Vollständigkeit
halber notiert.

> **Nachtrag 17.08.2026.** Diese Randfall-Prüfung ging an einem echten Defekt in
> derselben Funktion vorbei: `compute_array` schrieb sein Ergebnis über eine verkettete
> Fancy-Index-Zuweisung und war damit ein vollständiger No-op — die Kompression hat nie
> eine Porosität verändert. Die NaN-Sonde konnte das nicht sehen, weil ein No-op einen
> NaN-Eingang ebenfalls unverändert durchreicht: für diesen einen Testfall ist das
> richtige und das kaputte Verhalten identisch. Siehe
> [`Kompression_und_Parametervalidierung.md`](Kompression_und_Parametervalidierung.md).
> Lehre für weitere Randfall-Runden: eine Sonde, die nur „Eingabe kommt unverändert
> heraus" prüft, kann Stillstand nicht von korrektem Durchreichen unterscheiden — es
> braucht mindestens einen Fall, in dem sich der Wert ändern **muss**.

### I2-09 — `liquid_bridge`: NaN-Sättigung schaltet die Brückenverstärkung stumm ab — **mittel**

```
[ok] liquid_bridge/normal:    beta = 8.000e-21
[ok] liquid_bridge/sat=NaN:   beta = 8.000e-21
[ok] liquid_bridge/poro=NaN:  beta = 8.000e-21
[ok] liquid_bridge/liq=0:     beta = 3.515e-22
```

Bei NaN-Sättigung fällt der Kernel auf die reine Scher-Beta zurück (dokumentiert:
`if np.isnan(si) or np.isnan(sj): return beta`). Klasse **S/V-Grenzfall**: kein NaN im
Ergebnis, aber der gesamte Flüssigkeitsbeitrag verschwindet lautlos. Die Spanne
zwischen „mit Flüssigkeit" und „ohne" beträgt hier Faktor 23 — der Unterschied ist
also nicht klein.

### I2-10 — `liq_internalisation_agglomeration` ist robust — **kein Befund**

Alle sechs Randfälle (v_dry = 0, NaN-Eingänge, negative externe Flüssigkeit,
beide extern 0) liefern sauber 0.0 oder einen endlichen Wert. Keine Beanstandung.

---

## 3b. Gemessene Befunde — JIT-Kernel und Fenwick-Sampler

Getestet: `rebuild_r_pairdelta_pairwise` mit NaN/inf/0/negativ in Radius und Gewicht,
jeweils mit dem **echten** `delta_from_weights` des Solvers — nicht mit einer
selbstgebauten Vorbereinigung.

> **Korrektur einer eigenen Fehlmessung.** Ein erster Durchlauf hatte `delta` selbst
> sanitisiert und damit genau den Effekt maskiert, der gemessen werden sollte;
> außerdem wurde der Radius nur gegen den `constant`-Kernel getestet, der gar nicht
> vom Radius abhängt. Die folgenden Zahlen stammen aus der korrigierten Messung.

### Robust — kein Befund

`delta_from_weights` (`mcpbe_time_helper.py:57-60`) bereinigt NaN und negative
Gewichte zu `delta = 0`, und die JIT-Schleifen überspringen `delta <= 0`. Gemessen:

```
W[1]=NaN  -> delta=[10, 0, 5, 50]  out=[0.027, 0, 0.0245, 0.0795]   alle endlich
W[1]<0    -> delta=[10, 0, 5, 50]  out=[0.027, 0, 0.0245, 0.0795]   alle endlich
```

Das betroffene Partikel fällt heraus, die übrigen bleiben korrekt. Gut konstruiert.

### I2-13 — `W = +inf` wird **nicht** bereinigt und vergiftet alle Propensities — **hoch**

```
W[1]=inf  -> delta=[10, 50, 5, 50]   out=[inf, inf, inf, inf]   4/4 nicht endlich
```

`delta_from_weights` prüft `np.isfinite(delta) & (delta > 0.0)` — aber `delta` ist
`np.minimum(W, dW_const)`, und `min(inf, 50) = 50`. Der Filter greift also für NaN
und negative Werte, **nicht für +inf**. Das Partikel wird nicht übersprungen, `W_i · s`
wird `inf`, und über die Paarsumme wird die Propensity **jedes** Partikels `inf`.

Die Ereignisauswahl ist damit vollständig undefiniert — nicht nur für das eine
Partikel.

### I2-14 — Ein einziger NaN-Radius legt die Agglomeration **still** komplett lahm — **hoch**

Radiusabhängige Kernel, sonst identische Eingaben:

| Fall | `shear` total | `sum` total |
|---|---|---|
| normal | 1.431192e-13 | 6.677619e-15 |
| **R[1] = NaN** | **0.000000e+00** | **0.000000e+00** |
| R[1] = inf | inf (4/4 nicht endlich) | inf (4/4 nicht endlich) |
| R[1] = 0 | 7.388080e-14 | 5.356776e-15 |
| R[1] < 0 | 5.755680e-14 | 4.881934e-15 |

Der Mechanismus, Zeile für Zeile: `if bij <= 0.0: continue` ist für `bij = NaN`
**falsch**, das Paar wird also nicht übersprungen. `s` wird NaN, `val = W_i · s`
wird NaN, und der Abschluss `r[i] = val if val > 0.0 else 0.0` bildet NaN auf
**0.0** ab, weil `NaN > 0.0` falsch ist.

Ergebnis: kein NaN in der Ausgabe, kein Absturz, keine Warnung — aber jedes
Partikel, das mit dem defekten paart, bekommt Propensity 0. Bei einer vollen
Paarschleife ist das die **gesamte Population**. Die Agglomeration hört auf, und
nichts im Lauf weist darauf hin.

Klasse **V**, und das mit Abstand heimtückischste Muster dieser Runde: der
`else 0.0`-Zweig ist als Schutz gegen negative Werte gedacht und wirkt hier als
NaN-Schlucker.

### I2-15 — `FenwickSampler` validiert seine Eingaben nicht — **mittel**

```
normal            total=6.0   sample->2      ok
enthaelt 0        total=4.0   sample->2      ok
enthaelt NaN      total=nan   sample->0      <- still falsch
enthaelt inf      ValueError: s must be in [0,total)
enthaelt negativ  total=2.0   sample->2      <- Verteilung still verfaelscht
alle 0            ValueError: empty or non-positive total
```

Bei NaN liefert `total()` NaN und `sample()` gibt kommentarlos Index 0 zurück — die
Ereignisauswahl ist verfälscht, ohne dass etwas auffällt. Bei einem negativen
Gewicht wird dieses in die Summe eingerechnet (1 − 2 + 3 = 2), die
Auswahlwahrscheinlichkeiten stimmen nicht mehr, und auch das bleibt still.

Die beiden `ValueError`-Fälle sind demgegenüber das gewünschte Verhalten.

### Nicht gemessen

Der Vergleich `rebuild_r_pairdelta_moment` gegen `rebuild_r_pairdelta_pairwise`
wurde **nicht** durchgeführt: die Moment-Variante nimmt vorberechnete separable
Faktoren `(F, G, beta_ii, …)` entgegen, keine `(kid, p0)`. Ein belastbarer Vergleich
müsste diese Matrizen so aufbauen wie `_rebuild_all_propensities`. Die in
`Fundamentals.md` genannte Übereinstimmung von ~1e-15 ist damit hier **nicht**
bestätigt, aber auch nicht widerlegt.

---

## 3c. Gemessene Befunde — `ParticleMerger`

### Robust — kein Befund

Die NaN-Logik im Matching ist sorgfältig und in **beiden** Pfaden konsistent
(vektorisiert Z. 487-504, skalar Z. 622-640): NaN matcht NaN, NaN gegen endlich ist
ein Mismatch, sonst Toleranzvergleich. Der Hash-Schlüssel bildet NaN auf `None` ab
(Z. 552, 557) — ein echter Singleton, damit funktioniert der Dict-Lookup. Wäre dort
NaN selbst als Schlüssel gelandet, hätte der Lookup nur zufällig funktioniert, weil
`nan == nan` falsch ist. Sauber gelöst.

### I2-16 — Der Hash-Index findet 98,8 % der zulässigen Treffer nicht — **hoch (Wirksamkeit)**

Die Bin-Breite des Volumens und die Matching-Toleranz passen nicht zueinander:

```
bin_digits_volume = 8  -> Bin-Breite in log10 = 1.0e-08
                          entspricht relativer Volumenbreite = 2.303e-08
tol_rel           = 1e-06
-> das Toleranzfenster ist 43.4x BREITER als ein Bin
```

Zwei Partikel, die laut Toleranz verschmolzen werden **dürften**, liegen damit
typischerweise in verschiedenen Bins. Gemessen an 20.000 Zufallspaaren, die
garantiert innerhalb `tol_rel = 1e-6` liegen:

```
Paare innerhalb der Toleranz:  20.000
davon im GLEICHEN Hash-Bin:       231   (1,16 %)
-> 98,84 % der zulaessigen Treffer findet der Index nicht
```

**Und es gibt keinen Auffangpfad.** `_find_via_hash` schlägt genau einen Bin nach
und gibt bei leerem Ergebnis `-1` zurück; der lineare Scan läuft nur, wenn der Index
komplett abgeschaltet ist (`use_hash_index=False`) oder `force_linear_scan` gesetzt
wird. Nachbar-Bins werden nie konsultiert.

Der Docstring von `_find_via_hash` behandelt ausdrücklich den umgekehrten Fall
(„Hash collisions are possible … candidates are verified with exact tolerance
comparison") — also falsch-**positive**. Falsch-**negative** durch Bin-Grenzen sind
nicht adressiert.

**Folge:** Die Dedup-Logik verfehlt ihren dokumentierten Zweck weitgehend. Nutzer-Zitat
aus `Fundamentals.md` §4: *„Der Merger war so gedacht, dass keine redundanten
Child-Partikel entstehen."* Faktisch entstehen sie in ~99 % der Fälle trotzdem.
`n_comp` wächst schneller als nötig, die Rekonstruktion greift früher.

**Kein Physik- oder Massenfehler:** Duplikate sind physikalisch äquivalente Partikel,
die Bilanzen bleiben exakt. Es ist ein Wirksamkeits- und Laufzeitproblem.

Die Porositäts-/Sättigungsdimension hat das umgekehrte, unkritische Verhältnis:
Bin 1.0e-04 gegen Toleranz 4.0e-07 auf ε ≈ 0.4 — der Bin ist 250× breiter, also
über-einschließend, und der exakte Vergleich danach filtert korrekt nach.

---

## 3d. `ReconstructionMixin` — kein eigener Fehler, aber eine Wechselwirkung

### Robust — kein Befund

`_assert_reconstruction_is_safe` (Z. 109-160) verweigert die Rekonstruktion einer
Population mit Flüssigkeit oder Porosität, statt sie stillschweigend zu
verstümmeln. Der Docstring benennt präzise, was sonst passieren würde:
`liquid_volume`/`porosity`/`saturation` behalten die Werte des vorher an diesem
Index liegenden Partikels, und `V_flat[-1]` wird auf `sum(V_flat[:dim])` gesetzt,
verwirft also den Porenraum. Gemessener Verlust laut Kommentar: ~0,4 % Feststoff
je Rekonstruktionsschritt, kumulierend. Ein sauberer, bewusster Schutz.

### I2-17 — Wechselwirkung: kein Ventil für wachsendes `n_comp` bei nasser Population — **mittel**

Aus zwei für sich korrekten Entscheidungen entsteht eine Klemme:

1. **I2-16**: der Hash-Index findet 98,8 % der zulässigen Verschmelzungen nicht,
   `n_comp` wächst dadurch deutlich schneller als vorgesehen.
2. **`_assert_reconstruction_is_safe`**: die Rekonstruktion — das einzige Mittel,
   `n_comp` wieder zu senken — ist für nasse Populationen gesperrt.

Für einen Nassgranulationslauf gibt es damit **kein Ventil**: Duplikate sammeln
sich an, und bei Überschreiten von `recon_N_max` (Default-Trigger 4000) bricht der
Lauf ab, statt die Partikelzahl zu reduzieren.

In der aktuellen Testkonfiguration nicht akut — `test_powerlaw_rumpf_full.py` setzt
`solver.recon_enable = False`, und die Läufe bleiben unter der Schwelle. Bei
längeren Läufen oder größerer Anfangspopulation wird es das aber.

Der wirksamste Hebel ist I2-16: mit funktionierender Dedup entsteht das Problem
gar nicht erst.

---

## 4. Konstanten-Nachrechnung

### Nachgerechnet und **korrekt**

| Konstante | Sollwert | Gemessen |
|---|---|---|
| ΔV Kegelpille, gleich große Kugeln | (2/3)·π·r³ | 0.6666666667 ✓ |
| ΔV Layering | (1/3)·π·r³ | 0.3333333333 ✓ |
| σ(S) Stetigkeit bei S = 0.3 | kein Sprung | rel. 1.84e-09 ✓ |
| σ(S) Stetigkeit bei S = 0.8 | kein Sprung | rel. 1.25e-09 ✓ |
| `SAT_TRANSITION_WIDTH` | `SAT_WET − SAT_DRY` = 0.5 | 0.5 ✓ |
| `cone_model.PORO_MAX` vs `powerlaw_rumpf.poro_max` | identisch | 0.9999 = 0.9999 ✓ |

### I2-11 — `SAT_TRANSITION_WIDTH` ist redundant statt abgeleitet — **niedrig**

Die Konstante ist eigenständig definiert, nicht als `SAT_WET − SAT_DRY` berechnet.
Aktuell stimmt sie. Zur Demonstration der Folge einer Änderung wurde `SAT_WET` auf
0.9 gesetzt und `WIDTH` unverändert gelassen:

```
SAT_WET=0.9, WIDTH bleibt 0.5 -> links=7914.7059  rechts=7147.0588
                                 relativer Sprung = 1.074e-01
```

Ein Sprung von 10,7 % in der Festigkeit an der Regimegrenze — genau die Art
Sprungstelle, die Befund #1 der letzten Runde beseitigt hat. Als abgeleiteter Wert
(`WIDTH = WET − DRY`) wäre das konstruktiv unmöglich.

---

## 5. Statisches NaN-Inventar

Vollständige Erfassung über `wmcpbe/**/*.py` (ohne `Trials/`, ohne `Results/`).

### Erzeugende Stellen

| Ort | Klasse | Bemerkung |
|---|---|---|
| `helpers.py:490` | **S** | `porosity[:] = np.nan` bei `initial_porosity=None` — der Haupteintrittspunkt für Vollkörper |
| `kernels/porosity_growth/blueprint.py` 516, 575, 600, 619 | **U** | Rückgabe NaN für Merge/Fragment; Z. 516 wird fünf Zeilen später mit `max(0.0, min(1.0, …))` behandelt → **NaN wird zu 1.0** |
| `kernels/porosity_growth/incomplete_mixing.py:299` | **U** | siehe I2-01/I2-02 |
| `mcpbe_post.py` (10 Stellen), `mcpbe_base.py` (14 Stellen) | **P** | Auswertungs-Arrays, „kein Messwert" |
| `reconstruction_mixin.py:1148` | **P** | Rückgabewert einer Kennzahl |

### I2-12 — Die NaN→1.0-Falle existiert weiterhin in `blueprint.py` — **hoch**

```python
# blueprint.py
if V_pore_merged <= 0:
    poro_merged = np.nan          # Z. 516
else:
    poro_merged = V_pore_merged / v_dry_merged
poro_merged = max(0.0, min(1.0, poro_merged))    # Z. 521  -> NaN wird 1.0
```

In Python ist `min(1.0, nan) == 1.0`. Ein porenfreier Körper wird damit zu „reiner
Hohlraum", sein V_solid zu `V_dry · (1 − 1.0) = 0` — die Masse dieses Partikels
verschwindet. Exakt der Fehler, der in der letzten Runde in `cone_model` behoben
wurde; in der Basisklasse steht er noch.

Betrifft alle Kernel, die die Blueprint-Implementierung erben statt sie zu
überschreiben. Dieselbe Konstruktion nochmals in Z. 650.

### Abfragende Stellen (Verdichtung)

`isnan`-Treffer je Datei: `blueprint.py` 20, `mcpbe_nucleation.py` 18, `mcpbe_agg.py` 16,
`cone_model.py` 14, `particle_merger.py` 8, `mcpbe_base.py` 7, `mcpbe_break.py` 6.
Die Dichte zeigt, wie tief der Sentinel in die Physik hineinreicht.

---

## 5b. Laufzeitnachweis — entsteht im echten Lauf NaN?

Instrumentierung per Monkeypatch: nach **jedem** Nucleations-, Kontinuum-,
Agglomerations- und Bruchschritt werden alle sieben Physik-Arrays auf
nicht-endliche Werte geprüft, dazu die Invarianten. Konfiguration
`test_powerlaw_rumpf_full.py` mit `INITIAL_POROSITY = 0.8`, damit die
flüssigkeitsführenden Pfade überhaupt durchlaufen werden.

```
Pruefpunkte: 613
Keine inf-Werte und keine NaN ausserhalb der erlaubten Sentinel-Felder gefunden.

Invarianten:
  groesste rel. Abweichung V_dry*(1-eps) vs V_flat[0]:  1.917e-16
  groesste Porositaet:                                  0.840000
  groesste Saettigung:                                  0.069842
  groesstes V_liq_int / V_pore:                         0.069842
```

**Das ist der wichtigste Einzelbefund dieser Runde:** im laufenden Betrieb ist der
Zustand sauber. Die beiden `V_solid`-Darstellungen stimmen auf Maschinengenauigkeit
überein (1.9e-16 = wenige ULP), Porosität und Sättigung bleiben in ihren Grenzen,
und die interne Flüssigkeit überschreitet nie das Porenvolumen.

Damit sind **alle** unter I2-01 bis I2-15 dokumentierten Befunde **latent**: sie
brauchen entweder einen anderen Porositäts-Kernel (`incomplete_mixing`), einen
ungültigen Eingabewert oder bereits korrumpierte Daten. Keiner von ihnen feuert in
der Produktionskonfiguration.

Das entwertet sie nicht — es ordnet sie ein. Ein NaN, das erst durch einen
Kernelwechsel oder eine Datenkorruption entsteht, wird dann von den Mustern aus
Abschnitt 6 still verschluckt statt gemeldet.

---

## 6. Priorisierung

| # | Befund | Schwere | Klasse | Erreichbarkeit |
|---|---|---|---|---|
| I2-14 | **ein NaN-Radius stoppt die Agglomeration still komplett** | hoch | **V** | jeder NaN in `X`/`R` |
| I2-13 | `W = +inf` wird nicht bereinigt → alle Propensities `inf` | hoch | U | jeder inf in `W` |
| I2-12 | `blueprint.py` NaN→1.0, Masse verschwindet | hoch | U/V | über erbende Kernel |
| I2-07 | NaN-Sättigung wird „voll gesättigt" statt „trocken" | hoch | **V** | jede NaN-Sättigung |
| I2-01 | `incomplete_mixing` Nucleation liefert immer NaN | hoch | U | Kernel wählbar |
| I2-04 | Fragment-ε = 1.0 → Division durch Null, Abbruch | hoch | U | nur mit ε=1-Eltern |
| I2-02 | `incomplete_mixing` zwei Vollkörper → NaN | hoch | U | Kernel wählbar |
| I2-16 | Hash-Index verfehlt 98,8 % der zulässigen Treffer | hoch* | — | immer aktiv |
| I2-17 | kein Ventil für `n_comp` bei nasser Population | mittel | — | bei langen Läufen |
| I2-15 | `FenwickSampler` ohne Eingabevalidierung | mittel | V | NaN/negative Gewichte |
| I2-06 | `V = inf` → `rate = inf`, Einzelpfad ungeschützt | mittel | U | nur bei Datenkorruption |
| I2-09 | `liquid_bridge` NaN-Sättigung schaltet Brücke ab | mittel | S/V | jede NaN-Sättigung |
| I2-05 | ungültiges ε erzeugt/vernichtet Masse | mittel | V | nur bei ungültiger Eingabe |
| I2-03 | Fragment-Porosität NaN | mittel | S→U | NaN-Parent |
| I2-11 | `SAT_TRANSITION_WIDTH` redundant | niedrig | — | erst bei Änderung |
| I2-08 | `porosity_compression` reicht NaN durch | niedrig | **S** | beabsichtigt |

\* I2-16 ist als einziger Befund **dauerhaft aktiv**, verletzt aber keine Bilanz —
er kostet Laufzeit und Speicher, nicht Korrektheit.

### Das wiederkehrende Muster

Drei der vier schwersten Befunde (I2-14, I2-07, I2-12) haben dieselbe Bauform:

```python
x = max(0.0, min(1.0, x))      # NaN -> 1.0
r = val if val > 0.0 else 0.0  # NaN -> 0.0
if bij <= 0.0: continue        # NaN -> nicht uebersprungen
```

Ein Vergleich mit NaN ist **immer falsch**. Jede `if`-Abfrage, die einen Wert
gegen eine Schranke prüft, ordnet NaN damit stillschweigend dem `else`-Zweig zu —
und der ist überall so geschrieben, als sei dort der harmlose Fall. Das erzeugt
kein NaN im Ergebnis und keinen Absturz, sondern einen falschen Zahlenwert.

Ein globaler Endlichkeits-Assert auf den Physik-Arrays (siehe unten) würde alle
drei auf einen Schlag sichtbar machen.

### Beobachtung zur Frage „kann NaN ganz verschwinden?"

Der Sentinel selbst ist entbehrlich: `cone_model` und `mcpbe_break` bilden NaN
bereits auf 0.0 ab und behandeln „porenlos" damit als regulären Zustand statt als
Sonderfall. Der Umstieg wäre eine Designentscheidung mit drei Schritten:

1. `helpers.py:490` schreibt 0.0 statt NaN.
2. `blueprint.py` und `incomplete_mixing.py` geben 0.0 statt NaN zurück
   (behebt gleichzeitig I2-12, I2-01, I2-02, I2-03).
3. Die verbleibenden ~90 `isnan`-Abfragen in der Physik werden zu totem Code und
   können entfallen — **nicht** aber die in `mcpbe_post.py`/`mcpbe_base.py`, wo NaN
   legitim „kein Messwert" bedeutet.

Damit bliebe NaN ausschließlich im Post-Processing. Danach wäre ein globaler
`assert np.all(np.isfinite(...))` auf den Physik-Arrays möglich — der Detektor, der
heute fehlt.

**Nicht umgesetzt.** Erst zu entscheiden.

---

## 7. Was geprüft und **nicht** beanstandet wurde

- `liq_internalisation_agglomeration` — alle sechs Randfälle sauber
- `cone_model` — alle 8 Merge- und 5 Fragment-Randfälle sauber, keine Masseabweichung
- `V = NaN` in beiden Bruch-Kerneln → korrekt 0.0
- σ(S)-Stetigkeit an beiden Regimegrenzen
- Kegel- und Layering-Geometriekonstanten gegen die geschlossene Form
- `PORO_MAX`-Konsistenz zwischen Porositäts- und Bruch-Kernel
- `porosity_compression` — NaN-Maskierung im Aufrufer korrekt (aber: der Kernel selbst
  war zu diesem Zeitpunkt wirkungslos, siehe Nachtrag bei I2-08)
