# Offene Befunde (Stand 16.08.2026)

Gefunden bei der Verifikation des Breakage-Formel-Fix. Sortiert nach Dringlichkeit.

Nicht enthalten sind die bereits behobenen Punkte (Ratenformel, `solver.G`-Überschreibung,
`pl_q` wirkungslos, tote Validierung, Nucleation-Buchführung) — siehe
[`Breakage_Kernel_Formula_Divergenz.md`](Breakage_Kernel_Formula_Divergenz.md) und
[`Bias_Correction_und_Gewichtsdisziplin.md`](Bias_Correction_und_Gewichtsdisziplin.md).

> **Nachtrag 17.08.2026 — Baseline-Vorbehalt.** Sämtliche Vorher/Nachher-Vergleiche in
> diesem Dokument stammen aus Läufen, in denen die **Kompression wirkungslos** war
> (No-op in `porosity_compression.compute_array`, seither behoben). Die
> Porositätsangaben und die daraus abgeleiteten Ereigniszahlen sind damit nicht mehr
> reproduzierbar — die Ursachenketten bleiben gültig, die Zahlen sind es nicht. Siehe
> [`Kompression_und_Parametervalidierung.md`](Kompression_und_Parametervalidierung.md).
>
> Dasselbe gilt für die Agglomerationszahlen: neu angelegte Nucleations-Partikel hatten
> Propensity 0 und waren für den Sampler unsichtbar, weshalb `real_agg = 0` in diesen
> Läufen **kein physikalisches Ergebnis** war. Befund #7 („Agglomeration hängt an der
> Breakage") ist damit ein zweites Mal entkräftet — die dort beobachtete Kopplung war
> ein Artefakt: Breakage-Ereignisse haben die Propensities aufgefrischt und damit die
> Agglomeration überhaupt erst ermöglicht. Siehe
> [`Nucleation_Propensity_Blockade.md`](Nucleation_Propensity_Blockade.md).

---

## Bearbeitungsstand (16.08.2026, Abarbeitungsrunde)

| Befund | Status |
|---|---|
| #1 Rumpf-σ Singularität | **erledigt** — ε ≤ 0 / NaN ⇒ Rate 0, oberer Rand auf `poro_max = 0.9999` geklemmt |
| #2 `fastmath` schluckt NaN | **erledigt** — Guard `not (v > 0)` + lauter `isfinite`-Check im Aufrufer |
| #3 Batch schweigt, Einzel wirft | **erledigt** — beide liefern jetzt `0.0` |
| #4 kein vektorisierter Pfad | **erledigt** — `compute_rate_array` überschrieben, bitgleich zum Skalarpfad |
| #5 Test im Commit kaputt | **erledigt** — funktionierende Fassung inzwischen committet |
| #6 toter Konfigcode | **erledigt** — `_break_G` / `_break_pl_P1..P4` entfernt |
| #7 Agglomeration hängt an Breakage | **hinfällig** — siehe unten |
| #8 zwei Commits nie gemerged | **erledigt** — beide nach `pbe-core` portiert |
| #9 zwei Fallbacks für `pl_v` | **bewusst unverändert** — Vermerk steht jetzt im Code |
| #10 abweichende Fallback-Defaults | **erledigt** — löst sich mit #6 auf |

Zusätzlich in derselben Runde gefunden und behoben — Details in Abschnitt
„Nachträge" am Ende:

- **N1** Porositäts-Cap erzeugte abgeleitete Masse (`cone_model`)
- **N2** `max(0.0, min(1.0, NaN))` = 1.0 vernichtete die Masse von Vollkörpern
- **N3** ΔV der Kegelpille wurde negativ → Layering-Modell ergänzt
- **N4** wirkungslose Test-Assertions (`np.isclose` mit Default-`atol`)

### ⚠ Folge für `test_powerlaw_rumpf_full.py`

Gemessen, gleiche Konfiguration, Seed 42, `INITIAL_POROSITY = 0.0`:

| | vor der Runde | danach |
|---|---|---|
| Agglomeration | 33 | **0** |
| Breakage | 5.402 | **0** |
| Stokes-Ablehnungen | 1.074 | 1.099 |
| Porosität final (mean/max) | 0.003 / 0.200 | 0.000 / 0.000 |
| Feststoffmasse | exakt erhalten | exakt erhalten |

Ursachenkette: `INITIAL_POROSITY = 0.0` ⇒ alle Partikel porenfrei ⇒ neue Regel
„porenfrei bricht nicht" ⇒ keine Fragmente ⇒ Stokes lehnt jede Kollision ab.

**Die 5.402 Bruchereignisse der Baseline stammten ausschließlich aus dem
ε=0-Guard**, der porenfreien Partikeln die volle Basisrate zuwies (implizit
σ = 1 Pa). Der Lauf war also auf genau die Inversion kalibriert, die Befund #1
beseitigt. Das ist kein Regressionsfehler, sondern der beabsichtigte Effekt.

Warum die Nucleation die Lücke nicht schließt (gemessen über 195.013 Tropfen):

- `cone_model.compute_nucleation_porosity` liefert **per Design** ε = 0.0;
  195.005 von 195.054 Geometrie-Aufrufen nehmen diesen Zweig. Nucleation
  erzeugt also grundsätzlich keine Poren, nur `liquid_volume` wächst.
- Die erzwungene Agglomeration greift 24× (0,012 %). Auslöser ist
  `V_dry < liquid_volume + v_droplet`, also die *aufsummierte* Beladung; bei
  700 µm Partikeln liegt die mittlere Beladung 480× darunter.
- Diese 24 Kinder bekommen exakt ε = 0.2000 — der Fixpunktabbildung
  ε → 0.8·ε + 0.2 des Kegelmodells entsprechend. Sie tragen zusammen 0,012 %
  des Gesamtgewichts und werden regulär bei W = 0 entfernt.

**Offene Entscheidung:** `INITIAL_POROSITY` auf einen physikalischen Wert
(z.B. 0.8) setzen und `PL_P1` neu kalibrieren. Nicht ohne Rücksprache geändert.

---

## 1. Rumpf-σ: Singularität und Sprungstelle bei der Porosität — **hoch**

`kernels/breakage/powerlaw_rumpf.py`, `_compute_sigma_jit` (Z. 104) und `compute_rate` (Z. 483)

σ = `(1−ε)/ε · γ/x_s · k` geht für ε→1 gegen 0, damit `S = S_base/σ → ∞`. Zusätzlich
fängt `compute_rate` die Ränder mit `if poro <= 0 or poro >= 1: return base_rate` ab —
das erzeugt an beiden Enden einen **Sprung um viele Größenordnungen**.

Gemessen bei V = 2.058e-14 m³, `p1=3e-2, p2=1.0, g=1000, x_s=34 µm`:

| ε | S [1/s] |
|---|---|
| 0 (Guard) | 6.17e-13 |
| 1e-12 | **1.17e-28** |
| 0.5 | 1.17e-16 |
| 1−1e-12 | **1.17e-04** |
| 1 (Guard) | 6.17e-13 |

Zwischen ε=1−1e-12 und ε=1 springt die Rate also um **9 Größenordnungen nach unten**,
zwischen ε=0 und ε=1e-12 um **15 Größenordnungen**.

**Risiko:** Ein Partikel, dessen Porosität durch Nucleation/Kompression nahe an 1 gerät,
bekommt eine praktisch unendliche Bruchrate, dominiert die Event-Auswahl und lässt den
Zeitschritt kollabieren. Aktuell unkritisch, weil die Produktions-Konfiguration bei
ε ≈ 0.2–0.8 bleibt — es ist eine Falle, keine akute Störung.

**Vorschlag:** σ nach unten physikalisch begrenzen (σ_min statt des jetzigen `1e-30`),
oder ε vor der σ-Berechnung auf `[ε_min, ε_max]` klemmen, statt am Rand auf die
unkorrigierte Basisrate zurückzufallen. Der Rückfall auf `base_rate` bei ε=0 ist für
Vollkörper richtig, bei ε=1 aber unphysikalisch (ein reiner Hohlraum ist nicht so fest
wie ein Vollkörper).

## 2. `fastmath=True` schluckt NaN in der Ratenberechnung — **mittel**

`kernels/breakage/_base_rate.py` (`@njit(cache=True, fastmath=True)`)

Verifiziert durch Vergleich mit `.py_func`:

| Eingabe | JIT (`fastmath=True`) | reines Python |
|---|---|---|
| `v = nan` | `0.0` | `nan` |
| `v = inf` | `inf` | `inf` |

`fastmath` erlaubt dem Compiler, NaN-Freiheit anzunehmen; ein NaN-Volumen wird dadurch
still zu Rate `0.0`, statt sich fortzupflanzen. Eine Datenkorruption weiter oben
(z.B. ein NaN in `V_flat`) würde so **maskiert** statt sichtbar zu werden — das Partikel
bräche einfach nie.

`v = inf` ergibt `inf` als Rate und damit eine unendliche Propensity im Fenwick-Sampler.

Betrifft geerbtes Verhalten, nicht neu eingeführt (`fastmath=True` stand schon vorher auf
`_compute_base_rate_jit`). **Vorschlag:** entweder `fastmath` weglassen oder einen
expliziten `if not (v_particle > 0.0): return 0.0`-Guard, der NaN und Negatives gleich
behandelt, plus einen Endlichkeits-Check auf das Ergebnis.

## 3. Fehlender Bruch-Kernel: Batch-Pfad schweigt, Einzel-Pfad wirft — **mittel**

`mcpbe_break.py`: `_calc_break_rates_full` (Z. 142–149) vs. `_break_rate_single` (Z. 194)

Ohne Kernel und ohne MLP setzt der Batch-Pfad alle Raten auf `0.0` und kehrt kommentarlos
zurück („assume breakage is disabled"), während der Einzel-Pfad in derselben Lage
`RuntimeError("Breakage kernel not initialized...")` wirft. Dieselbe Konfiguration ist je
nach Codepfad einmal zulässig und einmal ein Fehler. Ein Lauf kann dadurch lange
fehlerfrei aussehen und erst beim ersten inkrementellen Update abbrechen.

## 4. `powerlaw_rumpf` hat keinen vektorisierten Pfad — **mittel (Performance)**

`kernels/breakage/powerlaw_rumpf.py`

Der Kernel überschreibt `compute_rate_array` nicht, fällt also auf die Schleife in
`kernels/base.py` (Z. 194–198) zurück: ein Python-Aufruf **pro Partikel pro Event**.
`power_law` ist dagegen vektorisiert. Ausgerechnet der produktiv genutzte Kernel ist
damit der langsame. Korrektheit ist nicht betroffen (die Schleife übergibt `particle_idx`,
die σ-Korrektur greift also — geprüft).

Der σ-Teil ist vektorisierbar: `_compute_sigma_jit` ist eine reine Funktion von
`(poro, saturation, x_s, …)` und ließe sich über die aktiven Slices von
`solver.porosity` / `solver.saturation` in einem Rutsch rechnen.

## 5. `test_powerlaw_rumpf_full.py` ist im Commit kaputt — **mittel**

Die **committete** Fassung (`b493120`) übergibt `compression_kernel_name`, das
`MCPBEBase.__init__` nicht mehr akzeptiert → `TypeError` schon bei der Solver-Erzeugung.
Nur die uncommittete Arbeitskopie läuft. Wer den Branch frisch auscheckt, kann den
Referenztest nicht ausführen — mir ist genau daran der Baseline-Vergleich gescheitert.
Betrifft vermutlich auch die daraus generierten `test_droplet_*.py`.

## 6. Toter Konfigurationscode in `_prepare_break_config` — **niedrig**

`mcpbe_break.py` Z. 35–39

`self._break_G`, `_break_pl_P1`, `_break_pl_P2`, `_break_pl_P3`, `_break_pl_P4` werden
gesetzt, aber **nirgends gelesen** (per Grep über `wmcpbe/*.py` verifiziert). Sie stammen
aus dem Pfad vor dem Kernel-Framework, in dem `calc_break_rate_1d` direkt aufgerufen
wurde. Sie suggerieren, die Solver-Attribute `pl_P1`/`pl_P2`/`G` steuerten noch die
Bruchrate — das tun jetzt ausschließlich die Kernel-Parameter. Verwechslungsgefahr.

Anders als `_break_pl_v` / `_break_pl_q`: die sind **aktiv** und gehören zur
Fragmentverteilung (`_build_break_function`) — die müssen bleiben.

## 7. Agglomeration hängt vollständig an der Breakage — **HINFÄLLIG**

> **Aufgelöst (16.08.2026).** Trugschluss aus einer unvollständigen
> Parameterstudie: mit steigender Flüssigkeitsmenge im System löst sich die
> Kopplung auf. Der ursprüngliche Text bleibt nur zur Dokumentation stehen —
> **nicht erneut als Befund aufwerfen.**


In `test_powerlaw_rumpf_full.py` mit `PL_P1=200`: `real agg=0` bei **10.064**
Stokes-Ablehnungen — es wird also gestoßen, aber nichts bleibt haften. Mit
`PL_P1=7.2e6`: `agg=194`, `break=3228`.

Ohne Bruch entstehen keine kleinen Fragmente, und das Stokes-Kriterium lehnt dann
ausnahmslos ab. Das ist physikalisch plausibel, heißt aber: das Testkriterium
„Agglomeration active" prüft indirekt die Breakage mit. Bei der Neukalibrierung von
`PL_P1` sollte man wissen, dass beide Kriterien am selben Parameter hängen.

---

# Teil 2: Abweichungen gegenüber `upstream/dev_monorepo`, die unbeabsichtigt wirken

Systematischer Vergleich `dev_eric` ↔ `upstream/dev_monorepo`. Der Großteil der
Unterschiede in `mcpbe/src/wmcpbe/` (≈36.000 Zeilen) ist die bewusste
Nassgranulations-Entwicklung. Die folgenden Punkte sehen dagegen nach Versehen aus.

## 8. `pbe-core/func/jit_mcpbe.py`: zwei Commits nie übernommen — **hoch**

Die gemeinsame Basis ist auf `dev_eric` **hinter** `dev_monorepo`. Gemeinsamer Stand war
`6e476f0 change jit_mcpbe`; danach hat `dev_monorepo` ergänzt:

- `811432d Add pair-delta correction`
- `d6d9539 update parallel repeats for wmcpbe`

Beides fehlt auf `dev_eric`. Konkret:

| | `dev_monorepo` | `dev_eric` |
|---|---|---|
| `nb_rebuild_ragg_weighted_pair_delta` | vorhanden | **fehlt** |
| `nb_pick_partner_weighted_pair_delta` | vorhanden | **fehlt** |
| Selbstkollision in `nb_rebuild_ragg_weighted` | `delta_ii = min(delta_i, W_i/2)` | alt: nur wenn `W_i > 2·delta_i` |

Das ist eine Divergenz durch **unterlassenen Merge**, keine bewusste Änderung
(`git log` zeigt keinen `dev_eric`-Commit auf dieser Datei nach `6e476f0`).

**Zwei Folgen:**

**(a) `wmcpbe_granulation` rechnet unkorrigiert — und wird von den Validierungsskripten benutzt.**

`mcpbe/src/wmcpbe_granulation/mcpbe_agg.py` ist auf `dev_eric` seit `71fb8ab` (dem
Trennungs-Commit) unverändert und weicht von `dev_monorepo` so ab:

| | `dev_monorepo` | `dev_eric` |
|---|---|---|
| Propensity | `nb_rebuild_ragg_weighted_pair_delta` | `nb_rebuild_ragg_weighted` (alt) |
| Partnerwahl | `nb_pick_partner_weighted_pair_delta` | `nb_pick_partner_weighted` (alt) |
| Selbstkollision | `delta_ii = min(delta_i, 0.5·W_i)` | **komplett abgelehnt** (`if i == j: return 0.0`) |
| dW-Deckelung | auf `delta_ii` | auf `delta_i` |

`dev_eric`s eigenes `wmcpbe` **hat** die Bias-Korrektur (siehe
`Bias_Correction_und_Gewichtsdisziplin.md`), `dev_monorepo`s Granulation ebenfalls —
`dev_eric`s `wmcpbe_granulation` ist damit die einzige Stelle im Repo ohne sie.
Zwangsläufig, denn die `_pair_delta`-Funktionen fehlen hier in `pbe-core`; ein Import
würde scheitern.

**Warum das zählt:** `wmcpbe_granulation` ist nicht totes Beiwerk. Es wird importiert von
`scripts/pbe_validation/granulation/validation.py` und
`scripts/pbe_validation/new/validation.py`. Validierungsläufe prüfen damit eine
Agglomerationsdynamik, die von der produktiv genutzten (`wmcpbe`) abweicht — und zwar
genau in dem Punkt, dessen Korrektur separat hergeleitet und dokumentiert wurde.
Ergebnisse dieser Skripte belegen also nicht das Verhalten des Produktionscodes.

**(b) Verweis ins Leere.**
`kernels/aggregation/jit_kernels.py:25` nennt
`pbe_core.func.jit_mcpbe.nb_rebuild_ragg_weighted_pair_delta` als *"the reference
implementation"* und *"numerical ground truth"*. **Diese Funktion existiert auf diesem
Branch nicht** — wer die eigene Implementierung dagegen prüfen will, findet nichts.

Nicht betroffen: das alte `mcpbe`-Paket nutzt `nb_rebuild_ragg` (ungewichtet).

## 9. Zwei Fallbacks für denselben Fragment-Exponenten widersprechen sich — **niedrig (entschärft)**

> **Nachgeprüft und heruntergestuft.** Der Widerspruch existiert im Code, ist aber
> **nicht erreichbar**: `pbe-core/src/pbe_core/base/base_solver.py:55` setzt
> `self.pl_v = 2` in der *gemeinsamen* Basisklasse, also in beiden Branches. Damit ist
> `solver.pl_v` immer definiert und beide Stellen lesen denselben Wert (empirisch
> geprüft für `BREAKFVAL` 1–5: CDF und Fragmentanzahl nutzen durchgehend `v=2.0`, kein
> `ValueError`). Es bleibt ein latenter Stolperstein für den Fall, dass der Default in
> `base_solver.py` je entfällt — kein aktiver Fehler. Der ursprüngliche Befund folgt
> zur Dokumentation.


`mcpbe_base.py:809-810` (`_compute_frag_num`) vs. `mcpbe_break.py:44-46`
(`_prepare_break_config`)

Beide lesen denselben BREAKFVAL-Exponenten `v`, mit identischem Aufbau — aber
**unterschiedlichem letzten Rückfallwert**:

```python
# mcpbe_base.py  -> Fragment-ANZAHL
v = float(v) if v is not None else float(getattr(self, "pl_v", 1.0))   # 1.0
# mcpbe_break.py -> Fragment-GRÖSSEN-CDF
... else float(getattr(self, "pl_v", 2.0))                             # 2.0
```

`dev_monorepo` nutzt an beiden Stellen `1.0`. `solver.pl_v` wird bei gesetztem
Bruch-Kernel von `kernel_integration.py` **nicht** geschrieben und hat in `MCPBEBase`
keinen Default — der Fallback greift also im Normalfall.

**Folge:** Für `BREAKFVAL` 3/4/5 (wo `p` von `v` abhängt) würde die Fragment*anzahl*
mit `v=1.0` und die Fragment*größenverteilung* mit `v=2.0` gerechnet — zwei
Einstellungen desselben Modellparameters, die nicht zusammenpassen. Bei `BREAKFVAL=3`
ergibt `v=1.0` sogar `p = v = 1.0` und damit direkt den `ValueError` in
`mcpbe_base.py:838` („Expected number of fragments p=1.000 must be > 1").

Aktuell **latent**, weil die Produktions- und Testkonfigurationen `BREAKFVAL=2`
festnageln (dort ist `p = 2.0` hartcodiert und `v` ungenutzt).

## 10. Weitere abweichende Fallback-Defaults (Folge von Befund 6) — **niedrig**

`mcpbe_break.py::_prepare_break_config`, jeweils `getattr(self, …, <default>)`:

| Konstante | `dev_monorepo` | `dev_eric` |
|---|---|---|
| `G` | 1.0 | 1000.0 |
| `pl_P1` | 1.0 | 0.03 |

Praktisch folgenlos, solange diese Attribute tot sind (Befund 6). Wird der
Nicht-Kernel-Pfad je reaktiviert, unterscheiden sich beide Branches ohne Vorwarnung um
Faktor 1000 in der Scherrate. Zusammen mit Befund 6 spricht das dafür, den ganzen Block
zu entfernen statt ihn zu pflegen.

---

## Nachtrag: was geprüft und **nicht** beanstandet wurde

- Volumen-Argument der Rate (`V_flat[-1]` = V_dry) — stimmt mit dem Gold-Standard überein.
- Propensity `W_i·S_i/δ_i`, `δ=min(50, W)` — identisch zu `dev_monorepo`.
- σ-Korrektur im Batch-Pfad — greift (Basis-Schleife übergibt `particle_idx`).
- Agglomerations-Kollisionsraten (`calc_beta`) — byte-identisch zwischen den Branches.
- Skalar- vs. Vektorpfad der Rate — max. relative Abweichung 1.7e-16 über 6.400 Stichproben.
- Parität zur Referenz — 3.000 zufällige `(breakrval, p1, p2, g, V)`-Kombinationen,
  relativer Fehler durchgehend exakt `0.000e+00`.
- Monotonie von S(V) für `breakrval` 2/3/4.
- Trennung `pl_v`/`pl_q` (Solver, Fragmentfunktion) vs. `p2` (Kernel, Rate) — sauber.
- Masseerhaltung unter `breakrval` 1–4 in reinen Bruchläufen (603/1/130/130 echte
  Events): Drift ≤ 1.9e-16, beide `V_solid`-Darstellungen exakt gleich, alle
  Zustandsgrenzen eingehalten. `breakrval=3` und `4` liefern identisch 130 Events —
  wie es sein muss, da sie in 1D dieselbe Formel sind.
- Nucleation-Deckelung numerisch (`Trials/check_frac_volume.py`): abgegebenes Volumen
  == Sollwert für Caps 0.01–3.0, größter Fehler bei 0.078× der Rauschgrenze.

Gegenüber `dev_monorepo` geprüft und als **bewusst** eingestuft (nicht zu beanstanden):

- `pbe-core/func/jit_kernel_break.py`: einzige Abweichung ist der defensive
  `else: return 0.0` in `breakage_func_1d` — dokumentiert und sinnvoll.
- `pbe-core/func/jit_kernel_agg.py`: byte-identisch.
- `lmc_adapter.py`: nur Docstring-Präzisierung (V_ges statt V_solid) — Verhalten gleich.
- `mcpbe_time_helper.py`: Entfernung von `W_MIN_ACTIVE` mit ausführlicher Herleitung;
  gleicht sich laut Kommentar **wieder an** `dev_monorepo` an.
- `reconstruction_mixin.py`: neuer Schutz `_assert_reconstruction_is_safe` gegen
  stillen Massenverlust bei nasser Population — bewusster Zugewinn.
- `fenwick_new.py`: keine abweichende Konstante.
- `mcpbe_post.py`: `alpha = 1.0` an der Interpolationsstelle in beiden Branches
  identisch (ein automatischer Konstantenvergleich meldete hier zunächst einen
  Unterschied — Fehlalarm, die abweichende `0.6` stammt aus einem Parameter-Dict).
