# Umsetzung der Audit-Befunde (17./18.08.2026)

Begleitdokument zu [`Audit_2026-08-17.md`](Audit_2026-08-17.md).
Hier steht, **was geändert wurde, was offen ist und warum**.

---

## Statusübersicht

| # | Thema | Status | Wo geändert |
|---|---|---|---|
| B-01 | `X` aus `V_solid` statt `V_dry` | ✅ **behoben** | `mcpbe_base` (neue Methoden), 5 Aufrufstellen |
| B-02 | `Vc` → Konzentration 8 Größenordnungen zu klein | ⚙️ **Testwerte angepasst** | Sonden-Konfiguration |
| B-03 | `SIZEEVAL` invertiert + andere Formel | ✅ **für Referenzconfig deaktiviert** (`SIZEEVAL=0`), Doku in Fundamentals.md | `Trials/test_powerlaw_rumpf_full.py` |
| B-04 | Merger-Hash-Index veraltet | ✅ **behoben** | `particle_merger`, `mcpbe_continuous_processes`, `mcpbe_base` |
| B-05 | Merger nur bei Breakage | ✅ **behoben** | `mcpbe_base._ensure_particle_merger` |
| B-06 | `agg_dW_max=1` vs. `break_dW_const=50` | ✅ **behoben, per `dev_monorepo`-Abgleich** | `Trials/test_powerlaw_rumpf_full.py` |
| B-07 | Kompression erzeugt Masse (ε→1) | ✅ **behoben** | `mcpbe_continuous_processes` |
| B-08 | `k_int` Größenordnung | ⚙️ **Testwerte angepasst** | Sonden-Konfiguration |
| B-09 | `_merge_liquid` toter Code | ✅ **behoben** | `mcpbe_agg` |
| B-10 | zwei Porositäts-Defaults | ✅ **behoben** (jetzt durchgängig 0.0) | `mcpbe_base` |
| B-11 | zweite Array-Liste im Vc-Doubling | ✅ **behoben** | `mcpbe_base` |
| B-12 | Hash-Index nach Vc-Doubling | ✅ **behoben** | `mcpbe_base` |
| B-13 | toter `compression`-Handler | ✅ **gelöscht** | `mcpbe_base.solve` |
| B-14 | unerreichbare Warnung | ✅ **zugänglich gemacht** | `kernel_integration` |
| B-15 | asymmetrisches Überschreiben | ✅ **behoben** + Toggle | `mcpbe_agg`, `mcpbe_base` |
| B-16 | `V_dry_old`-Alias | ✅ **behoben** (echte Kopie) | `mcpbe_continuous_processes` |
| B-17 | Massenverlust Breakage-Fallback | ✅ **behoben** | `mcpbe_break` |
| B-18 | `real_break_events` zu hoch | ✅ **behoben** + Zähler | `mcpbe_break` |
| B-19 | zwei Kollisionsgeschwindigkeiten (`v_rel`) | ✅ **`v_rel`/`E_coll` entfernt** | `mcpbe_agg`, `mcpbe_nucleation` |
| B-20 | Bruch-Runaway | ⚙️ **Testwerte angepasst** | Sonden-Konfiguration |
| B-25 | Energie-Kanal ohne Verbraucher | ✅ **vollständig entfernt** | 11 Dateien, s. u. |
| N-01 | Sammel-Loop mit eingefrorenem `a_tot` | ✅ **behoben** | `mcpbe_nucleation` |
| N-02 | Sampler-Staleness | ✅ **nicht bestätigt** (0 von 213 Stichproben) | — |
| N-03 | wt%-Laufzeitüberwachung (Relikt) | ✅ **entfernt** (168 Zeilen) | `mcpbe_nucleation` |
| N-04 | duplizierter Block + doppelte Logzeile | ✅ **entfernt** | `mcpbe_nucleation` |

### Verifikation nach den Änderungen

```
x_consistency_probe:   X inkonsistent  0 von 949  (vorher 64)
merger_and_mass_probe: ΔV_solid mit Merger    +0.000000 %
                       ΔV_solid ohne Merger   +0.000000 %  (vorher -0.001424 %)
                       Hash-Index 1023 Eintraege bei a_tot=1023, Ueberhang +0
                       Eintraege unter falschem Schluessel  0.0 %  (vorher 55 %)
porous_regime_probe:   eps0=0.0 -> X inkonsistent 0 %, falscher Schluessel 0 %
```

---

## B-03 · SIZEEVAL — ausführlich erklärt, **jetzt für die Referenzconfig entschieden**

Du sagst, dein Betreuer hat das für die **Trockenagglomeration** geschrieben.
Das passt zu allem, was ich im Code sehe – und erklärt genau, warum es hier
Probleme macht. Der Reihe nach.

### Was SIZEEVAL ursprünglich ist

`SIZEEVAL` ist ein **Schalter**, kein Wert. Er wählt aus, ob und wie die
**Kollisionseffizienz α** von der Partikelgröße abhängt.

α ist der Anteil der Zusammenstöße, die tatsächlich zum Verkleben führen:

```
tatsächliche Agglomerationsrate = β(r₁,r₂) · α
                                   ↑          ↑
                          wie oft treffen   wie oft bleiben
                          sie sich?         sie kleben?
```

In `pbe-core/src/pbe_core/base/base_solver.py:31` steht die Definition:

```python
self.SIZEEVAL = 1   # 1 = No size dependency, 2 = Model from Soos2007
```

Also:
- **`SIZEEVAL = 1`** → α ist konstant, keine Größenabhängigkeit. **Das ist der Default.**
- **`SIZEEVAL = 2`** → α wird mit einem Korrekturfaktor `corr_size` multipliziert.

### Was Soos2007 (SIZEEVAL = 2) physikalisch aussagt

Die Formel (aus `dpbe_core.py:378`, identisch in `mcpbe_jit.py:619`):

```
                exp( −X_SEL · (1 − λ)² )
corr_size  =  ─────────────────────────────        λ = min(R₁/R₂, R₂/R₁)
               ( R₁·R₂ / R₀² ) ^ Y_SEL
```

Zwei getrennte Aussagen, beide aus der **Trocken**agglomeration von Flocken:

**Zähler — der λ-Term.** λ ist das Größen*verhältnis*: 1 bei gleich großen
Partikeln, klein bei sehr ungleichen. `exp(−X_SEL·(1−λ)²)` ist maximal bei λ = 1
und fällt ab, je unterschiedlicher die Partner sind.
*Begründung:* Ein kleines Partikel, das auf ein großes trifft, wird von dessen
Strömungsfeld umgelenkt und trifft die Oberfläche gar nicht erst richtig.
Gleich große Partner haften am besten.

**Nenner — der Potenzterm.** `(R₁·R₂/R₀²)^Y_SEL` wächst mit der Größe, steht im
Nenner, dämpft also große Partikel.
*Begründung:* Große Flocken sind fraktal, locker und mechanisch schwach. Im
Scherfeld reißen sie beim Stoß wieder auseinander, bevor eine dauerhafte
Bindung entsteht. Das ist ein **glatter Potenzabfall** ohne bevorzugte Größe:
je größer, desto unwahrscheinlicher — monoton, ohne Maximum.

`X_SEL = 0.310601` und `Y_SEL = 1.06168` sind die an Messdaten gefitteten
Konstanten dieser Formel (Selomulya 2003 / Soos 2006/2007).

### Was in wmcpbe tatsächlich steht

`mcpbe_agg.py::_accept_sizeeval`:

```python
if int(getattr(self, "SIZEEVAL", 1)) == 0:
    return True                                   # (a) nur 0 schaltet ab
...
V_target = mean(V0[-1,:])                          # mittleres ANFANGSvolumen
sigma_V  = V_target * Y_SEL
size_factor = exp(-0.5*((Vi - V_target)/sigma_V)**2) \
            * exp(-0.5*((Vj - V_target)/sigma_V)**2)   # (b) Gauss statt Potenz
return u_acc <= size_factor
```

**Zwei unabhängige Abweichungen:**

**(a) Der Schalter ist invertiert.** Geprüft wird auf `== 0`. Der Default ist
aber `1`, und `1` heißt laut Definition „aus". Der Filter läuft also in jedem
Lauf, obwohl der Parameterwert das Gegenteil sagt. Um ihn abzuschalten, müsste
man `SIZEEVAL = 0` setzen — einen Wert, den die Konvention gar nicht kennt.

**(b) Die Formel ist eine andere Funktion.**

| | Soos2007 | wmcpbe |
|---|---|---|
| Form | Potenzabfall über `R₁·R₂` | **Gauß-Glocke** über `V` |
| Argument | Größen**verhältnis** λ + Größen**produkt** | **absolutes Volumen** jedes Partners |
| Verlauf | monoton fallend | **Maximum bei `V = V_target`** |
| `X_SEL` | benutzt | **gar nicht gelesen** |
| `Y_SEL` | Potenz-**Exponent** | relative **Standardabweichung** |

Der entscheidende inhaltliche Unterschied ist das **Maximum**. Soos2007 sagt
„größer ⇒ unwahrscheinlicher". Die Gauß-Form sagt „**es gibt eine bevorzugte
Größe, und Abweichung nach oben *und unten* wird bestraft**". Das ist eine
andere physikalische Aussage.

Und weil die Glocke auf das **Anfangs**volumen zentriert ist, wird der Filter
mit fortschreitender Granulation immer strenger:

| Granulat aus *k* Primärpartikeln (beide Partner) | Annahmewahrscheinlichkeit |
|---|---|
| 1 | 1.0 |
| 2 | 0.41 |
| 3 | 0.029 |
| 5 | 6.8 · 10⁻⁷ |
| 10 | 6.2 · 10⁻³² |

**Warum das für Trockenagglomeration trotzdem sinnvoll gewesen sein kann:** Dort
war das Ziel oft eine begrenzte Flockengröße im Gleichgewicht zwischen Wachstum
und Scherabbruch. Eine Glocke um eine charakteristische Größe modelliert genau
das — als Ersatz für einen fehlenden expliziten Bruchmechanismus.

**Warum es hier nicht passt:** wmcpbe *hat* einen expliziten Bruchmechanismus
(`powerlaw_rumpf`) und ein physikalisches Haftkriterium (`stokes_krit`,
Braumann 2007). Beide beschreiben denselben Effekt bereits — der Größenfilter
kommt als drittes, unbelegtes Modell obendrauf und deckelt das Wachstum
hart bei ~5 Primärpartikeln (≈ 58 µm bei 34 µm Ausgangsware).

### Ein zusätzlicher Detailfehler

`Vi`/`Vj` werden aus `V_flat[0, ·]` gelesen — das ist **V_solid**.
`V_target` kommt aus `V0[-1, :]` — das ist **V_dry**. Bei poröser Ausgangsware
sind das verschiedene Größen (V_solid = V_dry·(1−ε)), der Vergleich verschiebt
sich also mit der Anfangsporosität. Bei ε = 0 (deiner aktuellen Testconfig)
fällt das nicht auf, weil dort V_solid = V_dry.

### Warum ich nichts geändert habe

Es sind **drei getrennte Entscheidungen**, und alle drei sind fachlich, nicht
technisch:

1. **Soll bei Nassgranulation überhaupt eine empirische Größenabhängigkeit von
   α aktiv sein?** Meine Einschätzung: nein — `stokes_krit` und
   `powerlaw_rumpf` decken den Effekt physikalisch begründet ab. Aber das ist
   deine Modellentscheidung, ggf. mit deinem Betreuer.
2. **Falls ja: welche Formel?** Soos2007 (konsistent mit dpbe/mcpbe/qmom) oder
   die Gauß-Form? Für Letztere steht keine Quelle im Code.
3. **Der Schalter** muss in jedem Fall auf die Projektkonvention zurück
   (`1 = aus`, `2 = Soos2007`), sonst bleibt die Falle bestehen.

**Für einen Lauf ohne den Filter genügt heute:** `solver.SIZEEVAL = 0`.

### KORREKTUR (18.08.2026)

Nach Rücksprache mit deinem Betreuer: der Filter ist für euren Fall **nicht
relevant** -- er wurde für Trockenagglomeration geschrieben, WMCPBE deckt den
Effekt bereits über `powerlaw_rumpf` und `stokes_krit` ab. Frage 1 oben ist
damit beantwortet (nein). Fragen 2/3 (korrekte Formel bzw. Schalterkonvention
projektweit) bleiben offen, betreffen aber nur andere Nutzer dieses Codes.

**Geändert:**
- `Trials/test_powerlaw_rumpf_full.py` setzt jetzt `solver.SIZEEVAL = 0` explizit.
- `docs/Fundamentals.md` (Abschnitt 3.1) trägt dauerhaft nach, was der Schalter
  tut, warum er hier irrelevant ist und wie man ihn abschaltet -- fuer dich zum
  Nachlesen und fuer kuenftige Sessions.

**Warum das nicht bloss Doku war:** Der Filter war beim ersten B-06-Vergleich
unbeabsichtigt aktiv und hat das Ergebnis konfundiert (agg-dW-Summe 25 % zu
niedrig). Details unter B-06.

---

## B-19 · Was `v_rel` ist und warum es existiert

Du hast die Variable noch nie gesehen — kein Wunder, sie ist **lokal** und
existiert nur in zwei Funktionen:

- `mcpbe_agg.py:1019` in `_merged_porosity`
- `mcpbe_nucleation.py:2124` in `_manual_agglomerate_particles`

```python
rho    = 1000.0                                   # kg/m³, angenommene Granulatdichte
d_eff  = (X[i] + X[j]) * 0.5                      # mittlerer Durchmesser der Partner
v_rel  = G * d_eff                                # ← hier
E_coll = 0.5 * (rho * (V_dry_i + V_dry_j)) * v_rel**2
```

### Wozu es dient

`v_rel` ist die **Relativgeschwindigkeit**, mit der zwei Partikel aufeinander
treffen. Sie wird nur gebraucht, um daraus die **Stoßenergie** `E_coll` zu
bilden — und die geht als Argument `collision_energy` an den
Porositätswachstums-Kernel:

```
E_coll  →  porosity_growth_kernel.compute_merged_porosity(..., collision_energy=E_coll)
```

Die Idee dahinter: Ein **harter** Stoß drückt das entstehende Agglomerat
zusammen (weniger Poren), ein **weicher** Stoß lässt eine lockere Struktur
stehen (mehr Poren). Die Stoßenergie ist der Parameter, über den ein
Porositätsmodell das unterscheiden könnte.

### Woher `G · d` kommt

Das ist die Standardabschätzung für **scherinduzierte** Relativgeschwindigkeit.
In einer Scherströmung mit Scherrate `G` [1/s] bewegen sich zwei Punkte im
Abstand `d` mit der Geschwindigkeitsdifferenz

```
Δv = G · d
```

Zwei Partikel, deren Mittelpunkte etwa einen Partikeldurchmesser auseinander
liegen, treffen sich also mit ungefähr `G·d`. Dieselbe Logik steckt auch im
Kollisionskern `shear_chin1998` (β ∝ `G·(r₁+r₂)³`) — dort ist sie physikalisch
sauber begründet.

### Warum ich es als Befund geführt habe

Im **selben** Agglomerationsereignis werden zwei verschiedene Werte für
dieselbe physikalische Größe benutzt:

| Verwendung | Wert | Herkunft |
|---|---|---|
| Stoßenergie fürs Porositätswachstum | `v_rel = G · d_eff` | berechnet, größenabhängig |
| Stokes-Kriterium (`St ∝ m·U_coll`) | `U_coll` | **fester Konfigurationsparameter** |

Mit deinen Referenzwerten (`G = 5000 1/s`, `d ≈ 34 µm`):

```
v_rel  = 5000 · 34e-6 = 0.17 m/s
U_coll =                0.01 m/s        →  Faktor 17
```

Und der Unterschied **wächst**: `v_rel` ist proportional zum Durchmesser,
`U_coll` konstant. Bei 200 µm Granulaten wäre `v_rel = 1.0 m/s`, also Faktor 100.

Für einen Fit ist das relevant, weil `U_coll` und `G` damit zwei Stellschrauben
für **denselben** Effekt sind: der Optimierer kann den einen gegen den anderen
ausspielen, ohne dass sich am Ergebnis viel ändert — ein klassisches Zeichen für
Parameterentartung, die den Fit instabil macht.

### Korrektur (18.08.2026)

Auf deine Rückfrage hin habe ich nachgesehen, was die Porositätskernel mit
`collision_energy` machen. Antwort: **nichts.** Damit ist meine ursprüngliche
Schlussfolgerung hinfällig — es gibt *keine* Parameterentartung zwischen
`U_coll` und `G`, weil `v_rel` überhaupt keinen Verbraucher hat:

- `cone_model` (deine Referenz): Docstring sagt wörtlich *"Not used in this model"*
- `volume_mixing`: ebenso
- `incomplete_mixing`: nur unter `energy_dependent=True` — Default `False`,
  wird im Repo nirgends gesetzt
- `blueprint.py` rechnet damit, ist aber die Vorlagendatei und nicht registriert

Und auf der Bruchseite ist `breakage_energy` in `mcpbe_break.py:716` fest auf
`None` gesetzt — einzige Zuweisung im ganzen Modul.

`v_rel` und `E_coll` werden also pro Ereignis berechnet und weggeworfen. Der
Faktor 17 gegenüber `U_coll` ist damit **heute** folgenlos. Neu als
[B-25](Audit_2026-08-17.md#b-25) dokumentiert.

Was bleibt: Der Code *sieht* so aus, als steuere `G` über die Stoßenergie das
Porositätswachstum. Beim Überarbeiten des Scherraten-Konzepts nicht darauf
hereinfallen.

### 18.08.2026: Kanal vollständig entfernt

Deine Entscheidung: *"Entferne es. Es ist ein ehemaliges Missverständnis mit
einem Assistenten."* — `v_rel`, `E_coll`, `collision_energy`, `breakage_energy`
und jeder Parameter/Kwarg/Docstring-Absatz/auskommentierte Beispielblock, der
sich darauf bezog, ist jetzt aus dem gesamten Repository entfernt.

**11 Dateien geändert:**

| Datei | Was entfernt wurde |
|---|---|
| `mcpbe_agg.py` | Berechnung von `rho/d_eff/v_rel/E_coll` in `_merged_porosity`, der `collision_energy=`-Kwarg beim Kernelaufruf |
| `mcpbe_nucleation.py` | dieselbe Berechnung + derselbe Kwarg in `_manual_agglomerate_particles` |
| `kernel_integration.py` | `collision_energy`-Parameter aus `KernelManager.compute_merged_porosity` (Signatur + Weiterreichung) |
| `kernels/base.py` | `collision_energy`/`breakage_energy` aus den abstrakten Basissignaturen (`PorosityGrowthKernel`) |
| `kernels/porosity_growth/volume_mixing.py` | dieselben Parameter aus Signatur + Docstring |
| `kernels/porosity_growth/cone_model.py` | dieselben Parameter aus beiden Methoden (`compute_merged_porosity`, `compute_fragment_porosity` inkl. `_compute_fragment_porosity_multi`-Weiterreichung) |
| `kernels/porosity_growth/incomplete_mixing.py` | `energy_dependent`/`energy_threshold`-Parameter komplett (Defaults, Validierung, Nutzung in beiden Methoden) — das war der einzige Kernel, der den Kanal überhaupt las |
| `kernels/porosity_growth/blueprint.py` (Vorlagendatei) | `collision_energy`/`breakage_energy` aus beiden Signaturen, zwei komplette Docstring-Beispiele ("Energy-Dependent Pore Creation", "Stress-Dependent Pore Collapse"), vier auskommentierte Beispielblöcke, die Hilfsmethode `_estimate_collision_energy` |
| `kernels/aggregation/blueprint.py` (Vorlagendatei) | die ungenutzte Hilfsmethode `_compute_collision_energy` |
| `mcpbe_break.py` | `breakage_energy = None` und beide Weiterreichungen an den Porositätskernel |
| `Trials/test_liq_externalisation_breakage.py` | `breakage_energy=None`-Parameter aus der Test-Double-Kernelklasse (Interface-Konsistenz) |

Dazu vier Stellen in `kernels/KERNEL_UEBERSICHT.md` und
`kernels/porosity_growth/CONE_MODEL_README.md`, die den Kanal in
Code-Beispielen zeigten.

**Verifikation:** Referenzlauf (`test_powerlaw_rumpf_full`) nach der Entfernung
**bit-identisch** zum Lauf davor (t=0,2 s: 114 Ereignisse, agg=5, break=1,
n_comp=2107 — exakt gleich). `x_consistency_probe` und `merger_and_mass_probe`
liefern dieselben Werte wie vor der Entfernung. Das bestätigt, was B-25 schon
zeigte: der Kanal war wirkungslos, seine Entfernung ändert nichts am Verhalten.

**Was NICHT entfernt wurde, bewusst:** `U_coll` (Stokes-Kriterium) und der
Kollisionskern `shear_chin1998` — beide sind reale, wirksame Parameter. Nur der
tote Zweig ist weg. Die von dir separat erwähnte Überarbeitung des
Scherraten-Konzepts (`G`, `U_coll`, `shear_chin1998` vereinheitlichen) ist
davon unberührt und bleibt eine eigene Aufgabe.

---

## B-06 · dW-Asymmetrie — geloest durch Vergleich mit `upstream/dev_monorepo`

Du hattest B-06 zunaechst wie B-03 als offen markiert. Nach deiner Rueckfrage
habe ich es mit `upstream/dev_monorepo` (Commit `02dfac6`, aktuellster
verfuegbarer Stand des Betreuers) abgeglichen.

**Die Codedefaults sind bitgenau identisch** (`agg_dW_max=1.0`,
`break_dW_max=50.0` in `mcpbe_time_helper.py`) -- kein Unterschied zu seiner
Implementierung. Der Fehler lag nicht im Code, sondern in der **Testconfig**:
`test_powerlaw_rumpf_full.py` setzte keines von beiden explizit, beide fielen
also auf die Defaults zurueck -- und das ergibt zufaellig genau die 50:1-Asymmetrie.

**Keines seiner produktiven Validierungsskripte laesst das zu.** Drei
unabhaengige Stellen (`scripts/pbe_validation/validation.py`,
`scripts/pbe_validation/new/validation.py`,
`scripts/pbe_validation/new/reconstruction_monitor.py`) setzen identisch:

```python
solver.break_dW_max = 50.0
solver.agg_dW_min   = 1.0
solver.agg_dW_max   = 20.0
```

Verhaeltnis 2,5:1, durchgehend in dieser Groessenordnung ueber alle seine Skripte
(nie 50:1). `break_dW_min`/`break_dW_mode` tauchen zwar in seinen Skripten auf,
sind aber auch bei ihm toter Code (`_compute_dW` verzweigt nie darauf) -- reines
Rauschen, nicht mit uebernommen.

**Geaendert:** `Trials/test_powerlaw_rumpf_full.py` setzt jetzt
`agg_dW_min=1.0`, `agg_dW_max=20.0`, `break_dW_max=50.0` -- sein Muster,
woertlich. Kein Eingriff in `mcpbe_time_helper.py`/`mcpbe_agg.py`/`mcpbe_break.py`
noetig, da deren Defaults bereits korrekt waren.

Gemessen (500 Partikel, verkuerzte Referenzkonfiguration; `agg_events`/
`break_events` = `real_agg_events`/`real_break_events`, dW-Summen):

```
Alt: agg_events= 74  break_events= 160  a_tot=1023
Neu: agg_events= 80  break_events=1160  a_tot= 695
```

**Diese erste Messung war noch durch B-03 konfundiert** (SIZEEVAL-Filter
unbeabsichtigt aktiv). Mit `SIZEEVAL=0` faellt die agg-dW-Summe um 25 %:

```
                              SIZEEVAL=1 (Bug)   SIZEEVAL=0 (korrekt)
Neu, agg dW-Summe                    80                  60
Neu, break dW-Summe                1160                1020
```

Groessere Agglomerations-Pakete konzentrieren die Population schneller auf
weniger, groessere Partikel, deren (volumenabhaengige) Bruchrate entsprechend
steigt -- die physikalische Kopplung wirkt jetzt mit einer Aufloesung, die dem
Betreuer-Stand entspricht, statt mit einer zufaelligen 50:1-Verzerrung, und ohne
den zusaetzlichen SIZEEVAL-Confound.

**Klarstellung zu deiner Rueckfrage (18.08.2026):** `real_agg_events` und
`real_break_events` sind beide dW-Summen -- identischer Code fuer beide
Prozesse (`mcpbe_base.py:2154/2163`). Im *alten* dW-Setup fielen rohe
Ereigniszahl und dW-Summe fuer Agglomeration nur zufaellig zusammen, weil
`agg_dW_max=1` jedes Paket auf Groesse 1 zwang (roh=74=dW-Summe=74). Fuer
Breakage war das nie so (`break_dW_max=50` erzeugte immer variable Pakete: roh
110 vs. dW-Summe 160, schon im Alt-Setup).

---

## Antworten auf deine Rückfragen

### B-04 · „Macht Aktualisieren Sinn? Braucht das viel Rechenleistung?"

Ja, es macht Sinn, und der Aufwand ist beherrschbar — aber die Frage war genau
richtig gestellt. Was ich gebaut habe, in drei Stufen:

1. **`_key_of`-Rückwärtsabbildung** (`idx → registrierter Schlüssel`).
   Der Kern des Problems war nicht der Aufwand, sondern dass der Schlüssel beim
   Entfernen **neu geraten** wurde. Jetzt wird der Schlüssel gespeichert. Kosten:
   ein Dict-Eintrag pro Partikel. Das macht Entfernen und Umhängen **zuverlässig
   und billiger** als vorher (kein Neuberechnen mehr).

2. **Gezielte Meldung aus den kontinuierlichen Prozessen.**
   Kompression und Internalisierung wissen genau, welche Partikel sie angefasst
   haben, und melden nur diese. Kosten: O(k) für k geänderte Partikel.

3. **Vollabgleich einmal pro MC-Ereignis** (`reindex_all`).
   Weil es noch weitere Schreibstellen gibt, die ich nicht alle einzeln
   nachrüsten wollte. Ein Partikel, dessen gebinnter Schlüssel sich **nicht**
   geändert hat, kostet nur eine Schlüsselberechnung und einen Dict-Vergleich —
   kein Bucket wird angefasst. Nur echte Wechsel schreiben.

**Abschaltbar:** `solver._merger_full_reindex = False` lässt Stufe 3 weg;
Stufe 1 und 2 bleiben. Falls das Profiling zeigt, dass Stufe 3 stört, ist das
der Schalter.

Gemessen nach der Änderung: Hash-Index exakt so groß wie `a_tot`, **0 %**
Einträge unter falschem Schlüssel (vorher bis 99,8 %), Trefferquote 95,7 %.

### B-04 · „Wenn er ausgeschaltet ist, findet auch keine Pflege statt, oder?"

**Ja, korrekt** — und das gilt jetzt auf zwei Ebenen, die vorher vermischt waren:

| Schalter | Wirkung |
|---|---|
| `solver._enable_particle_merging = False` | **Kein Merger.** Jedes neue Partikel bekommt eine eigene Spalte. Null Pflegeaufwand, aber `n_comp` wächst ungebremst. |
| `solver._merger_use_hash_index = False` | Merger **bleibt aktiv**, aber ohne Hash. Die Suche wird ein vektorisierter NumPy-Scan über die aktive Scheibe. **Keinerlei Indexpflege** — jede Pflegefunktion steigt sofort aus. |

Die zweite Zeile ist genau der Fall, den du beschrieben hast: bei kleinem
`n_comp` ist der lineare Scan schneller, weil er als eine einzige
NumPy-Operation läuft, während der Hash-Pfad pro Suche **und pro
Eigenschaftsänderung** Python-Dict-Verkehr kostet.

Beide Schalter müssen **vor** dem ersten `_initialize_samplers()` gesetzt sein —
bzw. `_enable_particle_merging = False` wirkt auch nachträglich, weil die
Abschaltprüfung vor der „existiert schon"-Prüfung steht.

### B-16 · „Lösungsbedarf? Risikoarm?"

Ja, risikoarm und erledigt: `V_dry_old = v_dry_view` ist jetzt
`V_dry_old = v_dry_view.copy()`. Das war eine Zeile.

Der bisherige Code war *zufällig* korrekt (alle abgeleiteten Arrays wurden vor
dem Schreiben berechnet), aber der Name `..._old` für einen Puffer, in den drei
Zeilen später geschrieben wird, ist eine Falle für die nächste Änderung.
Eine Kopie von ~n Floats einmal pro Ereignis ist gegenüber dem Rest der
Kompression nicht messbar.

### B-13 · Bestätigung

Der `compression`-Handler-Zweig ist gelöscht. Deine Einschätzung stimmt: er
wurde mit Einführung des `ContinuousProcessesHandler` zusammengelegt und war
seither unerreichbar (`self.compression` wird nirgends gesetzt).

### B-14 · „Findet die Prüfung woanders statt?"

**Nein.** Es gab überhaupt keine wirksame Prüfung.

- `compression_kernel_name` wird in `KernelManager.__init__` nie gesetzt und ist
  auch kein `__init__`-Parameter. `hasattr(self, ...)` war daher **immer False**
  — die Migrationsanleitung war unerreichbar. Wer den alten Parameter an
  `MCPBESolver(...)` übergab, bekam nur einen nackten `TypeError`.
- **Behoben:** die Prüfung liest den Namen jetzt vom **Solver**
  (`getattr(solver, 'compression_kernel_name', None)`). Genau dort landet er,
  wenn eine alte Config-Datei ihn setzt (`base_solver._load_attributes` weist
  beliebige Attribute zu). Alte Configs bekommen jetzt die Erklärung, statt
  stillschweigend ohne Kompression zu laufen.
- Die drei anderen `hasattr`-Prüfungen waren immer **True** (Attribute werden in
  `__init__` gesetzt) und damit reines Rauschen — entfernt.

---

## Zu den Parameter-Befunden (B-02, B-08, B-20)

Du hast recht: Absolutwerte sind hinfällig, weil am Ende gegen Experimentaldaten
gefittet wird. Ich habe die **inhaltlichen Anmerkungen zu Zahlenwerten** deshalb
aus der Bewertung genommen. Was **nicht** hinfällig ist und bleiben sollte:

1. **`Vc` ist ein physikalischer Parameter, kein numerischer.** Er legt über
   `n = ΣW/Vc` die Konzentration fest. Wichtiger noch: die Agglomerationsrate
   skaliert mit `1/Vc`, die **Bruchrate gar nicht**. Ein „beliebiges" `Vc`
   verschiebt also das **Verhältnis** der beiden konkurrierenden Prozesse — und
   das kann kein Fit über die Kernelparameter sauber ausgleichen, weil es zwei
   Prozesse unterschiedlich trifft. `Vc` sollte aus Feststoffmasse, Dichte und
   Mischervolumen kommen.

2. **`k_int` ist nur relativ zum Tropfenvolumen sinnvoll.** τ ≈ 1/(k_int · L)
   mit L = Flüssigkeit pro Partikel. Der Kommentar „typical: 1e10 – 1e14" in
   `ContinuousProcessesConfig` ist als **absolute** Angabe irreführend — er gilt
   nur für ein bestimmtes L. Beim Fitten ist das ein Parameter, dessen sinnvoller
   Bereich sich mit der Tropfengröße mitverschiebt.

3. **B-20 ist kein reines Tuning-Problem.** Dass bei ε₀ = 0,6
   *null* Agglomerationen bei 119 678 Brüchen auftreten, liegt auch daran, dass
   `powerlaw_rumpf` bei S = 0 nur den trockenen, schwachen Ast kennt — und die
   Sättigung bleibt 0, solange die Internalisierung nicht greift. Es sind also
   drei gekoppelte Parameter (`Vc`, `k_int`, Bruchvorfaktor). Nach den jetzt
   erfolgten Fixes (insbesondere B-01: β war um Faktor 3 zu klein) sollte das
   Gleichgewicht neu vermessen werden, bevor daran gedreht wird.

---

## Geänderte Dateien

```
mcpbe/src/wmcpbe/mcpbe_base.py                    set_particle_dry_volume, sync_particle_diameters,
                                                  _ensure_particle_merger, Vc-Doubling, solve()
mcpbe/src/wmcpbe/mcpbe_agg.py                     compute_merged_liquid, _merge_pair, _merge_liquid entfernt
mcpbe/src/wmcpbe/mcpbe_break.py                   V_dry im Fallback, _last_break_dW, Zaehler, Merger-Erzeugung
mcpbe/src/wmcpbe/mcpbe_nucleation.py              set_particle_dry_volume an 2 Stellen
mcpbe/src/wmcpbe/mcpbe_continuous_processes.py    Porositaet+V_dry gemeinsam, X-Sync, Merger-Meldung, .copy()
mcpbe/src/wmcpbe/particle_merger.py               _key_of, _unlink, reindex_particles, reindex_all
mcpbe/src/wmcpbe/kernel_integration.py            Legacy-Warnung erreichbar, hasattr-Rauschen entfernt
```

**Nicht angefasst:** die Trial-Skripte und ihre Parameterwerte.
