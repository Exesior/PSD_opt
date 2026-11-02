# Sensitivitätsstudie WMCPBE-Nassgranulation

Ziel: herausfinden, **welche Parameter die simulierte Partikelgrößenverteilung wie
stark verändern**, mit Schwerpunkt auf dem Bruch. Die Methode lehnt sich an das
Betreuer-Skript `scripts/optframework_post/global_sensitivity_analysis.py` an:
Sobol-Indizes mit SALib (Saltelli-Stichprobe, S1 / ST / optional S2,
Bootstrap-Konfidenzintervalle, log-Parameter im Exponenten gezogen).

> **Zuerst lesen:** `PARAMETERLISTE.md` (alle Kandidaten, Einstufung, Messwerte). Stand: ganz unten.

## Dateien

| Datei | Zweck |
|---|---|
| `start_1_test.py` … `start_5_auswerten.py` | **Spyder-Startdateien:** öffnen, F5. Je Datei genau ein Schritt (Abschnitt „Auf dem Remote-Rechner“) |
| `sa_config_gesamt.py` | **Die Studie:** 18 Parameter, N, S2, Ergebnisordner |
| `sa_gemeinsam.py` | Für alle Studien gleich: Basisfall (id 1070, N_COMP 125), Seeds, Zeitlimit, Kennzahlen, Versuchsdaten |
| `PARAMETERLISTE.md` | Alle potenziell interessanten Parameter mit Einstufung, Bereich, Quelle (MA Tab. 3.1) und Einzeltest |
| `run_sensitivity.py` | Rechnen (Spyder F5). Fortsetzbar nach Abbruch, Fortschritt in Konsole + `progress.log` |
| `analyze_sensitivity.py` | Auswerten (Spyder F5): Sobol, Effektgrößen, Rauschen, Markierungen, Excel, Grafiken, `bericht.md` |
| `sa_core.py` | Ein Replikat bauen/rechnen/beobachten/speichern (Solver über den echten Fit-Adapter) |
| `sa_psd.py` | Verteilungs-Mathematik (Q3 an Maschen, Quantile, Verschiebung in Dekaden) |
| `sa_plots.py` | Grafiken der Auswertung |
| `test_synthetisch.py` | Test der ganzen Kette ohne Simulation (Ersatzmodell mit bekannten Sobol-Indizes) |
| `kalibrierung/` | Laptop-Messungen, mit denen Basisfall, Beschleunigung und Replikatzahl festgelegt wurden |

Es gibt **keine Änderung** an `mcpbe`, am Adapter oder am Optimizer. Der Solver wird
über `WetGranulationAdapter._build_solver` gebaut, also mit derselben
Kernel-Verdrahtung wie im KF3-Fit. `code_referenz.json` hält fest, mit welchem Stand
dieser Dateien kalibriert wurde (siehe „Auf dem Remote-Rechner“, Schritt 0).

## Auf dem Remote-Rechner

Voraussetzung: dieselbe Python-Umgebung wie für den Fit (`wmcpbe`, `pbe_optimizer`
importierbar) plus **SALib** (`pip install SALib`, getestet mit 1.6.0).
Den ganzen Ordner `studies/sensitivity/` kopieren (er braucht nur die installierten
Pakete, keine Pfade zum Repo). `kalibrierung/runs/`, `test_ausgabe_synthetisch/` und
`__pycache__/` müssen nicht mit.

0. **Gleicher Simulationscode wie auf dem Laptop.** Die Studie wurde mit dem Arbeitsstand
   von `mcpbe/` und `pbe-optimizer/` auf dem Laptop kalibriert. Dazu gehören nicht
   committete Änderungen und der Adapter `wet_granulation_adapter.py`, der nicht in Git
   ist. Auf den Remote-Rechner kommt er also entweder per Commit + Pull oder als Kopie der
   beiden Ordner.
   - `--check` vergleicht jede Solver-Datei (Paket `wmcpbe` ohne `Trials/`, Adapter samt
     Basisklasse) mit `code_referenz.json` aus diesem Ordner. Bei Abweichung meldet es
     `CHECK NICHT OK` mit den betroffenen Dateien.
   - Zusätzlich zeigt es, ob ein kurzer Testlauf (200 Events) dasselbe Ergebnis liefert
     wie auf dem Laptop. Bei gleichem Code und anderer numpy-/numba-Version kann das
     abweichen; die Studie ist dann statistisch, aber nicht bitgleich vergleichbar.
   - Der Runner legt bei jedem Start `code_stand.json` in den Ergebnisordner und warnt,
     wenn sich der Code zwischen zwei Starts ändert.
   - Nur auf dem Laptop und nur nach einer gewollten Code-Änderung:
     `--check --make-reference` schreibt die Referenz neu.

Alles läuft in **Spyder**: die jeweilige Startdatei öffnen und F5 drücken. Nichts muss
an Argumenten geändert werden. (Getestet mit einem echten Spyder-Kernel, Spyder 6.1, auch mit
den parallelen Prozessen. Voraussetzung ist Spyders Standard-Einstellung „Arbeitsverzeichnis =
Ordner der ausgeführten Datei“.)

1. **`start_1_test.py`** (≈ 3–5 min): Test der ganzen Kette ohne Simulation. Am Ende muss
   `TEST OK` stehen. Geprüft werden Design, Speicherformat, Auswertung, Grafiken, fehlende
   und ausgefallene Punkte, Fortsetzen, Design-Verlängerung und der Absturz eines
   Arbeitsprozesses.
2. **`start_2_check.py`** (Sekunden): Importe und SALib, Solver bauen, 200 Events, Code-Stand
   gegen den Laptop (Schritt 0). Endet mit `CHECK OK`.
3. **`start_3_benchmark.py`** (≈ 10 min): rechnet dieselben 36 Läufe des Basispunkts mit 6,
   9 und 12 Prozessen und empfiehlt die Prozesszahl. Die Empfehlung in `sa_gemeinsam.py`
   bei `WORKERS` eintragen. Das Ergebnis steht auch in `<Ergebnisordner>/benchmark/ergebnis.txt`;
   die Benchmark-Läufe zählen nicht zur Studie.
4. **`start_4_studie.py`:** Studie starten bzw. fortsetzen. Zuerst kommt der Plan
   (Punkte, Läufe, Schätzung), dann eine Zeile pro fertigem Lauf, z. B.

   ```
   [2026-10-18 14:03:12] Punkt   283: 3/5 fertig (52 s, 9123 Events) | gesamt 1415/7040 (20.1 %) | Rest ~14.2 h
   [2026-10-18 14:03:40] Punkt   283: 5/5 abgeschlossen
   ```

   Dieselben Zeilen stehen in `<Ergebnisordner>/progress.log`, also auch von außen
   prüfbar. Kaputte Läufe stehen zusätzlich in `failures.log`.
   **Zeitlimit:** Zuerst rechnet der Basispunkt (ohne Limit). Danach gilt für jeden Lauf
   `TIME_LIMIT_FACTOR` × die Median-Laufzeit des Basispunkts auf diesem Rechner. Das passt
   sich also von selbst an die Rechnergeschwindigkeit an und steht in `time_limit.json`.
   Läufe, die es reißen, haben den Status `timeout`.
   - **Fehlende Punkte werden nicht aufgefüllt** (anders als die KNN-Idee im
     Betreuer-Skript). Ausfälle häufen sich dort, wo die Raten hoch sind; aufgefüllte
     Werte würden genau den verursachenden Parameter kleinrechnen.
   - Stattdessen fällt der **ganze Saltelli-Block** weg, in dem ein Punkt nicht alle Seeds
     mit Status `ok` hat („ausgefallen“). SALib rechnet mit den restlichen Blöcken.
   - Sind mehr als 10 % der Blöcke ausgefallen, gibt es keine Sobol-Indizes. Die Grenze lässt
     sich mit `--max-dropped` ändern.
   - Noch nicht (fertig) gerechnete Blöcke zählen dabei **nicht** als ausgefallen, sondern als
     „offen/in Arbeit“.
   - Das ist streng: Ein Block hat in der Gesamt-Config 20 Punkte × 5 Seeds = 100 Läufe. Schon
     ~0,1 % fehlgeschlagene Läufe reichen für 10 % ausgefallene Blöcke. Die Parameterbereiche
     sind deshalb so gewählt, dass in Kalibrierung und Mini-Studie nichts ausgefallen ist.
   - Wo Ausfälle liegen, zeigt Blatt `Diskretisierung`, Spalte `share_failed`.
   - Ein Timeout wird beim nächsten Start neu gerechnet, wenn das Limit inzwischen größer
     ist (`TIME_LIMIT_FACTOR` erhöhen). Läufe mit Status `failed` oder `crashed` bleiben
     stehen; zum Wiederholen die Datei löschen.
   **Abbrechen / Fortsetzen:** Konsole stoppen (oder Rechner neu starten). Jeder fertige
   Lauf liegt sofort als eigene Datei in `<Ergebnisordner>/runs/`. Beim nächsten F5 auf
   `start_4_studie.py` werden fertige Läufe erkannt und übersprungen. Liegt im
   Ergebnisordner ein Design einer **anderen** Config, bricht das Skript ab, statt zu mischen.
   - Spyder: Der rote Stopp-Knopf trifft nur die Konsole. Bereits laufende
     Arbeitsprozesse rechnen ihren aktuellen Lauf noch zu Ende. Nach „Kernel neu starten“
     können `python.exe`-Prozesse übrig bleiben. Dann im Task-Manager beenden, sonst
     rechnen sie mit und verfälschen die Laufzeiten.
   - Stirbt ein Arbeitsprozess hart (z. B. Speicher), reißt er alle gerade laufenden Läufe
     mit. Das Skript baut einen neuen Pool und startet sie neu. Wer dreimal dabei war,
     rechnet am Ende allein: Stirbt er auch allein, ist er der Verursacher und wird mit
     Status `crashed` gespeichert (sein Block gilt als ausgefallen). Die übrigen rechnen
     normal (getestet). Die Meldung `OSError: handle is closed` danach ist harmlos.
5. **`start_5_auswerten.py`:** Ergebnisse stehen in `<Ergebnisordner>/analysis/`.
   - **Das geht auch zwischendurch:** Ausgewertet werden die fertigen Blöcke, das entspricht
     einer Studie mit kleinerem N. Der Bericht kennzeichnet das als „ZWISCHENSTAND“.
   - Unter 8 fertigen Blöcken gibt es noch keine Sobol-Indizes, nur Punkt-Kennzahlen und
     Markierungen.
   - Wurde `N_REPLICATES` erhöht und die neuen Seeds laufen noch: `--seeds 5` wertet mit den
     alten Seeds aus. Dazu in `start_5_auswerten.py` die Liste um `"--seeds", "5"` ergänzen.

## Was pro Lauf gespeichert wird

Eine `.npz` je Lauf (Parameterpunkt × Seed), Name `p<punkt>_r<replikat>.npz`:

- `meta_json`: alle Parameter, Seed, Status (`ok`, `timeout`, `particle_limit`,
  `failed`, `incomplete`, `crashed`), Wandzeit, Fehlertext. Nicht-`ok`-Läufe enthalten nur
  diese Metadaten.
- **Die ganze Verteilung:** die komplette Partikelliste bei 360 s und 600 s
  (`t360_V`, `t360_W`, `t360_poro`, `t360_sat`, `t360_liq`, ebenso `t600_*`).
  `V` = (Feststoffvolumen, Hüllvolumen) je Rechenpartikel, `W` = Gewicht.
- Q3-Kurven auf einem feinen log-Raster (10 µm … 5 mm, 241 Punkte) für **jeden**
  Schnappschuss (alle 30 s), dazu n_comp, Vc und die physikalischen Agglomerations- und
  Bruchzähler je Schnappschuss.
- Event-Zähler in 5-s-Fenstern (gesamt / akzeptierte Agglomeration / Bruch) und
  alle 2 s die gemessenen Prozessraten (siehe Diskretisierung).

Alle Kennzahlen rechnet erst die Auswertung aus diesen Rohdaten. Eine neue Kennzahl
braucht also keinen neuen Lauf. Platzbedarf: ~25 kB je Lauf, also ~0,5 GB für 20 000
Läufe. Ergebnisordner sind per `.gitignore` vom Repo ausgenommen.

## Das Maß: woran „wie stark“ festgemacht wird

Für jede Kennzahl Y (z. B. Q3 an einer Masche, log10 d50, Verschiebung) und jeden
Parameter:

1. **Sobol S1 / ST** (Anteil an der Streuung von Y, 0…1). S1 ist der Parameter allein,
   ST schließt alle Wechselwirkungen ein; ST − S1 ist also der Wechselwirkungsanteil.
   95-%-Intervalle per Bootstrap.
2. **Absolute Wirkung:** σ_ST = √(ST · Var(Y)) ist die Streuung von Y, die der Parameter
   verursacht, in der **Einheit von Y**. Sie wird verglichen mit
   - der **Wiederholstreuung des Versuchs** (AJ26/AJ29/AJ31 bei 600 s, gleiche
     Bedingungen): „Das Bruch-Niveau verschiebt Q3(250 µm) um das 4-Fache dessen, was der
     Versuch zwischen Wiederholungen streut“. Das ist die eigentliche Aussage:
     **Ist der Einfluss größer als das, was man im Experiment überhaupt auflösen kann?**
   - dem **MC-Rauschen** des Punktmittels (Standardfehler über die Seeds).
3. **Rauschanteil** = mittleres SE² / Var(Y). Liegt er über `NOISE_SHARE_LIMIT`, ist das
   Ergebnis für diese Kennzahl nicht belastbar → `N_REPLICATES` erhöhen. Die schon
   gerechneten Seeds bleiben gültig, denn die Seed-Liste ist präfixstabil.
4. **Verschiebung der Verteilung in Dekaden:** die Fläche zwischen der Q3-Kurve des
   Punkts und der des Basispunkts auf der log10-Größenachse (Wasserstein-1-Abstand in
   log x). Vorzeichenbehaftet heißt das: Wie viel gröber oder feiner ist die
   Verteilung im Mittel? 0,1 Dekaden ≈ 26 % Größenänderung.

Alle Punkte laufen mit **denselben Seeds** (Common Random Numbers). Unterschiede zwischen
Punkten kommen deshalb von den Parametern, nicht vom Zufallsgenerator.

## So liest man die Auswertung (`analysis/`)

| Datei | Was drinsteht |
|---|---|
| `bericht.md` | Kurzfassung: Läufe/Ausfälle, Markierungen, Rauschwarnungen, je Kennzahl die Parameter nach ST sortiert mit σ_ST und „× Versuch“ |
| `sensitivitaet_<studie>.xlsx` | Alle Tabellen: Sobol (S1/ST/S2 mit Intervallen), Rauschen, Haupteffekte, Diskretisierung/Ausfälle je Parameterbereich, Punkte (Spalte `zustand`: fertig / ausgefallen / in Arbeit / offen), einzelne Läufe. Blatt `Info` nennt das tatsächlich ausgewertete N |
| `1_sobol_indizes.png` | Balken S1 (hell) und ST (dunkel) je Parameter für die wichtigsten Kennzahlen |
| `2a_karte_ST.png` | Übersicht ST: Parameter × alle Kennzahlen |
| `2b_karte_effekt_vs_versuch.png` | **Die Kernaussage:** zurechenbare Streuung in Vielfachen der Versuchs-Wiederholstreuung. Ein Wert < 1 heißt, dass der Parameter weniger bewegt, als der Versuch ohnehin streut |
| `3_haupteffekte.png` | Kennzahl gegen jeden Parameter: Punkte, Mittel je Zehntel des Bereichs, Basispunkt, ±1 Versuchsstreuung |
| `4_verteilungen_je_parameter.png` | Q3-Kurven bei 600 s: Basis, Mittel des untersten/obersten Zehntels jedes Parameters, Hülle aller Punkte, Versuchspunkte. Erst bei vielen Punkten aussagekräftig: Bei einer Zwischenauswertung mit kleinem N mitteln sich die übrigen Parameter in einem Zehntel noch nicht heraus |
| `5_diskretisierung.png` | Verteilung des Events/s-Verhältnisses und wo im Parameterraum markiert wird |
| `6_rauschen.png` | Rauschanteil je Kennzahl |

Faustregeln:
- ST − S1 groß heißt, dass der Parameter vor allem über Wechselwirkungen wirkt.
- ST ≈ 0 (Intervall schließt 0 ein) heißt, dass der Parameter im untersuchten Bereich
  nichts bewirkt.
- Bei hohem Rauschanteil sind die Indizes dieser Kennzahl unsicher. Die stetigen Maße
  (log d10/d50/d90, Verschiebung) sind belastbarer als einzelne Maschenwerte.

## Diskretisierung (Markierung, kein Verbot)

Zwischen zwei MC-Events friert der Solver die Propensitäten ein. Die kontinuierlichen
Prozesse laufen danach über das ganze Zeitstück. Sind sie schneller als die
Event-Folge, ändert sich der Zustand zwischen zwei Events stark, und es entsteht ein
Diskretisierungsfehler. Die Bedingung (Vorgabe Eric):

  **mittlere Events/s ≥ 2 × max(k_comp, λ_int)**

- **Events/s:** alle MC-Events (Agglomeration akzeptiert oder abgelehnt, Bruch) geteilt
  durch die Simulationszeit.
- **k_comp:** die tatsächliche Kompressionsrate, `COMPRESSION_RATE · N_MIXER^COMPRESSION_C_MIXER` [1/s].
- **λ_int:** die tatsächliche Internalisierungsrate. Für jedes Partikel ist das die lokale
  Relaxationsrate der Internalisierungs-ODE dl/dt = k_int (L − l)(V_pore − l), also
  λ = k_int · ((V_pore − l) + (L − l)) [1/s]. Gemittelt wird über die Masse der
  Partikel, an denen der Prozess gerade wirklich arbeitet (freie Pore **und**
  Außenflüssigkeit). Das wird alle 2 s gemessen und über die Zeit gemittelt, in der
  Internalisierung stattfindet. Das Maximum über die Partikel wird zusätzlich
  gespeichert.

Läufe, die die Bedingung verfehlen, werden markiert (`disc_flag`). Die Auswertung
zeigt, in welchem Teil des Parameterraums das passiert (`5_diskretisierung.png`).
Zusätzlich gespeichert ist das kleinste Verhältnis in einem 5-s-Fenster
(`disc_ratio_local_min`, nur Information).

## Basisfall

KF3-Fit, Bestparameter **id 1070** (2100.sqlite, J = 0,0286). Die **einzige** Änderung
gegenüber dem Fit: **N_COMP 500 → 125** (W_init 3,19 → 12,75). Eric hat das am 18.10.
bestätigt. Box, Tropfen (150 µm, 6 pro Box), Zugabe (60 s) und alle Kernel-Parameter sind
wie im Fit. Gemessen in `kalibrierung/README_kalibrierung.md`:

- Dieselbe Verteilung wie mit 500 Rechenpartikeln: d50 257 gegen 248 µm, Verschiebung
  +0,008 Dekaden. Das gilt auch bei aktivem Bruch.
- Laufzeit je Replikat auf dem Laptop: 23 s statt 578 s.
- Diskretisierungs-Kriterium erfüllt: 8,7 Events/s gegen 2 × k_comp = 2,75/s.

## Parameter

Alle Kandidaten mit Einstufung, Bereich, Quelle und Einzeltest stehen in **`PARAMETERLISTE.md`**.
Kurzfassung der 18 Parameter in `sa_config_gesamt.py`:

- **Agglomeration:** AGG_AREF, AGG_A, STOKES_H_A, STOKES_U_COLL_REF, CONE_K_AGG
- **Bruch:** BREAK_SREF, BREAK_P2, RUMPF_ALPHA, RUMPF_K, CONTACT_ANGLE, CONE_K_BREAK
- **Porosität und Flüssigkeit:** MIN_POROSITY, COMPRESSION_RATE, LIQUID_INTERNALIZATION_RATE,
  INITIAL_POROSITY (ε0 der Primärpartikel, 0 … 0,8; Vorgabe Eric)
- **Nucleation:** DROPLET_DIAMETER (Box wächst über 150 µm mit)
- **Numerik:** AGG_DW_REL (dW Agg relativ zu W_init), NUCLEATION_BATCH_SIZE

Vier Parameter werden nicht roh, sondern umgerechnet gezogen (`kind` in der Config, Code in
`run_sensitivity.to_params`). Grund ist jeweils: Roh gezogen würden sie Laufzeit oder
Deutung verfälschen.

| Gezogen wird | Daraus gesetzt | Grund |
|---|---|---|
| `AGG_AREF` = log10(β_fit · r_ref^(−1,5·c_temp)), r_ref = 17 µm | `EKE_CORR_BETA` | Im Fit liegen β_fit und c_temp auf einer Linie (Korrelation −0,999). Unabhängig gezogen läge fast jeder Punkt dekadenweit daneben |
| `BREAK_SREF` = log10(P1 · V_ref^P2), d_ref = 300 µm | `BREAK_P1` | P2 dreht sonst nicht nur die Größenabhängigkeit, sondern verschiebt das Bruchniveau um Dekaden |
| `DROPLET_DIAMETER` | Box, W_init, Flussrate, dW Agg (× (d/150 µm)³ über 150 µm) | Es bleiben immer ≥ 6 Tropfen in der Box. Die Boxgröße selbst ändert nichts |
| `AGG_DW_REL` | `AGG_DW_MAX` = Wert · W_init | Absolut würde eine größere Box die Eventzahl vervielfachen (gemessen bis ~80×) |

**Bruch-Niveau bewusst über dem Fit:** Im Fit-Optimum (log10 S_ref = −2,44) bricht nichts.
Der Bereich 0 … 2 reicht von „praktisch wie der Fit“ (bei +1 nur 4 Events, Verschiebung ≈ 0)
bis „gesättigt“. Der Basispunkt liegt daher als Referenz außerhalb des Bereichs. Alle anderen
Parameter werden mit wählbar starkem Bruch bewertet.

**Wo die Grenzen herkommen:** Bei starkem Bruch explodiert die Zahl der Rechenpartikel,
und der Lauf reißt das Zeitlimit. In der Studie fehlt dann ein Punkt, und sein ganzer
Saltelli-Block wird verworfen. In einer Stichprobe aus 128 echten Design-Punkten der
endgültigen Config brauchten 2 Punkte (starker Bruch + schwache Festigkeit) ~22–24× die
Basis-Laufzeit, liefen aber durch. Das Zeitlimit liegt deshalb bei **40×**
(`TIME_LIMIT_FACTOR` in `sa_gemeinsam.py`); mit 20× wäre rechnerisch ein Drittel der
Blöcke ausgefallen. Einzelne Ecken (alles festigkeitsmindernd + stärkster Bruch + ε0 0,8 +
kleine Tropfen) reißen auch 40×; sie sind so selten, dass sie die 10-%-Grenze nicht
gefährden sollten. Details: `kalibrierung/README_kalibrierung.md`, Abschnitte 12–14.

## Laufzeit

Gemessen auf dem Laptop:

| Messung | Ergebnis |
|---|---|
| Mini-Studie Gesamt-Config (72 Punkte quer durch 16 Parameter, damals noch ohne U_ref, echter Runner) | im Mittel **1,17 × die Basis-Laufzeit**, max. ~4×, keine Timeouts, keine Markierung |
| Stichprobe 128 Punkte der endgültigen Config (18 Parameter) | Median **1,3 ×**, Mittel **1,9 ×** die Basis-Laufzeit; 2 Punkte ~22–24 × |
| Benchmark 4 gegen 8 Prozesse (Laptop, 4 Kerne / 8 logische), 16 gleiche Läufe | 587 gegen **640 Läufe/h** (+9 %): Hyperthreading halbiert fast die Geschwindigkeit je Prozess. Mit Faktor 1,9: N = 256 (25 620 Läufe) ~76 h, N = 128 ~38 h |

Der Remote-Rechner (6 schnellere Kerne / 12 logische) liegt darunter, grob geschätzt
**~40–70 h für N = 256** und **~20–35 h für N = 128**. `start_3_benchmark.py` gibt dort die
echte Hochrechnung aus (mit dem gemessenen Faktor 1,9 für den Parameterraum). Weil die
Blöcke der Reihe nach entstehen, ist das Ergebnis mit N = 128 nach der halben Zeit fertig
auswertbar; danach entscheidest du, ob es bis N = 256 weiterläuft.
- **Prozesszahl bei 6 Kernen / 12 logischen Prozessoren:** `start_3_benchmark.py` misst
  6, 9 und 12 Prozesse mit denselben Läufen und empfiehlt den Wert für `WORKERS`
  (`sa_gemeinsam.py`).
- **Zwischenstand nach der halben Zeit:** Die Blöcke entstehen der Reihe nach.
  `analyze_sensitivity.py` wertet jederzeit die fertigen Blöcke aus („ZWISCHENSTAND“). Nach
  der halben Rechenzeit ist das ein vollwertiges Ergebnis mit N = 128.
- **N später erhöhen geht:** Das Saltelli-Design ist präfix-stabil (das Design mit 2N enthält
  das mit N als erste Blöcke, geprüft). `N_SALTELLI` hochsetzen und F5 drücken: Gerechnete
  Läufe bleiben gültig. `CALC_SECOND_ORDER` dagegen nicht nachträglich ändern (anderes
  Design; das Skript bricht dann ab).
- **Rauschen und Seeds:** Mit 5 Seeds liegt der Standardfehler der stetigen Maße bei
  ≈ 0,005–0,02 Dekaden. Die Effekte reichen bis ~0,3 Dekaden, die Versuchsstreuung von d50
  liegt bei ≈ 0,035. Die Maschen-Werte Q3(100/250 µm) rauschen stärker, weil d50 genau auf
  der 250-µm-Masche liegt. Die Auswertung meldet den Rauschanteil je Kennzahl. Sind die
  Hauptkennzahlen betroffen: `N_REPLICATES` erhöhen, die Seeds sind präfix-stabil.

## Stand / Weiterarbeiten

Stand 2026-10-18, spätabends. Erics Entscheidungen sind umgesetzt,
auch Frage 7 (Verdichtung poröser Primärpartikel: so lassen). Startklar.

- **Neu am 18.10. abends (nach Erics Antworten):**
  - Spyder-Startdateien `start_1_test.py` … `start_5_auswerten.py`; `start_1` und `start_2`
    in einem echten Spyder-Kernel (spyder_kernels 3.1, `%runfile … --wdir` wie F5) getestet,
    inklusive Prozess-Pool und Absturz-Test.
  - ε0 mit fester Feststoffmasse (`porosity_mass_fixed`), Tropfen 50 … 225 µm,
    `sa_config_bruch.py` gelöscht, Benchmark für 6 / 9 / 12 Prozesse.
  - Befund: Die Kompaktierung verdichtet auch trockene poröse Primärpartikel (FRAGEN Nr. 7;
    Eric: so lassen).
- **Neu am 18.10. abends (davor):**
  - `INITIAL_POROSITY` (ε0, 0 … 0,8) auf Erics Wunsch in der Studie: jetzt 18 Parameter,
    5 120 Punkte, 25 620 Läufe. Messungen dazu: `kalibrierung/README_kalibrierung.md`,
    Abschnitt 12.
  - `TIME_LIMIT_FACTOR` 10 → 20, weil die Extrem-Ecke mit ε0 = 0,8 bis ~12,8× braucht.
  - Code-Abgleich: `code_referenz.json` (61 Solver-Dateien + Testlauf) wird von `--check`
    geprüft; der Runner legt `code_stand.json` in den Ergebnisordner.
- **Fertig und getestet:**
  - Synthetischer Test mit bekannten Sobol-Indizes (S1 ≈ 0,91 / 0,12 / 0, erwartet
    0,90 / 0,10 / 0). Er stellt alle drei Block-Zustände nach: ausgefallen (Timeouts),
    in Arbeit (halber Block, ein Punkt mit nur einem Seed) und offen (die letzten 40
    Blöcke). Geprüft wird, dass Sobol genau die fertigen Blöcke nutzt, nur ausgefallene
    gegen die 10-%-Grenze zählen und der Bericht „ZWISCHENSTAND“ meldet. Dazu kommen
    halb geschriebene Dateien, Design-Verlängerung und Zeitlimit-Logik.
  - Mini-Studie Bruch (76 Läufe) und Mini-Studie Gesamt (148 Läufe, damals 16 Parameter) mit
    dem echten Runner, jeweils mit Auswertung, zuletzt erneut ausgewertet mit dem
    Analyse-Code vom 18.10. Fortsetzen nach Abbruch ist getestet (die Gesamt-Mini-Studie
    wurde nach einem behobenen Fehler fortgesetzt).
  - Absturz von Arbeitsprozessen (im synthetischen Test, echter Prozess-Pool): Ein Lauf, der
    seinen Prozess jedes Mal umbringt, landet als `crashed`; alle mit ihm abgestürzten Läufe
    werden normal fertig.
  - Fortsetzen ohne `time_limit.json` mit alten Timeouts (synthetischer Test, 1. Ordnung wie
    die Produktions-Config): Nach „Fertig“ ist jeder Lauf ok, nichts wird abgeschnitten.
  - `--check` (18 Parameter, 5 120 Punkte) und `--dry-run` (schreibt nichts) laufen mit der
    endgültigen Config; `--benchmark` ist getestet.
  - Vier unabhängige Code-Prüfungen durch Agenten; alle echten Befunde sind behoben.
    - Aus der dritten: Block-Zustände, effektives N im Bericht, exakter Basispunkt, Schutz
      gegen Tippfehler in der Config, Dry-Run ohne Schreiben.
    - Aus der vierten: Isolation abgestürzter Läufe, kein Abschneiden beim Fortsetzen,
      Schutz gegen Basiswert-0-Abweichungen und gegen überschriebene Ziele, Rauschanteil
      über dieselben Punkte wie Sobol.
- **Inhaltliche Befunde** (für die MA):
  - Bruch macht die Verteilung nur feiner (bis −0,27 Dekaden). Grobgut über ~400 µm kann er
    nicht erzeugen.
  - Das Stokes-Kriterium wirkt in der Basis praktisch nur über die Filmdicke (h > h_a):
    St 1·10⁻³ … 3·10⁻², St* 2,6 … 4,8, kleinster Abstand ~60×. Deshalb sind U_ref und ρ_eff
    einzeln ohne Wirkung. Bei h_a = 1 µm schrumpft der Abstand auf ~16×, darum bleibt U_ref
    (steht auch für μ_b) in der Studie.
  - Die Batchsize ist ohne Wirkung, weil pro MC-Schritt < 1 Tropfen fällig ist. dW Bruch ist
    ohne Wirkung, weil die Granulat-Rechenpartikel W ≈ 0,25 (max. ~0,5) ≪ 1 haben.
  - Die Tropfengröße setzt die Granulatgröße. Die Boxgröße ist ohne Einfluss.
- **Nicht gemacht:**
  - Die volle Studie (nicht auf dem Laptop).
  - Der Benchmark auf dem Remote-Rechner selbst (`start_3_benchmark.py`, dort ~10 min).
- **Nächste Schritte:**
  1. `mcpbe/` und `pbe-optimizer/` auf dem Remote-Rechner auf den Laptop-Stand bringen
     (Schritt 0 oben).
  2. Ordner `studies/sensitivity/` auf den Remote-Rechner.
  3. In Spyder nacheinander `start_1_test.py` (`TEST OK`), `start_2_check.py` (`CHECK OK`),
     `start_3_benchmark.py` (Empfehlung in `WORKERS` eintragen), `start_4_studie.py`.
  4. Zwischendurch und am Ende `start_5_auswerten.py`.
