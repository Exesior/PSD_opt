# Kalibrierung (Laptop, i7-1065G7, 4 Kerne / 8 Threads)

Zweck: Basisfall, Beschleunigung und Replikatzahl der Sensitivitätsstudie festlegen.
Ausgangspunkt ist immer die KF3-Config plus die Bestparameter id 1070
(`kalib_run.py::base_params`). Jede Variante sind Überschreibungen davon.

```
python kalib_run.py --variant <name> --reps 4 --workers 4      # rechnet, schreibt runs/<name>/
python kalib_summary.py [<name> ...]                          # eine Zusammenfassung je Variante
```
(Beide Skripte brauchen den Ordner `studies/sensitivity` auf dem Python-Pfad. In der Konsole
geht das mit `PYTHONPATH=..`, in Spyder mit „Arbeitsverzeichnis = studies/sensitivity“.)

Variablen-Konvention bei Box/Tropfen-Änderungen (`scaled()`): Die Binder-Menge pro
Mischervolumen bleibt 0,106 (wie im Versuch). Bei kürzerer Zugabe steigt also die
Flussrate. `W_init = f · N_real / N_comp`.

## 1. Basisfall „orig“ (id 1070 genau wie im Fit)

4 Replikate, 4 parallel, numba-Standard-Threads:

| Größe | Wert |
|---|---|
| Laufzeit je Replikat | 578 ± 42 s (Server im Fit: ~123 s je Replikat) |
| Events | 16 272 = 27,1 /s |
| k_comp / λ_int | 1,37 /s / 0,02 /s → Verhältnis 19,7 (Kriterium ≥ 2 erfüllt) |
| Bruch-Events | **0** |
| J_KF3 (4 Seeds) | 0,0351 (Fit mit 10 Seeds: 0,0286), Streuung je Replikat 0,013 |
| d10/d50/d90 bei 600 s | 223 / 248 / 305 µm |
| Q3(250 µm) bei 600 s | 0,53 ± **0,23** je Replikat (Versuch: 0,42 ± 0,06) |
| log10 d50 je Replikat | ± 0,029 Dekaden; Verschiebung ± 0,013 Dekaden |

**Warum die Masche so stark rauscht:** Bei 600 s tragen 4–5 echte Granulate 96–98 % der
Masse (Summe der Gewichte der Rechenpartikel > 100 µm). Abgebildet sind sie durch
100–170 Rechenpartikel mit W ≈ 0,06. Ursache: nur 6 Tropfen in der Box (150 µm in
1e-10 m³), also 6 Keime. Ob diese wenigen Körner knapp über oder unter 250 µm liegen,
kippt Q3(250) um ±0,2. Stetige Maße (d50, Verschiebung in Dekaden) rauschen viel
weniger.

## 2. Wo die Zeit hingeht (cProfile, 60 s Sim, 1 Prozess)

93 % in `rebuild_r_pairdelta_pairwise_serial` (paarweise Agglomerations-Propensitäten,
O(n_comp²) pro Event, ~19 ms bei n_comp ≈ 500). Alles andere (Bruch-Raten,
kontinuierliche Prozesse, Nucleation, Merger) liegt zusammen bei < 5 %.
Folge: Laufzeit ≈ Events × n_comp². Bei fester Box sind die Events ∝ n_comp, also
Laufzeit ∝ n_comp³. Halbe Box mit halber Rechenpartikelzahl (W gleich) bringt
also ~8× Geschwindigkeit.

## 3. numba-Threads

4 Prozesse × 1 Thread gegen 4 Prozesse × Standard-Threads (120 s Sim): 185 s gegen 197 s.
Die Ergebnisse sind bitgleich (gleiche Eventzahlen). `run_sensitivity.py` setzt deshalb
`NUMBA_NUM_THREADS=1` und parallelisiert über Prozesse.

## 4. Tropfenzahl in der Box (Erics Test 50 gegen 200 Tropfen)

Gleiche Box (Vc = 1e-10 m³, N_comp = 500), gleiche Binder-Menge. Nur die Tropfen werden
kleiner, damit 50 bzw. 200 Tropfen in die Box passen. 4 Replikate, 4 parallel,
1 numba-Thread.

| Variante | Tropfen | Laufzeit | Events/s | d10/d50/d90 @600 s | Verschiebung zu orig | Streuung log d50 / Q3(250) je Repl. | J_KF3 |
|---|---|---|---|---|---|---|---|
| orig | 150 µm, 6 St. | 578 s | 27,1 | 223/248/305 µm | 0 | 0,029 Dek. / 0,23 | 0,035 |
| drop50 | 74 µm, 50 St. | 453 s | 14,6 | 128/182/239 µm | −0,135 Dek. (feiner) | 0,004 / 0,04 | 0,106 |
| drop200 | 47 µm, 200 St. | 372 s | 10,1 | 120/176/231 µm | −0,155 Dek. | 0,007 / 0,04 | 0,098 |

- 50 und 200 Tropfen geben fast dieselbe Verteilung (d50 182 gegen 176 µm). Bei 600 s
  sind es ~16–17 Granulate pro Box, nicht 50: Keime verbinden sich.
- Das Original mit 6 Tropfen liegt deutlich gröber. Die Vermutung: Mit nur 6 Tropfen in
  der Box ist die Verteilung durch die Tropfenzahl begrenzt (numerischer Effekt,
  „Tropfen und Box zu nah“), nicht durch die Physik. Zur Gegenprobe laufen dieselben
  150-µm-Tropfen in 2× und 4× größerer Box (Abschnitt 5).
- Das Diskretisierungs-Kriterium ist überall erfüllt (Verhältnis 7–20).

## 5. Gegenprobe: 150-µm-Tropfen in größerer Box; Rechenpartikel; Zugabedauer

| Variante | Box / Tropfen | N_comp (W_init) | Events | d10/d50/d90 @600 s | Verschiebung zu orig | Streuung log d50 je Repl. |
|---|---|---|---|---|---|---|
| orig | 1e-10 m³ / 6 | 500 (3,2) | 16 272 | 223/248/305 | 0 | 0,029 |
| **n250** | 1e-10 / 6 | 250 (6,4) | 7 938 | 223/244/308 | +0,002 | 0,016 |
| f2_n250 | 2e-10 / 12 | 250 (12,7) | 10 225 | 228/261/311 | +0,014 | 0,016 |
| f4_n250 | 4e-10 / 24 | 250 (25,5) | 16 862 | 225/256/327 | +0,008 | 0,018 |
| f8_n250 | 8e-10 / 48 | 250 (51) | 31 586 | 230/251/308 | +0,004 | 0,012 |
| liq30 | 1e-10 / 6, Zugabe 30 s | 500 (3,2) | 15 392 | 227/246/303 | +0,003 | 0,022 |

(Verschiebung: + = gröber als orig. Die Laufzeiten dieser Gruppe sind nicht vergleichbar:
f2/f4/f8_n250 und liq30 liefen versehentlich gleichzeitig, also mit 8 Prozessen auf 4 Kernen.
Saubere Zeiten folgen in Abschnitt 6.)

Folgerungen:
- **Die Vermutung aus Abschnitt 4 ist widerlegt:** Mit 150-µm-Tropfen geben 6, 12, 24 und
  48 Tropfen pro Box dieselbe Verteilung (d50 244–261 µm, Verschiebung ≤ 0,014 Dekaden,
  im Rahmen des Rauschens). Die Granulatgröße setzt die **Tropfengröße** (Keimgröße), nicht
  die Box. 6 Tropfen pro Box sind also kein numerisches Artefakt. Kleinere Tropfen (74 µm)
  ändern dagegen die Physik: Die Verteilung wird um 0,14 Dekaden feiner.
- **N_comp 250 statt 500:** dieselbe Verteilung (+0,002 Dekaden), halb so viele Events,
  ~3,8× schneller (152 s gegen 578 s je Replikat, beide sauber gemessen).
- **Größere Box** senkt das Rauschen je Replikat kaum (das hängt an N_comp, nicht an der
  Box), kostet aber Events ∝ Box. Mehr Seeds sind die billigere Art, Rauschen zu senken.
- **30 s Zugabe:** gleiche Verteilung, nur ~5 % weniger Events. Lohnt sich kaum.
- Q3(250 µm) streut bei allen Varianten stark (0,05–0,23 je Replikat). Der Grund: d50 liegt
  praktisch auf der 250-µm-Masche, dort ist die Summenkurve am steilsten. Das ist keine
  Schwäche der Simulation, sondern dieser einen Kennzahl. Die stetigen Maße (d10/d50/d90,
  Verschiebung) sind die belastbaren.

## 6. Saubere Laufzeiten (je Variante allein, 4 Prozesse × 1 Thread, 4 Replikate)

| Variante | N_comp | Laufzeit je Replikat | Events | d50 @600 s | Verschiebung zu orig |
|---|---|---|---|---|---|
| orig | 500 | 578 s* | 16 272 | 248 µm | 0 |
| n250 | 250 | ~80 s | 7 938 | 244 µm | +0,002 |
| n250_liq30 (30 s Zugabe) | 250 | 89 s | ~7 960 | – | – |
| **n125** | **125** | **23 s** | 5 245 | 257 µm | +0,008 |

\* orig mit numba-Standard-Threads; laut Abschnitt 3 macht das ~6 % aus.

- **N_comp 125 ist 25× schneller als das Original**, liefert dieselbe Verteilung und erfüllt
  das Diskretisierungs-Kriterium (8,7 Events/s gegen 2 × 1,37/s → Faktor 6,4).
- **30 s Zugabe bringt keine Zeitersparnis** (89 gegen 80 s) → Zugabe bleibt bei 60 s wie im Fit.

## 7. Bruch-Scan: Ab wann bricht etwas, und was kostet es?

Variiert wird die Bruch-Referenzrate `log10 S_ref = log10(BREAK_P1 · V_ref^BREAK_P2)` mit
d_ref = 300 µm (Basis id 1070: −2,44). Alles andere ist Basis (N_comp 125 bzw. 250).

Bruchraten der gespeicherten Partikel mit der Kernel-Formel nachgerechnet (sref = 0):
Granulate sind nass (Sättigung 0,84–0,99) → **Nass-Regime**, σ ≈ 48–54 kPa,
Porosität ≈ 0,187. Damit wirken α (RUMPF_ALPHA), Kontaktwinkel und Porosität auf die
Festigkeit, RUMPF_K (nur trocken) praktisch nicht. Primärpartikel (Porosität 0) brechen nie.

| sref | Bruch-Events (N125) | d10/d50/d90 @600 s | Verschiebung | Laufzeit (N125) | n_comp Ende |
|---|---|---|---|---|---|
| −2,44 (Basis) | 0 | 234/257/302 | 0 | 23 s | ~120 |
| +1 | 4 | 221/254/309 | ≈ 0 | 28 s | ~125 |
| +2 | 36 | 142/195/251 | −0,11 Dek. | 30 s | ~150 |
| +3 | 237 | 101/136/173 | −0,26 Dek. | 34 s | 240–430 |
| +4 | ~2 400 (abgebrochen) | – | – | **> 1 500 s (Timeout)** | 1 600–2 400, verdoppelt sich ~jede Minute ab 300 s |

N_comp 250 gegen 125 bei aktivem Bruch (sref = 2): Verschiebung −0,117 gegen −0,112
Dekaden, d50 194 gegen 195 µm → N_comp 125 bleibt gleichwertig.

## 8. Wo explodiert die Rechenpartikelzahl? (BREAK_P2 und Festigkeit)

Bei starkem Bruch zerbrechen auch die Bruchstücke immer weiter. Jedes Stück ist ein
Rechenpartikel, die Kosten wachsen ∝ n_comp². Läuft das aus dem Ruder, reißt der Lauf das
Zeitlimit. In der Studie fehlt dann ein Punkt, und sein ganzer Saltelli-Block wird
weggelassen. Die Parameterbereiche müssen deshalb fast ausschließlich im „sicheren“ Gebiet
liegen. N_comp 125, 4 Replikate je Zeile; „schwach“ = Kontaktwinkel 1,4 rad und
MIN_POROSITY 0,45 (die festigkeitsmindernde Ecke, ≈ +1,3 Dekaden effektive Rate).

| sref | P2 | Festigkeit | Laufzeit | n_comp Ende | Bruch-Events | d50 @600 s | Verschiebung |
|---|---|---|---|---|---|---|---|
| 3 | 1,66 | Basis | 34 s | 240–430 | 237 | 136 µm | −0,26 |
| 3 | 2,0 | Basis | 38 s | 210–290 | 151 | 157 µm | −0,21 |
| 3 | 1,0 | Basis | **> 10 min (explodiert)** | | | | |
| 3 | 0,5 | Basis | **Timeout 1 500 s**, n_comp ~2 250 | | | | |
| 4 | 1,66 | Basis | **Timeout 1 500 s**, n_comp ~2 400 | | | | |
| 2,5 | 1,66 | Basis | 34 s | 160–260 | 101 | 166 µm | −0,19 |
| 2,5 | 1,0 | Basis | 38 s | 290–490 | 296 | 138 µm | −0,26 |
| 2 | 1,0 | Basis | 33 s | 160–180 | 69 | 184 µm | −0,15 |
| 2 | 0,5 | Basis | 42 s | 275–440 | 276 | 149 µm | −0,23 |
| 2 | 1,0 | schwach | 51 s | 440–660 | 474 | 134 µm | −0,27 |
| 1,5 | 1,0 | schwach | 34 s | 220–310 | 138 | 160 µm | −0,20 |
| 1,5 | 0,5 | schwach | 43 s | 350–500 | 362 | 142 µm | −0,25 |

- Die Verschiebung sättigt bei ≈ −0,27 Dekaden (d50 ≈ 135 µm). Stärkerer Bruch macht die
  Verteilung kaum feiner, kostet aber exponentiell Rechenzeit.
- Die Festigkeitsparameter wirken deutlich: bei sref 2 / P2 1,0 −0,15 (Basis) gegen −0,27
  Dekaden (schwach).
- Das Diskretisierungs-Kriterium ist in allen Läufen erfüllt (Verhältnis 6,0–6,7).

Ecken bei sref 2,5 und P2 1,0 (Zeitlimit 300 s):

| Festigkeit | Ergebnis |
|---|---|
| schwach (δ 1,4, ε 0,45) | **alle 4 Timeout**, n_comp ~1 200 nach 300 s |
| mittel (δ 1,0, ε 0,35) | 133 / 244 / 303 s, 1× Timeout |

→ Vorschlag für die Studie: **BREAK_SREF 0 … 2, BREAK_P2 1,0 … 2,0**. Die schlimmste Ecke
dieses Bereichs (sref 2, P2 1,0, schwach) rechnet in 40–61 s, und der Bruch erreicht dort
trotzdem die Sättigung (−0,27 Dekaden).

## 9. Mini-Studie mit dem echten Runner (Ende-zu-Ende-Test)

Config = `sa_config_bruch.py` (am 18.10. auf Erics Entscheidung gelöscht) mit N = 4, 2 Seeds, Basis 4 Seeds, ohne S2 → 36 Punkte, 76
Läufe, 4 parallel. Ablauf wie geplant:
- Der Basispunkt läuft zuerst (27–30 s).
- Danach wird das Zeitlimit = 10 × 27 s = 275 s gesetzt und in `time_limit.json` gespeichert.
- Die Saltelli-Punkte brauchen im **Mittel 21,6 s** (max. 37 s).
- Die Hälfte der Läufe hat mehr als 10 Bruch-Events. Keine Timeouts, keine
  Diskretisierungs-Markierung (kleinstes Verhältnis 5,9).
- Die Auswertung lief durch. Bei nur 4 Blöcken rechnet sie absichtlich keine Sobol-Indizes.

## Zusammenfassung / Empfehlung

| Frage | Ergebnis | Empfehlung |
|---|---|---|
| Kleinere Tropfen? | ändern die Physik (−0,14 Dekaden) | nein, 150 µm bleiben |
| Größere Box? | gleiche Verteilung, teurer | nein, f = 1e-7 bleibt |
| Kürzere Zugabe? | gleiche Verteilung, nicht schneller | nein, 60 s bleiben |
| Weniger Rechenpartikel? | N_comp 125 = gleiche Verteilung, 25× schneller | **ja, N_comp 125** |
| Seeds | SE stetiger Maße ≈ 0,005–0,01 Dekaden bei 5 Seeds | 5 Seeds reichen für d-Werte/Verschiebung |
| Bruch-Bereich | sichtbar ab sref ~1, gesättigt ~2 (schwach) bzw. 3 | sref 0 … 2, P2 1 … 2 |

## 10. Einzeltests aller Kandidaten-Parameter (18.10.)

Jeder Kandidat einzeln an Unter- und Obergrenze seines vorgeschlagenen Bereichs: Basis n125,
4 Seeds, Zeitlimit 300 s. Erzeugt mit `kalib_run.py` (Varianten `o_*` bzw. Ecken `c_*`),
Tabelle mit `kalib_oat_table.py`. Achtung: Der Laptop lief bei dieser Messreihe langsamer;
eine Variante ohne jede Wirkung (z. B. o_rho_hi, bitgleich zur Basis) brauchte hier ~31 s
statt 23 s. Laufzeiten deshalb relativ zu ~31 s lesen.

Namen: aref = Agglomerations-Niveau a_ref (−5,8 / −4,8), agga = AGG_A (0,74 / 1,43),
uref = U_ref (0,002 / 0,2), ha = h_a (100 nm / 1 µm), rho = ρ_eff (300 / 1200),
ckagg = k_agg (0,4 / 1), eps = ε_min (0,15 / 0,45), comp = k_comp (0,0032 / 0,32),
kint = k_int (6,3e9 / 5e11), drop = Tropfen (75 / 225 µm, Box wächst mit),
dwagg = AGG_DW_MAX absolut (1 / 12), dwrel_lo = dW 1 bei hoher Agglomeration,
batch = Batchsize (1 / 50), dwbr = BREAK_DW_MAX (1 / 10 / 20, bei sref 2),
rk = RUMPF_K (2,2 / 2,8, bei sref 2), cdp = Keim-Startporosität 0,5 (ohne/mit Bruch).
Ecken: c_brk = alle festigkeitsmindernden Größen + sref 2 + BREAK_DW_MAX 1;
c_slow = dW 1 + a_ref −4,8 + 225-µm-Tropfen (absolutes dW); c_slow_rel = dasselbe mit
relativem dW (AGG_DW_MAX ∝ Box).

| Variante | Laeufe ok/TO | Laufzeit Mittel/Max [s] | Events/s | Diskr.-Verh. min | Bruch-Ev. | n_comp Ende | d50 @600 [um] | Verschiebung [Dek.] (+ = groeber) | Streuung je Seed |
|---|---|---|---|---|---|---|---|---|---|
| n125 | 4/0 | 23/25 | 8.7 | 5.9 | 0 | 117 | 257 | +0.000 | 0.016 |
| c_brk | 4/0 | 35/38 | 8.3 | 5.6 | 120 | 225 | 157 | -0.217 | 0.006 |
| c_slow | 0/4 | 309/309 | - | - | - | - | - | - | - |
| c_slow_rel | 4/0 | 144/148 | 57.0 | 40.8 | 0 | 120 | 370 | +0.187 | 0.022 |
| o_agga_hi | 4/0 | 35/37 | 8.8 | 6.1 | 0 | 124 | 264 | +0.017 | 0.021 |
| o_agga_lo | 4/0 | 36/37 | 8.8 | 6.1 | 0 | 122 | 309 | +0.058 | 0.020 |
| o_aref_hi | 4/0 | 49/50 | 13.2 | 8.7 | 0 | 113 | 321 | +0.095 | 0.040 |
| o_aref_lo | 4/0 | 18/18 | 4.5 | 3.2 | 0 | 133 | 209 | -0.329 | 0.013 |
| o_batch_hi | 4/0 | 31/33 | 8.7 | 5.9 | 0 | 117 | 257 | +0.000 | 0.016 |
| o_batch_lo | 4/0 | 34/36 | 8.7 | 5.9 | 0 | 117 | 257 | +0.000 | 0.016 |
| o_cdp_hi | 4/0 | 31/33 | 9.1 | 6.0 | 0 | 123 | 253 | +0.002 | 0.038 |
| o_cdp_hi_br | 4/0 | 35/38 | 8.6 | 5.7 | 37 | 150 | 195 | -0.118 | 0.014 |
| o_ckagg_hi | 4/0 | 33/35 | 8.9 | 6.1 | 0 | 121 | 242 | -0.020 | 0.009 |
| o_ckagg_lo | 4/0 | 30/31 | 8.2 | 5.8 | 0 | 120 | 234 | -0.027 | 0.010 |
| o_comp_hi | 4/0 | 31/33 | 8.7 | 2.6 | 0 | 117 | 252 | -0.003 | 0.020 |
| o_comp_lo | 4/0 | 33/34 | 8.8 | 272.6 | 0 | 124 | 252 | +0.003 | 0.022 |
| o_drop_hi | 4/0 | 77/78 | 27.5 | 19.2 | 0 | 131 | 347 | +0.047 | 0.012 |
| o_drop_hi_rel | 4/0 | 43/46 | 14.0 | 9.4 | 0 | 134 | 344 | +0.041 | 0.002 |
| o_drop_lo | 4/0 | 38/40 | 6.2 | 4.3 | 0 | 127 | 177 | -0.146 | 0.009 |
| o_dwagg_hi | 4/0 | 32/33 | 7.6 | 5.2 | 0 | 124 | 255 | +0.002 | 0.008 |
| o_dwagg_lo | 4/0 | 75/79 | 26.1 | 16.9 | 0 | 116 | 250 | -0.017 | 0.017 |
| o_dwbr_base | 4/0 | 37/39 | 8.6 | 5.8 | 36 | 150 | 195 | -0.120 | 0.022 |
| o_dwbr_hi | 4/0 | 38/40 | 8.6 | 5.8 | 36 | 150 | 195 | -0.120 | 0.022 |
| o_dwbr_lo | 4/0 | 38/40 | 8.6 | 5.8 | 36 | 150 | 195 | -0.120 | 0.022 |
| o_dwrel_lo | 4/0 | 140/141 | 41.6 | 29.5 | 0 | 117 | 284 | +0.069 | 0.028 |
| o_eps_hi | 4/0 | 32/34 | 9.1 | 6.1 | 0 | 125 | 254 | -0.008 | 0.008 |
| o_eps_lo | 4/0 | 34/35 | 9.2 | 6.5 | 0 | 124 | 250 | -0.007 | 0.020 |
| o_ha_hi | 4/0 | 32/34 | 8.9 | 6.0 | 0 | 121 | 254 | -0.014 | 0.010 |
| o_ha_lo | 4/0 | 34/36 | 8.8 | 5.8 | 0 | 121 | 238 | -0.015 | 0.019 |
| o_kint_hi | 4/0 | 32/33 | 8.8 | 5.9 | 0 | 117 | 254 | +0.008 | 0.026 |
| o_kint_lo | 4/0 | 31/33 | 8.7 | 5.9 | 0 | 116 | 258 | -0.003 | 0.016 |
| o_rho_hi | 4/0 | 31/33 | 8.7 | 5.9 | 0 | 117 | 257 | +0.000 | 0.016 |
| o_rho_lo | 4/0 | 35/38 | 8.7 | 5.9 | 0 | 117 | 257 | +0.000 | 0.016 |
| o_rk_hi | 4/0 | 36/38 | 8.6 | 5.8 | 39 | 154 | 197 | -0.121 | 0.011 |
| o_rk_lo | 4/0 | 35/37 | 8.6 | 5.8 | 38 | 154 | 198 | -0.120 | 0.015 |
| o_uref_hi | 4/0 | 31/33 | 8.7 | 5.9 | 0 | 117 | 257 | +0.000 | 0.016 |
| o_uref_lo | 4/0 | 32/34 | 8.7 | 5.9 | 0 | 117 | 257 | +0.000 | 0.016 |

Folgerungen (übernommen in `PARAMETERLISTE.md` / `sa_config_gesamt.py`):
- **Ohne Wirkung (bitgleich):** U_ref, ρ_eff, Batchsize, BREAK_DW_MAX. Nahezu ohne Wirkung:
  RUMPF_K, Keim-Startporosität. Stokes (aus den n125-Partikellisten nachgerechnet): St 1·10⁻³ … 3·10⁻²,
  St* 2,6 … 4,8, kleinster Abstand St*/St ≈ 60 (h_a = 500 nm), bei h_a = 1 µm nur ~16. In der
  Basis entscheidet deshalb praktisch nur h > h_a. U_ref bleibt trotzdem in der Studie, weil es
  zusammen mit großem h_a greifen kann.
- **Stark:** a_ref (−0,33 / +0,10), Tropfen (−0,15 / +0,04), Bruch (s. Abschn. 7/8).
  Mittel: AGG_A. Schwach: h_a, k_agg, ε_min, k_comp, k_int (ohne Bruch).
- **dW absolut ist gefährlich:** dW 1 + große Tropfen-Box → Timeout (~80× langsamer).
  Mit dW relativ zum Startgewicht (Box ×3,4 → dW ×3,4) rechnet dieselbe Ecke in 144 s
  (~4,6× Basis), und 225-µm-Tropfen allein brauchen 43 statt 77 s bei gleicher Verteilung.
  → In der Studie `AGG_DW_REL`.
- **Diskretisierung:** k_comp oben (0,32 → k = 3,1/s) senkt das Verhältnis auf ~2,6.
  Zusammen mit niedrigem a_ref (4,5 Events/s) fällt es unter 2. Dort greift die Markierung.

## 11. Mini-Studie Gesamt-Config (18.10.)

`sa_config_gesamt.py` mit N = 4, 2 Seeds, Basis 4 Seeds: 72 Punkte quer durch alle 16
Parameter, 148 Läufe, echter Runner, 4 parallel.
- Basispunkt 33 s (Laptop gedrosselt). Zeitlimit 10 × 33 = 327 s.
- **Saltelli-Läufe im Mittel 38,5 s = 1,17 × Basis**, Median 30 s, 90 % unter 73 s,
  Maximum 130 s.
- Keine Timeouts, keine Diskretisierungs-Markierung, Bruch-Events im Median 6.
- Verschiebung über die Punkte −0,29 … +0,25 Dekaden, log10 d50 2,16 … 2,67 (145–470 µm).
- Gefunden und behoben: ein Fehler beim atomaren Schreiben von `time_limit.json`. Die
  Studie wurde danach fortgesetzt; die 4 schon fertigen Basisläufe wurden übernommen.

## 12. Anfangsporosität der Primärpartikel ε0 (18.10.)

Auf Erics Wunsch ist `INITIAL_POROSITY` (ε0) mit 0 … 0,8 in der Studie. Der Adapter hält
den Durchmesser der Primärpartikel fest und setzt ihr Feststoffvolumen auf (1 − ε0) · V.
Bei 0,8 bleiben also nur 20 % Feststoff je Primärpartikel; auf denselben Binder kommt
5× weniger Feststoff. Poröse Primärpartikel können außerdem brechen (bei ε0 = 0 nie).
N_comp 125, 4 Seeds, Zeitlimit 600 s, Varianten `e0_*` und `*_e08`.

| Variante | Laufzeit | Events/s | Verhältnis | Bruch-Events | d10/d50/d90 @600 s | Verschiebung | J_KF3 |
|---|---|---|---|---|---|---|---|
| n125 (Basis, ε0 = 0) | 23 s | 8,7 | 6,4 | 0 | 234/257/302 | +0,01 | 0,034 |
| ε0 = 0,4 | 32 s | 7,8 | 5,7 | 0 | 193/212/286 | −0,07 | 0,052 |
| ε0 = 0,8 | 23 s | 2,7 | 2,0 (markiert) | 0 | 236/296/385 | +0,08 | 0,132 |
| ε0 = 0,8, k_int 5·10¹¹ | 24 s | 2,7 | 2,0 (markiert) | 0 | 218/295/385 | +0,07 | 0,131 |
| c_brk (Bruch-Ecke, ε0 = 0) | 35 s | 8,3 | 6,0 | 120 | 116/157/192 | −0,21 | 0,102 |
| c_brk + ε0 = 0,8 | **144–294 s** | 7,9 | 5,8 | 1 051 | 59/86/112 | −0,47 | 0,284 |
| c_slow_rel (ε0 = 0) | 144 s | 57 | 41,5 | 0 | 333/370/484 | +0,19 | 0,161 |
| c_slow_rel + ε0 = 0,8 | 72 s | 12,8 | 9,3 | 0 | 386/580/666 | +0,34 | 0,264 |

(Verschiebung gegen die N_comp-500-Läufe des Fits in Dekaden, + = gröber.)

- ε0 wirkt deutlich und **nicht monoton**: 0,4 macht feiner, 0,8 gröber und viel breiter.
- Bei hohem ε0 gibt es weniger Events (weniger Agglomeration). Das Diskretisierungs-
  Verhältnis fällt dann auf ~2 und die Punkte werden markiert. Die MC-Streuung je Seed
  steigt von 0,02 auf ~0,06 Dekaden (log10 d50).
- Zusammen mit dem stärksten Bruch zerbrechen die porösen Primärpartikel: d50 86 µm,
  n_comp bis ~1 300, Laufzeit bis 12,8× Basis.
- **Stichprobe aus 64 echten Design-Punkten** (je 1 Seed, Laptop, Limit 300 s): alle ok,
  Median 22 s, Mittel 24 s, Maximum 71 s (~3× Basis). Die 16 Punkte mit ε0 > 0,6 rechnen
  in 8–24 s, auch bei hohem Bruch-Niveau. Die teure Ecke braucht also alles zugleich
  (ε0 hoch, Bruch-Niveau oben, P2 ≈ 1, schwache Festigkeit) und ist selten.
  Diskretisierungs-Markierung: 3 von 64, alle mit ε0 > 0,6.
- → `TIME_LIMIT_FACTOR` von 10 auf **20** erhöht, damit auch die Extrem-Ecke durchläuft.

## 13. Tropfen bis 50 µm, ε0 mit fester Masse, Verdichtung der Primärpartikel (18.10. abends)

Entscheidungen Eric: Tropfen-Untergrenze 50 µm; bei ε0 bleibt die **Feststoffmasse fest**.
Umsetzung: Art `porosity_mass_fixed` in `run_sensitivity.to_params`: `INITIAL_POROSITY` = ε0,
`STOKES_RHO_SOLID` = 600 / (1 − ε0). Die Dichte geht im Solver nur in die Stokes-Masse
ρ · V_fest ein (Kollisionsradius und Bruch nutzen das Hüllvolumen); damit ist die Masse exakt
die der Basis. N_comp 125, 4 Seeds. Die Varianten `d050*` liefen gleichzeitig mit einem
anderen Test, ihre Laufzeiten sind daher eher zu hoch.

| Variante | Laufzeit | Events/s | Verhältnis | akz. Agglo | d10/d50/d90 @600 s | Verschiebung |
|---|---|---|---|---|---|---|
| Tropfen 75 µm (o_drop_lo) | 38 s | 6,2 | 4,5 | 1 761 | 133/177/233 | −0,14 |
| Tropfen 50 µm (d050) | 48 s | 5,3 | 3,9 | 1 805 | 122/176/228 | −0,16 |
| 50 µm + a_ref −4,8 + dW 1 (d050_fastagg) | 109 s | 18,3 | 13,3 | 2 269 | 194/270/343 | +0,03 |
| 50 µm + stärkster Bruch + ε0 0,8 (ρ 3000) | **> 600 s, alle Timeout** | – | – | – | n_comp ~1 600 | – |
| ε0 0,8, ρ 600 (e0_08, Abschn. 12) | 23 s | 2,7 | 2,0 | 784 | 236/296/385 | +0,08 |
| ε0 0,8, ρ 3000, h_a 1 µm, U_ref 0,2 | 24 s | 2,0 | 1,5 | 460 | 229/351/413 | +0,13 |

- 50 µm gibt praktisch dieselbe Verteilung wie 75 µm (wie schon 47 gegen 74 µm in
  Abschnitt 4): Unter ~75 µm setzt der Tropfen die Granulatgröße kaum noch.
- Mit fester Masse (ρ bis 3 000) greift bei ε0 0,8 das Stokes-Kriterium, sobald h_a und
  U_ref groß sind: weniger akzeptierte Agglomerationen, gröbere und breitere Verteilung.
- Die Extrem-Ecke (alles festigkeitsmindernd + stärkster Bruch + ε0 0,8) wird mit kleinen
  Tropfen noch teurer und reißt das Limit. Wie oft das im Design vorkommt, misst die
  Stichprobe unten.
- **Verdichtung der Primärpartikel:** Die Kompaktierung wirkt auf jedes Partikel mit
  Porosität > ε_min, auch auf trockene Primärpartikel. Aus den Partikellisten bei 360 s
  (Partikel mit Sättigung 0): ε0 0 → Porosität 0, 34,0 µm; ε0 0,4 → 0,2, 30,9 µm;
  ε0 0,8 → 0,2, 21,4 µm. Die Masse bleibt erhalten, aber das Ausgangspulver schrumpft.
  Entscheidung Eric (18.10.): so lassen, keine Änderung am Solver.

## 14. Stichprobe aus 128 Punkten der endgültigen Config (18.10. spätabends)

128 echte Design-Punkte von `sa_config_gesamt.py` (18 Parameter, Tropfen 50 … 225 µm, ε0 mit
fester Masse), gestreut über alle Zeilentypen der Saltelli-Blöcke, je 1 Seed, 4 parallel,
Limit 500 s (~20 × Basis 23 s).

- Median 29 s (1,3 × Basis), Mittel 42 s (1,8 ×), 3 Punkte über 150 s.
- **2 von 128 Timeouts** (Punkte 1497 und 3487): Bruch-Niveau 60–80 /s, Kontaktwinkel
  0,7–1,35 rad, ε_min ~0,4, Tropfen 88–127 µm, ε0 niedrig (0,10 / 0,32). Bei 500 s hatten sie
  575 bzw. 592 von 600 s Simulationszeit erreicht; die Partikelzahl verdoppelt sich gegen
  Ende alle 60–90 s (bis ~1 400–1 700).
- Ohne enges Limit (2 parallel) laufen beide durch: **439 s und 483 s** (mit 4 parallel
  ~520–560 s, also ~22–24 × Basis). Keine Explosion, nur langsam.
- Mit 20 × wären rechnerisch ~1/3 der Blöcke ausgefallen (je Block 20 Punkte à 5 Seeds),
  also keine Sobol-Indizes. → `TIME_LIMIT_FACTOR` = **40**.
- Laufzeit-Treiber (Regression von log Laufzeit auf die Parameter, R² 0,66; Faktor vom
  unteren zum oberen Bereichsende): AGG_AREF ×2,9, AGG_DW_REL ×0,43 (kleines dW teuer),
  ε0 ×0,41 (hohes ε0 billiger), AGG_A ×0,65, CONE_K_AGG ×1,5, Kontaktwinkel ×1,4,
  Bruch-Niveau ×1,4, ε_min ×1,4. Tropfen unter 75 µm ändern das Mittel nicht (42 s gegen 42 s).
- → `DESIGN_COST_FACTOR` (Hochrechnung im Benchmark) 1,2 → **1,9**.
