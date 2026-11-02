# Parameterliste: alle potenziell interessanten Parameter

Quellen:
- MA (Stand 18.10.2026), **Tabelle 3.1** und die Kapitel 2/3,
- der Code (`WetGranulationAdapter._build_solver`, Kernels),
- die Einzeltests in `kalibrierung/`.

Für jeden Parameter wurde einzeln an der Unter- und Obergrenze gemessen: Basis id 1070,
N_COMP 125, 4 Seeds. „Wirkung“ ist die Verschiebung der Q3-Kurve bei 600 s gegen die Basis
in Dekaden (+ = gröber). Zum Vergleich: Die Streuung je Seed liegt bei ~0,01–0,02 Dekaden,
die Wiederholstreuung des Versuchs (d50) bei ~0,035 Dekaden.

**Legende:**
- ✔ in der Studie (`sa_config_gesamt.py`)
- ✗ bewusst nicht drin (Grund dabei)
- ≡ redundant zu einem anderen Parameter
- ● fest, weil gemessen bzw. bekannt

**Ändern:**
- Einen ✔-Parameter herausnehmen: seine Zeile in `sa_config_gesamt.py` löschen. Er bleibt
  dann auf dem Basiswert.
  - AGG_A nur zusammen mit AGG_AREF herausnehmen, BREAK_P2 nur zusammen mit BREAK_SREF.
    Sonst verschiebt der verbliebene Parameter das Niveau mit.
- Die nach Messung weggelassenen Parameter (ρ_eff, dW Bruch, Keim-Startporosität)
  stehen fertig **auskommentiert** in der Config. Wieder aufnehmen: `#` entfernen (geprüft).
- Alles Weitere nach demselben Muster ergänzen; der Bereich steht in den Tabellen unten.

## Agglomeration

| Parameter (Code / MA) | Basis | Bereich | Einstufung | Begründung, Messung |
|---|---|---|---|---|
| `EKE_CORR_BETA` / β_fit, als **`AGG_AREF`** | log a_ref = −5,295 | −5,8 … −4,8 (log) | ✔ | Agglomerations-Niveau der Primärpartikel (s. u.). Wirkung **−0,33 / +0,10**; Laufzeit 0,6× / 1,6× |
| `AGG_A` / c_temp | 1,34 | 0,74 … 1,43 | ✔ | Bereich = beste 5 % des KF3-Fits. Bei festem a_ref: +0,06 / +0,02 |
| `STOKES_H_A` / h_a | 500 nm | 100 nm … 1 µm (log) | ✔ | Annahme (Tab. 3.1). Entscheidet über die Filmdicken-Prüfung h > h_a und über St* ∝ ln(h/h_a). −0,015 / −0,014 |
| `CONE_K_AGG` / k_agg | 0,89 | 0,4 … 1,0 | ✔ | gefittet, Bereich ≈ beste 5 %. −0,03 / −0,02 |
| `STOKES_U_COLL_REF` / U_ref | 0,016 m/s | 0,002 … 0,2 (log) | ✔ | Einzeln **bitgleich zur Basis** an beiden Enden. Grund (aus den Partikellisten nachgerechnet): St 1·10⁻³ … 3·10⁻², St* 2,6 … 4,8, kleinster Abstand St*/St ≈ 60 bei h_a = 500 nm. Bei h_a = 1 µm sind es aber nur ~16, und U_ref bis 12,5× kann dort zusammen mit h_a greifen. Darum drin |
| `STOKES_BINDER_VISCOSITY` / μ_b | 0,1 Pa s | – | ≡ | St ∝ U/μ, also exakt dieselbe Wirkung wie U_ref (Bereich 0,002–0,2 von U_ref entspricht μ_b 0,008–0,8). Nicht eigens getestet |
| `STOKES_RHO_SOLID` / ρ_eff | 600 kg/m³ | (300 … 1200) | ≡ | Einzeln **bitgleich zur Basis**. Höchstens Faktor 2 auf St, das bleibt selbst bei h_a = 1 µm mit Abstand ≥ 8 wirkungslos. Wird über ε0 (Masse fest) mitgeführt |
| `EKE_C_MIXER`, `STOKES_C_VEL` / c_coll, c_v | 0,0995 / 0,2852 | – | ≡ | Bei fester Drehzahl nur Faktoren auf β_fit bzw. U_ref |
| `N_MIXER` / n_mixer | 30 m/s | – | ● | Versuch |

**Warum `AGG_AREF` statt `EKE_CORR_BETA`:** Im Fit sind β_fit und c_temp entlang einer Linie
gekoppelt (Korrelation −0,999 in den besten 5 %). Ihre Steigung entspricht exakt der
Kernel-Formel für zwei Primärpartikel (r = 17 µm). Würde man beide unabhängig ziehen, läge
fast jeder Punkt dekadenweit neben dem Tal: sinnlose Ergebnisse und Laufzeit-Explosion.
`AGG_AREF` = log10(β_fit · r_ref^(−1,5·c_temp)) ist das Agglomerations-Niveau der
Primärpartikel. c_temp dreht dann nur die Größenabhängigkeit.

## Bruch

| Parameter | Basis | Bereich | Einstufung | Begründung, Messung |
|---|---|---|---|---|
| `BREAK_P1` / P1, als **`BREAK_SREF`** | log S_ref = −2,44 | 0 … 2 (log) | ✔ | Bruch-Niveau bei d_ref = 300 µm. Ab ~1 sichtbar, bei 2 mit schwacher Festigkeit gesättigt (−0,27). Darüber explodiert die Rechenpartikelzahl |
| `BREAK_P2` / P2 | 1,66 | 1,0 … 2,0 | ✔ | unter 1,0 mit starkem Bruch → Explosion (Timeouts) |
| `RUMPF_ALPHA` / k_wet | 1,0 | 1,0 … 1,33 | ✔ | Literatur (Tab. 3.1), nasse Festigkeit |
| `RUMPF_K` / k | 2,5 | 2,2 … 2,8 | ✔ | Formkorrektur (Eric: „halbwegs relevant“). Bei aktivem Bruch ±0,001, denn die Granulate sind nass (S 0,84–0,99) und k wirkt nur bei S < 0,8. Drin, damit die Studie das belegt |
| `CONTACT_ANGLE` / δ | 0 | 0 … 1,4 rad | ✔ | Annahme. cos δ ändert die nasse Festigkeit bis Faktor 6 |
| `CONE_K_BREAK` / k_break | 0,37 | 0 … 1 | ✔ | gefittet |
| `BREAK_DW_MAX` (Numerik) | 10 | (1 … 20) | ✗ | **bitgleich** bei aktivem Bruch. Granulat-Rechenpartikel haben W ≈ 0,25 (max. ~0,5) ≪ 1; die Grenze greift nie |
| `SURFACE_TENSION` / γ | 0,072 N/m | – | ● | Wasser |
| `BREAK_C_MIXER` / c_break | 0,6699 | – | ≡ | Faktor auf P1 |
| `BREAKRVAL`, Fragmentzahl v_f, Form q | 4 / 2 / 1 | – | ✗ | Vorgabe Eric |
| `BREAK_PORO_MAX` | 0,9999 | – | ✗ | numerische Klemme |

## Porosität und Flüssigkeit (wirken über die Festigkeit auf den Bruch)

| Parameter | Basis | Bereich | Einstufung | Begründung, Messung |
|---|---|---|---|---|
| `MIN_POROSITY` / ε_min | 0,2 | 0,15 … 0,45 | ✔ | Annahme. Ohne Bruch ±0,01; mit Bruch stark, (1−ε)/ε bis Faktor 4,6 in der Festigkeit |
| `COMPRESSION_RATE` / k_comp | 0,141 /s | 0,0032 … 0,32 (log) | ✔ | gefittet, Bereich ≈ beste 5 %. Ohne Bruch ±0,003. **Am oberen Ende sinkt das Diskretisierungs-Verhältnis auf ~2,6**; die Markierung greift dort, wo zusätzlich wenig Events sind |
| `LIQUID_INTERNALIZATION_RATE` / k_int | 1,44·10¹⁰ | 6,3·10⁹ … 5·10¹¹ (log) | ✔ | gefittet, Bereich ≈ beste 5 %. ±0,008; wirkt über die Sättigung (nass/trocken) |
| `COMPRESSION_C_MIXER` / c_comp | 0,6699 | – | ≡ | Faktor auf k_comp |
| `INITIAL_POROSITY` / ε0 | 0 | 0 … 0,8 | ✔ | **Vorgabe Eric (18.10.)**, **Feststoffmasse fest**: Die reale Porosität (~0,8) steckt bisher ganz in ρ_eff = 0,2·ρ_s; ε0 verschiebt einen Teil davon in echten Porenraum. Durchmesser und Anzahl bleiben, Feststoffvolumen = 1 − ε0, dafür ρ_eff = 600 / (1 − ε0) (bei 0,8 = ρ_s). Die Dichte geht im Solver nur in die Stokes-Masse ein; Kollisionsradius und Bruch nutzen das Hüllvolumen. **Achtung:** Die Kompaktierung verdichtet auch trockene Primärpartikel bis ε_min (bei ε0 0,8 schrumpfen sie 34 → 21 µm), siehe FRAGEN Nr. 7 (Eric: so lassen). Einzeltest: ε0 0,4 → −0,07 Dek., 0,8 → +0,08 Dek. (breiter), Events/s 8,7 → 2,7 (Diskretisierung markiert), MC-Streuung ~3× größer; mit stärkstem Bruch teuer (kalibrierung/README_kalibrierung.md, Abschnitte 12 und 13) |
| `CONE_DEFAULT_POROSITY` | 0 | (0,5) | ✗ | gemessen ±0,002 (auch mit Bruch). Greift nur bei Keimen ohne Zwangs-Agglomeration, die bei 150-µm-Tropfen (25× Primärvolumen) praktisch nicht vorkommen |

## Nucleation / Flüssigkeitszugabe

| Parameter | Basis | Bereich | Einstufung | Begründung, Messung |
|---|---|---|---|---|
| `DROPLET_DIAMETER` / d_drop | 150 µm | 50 … 225 µm (log) | ✔ | **Annahme** (Tab. 3.1). Setzt die Keim- und damit die Granulatgröße: −0,15 / +0,04 (d50 177 / 344 µm). Über 150 µm wächst die Box mit, damit immer ≥ 6 Tropfen drin sind. Untergrenze 50 µm (Eric 18.10.): ~160 Tropfen in der Box, Laufzeit ~2× Basis |
| `NUCLEATION_BATCH_SIZE` (Numerik) | 20 | 1 … 50 (log) | ✔ | Vorgabe Eric (MA-ToDo „Gitterstudie Batchwise“). Gemessen **bitgleich**: Pro MC-Schritt ist weniger als ein Tropfen fällig, das Tropfen-Gewicht greift nie. Erwartung ST = 0; die Studie belegt das über den ganzen Raum |
| Binder-Menge, Zugabedauer, ρ_liq | 106 g / 60 s / 1000 | – | ● | Versuch bzw. Wasser |

## Numerik

| Parameter | Basis | Bereich | Einstufung | Begründung, Messung |
|---|---|---|---|---|
| `AGG_DW_MAX` / ΔW Agg, als **`AGG_DW_REL`** | 5 (= 0,39 · W_init) | 0,078 … 0,78 · W_init (log) | ✔ | Vorgabe Eric. Relativ zum Startgewicht, damit eine größere Box nicht die Eventzahl vervielfacht (absolut: dW 1 + Box ×3,4 → ~80× langsamer). Basis-Box: −0,017 / +0,002, Laufzeit 2,4× am unteren Ende; bei hoher Agglomeration sichtbar (−0,026) |
| `N_COMP` | 125 | – | ● | entschieden (Eric). 25× schneller als 500, gleiche Verteilung |
| `SIM_FRACTION` / Vc | 1e-7 | – | ● | Box 1×–8× gemessen: gleiche Verteilung. Geht nur über die Tropfen-Box ein |
| `AGG_DW_MIN`, `MERGER_TOLERANCE`, `MAYBE_DOUBLE_VC`, `MAXITER`, `MAX_PARTICLES` | – | – | ✗ | Schutz- bzw. Technikgrößen |

## Material

| Parameter | Basis | Einstufung | Begründung |
|---|---|---|---|
| `INITIAL_DIAMETER` / x0 | 34 µm | ● | Analysenzertifikat (D50). Eine Änderung verschiebt die Primärpartikelzahl, das wäre eine eigene Frage |
| ρ_s, ρ_eff | 3050, 0,2·ρs | ● / ✗ | wirkt nur über Stokes, s. o. |

## Zusammenfassung

**18 Parameter in der Studie:**
- Agglomeration (5): AGG_AREF, AGG_A, STOKES_H_A, STOKES_U_COLL_REF, CONE_K_AGG
- Bruch (6): BREAK_SREF, BREAK_P2, RUMPF_ALPHA, RUMPF_K, CONTACT_ANGLE, CONE_K_BREAK
- Porosität/Flüssigkeit (4): MIN_POROSITY, COMPRESSION_RATE, LIQUID_INTERNALIZATION_RATE, INITIAL_POROSITY
- Nucleation (1): DROPLET_DIAMETER
- Numerik (2): AGG_DW_REL, NUCLEATION_BATCH_SIZE

**Nach Messung weggelassen**, weil sie ohne Wirkung sind:
- bitgleich: STOKES_RHO_SOLID (einzeln; läuft jetzt mit ε0 mit), BREAK_DW_MAX
- ±0,002: CONE_DEFAULT_POROSITY
- über U_ref abgedeckt: STOKES_BINDER_VISCOSITY

Die ersten drei stehen auskommentiert in der Config. Jeder kostet ~5 % mehr Läufe und
liefert erwartbar ST ≈ 0.

## Wie rund sind die Partikel?

- **Im Modell ist jedes Partikel eine Kugel**: Der Durchmesser wird aus V_dry berechnet. Auch
  Rumpfs Gleichung (MA Gl. 2.2) setzt kugelförmige Primärpartikel voraus. k (2,2–2,8) und
  k_wet (1,0–1,33) sind genau die Korrekturen dafür.
- **Wie rund Pural TH 130 wirklich ist, steht weder in der MA noch in den Daten, die ich habe.**
  Das Analysenzertifikat liefert nur D50 = 34 µm. Belastbar beantworten ließe sich das mit
  REM-Bildern oder einer Formangabe des Herstellers.
- Für die Studie reicht das: k wird über seinen Literaturbereich variiert, die Unsicherheit
  ist damit abgedeckt. In den Einzeltests wirkt k nicht, weil die Granulate im
  Nass-Regime sind (S > 0,8), wo nur k_wet und cos δ zählen.
