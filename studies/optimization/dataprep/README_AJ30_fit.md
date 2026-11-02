# AJ30 Nassgranulations-Fit — Handoff

Dieser Ordner + die zugehoerigen Dateien bringen die WMCPBE-Simulation so
weit, dass die Ray-Tune-Optimierung gegen die Sieb-Daten aus Versuch **AJ30**
laeuft. Gedacht zum **Wegkopieren auf einen leistungsstarken Rechner** — ein
einzelner Simulationslauf sind viele Millionen MC-Events.

## Dateien

| Datei | Zweck |
|---|---|
| `studies/optimization/opt_config_wet_granulation_full.py` | **Die** Config: alle Kernel-Parameter, `_FIT`-Liste (An/Aus je Parameter), DSMC-Skalierung |
| `studies/optimization/run_wet_granulation_fit.py` | Runner (startet die Optimierung, schreibt `best_params.json`) |
| `studies/optimization/export_wet_granulation_results.py` | Ergebnis-SQLite -> Tabelle (CSV + Excel, verlustfrei), Kurzuebersicht je `status` |
| `studies/optimization/data/AJ_Q3_sieve.xlsx` | **Eingabedaten** für `COST_MODE = "q3_sieve_mae"` (Standard der Config) — Blatt `Q3_sieve`. Liegt im Repo. |
| `studies/optimization/dataprep/prepare_aj_q3_sieve.py` | erzeugt `AJ_Q3_sieve.xlsx` aus den Sieb-Rohdateien aller Reihen (braucht `C:\Studium\...\MVM-IKFT Eirich-Mischer Data\`) |
| `studies/optimization/data/AJ30_xQ.xlsx` | **Eingabedaten** für `COST_MODE = "xq_mse"` — `x(Q, t)`, Blatt `x_Q3`. Liegt im Repo. |
| `studies/optimization/dataprep/prepare_aj30_xq.py` | erzeugt `AJ30_xQ.xlsx` aus der Sieb-Rohdatei (braucht `C:\Studium\...\260713_PSD_SieveTower_AJ30.xlsx`) |
| `studies/optimization/dataprep/prepare_aj30_q3.py` | Nebenprodukt: Q3(x)-Ansicht + Plot (nicht vom Runner benutzt) |
| `studies/optimization/dataprep/compute_aj30_scaling.py` | rechnet die DSMC-Skalierung aus den physik. Eckdaten nach |
| `pbe-optimizer/.../adapter/wet_granulation_adapter.py` | Adapter: Kernel-Verdrahtung, Datenformat, Kostenberechnung, Zeitlimit, `status` je Kandidat |

## Kostenfunktion: Schalter `COST_MODE`  (Eric, 2026-10-02)

Der Schalter sitzt im Adapter (`pop_params["COST_MODE"]`; fehlt der Eintrag, gilt
`"xq_mse"`). Eine Config dient genau EINEM Zweck: `opt_config_wet_granulation_full.py`
setzt `"q3_sieve_mae"` und `DATA_FILE = AJ_Q3_sieve.xlsx` fest; der Runner nimmt
`DATA_FILE` aus der Config. Für die andere Kostenfunktion eine eigene Config anlegen.

| `COST_MODE` | Daten | Vergleich | `score` |
|---|---|---|---|
| `"q3_sieve_mae"` (Fit-Config) | `AJ_Q3_sieve.xlsx`, Blatt `Q3_sieve` | Q3 an den Siebmaschen 100/250/500/800/1400/2000 um, Betrag | dimensionslos 0..1: mittlerer falsch einsortierter Massenanteil je Masche |
| `"xq_mse"` | `AJ30_xQ.xlsx`, Blatt `x_Q3` | x(Q, t) an 21 Quantilen (Betreuer-Konvention, unten) | MSE in m² |

```
J = sum_i  Gewicht_i * (1/6) * sum_k |Q3_sim(x_k, t_i) - Q3_i(x_k)|
```

* **Alle Siebklassen zählen nach Masse**: der Fehler an einer Masche ist genau der
  Massenanteil, der auf der falschen Seite liegt (kein Quadrat, keine Interpolation).
* **Jede Versuchsreihe zählt gleich** (je 1/5): AJ30 Probenreihe (S1..S9, je 1/45),
  AJ26 / AJ29 / AJ31 (10 min ohne Proben, Vollcharge), AJ40 (6 min ohne Proben).
  Die Gewichte stehen als Spalte `Gewicht` im Blatt (von Hand änderbar, Summe = 1).
  Gewicht je Sim-Zeit: 600 s 62 %, 360 s 22 %, die übrigen AJ30-Zeiten zusammen 16 %.
* Nicht dabei: 40-/60-min-Versuche (AJ42, AJ43) -- außerhalb der Sim-Zeit 600 s.
* Erreichbares Minimum wegen der Streuung der 10-min-Wiederholungen: J ≈ 0.0105.
* Blatt `Q3_sieve`: eine Zeile je Messung, Spalten `Reihe | Gruppe | Probe | t_sim_s |
  Gewicht | Sieb_g | 100 | 250 | ... | 2000` (Zahl-Köpfe = Masche in um, Werte = Q3).
  Jede `t_sim_s` muss in `t_vec` stehen.

## Datenformat  x(Q, t)  (Betreuer-Konvention, `COST_MODE = "xq_mse"`)

Blatt `x_Q3` in `AJ30_xQ.xlsx`:

```
Q     | 0     | 60    | 180   | ...      <- Kopf = Sim-Zeit [s]
0.00  | 0.034 | ...   | ...   | ...      <- Werte = Durchmesser x [mm]
0.05  | 0.034 | ...   | ...   | ...
...
1.00  | 0.034 | ...   | ...   | ...
```

Ziel des Fits ist der **Durchmesser x an festen Q3-Quantilen** (die inverse
CDF), nicht der CDF-Wert auf einem Durchmesser-Raster. Quantil-Raster
`Q = 0.00 .. 1.00` in Schritten von 0.05 (21 Werte, **gesamter** Groessenbereich).
Unterhalb der feinsten Sieb-Masche (100 um) ist die Verteilung nicht aufgeloest:
dort wird linear zwischen `(x = 34 um, Q3 = 0)` und 100 um interpoliert.
`x(Q = 1)` = erste Masche, bei der Q3 = 1.0 erreicht ist (sonst 2000 um).

Die 9 Proben S1..S9 sind um **90 s** nach vorne geschoben: Sim-t=0 = Start der
Binderzugabe (Prozess-t=90 s). t=0-Spalte = monodispers 34 um.

## Ausfuehren

Umgebung: conda env `Masterarbeitv4` (numpy, numba, scipy, ray, optuna, pandas, openpyxl).

```bash
# Kette testen (1 Iteration, + Direktauswertung):
python "studies/optimization/run_wet_granulation_fit.py" --check --n-iter 1

# Echter Lauf (n_iter aus der Config: 200 je aktivem _FIT-Parameter = 1600):
python "studies/optimization/run_wet_granulation_fit.py"

# Fortsetzen (Historie 500.sqlite, 500 weitere Kandidaten):
python "studies/optimization/run_wet_granulation_fit.py" --resume 500 --n-iter 500

# Auswerten (neueste <N>.sqlite im Ordner -> <N>_table.csv / .xlsx):
python "studies/optimization/export_wet_granulation_results.py" studies/optimization/wet_granulation_opt_results
```

Cluster-Job wie beim Lauf 1600: 8 Kerne, 72 h, Start mit dem Runner oben.
Warnungen/Fehler der Ray-Worker stehen im Job-Log; Status und Fehlergrund jedes
Kandidaten zusaetzlich in der SQLite.

Was optimiert wird, stellst du in `opt_config_wet_granulation_full.py` unter
`_FIT` ein (`True`/`False` je Parameter). Aktuell aktiv (9): `AGG_A`, `EKE_CORR_BETA`,
`STOKES_U_COLL_REF`, `BREAK_P1`, `BREAK_P2`, `COMPRESSION_RATE`,
`LIQUID_INTERNALIZATION_RATE`, `CONE_K_AGG`, `CONE_K_BREAK`.

Aggregation = `kinetic_blend` (Eric, 2026-10-02): Teilchengeschwindigkeit v ~ m^(-AGG_A/2),
`AGG_A` von 0 (alle gleich schnell) über 1 (= `eke_darelius2005`, bitgleich) bis 2
(ETM); der Adapter gibt `blend = AGG_A` unverändert an den Kernel (seit 2026-10-06;
Läufe davor, z. B. 3600_blend_nucfix, nutzten `blend = 2 * AGG_A`). `EKE_CORR_BETA` bleibt der
Vorfaktor corr_beta -- seine sinnvolle Größenordnung hängt stark von a ab.

## Laufzeit-Stellschrauben  (alle in `opt_config_wet_granulation_full.py`)

| Schluessel | Wert | Bedeutung |
|---|---|---|
| `algo_params["max_iter_time"]` | 1170 s | Zeitlimit **pro Kandidat** (alle Replikate zusammen); 0 = aus. Herleitung: 72 h Job - 7 h Puffer = 65 h fuer 1600 / 8 = 200 Kandidaten je Kern -> 1170 s. Garantiert fertig in 65 h, auch wenn jeder Kandidat das Limit ausschoepft. |
| `algo_params["max_concurrent"]` | 8 | parallele Kandidaten (= Kerne, die der Job braucht) |
| `algo_params["checkpoint_interval"]` | 50 | alle 50 fertigen Kandidaten eine Zwischen-SQLite `<n>.sqlite` (bei Job-Abbruch gehen hoechstens 49 verloren) |
| `opt_params["max_reuse"]` | 10000 | **nicht klein setzen** — bei `2` verlor jeder Worker nach 2 Kandidaten die Messdaten (Cluster-Lauf 1600: 1547 von 1550 Kandidaten = Score 10) |
| `pop_params["N_MC_REPLICATES"]` | 10 | Replikate je Kandidat; zusammen mit `max_iter_time` abwaegen |

Laeuft das Zeitlimit ab, stoppt das laufende Replikat, die restlichen starten
nicht. Ist mindestens 1 Replikat fertig, wird ueber die fertigen gemittelt und
normal bewertet (`status = timeout_partial`); sonst Score 10.1 (`status = timeout`).

## DSMC-Skalierung  (Experiment -> Simulation)

Das Kontrollvolumen `Vc` ist **[m³]** (nicht dimensionslos). Der dimensionslose
Knopf ist `SIM_FRACTION = f = Vc / V_mixer_exp`. Alles Vc-Abhängige wird daraus
abgeleitet — `f` ist die *einzige* Zahl, die du zum Verkleinern änderst.

```
EXPERIMENT (fest, Vc-invariant):
  V_mixer_exp  = 1.0 L
  m_solid_exp  = 200.06 g,  d0 = 34 um,  rho_eff = rho_Boehmit * 0.2 = 610 kg/m^3
  Q_binder_exp = 106 mL / 60 s = 1.7667e-6 m^3/s
  -> n_real   = N_real / V_mixer_exp        = 1.5937e13  1/m^3   (Anzahldichte)
  -> q_binder = Q_binder_exp / V_mixer_exp  = 1.7667e-3  1/s     (Binder-Ratendichte)

SIMULATION (Wahl: f ; N_comp FEST):
  N_comp              = 1000   (FEST -- Zahl der stochastischen Partikel = statist.
                                PSD-Aufloesung; W ist nur Mengen-Buchhaltung)
  Vc                  = f * V_mixer_exp
  INITIAL_WEIGHT (W)  = n_real * Vc / N_comp   (= N_real * f / N_comp ; skaliert mit f)
  VOLUMETRIC_FLOW_RATE = q_binder * Vc  (= Q_binder_exp * f)
  LIQUID_ADDITION_DURATION = 60 s   (Zeit, Vc-invariant)
  alle Kernel-Params, DROPLET_DIAMETER, N_MIXER, INITIAL_DIAMETER: Vc-invariant
```

Speedup bei `f < 1`: ~1/f (Events ∝ f, Kosten/Event = O(N_comp²) bleibt).
Preis: `dW_max/W` waechst ~1/f (jedes Parcel wird pro Event relativ groeber
abgebaut). Bei Bedarf `AGG/BREAK_DW_MAX` ~f mitziehen -- kostet dann aber den
Speedup wieder.

**`volumetric_flow_rate` ist NICHT die rohe Pumpen-Flussrate**, sondern
`q_binder * Vc` (= Q_exp * f). Bei `f = 1` fällt das zufällig zusammen; bei
`f < 1` musst du skalieren. Die Config macht das automatisch; ein `assert` prüft
`sum(W)/Vc = n_real` und `flow*Dauer/Vc = phi_binder` bei jedem Laden.

Tabelle für beliebiges f:  `python studies/optimization/dataprep/compute_aj30_scaling.py --fraction f`

Warum das erlaubt ist: alle Kernels sind **intensiv** (hängen nur von
Partikelgrößen + `n_real` ab). Nur `sum(W)` und `L_total` sind extensiv, beide
∝ Vc. Der Code selbst nutzt das (`_maybe_double_control_volume`: Vc x2 ->
Partikel x2, W gleich).

**Zwei Vorbehalte:**
- *PSD-Auflösung bleibt (N_comp fest), aber die „Mengen"-Granularität wird
  gröber:* bei kleinerem `f` ist `dW_max/W` größer → jedes Parcel fließt in
  gröberen Schritten in Agglomerate ab. Zweitrangig ggü. der Größen-Auflösung,
  aber im Blick behalten (bzw. `dW_max` ~f mitziehen).
- *Downscaling setzt räumlich homogenen Binder voraus* (der Ausschnitt Vc muss
  repräsentativ sein). Real wird an einer Düse gesprüht; über 60 s im
  Hochleistungsmischer plausibel, aber eine Annahme.

## Ergebnisse

| Ort | Inhalt |
|---|---|
| `studies/optimization/wet_granulation_opt_results/best_params.json` | bester Parametersatz + Score |
| `studies/optimization/wet_granulation_opt_results/<N>.sqlite` | volle Warm-Start-Historie, Tabelle `warm_params` (Spalten s. u.); Endstand `N = n_iter + n_iter_prev`, dazu Zwischenstaende alle `checkpoint_interval` |
| `studies/optimization/wet_granulation_ray/AJ30_xQ/` | Ray-Tune-Rohdaten (pro Trial `params.json`, `result.json`) |

Spalten von `warm_params` — eine Zeile je Kandidat, `id` = Reihenfolge der
Fertigstellung (in jeder Datei ab 1):

| Spalte | Inhalt |
|---|---|
| `id`, `data_name`, `param_json`, `score` | wie bisher; `score` = Kosten nach `COST_MODE`: J (0..1, `q3_sieve_mae`) bzw. MSE zwischen simuliertem und gemessenem `x(Q, t)` [m², `xq_mse`] |
| `status` | Ergebnisart, siehe unten |
| `n_rep_ok` / `n_rep_total` | ausgewertete / geplante MC-Replikate |
| `eval_seconds` | Rechenzeit des Kandidaten [s] |
| `failure_reason` | Klartext je gescheitertem Replikat, z. B. `seed=...: reached MAX_PARTICLES` |
| `worker_pid`, `reuse_num` | welcher Ray-Worker, wie oft dieser schon wiederverwendet war |
| `finished_at` | Uhrzeit der Fertigstellung |

| `status` | Bedeutung | `score` |
|---|---|---|
| `ok` | alle Replikate fertig | Kosten (J bzw. MSE) |
| `partial` | einige Replikate gescheitert, Mittel ueber den Rest | Kosten (J bzw. MSE) |
| `timeout_partial` | Zeitlimit, >= 1 Replikat fertig (Mittel ueber die fertigen) | Kosten (J bzw. MSE) |
| `timeout` | Zeitlimit, kein Replikat fertig | **10.1** |
| `failed` | alle Replikate gescheitert (z. B. `MAX_PARTICLES`) | 10 |
| `setup_error` | Programmfehler vor der Simulation | 10 |

Dateien aus Laeufen vor dieser Erweiterung (z. B. `wet_granulation_opt_1600`)
haben nur die ersten 4 Spalten; der Export-Skript liest beide.

## Offene Punkte / Annahmen (bitte pruefen)

1. **DSMC-Skalierung**: siehe eigener Abschnitt oben. Aktuell `SIM_FRACTION = 1e-6`
   (`1.0` = voller Mischer). Zum Verkleinern nur diese Zahl aendern.
2. **`EKE_CORR_BETA` = 5.75e-14** ist eine *physikalische Groessenordnungs-
   Schaetzung* (Agg-Rate ~0.03/s), NICHT der REF-Wert 5e-8 und kein Messwert.
   `_FIT`-Bounds aktuell `(-16, -6)`.
3. **`AGG_DW_MAX` / `BREAK_DW_MAX` = 5 / 10** (physik. Kollisionen pro MC-Event):
   bewusst klein (Genauigkeit, viele Events sind gewollt). Hochsetzen =
   schneller, aber gröber. Dominanter Kosten-/Genauigkeits-Knopf.
4. **`NUCLEATION_BATCH_SIZE = 20`** bewusst klein (Genauigkeit). Hochsetzen
   beschleunigt, verfaelscht die Verteilung.
5. **Tropfengroesse 150 um** ist eine erste Naeherung (Eric sucht bessere Daten).
6. **Laufzeit**: Auf dem Laptop ~14 Events/s (pairwise O(n^2), kinetic_blend hat keine
   Momentenform). Ein voller Lauf braucht die Rechenleistung des Zielrechners;
   ggf. `max_concurrent` in der Config hochsetzen.
7. **S1 (Sim-t=60 s)** ist in den Daten auffaellig (erste Probe direkt nach
   Binderzugabe, vor Homogenisierung) — beim Bewerten des Fits im Blick behalten.
8. **`STOKES_RHO_SOLID = 600`** ist der REF-Wert; skelettaler Boehmit ~3050.
