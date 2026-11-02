"""
KOSTENFUNKTIONS-VARIANTE KF4 (Eric, 2026-10-14) -- Daten wie KF2 (AJ40 6 min -> 360 s, Mittelwert
AJ26/AJ29/AJ31 10 min -> 600 s, je 1/2), aber verglichen gegen die GESAMTE Messlinie: Q3 zwischen
den Siebmaschen linear in log10(x) interpoliert, nur 100..2000 um (keine Extrapolation), 102
Stuetzstellen inkl. der echten Maschen, Trapezgewichte -> J = mittlere Flaeche zwischen den Kurven.
Daten: studies/optimization/data/KF4_NoSampling_Q3_interp.xlsx (Skript: studies/optimization/dataprep/prepare_kf4_interp.py)
Ergebnisse: studies/optimization/wet_granulation_opt_results_kf4_nosampling_interp/
Alles andere ist eine Kopie von opt_config_wet_granulation_full.py (Stand 2026-10-06);
die Varianten KF1/KF2/KF3/KF4 unterscheiden sich NUR in DATA_FILE und RESULT_DIR.

Konfigurations-/Parameter-Vorlage fuer das Fitten der Nassgranulations-
Simulation (WMCPBE) gegen die Sieb-Daten der Pural-TH130-Versuche (AJ30 mit
Probenahme + AJ26/AJ29/AJ31/AJ40 ohne Probenahme).

ZWECK
-----
Eine Datei, in der JEDER Parameter der acht Ziel-Kernels steht:

    Aggregation .................... kinetic_blend
    Agglomerations-Akzeptanz ...... stokes_dynamik
    Breakage ...................... powerlaw_rumpf_dynamic
    Porositaets-Kompression ....... porosity_compression_dynamik
    Fluessigkeits-Internalisierung  liquid_internalization
    Internalisierung b. Agglo. ... liq_internalisation_agglomeration   (parameterlos)
    Fluessigkeitsverteilung ...... uniform_weighted                    (parameterlos)
    Porositaets-Wachstum ......... cone_model

`pop_params` = alle festen ("geratenen") Werte + Wertequelle fuer alles, was
NICHT optimiert wird.
`_FIT` (unten) = die sichtbare An/Aus-Liste: pro Zeile ein Parameter mit
True/False und Suchraum. `opt_params` wird daraus erzeugt.

STAND (2026-10-02)
------------------
* Daten + Kostenfunktion (Eric 2026-10-02):
    studies/optimization/data/AJ_Q3_sieve.xlsx, Blatt "Q3_sieve"  (Skript: studies/optimization/dataprep/prepare_aj_q3_sieve.py)
    COST_MODE "q3_sieve_mae": Q3 an den Siebmaschen 100..2000 um, Betrag (1 % falsch
    einsortierte Masse = 1 % Einfluss). Jede Versuchsreihe gleich gewichtet: AJ30
    Probenreihe, AJ26/AJ29/AJ31 (10 min), AJ40 (6 min) je 1/5; AJ30-Proben je 1/45.
    Score J dimensionslos 0..1 = mittlerer falsch einsortierter Massenanteil je Masche.
* Aggregation (Eric 2026-10-02): kinetic_blend, Teilchengeschwindigkeit v ~ m^(-AGG_A/2).
  AGG_A = blend des Kernels (seit 2026-10-06, vorher 2*AGG_A) wird mitgefittet
  (0 = gleich schnell, 1 = EKE, 2 = ETM); Vorfaktor
  corr_beta = EKE_CORR_BETA wie bisher.
* Feste Kernel-Werte = aktueller Stand aus test_estimations_for_fit.py::TestConfig
  (Eric-Handtuning, 2026-09-10). Physik-Startwerte, KEINE Kalibrierung.
* Jede Kandidaten-Bewertung = Mittelwert aus N_MC_REPLICATES (=10) MC-Laeufen
  mit 10 FESTEN, verschiedenen Seeds; J des Mittelwerts gegen das Experiment.
  Gesucht wird mit CMA-ES (method="Cmaes").
* Adapter (pbe-optimizer/.../wet_granulation_adapter.py):
    - get_all_data liest Blatt "Q3_sieve", vergleicht Q3 an den Siebmaschen
    - Kernel-Verdrahtung = obiges Set (Vorlage: mcpbe/tests/bench/scenarios.py
      :: granulation_rumpf_dynamic_1d)
    - monodisperse Startpopulation aus INITIAL_DIAMETER
* DSMC-/Prozess-Skalierung: Experiment-Groessen (Vc-invariant) fest eingetragen,
  daraus per SIM_FRACTION (= Vc / V_mixer_exp) die Simulationsgroessen abgeleitet.
  f=1.0 = voller Mischer. Zum Verkleinern nur SIM_FRACTION aendern -- Vc, W_init
  und flow_rate skalieren ~f ; _N_COMP bleibt FEST (statist. PSD-Aufloesung).
  Ein assert prueft die zwei Invarianten (sum(W)/Vc = n_real ; flow*Dauer/Vc = phi_binder).
  Nachrechnen/Tabelle: studies/optimization/dataprep/compute_aj30_scaling.py --fraction f
* Diese Config ist zum Wegkopieren auf einen leistungsstarken Rechner gedacht;
  hier NICHT die volle Optimierung laufen lassen. Ein Lauf = viele Mio. MC-Events.

Zeitachse (Entscheidung Eric):
  Sim-t=0 == Prozess-t=90 s (Start Binderzugabe). Trockenmischen wird nicht
  simuliert. Alle Probenzeiten um 90 s nach vorne geschoben:
      S1..S9 (Prozess) 150,270,330,390,450,510,570,630,690
      S1..S9 (Sim)      60,180,240,300,360,420,480,540,600
  t_vec[0]=0 = Startzustand (monodispers d0=34 um).

Wertequellen:
  [KERNEL]      Kernel-Default
  [REF]         mcpbe/tests/bench/scenarios.py :: granulation_rumpf_dynamic_1d
                (lauffaehiger Golden-Reference-Fall, synthetisch, KEINE AJ30-Kalibrierung)
  [EXP]         Versuch AJ30 (260713_LabRecord_AJ30.xlsx)
  [SKALIERUNG]  aus den physik. Eckdaten unten hergeleitet
  [PHYS-EST]    physikalische Groessenordnungs-Schaetzung (kein Messwert)
"""

from pathlib import Path

import numpy as np


TRIALS_DIR = Path(__file__).resolve().parent

# Datenblatt (der Runner liest DATA_FILE aus dieser Config):
DATA_FILE = TRIALS_DIR / "data" / "KF4_NoSampling_Q3_interp.xlsx"     # data_name = "KF4_NoSampling_Q3_interp"
RESULT_DIR = TRIALS_DIR / "wet_granulation_opt_results_kf4_nosampling_interp"   # eigener Ordner je Kostenfunktion


# ===========================================================================
#  mixer speed -- EINE gemeinsame Zahl fuer alle drehzahlgetriebenen Kernels
# ===========================================================================
# # [EXP] Whirler speed waehrend Binderzugabe + Granulation = 30 m/s.
N_MIXER = 30.0


# ===========================================================================
#  DSMC-/Prozess-Skalierung   (Nachrechnen: studies/optimization/dataprep/compute_aj30_scaling.py)
# ===========================================================================
#
# --- EXPERIMENT  (fest, Vc-INVARIANT) ---------------------------------------
#   Feststoff-Startmasse .. 200.06 g   Pural TH130 (Laborprotokoll)
#   Primaerpartikel ....... rund, x = 34 um, monodispers
#   Effektiver Dichte rho_Boehmit * 0.2 = 3050 * 0.2 = 610 kg/m^3
#   Mischervolumen ........ 1 L
#   Binder ............... 106 g Wasser in 60 s
#
#   m_primary  = 610 * pi/6 * (34e-6)^3          = 1.2554e-11 kg
#   _N_REAL    = 0.20006 / m_primary             = 1.5937e10   reale Partikel im 1-L-Mischer
#   _Q_BINDER_EXP = (0.106/1000) / 60            = 1.7667e-6 m^3/s   Pumpen-Flussrate
#   -> Vc-invariant: n_real     = _N_REAL / _V_MIXER_EXP           = 1.5937e13 1/m^3
#   phi_binder = _Q_BINDER_EXP*_DURATION_S/_V_MIXER_EXP = 0.106

_V_MIXER_EXP  = 1.0e-3
_N_REAL       = 1.5937e10
_Q_BINDER_EXP = 1.7667e-6
_DURATION_S = 60

# --- SIMULATIONS-AUSSCHNITT  (die EINE Wahl) -------------------------------
#   SIM_FRACTION = f = Vc / V_mixer_exp .
SIM_FRACTION = 1e-7
_N_COMP      = 500            # FEST -- statistische Aufloesung der PSD

# --- daraus abgeleitet -------------------------------------------------
_VC        = SIM_FRACTION * _V_MIXER_EXP              # Kontrollvolumen [m^3]
_W_INIT    = SIM_FRACTION * _N_REAL / _N_COMP         # = n_real * Vc / N_comp ; skaliert mit f
_FLOW_RATE = SIM_FRACTION * _Q_BINDER_EXP             # = Ratendichte q_binder * Vc  [m^3/s]




# ===========================================================================
#  pop_params  --  ALLE Werte (fest, solange nicht in _FIT auf True)
# ===========================================================================
pop_params = {

    # ----- 0. mixer speed (gemeinsam) --------------------------------
    "N_MIXER": N_MIXER,                       # 30 m/s  [EXP]

    # ----- Kostenfunktion  (siehe Kopf) ------------------------------------
    "COST_MODE": "q3_sieve_mae",             # Q3 an den Siebmaschen, Betrag, alle Versuchsreihen

    # ----- 1. Monte-Carlo / DSMC ------------------------------------------
    "MC_SEED": 7,                          # BASIS-Seed; die N_MC_REPLICATES Seeds
                                            #   werden daraus fest abgeleitet (SeedSequence)
    "N_MC_REPLICATES": 10,                  # jede Bewertung = Mittelwert aus 10 MC-Laeufen
                                            #   (10 feste, verschiedene Seeds). MC-Rauschen /sqrt(10).
    "INITIAL_PARTICLES": _N_COMP,           # = _N_COMP (fest, statist. Aufloesung)  [SKALIERUNG]
    "INITIAL_WEIGHT": _W_INIT,              # = SIM_FRACTION * _N_REAL / _N_COMP     [SKALIERUNG]
    "CONTROL_VOLUME": _VC,                  # = SIM_FRACTION * _V_MIXER_EXP [m^3]    [SKALIERUNG]
    "MERGER_TOLERANCE": 1e-6,
    "MERGER_LOOKUP": "scan",                # "scan" = KEIN Hash-Index (jeden Event neu-keyen kostet mehr; 431->55 s auf rumpf_full). "hash" nur bei Bedarf.
    "MAXITER": 50_000_000,                  # "nicht abbrechen"-Guardrail; ein Lauf = viele Mio. Events
    "MAX_PARTICLES": 100_000,               # Guardrail; muss > INITIAL_PARTICLES sein
    "MAYBE_DOUBLE_VC": True,                # DSMC-Vc-Verdopplung bei zu wenig Partikeln
    "AGG_DW_MIN": 1.0,
    "AGG_DW_MAX": 5,                        # physik. Kollisionen/Event; klein=genau  [SKALIERUNG]
    "BREAK_DW_MAX": 10,                     #                                        [SKALIERUNG]
    "SIZEEVAL": 0,                          # [REF]

    # ----- 2. Ausgangspulver / Anfangszustand ---------------------------
    # [EXP] Pural TH130 (Boehmit), Feststoff-Start 200.06 g, Ausbeute 145.27 g.
    "INITIAL_DIAMETER": 34e-6,                # monodispers, 34 um   [EXP: Eric]
    "INITIAL_POROSITY": 0.0,                  # Vollkoerper-Start; Porositaet entsteht ueber cone_model

    # ----- 3. Fluessigkeitszugabe / Nucleation -------------------------
    # [EXP] 106 g Binder (Wasser) ueber 60 s.
    "DROPLET_DIAMETER": 150e-6,               # 150 um (erste Naeherung über Hochdruckreiniger)   [EXP]
    "VOLUMETRIC_FLOW_RATE": _FLOW_RATE,       # = SIM_FRACTION * _Q_BINDER_EXP [m^3/s]  [EXP/SKALIERUNG]
    "LIQUID_ADDITION_START": 0.0,             # Sim-Zeit Start (= Prozess 90 s)
    "LIQUID_ADDITION_DURATION": 60,           # 60 s (Prozess 90..150 s), Vc-invariant  [EXP]
    "NUCLEATION_BATCH_SIZE": 20.0,            # phys. Tropfen je Nucleation-Event.
    

    # ----- 4. kinetic_blend  (Aggregation) -----------------------------
    #   beta = corr_beta * n_mixer^c_mixer * (r1+r2)^2 * sqrt(r1^(-3b) + r2^(-3b)),  b = AGG_A
    #   (Adapter: blend = AGG_A; AGG_A = 1 ist eke_darelius2005 bitgleich)
    "AGG_A": 1.0,                             # blend: v ~ m^(-AGG_A/2); 0 gleich schnell, 1 = EKE, 2 = ETM
    "EKE_CORR_BETA": 5.75e-14,                # 5.75e-14 [PHYS-EST]  corr_beta von kinetic_blend
    "EKE_C_MIXER": 0.0995,                    # DEM C_FREQ                 [KERNEL]

    # ----- 5. stokes_dynamik  (Agglomerations-Akzeptanz) --------------
    #   St < St_crit -> Partikel bleiben zusammen ; U_coll = U_coll_ref * n_mixer^c_vel
    "STOKES_U_COLL_REF": 0.0794,             # [m/s] DEM-Fit-Vorfaktor     [REF]
    "STOKES_C_VEL": 0.2852,                  # DEM C_VEL                   [KERNEL]
    "STOKES_BINDER_VISCOSITY": 0.1,          # [Pa.s]                      [REF]
    "STOKES_RHO_SOLID": 600.0,               # [kg/m^3] REF-Wert; Boehmit skelettal ~3050  [REF]
    "STOKES_RHO_LIQUID": 1000.0,             # [kg/m^3] Wasser             [REF]
    "STOKES_H_A": 500e-9,                    # min. Filmdicke / Rauheit [m]  [REF]

    # ----- 6. powerlaw_rumpf_dynamic  (Breakage) ---------------------
    #   S = p1 * n_mixer^c_mixer * V^p2 / sigma(porositaet, saettigung)
    "BREAK_P1": 1e13,                        # Basisrate-Vorfaktor         [REF]
    "BREAK_P2": 1.0,                         # Volumen-Exponent            [REF]
    "BREAK_C_MIXER": 0.6699,                 # DEM C_BREAK                 [KERNEL]
    "BREAKRVAL": 4,                          # Ratenmodell-Variante 1..4   [REF]
    "RUMPF_K": 2.5,                          # trocken, Bereich [2.2, 2.8]  [REF]
    "RUMPF_ALPHA": 1.0,                      # nass,    Bereich [1.0, 1.33] [REF]
    "SURFACE_TENSION": 0.072,                # gamma [N/m] (Wasser)        [REF]
    "CONTACT_ANGLE": 0.0,                    # delta [rad], [0, pi/2)      [REF]
    "BREAK_X_S": None,                       # Sauterdurchmesser [m]; None -> aus solver.X0  [REF]
    "BREAK_PORO_MAX": 0.9999,                # obere Porositaets-Schranke fuer sigma  [KERNEL]

    # ----- 7. porosity_compression_dynamik ----------------------------
    #   k = rate * n_mixer^c_mixer ; eps(t+dt) = eps_min + (eps-eps_min)*exp(-k*dt)
    "COMPRESSION_RATE": 1.3441e-2,           # 1.3441e-2  (k=0.1 @ n_mixer=20)  [REF-kalibriert]
    "MIN_POROSITY": 0.2,                     # eps_min                     [REF]
    "COMPRESSION_C_MIXER": 0.6699,           # DEM C_BREAK                 [KERNEL]

    # ----- 8. liquid_internalization --------------------------------
    #   dl_intern/dt = k_int * l_ex * (v_pore - l_intern)
    "LIQUID_INTERNALIZATION_RATE": 1e10,     # k_int [1/(m^3 s)]   Eric-Handtuning 2026-09-10 (REF 1e11)

    # ----- 9. liq_internalisation_agglomeration  (parameterlos) ------
    "USE_LIQ_INTERNALISATION_AGGLOMERATION": True,   # nur an/aus

    # ----- 10. uniform_weighted  (parameterlos) ---------------------
    "USE_UNIFORM_WEIGHTED": True,                    # nur an/aus

    # ----- 11. cone_model  (Porositaets-Wachstum) ------------------
    "CONE_K_AGG": 1.0,                       # Formfaktor Agglo., Bereich [0,1]   [KERNEL/REF]
    "CONE_K_BREAK": 1.0,                     # Formfaktor Breakage, Bereich [0,1] (>1 warnt)  [KERNEL]
    "CONE_DEFAULT_POROSITY": 0.0,            # Startporositaet frisch nukleierter Partikel  [KERNEL/REF]
}


# ===========================================================================
#  Konsistenz-Check der Skalierung  (faellt beim Laden, wenn etwas driftet)
# ===========================================================================
# Prueft die zwei Vc-Invarianten auf den TATSAECHLICHEN pop_params-Werten --
# faengt also auch versehentliches Hand-Editieren einzelner Werte ab.
_n_density_target = _N_REAL / _V_MIXER_EXP
_phi_binder_target = _Q_BINDER_EXP * _DURATION_S / _V_MIXER_EXP
_n_density_have = (pop_params["INITIAL_PARTICLES"] * pop_params["INITIAL_WEIGHT"]
                   / pop_params["CONTROL_VOLUME"])
_phi_binder_have = (pop_params["VOLUMETRIC_FLOW_RATE"] * pop_params["LIQUID_ADDITION_DURATION"]
                    / pop_params["CONTROL_VOLUME"])
assert abs(_n_density_have / _n_density_target - 1.0) < 1e-3, (
    f"n_density inkonsistent: {_n_density_have:.4e} statt {_n_density_target:.4e} "
    f"(sum(W)/Vc muss = n_real sein)")
assert abs(_phi_binder_have / _phi_binder_target - 1.0) < 1e-3, (
    f"Binderkonzentration inkonsistent: {_phi_binder_have:.4f} statt {_phi_binder_target:.4f} "
    f"(flow_rate*Dauer/Vc muss = phi_binder sein -> flow_rate mit SIM_FRACTION skalieren!)")


# ===========================================================================
#  _FIT  --  die sichtbare An/Aus-Liste
# ===========================================================================
# Pro Zeile:   "NAME": (FIT?, suchraum)
#   FIT? = True   -> wird optimiert (pop_params-Wert wird ignoriert)
#   FIT? = False  -> bleibt fest auf pop_params[NAME]
# Suchraum:
#   {"bounds": (lo, hi), "log_scale": True}   -> Suchraum 10**lo .. 10**hi
#   {"bounds": (lo, hi), "log_scale": False}  -> Suchraum lo .. hi (linear)
#   {"int_bounds": (lo, hi)}                  -> ganze Zahlen lo..hi
#   {"choices": [a, b, c]}                    -> diskrete Auswahl
#
# Grenzen unten sind VORSCHLAEGE -- bitte pruefen.
_FIT = {
    # -- mixer speed (gemeinsam) ---------------------------------------
    "N_MIXER":                     (False, {"bounds": (5.0, 40.0),    "log_scale": False}),

    # -- 4. kinetic_blend ---------------------------------------------
    "AGG_A":                       (True,  {"bounds": (0.0, 2.0),     "log_scale": False}),
    "EKE_CORR_BETA":               (True,  {"bounds": (-16.0, -6.0), "log_scale": True}),   
    "EKE_C_MIXER":                 (False, {"bounds": (0.0, 0.3),     "log_scale": False}),

    # -- 5. stokes_dynamik -----------------------------------------
    "STOKES_U_COLL_REF":           (True, {"bounds": (0.0, 1),    "log_scale": False}),
    "STOKES_C_VEL":                (False, {"bounds": (0.0, 0.6),     "log_scale": False}),
    "STOKES_BINDER_VISCOSITY":     (False, {"bounds": (-3.0, 1.0),    "log_scale": True}),
    "STOKES_H_A":                  (False, {"bounds": (-8.0, -6.0),   "log_scale": True}),

    # -- 6. powerlaw_rumpf_dynamic --------------------------------
    "BREAK_P1":                    (True,  {"bounds": (10.0, 20.0),   "log_scale": True}),
    "BREAK_P2":                    (True, {"bounds": (0.5, 2.0),     "log_scale": False}),
    "BREAK_C_MIXER":               (False, {"bounds": (0.0, 1.5),     "log_scale": False}),
    "BREAKRVAL":                   (False, {"choices": [1, 2, 3, 4]}),
    "RUMPF_K":                     (False, {"bounds": (2.2, 2.8),     "log_scale": False}),
    "RUMPF_ALPHA":                 (False, {"bounds": (1.0, 1.33),    "log_scale": False}),
    "CONTACT_ANGLE":               (False, {"bounds": (0.0, 1.4),     "log_scale": False}),

    # -- 7. porosity_compression_dynamik ------------------------
    "COMPRESSION_RATE":            (True, {"bounds": (-5.0, -0.5),   "log_scale": True}),   
    "MIN_POROSITY":                (False, {"bounds": (0.15, 0.45),   "log_scale": False}),
    "COMPRESSION_C_MIXER":         (False, {"bounds": (0.0, 1.5),     "log_scale": False}),

    # -- 8. liquid_internalization ------------------------------
    "LIQUID_INTERNALIZATION_RATE": (True,  {"bounds": (9.0, 13.0),    "log_scale": True}),

    # -- 11. cone_model ----------------------------------------
    "CONE_K_AGG":                  (True, {"bounds": (0, 1.0),     "log_scale": False}),
    "CONE_K_BREAK":                (True, {"bounds": (0, 1.0),    "log_scale": False}),
    "CONE_DEFAULT_POROSITY":       (False, {"bounds": (0.0, 0.6),     "log_scale": False}),

    # -- Anfangszustand / Prozess (falls unsicher) --------------
    "INITIAL_DIAMETER":            (False, {"bounds": (-5.3, -4.3),   "log_scale": True}),   # 5..50 um
    "INITIAL_POROSITY":            (False, {"bounds": (0.0, 0.55),    "log_scale": False}),
    "DROPLET_DIAMETER":            (False, {"bounds": (-5.0, -3.0),   "log_scale": True}),
}

# opt_params aus _FIT bauen (Infrastruktur-Schluessel immer dabei)
opt_params = {
    "actor_wait": {"fixed": False},
    "wait_time": {"fixed": 0.0},
    # max_reuse: nach so vielen Kandidaten baut ein Ray-Worker seinen Adapter neu
    # (nur Schutz gegen Speicherlecks, laut Betreuer nicht zwingend noetig).
    # NICHT klein setzen: der neu gebaute Adapter bekommt die Messdaten nicht
    # zurueck (reset_config ruft get_all_data nicht auf) -> jeder weitere Kandidat
    # dieses Workers endet sofort mit status=setup_error / Score 10.
    # So geschehen im Cluster-Lauf wet_granulation_opt_1600 mit max_reuse=2.
    # 10000 >> Kandidaten pro Worker (n_iter / max_concurrent) -> nie neu gebaut.
    "max_reuse": {"fixed": 10000},
}
opt_params.update({name: spec for name, (fit, spec) in _FIT.items() if fit})

# Anzahl aktiv gefitteter Parameter (Information); Iterationszahl fest
# (Eric, 2026-10-07: 2000 statt Daumenwert 200/Param = 1800)
_N_ACTIVE_FIT = sum(1 for (fit, _spec) in _FIT.values() if fit)
_N_ITER = 2000


# ===========================================================================
#  config  --  wird vom Runner geladen
# ===========================================================================
config = {
    "multi_flag": False,
    "single_case": True,

    "algo_params": {
        "adapter": "wet_granulation",
        "dim": 1,

        # t_vec[0]=0 -> Startzustand (monodispers d0). Danach die 9 Sim-Zeiten.
        "t_init": np.array([0.0, 0.0]),
        "t_vec": np.array([0.0, 60.0, 180.0, 240.0, 300.0, 360.0, 420.0,
                           480.0, 540.0, 600.0]),
        "delta_t_start_step": 0,

        "sample_num": 1,
        "exp_data": False,
        "calc_init_N": False,
        "data_flag": "Q3",        # zugrundeliegende Verteilung (Masse); Ziel ist Q3 an den Siebmaschen
        "cost_flag": "MSE",       # nicht benutzt -- J rechnet der Adapter (COST_MODE, siehe Kopf)

        "method": "Cmaes",        # <- CMA-ES (Optuna CmaEsSampler). GP | TPE | Cmaes | NSGA | QMC
        "random_seed": 42,
        "n_iter": _N_ITER,        # = 200 * (Anzahl aktiver _FIT-Parameter)  [Daumenwert]

        "tune_storage_path": str(TRIALS_DIR / "wet_granulation_ray"),
        "verbose": 1,             # Ray-eigene Tabelle; die [fit N/M]-Zeile kommt vom Progress-Callback
        "multi_jobs": False,
        "cpus_per_trail": 1,
        # nur die n_iter-Trials parallelisieren (die 10 MC-Replikate bleiben seriell
        #   im Adapter -- sonst verschluckt sich der Rechner). CMA-ES mag 2..12.
        "max_concurrent": 8,

        # Zeitlimit pro KANDIDAT (alle N_MC_REPLICATES zusammen) [s]; 0 = kein Limit.
        #   Laeuft es ab: laufendes Replikat stoppt, restliche starten nicht.
        #   >= 1 Replikat fertig -> ausgewertet (status=timeout_partial),
        #   keins fertig -> Score 10.1 (status=timeout).
        #   Eric, 2026-10-07: 2 h = 7200 s. Im Lauf 3600_blend_nucfix brauchten die
        #   guten Kandidaten (Score < 0.10, echte Aggregation) 750..2400 s; je hoeher
        #   die Aggregationsrate, desto laenger. 2 h deckt sie mit Puffer ab.
        #   KEINE 65-h-Garantie mehr (frueher 1170 s, aus 72-h-Cluster-Job hergeleitet):
        #   liegen viele Kandidaten im langsamen Bereich, dauert ein Lauf laenger ->
        #   ueber checkpoint_interval / --resume fortsetzen.
        "max_iter_time": 7200,

        # Alle N fertigen Kandidaten eine Zwischen-SQLite <n>.sqlite schreiben; 0 = aus.
        #   (Cluster-Lauf 1600 lief mit 50.)
        "checkpoint_interval": 50,
    },

    "pop_params": pop_params,
    "opt_params": opt_params,
}
