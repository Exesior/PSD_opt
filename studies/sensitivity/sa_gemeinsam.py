"""
Gemeinsame Einstellungen ALLER Studien-Configs: Basisfall, Versuchsdaten, Kennzahlen, Grenzwerte.

Die Studien-Configs (sa_config_*.py) holen sich das mit `from sa_gemeinsam import *` und
legen selbst nur fest, WAS variiert wird und wie gross die Studie ist.

Werte sind feste Zahlen; Herleitungen stehen als Kommentar daneben (keine Live-Rechnung).
"""

# ===========================================================================
#  Basisfall = KF3-Config (studies/optimization/opt_config_wet_granulation_kf3_nosampling_logw.py)
#              + Bestparameter id 1070 + N_COMP 125
# ===========================================================================
# Einzige Aenderung gegenueber dem Fit: N_COMP 500 -> 125 (Eric 2026-10-18: "N_comp 125 ist gut").
# Gleiche Verteilung, ~25x schneller (kalibrierung/README_kalibrierung.md, Abschn. 5-7).
# DSMC-Skalierung wie in der KF3-Config:
#   SIM_FRACTION f = 1e-7 -> CONTROL_VOLUME = f * 1e-3 m^3 = 1e-10 m^3
#   N_real = 1.5937e10 Primaerpartikel im 1-L-Mischer -> f * N_real = 1593.7 in der Box
#   INITIAL_WEIGHT = 1593.7 / N_COMP = 1593.7 / 125 = 12.7496      (KF3: /500 = 3.1874)
#   VOLUMETRIC_FLOW_RATE = f * 1.7667e-6 m^3/s = 1.7667e-13 m^3/s (unveraendert)
#   Tropfen 150 um -> 0.106 * 1e-10 / (pi/6 * (150e-6)^3) = 6.0 Tropfen in der Box
BASE_PARAMS = {
    "N_MIXER": 30.0,
    "COST_MODE": "q3_sieve_mae",          # hier ohne Wirkung (nur fuer den Adapter)
    "MC_SEED": 7,                         # Basis der Seed-Liste (= Fit)
    "N_MC_REPLICATES": 1,                 # hier ohne Wirkung (Replikate macht der Runner)
    "INITIAL_PARTICLES": 125,             # N_COMP (KF3: 500)
    "INITIAL_WEIGHT": 12.7496,            # = 1593.7 / 125
    "CONTROL_VOLUME": 1e-10,
    "MERGER_TOLERANCE": 1e-06,
    "MERGER_LOOKUP": "scan",
    "MAXITER": 50_000_000,
    "MAX_PARTICLES": 100_000,
    "MAYBE_DOUBLE_VC": True,
    "AGG_DW_MIN": 1.0,
    "AGG_DW_MAX": 5,
    "BREAK_DW_MAX": 10,
    "SIZEEVAL": 0,
    "INITIAL_DIAMETER": 3.4e-05,
    "INITIAL_POROSITY": 0.0,
    "DROPLET_DIAMETER": 0.00015,
    "VOLUMETRIC_FLOW_RATE": 1.7667e-13,
    "LIQUID_ADDITION_START": 0.0,
    "LIQUID_ADDITION_DURATION": 60,
    "NUCLEATION_BATCH_SIZE": 20.0,
    # --- Bestparameter KF3 id 1070 ---
    "AGG_A": 1.340086659222402,
    "EKE_CORR_BETA": 1.310079085422099e-15,
    "EKE_C_MIXER": 0.0995,
    "STOKES_U_COLL_REF": 0.016017645998242614,
    "STOKES_C_VEL": 0.2852,
    "STOKES_BINDER_VISCOSITY": 0.1,
    "STOKES_RHO_SOLID": 600.0,
    "STOKES_RHO_LIQUID": 1000.0,
    "STOKES_H_A": 5e-07,
    "BREAK_P1": 4196244189526212.5,
    "BREAK_P2": 1.6645619890520489,
    "BREAK_C_MIXER": 0.6699,
    "BREAKRVAL": 4,
    "RUMPF_K": 2.5,
    "RUMPF_ALPHA": 1.0,
    "SURFACE_TENSION": 0.072,
    "CONTACT_ANGLE": 0.0,
    "BREAK_X_S": None,
    "BREAK_PORO_MAX": 0.9999,
    "COMPRESSION_RATE": 0.14080709816114967,
    "MIN_POROSITY": 0.2,
    "COMPRESSION_C_MIXER": 0.6699,
    "LIQUID_INTERNALIZATION_RATE": 14390824773.54635,
    "USE_LIQ_INTERNALISATION_AGGLOMERATION": True,
    "USE_UNIFORM_WEIGHTED": True,
    "CONE_K_AGG": 0.8894554328862593,
    "CONE_K_BREAK": 0.37469962481864244,
    "CONE_DEFAULT_POROSITY": 0.0,
}


# ===========================================================================
#  Ablauf (fuer alle Studien gleich)
# ===========================================================================
N_REPLICATES = 5               # Seeds je Punkt (Eric: 5; hoeher, falls Rauschen zu gross)
BASE_REPLICATES = 20           # Seeds am Basispunkt (Referenzkurve + Rauschschaetzung)
WORKERS = 12                    # Remote-Rechner: 6 Kerne / 12 Threads (mit --benchmark pruefen)
# Zeitlimit je Lauf = Faktor x Median-Laufzeit des Basispunkts auf dem rechnenden Rechner
# (der Basispunkt laeuft zuerst; Ergebnis in <Ergebnisordner>/time_limit.json). 0 = kein Limit.
# Stichprobe aus 128 echten Design-Punkten (kalibrierung, Abschn. 14): 2 von 128 brauchen
# ~22-24x Basis (starker Bruch + schwache Festigkeit), laufen aber durch. Mit 20x waeren
# rechnerisch ~1/3 der Bloecke ausgefallen. Daher 40 (18.10. spaetabends).
TIME_LIMIT_FACTOR = 40.0
SEC_PER_RUN_ESTIMATE = None    # nur fuer die Vorab-Schaetzung; --benchmark misst es

# Schnappschuesse alle 30 s; komplette Partikelliste bei 360 s (AJ40) und 600 s (AJ26/29/31)
T_SAVE = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0,
          360.0, 390.0, 420.0, 450.0, 480.0, 510.0, 540.0, 570.0, 600.0]
T_FULL = [360.0, 600.0]
BIN_DT = 5.0                   # Breite der Event-Zaehl-Fenster [s]
PROBE_DT = 2.0                 # Abstand der Raten-Messungen [s]


# ===========================================================================
#  Auswertung
# ===========================================================================
# Kennzahlen, fuer die Sobol-Indizes gerechnet werden (Namen siehe analyze_sensitivity.py).
#   log10_dXX_T   log10 des Quantil-Durchmessers [um] bei T s (stetig, rauscharm)
#   shift_T       Verschiebung der Q3-Kurve gegen den Basispunkt [Dekaden] (+ = groeber)
#   shift_abs_T   Flaeche zwischen den Kurven [Dekaden] (Betrag)
#   Q3_M_T        Q3 an der Masche M um bei T s (wie im Sieb; springt bei wenigen Granulaten)
#   J_KF3         KF3-Kosten des Punktmittels gegen die Versuche
#   break_per_primary / agg_per_primary   physikalische Bruch- / Agglomerationsereignisse
#                 je anfaenglichem Primaerpartikel (Wie aktiv ist der Mechanismus?)
OUTPUTS = [
    "log10_d10_600", "log10_d50_600", "log10_d90_600", "shift_600", "shift_abs_600",
    "Q3_100_600", "Q3_250_600", "Q3_500_600", "Q3_800_600",
    "log10_d10_360", "log10_d50_360", "log10_d90_360", "shift_360",
    "Q3_100_360", "Q3_250_360", "Q3_500_360",
    "J_KF3",
    "break_per_primary", "agg_per_primary", "n_comp_end", "ev_per_s",
]
PLOT_SOBOL_OUTPUTS = ["log10_d10_600", "log10_d50_600", "log10_d90_600", "shift_600",
                      "Q3_100_600", "Q3_250_600", "J_KF3", "break_per_primary", "agg_per_primary"]
PLOT_MAIN_EFFECT_OUTPUTS = ["log10_d50_600", "shift_600", "Q3_100_600", "break_per_primary"]

DISC_CRITERION_FACTOR = 2.0    # Events/s >= 2 x max(k_comp, lambda_int)  (Vorgabe Eric)
NOISE_SHARE_LIMIT = 0.10       # Rauschanteil > 10 % -> Indizes dieser Kennzahl vorsichtig lesen
BOOTSTRAP_RESAMPLES = 1000

# Versuchsdaten (studies/optimization/data/AJ_Q3_sieve.xlsx bzw. KF3_NoSampling_Q3_sieve_logw.xlsx),
# Q3 an den Maschen 100, 250, 500, 800, 1400, 2000 um
EXP_360 = [0.104914, 0.803040, 0.973487, 0.995511, 0.999042, 0.999581]            # AJ40, 360 s
EXP_600_REPEATS = [                                                                # 600 s, Wiederholungen
    [0.035684, 0.473779, 0.933153, 0.983846, 0.996805, 0.998975],                  # AJ26
    [0.038448, 0.442506, 0.926427, 0.983327, 0.997567, 0.999051],                  # AJ29
    [0.027029, 0.352731, 0.904597, 0.981857, 0.997902, 0.999568],                  # AJ31
]
EXP_600_MEAN = [0.033720, 0.423005, 0.921392, 0.983010, 0.997425, 0.999198]        # Mittel (KF3-Ziel)
MESH_WEIGHT = [0.282181, 0.227634, 0.164513, 0.145627, 0.129598, 0.050447]         # KF3 Maschengewicht
