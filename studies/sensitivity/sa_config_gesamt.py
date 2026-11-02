"""
SENSITIVITAETSSTUDIE "GESAMT" -- genau EIN Zweck: alle als relevant eingestuften Parameter
(physikalisch + die numerischen dW Agg und Batchsize) in EINER Sobol-Studie.

Auswahl und Begruendung je Parameter: PARAMETERLISTE.md (MA Tabelle 3.1 + Code + Einzeltests).
Basisfall, Versuchsdaten, Kennzahlen: sa_gemeinsam.py.

Einen Parameter herausnehmen = seine Zeile unten loeschen (er bleibt dann auf dem Basiswert).
Das geht nur VOR dem Start (anderes Design); der Runner bricht sonst ab, statt zu mischen.
"""

from sa_gemeinsam import *  # noqa: F401,F403  (Basisfall, Ablauf, Auswertung, Versuchsdaten)

STUDY_NAME = "gesamt_v1"
RESULT_DIR = "ergebnisse_gesamt_v1"         # relativ zu studies/sensitivity/

# Umfang (D = 18 Parameter). Punkte = N*(D+2) ohne S2 bzw. N*(2D+2) mit S2, je 5 Seeds (+20 Basis).
#   N = 256, ohne S2 -> 5120 Punkte -> 25 620 Laeufe   (Empfehlung)
#   N = 128, ohne S2 -> 2560 Punkte -> 12 820 Laeufe   (= erste Haelfte davon; Zwischenauswertung)
#   N = 128, mit S2  -> 4864 Punkte -> 24 340 Laeufe
# N darf spaeter erhoeht werden (Design praefix-stabil). CALC_SECOND_ORDER nicht nachtraeglich aendern.
N_SALTELLI = 256
CALC_SECOND_ORDER = False
SAMPLE_SEED = 20261018

# ===========================================================================
#  Variierte Parameter
# ===========================================================================
# Je Eintrag: name, kind, target (bei "direct"), bounds, log_scale, base (+ Zusatzangaben je kind)
#   bounds/base im ZIEH-RAUM: bei log_scale=True sind es log10-Werte. base = Wert am Basispunkt.
# Bereiche: gefittete Groessen ~ Band der besten 5 % des KF3-Fits; Annahmen ~ Literatur/Physik;
# Numerik ~ praktischer Bereich. Herleitung je Zeile in PARAMETERLISTE.md.
SA_PARAMETERS = [
    # --- Agglomeration -------------------------------------------------------------------
    # a_ref = EKE_CORR_BETA * r_ref^(-1.5 AGG_A), r_ref = 17 um (Primaerpartikel-Radius):
    # Agglomerations-Niveau der Primaerpartikel; Basis log10 = -5.295284830127166
    {"name": "AGG_AREF", "kind": "agg_rate_ref", "r_ref": 17e-6,
     "bounds": (-5.8, -4.8), "log_scale": True, "base": -5.295284830127166},
    {"name": "AGG_A", "kind": "direct", "target": "AGG_A",                      # c_temp
     "bounds": (0.74, 1.43), "log_scale": False, "base": 1.340086659222402},
    {"name": "STOKES_H_A", "kind": "direct", "target": "STOKES_H_A",            # h_a [m]: 100 nm .. 1 um
     "bounds": (-7.0, -6.0), "log_scale": True, "base": -6.301029995663981},
    # U_ref: einzeln ohne Wirkung (St/St* >= ~60), bei h_a = 1 um aber nur noch ~16x Abstand ->
    # zusammen mit grossem U_ref (bis 12.5x) kann das Stokes-Kriterium greifen. Steht auch fuer
    # die Binderviskositaet (St ~ U/mu, exakt gleiche Wirkung).
    {"name": "STOKES_U_COLL_REF", "kind": "direct", "target": "STOKES_U_COLL_REF",  # 0.002 .. 0.2 m/s
     "bounds": (-2.6989700043360187, -0.6989700043360187), "log_scale": True, "base": -1.7954013087943714},
    {"name": "CONE_K_AGG", "kind": "direct", "target": "CONE_K_AGG",
     "bounds": (0.4, 1.0), "log_scale": False, "base": 0.8894554328862593},
    # --- Bruch ---------------------------------------------------------------------------
    # S_ref = BREAK_P1 * V_ref^BREAK_P2, d_ref = 300 um; Basis log10 = -2.43703361357659
    {"name": "BREAK_SREF", "kind": "break_rate_ref", "d_ref": 300e-6,
     "bounds": (0.0, 2.0), "log_scale": True, "base": -2.43703361357659},
    {"name": "BREAK_P2", "kind": "direct", "target": "BREAK_P2",
     "bounds": (1.0, 2.0), "log_scale": False, "base": 1.6645619890520489},
    {"name": "RUMPF_ALPHA", "kind": "direct", "target": "RUMPF_ALPHA",          # k_wet
     "bounds": (1.0, 1.33), "log_scale": False, "base": 1.0},
    {"name": "RUMPF_K", "kind": "direct", "target": "RUMPF_K",                  # k (Formkorrektur)
     "bounds": (2.2, 2.8), "log_scale": False, "base": 2.5},
    {"name": "CONTACT_ANGLE", "kind": "direct", "target": "CONTACT_ANGLE",      # delta [rad]
     "bounds": (0.0, 1.4), "log_scale": False, "base": 0.0},
    {"name": "CONE_K_BREAK", "kind": "direct", "target": "CONE_K_BREAK",
     "bounds": (0.0, 1.0), "log_scale": False, "base": 0.37469962481864244},
    # --- Porositaet / Fluessigkeit (wirken ueber die Festigkeit auf den Bruch) -------------
    {"name": "MIN_POROSITY", "kind": "direct", "target": "MIN_POROSITY",        # eps_min
     "bounds": (0.15, 0.45), "log_scale": False, "base": 0.2},
    {"name": "COMPRESSION_RATE", "kind": "direct", "target": "COMPRESSION_RATE",  # k_comp [1/s]
     "bounds": (-2.5, -0.5), "log_scale": True, "base": -0.8513754516248897},
    {"name": "LIQUID_INTERNALIZATION_RATE", "kind": "direct", "target": "LIQUID_INTERNALIZATION_RATE",
     "bounds": (9.8, 11.7), "log_scale": True, "base": 10.158085685134333},     # k_int
    # Porositaet der Primaerpartikel beim Start (eps0), Vorgabe Eric 18.10. (spaetere Auswertung
    # in der MA). Feststoffmasse FEST: Durchmesser und Anzahl bleiben, Feststoff = (1 - eps0),
    # dafuer STOKES_RHO_SOLID = 600 / (1 - eps0) (0.2 rho_s bei 0 ... rho_s bei 0.8).
    {"name": "INITIAL_POROSITY", "kind": "porosity_mass_fixed",                  # eps0
     "bounds": (0.0, 0.8), "log_scale": False, "base": 0.0},
    # --- Nucleation ----------------------------------------------------------------------
    # Tropfendurchmesser [m] 50 .. 225 um (Untergrenze Eric 18.10.); ueber 150 um waechst die Box
    # mit (>= 6 Tropfen), darunter bleibt sie (bei 50 um ~160 Tropfen in der Box)
    {"name": "DROPLET_DIAMETER", "kind": "droplet_box", "d_box_ref": 150e-6,
     "bounds": (-4.301029995663981, -3.6478174818886), "log_scale": True, "base": -3.8239087409443187},
    # --- Numerik (Vorgabe Eric: dW Agg und Batchsize) ------------------------------------
    # dW Agg relativ zum Startgewicht: AGG_DW_MAX = Wert * INITIAL_WEIGHT; 0.078 .. 0.78
    # (= dW 1 .. 10 bei W_init 12.75); Basis 5 / 12.7496 = 0.392
    {"name": "AGG_DW_REL", "kind": "agg_dw_rel",
     "bounds": (-1.1054965596313227, -0.10549655963132273), "log_scale": True,
     "base": -0.40652655529530396},
    {"name": "NUCLEATION_BATCH_SIZE", "kind": "direct", "target": "NUCLEATION_BATCH_SIZE",
     "bounds": (0.0, 1.6989700043360187), "log_scale": True, "base": 1.3010299956639813},  # 1 .. 50

    # --- Nach Einzeltest OHNE Wirkung (bitgleich bzw. +-0.002), daher nicht drin --------
    #     Wieder aufnehmen: "#" am Zeilenanfang entfernen. Kostet je Parameter ~5 % Laufzeit.
    #     (STOKES_RHO_SOLID nicht zusammen mit INITIAL_POROSITY: eps0 setzt rho_eff fuer die feste Masse)
    # {"name": "STOKES_RHO_SOLID", "kind": "direct", "target": "STOKES_RHO_SOLID",    # 300 .. 1200 kg/m^3
    #  "bounds": (300.0, 1200.0), "log_scale": False, "base": 600.0},
    # {"name": "BREAK_DW_MAX", "kind": "direct", "target": "BREAK_DW_MAX",            # 1 .. 20
    #  "bounds": (0.0, 1.3010299956639813), "log_scale": True, "base": 1.0},
    # {"name": "CONE_DEFAULT_POROSITY", "kind": "direct", "target": "CONE_DEFAULT_POROSITY",  # 0 .. 0.5
    #  "bounds": (0.0, 0.5), "log_scale": False, "base": 0.0},
]

# Paare, die man nur ZUSAMMEN herausnehmen sollte:
#   AGG_A ohne AGG_AREF  -> c_temp verschiebt dann das Agglomerations-Niveau mit (Laufzeit!)
#   BREAK_P2 ohne BREAK_SREF -> P2 verschiebt dann das Bruch-Niveau um Dekaden
# Ohne DROPLET_DIAMETER bleibt die Box fest; AGG_DW_REL wirkt dann wie ein absolutes dW.
