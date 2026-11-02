"""
Schaetz-/Diagnoselauf fuer den AJ30-Nassgranulations-Fit.

Abgeleitet von ``test_powerlaw_rumpf_dynamic_full.py`` -- Kernel-Satz UND
Eingangsgroessen sind identisch zu dem, was der Optimierer fitten soll
(pbe-optimizer, Config ``studies/optimization/opt_config_wet_granulation_full.py``,
Adapter ``wet_granulation_adapter._build_solver``):

    Aggregation ............. kinetic_blend  (blend = AGG_A; AGG_A = 1 == eke_darelius2005)
    Agg.-Akzeptanz .......... stokes_dynamik
    Breakage ............... powerlaw_rumpf_dynamic
    Porositaets-Wachstum ... cone_model
    Porositaets-Kompression  porosity_compression_dynamik   (mixer-speed!)
    Fl.-Internalisierung ... liquid_internalization
    ... bei Agglomeration .. liq_internalisation_agglomeration
    Fl.-Verteilung ........ uniform_weighted

Der Solver laeuft mit ``verbose=True`` -- die ``[MC-PBE] Calculate t=... after N
events ...`` -Zeilen zeigen live den Fortschritt (Zeit, Events, n_comp, ms/Schritt),
genau wie beim Vorlagentest.

Zusaetzlich am Ende: die ``x(Q, t)`` SIM-vs-EXP-Tabelle, ``direct_delta`` --
die Kostenfunktion (MSE) zwischen simuliertem und experimentellem ``x(Q, t)``,
also genau die Groesse, die ``OptBase.calc_PSD_delta`` liefert -- und ein
Ergebnis-PNG (Q3(x) je Zeit, SIM vs EXP, Farbe = Zeit ; darunter die
Populations-Entwicklung; ``--no-plot`` / ``--plot-file`` zum Steuern).

direct_delta = mean( (x_sim - x_exp)^2 ) ueber alle (Quantil, Zeit)-Zellen, in m^2.
RMSE = sqrt(direct_delta) = mittlere Durchmesser-Abweichung je Punkt.
  1e-8 m^2 ~ 100 um (schlecht) ; 1e-10 ~ 10 um (gut) ; 10 = Sim gescheitert (Flag).

KOSTENFUNKTION per Schalter (TestConfig.COST_MODE bzw. --cost-mode, wie COST_MODE der
Fit-Config; Eric 2026-10-02):
  "q3_sieve_mae"  J = sum_i w_i * mean_k |Q3_sim(x_k, t_i) - Q3_exp,i(x_k)|
                  Q3 an den Siebmaschen 100..2000 um, alle Versuchsreihen gleich gewichtet
                  (studies/optimization/data/AJ_Q3_sieve.xlsx). J = mittlerer falsch einsortierter
                  Massenanteil je Masche (0 = perfekt). Plot: Q3(x)-Kurve der Simulation
                  (quasi kontinuierlich aus den Rechenpartikeln) + Siebpunkte an den Maschen.
  "xq_mse"        die bisherige x(Q, t)-MSE oben (nur AJ30).

WICHTIG
-------
Mit den echten AJ30-Groessen (W ~ 1.6e7, kleine dW_max, 100-um-Tropfen) ist
ein voller Lauf viele Millionen MC-Events. Fuer einen schnellen Blick:
    python studies/optimization/test_estimations_for_fit.py --maxiter 200000 --t-end 60

Aufruf:
    python studies/optimization/test_estimations_for_fit.py --maxiter 5000000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from wmcpbe import MCPBESolver


# =============================================================================
#  Experimentelle Daten  x(Q, t)
# =============================================================================
# studies/optimization  ->  PSD_opt  ist parents[2]
_PSD_OPT_ROOT = Path(__file__).resolve().parents[2]
EXP_DATA_FILE = _PSD_OPT_ROOT / "studies" / "optimization" / "data" / "AJ30_xQ.xlsx"          # COST_MODE "xq_mse"
SIEVE_DATA_FILE = _PSD_OPT_ROOT / "studies" / "optimization" / "data" / "AJ_Q3_sieve.xlsx"    # COST_MODE "q3_sieve_mae"
# Raster fuer die (quasi) kontinuierliche Q3(x)-Kurve der Simulation im Plot [um]
X_PLOT_UM = np.geomspace(20.0, 3000.0, 4000)

# AJ30-Zeitachse (Sim-Zeit = Prozesszeit - 90 s). t_vec[0]=0 = Startzustand.
AJ30_TIMES_S = (0.0, 60.0, 180.0, 240.0, 300.0, 360.0, 420.0, 480.0, 540.0, 600.0)


# =============================================================================
#  TestConfig  --  1:1 die AJ30-Fit-Groessen
#  (Quelle: studies/optimization/opt_config_wet_granulation_full.py  ->  pop_params)
# =============================================================================

class TestConfig:
    """Physikalische + numerische Parameter, identisch zum Fit-Setup."""

    SEED = 7                              # MC_SEED

    # -- Kostenfunktion (wie COST_MODE in der Fit-Config) -------------------
    #   "q3_sieve_mae": Q3 an den Siebmaschen, Betrag, alle Versuchsreihen gleich
    #                   gewichtet (SIEVE_DATA_FILE) ; Plot: Q3(x)-Kurve + Siebpunkte
    #   "xq_mse":       x(Q, t) an 21 Quantilen, MSE, nur AJ30 (EXP_DATA_FILE)
    COST_MODE = "q3_sieve_mae"

    # -- Zeit --------------------------------------------------------------
    T_END = 600.0                         # letzter Snapshot [s]

    # -- Ausgangspulver / Anfangszustand --------------------------------
    PARTICLE_DIAMETER = 34e-6             # INITIAL_DIAMETER, monodispers
    INITIAL_POROSITY = 0.0               # Vollkoerper-Start
    PARTICLE_DENSITY = 600.0             # STOKES_RHO_SOLID (nur stokes + Massen-Diagnose)

    # -- Fluessigkeitszugabe / Nucleation ------------------------------
    DROPLET_DIAMETER = 50e-6            # DROPLET_DIAMETER
    DROPLET_DENSITY = 1000.0             # STOKES_RHO_LIQUID
    LIQUID_ADDITION_START = 0.0
    NUCLEATION_DURATION = 60.0           # LIQUID_ADDITION_DURATION [s]  (Vc-invariant)
    BATCH_SIZE = 20.0                   # NUCLEATION_BATCH_SIZE (klein = genau)

    # -- kinetic_blend (Aggregation) ------------------------------------
    AGG_A = 1.0                          # AGG_A = blend: v ~ m^(-AGG_A/2); 0 = gleich schnell, 1 = EKE, 2 = ETM
    AGG_COEFFICIENT = 5.75e-12          # EKE_CORR_BETA [PHYS-EST] (corr_beta von kinetic_blend)
    C_MIXER_AGG = 0.0995                 # EKE_C_MIXER (= DEM C_FREQ)

    # -- stokes_dynamik ---------------------------------------------
    COLLISION_VELOCITY = 0.0794          # STOKES_U_COLL_REF [m/s]
    C_VEL = 0.2852                       # STOKES_C_VEL
    BINDER_VISCOSITY = 0.1               # STOKES_BINDER_VISCOSITY [Pa.s]
    H_A = 500e-9                         # STOKES_H_A [m]

    # -- powerlaw_rumpf_dynamic ------------------------------------
    PL_P1 = 1e11                         # BREAK_P1
    PL_P2 = 1.0                          # BREAK_P2
    C_MIXER_BREAK = 0.6699               # BREAK_C_MIXER (= DEM C_BREAK)
    BREAKRVAL = 4
    RUMPF_K = 2.5
    RUMPF_ALPHA = 1.0
    RUMPF_GAMMA = 0.072                  # SURFACE_TENSION [N/m]
    RUMPF_DELTA = 0.0                    # CONTACT_ANGLE [rad]
    BREAK_X_S = None                     # -> aus solver.X0
    BREAK_PORO_MAX = 0.9999

    # -- porosity_compression_dynamik -----------------------------
    COMPRESSION_RATE = 1.3441e-11         # COMPRESSION_RATE (k=0.1 @ n_mixer=20)
    MIN_POROSITY = 0.2
    C_MIXER_COMPRESSION = 0.6699

    # -- liquid_internalization -----------------------------------
    LIQ_INTERN_RATE = 1e10              # LIQUID_INTERNALIZATION_RATE

    # -- cone_model ---------------------------------------------
    CONE_K_AGG = 1.0
    CONE_K_BREAK = 1.0
    CONE_DEFAULT_POROSITY = 0.0

    # -- mixer speed (gemeinsam) --------------------------------
    N_MIXER = 30.0                       # N_MIXER [EXP]

    # -- DSMC-/Prozess-Skalierung  (identisch zur Config) ------
    #   EXPERIMENT (Vc-invariant):
    V_MIXER_EXP  = 1.0e-3                # 1 L
    N_REAL       = 1.5937e10             # reale Partikel im vollen Mischer
    Q_BINDER_EXP = 1.7667e-6             # Pumpen-Flussrate [m^3/s], voller Mischer
    #   AUSSCHNITT:  f skaliert Vc/W/flow ; N_comp bleibt FEST (statist. Aufloesung)
    SIM_FRACTION = 1e-7                   # f = Vc / V_mixer_exp
    INITIAL_PARTICLES = 700             # FEST
    #   abgeleitet (Reihenfolge zaehlt im Klassenkoerper):
    CONTROL_VOLUME      = SIM_FRACTION * V_MIXER_EXP
    INITIAL_WEIGHT      = SIM_FRACTION * N_REAL / INITIAL_PARTICLES   # skaliert mit f
    VOLUMETRIC_FLOW_RATE = SIM_FRACTION * Q_BINDER_EXP

    # -- Numerik ---------------------------------------------
    MERGER_TOLERANCE = 1e-6
    MAYBE_DOUBLE_VC = False
    AGG_DW_MIN = 1.0
    AGG_DW_MAX = 5.0          # < W_init (~16 bei f=1e-6) -> weighted-DSMC nicht degeneriert
    BREAK_DW_MAX = 10.0
    SIZEEVAL = 0
    MAXITER = 50_000_000
    MAX_PARTICLES = 100_000
    PREALLOC_PARTICLES = 20_000          # Kapazitaet sofort hierauf, statt schrittweise zu wachsen


# =============================================================================
#  Helfer
# =============================================================================

def fmt(value: float, unit: str = "") -> str:
    return f"{value:.3e} {unit}".strip()


def pct(value: float) -> str:
    return f"{value*100:+.4f}%"


def sec(title: str, char: str = "=") -> None:
    print(f"\n{char * 80}\n{title}\n{char * 80}")


def x_at_quantiles(v_snapshot: np.ndarray, weights: np.ndarray,
                   q_grid: np.ndarray) -> np.ndarray:
    """x an den Q3-Quantilen -- identisch zu
    ``wet_granulation_adapter._x_at_quantiles_from_snapshot``."""
    v_snapshot = np.asarray(v_snapshot, dtype=float)
    v_dry = np.asarray(v_snapshot[-1], dtype=float)
    v_solid = np.asarray(v_snapshot[0], dtype=float)
    diameters = np.cbrt(6.0 * v_dry / np.pi)
    weights = np.asarray(weights, dtype=float)
    amount = weights * v_solid                     # Masse-Proxy
    valid = (np.isfinite(diameters) & np.isfinite(amount)
             & (diameters > 0.0) & (amount > 0.0))
    if not np.any(valid):
        raise ValueError("Snapshot ohne positive, endliche gewichtete Partikel.")
    diameters = diameters[valid]
    amount = amount[valid]
    order = np.argsort(diameters)
    diameters = diameters[order]
    q3_cum = np.cumsum(amount[order])
    q3_cum /= q3_cum[-1]
    return np.interp(q_grid, q3_cum, diameters)


def read_exp_xq(path: Path, times_s) -> tuple[np.ndarray, np.ndarray]:
    """AJ30_xQ.xlsx -> (q_grid, data_exp[n_Q, n_t] in m)."""
    frame = pd.read_excel(path, sheet_name="x_Q3")
    q_grid = frame.iloc[:, 0].to_numpy(dtype=float)
    x_all_mm = frame.iloc[:, 1:].to_numpy(dtype=float)
    time_all = np.asarray([float(c) for c in frame.columns[1:]], dtype=float)
    cols = []
    for t in times_s:
        m = np.flatnonzero(np.isclose(time_all, t, atol=1e-6))
        if m.size != 1:
            raise ValueError(f"Keine eindeutige x_Q3-Spalte fuer t={t} s (vorhanden: {time_all.tolist()}).")
        cols.append(int(m[0]))
    return q_grid, x_all_mm[:, cols] * 1e-3


def _particles(v_snapshot: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Durchmesser [m] (aus V_dry) sortiert + kumulierte Massenanteile (W * V_solid)."""
    v_snapshot = np.asarray(v_snapshot, dtype=float)
    diameters = np.cbrt(6.0 * np.asarray(v_snapshot[-1], dtype=float) / np.pi)
    amount = np.asarray(weights, dtype=float) * np.asarray(v_snapshot[0], dtype=float)
    valid = (np.isfinite(diameters) & np.isfinite(amount)
             & (diameters > 0.0) & (amount > 0.0))
    if not np.any(valid):
        raise ValueError("Snapshot ohne positive, endliche gewichtete Partikel.")
    diameters, amount = diameters[valid], amount[valid]
    order = np.argsort(diameters)
    q3_cum = np.cumsum(amount[order])
    return diameters[order], q3_cum / q3_cum[-1]


def q3_below(v_snapshot: np.ndarray, weights: np.ndarray, x_m: np.ndarray) -> np.ndarray:
    """Q3(x) = Massenanteil mit Durchmesser < x, exakt aus den Rechenpartikeln -- identisch zu
    ``wet_granulation_adapter._q3_at_meshes_from_snapshot`` (dort mit x = Siebmaschen)."""
    d, q3_cum = _particles(v_snapshot, weights)
    n_below = np.searchsorted(d, np.asarray(x_m, dtype=float), side="left")
    return np.where(n_below > 0, q3_cum[np.maximum(n_below - 1, 0)], 0.0)


def read_exp_q3_sieve(path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """AJ_Q3_sieve.xlsx -> (Siebmaschen [um], Tabelle eine Zeile je Messung).

    Spalten: Reihe, Gruppe, Probe, t_sim_s, Gewicht, Sieb_g, je Masche [um] eine Q3-Spalte
    (Skript: studies/optimization/dataprep/prepare_aj_q3_sieve.py)."""
    frame = pd.read_excel(path, sheet_name="Q3_sieve")
    mesh_cols = []
    for c in frame.columns:
        try:
            float(c)
            mesh_cols.append(c)
        except (TypeError, ValueError):
            pass
    mesh_um = np.asarray([float(c) for c in mesh_cols], dtype=float)
    obs = frame.rename(columns={c: float(c) for c in mesh_cols})
    if abs(obs["Gewicht"].sum() - 1.0) > 1e-9:
        raise ValueError(f"Gewichte in {path.name} summieren nicht zu 1.")
    return mesh_um, obs


# =============================================================================
#  Solver bauen  --  identisch zu wet_granulation_adapter._build_solver
# =============================================================================

def build_solver(cfg: TestConfig, t_vec: np.ndarray, initial_diameters: np.ndarray,
                 seed: int | None = None, verbose: bool = True) -> MCPBESolver:
    n_mixer = cfg.N_MIXER
    k_int = cfg.LIQ_INTERN_RATE

    solver = MCPBESolver(
        dim=1,
        t_vec=t_vec,
        verbose=verbose,   # [MC-PBE] Calculate t=... -Fortschrittszeilen in der Console
        load_attr=False,
        init=False,
        seed=int(cfg.SEED if seed is None else seed),
        maybe_double_control_volume=bool(cfg.MAYBE_DOUBLE_VC),
        recon_enable=False,
        agg_kernel_name="kinetic_blend",
        agg_kernel_params={
            "corr_beta": float(cfg.AGG_COEFFICIENT),
            "blend": float(cfg.AGG_A),
            "n_mixer": n_mixer,
            "c_mixer": float(cfg.C_MIXER_AGG),
        },
        agg_acceptance_kernel_name="stokes_dynamik",
        agg_acceptance_kernel_params={
            "U_coll_ref": float(cfg.COLLISION_VELOCITY),
            "n_mixer": n_mixer,
            "c_vel": float(cfg.C_VEL),
            "binder_viscosity": float(cfg.BINDER_VISCOSITY),
            "rho_solid": float(cfg.PARTICLE_DENSITY),
            "rho_liquid": float(cfg.DROPLET_DENSITY),
            "h_a": float(cfg.H_A),
        },
        break_kernel_name="powerlaw_rumpf_dynamic",
        break_kernel_params={
            "p1": float(cfg.PL_P1),
            "p2": float(cfg.PL_P2),
            "n_mixer": n_mixer,
            "c_mixer": float(cfg.C_MIXER_BREAK),
            "breakrval": int(cfg.BREAKRVAL),
            "k": float(cfg.RUMPF_K),
            "alpha": float(cfg.RUMPF_ALPHA),
            "gamma": float(cfg.RUMPF_GAMMA),
            "delta": float(cfg.RUMPF_DELTA),
            "x_s": cfg.BREAK_X_S,
            "poro_max": float(cfg.BREAK_PORO_MAX),
        },
        porosity_growth_kernel_name="cone_model",
        porosity_growth_kernel_params={
            "k_agg": float(cfg.CONE_K_AGG),
            "k_break": float(cfg.CONE_K_BREAK),
            "default_porosity": float(cfg.CONE_DEFAULT_POROSITY),
        },
        porosity_compression_kernel_name="porosity_compression_dynamik",
        porosity_compression_kernel_params={
            "rate": float(cfg.COMPRESSION_RATE),
            "min_porosity": float(cfg.MIN_POROSITY),
            "n_mixer": n_mixer,
            "c_mixer": float(cfg.C_MIXER_COMPRESSION),
        },
        liquid_internalization_kernel_name="liquid_internalization",
        liquid_internalization_kernel_params={"k_int": k_int},
        liq_internalisation_agglomeration_kernel_name="liq_internalisation_agglomeration",
        liquid_dist_kernel_name="uniform_weighted",
    )

    solver.process_type = "mix"
    solver.agg_propensity_mode = "moment"   # kinetic_blend hat keine Momentenform -> pairwise
    solver.SIZEEVAL = int(cfg.SIZEEVAL)
    solver.agg_dW_min = float(cfg.AGG_DW_MIN)
    solver.agg_dW_max = float(cfg.AGG_DW_MAX)
    solver.break_dW_max = float(cfg.BREAK_DW_MAX)
    solver._enable_particle_merging = True
    solver.merger_lookup = "scan"          # KEIN Hash-Index (jeden Event neu-keyen kostet mehr); 431 s -> 55 s auf rumpf_full
    solver._merger_use_hash_index = False
    solver._fragment_merge_tol = float(cfg.MERGER_TOLERANCE)

    porosity = float(cfg.INITIAL_POROSITY)
    diameters = np.asarray(initial_diameters, dtype=float)
    v_dry = np.pi / 6.0 * diameters ** 3
    v_flat = np.vstack((v_dry * (1.0 - porosity), v_dry))
    weights = np.full(diameters.size, float(cfg.INITIAL_WEIGHT))
    solver.Vc = float(cfg.CONTROL_VOLUME)
    solver._initialize_particles(init_Vc=False, V_flat=v_flat, W_init=weights)
    # Kapazitaet SOFORT auf PREALLOC_PARTICLES hochziehen -- spart das
    # schrittweise Array-Verdoppeln + Kopieren (die langsamen "Capacity grown"-
    # Zeilen). Nutzt die vorhandene Solver-API, kein Core-Eingriff.
    # verbose kurz aus: _ensure_capacity_for liest sonst _last_verbose_time /
    # _elapsed, die erst in solve() gesetzt werden.
    _prealloc = int(getattr(cfg, "PREALLOC_PARTICLES", 0))
    if _prealloc > solver.a_tot:
        _v, solver.VERBOSE = solver.VERBOSE, False
        solver._ensure_capacity_for(_prealloc - solver.a_tot)
        solver.VERBOSE = _v
    solver.porosity[:solver.a_tot] = porosity
    solver.liquid_volume[:solver.a_tot] = 0.0
    solver.saturation[:solver.a_tot] = 0.0
    solver._init_lmc()
    solver._initialize_samplers()

    solver.create_nucleation_handler(
        enabled=True,
        volumetric_flow_rate=float(cfg.VOLUMETRIC_FLOW_RATE),
        droplet_diameter=float(cfg.DROPLET_DIAMETER),
        liquid_addition_start=float(cfg.LIQUID_ADDITION_START),
        liquid_addition_duration=float(cfg.NUCLEATION_DURATION),
        batch_size=float(cfg.BATCH_SIZE),
    )
    solver.create_continuous_processes_handler(
        enabled=True,
        k_int=k_int,
        compression_enabled=True,
        compression_rate=float(cfg.COMPRESSION_RATE),
        min_porosity=float(cfg.MIN_POROSITY),
    )
    return solver


# =============================================================================
#  Hauptlauf
# =============================================================================

def _plot_results(cfg, q_grid, fit_times, data_mod, data_exp_fit,
                  t_vec, solver, n_save, direct_delta, rmse_um, plot_file,
                  data_std=None, n_rep=1, sieve=None):
    """Q3(x) SIM vs EXP je Zeit (oben, Durchmesser auf x / Q3 auf y) +
    Populations-Entwicklung (unten).

    COST_MODE "xq_mse" (sieve=None): dieselben x(Q)-Stuetzstellen wie in der
    Kostenfunktion, nur mit getauschten Achsen dargestellt.
    COST_MODE "q3_sieve_mae" (sieve=dict): SIM als (quasi) kontinuierliche Q3(x)-Kurve
    aus den Rechenpartikeln, EXP als Punkte an den Siebmaschen (Dreieck = AJ30-Probe,
    Kreis = Versuch ohne Probenahme). Der senkrechte Abstand an einer Masche ist genau
    der Fehler, den die Kostenfunktion zaehlt.
    Zeiten als diskrete Legende (Farben wie bisher: viridis, normiert auf den Bereich
    der Fit-Zeitpunkte).

    Speichert IMMER ein PNG und zeigt es an:
      - haengt aus einer wiederverwendeten Konsole noch Agg -> auf QtAgg zwingen;
      - interaktives Backend -> plt.show() (Spyder-Plotfenster / Terminal-Fenster);
      - bleibt es nicht-interaktiv (headless) -> PNG im Standard-Bildbetrachter
        oeffnen. So kommt das Bild in jedem Fall an.
    """
    try:
        import matplotlib
        # In einer WIEDERVERWENDETEN Spyder-/IPython-Konsole kann noch ein
        # nicht-interaktives Backend (Agg) aus einem frueheren Lauf haengen
        # -> plt.show() zeigt dann nichts ("FigureCanvasAgg is non-interactive").
        # Dann auf ein GUI-Backend zwingen (Qt ist im env vorhanden).
        _NONINT = {"agg", "pdf", "ps", "svg", "cairo", "pgf", "template"}
        if matplotlib.get_backend().lower() in _NONINT:
            for _gui in ("QtAgg", "Qt5Agg", "TkAgg"):
                try:
                    matplotlib.use(_gui, force=True)
                    break
                except Exception:
                    continue
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.lines import Line2D
    except Exception as exc:  # pragma: no cover
        print(f"  (Plot uebersprungen: {exc})")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8.5),
                                   gridspec_kw={"height_ratios": [2.2, 1]})
    cmap = plt.get_cmap("viridis")
    ax1.set_xscale("log")
    ax1.set_xlabel("Durchmesser x [um]")
    ax1.set_ylabel("Q3(x) [-]")
    ax1.set_ylim(-0.02, 1.02)
    _rep = f"   |   Mittelwert aus {n_rep} MC-Laeufen" if n_rep and n_rep > 1 else ""
    if sieve is None:
        ax1.set_title(f"AJ30  Q3(x) je Zeit:  Simulation vs. Experiment{_rep}\n"
                      f"SIM_FRACTION f = {cfg.SIM_FRACTION:g}   |   "
                      f"direct_delta = {direct_delta:.3e} m^2   RMSE = {rmse_um:.0f} um")
    else:
        ax1.set_title(f"Q3(x) je Zeit:  Simulation vs. Siebdaten (alle Reihen){_rep}\n"
                      f"SIM_FRACTION f = {cfg.SIM_FRACTION:g}   |   "
                      f"J = {direct_delta:.4f}  (Q3 an den Maschen, Betrag; 0 = perfekt)")
    ax1.grid(alpha=0.3, which="both")
    # Q3(x): pro Zeit eine Kurve (Durchmesser auf x, Q3 auf y), Farbe = Zeit.
    # t=0 (monodispers) uebersprungen -- keine Verteilung.
    tmask = fit_times > 0.0
    plot_times = fit_times[tmask]
    if plot_times.size and sieve is not None:
        lo, hi = float(plot_times.min()), float(plot_times.max())
        norm = Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1.0)
        obs, mesh_um = sieve["obs"], sieve["mesh_um"]
        time_handles = []
        for j, t in zip(np.flatnonzero(tmask), plot_times):
            c = cmap(norm(t))
            if sieve.get("curves_std") is not None:
                # 1-sigma-Schlauch: gefuellte Flaeche + duenne Randlinien bei Mittel +/- sigma
                lo_s = np.clip(sieve["curves"][j] - sieve["curves_std"][j], 0.0, 1.0)
                hi_s = np.clip(sieve["curves"][j] + sieve["curves_std"][j], 0.0, 1.0)
                ax1.fill_between(X_PLOT_UM, lo_s, hi_s, color=c, alpha=0.3, lw=0)
                ax1.plot(X_PLOT_UM, lo_s, "-", color=c, lw=0.7, alpha=0.8)
                ax1.plot(X_PLOT_UM, hi_s, "-", color=c, lw=0.7, alpha=0.8)
            ax1.plot(X_PLOT_UM, sieve["curves"][j], "-", color=c, lw=1.6)
            for _, row in obs[np.isclose(obs["t_sim_s"], t)].iterrows():
                aj30 = str(row["Reihe"]) == "AJ30"
                ax1.plot(mesh_um, row[list(mesh_um)].to_numpy(float), "^" if aj30 else "o",
                         color=c, ms=6 if aj30 else 7, mec="white" if aj30 else "black", mew=0.8,
                         ls="none", zorder=5)
            time_handles.append(Line2D([], [], color=c, lw=3, label=f"{t:g} s"))
        ax1.set_xlim(X_PLOT_UM[0], X_PLOT_UM[-1])
        ax1.plot([], [], "-", color="0.3", label="SIM (Mittelwert)" if n_rep > 1 else "SIM")
        if sieve.get("curves_std") is not None:
            ax1.fill_between([], [], [], color="0.3", alpha=0.3, label="SIM +/- 1 sigma (MC)")
        ax1.plot([], [], "^", color="0.5", mec="white", ls="none", label="EXP AJ30-Probe (Sieb)")
        ax1.plot([], [], "o", color="0.5", mec="black", ls="none", label="EXP ohne Probenahme (Sieb)")
        style_legend = ax1.legend(loc="lower right", fontsize=9)
        ax1.add_artist(style_legend)
        ax1.legend(handles=time_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                   fontsize=9, title="Sim-Zeit\n(= Prozesszeit - 90 s)", title_fontsize=8.5,
                   borderaxespad=0.0)
    elif plot_times.size:
        lo, hi = float(plot_times.min()), float(plot_times.max())
        norm = Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1.0)
        time_handles = []
        for j, t in zip(np.flatnonzero(tmask), plot_times):
            c = cmap(norm(t))
            if data_std is not None:
                # 1-sigma-Schlauch: gefuellte Flaeche + duenne Randlinien bei Mittel +/- sigma
                lo_s = (data_mod[:, j] - data_std[:, j]) * 1e6
                hi_s = (data_mod[:, j] + data_std[:, j]) * 1e6
                ax1.fill_betweenx(q_grid, lo_s, hi_s, color=c, alpha=0.3, lw=0)
                ax1.plot(lo_s, q_grid, "-", color=c, lw=0.7, alpha=0.8)
                ax1.plot(hi_s, q_grid, "-", color=c, lw=0.7, alpha=0.8)
            ax1.plot(data_mod[:, j] * 1e6, q_grid, "-o", color=c, ms=4, lw=1.6)
            ax1.plot(data_exp_fit[:, j] * 1e6, q_grid, "--x", color=c, ms=6, lw=1.2, alpha=0.9)
            time_handles.append(Line2D([], [], color=c, lw=3, label=f"{t:g} s"))
        ax1.plot([], [], "-o", color="0.3", label="SIM (Mittelwert)" if n_rep > 1 else "SIM")
        ax1.plot([], [], "--x", color="0.3", label="EXP (Sieb)")
        if data_std is not None:
            ax1.fill_between([], [], [], color="0.3", alpha=0.3, label="SIM +/- 1 sigma (MC)")
        style_legend = ax1.legend(loc="upper left", fontsize=9)
        ax1.add_artist(style_legend)
        # diskrete Zeit-Legende rechts neben dem Plot (dort sass vorher die Farbskala)
        ax1.legend(handles=time_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                   fontsize=9, title="Sim-Zeit\n(= Prozesszeit - 90 s)", title_fontsize=8.5,
                   borderaxespad=0.0)
    else:
        ax1.text(0.5, 0.5, "keine Fit-Zeitpunkte > 0 erreicht", transform=ax1.transAxes,
                 ha="center", va="center")

    m = int(n_save)
    tt = t_vec[:m]
    ncomp = np.array([len(solver.W_save[k]) for k in range(m)])
    agg = np.asarray(getattr(solver, "real_agg_events_save", [0.0]))[:m]
    brk = np.asarray(getattr(solver, "real_break_events_save", [0.0]))[:m]
    ax2.plot(tt, ncomp, "-", color="tab:blue", label="n_comp (Rechenpartikel)")
    ax2.set_xlabel("Sim-Zeit [s]")
    ax2.set_ylabel("n_comp", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.grid(alpha=0.3)
    ax2b = ax2.twinx()
    ax2b.plot(tt, agg, "-", color="tab:green", label="agg-Events (kum.)")
    ax2b.plot(tt, brk, "-", color="tab:red", label="break-Events (kum.)")
    ax2b.set_ylabel("Events (kumuliert)")
    ax2.axvspan(0, cfg.NUCLEATION_DURATION, color="0.85", alpha=0.5, zorder=0)
    lines = ax2.get_lines() + ax2b.get_lines()
    ax2.legend(lines, [ln.get_label() for ln in lines], loc="upper left", fontsize=8)
    ax2.set_title("Populations-Entwicklung", fontsize=10)

    fig.tight_layout()
    path = Path(plot_file) if plot_file else (
        EXP_DATA_FILE.parent / f"AJ30_estimation_f{cfg.SIM_FRACTION:g}.png")
    fig.savefig(path, dpi=130)
    print(f"\n  PLOT gespeichert: {path}")

    # Anzeigen: GUI-Fenster, wenn ein interaktives Backend aktiv ist
    # (Spyder-Plot-Fenster ; Terminal: blockiert bis geschlossen). Sonst
    # (Agg haengt aus wiederverwendeter Konsole / headless) das PNG direkt
    # im Standard-Bildbetrachter oeffnen -- so kommt es garantiert an.
    shown = False
    if matplotlib.get_backend().lower() not in _NONINT:
        try:
            plt.show()
            shown = True
        except Exception:
            shown = False
    if not shown:
        print(f"  (Backend '{matplotlib.get_backend()}' nicht-interaktiv -- oeffne das PNG)")
        try:
            os.startfile(str(path))                   # nur Windows
        except AttributeError:                         # macOS / Linux
            import subprocess as _sp
            _sp.run(["open" if sys.platform == "darwin" else "xdg-open", str(path)],
                    check=False)
        except Exception:
            pass
    return path


def _finish_q3_sieve(cfg, solver, fit_reached, times_s, t_vec, n_save, mesh_um, obs,
                     replicates, rep_seeds, initial_diameters, plot, plot_file,
                     info: Dict[str, Any]) -> Dict[str, Any]:
    """Auswertung fuer COST_MODE "q3_sieve_mae" -- dieselbe Rechnung wie im Adapter:

        J = sum_i w_i * mean_k |Q3_sim(x_k, t_i) - Q3_exp,i(x_k)|

    Q3_sim = Mittelwert der Replikate an den Siebmaschen. Erreicht der Lauf nicht alle
    Messzeiten (--t-end < 600), zaehlen nur die erreichten Messungen (Gewichte renormiert).
    """
    fit_times = np.array([times_s[i] for i, _ in fit_reached])
    ks = [k for _, k in fit_reached]
    mesh_m = mesh_um * 1e-6
    x_plot_m = X_PLOT_UM * 1e-6

    use = obs[[bool(np.any(np.isclose(fit_times, t))) for t in obs["t_sim_s"]]].reset_index(drop=True)
    if use.empty:
        print("\n!! Keine Messzeit erreicht -- J nicht berechenbar.")
        return {"direct_delta": None, "cost_mode": "q3_sieve_mae", **info}
    row_of = np.array([int(np.flatnonzero(np.isclose(fit_times, t))[0]) for t in use["t_sim_s"]])
    w_reached = float(use["Gewicht"].sum())
    w = use["Gewicht"].to_numpy(float) / w_reached
    exp = use[list(mesh_um)].to_numpy(float)

    def evaluate(s):
        q3 = np.vstack([q3_below(s.V_save[k], s.W_save[k], mesh_m) for k in ks])        # (n_fit, n_Masche)
        curves = np.vstack([q3_below(s.V_save[k], s.W_save[k], x_plot_m) for k in ks])  # (n_fit, n_x)
        return q3, curves

    def cost(q3):
        return float(np.dot(w, np.abs(q3[row_of] - exp).mean(axis=1)))

    q3, curves = evaluate(solver)
    q3s, curve_list, rep_J = [q3], [curves], [cost(q3)]
    if replicates > 1:
        kmax = max(ks)
        sec(f"WEITERE {replicates - 1} MC-REPLIKATE (Seeds {rep_seeds[1:]})", "-")
        for r, sd in enumerate(rep_seeds[1:], start=2):
            s_r = build_solver(cfg, t_vec, initial_diameters, seed=sd, verbose=False)
            try:
                s_r.solve(maxiter=int(cfg.MAXITER), max_particles=int(cfg.MAX_PARTICLES))
            except Exception as exc:
                print(f"  Replikat {r} (seed {sd}) gescheitert: {exc}")
                continue
            if len(s_r.V_save) <= kmax:
                print(f"  Replikat {r} (seed {sd}): zu wenige Snapshots -- uebersprungen")
                continue
            q3_r, c_r = evaluate(s_r)
            q3s.append(q3_r)
            curve_list.append(c_r)
            rep_J.append(cost(q3_r))
            print(f"  Replikat {r} (seed {sd}): J = {rep_J[-1]:.4f}")
    n_rep_ok = len(q3s)
    q3 = np.mean(q3s, axis=0)
    curves_mean = np.mean(curve_list, axis=0)
    curves_std = np.std(curve_list, axis=0, ddof=1) if n_rep_ok > 1 else None
    J = cost(q3)

    sec("Q3 AN DEN SIEBMASCHEN:  SIM / EXP", "=")
    print("  Reihe Probe   t [s]  Gewicht | " + " | ".join(f"<{m:g} um".center(13) for m in mesh_um))
    for i, row in use.iterrows():
        cells = " | ".join(f"{q3[row_of[i], k]:.3f} / {exp[i, k]:.3f}" for k in range(mesh_um.size))
        print(f"  {str(row['Reihe']):5s} {str(row['Probe']):5s} {row['t_sim_s']:6.0f}  {w[i]:.4f} | {cells}")

    sec("KOSTEN J  (Q3 an den Maschen, Betrag)", "=")
    if n_rep_ok > 1:
        print(f"  erfolgreiche Replikate        : {n_rep_ok}/{replicates}")
        print(f"  J je Replikat                 : " + "  ".join(f"{d:.4f}" for d in rep_J))
    if w_reached < 1.0 - 1e-9:
        print(f"  !! nur Messungen bis t = {fit_times.max():g} s erreicht ({w_reached:.1%} des Gewichts) "
              f"-- Gewichte darauf renormiert.")
    print(f"  J (Mittelwert der Replikate)  : {J:.6f}   (0 = perfekt ; max 1 ; 10 = Sim gescheitert)")
    err = w * np.abs(q3[row_of] - exp).mean(axis=1)
    by_group = pd.Series(err, index=use["Gruppe"]).groupby(level=0).sum()
    print("  Beitrag je Versuchsreihe      : " + "   ".join(f"{g}: {v:.4f}" for g, v in by_group.items()))
    by_mesh = (w[:, None] * np.abs(q3[row_of] - exp)).sum(axis=0) / mesh_um.size
    print("  Beitrag je Masche             : " + "   ".join(f"<{m:g}: {v:.4f}" for m, v in zip(mesh_um, by_mesh)))

    if plot:
        _plot_results(cfg, None, fit_times, None, None, t_vec, solver, n_save, J, None, plot_file,
                      n_rep=n_rep_ok, sieve=dict(obs=use, mesh_um=mesh_um, curves=curves_mean,
                                                 curves_std=curves_std))
    return {"direct_delta": J, "cost_mode": "q3_sieve_mae", **info,
            "replicates_ok": n_rep_ok, "replicate_deltas": rep_J}


def run_estimation(maxiter: int | None = None, t_end: float | None = None,
                   fraction: float | None = None, progress_dt: float = 10.0,
                   plot: bool = True, plot_file: str | None = None,
                   replicates: int = 1, cost_mode: str | None = None) -> Dict[str, Any]:
    cfg = TestConfig()
    cost_mode = str(cost_mode or cfg.COST_MODE)
    if cost_mode not in ("xq_mse", "q3_sieve_mae"):
        raise ValueError(f"cost_mode muss 'xq_mse' oder 'q3_sieve_mae' sein, nicht {cost_mode!r}.")
    replicates = max(1, int(replicates))
    # dieselbe feste Seed-Liste wie im Adapter (SeedSequence(MC_SEED)); bei
    # replicates==1 bleibt es beim schlichten cfg.SEED (rueckwaertskompatibel).
    rep_seeds = ([int(s) for s in np.random.SeedSequence(int(cfg.SEED)).generate_state(replicates)]
                 if replicates > 1 else [int(cfg.SEED)])
    if maxiter is not None:
        cfg.MAXITER = int(maxiter)
    if t_end is not None:
        cfg.T_END = float(t_end)
    if fraction is not None:
        # SIM_FRACTION zur Laufzeit aendern -> abgeleitete Werte neu berechnen
        # (N_comp bleibt FEST; Vc, W, flow skalieren mit f)
        cfg.SIM_FRACTION = float(fraction)
        cfg.CONTROL_VOLUME = cfg.SIM_FRACTION * cfg.V_MIXER_EXP
        cfg.INITIAL_WEIGHT = cfg.SIM_FRACTION * cfg.N_REAL / cfg.INITIAL_PARTICLES
        cfg.VOLUMETRIC_FLOW_RATE = cfg.SIM_FRACTION * cfg.Q_BINDER_EXP

    # Skalierungs-Check (dieselben zwei Invarianten wie in der Config)
    _nd = cfg.INITIAL_PARTICLES * cfg.INITIAL_WEIGHT / cfg.CONTROL_VOLUME
    _phi = cfg.VOLUMETRIC_FLOW_RATE * cfg.NUCLEATION_DURATION / cfg.CONTROL_VOLUME
    assert abs(_nd / (cfg.N_REAL / cfg.V_MIXER_EXP) - 1.0) < 1e-3, "n_density inkonsistent"
    assert abs(_phi / (cfg.Q_BINDER_EXP * cfg.NUCLEATION_DURATION / cfg.V_MIXER_EXP) - 1.0) < 1e-3, "L/Vc inkonsistent"

    # Fit-Zeitpunkte (= Spalten in AJ30_xQ.xlsx) fuer direct_delta:
    times_s = np.array([t for t in AJ30_TIMES_S if t <= cfg.T_END + 1e-9], dtype=float)
    # Dichtes t_vec fuer den Solver -> haeufige [MC-PBE] Calculate ... -Fortschrittszeilen.
    # (Ein Snapshot wird nur an t_vec-Punkten gespeichert; dort druckt verbose.)
    if progress_dt and progress_dt > 0:
        grid = np.arange(0.0, cfg.T_END + 1e-9, float(progress_dt))
        t_vec = np.unique(np.concatenate([grid, times_s]))
    else:
        t_vec = times_s.copy()
    # Index jedes Fit-Zeitpunkts im (dichten) t_vec
    fit_idx = np.array([int(np.flatnonzero(np.isclose(t_vec, t))[0]) for t in times_s])

    particle_volume = np.pi / 6.0 * cfg.PARTICLE_DIAMETER ** 3
    droplet_volume = np.pi / 6.0 * cfg.DROPLET_DIAMETER ** 3
    expected_liquid = cfg.VOLUMETRIC_FLOW_RATE * cfg.NUCLEATION_DURATION
    expected_droplets = expected_liquid / droplet_volume

    sec("AJ30 FIT-SCHAETZUNG  --  Kernel-Satz + Groessen wie im Fit")
    print(f"  agg=kinetic_blend (AGG_A={cfg.AGG_A:g})  acc=stokes_dynamik  break=powerlaw_rumpf_dynamic")
    print("  growth=cone_model  compression=porosity_compression_dynamik")
    print("  liq=liquid_internalization (+ _agglomeration)  dist=uniform_weighted")

    sec("EINGANGSGROESSEN", "-")
    print(f"  SIM_FRACTION f    :  {cfg.SIM_FRACTION:g}   (Vc = f * {fmt(cfg.V_MIXER_EXP,'m^3')})")
    print(f"  Primaerpartikel d :  {cfg.PARTICLE_DIAMETER*1e6:.1f} um   V = {fmt(particle_volume,'m^3')}")
    print(f"  INITIAL_POROSITY  :  {cfg.INITIAL_POROSITY:.2f}")
    print(f"  N_comp (fest) / W :  {cfg.INITIAL_PARTICLES}  /  {fmt(cfg.INITIAL_WEIGHT)}  (W ~ f)")
    print(f"  Vc                :  {fmt(cfg.CONTROL_VOLUME,'m^3')}")
    print(f"  sum(W) (Partikelzahl):  {fmt(cfg.INITIAL_PARTICLES*cfg.INITIAL_WEIGHT)}   (reale Partikel in Vc, dimensionslos)")
    print(f"  n_phys = sum(W)/Vc  :  {fmt(cfg.INITIAL_PARTICLES*cfg.INITIAL_WEIGHT/cfg.CONTROL_VOLUME,'1/m^3')}  "
          f"(Anzahldichte, f-invariant -- DAS ist das 'n_phys' der [MC-PBE]-Konsolenzeilen; sinkt dort mit der Agglomeration)")
    print(f"  Tropfen d         :  {cfg.DROPLET_DIAMETER*1e6:.1f} um   V = {fmt(droplet_volume,'m^3')}")
    print(f"  flow_rate         :  {fmt(cfg.VOLUMETRIC_FLOW_RATE,'m^3/s')}   Fenster [0, {cfg.NUCLEATION_DURATION:.0f}] s")
    print(f"  erwartete Tropfen :  {expected_droplets:.3e}   (batch_size {cfg.BATCH_SIZE:g})")
    print(f"  n_mixer           :  {cfg.N_MIXER:.1f} m/s")
    print(f"  agg corr_beta / AGG_A :  {fmt(cfg.AGG_COEFFICIENT)} / {cfg.AGG_A:g}   (c_mixer {cfg.C_MIXER_AGG})")
    print(f"  BREAK p1          :  {fmt(cfg.PL_P1)}   (c_mixer {cfg.C_MIXER_BREAK}, breakrval {cfg.BREAKRVAL})")
    print(f"  COMPRESSION rate  :  {fmt(cfg.COMPRESSION_RATE)}   min_poro {cfg.MIN_POROSITY}")
    print(f"  k_int             :  {fmt(cfg.LIQ_INTERN_RATE)}")
    print(f"  agg/break dW_max  :  {cfg.AGG_DW_MAX:g} / {cfg.BREAK_DW_MAX:g}")
    print(f"  Fit-Zeitpunkte [s]:  {times_s.tolist()}")
    print(f"  Snapshots (t_vec) :  {t_vec.size} Punkte, {t_vec[0]:g}..{t_vec[-1]:g} s "
          f"(alle ~{progress_dt:g} s -> Fortschrittszeilen)")
    print(f"  maxiter           :  {cfg.MAXITER:,}   max_particles {cfg.MAX_PARTICLES:,}")
    print(f"  Kostenfunktion    :  COST_MODE = {cost_mode}")

    if cost_mode == "q3_sieve_mae":
        if not SIEVE_DATA_FILE.is_file():
            raise FileNotFoundError(
                f"{SIEVE_DATA_FILE} fehlt -- zuerst studies/optimization/dataprep/prepare_aj_q3_sieve.py laufen lassen."
            )
        mesh_um, obs_sieve = read_exp_q3_sieve(SIEVE_DATA_FILE)
        q_grid = data_exp = None
        print(f"\n  exp. Daten        :  {SIEVE_DATA_FILE.name}   {len(obs_sieve)} Messungen, "
              f"Maschen {mesh_um.tolist()} um")
    else:
        if not EXP_DATA_FILE.is_file():
            raise FileNotFoundError(
                f"{EXP_DATA_FILE} fehlt -- zuerst studies/optimization/dataprep/prepare_aj30_xq.py laufen lassen."
            )
        q_grid, data_exp = read_exp_xq(EXP_DATA_FILE, times_s)
        print(f"\n  exp. Daten        :  {EXP_DATA_FILE.name}   Q-Raster {q_grid.tolist()}")

    # ------------------------------------------------------------------
    #  Kosten-Abschaetzung VOR dem Lauf
    # ------------------------------------------------------------------
    n_droplets_total = expected_liquid / droplet_volume
    est_nuc_events = n_droplets_total / cfg.BATCH_SIZE
    forced_merges_per_drop = max(1.0, droplet_volume / particle_volume)  # V_dry(Partikel) < V(Tropfen) -> Zwangs-Agglo.
    sec("KOSTEN-ABSCHAETZUNG", "-")
    print(f"  Tropfen gesamt (phys.)        : {n_droplets_total:.3e}")
    print(f"  -> Nucleation-Events (~/batch): {est_nuc_events:.3e}")
    if droplet_volume > particle_volume:
        print(f"  !! Tropfen ({cfg.DROPLET_DIAMETER*1e6:.0f} um) > Primaerpartikel "
              f"({cfg.PARTICLE_DIAMETER*1e6:.0f} um):")
        print(f"     jeder Tropfen erzwingt ~{forced_merges_per_drop:.0f} Agglomerationen "
              f"(V_Tropfen / V_Partikel), bis V_dry >= V_Tropfen.")
        print(f"  -> grobe Operationszahl nur fuer Nucleation: "
              f"{est_nuc_events*forced_merges_per_drop:.2e}")
    print("  (Der Wert unten wird an der gemessenen Event-Rate hochgerechnet.)")

    initial_diameters = np.full(cfg.INITIAL_PARTICLES, cfg.PARTICLE_DIAMETER, dtype=float)

    sec("SOLVER BAUEN", "-")
    solver = build_solver(cfg, t_vec, initial_diameters, seed=rep_seeds[0])
    v_dry0 = solver.V_flat[-1, :solver.a_tot]
    w0 = solver.W[:solver.a_tot]
    initial_solid_volume = float(np.sum(v_dry0 * (1.0 - cfg.INITIAL_POROSITY) * w0))
    initial_solid_mass = initial_solid_volume * cfg.PARTICLE_DENSITY
    print(f"  a_tot {solver.a_tot}   Vc {fmt(solver.Vc,'m^3')}   W[0] {fmt(float(solver.W[0]))}")
    print(f"  init. Feststoffmasse (Diagnose): {fmt(initial_solid_mass,'kg')}")

    # Zwangs-Agglomerationen bei der Nucleation mitzaehlen
    nuc_counters = {"drops": 0, "forced_merges": 0}
    _orig_drop = solver.nucleation._distribute_one_droplet_with_dW
    _orig_merge = getattr(solver.nucleation, "_perform_nucleation_agglomeration", None)

    def _wrap_drop(*a, **k):
        nuc_counters["drops"] += 1
        return _orig_drop(*a, **k)
    solver.nucleation._distribute_one_droplet_with_dW = _wrap_drop
    if _orig_merge is not None:
        def _wrap_merge(*a, **k):
            nuc_counters["forced_merges"] += 1
            return _orig_merge(*a, **k)
        solver.nucleation._perform_nucleation_agglomeration = _wrap_merge

    sec("SIMULATION LAEUFT ...", "=")
    t0 = time.time()
    hit_particle_limit = False
    solve_error = None
    try:
        solver.solve(maxiter=int(cfg.MAXITER), max_particles=int(cfg.MAX_PARTICLES))
    except Exception as exc:  # pragma: no cover
        solve_error = str(exc)
        import traceback
        traceback.print_exc()
    elapsed = time.time() - t0
    hit_particle_limit = bool(getattr(solver, "_hit_particle_limit", False))

    # V_save ist auf t_vec ausgerichtet (V_save[k] <-> t_vec[k]).
    n_save = len(solver.V_save)
    # Fit-Zeitpunkte, deren Snapshot erreicht wurde: (i in times_s, k in t_vec/V_save)
    fit_reached = [(i, int(k)) for i, k in enumerate(fit_idx) if int(k) < n_save]
    reached = len(fit_reached)

    sec("LAUF-STATUS", "=")
    print(f"  Wandzeit            : {elapsed:.1f} s")
    agg_events = int(getattr(solver, "real_agg_events", 0))
    break_events = int(getattr(solver, "real_break_events", 0))
    total_events = agg_events + break_events
    print(f"  Events              : agg {agg_events:,}   break {break_events:,}   "
          f"({total_events/elapsed:,.0f}/s)" if elapsed > 0 else "")
    print(f"  Snapshots           : {n_save} / {t_vec.size}   (Fit-Zeitpunkte erreicht: {reached}/{times_s.size})")
    print(f"  Partikel-Limit       : {hit_particle_limit}   (MAX_PARTICLES {cfg.MAX_PARTICLES:,})")
    if solve_error:
        print(f"  solve() Exception   : {solve_error}")
    print(f"  Nucleation droplet-events : {nuc_counters['drops']:,}   "
          f"Zwangs-Agglomerationen: {nuc_counters['forced_merges']:,}")
    if nuc_counters["drops"] > 0:
        print(f"    -> {nuc_counters['forced_merges']/nuc_counters['drops']:.1f} Zwangs-Merges pro Tropfen-Event")
        done_frac = solver.nucleation.get_statistics()["liquid_volume_added_total"] / max(expected_liquid, 1e-30)
        if done_frac >= 1.0:
            pass
        elif done_frac > 1e-3 and elapsed > 0:
            est_full = elapsed / done_frac
            print(f"    -> Binderzugabe zu {done_frac*100:.2f} % fertig in {elapsed:.0f} s "
                  f"-> Hochrechnung nur Nucleation: ~{est_full:.0f} s ({est_full/3600:.1f} h)")
        elif elapsed > 0:
            print(f"    -> Binderzugabe erst zu {done_frac*100:.4f} % fertig "
                  f"(Hochrechnung noch nicht aussagekraeftig)")
    nuc = solver.nucleation.get_statistics()
    print(f"  Binder zugegeben    : {fmt(nuc['liquid_volume_added_total'],'m^3')} "
          f"(Soll {fmt(expected_liquid,'m^3')})")
    print(f"  Tropfen (stat.)     : {nuc['droplets_added_total']:.3e}")
    Wsum = float(np.sum(solver.W[:solver.a_tot]))
    Wsum0 = cfg.INITIAL_PARTICLES * cfg.INITIAL_WEIGHT
    print(f"  sum(W) (Partikelzahl): {fmt(Wsum)}   (Start {fmt(Wsum0)} -> x{Wsum/Wsum0:.3f} ; "
          f"Netto-Agglomeration senkt die Partikelzahl, Feststoffmasse bleibt)")
    print(f"  n_phys = sum(W)/Vc  : {fmt(Wsum/solver.Vc,'1/m^3')}   "
          f"(= 'n_phys' der Konsolenzeilen ; Start {fmt(Wsum0/cfg.CONTROL_VOLUME,'1/m^3')})")
    Xum = solver.X[:solver.a_tot] * 1e6
    print(f"  n_comp final        : {solver.a_tot}")
    print(f"  x final [um]        : min {np.min(Xum):.1f}  median {np.median(Xum):.1f}  max {np.max(Xum):.1f}")

    if reached < 1:
        print("\n!! Kein einziger Snapshot -- direct_delta nicht berechenbar.")
        return {"direct_delta": None, "reached": 0, "elapsed": elapsed}

    if cost_mode == "q3_sieve_mae":
        return _finish_q3_sieve(
            cfg, solver, fit_reached, times_s, t_vec, n_save, mesh_um, obs_sieve,
            replicates, rep_seeds, initial_diameters, plot, plot_file,
            info={"reached": reached, "n_save": n_save, "agg_events": agg_events,
                  "break_events": break_events, "elapsed": elapsed,
                  "hit_particle_limit": hit_particle_limit, "solve_error": solve_error})

    # -------------------------------------------------------------------
    #  direct_delta  =  MSE( x_sim(Q,t) , x_exp(Q,t) )   an den erreichten Fit-Zeitpunkten
    # -------------------------------------------------------------------
    fit_times = np.array([times_s[i] for i, _ in fit_reached])
    data_mod = np.column_stack([
        x_at_quantiles(solver.V_save[k], solver.W_save[k], q_grid)
        for _, k in fit_reached
    ])
    data_exp_fit = data_exp[:, [i for i, _ in fit_reached]]
    direct_delta = float(np.mean((data_mod - data_exp_fit) ** 2))   # == sklearn MSE
    rmse_um = float(np.sqrt(direct_delta)) * 1e6

    # ------------------------------------------------------------------
    #  Weitere MC-Replikate (--replicates): identisch zur Adapter-Logik.
    #  Feste, verschiedene Seeds (SeedSequence(cfg.SEED)); verglichen wird
    #  der MITTELWERT der x(Q,t) -- genau das bewertet spaeter der Optimierer.
    # ------------------------------------------------------------------
    data_mod_std = None
    rep_deltas = [direct_delta]
    n_rep_ok = 1
    if replicates > 1:
        mods = [data_mod]
        _kmax = max(k for _, k in fit_reached)
        sec(f"WEITERE {replicates - 1} MC-REPLIKATE (Seeds {rep_seeds[1:]})", "-")
        for r, sd in enumerate(rep_seeds[1:], start=2):
            s_r = build_solver(cfg, t_vec, initial_diameters, seed=sd, verbose=False)
            try:
                s_r.solve(maxiter=int(cfg.MAXITER), max_particles=int(cfg.MAX_PARTICLES))
            except Exception as exc:
                print(f"  Replikat {r} (seed {sd}) gescheitert: {exc}")
                continue
            if len(s_r.V_save) <= _kmax:
                print(f"  Replikat {r} (seed {sd}): zu wenige Snapshots -- uebersprungen")
                continue
            m_r = np.column_stack([x_at_quantiles(s_r.V_save[k], s_r.W_save[k], q_grid)
                                   for _, k in fit_reached])
            d_r = float(np.mean((m_r - data_exp_fit) ** 2))
            rep_deltas.append(d_r)
            mods.append(m_r)
            print(f"  Replikat {r} (seed {sd}): direct_delta = {d_r:.4e}   RMSE {np.sqrt(d_r)*1e6:.0f} um")
        n_rep_ok = len(mods)
        stack = np.stack(mods, axis=0)
        data_mod = stack.mean(axis=0)
        data_mod_std = stack.std(axis=0, ddof=1) if n_rep_ok > 1 else None
        direct_delta = float(np.mean((data_mod - data_exp_fit) ** 2))
        rmse_um = float(np.sqrt(direct_delta)) * 1e6
        sec("ENSEMBLE  (Mittelwert -- das bewertet der Optimierer)", "=")
        print(f"  erfolgreiche Replikate       : {n_rep_ok}/{replicates}")
        print(f"  direct_delta je Replikat [m^2]: " + "  ".join(f"{d:.3e}" for d in rep_deltas))
        print(f"  direct_delta(Mittelwert) [m^2]: {direct_delta:.6e}   -> RMSE {rmse_um:.1f} um")

    sec("x(Q, t):  SIM  vs  EXP   [um]", "=")
    print("  Q     " + "".join(f"| t={int(t):>4}          " for t in fit_times))
    for i, q in enumerate(q_grid):
        cells = "".join(
            f"| {data_mod[i, k]*1e6:6.1f} / {data_exp_fit[i, k]*1e6:6.1f} "
            for k in range(len(fit_reached))
        )
        print(f"  {q:4.2f}  {cells}")

    sec("DIRECT_DELTA", "=")
    if reached < times_s.size:
        print(f"  !! nur {reached}/{times_s.size} Fit-Zeitpunkte erreicht -- Wert nur ueber diese.")
    print(f"  direct_delta = MSE( x_sim(Q,t) - x_exp(Q,t) )  ueber {q_grid.size} Quantile x {len(fit_reached)} Zeiten")
    print(f"  direct_delta [m^2]           : {direct_delta:.6e}   (0 = perfekt ; 10 = Sim gescheitert)")
    print(f"  entspr. RMSE                  : {rmse_um:.2f} um   (mittlere Durchmesser-Abweichung je Punkt)")
    per_t = np.sqrt(np.mean((data_mod - data_exp_fit) ** 2, axis=0)) * 1e6
    print(f"  RMSE je Zeit [um]            : " +
          "  ".join(f"t={int(t)}:{e:.0f}" for t, e in zip(fit_times, per_t)))

    if plot:
        _plot_results(cfg, q_grid, fit_times, data_mod, data_exp_fit,
                      t_vec, solver, n_save, direct_delta, rmse_um, plot_file,
                      data_std=data_mod_std, n_rep=n_rep_ok)

    return {
        "direct_delta": direct_delta,
        "rmse_um": rmse_um,
        "reached": reached,
        "n_save": n_save,
        "agg_events": agg_events,
        "break_events": break_events,
        "elapsed": elapsed,
        "hit_particle_limit": hit_particle_limit,
        "solve_error": solve_error,
        "replicates_ok": n_rep_ok,
        "replicate_deltas": rep_deltas,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxiter", type=int, default=None,
                    help="MC-Event-Limit ueberschreiben (Default: TestConfig.MAXITER).")
    ap.add_argument("--t-end", type=float, default=None,
                    help="Nur bis zu dieser Sim-Zeit [s] rechnen (Default 600).")
    ap.add_argument("--fraction", "-f", type=float, default=None,
                    help="SIM_FRACTION ueberschreiben (Default: TestConfig.SIM_FRACTION = 1.0).")
    ap.add_argument("--progress-dt", type=float, default=10.0,
                    help="Abstand der Snapshot-/Fortschrittszeilen [s] (Default 10; 0 = nur Fit-Zeitpunkte). "
                         "Kleiner = mehr [MC-PBE]-Zeilen, aber mehr Snapshot-Speicher.")
    ap.add_argument("--no-plot", action="store_true", help="Kein Ergebnis-PNG erzeugen.")
    ap.add_argument("--plot-file", type=str, default=None, help="Pfad fuer das Ergebnis-PNG.")
    ap.add_argument("--replicates", type=int, default=1,
                    help="Anzahl MC-Laeufe mit festen, verschiedenen Seeds (wie im Fit-Adapter). "
                         "Verglichen/geplottet wird der Mittelwert. Default 1.")
    ap.add_argument("--cost-mode", choices=("q3_sieve_mae", "xq_mse"), default=None,
                    help="Kostenfunktion (Default: TestConfig.COST_MODE).")
    args = ap.parse_args()

    res = run_estimation(maxiter=args.maxiter, t_end=args.t_end, fraction=args.fraction,
                         progress_dt=args.progress_dt, replicates=args.replicates,
                         plot=not args.no_plot, plot_file=args.plot_file,
                         cost_mode=args.cost_mode)
    sec("FERTIG", "=")
    dd = res.get("direct_delta")
    print(f"  direct_delta = {dd:.6e}" if dd is not None else "  direct_delta = n/a")
    sys.exit(0)
