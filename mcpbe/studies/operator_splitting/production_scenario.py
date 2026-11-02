"""Produktions-Szenario der Nassgranulation fuer die Splitting-Diagnose.

Bildet EINEN Replikat-Lauf des Fits 1:1 nach:

    Werte ....... studies/optimization/opt_config_wet_granulation_full.py, `pop_params`
                  (Stand 2026-10-07; KF1/KF2/KF3 haben identische Kernel-Werte,
                  sie unterscheiden sich nur in DATA_FILE / RESULT_DIR)
    Verdrahtung . pbe-optimizer/.../wet_granulation_adapter.py, `_build_solver`
                  + `solve(maxiter=MAXITER, max_particles=MAX_PARTICLES)`
    Seed ........ erster der 10 Replikat-Seeds: SeedSequence(MC_SEED).generate_state(10)[0]

Warum nicht `tests/bench/scenarios.py::build`: der Adapter setzt Dinge, die
`Scenario` nicht ausdruecken kann (liquid_dist_kernel uniform_weighted,
MAYBE_DOUBLE_VC, Merger-Einstellungen, `max_particles`). Und die Datei bleibt
unangetastet (rein additiver Ordner).

BEWUSSTE DUPLIKATION: Werte und Verdrahtung sind hier KOPIERT, nicht
importiert. Aendert sich Config oder Adapter, muss das hier nachgezogen werden.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from wmcpbe import MCPBESolver

NAME = "wet_granulation_production"

# --- Skalierung, wie in der Config hergeleitet --------------------------------
_V_MIXER_EXP = 1.0e-3
_N_REAL = 1.5937e10
_Q_BINDER_EXP = 1.7667e-6
SIM_FRACTION = 1e-7
_N_COMP = 500
_VC = SIM_FRACTION * _V_MIXER_EXP
_W_INIT = SIM_FRACTION * _N_REAL / _N_COMP
_FLOW_RATE = SIM_FRACTION * _Q_BINDER_EXP

# --- pop_params (nur die Schluessel, die _build_solver / solve lesen) ---------
POP: Dict[str, Any] = {
    "N_MIXER": 30.0,
    "MC_SEED": 7,
    "N_MC_REPLICATES": 10,
    "INITIAL_PARTICLES": _N_COMP,
    "INITIAL_WEIGHT": _W_INIT,
    "CONTROL_VOLUME": _VC,
    "MERGER_TOLERANCE": 1e-6,
    "MERGER_LOOKUP": "scan",
    "MAXITER": 50_000_000,
    "MAX_PARTICLES": 100_000,
    "MAYBE_DOUBLE_VC": True,
    "AGG_DW_MIN": 1.0,
    "AGG_DW_MAX": 5,
    "BREAK_DW_MAX": 10,
    "SIZEEVAL": 0,
    "INITIAL_DIAMETER": 34e-6,
    "INITIAL_POROSITY": 0.0,
    "DROPLET_DIAMETER": 150e-6,
    "VOLUMETRIC_FLOW_RATE": _FLOW_RATE,
    "LIQUID_ADDITION_START": 0.0,
    "LIQUID_ADDITION_DURATION": 60,
    "NUCLEATION_BATCH_SIZE": 20.0,
    "AGG_A": 1.0,
    "EKE_CORR_BETA": 5.75e-14,
    "EKE_C_MIXER": 0.0995,
    "STOKES_U_COLL_REF": 0.0794,
    "STOKES_C_VEL": 0.2852,
    "STOKES_BINDER_VISCOSITY": 0.1,
    "STOKES_RHO_SOLID": 600.0,
    "STOKES_RHO_LIQUID": 1000.0,
    "STOKES_H_A": 500e-9,
    "BREAK_P1": 1e13,
    "BREAK_P2": 1.0,
    "BREAK_C_MIXER": 0.6699,
    "BREAKRVAL": 4,
    "RUMPF_K": 2.5,
    "RUMPF_ALPHA": 1.0,
    "SURFACE_TENSION": 0.072,
    "CONTACT_ANGLE": 0.0,
    "BREAK_X_S": None,
    "BREAK_PORO_MAX": 0.9999,
    "COMPRESSION_RATE": 1.3441e-2,
    "MIN_POROSITY": 0.2,
    "COMPRESSION_C_MIXER": 0.6699,
    "LIQUID_INTERNALIZATION_RATE": 1e10,
    "USE_LIQ_INTERNALISATION_AGGLOMERATION": True,
    "USE_UNIFORM_WEIGHTED": True,
    "CONE_K_AGG": 1.0,
    "CONE_K_BREAK": 1.0,
    "CONE_DEFAULT_POROSITY": 0.0,
}

# --- Bester Kandidat der Optimierung (eigenes Szenario, kein Schalter) -------
# Quelle: von Eric kopierte Ergebniszeile (2026-10-14), Reihenfolge/id 1070,
# finished_at 2026-10-12 04:59:10, status ok, 10/10 Replikate, score 0.028586307.
# Werte so, wie sie in der Zeile stehen (Excel-Anzeige, Dezimalkomma -> Punkt);
# EKE_CORR_BETA und BREAK_P1 sind dort nur auf 6 Stellen angezeigt.
# Alle anderen Werte = POP (lokale Config), angenommen identisch zum Fit-Lauf.
NAME_BEST_1070 = "wet_granulation_best_1070"
POP_BEST_1070: Dict[str, Any] = {
    **POP,
    "AGG_A": 1.340086659,
    "EKE_CORR_BETA": 1.31008e-15,
    "STOKES_U_COLL_REF": 0.016017646,
    "BREAK_P1": 4.19624e15,
    "BREAK_P2": 1.664561989,
    "COMPRESSION_RATE": 0.140807098,
    "LIQUID_INTERNALIZATION_RATE": 14390824774.0,
    "CONE_K_AGG": 0.889455433,
    "CONE_K_BREAK": 0.374699625,
}

# --- N_comp-Scan (Event-Rate-Studie) ------------------------------------------
# Bei W < AGG_DW_MAX ist die Event-Rate proportional zu N_comp (gemessen:
# 9.3 / 23 / 46 / 93 1/s fuer N_comp 100/250/500/1000 bei W ~ 0.16..3.2).
# Damit N_comp die Rate steuert, bleibt das Paketgewicht W = _W_INIT FEST und
# Vc und Flussrate wandern mit: f_n = SIM_FRACTION * n / _N_COMP. Die beiden
# Invarianten der Config (sum(W)/Vc = n_real, flow*Dauer/Vc = phi_binder) gelten
# damit fuer jedes n; geprueft in pop_best_1070_ncomp.
def name_best_1070_ncomp(n: int) -> str:
    return f"{NAME_BEST_1070}_ncomp{int(n)}"


def pop_best_1070_ncomp(n: int) -> Dict[str, Any]:
    f_n = SIM_FRACTION * int(n) / _N_COMP
    pop = {
        **POP_BEST_1070,
        "INITIAL_PARTICLES": int(n),
        "INITIAL_WEIGHT": f_n * _N_REAL / int(n),
        "CONTROL_VOLUME": f_n * _V_MIXER_EXP,
        "VOLUMETRIC_FLOW_RATE": f_n * _Q_BINDER_EXP,
    }
    n_density = pop["INITIAL_PARTICLES"] * pop["INITIAL_WEIGHT"] / pop["CONTROL_VOLUME"]
    phi = pop["VOLUMETRIC_FLOW_RATE"] * pop["LIQUID_ADDITION_DURATION"] / pop["CONTROL_VOLUME"]
    assert abs(n_density / (_N_REAL / _V_MIXER_EXP) - 1.0) < 1e-9, n_density
    assert abs(phi / (_Q_BINDER_EXP * 60 / _V_MIXER_EXP) - 1.0) < 1e-9, phi
    assert abs(pop["INITIAL_WEIGHT"] / _W_INIT - 1.0) < 1e-12
    return pop


# t_vec der Config: Startzustand + die 9 Sim-Zeiten der Proben
T_VEC = np.array([0.0, 60.0, 180.0, 240.0, 300.0, 360.0, 420.0, 480.0, 540.0, 600.0])


def replicate_seed(k: int = 0) -> int:
    """k-ter der festen Replikat-Seeds des Adapters."""
    n_rep = int(POP["N_MC_REPLICATES"])
    return int(np.random.SeedSequence(int(POP["MC_SEED"])).generate_state(n_rep)[k])


def build_production(seed: int | None = None, t_vec: np.ndarray | None = None,
                     pop: Dict[str, Any] | None = None) -> MCPBESolver:
    """Kopie von `wet_granulation_adapter._build_solver`; Werte aus `pop` (Default POP)."""
    v = (POP if pop is None else pop).get
    t_vec = T_VEC if t_vec is None else np.asarray(t_vec, dtype=float)

    n_mixer = float(v("N_MIXER"))
    agg_a = float(v("AGG_A"))
    k_int = float(v("LIQUID_INTERNALIZATION_RATE"))
    compression_rate = float(v("COMPRESSION_RATE"))
    min_porosity = float(v("MIN_POROSITY"))
    seed_val = replicate_seed(0) if seed is None else int(seed)

    solver = MCPBESolver(
        dim=1,
        t_vec=t_vec,
        verbose=False,
        load_attr=False,
        init=False,
        seed=seed_val,
        maybe_double_control_volume=bool(v("MAYBE_DOUBLE_VC")),
        recon_enable=False,
        agg_kernel_name="kinetic_blend",
        agg_kernel_params={
            "corr_beta": float(v("EKE_CORR_BETA")),
            "blend": agg_a,
            "n_mixer": n_mixer,
            "c_mixer": float(v("EKE_C_MIXER")),
        },
        agg_acceptance_kernel_name="stokes_dynamik",
        agg_acceptance_kernel_params={
            "U_coll_ref": float(v("STOKES_U_COLL_REF")),
            "n_mixer": n_mixer,
            "c_vel": float(v("STOKES_C_VEL")),
            "binder_viscosity": float(v("STOKES_BINDER_VISCOSITY")),
            "rho_solid": float(v("STOKES_RHO_SOLID")),
            "rho_liquid": float(v("STOKES_RHO_LIQUID")),
            "h_a": float(v("STOKES_H_A")),
        },
        break_kernel_name="powerlaw_rumpf_dynamic",
        break_kernel_params={
            "p1": float(v("BREAK_P1")),
            "p2": float(v("BREAK_P2")),
            "n_mixer": n_mixer,
            "c_mixer": float(v("BREAK_C_MIXER")),
            "breakrval": int(v("BREAKRVAL")),
            "k": float(v("RUMPF_K")),
            "alpha": float(v("RUMPF_ALPHA")),
            "gamma": float(v("SURFACE_TENSION")),
            "delta": float(v("CONTACT_ANGLE")),
            "x_s": v("BREAK_X_S"),
            "poro_max": float(v("BREAK_PORO_MAX")),
        },
        porosity_growth_kernel_name="cone_model",
        porosity_growth_kernel_params={
            "k_agg": float(v("CONE_K_AGG")),
            "k_break": float(v("CONE_K_BREAK")),
            "default_porosity": float(v("CONE_DEFAULT_POROSITY")),
        },
        porosity_compression_kernel_name="porosity_compression_dynamik",
        porosity_compression_kernel_params={
            "rate": compression_rate,
            "min_porosity": min_porosity,
            "n_mixer": n_mixer,
            "c_mixer": float(v("COMPRESSION_C_MIXER")),
        },
        liquid_internalization_kernel_name="liquid_internalization",
        liquid_internalization_kernel_params={"k_int": k_int},
        liq_internalisation_agglomeration_kernel_name=(
            "liq_internalisation_agglomeration"
            if bool(v("USE_LIQ_INTERNALISATION_AGGLOMERATION")) else None
        ),
        liquid_dist_kernel_name="uniform_weighted" if bool(v("USE_UNIFORM_WEIGHTED")) else None,
    )

    solver.process_type = "mix"
    solver.agg_propensity_mode = "moment"
    solver.SIZEEVAL = int(v("SIZEEVAL"))
    solver.agg_dW_min = float(v("AGG_DW_MIN"))
    solver.agg_dW_max = float(v("AGG_DW_MAX"))
    solver.break_dW_max = float(v("BREAK_DW_MAX"))
    solver._enable_particle_merging = True
    solver.merger_lookup = str(v("MERGER_LOOKUP"))
    solver._merger_use_hash_index = (solver.merger_lookup != "scan")
    solver._fragment_merge_tol = float(v("MERGER_TOLERANCE"))

    porosity = float(v("INITIAL_POROSITY"))
    diameters = np.full(int(v("INITIAL_PARTICLES")), float(v("INITIAL_DIAMETER")))
    v_dry = np.pi / 6.0 * diameters**3
    v_flat = np.vstack((v_dry * (1.0 - porosity), v_dry))
    weights = np.full(diameters.size, float(v("INITIAL_WEIGHT")))
    solver.Vc = float(v("CONTROL_VOLUME"))
    solver._initialize_particles(init_Vc=False, V_flat=v_flat, W_init=weights)
    solver.porosity[:solver.a_tot] = porosity
    solver.liquid_volume[:solver.a_tot] = 0.0
    solver.saturation[:solver.a_tot] = 0.0
    solver._init_lmc()
    solver._initialize_samplers()

    solver.create_nucleation_handler(
        enabled=True,
        volumetric_flow_rate=float(v("VOLUMETRIC_FLOW_RATE")),
        droplet_diameter=float(v("DROPLET_DIAMETER")),
        liquid_addition_start=float(v("LIQUID_ADDITION_START")),
        liquid_addition_duration=float(v("LIQUID_ADDITION_DURATION")),
        batch_size=float(v("NUCLEATION_BATCH_SIZE")),
    )
    solver.create_continuous_processes_handler(
        enabled=True,
        k_int=k_int,
        compression_enabled=True,
        compression_rate=compression_rate,
        min_porosity=min_porosity,
    )
    return solver
