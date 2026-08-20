"""Bauplan fuer einen einzelnen Referenzlauf.

Kernidee
--------
Ein Worker-Prozess bekommt **kein fertiges Solver-Objekt**, sondern nur ein
Rezept: ein flaches ``dict`` aus Zahlen und Strings plus einen Seed. Er baut
den Solver damit selbst.

Warum nicht das Objekt selbst verschicken? Drei Unterobjekte des Solvers
(``_particle_merger``, ``continuous_processes``, ``nucleation``) halten eine
Rueckreferenz ``self.solver``. Ein ``copy.deepcopy`` des Solver-Zustands
erzeugt davon Kopien, die auf einen *anderen* (verwaisten) Solver zeigen --
die Physik wuerde dann still in ein Objekt schreiben, das niemand ausliest.
Ausserdem haelt ``NucleationHandler`` eine Referenz auf den Zufallsgenerator
(``self._rng = solver._rng``), die ein neu gesetzter Seed nicht mehr erreicht;
alle Wiederholungen wuerden identisch nukleieren.

Wird der Solver dagegen im Worker gebaut, entstehen diese Objekte dort und
zeigen automatisch auf den richtigen Solver.

Die Parameterwerte in ``REFERENCE_PARAMS`` sind 1:1 aus
``Trials/test_powerlaw_rumpf_full.py`` (Klasse ``TestConfig``) uebernommen.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


# ---------------------------------------------------------------------------
# Referenzkonfiguration
# ---------------------------------------------------------------------------
# Identisch zu TestConfig in test_powerlaw_rumpf_full.py. Einzige Ausnahme:
# t_total ist hier ein normaler Parameter, damit fuer Rauchtests eine kurze
# Laufzeit gesetzt werden kann, ohne den Rest anzufassen.
REFERENCE_PARAMS: Dict[str, Any] = {
    # --- Zeit ---
    "t_total": 100.0,                 # [s]
    "t_write": 0.2,                   # Ausgabeintervall [s]
    "maxiter": int(1e9),

    # --- Partikel (Anfangszustand) ---
    "particle_diameter": 34e-6,       # [m]
    "particle_density": 500.0,        # [kg/m^3]
    "initial_porosity": 0.0,          # [-]
    "initial_particles": 2000,        # Rechenpartikel
    "initial_weight": 600.0,          # Gewicht je Rechenpartikel
    "control_volume": 1.0,            # [m^3]

    # --- Tropfen ---
    "droplet_diameter": 20e-6,        # [m]
    "droplet_density": 1000.0,        # [kg/m^3]

    # --- Prozess ---
    "volumetric_flow_rate": 3e-10,    # [m^3/s]
    "nucleation_duration": 10.0,      # [s]
    "agg_coefficient": 4.0,           # Vorfaktor Kollisionskernel
    "batch_size": 20,                 # Tropfen je Nukleationsereignis

    # --- Bruch (PowerLaw-Rumpf) ---
    "pl_p1": 4e13,
    "pl_p2": 1.0,
    "g": 1000.0,                      # Scherrate [1/s]
    "breakrval": 4,
    "rumpf_k": 2.5,
    "rumpf_alpha": 1.0,
    "rumpf_gamma": 0.072,             # [N/m]
    "rumpf_delta": 0.0,               # [rad]

    # --- Kompression ---
    "compression_enabled": True,
    "compression_rate": 0.02,         # [1/s]
    "min_porosity": 0.2,

    # --- Fluessigkeits-Internalisierung ---
    "liq_intern_enabled": True,
    "liq_intern_rate": 1e12,          # k_int

    # --- Agglomerations-Akzeptanz (Stokes) ---
    "binder_viscosity": 0.1,          # [Pa*s]
    "collision_velocity": 0.5,        # [m/s]
    "h_a": 500e-9,                    # [m]

    # --- Numerik ---
    "agg_dW_min": 1.0,
    "agg_dW_max": 20.0,
    "break_dW_max": 50.0,
    "sizeeval": 0,
    "agg_propensity_mode": "moment",
    "process_type": "mix",
    "recon_enable": False,

    # --- Merger (rein numerisch: haelt n_comp klein) ---
    # ACHTUNG: TestConfig in test_powerlaw_rumpf_full.py definiert
    # MERGER_TOLERANCE = 1e-4, wendet den Wert aber nie auf den Solver an. Der
    # Referenzlauf verwendet also den Solver-Default 1e-6. Genau dieser Wert
    # steht hier, damit die Referenz bitgenau reproduziert bleibt.
    "merger_tolerance": 1e-6,        # -> solver._fragment_merge_tol
    "merger_use_hash_index": True,   # -> solver._merger_use_hash_index
    "enable_particle_merging": True, # -> solver._enable_particle_merging
}


def resolve_params(params: Dict[str, Any] | None) -> Dict[str, Any]:
    """Ergaenze fehlende Schluessel aus der Referenz."""
    merged = dict(REFERENCE_PARAMS)
    merged.update(params or {})
    return merged


def build_reference_solver(params: Dict[str, Any], seed: int, verbose: bool = False):
    """Baue den vollstaendig konfigurierten Referenz-Solver.

    Parameters
    ----------
    params
        Rezept, siehe :data:`REFERENCE_PARAMS`. Fehlende Schluessel werden aus
        der Referenz ergaenzt.
    seed
        Seed fuer diesen einen Lauf. Erzeugt den einzigen Zufallsgenerator des
        Solvers; alle Untermodule leiten sich davon ab.
    verbose
        Fortschrittsausgabe des Solvers. Im Parallelbetrieb immer ``False``,
        sonst schreiben mehrere Prozesse gleichzeitig in dieselbe Konsole.

    Returns
    -------
    MCPBESolver
        Fertig initialisiert, bereit fuer ``solver.solve(maxiter=...)``.
    """
    from wmcpbe import MCPBESolver

    p = resolve_params(params)

    t_vec = np.linspace(
        0.0,
        float(p["t_total"]),
        int(round(float(p["t_total"]) / float(p["t_write"]))) + 1,
    )

    # Ein einziger RNG-Ursprung pro Lauf. Der Solver reicht ihn an Nucleation,
    # Merger und Kernel weiter -- deshalb darf er nur hier entstehen.
    rng = np.random.default_rng(seed)

    solver = MCPBESolver(
        dim=1,
        t_vec=t_vec,
        verbose=bool(verbose),
        load_attr=False,          # keine Config-Datei lesen: N Prozesse, eine Datei
        init=True,                # bewusst wie im Referenzskript, siehe Hinweis unten
        rng=rng,
        agg_kernel_name="shear_chin1998",
        agg_kernel_params={"corr_beta": float(p["agg_coefficient"]), "g": float(p["g"])},
        agg_acceptance_kernel_name="stokes_krit",
        agg_acceptance_kernel_params={
            "U_coll": float(p["collision_velocity"]),
            "binder_viscosity": float(p["binder_viscosity"]),
            "rho_solid": float(p["particle_density"]),
            "rho_liquid": float(p["droplet_density"]),
            "h_a": float(p["h_a"]),
        },
        break_kernel_name="powerlaw_rumpf",
        break_kernel_params={
            "p1": float(p["pl_p1"]),
            "p2": float(p["pl_p2"]),
            "g": float(p["g"]),
            "breakrval": int(p["breakrval"]),
            "k": float(p["rumpf_k"]),
            "alpha": float(p["rumpf_alpha"]),
            "gamma": float(p["rumpf_gamma"]),
            "delta": float(p["rumpf_delta"]),
            "x_s": None,
        },
        porosity_growth_kernel_name="cone_model",
        porosity_growth_kernel_params={},
        porosity_compression_kernel_name="porosity_compression",
        porosity_compression_kernel_params={
            "rate": float(p["compression_rate"]),
            "min_porosity": float(p["min_porosity"]),
        },
        liquid_internalization_kernel_name="liquid_internalization",
        liquid_internalization_kernel_params={"k_int": float(p["liq_intern_rate"])},
        liq_internalisation_agglomeration_kernel_name="liq_internalisation_agglomeration",
        liq_internalisation_agglomeration_kernel_params={},
    )
    # HINWEIS zu init=True: der Konstruktor initialisiert damit bereits eine
    # Default-Population, die unten sofort ueberschrieben wird. Das ist
    # redundant, wird aber absichtlich beibehalten -- das Referenzskript macht
    # es genauso, und ein Weglassen wuerde den Zustand des Zufallsgenerators
    # verschieben. Damit waeren die Zahlen nicht mehr mit
    # test_powerlaw_rumpf_full.py vergleichbar.

    solver.mcpbe_debug = False
    solver.agg_propensity_mode = str(p["agg_propensity_mode"])
    solver.process_type = str(p["process_type"])
    solver.recon_enable = bool(p["recon_enable"])

    # Paketgroessen muessen stehen, BEVOR _initialize_samplers() sie liest.
    solver.agg_dW_min = float(p["agg_dW_min"])
    solver.agg_dW_max = float(p["agg_dW_max"])
    solver.break_dW_max = float(p["break_dW_max"])
    solver.SIZEEVAL = int(p["sizeeval"])

    # Merger-Stellschrauben: werden in _ensure_particle_merger() vom Solver
    # gelesen, das wiederum aus _initialize_samplers() heraus laeuft. Sie
    # muessen deshalb VOR _initialize_samplers() stehen.
    solver._fragment_merge_tol = float(p["merger_tolerance"])
    solver._merger_use_hash_index = bool(p["merger_use_hash_index"])
    solver._enable_particle_merging = bool(p["enable_particle_merging"])

    # V_flat-Aufbau fuer dim=1:
    #   V_flat[0, :] = V_solid  (Feststoffvolumen, bleibt bei Kompression erhalten)
    #   V_flat[1, :] = V_dry    (geometrisches Trockenvolumen = V_solid + V_pore)
    n0 = int(p["initial_particles"])
    particle_volume = (4.0 / 3.0) * np.pi * (float(p["particle_diameter"]) / 2.0) ** 3
    V_flat = np.zeros((2, n0), dtype=float)
    V_flat[0, :] = particle_volume * (1.0 - float(p["initial_porosity"]))
    V_flat[1, :] = particle_volume
    W_init = np.full(n0, float(p["initial_weight"]), dtype=float)

    solver.Vc = float(p["control_volume"])
    solver._initialize_particles(init_Vc=False, V_flat=V_flat, W_init=W_init)

    solver.porosity[: solver.a_tot] = float(p["initial_porosity"])
    solver.liquid_volume[: solver.a_tot] = 0.0
    solver.saturation[: solver.a_tot] = 0.0

    solver._initialize_samplers()

    solver.create_nucleation_handler(
        enabled=True,
        volumetric_flow_rate=float(p["volumetric_flow_rate"]),
        droplet_diameter=float(p["droplet_diameter"]),
        liquid_addition_start=0.0,
        liquid_addition_duration=float(p["nucleation_duration"]),
        batch_size=float(p["batch_size"]),
    )

    if bool(p["liq_intern_enabled"]) or bool(p["compression_enabled"]):
        solver.create_continuous_processes_handler(
            enabled=True,
            k_int=float(p["liq_intern_rate"]) if bool(p["liq_intern_enabled"]) else 0.0,
            compression_enabled=bool(p["compression_enabled"]),
            compression_rate=float(p["compression_rate"]),
            min_porosity=float(p["min_porosity"]),
        )

    return solver


def expected_liquid_volume(params: Dict[str, Any] | None = None) -> float:
    """Sollmenge zugefuehrter Fluessigkeit [m^3] -- Referenz fuer die Bilanz."""
    p = resolve_params(params)
    return float(p["volumetric_flow_rate"]) * float(p["nucleation_duration"])


def expected_droplet_count(params: Dict[str, Any] | None = None) -> float:
    """Sollzahl zugefuehrter Tropfen -- Referenz fuer die Nukleationspruefung."""
    p = resolve_params(params)
    droplet_volume = (4.0 / 3.0) * np.pi * (float(p["droplet_diameter"]) / 2.0) ** 3
    return expected_liquid_volume(p) / droplet_volume
