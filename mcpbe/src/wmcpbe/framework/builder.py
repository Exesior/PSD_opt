"""Recipe for a single reference run.

Core idea
---------
A worker process receives **no finished solver object**, only a recipe: a flat
``dict`` of numbers and strings plus a seed. It builds the solver itself.

Why not ship the object? Three of the solver's sub-objects
(``_particle_merger``, ``continuous_processes``, ``nucleation``) hold a back
reference ``self.solver``. A ``copy.deepcopy`` of the solver state produces
copies of those pointing at a *different*, orphaned solver -- the physics would
then quietly write into an object nobody reads. On top of that,
``NucleationHandler`` keeps a reference to the random generator
(``self._rng = solver._rng``) that a newly set seed no longer reaches, so every
repeat would nucleate identically.

Built inside the worker, those objects are created there and point at the right
solver automatically.

The values in ``REFERENCE_PARAMS`` are taken 1:1 from
``Trials/test_powerlaw_rumpf_full.py`` (class ``TestConfig``).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


# ---------------------------------------------------------------------------
# Reference configuration
# ---------------------------------------------------------------------------
# Identical to TestConfig in test_powerlaw_rumpf_full.py, with one exception:
# t_total is an ordinary parameter here, so a smoke test can shorten the run
# without touching anything else.
REFERENCE_PARAMS: Dict[str, Any] = {
    # --- time ---
    "t_total": 100.0,                 # [s]
    "t_write": 0.2,                   # output interval [s]
    "maxiter": int(1e9),

    # --- particles (initial state) ---
    "particle_diameter": 34e-6,       # [m]
    "particle_density": 500.0,        # [kg/m^3]
    "initial_porosity": 0.0,          # [-]
    "initial_particles": 2000,        # computational particles
    "initial_weight": 600.0,          # weight per computational particle
    "control_volume": 1.0,            # [m^3]

    # --- droplets ---
    "droplet_diameter": 20e-6,        # [m]
    "droplet_density": 1000.0,        # [kg/m^3]

    # --- process ---
    "volumetric_flow_rate": 3e-10,    # [m^3/s]
    "nucleation_duration": 10.0,      # [s]
    "agg_coefficient": 4.0,           # collision kernel prefactor
    "batch_size": 20,                 # droplets per nucleation event

    # --- breakage (PowerLaw-Rumpf) ---
    "pl_p1": 4e13,
    "pl_p2": 1.0,
    "g": 1000.0,                      # shear rate [1/s]
    "breakrval": 4,
    "rumpf_k": 2.5,
    "rumpf_alpha": 1.0,
    "rumpf_gamma": 0.072,             # [N/m]
    "rumpf_delta": 0.0,               # [rad]

    # --- compression ---
    "compression_enabled": True,
    "compression_rate": 0.02,         # [1/s]
    "min_porosity": 0.2,

    # --- liquid internalisation ---
    "liq_intern_enabled": True,
    "liq_intern_rate": 1e12,          # k_int

    # --- agglomeration acceptance (Stokes) ---
    "binder_viscosity": 0.1,          # [Pa*s]
    "collision_velocity": 0.5,        # [m/s]
    "h_a": 500e-9,                    # [m]

    # --- numerics ---
    "agg_dW_min": 1.0,
    "agg_dW_max": 20.0,
    "break_dW_max": 50.0,
    "sizeeval": 0,
    "agg_propensity_mode": "moment",
    "process_type": "mix",
    "recon_enable": False,

    # --- merger (purely numerical: keeps n_comp small) ---
    # NOTE: TestConfig in test_powerlaw_rumpf_full.py defines
    # MERGER_TOLERANCE = 1e-4 but never applies it to the solver, so the
    # reference run uses the solver default of 1e-6. That value is repeated
    # here so the reference stays bit-for-bit reproducible.
    "merger_tolerance": 1e-6,        # -> solver._fragment_merge_tol
    "merger_lookup": "scan",         # -> solver.merger_lookup (scan|hash|hash_lazy)
    "enable_particle_merging": True, # -> solver._enable_particle_merging
}


def resolve_params(params: Dict[str, Any] | None) -> Dict[str, Any]:
    """Fill missing keys from the reference configuration."""
    merged = dict(REFERENCE_PARAMS)
    merged.update(params or {})
    return merged


def build_reference_solver(params: Dict[str, Any], seed: int, verbose: bool = False):
    """Build the fully configured reference solver.

    Parameters
    ----------
    params
        Recipe, see :data:`REFERENCE_PARAMS`. Missing keys are filled from it.
    seed
        Seed for this one run. Creates the solver's only random generator; all
        submodules derive from it.
    verbose
        Solver progress output. Always ``False`` when running in parallel,
        otherwise several processes write into the same console at once.

    Returns
    -------
    MCPBESolver
        Fully initialised, ready for ``solver.solve(maxiter=...)``.
    """
    from wmcpbe import MCPBESolver

    p = resolve_params(params)

    t_vec = np.linspace(
        0.0,
        float(p["t_total"]),
        int(round(float(p["t_total"]) / float(p["t_write"]))) + 1,
    )

    # One RNG origin per run. The solver passes it on to nucleation, merger and
    # kernels -- which is why it may only be created here.
    rng = np.random.default_rng(seed)

    solver = MCPBESolver(
        dim=1,
        t_vec=t_vec,
        verbose=bool(verbose),
        load_attr=False,          # read no config file: N processes, one file
        init=True,                # deliberately as in the reference script, see note below
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
    # On init=True: the constructor already builds a default population that is
    # overwritten immediately below. That is redundant, but kept on purpose --
    # the reference script does the same, and dropping it would shift the state
    # of the random generator. The numbers would then no longer be comparable
    # with test_powerlaw_rumpf_full.py.

    solver.mcpbe_debug = False
    solver.agg_propensity_mode = str(p["agg_propensity_mode"])
    solver.process_type = str(p["process_type"])
    solver.recon_enable = bool(p["recon_enable"])

    # Packet sizes must be set BEFORE _initialize_samplers() reads them.
    solver.agg_dW_min = float(p["agg_dW_min"])
    solver.agg_dW_max = float(p["agg_dW_max"])
    solver.break_dW_max = float(p["break_dW_max"])
    solver.SIZEEVAL = int(p["sizeeval"])

    # Merger settings: read by the solver in _ensure_particle_merger(), which
    # in turn runs from _initialize_samplers(). They therefore have to be set
    # BEFORE _initialize_samplers(). MCPBESolver(init=True) already builds a
    # merger in the constructor; _ensure_particle_merger() rebuilds it here when
    # the config below no longer matches, so setting these after construction
    # takes effect.
    solver._fragment_merge_tol = float(p["merger_tolerance"])
    solver.merger_lookup = str(p["merger_lookup"])
    solver._enable_particle_merging = bool(p["enable_particle_merging"])

    # V_flat layout for dim=1:
    #   V_flat[0, :] = V_solid  (solid volume, unchanged by compression)
    #   V_flat[1, :] = V_dry    (geometric dry volume = V_solid + V_pore)
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
    """Target amount of liquid added [m^3] -- the reference for the balance."""
    p = resolve_params(params)
    return float(p["volumetric_flow_rate"]) * float(p["nucleation_duration"])


def expected_droplet_count(params: Dict[str, Any] | None = None) -> float:
    """Target number of droplets added -- the reference for the nucleation check."""
    p = resolve_params(params)
    droplet_volume = (4.0 / 3.0) * np.pi * (float(p["droplet_diameter"]) / 2.0) ** 3
    return expected_liquid_volume(p) / droplet_volume
