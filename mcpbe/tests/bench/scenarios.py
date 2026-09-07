"""Reproducible WMCPBE benchmark / regression scenarios.

Every scenario builds a solver from an explicit, self-contained configuration
(``load_attr=False``) so results never depend on whatever ``MCPBE_config.py``
happens to sit in the current working directory.

The scenarios are used for two things:

1. ``bench_wmcpbe.py``     - wall-clock timing of the hot paths.
2. ``golden_reference.py`` - bit-exact regression fingerprints. Every scenario
   is seeded, so a behaviour-preserving refactor must reproduce the recorded
   fingerprint exactly.

Design rule: every scenario is bounded by a **fixed event budget**
(``Scenario.maxiter``), not by simulated time. That makes the wall-clock number
directly proportional to *cost per Monte-Carlo event*, which is exactly the
quantity the optimisation work targets, and it keeps the whole suite inside a
predictable time budget regardless of kernel parameters.

Keep this file free of dependencies beyond numpy + wmcpbe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from wmcpbe import MCPBESolver
from wmcpbe.kernels.mixer_speed import C_BREAK, N_MIXER_DEFAULT


@dataclass(frozen=True)
class Scenario:
    """A single, fully specified simulation run."""

    name: str
    dim: int
    process_type: str
    a0: float
    t_end: float
    maxiter: int
    n_save: int = 7
    seed: int = 20240607
    agg_kernel_name: Optional[str] = "shear_chin1998"
    agg_kernel_params: Dict[str, Any] = field(
        default_factory=lambda: {"corr_beta": 1e-3, "g": 1000.0}
    )
    break_kernel_name: Optional[str] = None
    break_kernel_params: Dict[str, Any] = field(default_factory=dict)
    solver_attrs: Dict[str, Any] = field(default_factory=dict)
    # Nucleation + continuous-process handlers (granulation runs only).
    granulation: bool = False
    notes: str = ""

    # --- optional kernel slots -------------------------------------------
    # All default to None, i.e. "not configured", which is exactly what the
    # solver saw before these fields existed. A scenario that leaves them unset
    # is byte-identical to the pre-extension behaviour.
    agg_acceptance_kernel_name: Optional[str] = None
    agg_acceptance_kernel_params: Dict[str, Any] = field(default_factory=dict)
    porosity_growth_kernel_name: Optional[str] = None
    porosity_growth_kernel_params: Dict[str, Any] = field(default_factory=dict)
    porosity_compression_kernel_name: Optional[str] = None
    porosity_compression_kernel_params: Dict[str, Any] = field(default_factory=dict)
    liquid_internalization_kernel_name: Optional[str] = None
    liquid_internalization_kernel_params: Dict[str, Any] = field(default_factory=dict)
    liq_internalisation_agglomeration_kernel_name: Optional[str] = None
    liq_internalisation_agglomeration_kernel_params: Dict[str, Any] = field(
        default_factory=dict
    )

    # --- optional handler / init overrides -------------------------------
    #: Replaces the defaults in ``_attach_granulation_handlers``. Only read when
    #: ``granulation`` is True.
    nucleation_params: Optional[Dict[str, Any]] = None
    continuous_params: Optional[Dict[str, Any]] = None
    #: Monodisperse init through ``V_flat``/``W_init`` instead of the c/x/PGV
    #: path, mirroring how the Trials scripts set a run up (see Modul 5.1,
    #: mode 2). Keys: particle_diameter [m], initial_porosity [-],
    #: initial_particles [-], initial_weight [-], control_volume [m^3].
    init_particles: Optional[Dict[str, Any]] = None


def build(scenario: Scenario) -> MCPBESolver:
    """Instantiate and initialise a solver for ``scenario`` (does not solve)."""
    t_vec = np.linspace(0.0, scenario.t_end, scenario.n_save)

    solver = MCPBESolver(
        dim=scenario.dim,
        t_vec=t_vec,
        verbose=False,
        load_attr=False,
        init=False,
        seed=scenario.seed,
        agg_kernel_name=scenario.agg_kernel_name,
        agg_kernel_params=dict(scenario.agg_kernel_params) or None,
        break_kernel_name=scenario.break_kernel_name,
        break_kernel_params=dict(scenario.break_kernel_params) or None,
        agg_acceptance_kernel_name=scenario.agg_acceptance_kernel_name,
        agg_acceptance_kernel_params=dict(scenario.agg_acceptance_kernel_params) or None,
        porosity_growth_kernel_name=scenario.porosity_growth_kernel_name,
        porosity_growth_kernel_params=dict(scenario.porosity_growth_kernel_params) or None,
        porosity_compression_kernel_name=scenario.porosity_compression_kernel_name,
        porosity_compression_kernel_params=(
            dict(scenario.porosity_compression_kernel_params) or None
        ),
        liquid_internalization_kernel_name=scenario.liquid_internalization_kernel_name,
        liquid_internalization_kernel_params=(
            dict(scenario.liquid_internalization_kernel_params) or None
        ),
        liq_internalisation_agglomeration_kernel_name=(
            scenario.liq_internalisation_agglomeration_kernel_name
        ),
        liq_internalisation_agglomeration_kernel_params=(
            dict(scenario.liq_internalisation_agglomeration_kernel_params) or None
        ),
    )

    dim = scenario.dim
    solver.process_type = scenario.process_type
    solver.c = np.full(dim, 1e-2 / dim)
    solver.x = np.full(dim, 1e-6)
    solver.PGV = np.full(dim, "mono")
    solver.SIG = np.full(dim, 0.1)
    solver.a0 = scenario.a0
    solver.alpha_prim = np.ones(dim**2)
    solver.maybe_double_control_volume = False

    # solver_attrs are applied before _initialize_samplers(), so anything the
    # merger reads there takes effect -- e.g. solver_attrs={"merger_lookup":
    # "hash"} to compare the hash index against the default linear scan.
    for key, value in scenario.solver_attrs.items():
        setattr(solver, key, value)

    if scenario.init_particles is None:
        solver._initialize_particles()
    else:
        _initialize_monodisperse(solver, scenario.init_particles)
    solver._init_lmc()
    solver._initialize_samplers()

    if scenario.granulation:
        _attach_granulation_handlers(
            solver,
            nucleation_params=scenario.nucleation_params,
            continuous_params=scenario.continuous_params,
        )

    return solver


def _initialize_monodisperse(solver: MCPBESolver, spec: Dict[str, Any]) -> None:
    """Init from an explicit ``V_flat`` / ``W_init``, as the Trials scripts do.

    ``V_flat[0]`` is the solid volume, ``V_flat[-1]`` the dry volume; the two
    differ by the pore space, hence the ``(1 - porosity)`` factor. Porosity,
    liquid and saturation are set AFTER ``_initialize_particles`` (which zeroes
    them) and BEFORE ``_initialize_samplers`` (which reads them for the breakage
    rate) -- see the ordering rule in the project overview.
    """
    diameter = float(spec["particle_diameter"])
    poro = float(spec.get("initial_porosity", 0.0))
    n_part = int(spec["initial_particles"])
    weight = float(spec["initial_weight"])

    particle_volume = (4.0 / 3.0) * np.pi * (diameter / 2.0) ** 3

    V_flat = np.zeros((solver.dim + 1, n_part), dtype=float)
    V_flat[0, :] = particle_volume * (1.0 - poro)
    V_flat[-1, :] = particle_volume
    W_init = np.full(n_part, weight, dtype=float)

    solver.Vc = float(spec.get("control_volume", 1.0))
    solver._initialize_particles(init_Vc=False, V_flat=V_flat, W_init=W_init)

    a = solver.a_tot
    solver.porosity[:a] = poro
    solver.liquid_volume[:a] = 0.0
    solver.saturation[:a] = 0.0


def _attach_granulation_handlers(
    solver: MCPBESolver,
    nucleation_params: Optional[Dict[str, Any]] = None,
    continuous_params: Optional[Dict[str, Any]] = None,
) -> None:
    """Attach nucleation + continuous processes (liquid uptake and porosity).

    Flow rate is deliberately small in the default: nucleation events are far
    cheaper than agglomeration events, and an aggressive flow rate would swamp
    the benchmark with nucleation bookkeeping instead of exercising the
    agglomeration path. A scenario that models a real process passes its own
    ``nucleation_params`` / ``continuous_params`` instead.
    """
    nuc = nucleation_params or dict(
        enabled=True,
        volumetric_flow_rate=5e-19,
        droplet_diameter=1e-7,
        liquid_addition_duration=2.0,
        liquid_addition_start=0.0,
        batch_size=25.0,
    )
    cont = continuous_params or dict(
        enabled=True,
        k_int=1e12,
        compression_enabled=True,
        compression_rate=0.02,
        min_porosity=0.3,
    )
    solver.create_nucleation_handler(**nuc)
    solver.create_continuous_processes_handler(**cont)


# --- shared kernel parameter sets ------------------------------------------
SHEAR = {"corr_beta": 1e-3, "g": 1000.0}
# breakrval=1 -> constant single-particle breakage rate S = p1. Small p1 keeps
# the event rate in a range where the run stays inside its event budget without
# the time step collapsing. Volume-dependent variants (breakrval=3/4) are
# covered by the unit tests, not by the benchmark.
# NOTE: pl_v / pl_q are breakage FUNCTION parameters (fragment size
# distribution) and are rejected by the rate kernels -- the rate's volume
# exponent is p2. Der BREAKRVAL-Schalter steckt seit 20.08.2026 in jedem
# Bruchkernel selbst (kein gemeinsames Modul mehr), siehe power_law.py.
POWER_LAW = {"p1": 1e-2, "p2": 1.0, "g": 1000.0, "breakrval": 1}
# BREAKFVAL selects the fragment-count model in _compute_frag_num(); pin it so
# scenarios do not silently inherit the BaseSolver default.
BREAK_ATTRS = {"break_dW_max": 50.0, "BREAKFVAL": 2}

# Mixer-speed compression variants. The dynamic kernels replace the fixed rate
# k of `porosity_compression` with k = rate * n_mixer**c_mixer (the `_rumpf`
# one additionally / sigma(eps, S)). The rate values below are picked so that at
# n_mixer = 20 the effective k reproduces the 0.1 that
# `granulation_rumpf_dynamic_1d` uses with the static kernel -- the three
# granulation scenarios then differ ONLY in how the compression rate is formed.
_COMPRESSION_K_TARGET = 0.1
_COMPRESSION_N_FACTOR = N_MIXER_DEFAULT ** C_BREAK
_COMPRESSION_DYNAMIK_RATE = _COMPRESSION_K_TARGET / _COMPRESSION_N_FACTOR
# Reference strength: dry regime, eps = 0.4, x_s = 34 um (the monodisperse init
# diameter), default k_rumpf = 2.5, gamma = 0.072 -> sigma = phi(S=0)*(1-eps)/eps.
_COMPRESSION_SIGMA_REF = (0.072 / 34e-6) * 2.5 * (1.0 - 0.4) / 0.4
_COMPRESSION_DYNAMIK_RUMPF_RATE = (
    _COMPRESSION_K_TARGET * _COMPRESSION_SIGMA_REF / _COMPRESSION_N_FACTOR
)


SCENARIOS: Dict[str, Scenario] = {
    s.name: s
    for s in (
        Scenario(
            name="agg_shear_1d",
            dim=1,
            process_type="agglomeration",
            a0=500,
            t_end=120.0,
            maxiter=400,
            notes="Baseline granulation kernel, 1D.",
        ),
        Scenario(
            name="agg_shear_1d_large",
            dim=1,
            process_type="agglomeration",
            a0=2000,
            t_end=40.0,
            maxiter=150,
            notes="Same kernel at 4x particle count: exposes O(n^2) scaling.",
        ),
        Scenario(
            name="agg_constant_1d",
            dim=1,
            process_type="agglomeration",
            a0=500,
            t_end=2.0e6,
            maxiter=400,
            agg_kernel_name="constant",
            agg_kernel_params={"corr_beta": 1e-3},
        ),
        Scenario(
            name="agg_sum_1d",
            dim=1,
            process_type="agglomeration",
            a0=500,
            t_end=2.0e13,
            maxiter=400,
            agg_kernel_name="sum",
            agg_kernel_params={"corr_beta": 1e-3},
        ),
        Scenario(
            name="agg_brownian_1d",
            dim=1,
            process_type="agglomeration",
            a0=500,
            t_end=2.0e-4,
            maxiter=400,
            agg_kernel_name="brownian_tsouris1995",
            agg_kernel_params={"corr_beta": 1e-3},
        ),
        Scenario(
            name="agg_shear_2d",
            dim=2,
            process_type="agglomeration",
            a0=500,
            t_end=120.0,
            maxiter=300,
            # BREAKFVAL must be set even for pure agglomeration: the BaseSolver
            # default (3) makes _compute_frag_num() raise for dim=2.
            solver_attrs={"BREAKFVAL": 2},
        ),
        Scenario(
            name="break_powerlaw_1d",
            dim=1,
            process_type="breakage",
            a0=500,
            t_end=300.0,
            maxiter=1500,
            break_kernel_name="power_law",
            break_kernel_params=POWER_LAW,
            solver_attrs=dict(BREAK_ATTRS),
        ),
        Scenario(
            name="mix_1d",
            dim=1,
            process_type="mix",
            a0=500,
            t_end=120.0,
            maxiter=300,
            break_kernel_name="power_law",
            break_kernel_params=POWER_LAW,
            solver_attrs=dict(BREAK_ATTRS),
            notes="Worst case: every event rebuilds both agg and break state.",
        ),
        Scenario(
            name="granulation_1d",
            dim=1,
            process_type="agglomeration",
            a0=400,
            t_end=4.0,
            maxiter=300,
            granulation=True,
            notes="Agglomeration + nucleation + liquid internalisation + compression.",
        ),
        # The full wet-granulation production configuration, mirroring
        # Trials/test_powerlaw_rumpf_dynamic_full.py (same kernel parameters,
        # same seed, same monodisperse init). The run is bounded by a fixed
        # event budget rather than by simulated time, per the design rule above.
        #
        # This is the scenario that matters for the propensity-rebuild work:
        # EKE is NOT separable, so it takes the compiled O(n^2) pairwise path
        # (no moment form), and all three handler paths are live -- nucleation,
        # liquid internalisation and porosity compression. That combination is
        # what makes solve() rebuild the propensities ~1.6x per iteration.
        #
        # Why maxiter=1600 and not something cheaper: the Rumpf strength model
        # only lets particles break once they have grown porous and wet, so the
        # FIRST breakage event lands at iteration ~495 (t ~ 35 s). A shorter
        # budget exercises the agglomeration rebuild path only and leaves
        # `_do_one_break`'s full agglomeration rebuild (mcpbe_break.py, mix mode)
        # completely uncovered. At 1600 events roughly 23 breakage events fire,
        # which is ample: for a bit-exactness fingerprint a single one suffices,
        # because any divergence propagates through the rest of the RNG stream.
        Scenario(
            name="granulation_rumpf_dynamic_1d",
            dim=1,
            process_type="mix",
            a0=1000,
            t_end=120.0,
            maxiter=1600,
            seed=205,
            agg_kernel_name="eke_darelius2005",
            agg_kernel_params={"corr_beta": 5e-8, "n_mixer": 20.0},
            agg_acceptance_kernel_name="stokes_dynamik",
            agg_acceptance_kernel_params={
                "U_coll_ref": 0.0794,
                "n_mixer": 20.0,
                "binder_viscosity": 0.1,
                "rho_solid": 600.0,
                "rho_liquid": 1000.0,
                "h_a": 500e-9,
            },
            break_kernel_name="powerlaw_rumpf_dynamic",
            break_kernel_params={
                "p1": 1e13,
                "p2": 1.0,
                "n_mixer": 20.0,
                "breakrval": 4,
                "k": 2.5,
                "alpha": 1.0,
                "gamma": 0.072,
                "delta": 0.0,
                "x_s": None,
            },
            porosity_growth_kernel_name="cone_model",
            porosity_compression_kernel_name="porosity_compression",
            porosity_compression_kernel_params={"rate": 0.1, "min_porosity": 0.2},
            liquid_internalization_kernel_name="liquid_internalization",
            liquid_internalization_kernel_params={"k_int": 1e11},
            liq_internalisation_agglomeration_kernel_name=(
                "liq_internalisation_agglomeration"
            ),
            solver_attrs={
                "agg_propensity_mode": "moment",
                "recon_enable": False,
                "agg_dW_min": 1.0,
                "agg_dW_max": 20.0,
                "break_dW_max": 50.0,
                "SIZEEVAL": 0,
            },
            init_particles={
                "particle_diameter": 34e-6,
                "initial_porosity": 0.0,
                "initial_particles": 1000,
                "initial_weight": 600.0,
                "control_volume": 1.0,
            },
            granulation=True,
            nucleation_params=dict(
                enabled=True,
                volumetric_flow_rate=1.635e-10,
                droplet_diameter=20e-6,
                liquid_addition_start=0.0,
                liquid_addition_duration=20.0,
                batch_size=20,
            ),
            continuous_params=dict(
                enabled=True,
                k_int=1e11,
                compression_enabled=True,
                compression_rate=0.1,
                min_porosity=0.2,
            ),
            notes=(
                "Production wet granulation (EKE + stokes_dynamik + "
                "powerlaw_rumpf_dynamic + cone_model + nucleation + continuous "
                "processes). Mirrors Trials/test_powerlaw_rumpf_dynamic_full.py."
            ),
        ),
        # Two deliberate copies of granulation_rumpf_dynamic_1d that change ONLY
        # the porosity-compression slot: the mixer-speed variant, and the
        # mixer-speed + Rumpf-strength variant. Same seed, same everything else,
        # so a fingerprint diff against granulation_rumpf_dynamic_1d isolates
        # exactly the compression-kernel effect. Both keep n_mixer = 20 to match
        # the other mixer-speed kernels (assert_consistent_mixer_speed).
        Scenario(
            name="granulation_compression_dynamik_1d",
            dim=1,
            process_type="mix",
            a0=1000,
            t_end=120.0,
            maxiter=1600,
            seed=205,
            agg_kernel_name="eke_darelius2005",
            agg_kernel_params={"corr_beta": 5e-8, "n_mixer": 20.0},
            agg_acceptance_kernel_name="stokes_dynamik",
            agg_acceptance_kernel_params={
                "U_coll_ref": 0.0794,
                "n_mixer": 20.0,
                "binder_viscosity": 0.1,
                "rho_solid": 600.0,
                "rho_liquid": 1000.0,
                "h_a": 500e-9,
            },
            break_kernel_name="powerlaw_rumpf_dynamic",
            break_kernel_params={
                "p1": 1e13,
                "p2": 1.0,
                "n_mixer": 20.0,
                "breakrval": 4,
                "k": 2.5,
                "alpha": 1.0,
                "gamma": 0.072,
                "delta": 0.0,
                "x_s": None,
            },
            porosity_growth_kernel_name="cone_model",
            porosity_compression_kernel_name="porosity_compression_dynamik",
            porosity_compression_kernel_params={
                "rate": _COMPRESSION_DYNAMIK_RATE,
                "min_porosity": 0.2,
                "n_mixer": 20.0,
            },
            liquid_internalization_kernel_name="liquid_internalization",
            liquid_internalization_kernel_params={"k_int": 1e11},
            liq_internalisation_agglomeration_kernel_name=(
                "liq_internalisation_agglomeration"
            ),
            solver_attrs={
                "agg_propensity_mode": "moment",
                "recon_enable": False,
                "agg_dW_min": 1.0,
                "agg_dW_max": 20.0,
                "break_dW_max": 50.0,
                "SIZEEVAL": 0,
            },
            init_particles={
                "particle_diameter": 34e-6,
                "initial_porosity": 0.0,
                "initial_particles": 1000,
                "initial_weight": 600.0,
                "control_volume": 1.0,
            },
            granulation=True,
            nucleation_params=dict(
                enabled=True,
                volumetric_flow_rate=1.635e-10,
                droplet_diameter=20e-6,
                liquid_addition_start=0.0,
                liquid_addition_duration=20.0,
                batch_size=20,
            ),
            continuous_params=dict(
                enabled=True,
                k_int=1e11,
                compression_enabled=True,
                compression_rate=0.1,
                min_porosity=0.2,
            ),
            notes=(
                "granulation_rumpf_dynamic_1d with the mixer-speed compression "
                "kernel (k = rate * n_mixer**C_BREAK) instead of a fixed rate. "
                "rate is calibrated so k == 0.1 exactly at n_mixer = 20, so the "
                "fingerprint is EXPECTED to equal granulation_rumpf_dynamic_1d "
                "-- a drop-in-equivalence guard. The conservation suite still "
                "exercises the dynamik compute_array path independently."
            ),
        ),
        Scenario(
            name="granulation_compression_dynamik_rumpf_1d",
            dim=1,
            process_type="mix",
            a0=1000,
            t_end=120.0,
            maxiter=1600,
            seed=205,
            agg_kernel_name="eke_darelius2005",
            agg_kernel_params={"corr_beta": 5e-8, "n_mixer": 20.0},
            agg_acceptance_kernel_name="stokes_dynamik",
            agg_acceptance_kernel_params={
                "U_coll_ref": 0.0794,
                "n_mixer": 20.0,
                "binder_viscosity": 0.1,
                "rho_solid": 600.0,
                "rho_liquid": 1000.0,
                "h_a": 500e-9,
            },
            break_kernel_name="powerlaw_rumpf_dynamic",
            break_kernel_params={
                "p1": 1e13,
                "p2": 1.0,
                "n_mixer": 20.0,
                "breakrval": 4,
                "k": 2.5,
                "alpha": 1.0,
                "gamma": 0.072,
                "delta": 0.0,
                "x_s": None,
            },
            porosity_growth_kernel_name="cone_model",
            porosity_compression_kernel_name="porosity_compression_dynamik_rumpf",
            porosity_compression_kernel_params={
                "rate": _COMPRESSION_DYNAMIK_RUMPF_RATE,
                "min_porosity": 0.2,
                "n_mixer": 20.0,
                "k": 2.5,
                "alpha": 1.0,
                "gamma": 0.072,
                "delta": 0.0,
                "x_s": None,
            },
            liquid_internalization_kernel_name="liquid_internalization",
            liquid_internalization_kernel_params={"k_int": 1e11},
            liq_internalisation_agglomeration_kernel_name=(
                "liq_internalisation_agglomeration"
            ),
            solver_attrs={
                "agg_propensity_mode": "moment",
                "recon_enable": False,
                "agg_dW_min": 1.0,
                "agg_dW_max": 20.0,
                "break_dW_max": 50.0,
                "SIZEEVAL": 0,
            },
            init_particles={
                "particle_diameter": 34e-6,
                "initial_porosity": 0.0,
                "initial_particles": 1000,
                "initial_weight": 600.0,
                "control_volume": 1.0,
            },
            granulation=True,
            nucleation_params=dict(
                enabled=True,
                volumetric_flow_rate=1.635e-10,
                droplet_diameter=20e-6,
                liquid_addition_start=0.0,
                liquid_addition_duration=20.0,
                batch_size=20,
            ),
            continuous_params=dict(
                enabled=True,
                k_int=1e11,
                compression_enabled=True,
                compression_rate=0.1,
                min_porosity=0.2,
            ),
            notes=(
                "granulation_rumpf_dynamic_1d with the mixer-speed + "
                "Rumpf-strength compression kernel "
                "(k = rate * n_mixer**C_BREAK / sigma(eps, S))."
            ),
        ),
    )
}


def make_solver(name: str) -> MCPBESolver:
    return build(SCENARIOS[name])


# Scenarios used to force Numba compilation of every JIT path before timing.
# Without this the first measured run pays several hundred ms of LLVM compile
# time and the numbers are meaningless.
_WARMUP = ("agg_shear_1d", "agg_constant_1d", "agg_sum_1d", "break_powerlaw_1d", "mix_1d")


def warmup() -> None:
    """Run each JIT-backed code path once on a tiny problem."""
    for name in _WARMUP:
        scenario = SCENARIOS[name]
        tiny = Scenario(
            **{
                **scenario.__dict__,
                "name": f"{scenario.name}__warmup",
                "a0": 24,
                "maxiter": 3,
            }
        )
        try:
            build(tiny).solve(maxiter=tiny.maxiter)
        except Exception:  # pragma: no cover - warmup must never break a run
            pass
