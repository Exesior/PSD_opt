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

    for key, value in scenario.solver_attrs.items():
        setattr(solver, key, value)

    solver._initialize_particles()
    solver._init_lmc()
    solver._initialize_samplers()

    if scenario.granulation:
        _attach_granulation_handlers(solver)

    return solver


def _attach_granulation_handlers(solver: MCPBESolver) -> None:
    """Attach nucleation + continuous processes (liquid uptake and porosity).

    Flow rate is deliberately small: nucleation events are far cheaper than
    agglomeration events, and an aggressive flow rate would swamp the benchmark
    with nucleation bookkeeping instead of exercising the agglomeration path.
    """
    solver.create_nucleation_handler(
        enabled=True,
        volumetric_flow_rate=5e-19,
        droplet_diameter=1e-7,
        liquid_addition_duration=2.0,
        liquid_addition_start=0.0,
        batch_size=25.0,
    )
    solver.create_continuous_processes_handler(
        enabled=True,
        k_int=1e12,
        compression_enabled=True,
        compression_rate=0.02,
        min_porosity=0.3,
    )


# --- shared kernel parameter sets ------------------------------------------
SHEAR = {"corr_beta": 1e-3, "g": 1000.0}
# breakrval=1 -> constant single-particle breakage rate S = p1. Small p1 keeps
# the event rate in a range where the run stays inside its event budget without
# the time step collapsing. Volume-dependent variants (breakrval=4/5) are
# covered by the unit tests, not by the benchmark.
POWER_LAW = {"p1": 1e-2, "p2": 1.0, "g": 1000.0, "breakrval": 1, "pl_v": 2.0, "pl_q": 1.0}
# BREAKFVAL selects the fragment-count model in _compute_frag_num(); pin it so
# scenarios do not silently inherit the BaseSolver default.
BREAK_ATTRS = {"break_dW_max": 50.0, "BREAKFVAL": 2}


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
            name="agg_liquid_bridge_1d",
            dim=1,
            process_type="agglomeration",
            a0=150,
            t_end=120.0,
            maxiter=40,
            agg_kernel_name="liquid_bridge",
            agg_kernel_params={"corr_beta": 1e-3, "g": 1000.0},
            notes="Hits the non-JIT fallback: O(n^2) Python kernel calls per event.",
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
