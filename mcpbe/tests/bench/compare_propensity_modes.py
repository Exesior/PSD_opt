"""Compare the two aggregation-propensity evaluation modes.

``agg_propensity_mode`` selects how ``r_i = sum_j W_j beta(i,j)`` is evaluated:

``pairwise``  literal O(n^2) double loop - the numerical reference and default
``moment``    algebraically exact O(n) closed form

The two are mathematically identical but sum in a different order, so they
differ at the ULP level and Monte-Carlo trajectories eventually diverge. This
script quantifies both sides of that trade:

1. **Kernel accuracy** - direct ``r_i`` comparison on identical state.
2. **Kernel scaling**  - rebuild cost vs. particle count.
3. **Ensemble agreement** - moments from N seeded runs in each mode, compared
   against the Monte-Carlo spread. If the mode difference is well inside the
   seed-to-seed spread, the choice is statistically irrelevant.

    python -m tests.bench.compare_propensity_modes --seeds 8
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace

import numpy as np

from .scenarios import SCENARIOS, build, warmup

MODES = ("pairwise", "moment")


# ---------------------------------------------------------------------------
# 1 + 2: direct kernel comparison and scaling
# ---------------------------------------------------------------------------
def kernel_study(sizes=(250, 500, 1000, 2000, 4000), repeats: int = 5) -> None:
    from wmcpbe.kernels.aggregation.jit_kernels import (
        rebuild_r_array_shear,
        rebuild_r_array_shear_moment,
    )

    rng = np.random.default_rng(12345)
    print("\n[1/3] propensity rebuild: accuracy and scaling (shear kernel)")
    print(
        f"{'n':>6} {'pairwise [ms]':>14} {'moment [ms]':>12} {'speedup':>9} "
        f"{'max rel dev':>12} {'max ULP':>8}"
    )
    print("-" * 68)

    for n in sizes:
        R = rng.uniform(0.4e-6, 5e-6, n)
        W = rng.uniform(1.0, 80.0, n)
        a = np.zeros(n)
        b = np.zeros(n)

        rebuild_r_array_shear(1e-3, 1000.0, R, W, a)
        rebuild_r_array_shear_moment(1e-3, 1000.0, R, W, b)

        t0 = time.perf_counter()
        for _ in range(repeats):
            rebuild_r_array_shear(1e-3, 1000.0, R, W, a)
        t_pair = (time.perf_counter() - t0) / repeats

        t0 = time.perf_counter()
        for _ in range(repeats):
            rebuild_r_array_shear_moment(1e-3, 1000.0, R, W, b)
        t_mom = (time.perf_counter() - t0) / repeats

        denom = np.maximum(np.abs(a), 1e-300)
        rel = float(np.max(np.abs(a - b) / denom))
        ulp = float(np.max(np.abs(a - b) / np.maximum(np.spacing(np.abs(a)), 1e-300)))

        print(
            f"{n:6d} {t_pair*1e3:14.4f} {t_mom*1e3:12.4f} "
            f"{t_pair/t_mom:8.1f}x {rel:12.2e} {ulp:8.0f}"
        )


# ---------------------------------------------------------------------------
# 3: ensemble agreement
# ---------------------------------------------------------------------------
def _moments(solver) -> np.ndarray:
    """M0, M1, M2 of the final population (solid volume basis)."""
    a = solver.a_tot
    W = solver.W[:a]
    V = solver.V_flat[-1, :a]
    return np.array([np.sum(W), np.sum(W * V), np.sum(W * V * V)], dtype=float)


def ensemble_study(name: str, n_seeds: int) -> None:
    base = SCENARIOS[name]
    print(f"\n[3/3] ensemble agreement on {name!r} ({n_seeds} seeds per mode)")

    results = {}
    timings = {}
    for mode in MODES:
        moments = []
        t_total = 0.0
        for k in range(n_seeds):
            scenario = replace(base, seed=base.seed + 1000 * k)
            solver = build(scenario)
            solver.agg_propensity_mode = mode
            t0 = time.perf_counter()
            solver.solve(maxiter=scenario.maxiter)
            t_total += time.perf_counter() - t0
            moments.append(_moments(solver))
        results[mode] = np.asarray(moments)
        timings[mode] = t_total

    print(f"  wall clock: pairwise {timings['pairwise']:.3f}s, moment {timings['moment']:.3f}s "
          f"({timings['pairwise']/max(timings['moment'],1e-12):.2f}x)")
    print(f"  {'moment':<6} {'pairwise mean':>16} {'moment mean':>14} "
          f"{'| delta |':>11} {'MC std':>11} {'delta / std':>12}")
    print("  " + "-" * 76)

    labels = ("M0", "M1", "M2")
    for idx, label in enumerate(labels):
        a = results["pairwise"][:, idx]
        b = results["moment"][:, idx]
        pooled_std = float(np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))) if n_seeds > 1 else 0.0
        delta = abs(float(a.mean() - b.mean()))
        ratio = delta / pooled_std if pooled_std > 0 else float("nan")
        print(
            f"  {label:<6} {a.mean():16.6e} {b.mean():14.6e} "
            f"{delta:11.3e} {pooled_std:11.3e} {ratio:12.3f}"
        )
    print(
        "  Interpretation: |delta| / MC std well below 1 means the mode choice is\n"
        "  indistinguishable from ordinary Monte-Carlo seed noise."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="agg_shear_1d", choices=sorted(SCENARIOS))
    parser.add_argument("--seeds", type=int, default=8)
    args = parser.parse_args(argv)

    warmup()
    kernel_study()

    print("\n[2/3] end-to-end solver timing")
    for mode in MODES:
        scenario = SCENARIOS["agg_shear_1d_large"]
        solver = build(scenario)
        solver.agg_propensity_mode = mode
        t0 = time.perf_counter()
        solver.solve(maxiter=scenario.maxiter)
        dt = time.perf_counter() - t0
        print(
            f"  agg_shear_1d_large  mode={mode:<9} {dt:7.3f}s  "
            f"({dt / max(solver._iter_count, 1) * 1e3:.3f} ms/event, n~{solver.a_tot})"
        )

    ensemble_study(args.scenario, args.seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
