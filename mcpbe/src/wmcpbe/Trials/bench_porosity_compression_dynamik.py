"""Runtime cost of the porosity-compression kernels: static vs. mixer-speed vs.
mixer-speed + Rumpf strength.

Two measurements:

1. ``compute_array`` in isolation, on a realistic mid-run population -- this is
   the per-Monte-Carlo-event cost that ``_apply_compression`` adds. Isolates
   what the Rumpf sigma(eps, S) evaluation costs on top of the bare
   exponential.

2. The full ``solve()`` wall time for the three otherwise-identical granulation
   scenarios, so the isolated number can be put in proportion to a real run
   (which is dominated by the O(n^2) EKE propensity rebuild).

Usage
-----
    python -m wmcpbe.Trials.bench_porosity_compression_dynamik
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # -> mcpbe/

from tests.bench.scenarios import SCENARIOS, build, warmup  # noqa: E402
from wmcpbe.kernels.continuous_processes import get_continuous_kernel  # noqa: E402


_SCEN = {
    "porosity_compression": "granulation_rumpf_dynamic_1d",
    "porosity_compression_dynamik": "granulation_compression_dynamik_1d",
    "porosity_compression_dynamik_rumpf": "granulation_compression_dynamik_rumpf_1d",
}


def _mid_run_population(events: int = 500):
    """A realistic (porosity, saturation, X0) snapshot from a granulation run."""
    solver = build(SCENARIOS["granulation_compression_dynamik_rumpf_1d"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver.solve(maxiter=events)
    a = solver.a_tot
    return solver, a


def bench_compute_array(reps: int = 2000) -> None:
    print("\n1. compute_array auf einer realistischen Population "
          "(Kosten pro MC-Ereignis)")
    solver, a = _mid_run_population()
    poro = solver.porosity[:a].copy()
    dt = 0.05
    n_porous = int(np.count_nonzero(~np.isnan(poro) & (poro > 0.2)))
    print(f"   a_tot = {a},  davon komprimierbar (eps > 0.2): {n_porous}")

    kernels = {
        "porosity_compression": get_continuous_kernel(
            "porosity_compression", rate=0.1, min_porosity=0.2),
        "porosity_compression_dynamik": get_continuous_kernel(
            "porosity_compression_dynamik", rate=0.0134, min_porosity=0.2, n_mixer=20.0),
        "porosity_compression_dynamik_rumpf": get_continuous_kernel(
            "porosity_compression_dynamik_rumpf", rate=106.7, min_porosity=0.2,
            n_mixer=20.0, k=2.5, alpha=1.0, gamma=0.072, delta=0.0),
    }

    # NOTE: on small arrays the numpy per-call overhead dominates and the first
    # kernel measured "pays" for allocator / cache warmup the next ones inherit,
    # so the static-vs-dynamik ratio here carries an ordering bias of ~20-30 %.
    # static and dynamik run identical array ops (k is folded once at
    # construction), so read them as equal; the dynamik_rumpf figure -- the
    # extra sigma(eps, S) array work -- is the real signal.
    base = None
    for name, kern in kernels.items():
        # warmup
        for _ in range(50):
            kern.compute_array(poro, dt, solver=solver)
        t0 = time.perf_counter()
        for _ in range(reps):
            kern.compute_array(poro, dt, solver=solver)
        el = (time.perf_counter() - t0) / reps
        if base is None:
            base = el
        print(f"   {name:38s} {el * 1e6:8.2f} us/Aufruf   "
              f"{el / base:5.2f}x static   "
              f"{el / max(n_porous, 1) * 1e9:6.1f} ns/komprimierbarem Partikel")

    # Isolate the sigma evaluation itself
    kr = kernels["porosity_compression_dynamik_rumpf"]
    sat = solver.saturation[:a][~np.isnan(poro) & (poro > 0.2)]
    e = poro[~np.isnan(poro) & (poro > 0.2)]
    x_s = kr._get_x_s(solver)
    for _ in range(50):
        kr._sigma(e, sat, x_s)
    t0 = time.perf_counter()
    for _ in range(reps):
        kr._sigma(e, sat, x_s)
    el = (time.perf_counter() - t0) / reps
    print(f"   -> davon nur sigma(eps,S):              {el * 1e6:8.2f} us/Aufruf")


def bench_full_solve(repeats: int = 3) -> None:
    print("\n2. Volle solve()-Wandzeit der drei Szenarien (identisch bis auf "
          "den Kompressionskernel)")
    warmup()
    for name, scen in _SCEN.items():
        sc = SCENARIOS[scen]
        best = float("inf")
        for _ in range(repeats):
            solver = build(sc)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t0 = time.perf_counter()
                solver.solve(maxiter=sc.maxiter)
                best = min(best, time.perf_counter() - t0)
        comp = solver.continuous_processes._particles_compressed_total
        print(f"   {name:38s} {best:7.2f} s   ({sc.maxiter} Ereignisse, "
              f"{comp} Partikel-Kompressionsschritte)")


if __name__ == "__main__":
    print("=" * 78)
    print("LAUFZEIT: Porositaets-Kompressionskernel  static / dynamik / dynamik_rumpf")
    print("=" * 78)
    bench_compute_array()
    bench_full_solve()
    print("\n" + "=" * 78)
