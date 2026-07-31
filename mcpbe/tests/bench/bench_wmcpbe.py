"""Wall-clock benchmark for the WMCPBE solver.

    python -m tests.bench.bench_wmcpbe --repeats 5
    python -m tests.bench.bench_wmcpbe --only agg_shear_1d_large --scale 4
    python -m tests.bench.bench_wmcpbe --repeats 5 --out bench.json
    python -m tests.bench.bench_wmcpbe --compare before.json --out after.json

Each scenario is run ``--repeats`` times and the **minimum** is reported: the
minimum is the least noisy estimator of the underlying cost on a machine with
other work running. ``--scale`` multiplies every scenario's event budget, which
is useful for looking at behaviour closer to a production run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from statistics import median
from typing import Any, Dict, List

from .scenarios import SCENARIOS, build, warmup


def run_once(name: str, scale: float, mode: str | None = None) -> tuple[float, int, int]:
    scenario = SCENARIOS[name]
    if scale != 1.0:
        scenario = replace(scenario, maxiter=max(1, int(scenario.maxiter * scale)))
    solver = build(scenario)
    if mode is not None:
        solver.agg_propensity_mode = mode

    t0 = time.perf_counter()
    solver.solve(maxiter=scenario.maxiter)
    elapsed = time.perf_counter() - t0
    return elapsed, int(solver._iter_count), int(solver.a_tot)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--out", default=None, help="Write results as JSON")
    parser.add_argument("--compare", default=None, help="Baseline JSON to compare against")
    parser.add_argument(
        "--propensity-mode",
        default=None,
        choices=("pairwise", "moment"),
        help="Override MCPBEAgg.agg_propensity_mode for every scenario.",
    )
    args = parser.parse_args(argv)

    names = args.only or list(SCENARIOS)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        parser.error(f"unknown scenario(s): {unknown}")

    print("[bench] warming up JIT ...", flush=True)
    warmup()

    results: Dict[str, Any] = {}
    header = f"{'scenario':<24} {'best [s]':>9} {'median [s]':>11} {'events':>8} {'ms/event':>9}"
    print(f"\n[bench] {args.repeats} repeat(s), scale={args.scale}")
    print(header)
    print("-" * len(header))

    for name in names:
        timings = []
        events = a_tot = 0
        for _ in range(args.repeats):
            elapsed, events, a_tot = run_once(name, args.scale, args.propensity_mode)
            timings.append(elapsed)
        best = min(timings)
        per_event = (best / events * 1e3) if events else float("nan")
        results[name] = {
            "best": best,
            "median": median(timings),
            "events": events,
            "a_tot": a_tot,
            "ms_per_event": per_event,
        }
        print(f"{name:<24} {best:9.4f} {median(timings):11.4f} {events:8d} {per_event:9.4f}")

    total_best = sum(r["best"] for r in results.values())
    print("-" * len(header))
    print(f"{'TOTAL':<24} {total_best:9.4f}")

    if args.compare:
        with open(args.compare, "r", encoding="utf-8") as fh:
            base = json.load(fh)
        print(f"\n{'scenario':<24} {'before [s]':>11} {'after [s]':>10} {'speedup':>9}")
        print("-" * 58)
        total_before = 0.0
        total_after = 0.0
        for name, rec in results.items():
            if name not in base:
                continue
            before = base[name]["best"]
            after = rec["best"]
            total_before += before
            total_after += after
            print(f"{name:<24} {before:11.4f} {after:10.4f} {before / after:8.2f}x")
        print("-" * 58)
        if total_after > 0:
            print(
                f"{'TOTAL':<24} {total_before:11.4f} {total_after:10.4f} "
                f"{total_before / total_after:8.2f}x"
            )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\n[bench] wrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
