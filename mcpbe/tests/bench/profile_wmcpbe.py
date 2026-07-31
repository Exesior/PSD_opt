"""cProfile a single WMCPBE scenario and print the hottest call sites.

    python -m tests.bench.profile_wmcpbe agg_shear_1d_large --top 25
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys

from .scenarios import SCENARIOS, make_solver, warmup


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=sorted(SCENARIOS))
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--sort", default="tottime", choices=("tottime", "cumtime"))
    parser.add_argument("--out", default=None, help="Optional .prof output path")
    args = parser.parse_args(argv)

    warmup()
    scenario = SCENARIOS[args.scenario]
    solver = make_solver(args.scenario)

    profiler = cProfile.Profile()
    profiler.enable()
    solver.solve(maxiter=scenario.maxiter)
    profiler.disable()

    if args.out:
        profiler.dump_stats(args.out)

    print(f"\n=== {args.scenario}: {solver._iter_count} events, a_tot={solver.a_tot} ===")
    pstats.Stats(profiler).sort_stats(args.sort).print_stats(args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
