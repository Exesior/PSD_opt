"""N_comp-Scan mit den besten Parametern (Kandidat 1070): Splitting-Fehler gegen Event-Rate.

N_comp in (25, 50, 100, 500) x 3 Seeds (die ersten 3 Replikat-Seeds des Adapters).
W bleibt fest, Vc und Flussrate wandern mit N_comp (siehe production_scenario.
pop_best_1070_ncomp), dadurch skaliert die Event-Rate ~ N_comp.
Zusaetzlich: Config-Startwerte, Seed 0, erneut (theta-Schaetzer korrigiert).
Der kleinste Fall (N_comp 25, Seed 0) laeuft mit Bitgleichheits-Pruefung.

Aufruf (aus dem Ordner `mcpbe`):
    python -m studies.operator_splitting.scan_ncomp
"""

from __future__ import annotations

import sys
import time

from . import production_scenario as prod
from .diagnose_splitting import run_one

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

N_COMPS = (25, 50, 100, 500)   # klein zuerst: wenige Events, schnell
N_SEEDS = 3


def main() -> int:
    t0 = time.perf_counter()
    for n in N_COMPS:
        pop = prod.pop_best_1070_ncomp(n)
        name = prod.name_best_1070_ncomp(n)
        for k in range(N_SEEDS):
            seed = prod.replicate_seed(k)
            run_one(name, (lambda p=pop, s=seed: prod.build_production(seed=s, pop=p)),
                    int(pop["MAXITER"]), int(pop["MAX_PARTICLES"]), seed,
                    check_identical=(n == N_COMPS[0] and k == 0), pop=pop, quiet=True)
            print(f"  [scan] N_comp {n}, Seed {k + 1}/{N_SEEDS} fertig, "
                  f"{time.perf_counter() - t0:.0f} s gesamt", flush=True)

    seed = prod.replicate_seed(0)
    run_one(prod.NAME, lambda: prod.build_production(seed=seed, pop=prod.POP),
            int(prod.POP["MAXITER"]), int(prod.POP["MAX_PARTICLES"]), seed,
            pop=prod.POP, quiet=True)
    print(f"[scan] fertig, {time.perf_counter() - t0:.0f} s gesamt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
