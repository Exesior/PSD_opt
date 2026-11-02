"""Start-Event-Rate R0 = 30 / t(30. Event) gegen f = Vc/V_mixer und N_comp (beste Parameter 1070).

W = f * N_real / N_comp wandert hier MIT (anders als im N_comp-Scan), damit
beide Regime sichtbar werden: W >= AGG_DW_MAX -> R ~ Vc ; W < AGG_DW_MAX -> R ~ N_comp.
Nur 30 Events je Punkt (Millisekunden bis 1 s). Ergebnis: results/rate_scaling.csv (repr-genau).

Aufruf (aus dem Ordner `mcpbe`):
    python -m studies.operator_splitting.rate_scaling
"""

from __future__ import annotations

import csv
import sys

from . import production_scenario as prod
from .diagnose_splitting import RESULTS_DIR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FRACTIONS = (1e-8, 1e-7, 1e-6, 1e-5)
N_COMPS = (100, 250, 500, 1000)
N_EVENTS = 30


def main() -> int:
    rows = []
    for f in FRACTIONS:
        for n in N_COMPS:
            pop = dict(prod.POP_BEST_1070)
            pop["INITIAL_PARTICLES"] = n
            pop["CONTROL_VOLUME"] = f * prod._V_MIXER_EXP
            pop["INITIAL_WEIGHT"] = f * prod._N_REAL / n
            pop["VOLUMETRIC_FLOW_RATE"] = f * prod._Q_BINDER_EXP
            s = prod.build_production(seed=prod.replicate_seed(0), pop=pop)
            s.solve(maxiter=N_EVENTS, max_particles=int(pop["MAX_PARTICLES"]))
            t = float(s._elapsed)
            r0 = N_EVENTS / t if t > 0 else float("nan")
            regime = "W>=dW_max (R~Vc)" if pop["INITIAL_WEIGHT"] >= pop["AGG_DW_MAX"] else "W<dW_max (R~N_comp)"
            rows.append({"f_Vc_durch_Vmixer": f, "N_comp": n, "Vc_m3": pop["CONTROL_VOLUME"],
                         "W_init": pop["INITIAL_WEIGHT"], "AGG_DW_MAX": float(pop["AGG_DW_MAX"]),
                         "Regime": regime, "N_Events": N_EVENTS, "t_nach_N_Events_s": t,
                         "R0_Events_pro_s": r0})
            print(f"f={f:.0e} N_comp={n:5d} W={pop['INITIAL_WEIGHT']:.3g} R0={r0:.4g} /s", flush=True)
    out = RESULTS_DIR / "rate_scaling.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"Geschrieben: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
