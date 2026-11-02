"""delta_Bruch (mit Vorzeichen, lineare Achse inkl. 0) ueber der Zeit und ueber der Event-Rate.

delta_break = (A_break nach - A_break vor) / A_break vor  des kontinuierlichen Schritts,
eine Zahl pro Iteration (Punkt). Negativ = der Schritt senkt die Bruchrate; das Event
bei t_n wurde also mit einer zu HOHEN Bruchrate gewuerfelt.

    Panel 1  delta_break ueber t, Seed 1 je N_comp
    Panel 2  delta_break ueber R = 1/dt, alle 3 Seeds je N_comp, plus Mittelwert je R-Klasse

Aufruf (aus dem Ordner `mcpbe`):
    python -m studies.operator_splitting.plot_break_delta [--out PNG]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from . import production_scenario as prod
from .plot_splitting import PALETTE, TEXT_2, GRID

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS = Path(__file__).resolve().parent / "results"
N_COMPS = (500, 100, 50, 25)   # grosse Punktwolke zuerst, kleine obenauf


def _runs(n: int):
    name = prod.name_best_1070_ncomp(n)
    out = []
    for k in range(3):
        f = RESULTS / f"{name}_n{int(prod.POP['MAXITER'])}_s{prod.replicate_seed(k)}" / "per_iteration.npz"
        if f.exists():
            d = np.load(f)
            out.append({key: d[key] for key in ("t", "dt", "delta_break")})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "delta_break_linear.png"))
    args = ap.parse_args(argv)

    colors = dict(zip(sorted(N_COMPS), PALETTE))   # gleiche Farben wie die anderen Plots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8.5))
    edges = np.logspace(-1.5, 2.0, 22)
    centers = np.sqrt(edges[:-1] * edges[1:])

    for n in N_COMPS:
        runs = _runs(n)
        if not runs:
            continue
        c = colors[n]
        r0 = runs[0]
        axes[0].plot(r0["t"], r0["delta_break"], ls="none", marker="o", ms=2.5,
                     color=c, alpha=0.5)
        R = np.concatenate([1.0 / r["dt"] for r in runs])
        db = np.concatenate([r["delta_break"] for r in runs])
        axes[1].plot(R, db, ls="none", marker="o", ms=2.5, color=c, alpha=0.35)
        idx = np.digitize(R, edges) - 1
        mean = np.array([np.nanmean(db[idx == k]) if np.sum(idx == k) >= 5 else np.nan
                         for k in range(centers.size)])
        axes[1].plot(centers, mean, color=c, lw=2)

    titles = ["delta_Bruch je Iteration ueber der Zeit (Seed 1)",
              "delta_Bruch je Iteration ueber der Event-Rate (3 Seeds), Linie = Mittelwert je R-Klasse"]
    xlabels = ["Simulationszeit t [s]", "Event-Rate R = 1/dt [Events/s]"]
    for ax, t, xl in zip(axes, titles, xlabels):
        ax.axhline(0.0, color=TEXT_2, lw=1)
        ax.set_title(t, loc="left", fontsize=11, color="#0b0b0b")
        ax.set_ylabel("delta_Bruch [-]", color=TEXT_2)
        ax.set_xlabel(xl, color=TEXT_2)
        ax.grid(True, color=GRID, lw=0.8)
        ax.tick_params(colors=TEXT_2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ymin = min(ax.get_ylim()[0], -0.05)
        ax.set_ylim(ymin * 1.05, -0.03 * ymin)   # 0 sichtbar, etwas Luft nach oben
    axes[0].axvline(60.0, color=TEXT_2, lw=1, ls=":")
    axes[1].set_xscale("log")

    # eigene Legenden-Symbole: die geplotteten Punkte selbst bleiben klein und transparent
    from matplotlib.lines import Line2D
    present = [n for n in sorted(N_COMPS) if _runs(n)]
    handles = [Line2D([], [], ls="none", marker="o", ms=7, color=colors[n]) for n in present]
    labs = [f"N_comp {n}" for n in present]
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(handles, labs, loc="lower center", bbox_to_anchor=(0.5, 0.955),
               ncol=len(labs), frameon=False)
    fig.savefig(args.out, dpi=150)
    print(f"Geschrieben: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
