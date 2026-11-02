"""Splitting-Kennzahlen gegen die Event-Rate R = 1/dt (N_comp-Scan, beste Parameter).

Jede Iteration jedes Scan-Laufs liefert einen Punkt (R, Kennzahl). Die Iterationen
werden in logarithmische R-Klassen einsortiert; je N_comp (= ein Fall, alle 3 Seeds
zusammen) eine Farbe.

    Panel 1  theta_max: Maximum je R-Klasse (Linie) + Referenz k_eff / R (grau gestrichelt)
    Panel 2  |delta| Gesamtrate (durchgezogen) und Bruchrate (gestrichelt): MITTELWERT je R-Klasse
             (der Mittelwert ist, was sich als systematischer Fehler aufsummiert; das Maximum
             springt nur danach, ob gerade ein kompressibles Partikel existiert)
    Panel 3  Akzeptanz-Flips je 1000 Stokes-Tests je R-Klasse

Aufruf (aus dem Ordner `mcpbe`):
    python -m studies.operator_splitting.plot_rate_dependence [--out PNG]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from . import production_scenario as prod
from .plot_splitting import PALETTE, TEXT_2, GRID

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS = Path(__file__).resolve().parent / "results"
N_COMPS = (25, 50, 100, 500)
# k_eff der Kompression bei den besten Parametern: rate * n_mixer ** c_mixer
K_EFF = (prod.POP_BEST_1070["COMPRESSION_RATE"]
         * prod.POP_BEST_1070["N_MIXER"] ** prod.POP_BEST_1070["COMPRESSION_C_MIXER"])


def _collect(n: int):
    name = prod.name_best_1070_ncomp(n)
    cols = {}
    for folder in sorted(RESULTS.glob(f"{name}_n*")):
        if not (folder / "per_iteration.npz").exists():
            continue
        d = np.load(folder / "per_iteration.npz")
        for k in ("dt", "theta_max", "delta_rate", "delta_break", "acc_checked", "flip"):
            cols.setdefault(k, []).append(d[k])
    return {k: np.concatenate(v) for k, v in cols.items()} if cols else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "splitting_vs_event_rate.png"))
    ap.add_argument("--bins-per-decade", type=int, default=8)
    args = ap.parse_args(argv)

    data = {n: _collect(n) for n in N_COMPS}
    data = {n: c for n, c in data.items() if c is not None}
    r_all = np.concatenate([1.0 / c["dt"] for c in data.values()])
    lo, hi = np.floor(np.log10(r_all.min()) * 2) / 2, np.ceil(np.log10(r_all.max()) * 2) / 2
    edges = np.logspace(lo, hi, int((hi - lo) * args.bins_per_decade) + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])

    fig, axes = plt.subplots(3, 1, figsize=(10, 9.5), sharex=True)
    for (n, c), color in zip(data.items(), PALETTE):
        R = 1.0 / c["dt"]
        idx = np.digitize(R, edges) - 1
        th = np.full(centers.size, np.nan)
        dr = np.full(centers.size, np.nan)
        db = np.full(centers.size, np.nan)
        fl = np.full(centers.size, np.nan)
        for k in range(centers.size):
            m = idx == k
            if m.sum() < 5:
                continue
            th[k] = np.nanmax(c["theta_max"][m]) if np.any(c["theta_max"][m] > 0) else np.nan
            dr[k] = np.nanmean(np.abs(c["delta_rate"][m]))
            db[k] = np.nanmean(np.abs(c["delta_break"][m]))
            n_chk = c["acc_checked"][m].sum()
            fl[k] = 1000.0 * c["flip"][m].sum() / n_chk if n_chk > 0 else np.nan
        label = f"N_comp {n}"
        axes[0].plot(centers, th, color=color, lw=2, marker="o", ms=4, label=label)
        axes[1].plot(centers, dr, color=color, lw=2, ls="-", marker="o", ms=4)
        axes[1].plot(centers, db, color=color, lw=2, ls="--", marker="o", ms=4)
        axes[2].plot(centers, fl, color=color, lw=2, marker="o", ms=4)

    axes[0].plot(edges, K_EFF / edges, color=TEXT_2, lw=1, ls="--",
                 label=f"k_eff / R  (k_eff = {K_EFF:.3f} 1/s)")
    titles = ["theta = k_eff * dt: Maximum je Event-Rate-Klasse",
              "|delta|: Gesamtrate (durchgezogen), Bruchrate (gestrichelt), Mittelwert je Klasse",
              "Akzeptanz-Flips je 1000 Stokes-Tests"]
    ylabels = ["theta [-]", "|delta| [-]", "Flips / 1000 Tests"]
    for ax, t, yl in zip(axes, titles, ylabels):
        ax.set_title(t, loc="left", fontsize=11, color="#0b0b0b")
        ax.set_ylabel(yl, color=TEXT_2)
        ax.set_xscale("log")
        ax.grid(True, color=GRID, lw=0.8, which="both")
        ax.tick_params(colors=TEXT_2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[-1].set_xlabel("Event-Rate R = 1/dt [Events/s]", color=TEXT_2)

    handles, labs = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.legend(handles, labs, loc="lower center", bbox_to_anchor=(0.5, 0.945),
               ncol=min(5, len(labs)), frameon=False)
    fig.savefig(args.out, dpi=150)
    print(f"Geschrieben: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
