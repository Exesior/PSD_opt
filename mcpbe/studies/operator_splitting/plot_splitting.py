"""Plot der Splitting-Diagnose: Kennzahlen ueber der Zeit, mehrere Faelle ueberlagert.

Ein "Fall" = ein Ergebnisordner von diagnose_splitting.py (results/<...>/).
Jeder Fall bekommt eine Farbe; die Panels (Zeilen) sind die Kennzahlen:

    1  dt_n                         Zeit zwischen zwei Events
    2  theta_max = max k_eff * dt   wie weit die Kompression pro Intervall kommt
    3  |delta|  Agglomeration (durchgezogen) / Bruch (gestrichelt)
    4  Flips kumuliert (durchgezogen) / sqrt(N_akzeptiert) kumuliert (gestrichelt)

Panels 1-3 zeigen je Zeitfenster den Median (Linie) und das 10..90-%-Band.
Werte = 0 (z. B. theta, solange nichts poroes ist) fehlen auf der log-Achse.

Aufruf (aus dem Ordner `mcpbe`):
    python -m studies.operator_splitting.plot_splitting \
        results/<fall1> [results/<fall2> ...] [--labels A B] [--mark 60] [--bins 300]
Relative Pfade werden gegen den Ordner dieses Skripts aufgeloest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
# Kategoriale Reihenfolge der dataviz-Referenzpalette (validiert, feste Reihenfolge)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT_2 = "#52514e"
GRID = "#e4e3df"


def _load(folder: Path):
    d = np.load(folder / "per_iteration.npz")
    cols = {k: d[k] for k in d.files}
    sm = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    return cols, sm


def _binned(t, y, edges):
    """Median und 10/90-%-Quantil je Zeitfenster; nur positive, endliche Werte."""
    idx = np.digitize(t, edges) - 1
    n = len(edges) - 1
    med = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    ok = np.isfinite(y) & (y > 0)
    for k in range(n):
        m = ok & (idx == k)
        if np.any(m):
            lo[k], med[k], hi[k] = np.quantile(y[m], [0.1, 0.5, 0.9])
    return med, lo, hi


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--mark", type=float, nargs="*", default=[],
                    help="senkrechte Hilfslinien [s], z. B. Ende der Binderzugabe")
    ap.add_argument("--bins", type=int, default=300)
    ap.add_argument("--out", default=None, help="PNG-Pfad (Default: im ersten Ordner)")
    args = ap.parse_args(argv)

    folders = [(Path(f) if Path(f).is_absolute() else HERE / f) for f in args.folders]
    if len(folders) > len(PALETTE):
        raise SystemExit(f"Hoechstens {len(PALETTE)} Faelle (feste Palette).")
    labels = args.labels or [f.name for f in folders]
    if len(labels) != len(folders):
        raise SystemExit("--labels braucht genau einen Eintrag je Ordner.")

    data = [_load(f) for f in folders]
    t_max = max(float(c["t"][-1]) for c, _ in data)
    edges = np.linspace(0.0, t_max, args.bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    titles = [
        "Zeit zwischen zwei Events  dt",
        "Kompression pro Intervall  theta = max k_eff * dt",
        "Propensity-Drift |delta|:  Agglomeration (durchgezogen), Bruch (gestrichelt)",
        "Akzeptanz-Flips kumuliert (durchgezogen) vs. sqrt(N_akzeptiert) (gestrichelt)",
    ]
    ylabels = ["dt [s]", "theta [-]", "|delta| [-]", "Anzahl [-]"]

    for (cols, _sm), color, label in zip(data, PALETTE, labels):
        t = cols["t"]
        for ax, key, ls in ((axes[0], "dt", "-"), (axes[1], "theta_max", "-"),
                            (axes[2], "delta_agg", "-"), (axes[2], "delta_break", "--")):
            med, lo, hi = _binned(t, np.abs(cols[key]), edges)
            ax.plot(centers, med, color=color, lw=2, ls=ls,
                    label=label if (ax is axes[0]) else None)
            ax.fill_between(centers, lo, hi, color=color, alpha=0.15, lw=0)

        checked = cols["acc_checked"] == 1
        flips = np.cumsum(cols["flip"])
        n_acc = np.cumsum(checked & (cols["acc_old"] == 1))
        axes[3].plot(t, flips, color=color, lw=2, ls="-")
        axes[3].plot(t, np.sqrt(n_acc), color=color, lw=2, ls="--")

    for ax, title, yl in zip(axes, titles, ylabels):
        ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
        ax.set_ylabel(yl, color=TEXT_2)
        ax.grid(True, color=GRID, lw=0.8)
        ax.tick_params(colors=TEXT_2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    # log nur, wenn die Werte mehr als eine Dekade ueberspannen; sonst linear
    # (eine log-Achse ueber Faktor 1.03 bekommt nur unlesbare Zwischenticks)
    for ax in axes[:3]:
        ys = np.concatenate([ln.get_ydata() for ln in ax.get_lines()])
        ys = ys[np.isfinite(ys) & (ys > 0)]
        if ys.size and ys.max() / ys.min() > 10.0:
            ax.set_yscale("log")
    # Hilfslinien erst nach der Skalenwahl (ihre y-Daten 0..1 wuerden sie verfaelschen)
    for ax in axes:
        for x in args.mark:
            ax.axvline(x, color=TEXT_2, lw=1, ls=":")
    axes[-1].set_xlabel("Simulationszeit t [s]", color=TEXT_2)
    axes[-1].set_xlim(0.0, t_max)

    handles, labs = axes[0].get_legend_handles_labels()
    ncol = min(3, len(labs))
    top = 1.0 - 0.02 - 0.022 * int(np.ceil(len(labs) / ncol))   # Platz je Legendenzeile
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.legend(handles, labs, loc="lower center", bbox_to_anchor=(0.5, top + 0.005),
               ncol=ncol, frameon=False)

    out = Path(args.out) if args.out else folders[0] / "splitting_over_time.png"
    fig.savefig(out, dpi=150)
    print(f"Geschrieben: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
