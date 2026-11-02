"""
Grafiken der Sensitivitaetsauswertung (von analyze_sensitivity.py aufgerufen).

Farben: validierte Referenzpalette (dataviz-Skill, references/palette.md), feste
Reihenfolge, nie zyklisch. Messwerte nur als Marker (keine durchgezogenen Linien).
Alle Bilder als PNG nach <Ergebnisordner>/analysis/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sa_psd import GRID_UM, MESHES_UM

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRIDC, AXISC, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

LABELS = {
    "Q3_100_600": "Q3(100 um), 600 s", "Q3_250_600": "Q3(250 um), 600 s", "Q3_500_600": "Q3(500 um), 600 s",
    "Q3_800_600": "Q3(800 um), 600 s", "Q3_100_360": "Q3(100 um), 360 s", "Q3_250_360": "Q3(250 um), 360 s",
    "Q3_500_360": "Q3(500 um), 360 s", "Q3_800_360": "Q3(800 um), 360 s",
    "log10_d10_600": "log10 d10 [um], 600 s", "log10_d50_600": "log10 d50 [um], 600 s",
    "log10_d90_600": "log10 d90 [um], 600 s", "log10_d50_360": "log10 d50 [um], 360 s",
    "shift_600": "Verschiebung zur Basis [Dekaden], 600 s", "shift_abs_600": "Flaeche zur Basis [Dekaden], 600 s",
    "shift_360": "Verschiebung zur Basis [Dekaden], 360 s",
    "J_KF3": "KF3-Kosten J", "break_per_primary": "Bruch-Ereignisse je Primaerpartikel",
    "agg_per_primary": "Agglomerationen je Primaerpartikel", "n_comp_end": "Rechenpartikel am Ende",
    "ev_per_s": "MC-Events je Sim-Sekunde",
}


def _style():
    matplotlib.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": AXISC, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "font.size": 9, "axes.titlesize": 9.5, "axes.titlecolor": INK, "legend.frameon": False,
        "lines.linewidth": 2.0,
    })


def _label(y):
    return LABELS.get(y, y)


def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grafik: {path.name}")


def _grid(n, ncol):
    nrow = int(np.ceil(n / ncol))
    return nrow, ncol


# ---------------------------------------------------------------------------
def plot_sobol_bars(ana, sob_df, outputs, names):
    outputs = [y for y in outputs if y in set(sob_df.get("output", []))]
    if not outputs:
        return
    n = len(outputs)
    nrow, ncol = _grid(n, 3)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 0.42 * len(names) * nrow + 1.2 * nrow),
                             squeeze=False, sharex=True)
    ypos = np.arange(len(names))
    for ax, y in zip(axes.flat, outputs):
        g = sob_df[sob_df["output"] == y].set_index("parameter").reindex(names)
        ax.barh(ypos + 0.18, g["S1"], height=0.34, color="#86b6ef", label="S1 (Haupteffekt)",
                xerr=g["S1_conf"], ecolor=INK2, capsize=2)
        ax.barh(ypos - 0.18, g["ST"], height=0.34, color="#1c5cab", label="ST (gesamt)",
                xerr=g["ST_conf"], ecolor=INK2, capsize=2)
        ax.set_yticks(ypos, names)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.set_title(_label(y))
    for ax in axes.flat[n:]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, ana / "1_sobol_indizes.png")


def plot_heatmaps(ana, sob_df, outputs, names):
    if not len(sob_df) or "ST" not in sob_df.columns:
        return
    outputs = [y for y in outputs if y in set(sob_df["output"])]
    for col, fname, title, vmax in (
            ("ST", "2a_karte_ST.png", "Gesamteffekt ST (Anteil an der Streuung der Kennzahl)", 1.0),
            ("sigma_ST_per_exp_sd", "2b_karte_effekt_vs_versuch.png",
             "Zurechenbare Streuung sigma_ST in Vielfachen der Versuchs-Wiederholstreuung", None)):
        M = sob_df.pivot_table(index="parameter", columns="output", values=col).reindex(index=names, columns=outputs)
        M = M.loc[:, ~M.isna().all()]                 # z. B. Kennzahlen ohne Versuchs-Massstab
        if M.empty:
            continue
        outputs_m = list(M.columns)
        fig, ax = plt.subplots(figsize=(1.0 * len(outputs_m) + 3.0, 0.45 * len(names) + 2.2))
        vm = vmax if vmax is not None else float(np.nanpercentile(M.to_numpy(), 95)) or 1.0
        im = ax.imshow(M.to_numpy(dtype=float), cmap=SEQ, vmin=0, vmax=vm, aspect="auto")
        ax.set_xticks(range(len(outputs_m)), [_label(y) for y in outputs_m], rotation=40, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.grid(False)
        for i in range(len(names)):
            for j in range(len(outputs_m)):
                v = M.iat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}" if col == "ST" else f"{v:.1f}", ha="center", va="center",
                            fontsize=7.5, color=SURF if v > 0.6 * vm else INK)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        _save(fig, ana / fname)


def plot_main_effects(ana, cfg, names, X, mean, ab, base_id, yard):
    outs = [y for y in cfg["PLOT_MAIN_EFFECT_OUTPUTS"] if y in mean.columns]
    if not outs or len(ab) < 20:
        return
    specs = cfg["SA_PARAMETERS"]
    fig, axes = plt.subplots(len(outs), len(names), figsize=(2.3 * len(names), 2.0 * len(outs)),
                             squeeze=False, sharey="row")
    for r, y in enumerate(outs):
        yv = mean.loc[ab, y].to_numpy(dtype=float)
        base_v = float(mean.loc[base_id, y]) if base_id in mean.index else np.nan
        sd = yard.get(y)
        for c, n in enumerate(names):
            ax = axes[r, c]
            xv = X[ab, c]
            ax.scatter(xv, yv, s=6, color=MUTED, alpha=0.45, linewidths=0)
            lo, hi = specs[c]["bounds"]
            edges = np.linspace(lo, hi, 11)
            mid = 0.5 * (edges[1:] + edges[:-1])
            bm = [np.nanmean(yv[(xv >= a) & (xv < b)]) if np.any((xv >= a) & (xv < b)) else np.nan
                  for a, b in zip(edges[:-1], edges[1:])]
            ax.plot(mid, bm, color=CAT[0], marker="o", ms=4)
            if np.isfinite(base_v):
                ax.axhline(base_v, color=INK2, ls="--", lw=1)
                if sd:
                    ax.axhspan(base_v - sd, base_v + sd, color=CAT[1], alpha=0.12, lw=0)
            if r == len(outs) - 1:
                ax.set_xlabel(("log10 " if specs[c].get("log_scale") else "") + n)
            if c == 0:
                ax.set_ylabel(_label(y).replace(", ", "\n").replace(" [", "\n["), fontsize=8)
    fig.suptitle("Haupteffekte: Punkte (grau), Mittel je Zehntel des Bereichs (blau), Basis (gestrichelt), "
                 "+-1 Versuchs-Wiederholstreuung um die Basis (orange Band)", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, ana / "3_haupteffekte.png")


def plot_distributions(ana, cfg, names, X01, pooled, base_id, t_end):
    pts = [p for p in sorted(pooled) if p != base_id and t_end in pooled[p]]
    if len(pts) < 20 or base_id not in pooled:
        return
    curves = np.array([pooled[p][t_end] for p in pts])
    lo, hi = np.nanpercentile(curves, 5, axis=0), np.nanpercentile(curves, 95, axis=0)
    base = pooled[base_id][t_end]
    exp = np.asarray(cfg["EXP_600_REPEATS"], dtype=float)
    n = len(names)
    nrow, ncol = _grid(n + 1, 3)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.0 * nrow), squeeze=False, sharex=True, sharey=True)
    pidx = np.array(pts)
    for ax, (i, name) in zip(axes.flat, enumerate(names)):
        ax.fill_between(GRID_UM, lo, hi, color=GRIDC, lw=0, label="5-95 % aller Punkte")
        x = X01[pidx, i]
        low = curves[x <= np.quantile(x, 0.1)].mean(axis=0)
        high = curves[x >= np.quantile(x, 0.9)].mean(axis=0)
        ax.plot(GRID_UM, low, color=CAT[0], label="unterstes Zehntel des Parameters")
        ax.plot(GRID_UM, high, color=CAT[1], label="oberstes Zehntel des Parameters")
        ax.plot(GRID_UM, base, color=INK, lw=1.4, label="Basispunkt")
        for k, row in enumerate(exp):
            ax.plot(MESHES_UM, row, ls="none", marker="o", ms=4, mfc="none", mec=INK2,
                    label="Versuch AJ26/29/31 (600 s)" if k == 0 else None)
        ax.set_xscale("log")
        ax.set_title(name)
        ax.set_xlim(20, 3000)
        ax.set_ylim(0, 1.02)
    spare = list(axes.flat)[n:]
    for ax in spare:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    spare[0].legend(handles, labels, loc="center")
    for ax in list(axes.flat)[:n]:
        ax.tick_params(labelbottom=True)       # sharex blendet sonst die Beschriftung ueber leeren Zellen aus
        ax.set_xlabel("Durchmesser [um]")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Q3 bei {t_end} s")
    fig.tight_layout()
    _save(fig, ana / "4_verteilungen_je_parameter.png")


def plot_discretization(ana, cfg, names, pts_sa, flag_df):
    if "disc_ratio" not in pts_sa.columns:
        return
    fac = float(cfg["DISC_CRITERION_FACTOR"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    r = pts_sa["disc_ratio"].dropna()
    r = r[r > 0]
    if len(r):
        lo, hi = max(float(r.min()), 1e-2), float(r.max())
        lo, hi = min(lo, fac) / 1.5, max(hi, fac) * 1.5          # Linie und schmale Verteilungen sichtbar
        a1.hist(r, bins=np.logspace(np.log10(lo), np.log10(hi), 40), color="#6da7ec")
        a1.set_xscale("log")
    a1.axvline(fac, color=INK, ls="--", lw=1)
    a1.set_xlabel("Events/s / max(k_comp, lambda_int)  (Punktmittel)")
    a1.set_ylabel("Anzahl Punkte")
    a1.set_title(f"Diskretisierung: links der Linie (< {fac:g}) markiert")
    for k, n in enumerate(names[: len(CAT)]):
        g = flag_df[flag_df["parameter"] == n]
        a2.plot(g["decile"], g["share_flagged"] * 100, color=CAT[k], marker="o", ms=3.5, label=n)
    a2.set_xlabel("Zehntel des Parameterbereichs (1 = unten)")
    a2.set_ylabel("markierte Punkte [%]")
    a2.set_title("Wo die Bedingung verletzt wird")
    a2.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    _save(fig, ana / "5_diskretisierung.png")


def plot_noise(ana, cfg, noise_df):
    if not len(noise_df):
        return
    lim = float(cfg["NOISE_SHARE_LIMIT"])
    d = noise_df.dropna(subset=["noise_share"])
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(d) + 1.4))
    ax.barh(range(len(d)), d["noise_share"] * 100, color="#3987e5")
    ax.set_yticks(range(len(d)), [_label(y) for y in d["output"]])
    ax.invert_yaxis()
    ax.axvline(lim * 100, color=INK, ls="--", lw=1)
    ax.set_xlabel("Rauschanteil = mittleres SE^2 / Var(Kennzahl) [%]")
    ax.set_title(f"MC-Rauschen je Kennzahl (Linie = Grenze {lim:.0%})")
    fig.tight_layout()
    _save(fig, ana / "6_rauschen.png")


def make_all_plots(ana, cfg, names, X, X01, mean, se, pooled, base_id, t_end, sob_df, noise_df,
                   flag_df, ab, yard):
    _style()
    pts_sa = mean.loc[mean.index != base_id]
    if len(sob_df) and "ST" in sob_df.columns:
        plot_sobol_bars(ana, sob_df, cfg["PLOT_SOBOL_OUTPUTS"], names)
        plot_heatmaps(ana, sob_df, list(cfg["OUTPUTS"]), names)
    plot_main_effects(ana, cfg, names, X, mean, ab, base_id, yard)
    plot_distributions(ana, cfg, names, X01, pooled, base_id, t_end)
    plot_discretization(ana, cfg, names, pts_sa, flag_df)
    plot_noise(ana, cfg, noise_df)
