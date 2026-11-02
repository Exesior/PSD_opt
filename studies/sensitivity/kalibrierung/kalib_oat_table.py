"""
Tabelle der Einzeltests (one-at-a-time) gegen den Basisfall n125.

    python kalib_oat_table.py            -> Markdown-Tabelle auf der Konsole + runs/oat_tabelle.md

Je Variante: Laufzeit (Mittel/Max), Timeouts, Events/s, Diskretisierungs-Verhaeltnis,
Bruch-Events, d50 @600 s, Verschiebung gegen n125 [Dekaden] und deren Streuung ueber die Seeds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from kalib_summary import RUNS, load_variant
from sa_core import PROBE_COLUMNS
from sa_psd import log_shift_decades, q3_curve_quantile

PC = {c: i for i, c in enumerate(PROBE_COLUMNS)}


def row(name: str, ref_q3):
    reps = load_variant(name)
    if not reps:
        return None
    ok = [r for r in reps if r["meta"]["status"] == "ok"]
    walls = [r["meta"]["wall_s"] for r in reps]
    out = {"variant": name, "n": len(reps), "n_ok": len(ok),
           "n_timeout": sum(r["meta"]["status"] == "timeout" for r in reps),
           "wall_mean": float(np.mean(walls)), "wall_max": float(np.max(walls))}
    if not ok:
        return out
    t_end = ok[0]["meta"]["t_end"]
    ev = np.array([r["events_per_bin"].sum() / t_end for r in ok])
    k_comp = ok[0]["meta"]["k_comp"]
    lam = []
    for r in ok:
        pr = r["probe"]
        cols = [str(c) for c in r.get("probe_columns", PROBE_COLUMNS)]
        lm = pr[:, cols.index("lam_int_mean")]
        lam.append(np.mean(lm[np.isfinite(lm)]) if np.isfinite(lm).any() else 0.0)
    ratio = ev / np.maximum(k_comp, np.array(lam))
    q3 = np.array([r["q3_grid"][-1] for r in ok])
    shifts = [log_shift_decades(ref_q3, c)[0] for c in q3]
    out.update({
        "ev_per_s": float(ev.mean()), "disc_ratio_min": float(ratio.min()),
        "breaks": float(np.mean([r["break_per_bin"].sum() for r in ok])),
        "n_comp_end": float(np.mean([r["n_comp_snap"][-1] for r in ok])),
        "d50": q3_curve_quantile(q3.mean(0), 0.5),
        "shift": log_shift_decades(ref_q3, q3.mean(0))[0],
        "shift_sd": float(np.std(shifts, ddof=1)) if len(shifts) > 1 else 0.0,
    })
    return out


def main(prefixes):
    base = [r for r in load_variant("n125") if r["meta"]["status"] == "ok"]
    ref_q3 = np.mean([r["q3_grid"][-1] for r in base], axis=0)
    names = sorted(d.name for d in RUNS.iterdir() if d.is_dir() and d.name.startswith(tuple(prefixes)))
    lines = ["| Variante | Laeufe ok/TO | Laufzeit Mittel/Max [s] | Events/s | Diskr.-Verh. min | Bruch-Ev. | n_comp Ende | d50 @600 [um] | Verschiebung [Dek.] (+ = groeber) | Streuung je Seed |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for n in ["n125"] + names:
        r = row(n, ref_q3)
        if r is None:
            continue
        if "d50" not in r:
            lines.append(f"| {n} | {r['n_ok']}/{r['n_timeout']} | {r['wall_mean']:.0f}/{r['wall_max']:.0f} | - | - | - | - | - | - | - |")
            continue
        lines.append(f"| {n} | {r['n_ok']}/{r['n_timeout']} | {r['wall_mean']:.0f}/{r['wall_max']:.0f} | "
                     f"{r['ev_per_s']:.1f} | {r['disc_ratio_min']:.1f} | {r['breaks']:.0f} | {r['n_comp_end']:.0f} | "
                     f"{r['d50']:.0f} | {r['shift']:+.3f} | {r['shift_sd']:.3f} |")
    text = "\n".join(lines)
    print(text)
    (RUNS / "oat_tabelle.md").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or ["o_", "c_"])
