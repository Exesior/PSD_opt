"""
Zusammenfassung der Kalibrier-Laeufe: eine Zeile je Variante.

    python kalib_summary.py [variante ...]      (ohne Argument: alle Ordner in runs/)

Spalten: Laufzeit, Events/s, gemessene Prozessraten, Diskretisierungs-Verhaeltnis,
Q3 an den Maschen bei 360/600 s (Mittel +- Streuung der Replikate), KF3-Kosten J,
d50, Verschiebung gegenueber 'orig' in Dekaden, Bruch-Events.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from sa_core import PROBE_COLUMNS, read_replicate
from sa_psd import GRID_UM, MESHES_UM, diameters_and_mass, log_shift_decades, q3_at, q3_curve_quantile

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"

# KF3-Ziel (studies/optimization/data/KF3_NoSampling_Q3_sieve_logw.xlsx)
EXP_360 = np.array([0.104914, 0.803040, 0.973487, 0.995511, 0.999042, 0.999581])   # AJ40
EXP_600 = np.array([0.033720, 0.423005, 0.921392, 0.983010, 0.997425, 0.999198])   # Mittel AJ26/29/31
MESH_W = np.array([0.282181, 0.227634, 0.164513, 0.145627, 0.129598, 0.050447])


def load_variant(name: str) -> list[dict]:
    files = sorted((RUNS / name).glob("p*_r*.npz"))
    return [read_replicate(f) for f in files]


def summarize(name: str, ref_q3_600=None) -> dict:
    reps = [r for r in load_variant(name) if r["meta"]["status"] == "ok"]
    if not reps:
        return {"variant": name, "n_ok": 0}
    meta = reps[0]["meta"]
    p = meta["params"]
    t_end = meta["t_end"]
    ci = {c: i for i, c in enumerate(PROBE_COLUMNS)}

    wall = np.array([r["meta"]["wall_s"] for r in reps])
    events = np.array([r["events_per_bin"].sum() for r in reps])
    ev_rate = events / t_end
    brk = np.array([r["break_per_bin"].sum() for r in reps])
    agg = np.array([r["agg_acc_per_bin"].sum() for r in reps])

    lam_mean_t, lam_all_t, lam_max = [], [], []
    for r in reps:
        pr = r["probe"]
        lm = pr[:, ci["lam_int_mean"]]
        frac = pr[:, ci["lam_int_massfrac"]]
        act = np.isfinite(lm)
        lam_mean_t.append(np.mean(lm[act]) if act.any() else 0.0)       # Mittel ueber aktive Zeit
        lam_all_t.append(np.mean(np.where(act, lm * frac, 0.0)))           # Masse aller Partikel, ganze Zeit
        lam_max.append(np.nanmax(pr[:, ci["lam_int_max"]]) if act.any() else 0.0)
    k_comp = meta["k_comp"]
    lam = float(np.mean(lam_mean_t))
    ratio = float(np.mean(ev_rate)) / max(k_comp, lam)

    def mesh_vals(r, t):
        d, m = diameters_and_mass(r[f"t{t}_V"], r[f"t{t}_W"])
        return q3_at(d, m, MESHES_UM)
    q360 = np.array([mesh_vals(r, 360) for r in reps])
    q600 = np.array([mesh_vals(r, int(t_end)) for r in reps]) if int(t_end) == 600 else None
    out = {
        "variant": name, "n_ok": len(reps),
        "Vc": p["CONTROL_VOLUME"], "N_comp": p["INITIAL_PARTICLES"],
        "drop_um": p["DROPLET_DIAMETER"] * 1e6,
        "n_drops": p["VOLUMETRIC_FLOW_RATE"] * p["LIQUID_ADDITION_DURATION"] / (np.pi / 6 * p["DROPLET_DIAMETER"] ** 3),
        "liq_s": p["LIQUID_ADDITION_DURATION"],
        "wall_s": wall.mean(), "wall_sd": wall.std(ddof=1) if wall.size > 1 else 0.0,
        "events": events.mean(), "ev_per_s": ev_rate.mean(), "k_comp": k_comp,
        "lam_int": lam, "lam_int_all": float(np.mean(lam_all_t)), "lam_int_max": float(np.max(lam_max)),
        "ratio": ratio, "agg_acc": agg.mean(), "breaks": brk.mean(),
        "q360": q360.mean(0), "q360_sd": q360.std(0, ddof=1) if len(reps) > 1 else q360[0] * 0,
    }
    if q600 is not None:
        out["q600"] = q600.mean(0)
        out["q600_sd"] = q600.std(0, ddof=1) if len(reps) > 1 else q600[0] * 0
        out["J"] = 0.5 * MESH_W @ np.abs(q360.mean(0) - EXP_360) + 0.5 * MESH_W @ np.abs(q600.mean(0) - EXP_600)
        j_rep = [0.5 * MESH_W @ np.abs(a - EXP_360) + 0.5 * MESH_W @ np.abs(b - EXP_600) for a, b in zip(q360, q600)]
        out["J_rep_sd"] = float(np.std(j_rep, ddof=1)) if len(j_rep) > 1 else 0.0
        q3g = np.mean([r["q3_grid"][-1] for r in reps], axis=0)
        out["d10"], out["d50"], out["d90"] = (q3_curve_quantile(q3g, q) for q in (0.1, 0.5, 0.9))
        out["q3g_600"] = q3g
        ld50 = [np.log10(q3_curve_quantile(r["q3_grid"][-1], 0.5)) for r in reps]
        out["ld50_rep_sd"] = float(np.std(ld50, ddof=1)) if len(reps) > 1 else 0.0
        ref = ref_q3_600 if ref_q3_600 is not None else q3g
        sh = [log_shift_decades(ref, r["q3_grid"][-1])[0] for r in reps]
        out["shift_rep_sd"] = float(np.std(sh, ddof=1)) if len(reps) > 1 else 0.0
        if ref_q3_600 is not None:
            out["shift"], out["shift_abs"] = log_shift_decades(ref_q3_600, q3g)
    return out


def main(names):
    if not names:
        names = sorted(d.name for d in RUNS.iterdir() if d.is_dir())
    ref = summarize("orig") if (RUNS / "orig").exists() else None
    ref_q = ref.get("q3g_600") if ref else None
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    for n in names:
        s = summarize(n, ref_q)
        if s.get("n_ok", 0) == 0:
            print(f"{n}: keine fertigen Replikate"); continue
        print(f"=== {n}  ({s['n_ok']} Repl.)  Vc={s['Vc']:.2g} N_comp={s['N_comp']} Tropfen {s['drop_um']:.0f} um "
              f"({s['n_drops']:.0f} St.) Zugabe {s['liq_s']:g} s")
        print(f"    Laufzeit {s['wall_s']:.0f} +- {s['wall_sd']:.0f} s | Events {s['events']:.0f} = {s['ev_per_s']:.1f}/s "
              f"| k_comp {s['k_comp']:.3g}/s, lam_int {s['lam_int']:.3g}/s (alle {s['lam_int_all']:.3g}, max {s['lam_int_max']:.3g}) "
              f"| Verhaeltnis {s['ratio']:.1f} {'OK' if s['ratio'] >= 2 else 'ZU KLEIN'}")
        print(f"    akzept. Agglo {s['agg_acc']:.0f}, Bruch-Events {s['breaks']:.0f}")
        print(f"    Q3@360 {s['q360']}  sd {s['q360_sd']}   exp {EXP_360}")
        if "q600" in s:
            print(f"    Q3@600 {s['q600']}  sd {s['q600_sd']}   exp {EXP_600}")
            extra = f" | Verschiebung zu orig {s['shift']:+.3f} dec (|.| {s['shift_abs']:.3f})" if "shift" in s else ""
            print(f"    J_KF3 {s['J']:.4f} (Repl.-sd {s['J_rep_sd']:.4f}) | d10/50/90 @600 "
                  f"{s['d10']:.0f}/{s['d50']:.0f}/{s['d90']:.0f} um{extra}")
            print(f"    Streuung je Replikat: log10 d50 {s['ld50_rep_sd']:.3f} dec, Verschiebung {s['shift_rep_sd']:.3f} dec "
                  f"-> SE bei 5 Seeds {s['ld50_rep_sd']/np.sqrt(5):.3f} / {s['shift_rep_sd']/np.sqrt(5):.3f} dec")


if __name__ == "__main__":
    main(sys.argv[1:])
