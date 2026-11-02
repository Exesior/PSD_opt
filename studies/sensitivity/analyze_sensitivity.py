"""
Sensitivitaetsstudie auswerten (Spyder: F5; Konsole: python analyze_sensitivity.py --config ...).

Liest alle Laeufe aus <Ergebnisordner>/runs/, rechnet je Lauf die Kennzahlen,
mittelt je Parameterpunkt ueber die Seeds und bestimmt fuer jede Kennzahl

  1. Sobol-Indizes (SALib, wie im Betreuer-Skript): S1 (Haupteffekt), ST (Gesamteffekt
     inkl. Wechselwirkungen), optional S2 (Paare), je mit 95-%-Bootstrap-Intervall.
  2. ABSOLUTE Groesse des Einflusses: zurechenbare Streuung sigma_ST = sqrt(ST * Var(Y))
     in den Einheiten der Kennzahl, verglichen mit
        - der Wiederholstreuung des Versuchs (AJ26/AJ29/AJ31 bei 600 s) und
        - dem MC-Rauschen des Punktmittels (Standardfehler ueber die Seeds).
  3. Rauschanteil = mittleres SE^2 / Var(Y). Ist er gross, sind die Indizes nicht
     belastbar -> mehr Replikate (Config N_REPLICATES hoch, fertige Laeufe bleiben gueltig).
  4. Markierung der Diskretisierungs-Bedingung (Events/s >= 2 x max(k_comp, lambda_int))
     je Punkt und wo im Parameterraum sie verletzt wird.

Fehlende Punkte (Timeout, Partikel-Limit, Absturz, oder nicht alle Seeds fertig):
der GANZE Saltelli-Block, in dem sie liegen, wird weggelassen (SALib rechnet mit
jedem N, solange die Blockstruktur stimmt). Auffuellen (KNN, Idee im Betreuer-Skript)
wird bewusst NICHT gemacht: fehlende Punkte haeufen sich dort, wo die Raten hoch sind,
und aufgefuellte Zeilen wuerden genau den verursachenden Parameter kleinrechnen.

Ausgabe in <Ergebnisordner>/analysis/: Excel + CSV, Grafiken (PNG), bericht.md.
Kann jederzeit (auch waehrend der Rechnung) gestartet werden: fehlen noch zu viele
Punkte, werden nur die Punkt-Kennzahlen und Markierungen geschrieben.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from run_sensitivity import HERE, config_fingerprint, load_config
from sa_core import BASE_ID, PROBE_COLUMNS, read_replicate, run_files
from sa_psd import (GRID_UM, MESHES_UM, diameters_and_mass, log_shift_decades, q3_at,
                    q3_curve_quantile, quantile_diameter)

SPYDER_ARGS = ["--config", "sa_config_gesamt.py"]

# Hoechstens dieser Anteil an Saltelli-Bloecken darf fehlen, sonst keine Sobol-Indizes.
MAX_DROPPED_BLOCK_SHARE = 0.10


# ===========================================================================
#  Kennzahlen eines Laufs
# ===========================================================================
def replicate_metrics(data: dict, criterion_factor: float) -> tuple[dict, dict]:
    """(Skalar-Kennzahlen, Kurven) eines Replikats. Kurven: Q3 auf GRID_UM bei 360 / Ende."""
    meta = data["meta"]
    p = meta["params"]
    t_end = float(meta["t_end"])
    m = {"point": meta["point"], "rep": meta["rep"], "seed": meta["seed"],
         "status": meta["status"], "wall_s": meta["wall_s"]}
    curves = {}
    if meta["status"] != "ok":
        return m, curves

    tend = int(round(t_end))
    for t in sorted({360, tend}):
        key = f"t{t}_V"
        if key not in data:
            continue
        d, mass = diameters_and_mass(data[key], data[f"t{t}_W"])
        q = q3_at(d, mass, MESHES_UM)
        for mesh, val in zip(MESHES_UM, q):
            m[f"Q3_{int(mesh)}_{t}"] = float(val)
        for qq in (0.1, 0.5, 0.9):
            m[f"log10_d{int(qq * 100)}_{t}"] = float(np.log10(quantile_diameter(d, mass, qq)))
    t_snap = data["t_snap"]
    for t in sorted({360, tend}):
        idx = np.flatnonzero(np.isclose(t_snap, t))
        if idx.size:
            curves[t] = data["q3_grid"][int(idx[0])]

    # Prozess-Zaehler (Events = MC-Schritte; "real" = physikalisch, auf die Anfangsbox bezogen)
    pc = [str(c) for c in data.get("probe_columns", PROBE_COLUMNS)]
    pr = data["probe"]
    col = {c: i for i, c in enumerate(pc)}
    ev = int(data["events_per_bin"].sum())
    w0 = float(p["INITIAL_PARTICLES"]) * float(p["INITIAL_WEIGHT"])        # reale Primaerpartikel in der Box
    if "real_agg_norm" in col:
        real_agg, real_break = float(pr[-1, col["real_agg_norm"]]), float(pr[-1, col["real_break_norm"]])
    else:                                                                   # altes Dateiformat
        real_agg, real_break = float(data["real_agg_snap"][-1]), float(data["real_break_snap"][-1])
    m["events"] = ev
    m["ev_per_s"] = ev / t_end
    m["break_events"] = int(data["break_per_bin"].sum())
    m["agg_events"] = int(data["agg_acc_per_bin"].sum())
    m["break_per_primary"] = real_break / w0
    m["agg_per_primary"] = real_agg / w0
    m["n_comp_end"] = int(data["n_comp_snap"][-1])
    m["vc_factor_end"] = float(data["Vc_snap"][-1] / data["Vc_snap"][0])

    # Diskretisierung: mittlere Events/s gegen die tatsaechlichen Prozessraten
    lam = pr[:, col["lam_int_mean"]]
    act = np.isfinite(lam)
    lam_mean = float(np.mean(lam[act])) if act.any() else 0.0
    k_comp = float(meta["k_comp"])
    m["k_comp"] = k_comp
    m["lam_int_mean"] = lam_mean
    m["lam_int_max"] = float(np.nanmax(pr[:, col["lam_int_max"]])) if act.any() else 0.0
    m["disc_ratio"] = m["ev_per_s"] / max(k_comp, lam_mean, 1e-300)
    m["disc_flag"] = bool(m["disc_ratio"] < criterion_factor)
    # Zusatzinfo (nicht das Kriterium): kleinstes Verhaeltnis in einem Zeitfenster,
    # Fenster = Bin des Event-Zaehlers, Rate = Interpolation der Messungen.
    bin_dt = float(meta["bin_dt"])
    ev_bin = data["events_per_bin"][: int(np.ceil(t_end / bin_dt))] / bin_dt
    t_mid = (np.arange(ev_bin.size) + 0.5) * bin_dt
    lam_t = np.interp(t_mid, pr[:, col["t"]], np.where(act, lam, 0.0))
    comp_on = np.interp(t_mid, pr[:, col["t"]], pr[:, col["comp_massfrac"]]) > 1e-3
    need = np.maximum(np.where(comp_on, k_comp, 0.0), lam_t)
    with np.errstate(divide="ignore", invalid="ignore"):
        loc = np.where(need > 0, ev_bin / need, np.inf)
    m["disc_ratio_local_min"] = float(np.min(loc)) if loc.size else np.inf
    return m, curves


# ===========================================================================
#  Versuchs-Massstab
# ===========================================================================
def experiment_yardsticks(cfg: dict) -> dict:
    """Wiederholstreuung des Versuchs (AJ26/AJ29/AJ31, 600 s) je Kennzahl, als Standardabweichung."""
    rows = np.asarray(cfg["EXP_600_REPEATS"], dtype=float)          # 3 x 6 (Maschen)
    ys = {}
    sd = rows.std(axis=0, ddof=1)
    for mesh, s in zip(MESHES_UM, sd):
        ys[f"Q3_{int(mesh)}_600"] = float(s)
        ys[f"Q3_{int(mesh)}_360"] = float(s)       # bei 360 s nur EIN Versuch -> 600-s-Streuung als Ersatz
    # Kurven der drei Versuche: Stuetzstelle (Startdurchmesser, Q3 = 0) + Maschen, linear in log x
    x_anchor = np.concatenate([[float(cfg["BASE_PARAMS"]["INITIAL_DIAMETER"]) * 1e6], MESHES_UM])
    curves = []
    for r in rows:
        y = np.concatenate([[0.0], r])
        curves.append(np.interp(np.log10(GRID_UM), np.log10(x_anchor), y, left=0.0, right=float(y[-1])))
    for qq in (10, 50, 90):
        vals = [np.log10(q3_curve_quantile(c, qq / 100)) for c in curves]
        ys[f"log10_d{qq}_600"] = ys[f"log10_d{qq}_360"] = float(np.std(vals, ddof=1))
    # Verschiebung: jede Versuchskurve gegen die mittlere Versuchskurve, SD (bzw. RMS) daraus
    mean_c = np.mean(curves, axis=0)
    s_signed = [log_shift_decades(mean_c, c)[0] for c in curves]
    s_abs = [log_shift_decades(mean_c, c)[1] for c in curves]
    n = len(curves)
    ys["shift_600"] = ys["shift_360"] = float(np.std(s_signed, ddof=1))
    ys["shift_abs_600"] = ys["shift_abs_360"] = float(np.sqrt(np.sum(np.square(s_abs)) / (n - 1)))
    return ys


# ===========================================================================
#  Punkte
# ===========================================================================
def collect(cfg: dict, out_dir: Path):
    files = run_files(out_dir / "runs")
    if not files:
        raise SystemExit(f"Keine Laeufe in {out_dir / 'runs'}")
    fac = float(cfg["DISC_CRITERION_FACTOR"])
    rows, curves = [], {}
    for i, f in enumerate(files):
        data = read_replicate(f)
        m, c = replicate_metrics(data, fac)
        rows.append(m)
        if c:
            curves[(m["point"], m["rep"])] = c
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(files)} Laeufe gelesen", flush=True)
    return pd.DataFrame(rows), curves


def point_table(rep_df: pd.DataFrame, curves: dict, base_id: int, t_end: int,
                exp_360: np.ndarray, exp_600: np.ndarray, mesh_w: np.ndarray):
    """Mittel und Standardfehler je Punkt + Kennzahlen aus der gemittelten Kurve."""
    ok = rep_df[rep_df["status"] == "ok"]
    num_cols = [c for c in ok.columns if c not in ("point", "rep", "seed", "status", "disc_flag")
                and pd.api.types.is_numeric_dtype(ok[c])]
    g = ok.groupby("point")
    n = g.size()
    mean = g[num_cols].mean()
    se = g[num_cols].std(ddof=1).div(np.sqrt(n), axis=0)
    mean["n_ok"] = n
    mean["disc_flag_any"] = g["disc_flag"].agg(lambda s: bool(np.any(s.astype(bool))))
    mean["disc_flag_share"] = g["disc_flag"].agg(lambda s: float(np.mean(s.astype(float))))

    # gemittelte Q3-Kurven je Punkt
    sums, cnt = {}, {}
    for (pt, _), c in curves.items():
        cnt[pt] = cnt.get(pt, 0) + 1
        for t, q in c.items():
            sums.setdefault(pt, {})[t] = sums.get(pt, {}).get(t, 0.0) + q
    pooled = {pt: {t: q / cnt[pt] for t, q in d.items()} for pt, d in sums.items()}
    if base_id not in pooled:
        raise SystemExit("Der Basispunkt hat noch keinen fertigen Lauf -> keine Referenz.")
    ref = pooled[base_id]

    # Verschiebung gegen die Basis: Punktmittel aus der gemittelten Kurve, SE aus den Replikaten
    times = [t for t in sorted({360, t_end}) if t in ref]
    for t in times:
        for kind, k in (("shift", 0), ("shift_abs", 1)):
            col = f"{kind}_{t}"
            mean[col] = pd.Series({pt: log_shift_decades(ref[t], pooled[pt][t])[k]
                                   for pt in mean.index if pt in pooled and t in pooled[pt]})
            per_rep = {}
            for (pt, _), c in curves.items():
                if t in c:
                    per_rep.setdefault(pt, []).append(log_shift_decades(ref[t], c[t])[k])
            se[col] = pd.Series({pt: (np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan)
                                 for pt, v in per_rep.items()})
    # KF3-Kosten des Punktmittels (wie im Fit: Mittel der Q3 ueber die Seeds)
    q360 = mean[[f"Q3_{int(x)}_360" for x in MESHES_UM]].to_numpy()
    q600 = mean[[f"Q3_{int(x)}_{t_end}" for x in MESHES_UM]].to_numpy()
    mean["J_KF3"] = 0.5 * np.abs(q360 - exp_360) @ mesh_w + 0.5 * np.abs(q600 - exp_600) @ mesh_w
    return mean, se, pooled


# ===========================================================================
#  Sobol
# ===========================================================================
def block_step(d: int, second: bool) -> int:
    return 2 * d + 2 if second else d + 2


def saltelli_ab_rows(n_rows: int, d: int, second: bool) -> np.ndarray:
    """Indizes der unabhaengigen A- und B-Zeilen der Saltelli-Matrix.

    Layout von SALib.sample.sobol: je Block [A, AB_1..AB_D, (BA_1..BA_D,) B].
    """
    step = block_step(d, second)
    k = np.arange(n_rows // step)
    return np.sort(np.concatenate([k * step, k * step + step - 1]))


def complete_blocks(valid: np.ndarray, d: int, second: bool) -> np.ndarray:
    """Bool je Block: alle Zeilen des Blocks vorhanden?"""
    step = block_step(d, second)
    return valid.reshape(-1, step).all(axis=1)


def run_sobol(problem: dict, Y: np.ndarray, second: bool, seed: int, resamples: int):
    from SALib.analyze import sobol
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sobol.analyze(problem, Y, calc_second_order=second, num_resamples=resamples,
                             conf_level=0.95, print_to_console=False, seed=seed)


# ===========================================================================
#  Hauptprogramm
# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="sa_config_gesamt.py")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--max-dropped", type=float, default=MAX_DROPPED_BLOCK_SHARE,
                    help="hoechster Anteil ausgefallener Saltelli-Bloecke, bis zu dem noch Sobol gerechnet wird")
    ap.add_argument("--seeds", type=int, default=None,
                    help="nur die ersten K Seeds je Punkt auswerten (Standard: N_REPLICATES der Config). "
                         "Nuetzlich fuer eine Zwischenauswertung, nachdem N_REPLICATES erhoeht wurde.")
    args = ap.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = HERE / cfg_path
    cfg = load_config(cfg_path)
    out_dir = Path(args.out) if args.out else Path(cfg["RESULT_DIR"])
    if not out_dir.is_absolute():
        out_dir = HERE / out_dir
    ana = out_dir / "analysis"
    ana.mkdir(parents=True, exist_ok=True)

    # Design + Parameterdefinition aus dem Ergebnisordner (nicht aus der evtl. geaenderten Config)
    with np.load(out_dir / "design.npz", allow_pickle=False) as z:
        X = z["X"]
        x_base = z["x_base"]
        names = [str(n) for n in z["names"]]
        bounds = z["bounds"]
        second = bool(z["calc_second_order"])
        fp_design = str(z["fingerprint"])
        sa_params = json.loads(str(z["sa_parameters_json"]))
    if fp_design != config_fingerprint(cfg):
        print("WARNUNG: Die Config wurde seit dem Rechnen geaendert. Ausgewertet wird das Design "
              "im Ergebnisordner; Kennzahl-Einstellungen kommen aus der aktuellen Config.")
    cfg = dict(cfg)
    cfg["SA_PARAMETERS"] = sa_params
    problem = {"num_vars": len(names), "names": names, "bounds": bounds.tolist()}
    n_pts = X.shape[0]
    base_id = BASE_ID
    n_rep_req = int(args.seeds) if args.seeds else int(cfg["N_REPLICATES"])
    cfg["N_REPLICATES"] = n_rep_req
    t_end = int(round(float(cfg["T_SAVE"][-1])))

    print(f"Lese Laeufe aus {out_dir / 'runs'} ...", flush=True)
    rep_df, curves = collect(cfg, out_dir)
    # Nur die Seeds der aktuellen Config (wurde N_REPLICATES verkleinert, liegen evtl. mehr Dateien da;
    # gemittelt wird an JEDEM Punkt ueber dieselben Seeds)
    keep = np.where(rep_df["point"] == BASE_ID, rep_df["rep"] < int(cfg["BASE_REPLICATES"]),
                    rep_df["rep"] < n_rep_req)
    rep_df = rep_df[keep].reset_index(drop=True)
    curves = {k: v for k, v in curves.items()
              if k[1] < (int(cfg["BASE_REPLICATES"]) if k[0] == BASE_ID else n_rep_req)}
    exp_360 = np.asarray(cfg["EXP_360"], dtype=float)
    exp_600 = np.asarray(cfg["EXP_600_MEAN"], dtype=float)
    mesh_w = np.asarray(cfg["MESH_WEIGHT"], dtype=float)
    mean, se, pooled = point_table(rep_df, curves, base_id, t_end, exp_360, exp_600, mesh_w)
    yard = experiment_yardsticks(cfg)

    # Punkt-Tabelle mit Parameterwerten (Zieh-Raum: log10 fuer log-Parameter)
    pts = pd.DataFrame(X, columns=[f"x_{n}" for n in names])
    pts.loc[base_id] = x_base
    pts = pts.join(mean, how="left")
    status_counts = rep_df.groupby(["point", "status"]).size().unstack(fill_value=0)
    pts = pts.join(status_counts.add_prefix("status_"), how="left")

    # Ein Punkt zaehlt nur, wenn ALLE geforderten Seeds ok sind (sonst andere Seed-Menge)
    n_ok = mean["n_ok"].reindex(range(n_pts)).fillna(0).to_numpy()
    valid = n_ok >= n_rep_req
    # ausgefallen = mindestens ein geforderter Seed hat eine Datei mit Status != ok
    bad_pts = rep_df.loc[(rep_df["status"] != "ok") & (rep_df["point"] < n_pts), "point"].unique()
    failed = np.isin(np.arange(n_pts), bad_pts)
    has_run = np.isin(np.arange(n_pts), rep_df["point"].to_numpy())
    step = block_step(len(names), second)
    blocks_ok = complete_blocks(valid, len(names), second)
    blocks_failed = failed.reshape(-1, step).any(axis=1)
    n_blocks = blocks_ok.size
    # Drei Zustaende je Block:
    #   fertig        alle Punkte mit allen Seeds ok          -> geht in Sobol ein
    #   ausgefallen   ein Seed eines Punkts nicht ok          -> weggelassen, zaehlt gegen die Grenze
    #   offen         noch nicht (fertig) gerechnet            -> weggelassen, zaehlt NICHT gegen die Grenze
    # Weil die Bloecke der Reihe nach gerechnet werden, ist das Ergebnis bei offenen Bloecken
    # einfach die Studie mit kleinerem N (das Saltelli-Design ist praefix-stabil).
    n_failed_blocks = int(blocks_failed.sum())
    n_open = int((~blocks_ok & ~blocks_failed).sum())
    rows_keep = np.repeat(blocks_ok, step)
    n_decided = int(blocks_ok.sum()) + n_failed_blocks
    dropped_share = n_failed_blocks / n_decided if n_decided else 1.0
    pts_with_runs = np.flatnonzero(has_run)
    # Zustand je Punkt fuer punkte.csv / Excel (offene Punkte stehen mit leeren Werten drin)
    state = np.where(failed, "ausgefallen", np.where(valid, "fertig", np.where(has_run, "in Arbeit", "offen")))
    pts["zustand"] = pd.Series(state, index=range(n_pts))
    pts.loc[base_id, "zustand"] = "Basis"
    print(f"Punkte komplett: {int(valid.sum())}/{n_pts}; Saltelli-Bloecke: {int(blocks_ok.sum())} fertig, "
          f"{n_failed_blocks} ausgefallen ({dropped_share:.1%}), {n_open} offen/in Arbeit"
          + (" -> ZWISCHENSTAND" if n_open else ""))

    outputs = [y for y in cfg["OUTPUTS"] if y in mean.columns]
    use_sobol = bool(blocks_ok.sum() >= 8 and dropped_share <= args.max_dropped)

    # --- Rauschanteil je Kennzahl -----------------------------------------
    # dieselben Punkte wie Sobol (behaltene Bloecke), sonst alle kompletten Punkte
    sel = list(np.flatnonzero(rows_keep)) if use_sobol else list(np.flatnonzero(valid))
    noise = []
    for y in outputs:
        v = mean.loc[sel, y].var(ddof=1) if sel else np.nan
        s2 = (se.loc[sel, y] ** 2).mean() if (y in se.columns and sel) else np.nan
        base_se = se.loc[base_id, y] if (y in se.columns and base_id in se.index) else np.nan
        noise.append({"output": y, "var_Y": v, "mean_SE2": s2,
                      "noise_share": s2 / v if v and v > 0 else np.nan,
                      "SE_base": base_se, "exp_sd": yard.get(y, np.nan)})
    noise_df = pd.DataFrame(noise)

    # --- Sobol -------------------------------------------------------------
    sob_rows, s2_rows = [], []
    if use_sobol:
        ab_all = saltelli_ab_rows(n_pts, len(names), second)
        for y in outputs:
            Y_all = mean[y].reindex(range(n_pts)).to_numpy(dtype=float)
            Y = Y_all[rows_keep]
            if not np.all(np.isfinite(Y)):
                sob_rows.append({"output": y, "parameter": "-", "note": "nicht endliche Werte"})
                continue
            # Varianz wie SALib: ueber die A- und B-Zeilen der behaltenen Bloecke (ddof 0)
            ab = ab_all[rows_keep[ab_all]]
            var_ab = float(np.var(Y_all[ab]))
            if not np.isfinite(var_ab) or var_ab <= 1e-30:
                sob_rows.append({"output": y, "parameter": "-", "note": "Kennzahl konstant -> keine Indizes"})
                continue
            Si = run_sobol(problem, Y, second, seed=int(cfg["SAMPLE_SEED"]),
                           resamples=int(cfg["BOOTSTRAP_RESAMPLES"]))
            nrow = noise_df.set_index("output").loc[y]
            for i, n in enumerate(names):
                st = float(Si["ST"][i])
                sigma_i = float(np.sqrt(max(st, 0.0) * var_ab))
                sob_rows.append({
                    "output": y, "parameter": n,
                    "S1": float(Si["S1"][i]), "S1_conf": float(Si["S1_conf"][i]),
                    "ST": st, "ST_conf": float(Si["ST_conf"][i]),
                    "sigma_ST": sigma_i,
                    "sigma_ST_per_exp_sd": sigma_i / yard[y] if yard.get(y) else np.nan,
                    "sigma_ST_per_MC_SE": (sigma_i / np.sqrt(nrow["mean_SE2"])
                                           if np.isfinite(nrow["mean_SE2"]) and nrow["mean_SE2"] > 0 else np.nan),
                    "blocks_used": int(blocks_ok.sum()),
                })
            if second:
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        s2_rows.append({"output": y, "pair": f"{names[i]} x {names[j]}",
                                        "S2": float(Si["S2"][i, j]), "S2_conf": float(Si["S2_conf"][i, j])})
    else:
        why = (f"erst {int(blocks_ok.sum())} fertige Bloecke (mindestens 8 noetig)" if blocks_ok.sum() < 8 else
               f"{dropped_share:.1%} der entschiedenen Bloecke ausgefallen (Grenze {args.max_dropped:.0%}, "
               f"aenderbar mit --max-dropped); wo: Blatt 'Diskretisierung', Spalte share_failed")
        print(f"Keine Sobol-Indizes: {why}.")
    sob_df = pd.DataFrame(sob_rows)
    s2_df = pd.DataFrame(s2_rows)

    # --- Haupteffekte (Spanne der Bin-Mittel ueber die unabhaengigen A/B-Zeilen) ---
    X01 = (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    ab = saltelli_ab_rows(n_pts, len(names), second)
    ab = ab[valid[ab]]
    eff_rows = []
    for y in outputs:
        if ab.size < 20:
            break
        yv = mean.loc[ab, y].to_numpy(dtype=float)
        for i, n in enumerate(names):
            bins = np.clip((X01[ab, i] * 10).astype(int), 0, 9)
            bm = np.array([np.nanmean(yv[bins == b]) if np.any(bins == b) else np.nan for b in range(10)])
            span = float(np.nanmax(bm) - np.nanmin(bm))
            eff_rows.append({"output": y, "parameter": n, "main_effect_span": span,
                             "span_per_exp_sd": span / yard[y] if yard.get(y) else np.nan})
    eff_df = pd.DataFrame(eff_rows)

    # --- Diskretisierung + Ausfaelle: wo im Parameterraum? (nur gerechnete Punkte) ---
    flag_rows = []
    pts_sa = pts.loc[pts.index < n_pts]
    flag = pts_sa["disc_flag_any"].astype(float)
    for i, n in enumerate(names):
        dec = np.clip((X01[:, i] * 10).astype(int), 0, 9)
        for b in range(10):
            sel = np.flatnonzero((dec == b) & has_run)
            flag_rows.append({"parameter": n, "decile": b + 1, "n_points": int(sel.size),
                              "share_flagged": float(flag.loc[sel].mean()) if sel.size else np.nan,
                              "share_failed": float(failed[sel].mean()) if sel.size else np.nan})
    flag_df = pd.DataFrame(flag_rows)

    # --- Schreiben -----------------------------------------------------------
    study = cfg["STUDY_NAME"]
    n_design = n_pts // step
    info = pd.DataFrame({"Schluessel": ["Studie", "Parameter", "N (Saltelli) im Design",
                                        "N tatsaechlich ausgewertet (= fertige Bloecke)", "2. Ordnung",
                                        "Punkte im Design (+ Basispunkt)", "Seeds je Punkt", "Laeufe gelesen",
                                        "Laeufe ok", "Punkte komplett", "Bloecke ausgefallen",
                                        "Bloecke offen/in Arbeit", "Punkte mit Diskretisierungs-Markierung",
                                        "Kriterium"],
                         "Wert": [study, ", ".join(names), n_design, int(blocks_ok.sum()), second,
                                  n_pts + 1, n_rep_req, len(rep_df), int((rep_df["status"] == "ok").sum()),
                                  int(valid.sum()), n_failed_blocks, n_open,
                                  f"{int(np.nansum(flag.to_numpy()))} von {int(has_run.sum())} gerechneten",
                                  f"Events/s >= {cfg['DISC_CRITERION_FACTOR']} x max(k_comp, lambda_int)"]})
    pts.to_csv(ana / "punkte.csv", index_label="punkt")
    rep_df.to_csv(ana / "laeufe.csv", index=False)
    if len(sob_df):
        sob_df.to_csv(ana / "sobol.csv", index=False)
    if len(s2_df):
        s2_df.to_csv(ana / "sobol_s2.csv", index=False)
    pids = sorted(pooled)
    np.savez_compressed(ana / "q3_kurven_punktmittel.npz", grid_um=GRID_UM, points=np.array(pids),
                        q3_end=np.array([pooled[p].get(t_end, np.full(GRID_UM.size, np.nan)) for p in pids]),
                        q3_360=np.array([pooled[p].get(360, np.full(GRID_UM.size, np.nan)) for p in pids]))
    write_report(ana, cfg, names, sob_df, noise_df, pts_sa.loc[has_run[pts_sa.index]], rep_df, valid,
                 blocks_ok, n_open, n_failed_blocks)

    if not args.no_plots:
        from sa_plots import make_all_plots
        make_all_plots(ana, cfg, names, X, X01, mean, se, pooled, base_id, t_end, sob_df, noise_df,
                       flag_df, ab, yard)

    # Excel zuletzt: ist die Datei gerade in Excel offen, gehen nur die Tabellen verloren
    xlsx = ana / f"sensitivitaet_{study}.xlsx"
    try:
        with pd.ExcelWriter(xlsx) as xw:
            info.to_excel(xw, sheet_name="Info", index=False)
            if len(sob_df):
                sob_df.to_excel(xw, sheet_name="Sobol_S1_ST", index=False)
            if len(s2_df):
                s2_df.to_excel(xw, sheet_name="Sobol_S2", index=False)
            noise_df.to_excel(xw, sheet_name="Rauschen", index=False)
            eff_df.to_excel(xw, sheet_name="Haupteffekte", index=False)
            flag_df.to_excel(xw, sheet_name="Diskretisierung", index=False)
            pts.to_excel(xw, sheet_name="Punkte", index_label="punkt")
            rep_df.to_excel(xw, sheet_name="Laeufe", index=False)
        print(f"Tabellen: {xlsx}")
    except PermissionError:
        print(f"WARNUNG: {xlsx.name} ist gesperrt (in Excel offen?) -> nicht geschrieben. "
              f"Die CSV-Dateien und Grafiken sind aktuell.")
    print(f"Fertig. Ergebnisse in {ana}")


def write_report(ana, cfg, names, sob_df, noise_df, pts_run, rep_df, valid, blocks_ok, n_open=0,
                 n_failed_blocks=0):
    """Kurzbericht in Klartext: Rangfolge je Kennzahl + Warnungen. pts_run = gerechnete Punkte."""
    L = [f"# Sensitivitaetsstudie {cfg['STUDY_NAME']}", ""]
    st = rep_df["status"].value_counts().to_dict()
    L.append(f"Laeufe: {len(rep_df)} ({', '.join(f'{k}: {v}' for k, v in st.items())}).")
    L.append(f"Punkte mit allen Seeds ok: {int(valid.sum())} von {valid.size}. Saltelli-Bloecke: "
             f"{int(blocks_ok.sum())} fertig (gehen ein), {n_failed_blocks} ausgefallen, {n_open} offen/in Arbeit, "
             f"von {blocks_ok.size}. Unvollstaendige Bloecke werden weggelassen, nicht aufgefuellt.")
    has_sobol = len(sob_df) and "ST" in sob_df.columns
    if n_open and has_sobol:
        L.append(f"**ZWISCHENSTAND:** {n_open} Bloecke sind noch nicht fertig. Die Indizes gelten fuer die "
                 f"{int(blocks_ok.sum())} fertigen Bloecke (= Studie mit N = {int(blocks_ok.sum())}).")
    elif not has_sobol:
        L.append("**Noch keine Sobol-Indizes** (zu wenige fertige Bloecke oder zu viele Ausfaelle, siehe Konsole).")
    flagged = pts_run["disc_flag_any"].astype(float)
    L.append(f"Diskretisierungs-Markierung (Events/s < {cfg['DISC_CRITERION_FACTOR']} x max(k_comp, lambda_int)): "
             f"{int(np.nansum(flagged))} von {len(pts_run)} gerechneten Punkten.")
    L.append("")
    if len(noise_df):
        bad = noise_df[noise_df["noise_share"] > float(cfg["NOISE_SHARE_LIMIT"])]
        if len(bad):
            L.append(f"**Rauschen gross** (Anteil > {cfg['NOISE_SHARE_LIMIT']:.0%}) bei: "
                     + ", ".join(f"{r.output} ({r.noise_share:.0%})" for r in bad.itertuples())
                     + " -> diese Indizes vorsichtig lesen bzw. N_REPLICATES erhoehen. (Der Rauschanteil "
                       "ist eine obere Schranke: mit gleichen Seeds an allen Punkten ist das Rauschen "
                       "zwischen Punkten korreliert und wirkt schwaecher auf die Indizes.)")
            L.append("")
    if len(sob_df) and "ST" in sob_df.columns:
        L.append("## Rangfolge je Kennzahl (nach ST)")
        L.append("sigma = zurechenbare Streuung der Kennzahl (Einheit der Kennzahl); "
                 "x Versuch = in Vielfachen der Wiederholstreuung des Versuchs.")
        for y, g in sob_df.dropna(subset=["ST"]).groupby("output", sort=False):
            g = g.sort_values("ST", ascending=False)
            parts = [f"{r.parameter}: ST {r.ST:.2f}+-{r.ST_conf:.2f}, S1 {r.S1:.2f}, "
                     f"sigma {r.sigma_ST:.3g} ({r.sigma_ST_per_exp_sd:.1f}x Versuch)"
                     for r in g.itertuples()]
            L.append(f"- **{y}**: " + " | ".join(parts))
    (ana / "bericht.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else SPYDER_ARGS)
