"""
Kalibrier-Laeufe fuer die Sensitivitaetsstudie (Laptop): Laufzeit, Events/s,
Raten der kontinuierlichen Prozesse und Verteilung fuer Varianten des Basisfalls.

Basisfall = KF3-Config (studies/optimization/opt_config_wet_granulation_kf3_nosampling_logw.py)
            + Bestparameter id 1070 aus 2100.sqlite.
Varianten = Beschleunigungen (Box, Tropfen, Dauer der Fluessigkeitszugabe) und
            Bruch-Tests. Jede Variante ist ein dict von Ueberschreibungen.

Aufruf (Konsole oder Spyder-Argumente unten):
    python kalib_run.py --variant orig --reps 4 --workers 4
Ergebnis: kalibrierung/runs/<variant>/p0000_r00.npz ...  + Zeile in runs/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import runpy
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPO = STUDY.parents[1]

from sa_core import replicate_seeds, run_replicate  # noqa: E402  (Ordner liegt per Aufruf-Pfad auf sys.path)

# Spyder (F5, keine Argumente) -> diese Argumente
SPYDER_ARGS = ["--variant", "orig", "--reps", "4", "--workers", "4"]

KF3_CONFIG = REPO / "studies" / "optimization" / "opt_config_wet_granulation_kf3_nosampling_logw.py"

# Bestparameter KF3, id 1070 (2100.sqlite, J = 0.028586)
ID1070 = {
    "AGG_A": 1.340086659222402,
    "EKE_CORR_BETA": 1.310079085422099e-15,
    "STOKES_U_COLL_REF": 0.016017645998242614,
    "BREAK_P1": 4196244189526212.5,
    "BREAK_P2": 1.6645619890520489,
    "COMPRESSION_RATE": 0.14080709816114967,
    "LIQUID_INTERNALIZATION_RATE": 14390824773.54635,
    "CONE_K_AGG": 0.8894554328862593,
    "CONE_K_BREAK": 0.37469962481864244,
}

# Experiment-Eckdaten (Vc-invariant, aus der KF3-Config)
V_MIXER_EXP = 1.0e-3
N_REAL = 1.5937e10
Q_BINDER_EXP = 1.7667e-6
DURATION_EXP = 60.0
PHI_BINDER = Q_BINDER_EXP * DURATION_EXP / V_MIXER_EXP      # 0.106 m^3 Binder je m^3 Mischer


def base_params() -> dict:
    cfg = runpy.run_path(str(KF3_CONFIG))
    params = dict(cfg["pop_params"])
    params.update(ID1070)
    params["N_MC_REPLICATES"] = 1
    return params


def scaled(params: dict, *, f: float | None = None, n_comp: int | None = None,
           droplet_d: float | None = None, n_droplets: float | None = None,
           liq_duration: float | None = None) -> dict:
    """Box/Tropfen/Zugabedauer konsistent umstellen (Gesamt-Binder je Volumen bleibt 0.106)."""
    p = dict(params)
    f = float(f if f is not None else p["CONTROL_VOLUME"] / V_MIXER_EXP)
    n_comp = int(n_comp if n_comp is not None else p["INITIAL_PARTICLES"])
    dur = float(liq_duration if liq_duration is not None else p["LIQUID_ADDITION_DURATION"])
    vc = f * V_MIXER_EXP
    p["CONTROL_VOLUME"] = vc
    p["INITIAL_PARTICLES"] = n_comp
    p["INITIAL_WEIGHT"] = f * N_REAL / n_comp
    p["LIQUID_ADDITION_DURATION"] = dur
    p["VOLUMETRIC_FLOW_RATE"] = PHI_BINDER * vc / dur
    if n_droplets is not None:
        v_drop = PHI_BINDER * vc / float(n_droplets)
        p["DROPLET_DIAMETER"] = float(np.cbrt(6.0 * v_drop / np.pi))
    elif droplet_d is not None:
        p["DROPLET_DIAMETER"] = float(droplet_d)
    return p


def n_droplets_in_box(p: dict) -> float:
    v_drop = np.pi / 6.0 * p["DROPLET_DIAMETER"] ** 3
    return p["VOLUMETRIC_FLOW_RATE"] * p["LIQUID_ADDITION_DURATION"] / v_drop


def variants() -> dict:
    b = base_params()
    v = {
        "orig": b,                                                    # wie der Fit (6 Tropfen)
        "drop50": scaled(b, n_droplets=50),                           # gleiche Box, kleinere Tropfen
        "drop200": scaled(b, n_droplets=200),
        "liq30": scaled(b, liq_duration=30.0),                        # Zugabe in 30 s statt 60 s
        "drop50_liq30": scaled(b, n_droplets=50, liq_duration=30.0),
        "drop200_liq30": scaled(b, n_droplets=200, liq_duration=30.0),
        # halbe Box, halbe Rechenpartikelzahl (W_init bleibt 3.19)
        "f05_drop50": scaled(b, f=0.5e-7, n_comp=250, n_droplets=50),
        "f05_drop200": scaled(b, f=0.5e-7, n_comp=250, n_droplets=200),
        "f05_drop50_liq30": scaled(b, f=0.5e-7, n_comp=250, n_droplets=50, liq_duration=30.0),
        # Tropfen 150 um bleiben (Granulatgroesse!), nur Rechenpartikel / Box aendern
        "n250": scaled(b, n_comp=250),                                # W_init 6.37
        "n250_liq30": scaled(b, n_comp=250, liq_duration=30.0),
        "f2": scaled(b, f=2e-7),                                      # 12 Tropfen, W_init 6.37
        "f4": scaled(b, f=4e-7),                                      # 24 Tropfen, W_init 12.7
        "f4_n250": scaled(b, f=4e-7, n_comp=250),                     # 24 Tropfen, W_init 25.5
        "f2_n250": scaled(b, f=2e-7, n_comp=250),                     # 12 Tropfen, W_init 12.7
        "f8_n250": scaled(b, f=8e-7, n_comp=250),                     # 48 Tropfen, W_init 51
        "n125": scaled(b, n_comp=125),                                # W_init 12.7
    }
    return v


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--first-rep", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--t-end", type=float, default=600.0)
    ap.add_argument("--time-limit", type=float, default=1500.0, help="Wandzeit-Limit je Replikat [s]")
    ap.add_argument("--set", nargs="*", default=[], help="KEY=WERT Ueberschreibungen (float)")
    ap.add_argument("--base", default=None, help="Ausgangsvariante, wenn --variant ein neuer Name ist")
    ap.add_argument("--sref", type=float, default=None,
                    help="log10 der Bruch-Referenzrate P1*V_ref^P2 (d_ref = 300 um); setzt BREAK_P1 nach --set")
    ap.add_argument("--aref", type=float, default=None,
                    help="log10 a_ref = log10(EKE_CORR_BETA * r_ref^(-1.5 AGG_A)), r_ref = 17 um; nach --set")
    ap.add_argument("--drop", type=float, default=None,
                    help="Tropfendurchmesser [um]; > 150 um -> Box (Vc, W_init, Flussrate) waechst mit (d/150)^3. "
                         "Anders als kind droplet_box waechst AGG_DW_MAX NICHT mit (dafuer --set AGG_DW_MAX=...)")
    args = ap.parse_args(argv)

    all_v = variants()
    src = args.base or args.variant
    if src not in all_v:
        raise SystemExit(f"unbekannte Variante {src!r}; bekannt: {sorted(all_v)}")
    params = dict(all_v[src])
    for item in args.set:
        key, val = item.split("=", 1)
        params[key] = float(val)
    if args.sref is not None:
        v_ref = np.pi / 6.0 * (300e-6) ** 3
        params["BREAK_P1"] = 10.0 ** args.sref / v_ref ** params["BREAK_P2"]
    if args.aref is not None:
        params["EKE_CORR_BETA"] = 10.0 ** args.aref * (17e-6) ** (1.5 * params["AGG_A"])
    if args.drop is not None:
        grow = max(1.0, (args.drop / 150.0) ** 3)
        for key in ("CONTROL_VOLUME", "INITIAL_WEIGHT", "VOLUMETRIC_FLOW_RATE"):
            params[key] = params[key] * grow
        params["DROPLET_DIAMETER"] = args.drop * 1e-6

    t_save = np.unique(np.concatenate([np.arange(0.0, args.t_end + 1e-9, 30.0), [360.0, args.t_end]]))
    t_save = t_save[t_save <= args.t_end]
    out_dir = HERE / "runs" / args.variant
    seeds = replicate_seeds(int(params["MC_SEED"]), args.first_rep + args.reps)
    tasks = []
    for r in range(args.first_rep, args.first_rep + args.reps):
        tasks.append({
            "params": params, "sa_values": {}, "point": 0, "rep": r, "seed": seeds[r],
            "t_save": t_save, "t_full": [t for t in (360.0, args.t_end) if t <= args.t_end],
            "bin_dt": 5.0, "probe_dt": 2.0, "time_limit_s": float(args.time_limit),
            "out_path": str(out_dir / f"p0000_r{r:02d}.npz"), "study": f"kalib_{args.variant}",
        })
    print(f"Variante {args.variant}: Vc={params['CONTROL_VOLUME']:.3g} m^3, N_comp={params['INITIAL_PARTICLES']}, "
          f"Tropfen {params['DROPLET_DIAMETER']*1e6:.1f} um -> {n_droplets_in_box(params):.1f} Tropfen in der Box, "
          f"Zugabe {params['LIQUID_ADDITION_DURATION']:g} s", flush=True)
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_replicate, t) for t in tasks]
        for fu in as_completed(futs):
            res = fu.result()
            rows.append(res)
            print(f"  Replikat {res['rep']}: {res['status']}, {res['wall_s']:.1f} s, "
                  f"{res['events']} Events, n_comp_end {res['n_comp_end']} {res['error']}", flush=True)
    total = time.perf_counter() - t0
    summ = HERE / "runs" / "summary.csv"
    new = not summ.exists()
    with open(summ, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["variant", "set", "reps", "workers", "wall_total_s", "wall_mean_s", "events_mean"])
        w.writerow([args.variant, " ".join(args.set), args.reps, args.workers, f"{total:.1f}",
                    f"{np.mean([r['wall_s'] for r in rows]):.1f}",
                    f"{np.mean([r['events'] for r in rows]):.0f}"])
    print(f"fertig in {total:.1f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else SPYDER_ARGS)
