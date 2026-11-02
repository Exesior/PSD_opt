"""
Kernel-Vergleich mit den besten KF3-Parametern (Eric, 2026-10-13).

6 Tests = 3 Aggregationskernels x 2 Bruchkernels, je 10 MC-Replikate (die 10 festen
Fit-Seeds), volle Simulationszeit (t_vec der KF3-Config, 0..600 s).

    Aggregation   EKE    eke_darelius2005   (blend = 1)
                  ETM    etm_darelius2005   (blend = 2)
                  Blend  kinetic_blend      (blend = AGG_A, wie gefittet)
    Bruch         Rumpf     powerlaw_rumpf_dynamic   S = P1 * n^c * V^P2 / sigma(eps, S)   (wie gefittet)
                  Powerlaw  power_law                S = P1 * G   * V^P2                   (ohne Rumpf-Festigkeit)

Parameter
---------
Basis ist der beste Kandidat (kleinster Score) der neuesten SQLite im RESULT_DIR der
KF3-Config. Alle gefitteten Parameter bleiben gleich; nur die Kernel-Vorfaktoren, die es
im jeweiligen Kernel anders gibt, werden so umgerechnet, dass die Rate "ungefaehr passt":

* Aggregation: corr_beta von EKE/ETM so, dass der Aggregationsindex
      A = log10(corr_beta) - 1.5 * blend * log10(r0),   r0 = INITIAL_DIAMETER / 2
  dem gefitteten Blend-Wert entspricht (gleiche Kollisionsrate zweier Primaerpartikel):
      corr_beta(blend) = EKE_CORR_BETA_fit * r0 ** (1.5 * (blend - AGG_A_fit))
* Bruch: Powerlaw bekommt G = n_mixer ** BREAK_C_MIXER (wie der dynamische Kernel) und
      P1_powerlaw = BREAK_P1_fit / sigma_ref
  sigma_ref = Rumpf-Festigkeit eines "typischen" Granulats: W-gewichtetes harmonisches
  Mittel von sigma(eps, S) ueber alle poroesen Partikel der Rumpf-Laeufe DESSELBEN
  Aggregationskernels bei 360 s und 600 s (harmonisch, weil die Rate ~ 1/sigma ist).
  Deshalb laufen erst die drei Rumpf-Tests (Phase 1), dann die Powerlaw-Tests (Phase 2).
  Unterschied, der bleibt: Rumpf bricht poroese-freie Partikel (eps = 0 / NaN) gar nicht,
  Powerlaw bricht jedes Partikel.

Alles andere (Akzeptanz, Porositaet, Kompression, Fluessigkeit, Nucleation, Merger, ...)
ist exakt WetGranulationAdapter._build_solver: der Adapter baut den Solver, dieses Skript
tauscht nur Name/Parameter des Aggregations- und Bruchkernels im Konstruktor-Aufruf.

Neuer Rechner (einmalig)
------------------------
1. Python-Umgebung mit numpy, pandas, numba, scipy, scikit-learn, matplotlib, openpyxl, h5py
   (am einfachsten: Kopie der conda-Umgebung "Masterarbeitv4").
2. Die Repo-Pakete bekannt machen -- im Ordner PSD_opt einmal ausfuehren:
       python -c "import site,os; d=[p for p in site.getsitepackages() if p.endswith('site-packages')][0]; f=os.path.join(d,'psd_opt.pth'); open(f,'w').write(chr(10).join(os.path.abspath(x) for x in ['mcpbe/src','pbe-core/src','dpbe/src','pbe-optimizer/src'])); print(f)"
3. Kandidat: liegt als Kopie in studies/optimization/kernel_vergleich_input/ (siehe SPYDER_ARGS unten).

Start per "Play" (Spyder/IDE, ohne Argumente)
---------------------------------------------
Nimmt SPYDER_ARGS (unten) und rechnet erst "sim", dann "auswertung". Geht der
Parallelbetrieb in der IDE nicht, rechnet das Skript automatisch nacheinander weiter.

Aufruf (Terminal, aus dem Repo-Wurzelordner oder von ueberall)
--------------------------------------------------------------
    python studies/optimization/kernel_vergleich.py sim --workers 10
    python studies/optimization/kernel_vergleich.py auswertung
Optionen:
    --workers N        parallele Prozesse (Default: Anzahl CPUs, max. 30)
    --max-seconds S    Zeitlimit je Replikat (Default 7200 = 2 h); abgebrochene Replikate
                       werden als status=timeout gespeichert und in der Auswertung ausgelassen
    --sqlite PFAD      anderer Kandidaten-Lauf statt neuester SQLite der KF3-Config
    --smoke            Funktionstest: 60 statt 500 Rechenpartikel, 2 Replikate, eigener
                       Ausgabeordner *_smoke (physikalisch NICHT aussagekraeftig)
Bereits gerechnete Replikate (npz vorhanden) werden uebersprungen -> nach einem Abbruch
einfach denselben Befehl erneut starten.

Ausgabe: studies/optimization/kernel_vergleich_results/
    npz/<Test>_s<Seed>.npz           Rohdaten je Replikat (alle Snapshots)
    kernel_vergleich.xlsx            Blatt "Uebersicht" + je Test ein Blatt mit den Endergebnissen
                                     (600 s): Partikelliste je Replikat nebeneinander
                                     (Zeile = Rechenpartikel n_comp) und Dichteverteilung q3(x)
    Bruch_<Rumpf|Powerlaw>.png       3 Aggregationskernels bei festem Bruchkernel
    Agg_<EKE|ETM|Blend>.png          2 Bruchkernels bei festem Aggregationskernel
"""

from __future__ import annotations

import argparse
import json
import math
import os
import runpy
import sqlite3
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TRIALS = REPO / "studies" / "optimization"
CONFIG = TRIALS / "opt_config_wet_granulation_kf3_nosampling_logw.py"
OUT = TRIALS / "kernel_vergleich_results"
HELPER_KEYS = {"actor_wait", "wait_time", "max_reuse", "__exp_paths", "__known_params"}

AGG = {  # Kurzname -> (Kernel, blend; None = gefittetes AGG_A)
    "EKE": ("eke_darelius2005", 1.0),
    "ETM": ("etm_darelius2005", 2.0),
    "Blend": ("kinetic_blend", None),
}
BREAK = {  # Kurzname -> Kernel
    "Rumpf": "powerlaw_rumpf_dynamic",
    "Powerlaw": "power_law",
}
T_SHOW = (360.0, 600.0)          # 6 min (AJ40) und 10 min (Mittel AJ26/29/31)
X_EDGES_UM = np.geomspace(20.0, 3000.0, 41)    # Klassen der Dichteverteilung q3
X_GRID_UM = np.geomspace(20.0, 3000.0, 800)    # Raster fuer die Q3-Kurven


def test_name(agg: str, brk: str) -> str:
    return f"{agg}_{brk}"


# ===========================================================================
#  Kandidat + Adapter
# ===========================================================================
def best_candidate(sqlite_path: Path | None) -> dict:
    if sqlite_path is None:
        result_dir = Path(runpy.run_path(str(CONFIG))["RESULT_DIR"])
        files = [p for p in result_dir.glob("*.sqlite") if p.stem.isdigit()]
        if not files:
            raise FileNotFoundError(f"keine <N>.sqlite in {result_dir}")
        sqlite_path = max(files, key=lambda p: int(p.stem))
    con = sqlite3.connect(str(sqlite_path))
    try:
        cid, score, pj = con.execute(
            "SELECT id, score, param_json FROM warm_params ORDER BY score LIMIT 1").fetchone()
    finally:
        con.close()
    params = {k: v for k, v in json.loads(pj).items() if k not in HELPER_KEYS}
    return {"sqlite": str(sqlite_path), "id": int(cid), "score": float(score), "params": params}


def make_adapter(cand: dict, smoke: bool):
    from pbe_optimizer.adapter.wet_granulation_adapter import WetGranulationAdapter
    from pbe_optimizer.optimizer.opt_core import OptCore

    cfg = runpy.run_path(str(CONFIG))
    pop = dict(cfg["config"]["pop_params"])
    if smoke:  # gleiche Partikeldichte n = N * W / Vc, nur weniger Rechenpartikel
        factor = pop["INITIAL_PARTICLES"] / 60.0
        pop["INITIAL_PARTICLES"] = 60
        pop["INITIAL_WEIGHT"] = pop["INITIAL_WEIGHT"] * factor
    t_vec = np.asarray(cfg["config"]["algo_params"]["t_vec"], dtype=float)
    core = OptCore()
    core.t_vec = t_vec
    core.dim = 1
    core.max_iter_time = 0.0
    ad = WetGranulationAdapter(opt=core, dim=1, t_vec=t_vec, load_attr=False)
    for k, v in {**pop, **cand["params"]}.items():
        setattr(ad, k, v)
    data_file = Path(cfg["DATA_FILE"])
    ad.get_all_data(data_file)
    return ad, pop, t_vec, data_file


def kernel_settings(ad, agg: str, brk: str, sigma_ref: float | None) -> dict:
    """Name + Parameter der beiden getauschten Kernels (None = wie im Adapter)."""
    v = ad._value
    a_fit = float(v("AGG_A", 1.0))
    corr_fit = float(v("EKE_CORR_BETA", 5e-8))
    r0 = 0.5 * float(v("INITIAL_DIAMETER", 34e-6))
    n_mixer = float(v("N_MIXER", 20.0))
    out = {"A": math.log10(corr_fit) - 1.5 * a_fit * math.log10(r0), "r0": r0}

    kernel, blend = AGG[agg]
    if blend is None:
        out.update(agg_name=None, agg_params=None, blend=a_fit, corr_beta=corr_fit)
    else:
        corr = corr_fit * r0 ** (1.5 * (blend - a_fit))
        out.update(agg_name=kernel, blend=blend, corr_beta=corr,
                   agg_params={"corr_beta": corr, "n_mixer": n_mixer,
                               "c_mixer": float(v("EKE_C_MIXER", 0.0995))})

    p1_fit = float(v("BREAK_P1", 1e13))
    g = n_mixer ** float(v("BREAK_C_MIXER", 0.6699))
    if BREAK[brk] == "powerlaw_rumpf_dynamic":
        out.update(break_name=None, break_params=None, p1=p1_fit, g=g, sigma_ref=None)
    else:
        if sigma_ref is None:
            raise ValueError("Powerlaw braucht sigma_ref aus den Rumpf-Laeufen.")
        p1 = p1_fit / sigma_ref
        out.update(break_name=BREAK[brk], p1=p1, g=g, sigma_ref=sigma_ref,
                   break_params={"p1": p1, "p2": float(v("BREAK_P2", 1.0)), "g": g,
                                 "breakrval": int(v("BREAKRVAL", 4))})
    return out


@contextmanager
def swapped_kernels(settings: dict):
    """Tauscht im MCPBESolver-Aufruf des Adapters nur Aggregations-/Bruchkernel."""
    import pbe_optimizer.adapter.wet_granulation_adapter as wga
    original = wga.MCPBESolver

    def factory(*args, **kwargs):
        if settings["agg_name"] is not None:
            kwargs["agg_kernel_name"] = settings["agg_name"]
            kwargs["agg_kernel_params"] = settings["agg_params"]
        if settings["break_name"] is not None:
            kwargs["break_kernel_name"] = settings["break_name"]
            kwargs["break_kernel_params"] = settings["break_params"]
        return original(*args, **kwargs)

    wga.MCPBESolver = factory
    try:
        yield
    finally:
        wga.MCPBESolver = original


# ===========================================================================
#  Ein Replikat (laeuft im Worker-Prozess)
# ===========================================================================
def run_replicate(task: dict) -> dict:
    import warnings
    warnings.simplefilter("ignore")
    out_file = Path(task["out_file"])
    ad, pop, t_vec, _ = make_adapter(task["cand"], task["smoke"])
    settings = kernel_settings(ad, task["agg"], task["brk"], task["sigma_ref"])
    with swapped_kernels(settings):
        solver = ad._build_solver(t_vec, seed=task["seed"])

    flag = {"cancel": False}
    solver.cancel_flag = flag
    timer = threading.Timer(task["max_seconds"], flag.__setitem__, args=("cancel", True))
    timer.daemon = True
    t0 = time.perf_counter()
    timer.start()
    try:
        solver.solve(maxiter=int(pop["MAXITER"]), max_particles=int(pop["MAX_PARTICLES"]))
    finally:
        timer.cancel()
    seconds = time.perf_counter() - t0
    complete = len(solver.V_save) == t_vec.size
    status = "ok" if complete else ("timeout" if flag["cancel"] else "abgebrochen")
    if getattr(solver, "_hit_particle_limit", False):
        status = "max_particles"

    data = {"t_vec": t_vec, "seed": task["seed"], "status": status, "seconds": seconds,
            "settings": json.dumps({k: v for k, v in settings.items() if k not in ("agg_params", "break_params")}
                                   | {"agg_params": settings["agg_params"], "break_params": settings["break_params"]}),
            "agg_events": np.asarray(solver.real_agg_events_save, dtype=float),
            "break_events": np.asarray(solver.real_break_events_save, dtype=float)}
    for i in range(len(solver.V_save)):
        data[f"V_{i}"] = np.asarray(solver.V_save[i], dtype=float)          # [V_solid, ..., V_dry]
        data[f"W_{i}"] = np.asarray(solver.W_save[i], dtype=float)
        data[f"poro_{i}"] = np.asarray(solver.porosity_save[i], dtype=float)
        data[f"sat_{i}"] = np.asarray(solver.saturation_save[i], dtype=float)
        data[f"liq_{i}"] = np.asarray(solver.liquid_volume_save[i], dtype=float)

    # Rumpf: Festigkeit sigma der poroesen Partikel bei 360 s / 600 s (fuer sigma_ref)
    if complete and BREAK[task["brk"]] == "powerlaw_rumpf_dynamic":
        bk = solver.kernel_manager.break_kernel
        x_s = bk._get_x_s(solver)
        for t_show in T_SHOW:
            i = int(np.flatnonzero(np.isclose(t_vec, t_show))[0])
            poro, sat, w = data[f"poro_{i}"], data[f"sat_{i}"], data[f"W_{i}"]
            ok = np.isfinite(poro) & (poro > 0.0)
            sig = np.array([bk._compute_sigma(min(p, bk.poro_max), s, x_s) for p, s in zip(poro[ok], sat[ok])])
            data[f"sigma_{i}"] = sig
            data[f"sigma_w_{i}"] = w[ok]
        data["x_s"] = x_s
    solver._close()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(".tmp.npz")
    np.savez(tmp, **data)
    os.replace(tmp, out_file)
    return {"test": task["test"], "seed": task["seed"], "status": status, "seconds": seconds,
            "agg": float(data["agg_events"][-1]) if len(data["agg_events"]) else float("nan"),
            "break": float(data["break_events"][-1]) if len(data["break_events"]) else float("nan")}


# ===========================================================================
#  Simulation (Phase 1: Rumpf, Phase 2: Powerlaw)
# ===========================================================================
def sigma_ref_from_rumpf(npz_dir: Path, agg: str) -> float:
    num, den = 0.0, 0.0
    for f in sorted(npz_dir.glob(f"{test_name(agg, 'Rumpf')}_s*.npz")):
        z = np.load(f)
        if str(z["status"]) != "ok":
            continue
        t_vec = z["t_vec"]
        for t_show in T_SHOW:
            i = int(np.flatnonzero(np.isclose(t_vec, t_show))[0])
            if f"sigma_{i}" in z.files and len(z[f"sigma_{i}"]):
                w, s = z[f"sigma_w_{i}"], z[f"sigma_{i}"]
                num += float(w.sum())
                den += float((w / s).sum())
    if den <= 0.0:
        raise RuntimeError(f"{agg}: keine poroesen Partikel in den Rumpf-Laeufen -> sigma_ref unbestimmt.")
    return num / den


def simulate(args) -> None:
    out = OUT.with_name(OUT.name + "_smoke") if args.smoke else OUT
    npz_dir = out / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    cand = best_candidate(Path(args.sqlite) if args.sqlite else None)
    (out / "kandidat.json").write_text(json.dumps(cand, indent=2), encoding="utf-8")
    ad, _, _, _ = make_adapter(cand, args.smoke)
    seeds = ad._resolve_mc_seeds()[: (2 if args.smoke else len(ad._resolve_mc_seeds()))]
    print(f"Kandidat id {cand['id']} (Score {cand['score']:.5g}) aus {cand['sqlite']}", flush=True)
    print(f"{len(seeds)} Replikate je Test, {args.workers} Prozesse, Zeitlimit {args.max_seconds:g} s", flush=True)

    def run_phase(brk: str, sigma_refs: dict) -> None:
        tasks = []
        for agg in AGG:
            name = test_name(agg, brk)
            for s in seeds:
                f = npz_dir / f"{name}_s{s}.npz"
                if f.exists():
                    continue
                tasks.append({"test": name, "agg": agg, "brk": brk, "seed": int(s), "cand": cand,
                              "smoke": args.smoke, "max_seconds": args.max_seconds,
                              "sigma_ref": sigma_refs.get(agg), "out_file": str(f)})
        print(f"\n== Phase {brk}: {len(tasks)} offene Replikate", flush=True)
        if not tasks:
            return
        def report(r: dict) -> None:
            print(f"  {r['test']:15s} Seed {r['seed']:>10}: {r['status']:13s} {r['seconds']:7.0f} s  "
                  f"agg {r['agg']:.4g}  break {r['break']:.4g}", flush=True)

        if args.workers > 1:
            try:
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    futures = [pool.submit(run_replicate, t) for t in tasks]
                    for fut in as_completed(futures):
                        report(fut.result())
                return
            except Exception as exc:  # BrokenProcessPool, Pickling-/Import-Fehler im Kindprozess, ...
                # z. B. Spyder unter Windows ohne externes Terminal: Mehrprozess geht nicht ->
                # die noch fehlenden Replikate nacheinander rechnen (fertige npz bleiben erhalten).
                # Ein echter Fehler in der Simulation taucht dann unten erneut auf.
                print(f"  Parallelbetrieb nicht moeglich ({type(exc).__name__}: {exc}) "
                      f"-> rechne die restlichen Replikate nacheinander.", flush=True)
        for t in tasks:
            if not Path(t["out_file"]).exists():
                report(run_replicate(t))

    run_phase("Rumpf", {})
    sigma_refs = {agg: sigma_ref_from_rumpf(npz_dir, agg) for agg in AGG}
    for agg, s in sigma_refs.items():
        print(f"sigma_ref {agg}: {s:.4g} Pa", flush=True)
    (out / "sigma_ref.json").write_text(json.dumps(sigma_refs, indent=2), encoding="utf-8")
    run_phase("Powerlaw", sigma_refs)
    print(f"\nFertig. Auswertung:  python {Path(__file__).name} auswertung" + (" --smoke" if args.smoke else ""))


# ===========================================================================
#  Auswertung: Excel + Grafiken
# ===========================================================================
def load_test(npz_dir: Path, name: str) -> list:
    reps = []
    for f in sorted(npz_dir.glob(f"{name}_s*.npz")):
        z = np.load(f)
        if str(z["status"]) == "ok":
            reps.append(z)
    return reps


def snapshot(z, i: int) -> dict:
    v = z[f"V_{i}"]
    return {"d_um": np.cbrt(6.0 * v[-1] / np.pi) * 1e6, "V_solid": v[0], "V_dry": v[-1],
            "poro": z[f"poro_{i}"], "sat": z[f"sat_{i}"], "liq": z[f"liq_{i}"], "W": z[f"W_{i}"]}


def q3_curve(d_um: np.ndarray, mass: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Massenanteil mit Durchmesser < x (Sieb-Definition)."""
    o = np.argsort(d_um)
    cum = np.r_[0.0, np.cumsum(mass[o])] / mass.sum()
    return cum[np.searchsorted(d_um[o], x, side="left")]


def write_excel(path: Path, sheets: dict) -> None:
    import openpyxl.cell._writer as _xw
    original = _xw.safe_string
    _xw.safe_string = lambda x: repr(x) if isinstance(x, float) and math.isfinite(x) else original(x)
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            for name, frame in sheets.items():
                frame.to_excel(xw, sheet_name=name, index=False)
                xw.sheets[name].freeze_panes = "B2"
    finally:
        _xw.safe_string = original


def kf3_cost(ad, reps: list) -> float:
    """KF3-Kosten des Replikat-Mittels (wie im Adapter)."""
    q3 = []
    for z in reps:
        rows = []
        for i in range(len(z["t_vec"])):
            v = z[f"V_{i}"]
            rows.append(ad._q3_at_meshes_from_snapshot(v, z[f"W_{i}"], ad._mesh_m))
        q3.append(np.vstack(rows)[ad._obs_tidx])
    ad.data_mod = np.stack(q3).mean(axis=0)
    return ad.calc_delta_pop(None, ad._data_exp)


def kf3_static(data_file: Path) -> float:
    frame = pd.read_excel(data_file, sheet_name="Q3_sieve")
    mesh = [c for c in frame.columns if isinstance(c, (int, float))]
    q, w = frame[mesh].to_numpy(float), frame["Gewicht"].to_numpy(float)
    sheets = pd.ExcelFile(data_file).sheet_names
    m = (pd.read_excel(data_file, sheet_name="Maschengewicht")[mesh].to_numpy(float)[0]
         if "Maschengewicht" in sheets else np.full(len(mesh), 1.0 / len(mesh)))
    c = []
    for k in range(len(mesh)):
        o = np.argsort(q[:, k]); cw = np.cumsum(w[o])
        c.append(q[o, k][np.searchsorted(cw, 0.5 * cw[-1])])
    return float(np.dot(w, np.abs(np.asarray(c) - q) @ m))


def evaluate(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    out = OUT.with_name(OUT.name + "_smoke") if args.smoke else OUT
    npz_dir = out / "npz"
    cand = json.loads((out / "kandidat.json").read_text(encoding="utf-8"))
    ad, _, t_vec, data_file = make_adapter(cand, args.smoke)
    j_static = kf3_static(data_file)
    tests = {test_name(a, b): load_test(npz_dir, test_name(a, b)) for b in BREAK for a in AGG}
    sigma_refs = json.loads((out / "sigma_ref.json").read_text(encoding="utf-8")) \
        if (out / "sigma_ref.json").exists() else {}

    # ---------------- Excel
    overview, sheets = [], {}
    i_end = len(t_vec) - 1
    for name, reps in tests.items():
        agg, brk = name.split("_")
        n_all = len(list(npz_dir.glob(f"{name}_s*.npz")))
        st = kernel_settings(ad, agg, brk, sigma_refs.get(agg)) if (brk == "Rumpf" or agg in sigma_refs) else {}
        row = {"Test": name, "Aggregation": AGG[agg][0], "Bruch": BREAK[brk],
               "Replikate ok": len(reps), "Replikate gesamt": n_all,
               "blend": st.get("blend"), "corr_beta": st.get("corr_beta"), "A": st.get("A"),
               "Bruch P1": st.get("p1"), "Bruch G = n^c": st.get("g"), "sigma_ref [Pa]": st.get("sigma_ref")}
        if reps:
            row.update({
                "Rechenzeit Mittel [s]": float(np.mean([float(z["seconds"]) for z in reps])),
                "Aggregationen Mittel": float(np.mean([z["agg_events"][-1] for z in reps])),
                "Brueche Mittel": float(np.mean([z["break_events"][-1] for z in reps])),
                "n_comp Ende Mittel": float(np.mean([len(z[f"W_{i_end}"]) for z in reps])),
                "J (KF3, Mittel der Replikate)": kf3_cost(ad, reps),
            })
            row["S = 1 - J/J_static"] = 1.0 - row["J (KF3, Mittel der Replikate)"] / j_static
        overview.append(row)
        if not reps:
            continue

        # Partikelliste 600 s, Replikate nebeneinander, Zeile = Rechenpartikel
        n_max = max(len(z[f"W_{i_end}"]) for z in reps)
        block = {"n_comp (Partikel-Nr.)": np.arange(1, n_max + 1)}
        for r, z in enumerate(reps, start=1):
            s = snapshot(z, i_end)
            pad = lambda a: np.r_[a, np.full(n_max - len(a), np.nan)]
            block[f"R{r} d_aussen [um]"] = pad(s["d_um"])
            block[f"R{r} V_solid [m3]"] = pad(s["V_solid"])
            block[f"R{r} V_dry [m3]"] = pad(s["V_dry"])
            block[f"R{r} Porositaet"] = pad(s["poro"])
            block[f"R{r} Saettigung"] = pad(s["sat"])
            block[f"R{r} Fluessigkeit [m3]"] = pad(s["liq"])
            block[f"R{r} Gewicht W"] = pad(s["W"])
        particles = pd.DataFrame(block)

        # Dichteverteilung q3(x) = dQ3 / dln(x) je Klasse, Replikate nebeneinander
        lo, hi = X_EDGES_UM[:-1], X_EDGES_UM[1:]
        dens = {"x_unten [um]": lo, "x_oben [um]": hi, "x_mitte [um]": np.sqrt(lo * hi)}
        q_all = []
        for r, z in enumerate(reps, start=1):
            s = snapshot(z, i_end)
            q3 = q3_curve(s["d_um"], s["W"] * s["V_solid"], X_EDGES_UM)
            q3_lnx = np.diff(q3) / np.diff(np.log(X_EDGES_UM))
            dens[f"R{r} q3_lnx"] = q3_lnx
            q_all.append(q3_lnx)
        dens["Mittel q3_lnx"] = np.mean(q_all, axis=0)
        dens["Std q3_lnx"] = np.std(q_all, axis=0, ddof=1) if len(q_all) > 1 else np.nan
        dens = pd.DataFrame(dens)
        spacer = pd.DataFrame({" ": [np.nan] * max(len(particles), len(dens))})
        sheet = pd.concat([particles.reindex(range(len(spacer))), spacer,
                           dens.reindex(range(len(spacer)))], axis=1)
        sheets[name] = sheet

    over = pd.DataFrame(overview)
    info = pd.DataFrame({
        "Feld": ["Kandidat", "Fit-Score (KF3, 10 Replikate)", "Quelle", "J_static (KF3)", "Endzeitpunkt der Blaetter",
                 "Dichteverteilung", "Smoke-Test"],
        "Wert": [cand["id"], cand["score"], cand["sqlite"], j_static, f"{t_vec[-1]:g} s",
                 "q3_lnx = dQ3/dln(x), massenbasiert (W * V_solid), Durchmesser aus V_dry", bool(args.smoke)]})
    params = pd.DataFrame({"Parameter": list(cand["params"]), "Wert": list(cand["params"].values())})
    xlsx = out / "kernel_vergleich.xlsx"
    write_excel(xlsx, {"Uebersicht": over, "Kandidat": pd.concat([info, pd.DataFrame({"Feld": [""], "Wert": [""]}),
                                                                  params.rename(columns={"Parameter": "Feld"})]),
                       **sheets})
    print(over.to_string(index=False))
    print(f"\nExcel: {xlsx}")

    # ---------------- Grafiken
    exp_kf3 = pd.read_excel(data_file, sheet_name="Q3_sieve")
    mesh = [c for c in exp_kf3.columns if isinstance(c, (int, float))]
    mesh_um = np.array([float(c) for c in mesh])
    all_src = pd.read_excel(TRIALS / "data" / "AJ_Q3_sieve.xlsx", sheet_name="Q3_sieve")
    singles = all_src[all_src.Reihe.isin(["AJ26", "AJ29", "AJ31"])]
    cmap = plt.get_cmap("viridis")
    tcol = {T_SHOW[0]: cmap(0.25), T_SHOW[1]: cmap(0.75)}
    plt.rcParams.update({"axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
                         "axes.axisbelow": True, "font.size": 10})

    def figure(names_styles: list, title: str, path: Path) -> None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9.5), gridspec_kw={"height_ratios": [2.2, 1]})
        for t_show in T_SHOW:
            i = int(np.flatnonzero(np.isclose(t_vec, t_show))[0])
            c = tcol[t_show]
            for name, ls, label in names_styles:
                reps = tests[name]
                if not reps:
                    continue
                curves = []
                for z in reps:
                    s = snapshot(z, i)
                    curves.append(q3_curve(s["d_um"], s["W"] * s["V_solid"], X_GRID_UM))
                curves = np.vstack(curves)
                mean = curves.mean(axis=0)
                sd = curves.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
                ax1.fill_between(X_GRID_UM, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1),
                                 color=c, alpha=0.15, lw=0, step="post")
                ax1.plot(X_GRID_UM, mean, ls=ls, color=c, lw=1.8, drawstyle="steps-post")
            row = exp_kf3[np.isclose(exp_kf3.t_sim_s, t_show)]
            if len(row):
                ax1.plot(mesh_um, row[mesh].to_numpy(float)[0], "o", color=c, mec="black", mew=0.8, ms=7, ls="none",
                         zorder=5)
            if np.isclose(t_show, 600.0):
                for _, r in singles.iterrows():
                    ax1.plot(mesh_um, r[mesh].to_numpy(float), "o", color=c, ms=4, alpha=0.6, mec="none", ls="none",
                             zorder=4)
        ax1.set_xscale("log")
        ax1.set_xlim(20, 3000)
        ax1.set_ylim(-0.02, 1.02)
        ax1.set_xlabel("Durchmesser x [µm] (log)")
        ax1.set_ylabel("Q3(x) [-]")
        style = [Line2D([], [], color="0.3", ls=ls, lw=1.8, label=f"SIM {label} (Mittelwert)")
                 for _, ls, label in names_styles]
        style += [Patch(facecolor="0.3", alpha=0.25, label="SIM +/- 1 sigma (MC)"),
                  Line2D([], [], color="0.5", marker="o", mec="black", ls="none", ms=7,
                         label="EXP ohne Probenahme (Sieb; 600 s = Mittel)"),
                  Line2D([], [], color="0.5", marker="o", mec="none", ls="none", ms=4, alpha=0.6,
                         label="EXP einzelne 10-min-Versuche")]
        leg = ax1.legend(handles=style, loc="lower right", fontsize=8.5)
        ax1.add_artist(leg)
        ax1.legend(handles=[Line2D([], [], color=tcol[t], lw=3, label=f"{t:g} s") for t in T_SHOW],
                   loc="upper left", bbox_to_anchor=(1.01, 1.0), title="t_sim", fontsize=9)
        ax1.set_title(title, fontsize=10)

        ax2b = ax2.twinx()
        for name, ls, label in names_styles:
            reps = tests[name]
            if not reps:
                continue
            n_comp = np.mean([[len(z[f"W_{i}"]) for i in range(len(t_vec))] for z in reps], axis=0)
            agg = np.mean([z["agg_events"][: len(t_vec)] for z in reps], axis=0)
            brk = np.mean([z["break_events"][: len(t_vec)] for z in reps], axis=0)
            ax2.plot(t_vec, n_comp, ls=ls, color="tab:blue", lw=1.6)
            ax2b.plot(t_vec, agg, ls=ls, color="tab:green", lw=1.6)
            ax2b.plot(t_vec, brk, ls=ls, color="tab:red", lw=1.6)
        ax2.axvspan(0, float(ad._value("LIQUID_ADDITION_DURATION", 60.0)), color="0.85", alpha=0.5, zorder=0)
        ax2.set_xlabel("t_sim [s]")
        ax2.set_ylabel("n_comp", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")
        ax2b.set_ylabel("Ereignisse (kum., Mittel)")
        ax2.legend(handles=[Line2D([], [], color="tab:blue", label="n_comp (Rechenpartikel)"),
                            Line2D([], [], color="tab:green", label="agg-Events (kum.)"),
                            Line2D([], [], color="tab:red", label="break-Events (kum.)")]
                   + [Line2D([], [], color="0.3", ls=ls, label=label) for _, ls, label in names_styles],
                   loc="upper left", fontsize=8, ncol=2)
        fig.suptitle(f"Kernel-Vergleich mit KF3-Bestparametern (id {cand['id']})"
                     + ("  [SMOKE-TEST]" if args.smoke else ""), fontsize=11)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"Grafik: {path}")

    agg_styles = {"EKE": "-", "ETM": "--", "Blend": ":"}
    brk_styles = {"Rumpf": "-", "Powerlaw": "--"}
    for brk in BREAK:
        figure([(test_name(a, brk), agg_styles[a], f"{a} ({AGG[a][0]})") for a in AGG],
               f"Bruch: {brk} ({BREAK[brk]}) – drei Aggregationskernels", out / f"Bruch_{brk}.png")
    for agg in AGG:
        figure([(test_name(agg, b), brk_styles[b], f"{b} ({BREAK[b]})") for b in BREAK],
               f"Aggregation: {agg} ({AGG[agg][0]}) – zwei Bruchkernels", out / f"Agg_{agg}.png")


# ===========================================================================
#  SPYDER / "Play" (keine Kommandozeilen-Argumente): diese Liste wird benutzt.
#  Ohne Schritt-Angabe laufen nacheinander "sim" und "auswertung".
#  Kandidat = Kopie der KF3-SQLite im Repo (Pfad relativ zum Repo-Ordner).
#  Neuerer KF3-Stand: Datei in studies/optimization/kernel_vergleich_input/ ablegen und hier eintragen.
#  Funktionstest zuerst: "--smoke" an die Liste anhaengen.
# ===========================================================================
SPYDER_ARGS: list[str] = ["--sqlite", "studies/optimization/kernel_vergleich_input/KF3_1100.sqlite"]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):   # Spyder-Konsole (TTYOutStream) kann das nicht
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("schritt", nargs="?", choices=["sim", "auswertung"], default=None,
                    help="ohne Angabe: erst sim, dann auswertung")
    ap.add_argument("--workers", type=int, default=min(30, os.cpu_count() or 1))
    ap.add_argument("--max-seconds", type=float, default=7200.0)
    ap.add_argument("--sqlite", default=None, help="absolut oder relativ zum Repo-Ordner")
    ap.add_argument("--smoke", action="store_true")
    argv = sys.argv[1:] if len(sys.argv) > 1 else SPYDER_ARGS
    if not sys.argv[1:]:
        print(f"[Spyder/Play-Modus] keine Argumente -> SPYDER_ARGS = {SPYDER_ARGS}", flush=True)
    args = ap.parse_args(argv)
    if args.sqlite and not Path(args.sqlite).is_absolute():
        args.sqlite = str(REPO / args.sqlite)
    if args.schritt in (None, "sim"):
        simulate(args)
    if args.schritt in (None, "auswertung"):
        evaluate(args)


if __name__ == "__main__":
    main()
