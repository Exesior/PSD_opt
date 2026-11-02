"""
Test der ganzen Kette OHNE Simulation: Design -> Lauf-Dateien -> Auswertung -> Grafiken.

Statt WMCPBE erzeugt ein Ersatzmodell mit BEKANNTER Formel die Partikellisten:
    log10 d50 = log10(200) + 0.30*a + 0.10*b + 0.0*c      (a, b, c gleichverteilt in [0, 1])
    + kleines Seed-Rauschen
Erwartung (analytisch, Varianzen 0.09/12 : 0.01/12 : 0):
    S1(a) ~ 0.90, S1(b) ~ 0.10, S1(c) ~ 0; ST ebenso (keine Wechselwirkung).
Das Dateiformat wird ueber sa_core._collect erzeugt -> identisch zum echten Lauf.

Dazu zwei kleine Szenarien:
  - Fortsetzen ohne time_limit.json mit alten Timeouts (1. Ordnung, run_sensitivity.main)
  - Absturz-Isolation im echten Prozess-Pool (ein Lauf bringt seinen Prozess immer um)

Aufruf:  python test_synthetisch.py   (im Ordner studies/sensitivity; dauert ~1 min) oder Spyder F5
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import analyze_sensitivity as ana
import run_sensitivity as rs
from sa_core import PROBE_COLUMNS, _collect

STUDY = Path(__file__).resolve().parent

CONFIG_TEXT = r'''
STUDY_NAME = "synthetisch"
RESULT_DIR = "unused"
N_SALTELLI = 256
CALC_SECOND_ORDER = True
N_REPLICATES = 3
BASE_REPLICATES = 6
SAMPLE_SEED = 1
WORKERS = 1
TIME_LIMIT_FACTOR = 0
SEC_PER_RUN_ESTIMATE = None
T_SAVE = [0.0, 360.0, 600.0]
T_FULL = [360.0, 600.0]
BIN_DT = 5.0
PROBE_DT = 2.0
from sa_core import REQUIRED_KEYS
BASE_PARAMS = {k: 0.0 for k in REQUIRED_KEYS}      # Platzhalter fuer alle Adapter-Schluessel
BASE_PARAMS.update({"INITIAL_PARTICLES": 500, "INITIAL_WEIGHT": 3.0, "MC_SEED": 7, "COMPRESSION_RATE": 0.1,
                    "N_MIXER": 30.0, "COMPRESSION_C_MIXER": 0.6699, "LIQUID_INTERNALIZATION_RATE": 1e10,
                    "MIN_POROSITY": 0.2, "MAXITER": 1000, "MAX_PARTICLES": 1000, "INITIAL_DIAMETER": 34e-6,
                    "AGG_A": 0.5, "BREAK_P2": 0.5, "RUMPF_K": 0.5})
# Ziele sind echte Adapter-Schluessel (load_config lehnt unbekannte ab); das Ersatzmodell liest sie
SA_PARAMETERS = [
    {"name": "A", "kind": "direct", "target": "AGG_A", "bounds": (0.0, 1.0), "log_scale": False, "base": 0.5},
    {"name": "B", "kind": "direct", "target": "BREAK_P2", "bounds": (0.0, 1.0), "log_scale": False, "base": 0.5},
    {"name": "C", "kind": "direct", "target": "RUMPF_K", "bounds": (0.0, 1.0), "log_scale": False, "base": 0.5},
]
EXP_360 = [0.1, 0.8, 0.97, 0.995, 0.999, 0.9996]
EXP_600_MEAN = [0.034, 0.423, 0.921, 0.983, 0.997, 0.999]
EXP_600_REPEATS = [[0.035684, 0.473779, 0.933153, 0.983846, 0.996805, 0.998975],
                   [0.038448, 0.442506, 0.926427, 0.983327, 0.997567, 0.999051],
                   [0.027029, 0.352731, 0.904597, 0.981857, 0.997902, 0.999568]]
MESH_WEIGHT = [0.282181, 0.227634, 0.164513, 0.145627, 0.129598, 0.050447]
OUTPUTS = ["log10_d50_600", "Q3_250_600", "shift_600", "J_KF3"]
PLOT_SOBOL_OUTPUTS = ["log10_d50_600", "Q3_250_600", "shift_600"]
PLOT_MAIN_EFFECT_OUTPUTS = ["log10_d50_600", "shift_600"]
DISC_CRITERION_FACTOR = 2.0
NOISE_SHARE_LIMIT = 0.10
BOOTSTRAP_RESAMPLES = 200
'''


def fake_run(task: dict) -> None:
    """Ersatz fuer run_replicate: Partikelliste aus der bekannten Formel, gleiches Dateiformat."""
    p = task["params"]
    rng = np.random.default_rng(int(task["seed"]))
    n = 400
    t_save = np.asarray(task["t_save"], dtype=float)
    V_save, W_save = [], []
    for t in t_save:
        frac = t / t_save[-1]
        mu = np.log10(34.0) + frac * (np.log10(200.0) - np.log10(34.0)
                                      + 0.30 * p["AGG_A"] + 0.10 * p["BREAK_P2"] + 0.0 * p["RUMPF_K"])
        mu += rng.normal(0, 0.003)                      # Seed-Rauschen (Common Random Numbers)
        d_um = 10 ** (mu + rng.normal(0, 0.15, n))
        v = np.pi / 6 * (d_um * 1e-6) ** 3
        V_save.append(np.vstack([0.8 * v, v]))
        W_save.append(np.full(n, 3.0))
    solver = SimpleNamespace(V_save=V_save, W_save=W_save, Vc_save=[1e-10] * len(t_save),
                             real_agg_events_save=list(np.linspace(0, 500, len(t_save))),
                             real_break_events_save=list(np.linspace(0, 50, len(t_save))),
                             porosity_save=[np.full(n, 0.2)] * len(t_save),
                             saturation_save=[np.full(n, 0.5)] * len(t_save),
                             liquid_volume_save=[np.zeros(n)] * len(t_save), a_tot=n)
    nb = int(np.ceil(t_save[-1] / task["bin_dt"])) + 1
    probe = SimpleNamespace(events_per_bin=np.full(nb, 100, dtype=np.int64),
                            agg_acc_per_bin=np.full(nb, 10, dtype=np.int64),
                            break_per_bin=np.full(nb, 1, dtype=np.int64),
                            probe_rows=[(t, n, 1e-10, 0.01, 0.05, 0.3, 1500.0, 0.0, 500.0 * t / 600, 50.0 * t / 600)
                                        for t in np.arange(0, 601, 2.0)])
    assert len(probe.probe_rows[0]) == len(PROBE_COLUMNS)
    payload = _collect(task, solver, probe, "ok", "", 0.01, t_save, task["t_full"])
    out = Path(task["out_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="sa_synth_"))
    try:
        cfg_file = tmp / "cfg_synth.py"
        cfg_file.write_text(CONFIG_TEXT, encoding="utf-8")
        cfg = rs.load_config(cfg_file)
        out = tmp / "out"
        X = rs.prepare_design(cfg, out)
        tasks = rs.build_tasks(cfg, X, out)
        # Drei Block-Zustaende nachstellen (Block = 8 Punkte, 256 Bloecke):
        #   ausgefallen  : Punkte p % 97 == 5 -> Seed 1 laeuft in den Timeout
        #   in Arbeit    : Block 215 nur zur Haelfte gerechnet
        #   offen        : Bloecke 216..255 gar nicht gerechnet
        step, first_open, half_block = 8, 216, 215
        bad_blocks = set()
        for t in tasks:
            p = t["point"]
            if p < len(X) and p >= first_open * step:
                continue                                          # offen
            if p < len(X) and p // step == half_block and p % step >= 4 and not (p % step == 4 and t["rep"] == 0):
                continue                                          # in Arbeit (Punkt 4 hat nur Seed 0)
            if p < len(X) and p % 97 == 5 and t["rep"] == 1:
                payload = _collect(t, None, None, "timeout", "", 1.0, np.asarray(t["t_save"]), t["t_full"])
                Path(t["out_path"]).parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(t["out_path"], **payload)
                if p // step != half_block:
                    bad_blocks.add(p // step)
            else:
                fake_run(t)
        n_bad = len(bad_blocks)
        # eine halb geschriebene Datei darf die Auswertung nicht stoeren
        (out / "runs" / "_part_p00003_r00_999.npz").write_bytes(b"kaputt")
        X2 = rs.prepare_design(cfg, out)          # Fortsetzen: gleiches Design muss akzeptiert werden
        assert np.array_equal(X, X2)
        ana.main(["--config", str(cfg_file), "--out", str(out)])
        import pandas as pd
        sob = pd.read_csv(out / "analysis" / "sobol.csv")
        d50 = sob[sob["output"] == "log10_d50_600"].set_index("parameter")
        print(d50[["S1", "S1_conf", "ST", "ST_conf", "blocks_used"]])
        # genutzt = fertig gerechnete Bloecke (0..214) minus die ausgefallenen
        assert d50["blocks_used"].iloc[0] == half_block - n_bad, (d50["blocks_used"].iloc[0], half_block, n_bad)
        rep_txt = (out / "analysis" / "bericht.md").read_text(encoding="utf-8")
        assert "ZWISCHENSTAND" in rep_txt and f"{n_bad} ausgefallen" in rep_txt, rep_txt[:400]
        assert abs(d50.loc["A", "S1"] - 0.90) < 0.08, d50.loc["A", "S1"]
        assert abs(d50.loc["B", "S1"] - 0.10) < 0.06, d50.loc["B", "S1"]
        assert abs(d50.loc["C", "ST"]) < 0.03, d50.loc["C", "ST"]
        pts = pd.read_csv(out / "analysis" / "punkte.csv", index_col="punkt")
        z = pts["zustand"].value_counts()
        print("Zustaende:", z.to_dict())
        assert z["offen"] == (len(X) // step - first_open) * step + 3, z     # 40 Bloecke + Rest des halben Blocks
        assert z["in Arbeit"] == 1 and z["Basis"] == 1, z                   # Punkt mit nur einem Seed
        assert z["ausgefallen"] == n_bad, z                                 # je ausgefallenem Block ein Punkt
        assert (pts.loc[pts["zustand"] == "fertig", "n_ok"] == 3).all()     # N_REPLICATES = 3
        pngs = sorted(p.name for p in (out / "analysis").glob("*.png"))
        print("Grafiken:", pngs)
        assert len(pngs) >= 5
        # Zeitlimit-Pfad: aus den Basispunkt-Laeufen bestimmen, speichern, Faktor-Wechsel
        import json
        prog = rs.Progress(out, len(tasks), 0, 1)
        lim = rs.resolve_time_limit(dict(cfg, TIME_LIMIT_FACTOR=10.0), out, tasks, rs.BASE_ID, prog)
        info = json.loads((out / "time_limit.json").read_text(encoding="utf-8"))
        assert abs(lim - 10.0 * info["base_median_wall_s"]) < 1e-12 and info["factor"] == 10.0
        lim2 = rs.resolve_time_limit(dict(cfg, TIME_LIMIT_FACTOR=20.0), out, tasks, rs.BASE_ID, prog)
        assert abs(lim2 - 2 * lim) < 1e-12
        assert abs(rs.stored_time_limit(out, 20.0) - lim2) < 1e-12
        # Design verlaengern (N 256 -> 512): alte Punkte bleiben, kleineres N wird abgelehnt
        cfg_big = dict(cfg, N_SALTELLI=512)
        X_big = rs.prepare_design(cfg_big, out)
        assert X_big.shape[0] == 2 * X.shape[0] and np.array_equal(X_big[: X.shape[0]], X)
        try:
            rs.prepare_design(dict(cfg, N_SALTELLI=128), out)
            raise AssertionError("kleineres N haette abgelehnt werden muessen")
        except SystemExit:
            pass
        keep = STUDY / "test_ausgabe_synthetisch"
        if keep.exists():
            shutil.rmtree(keep)
        shutil.copytree(out / "analysis", keep)

        scenario_first_order_resume(tmp)
        scenario_crash_isolation(tmp)
        print("TEST OK")
        print(f"Auswertung zum Anschauen: {keep}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _small_config(tmp: Path, name: str) -> Path:
    """Kleine Studie 1. Ordnung (wie die Produktions-Config): D = 3, Block = 5 Punkte, N = 16."""
    text = (CONFIG_TEXT.replace("N_SALTELLI = 256", "N_SALTELLI = 16")
            .replace("CALC_SECOND_ORDER = True", "CALC_SECOND_ORDER = False")
            .replace("TIME_LIMIT_FACTOR = 0", "TIME_LIMIT_FACTOR = 10.0")
            .replace("N_REPLICATES = 3", "N_REPLICATES = 2")
            .replace("BASE_REPLICATES = 6", "BASE_REPLICATES = 3"))
    f = tmp / f"{name}.py"
    f.write_text(text, encoding="utf-8")
    return f


def scenario_first_order_resume(tmp: Path) -> None:
    """Fortsetzen ohne time_limit.json: Basispunkt unvollstaendig, Timeouts unter einem
    vorlaeufigen kleineren Limit. Nach main() muss ALLES ok sein (nichts abgeschnitten),
    danach Sobol 1. Ordnung ueber alle 16 Bloecke ohne ZWISCHENSTAND."""
    cfg_file = _small_config(tmp, "cfg_resume")
    cfg = rs.load_config(cfg_file)
    out = tmp / "out_resume"
    X = rs.prepare_design(cfg, out)
    tasks = rs.build_tasks(cfg, X, out)
    n_to = 0
    for t in tasks:
        p, r = t["point"], t["rep"]
        if p == rs.BASE_ID:
            if r < 2:
                fake_run(t)                       # Basis-Seed 2 fehlt -> kein Limit gespeichert
        elif p < 40:
            if p % 4 == 0 and r == 0:             # Timeout unter vorlaeufigem Limit 0,05 s
                t2 = dict(t, time_limit_s=0.05)
                payload = _collect(t2, None, None, "timeout", "", 0.05, np.asarray(t["t_save"]), t["t_full"])
                Path(t["out_path"]).parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(t["out_path"], **payload)
                n_to += 1
            else:
                fake_run(t)
    calls = []

    def fake_pool(task_list, workers, prog, done_per_point):
        calls.append(len(task_list))
        for t in task_list:
            fake_run(t)                           # Wandzeit 0,01 s -> Limit 10 x 0,01 = 0,1 s

    real_pool = rs.run_pool
    rs.run_pool = fake_pool
    try:
        rs.main(["--config", str(cfg_file), "--out", str(out)])
    finally:
        rs.run_pool = real_pool
    from sa_core import read_meta
    not_ok = [(t["point"], t["rep"]) for t in tasks
              if (read_meta(Path(t["out_path"])) or {}).get("status") != "ok"]
    assert not not_ok, f"nach 'Fertig' nicht ok: {not_ok[:10]}"
    n_missing = sum(1 for t in tasks if t["point"] != rs.BASE_ID and t["point"] >= 40)
    assert calls == [1, n_missing + n_to], (calls, n_missing, n_to)
    ana.main(["--config", str(cfg_file), "--out", str(out), "--no-plots"])
    import pandas as pd
    sob = pd.read_csv(out / "analysis" / "sobol.csv")
    assert (sob.loc[sob["output"] == "log10_d50_600", "blocks_used"] == 16).all()
    assert "ZWISCHENSTAND" not in (out / "analysis" / "bericht.md").read_text(encoding="utf-8")
    print(f"Szenario Fortsetzen/1. Ordnung OK ({n_to} alte Timeouts neu gerechnet, 16 Bloecke)")


CRASH_POINT = 3


def crash_worker(task: dict) -> dict:
    """Ersatz-Lauf im Arbeitsprozess; Punkt CRASH_POINT / Seed 1 beendet den Prozess IMMER hart."""
    if task["point"] == CRASH_POINT and task["rep"] == 1:
        os._exit(1)
    time.sleep(0.2)
    fake_run(task)
    return {"point": task["point"], "rep": task["rep"], "status": "ok", "wall_s": 0.2, "events": 1,
            "n_comp_end": 0, "error": ""}


def scenario_crash_isolation(tmp: Path) -> None:
    """Echter Prozess-Pool: ein Lauf bringt seinen Prozess jedes Mal um. Er muss als 'crashed'
    gespeichert werden, alle mit ihm abgestuerzten Unschuldigen muessen 'ok' werden."""
    import importlib
    from collections import defaultdict
    from sa_core import read_meta
    if str(STUDY) not in sys.path:
        sys.path.insert(0, str(STUDY))            # Kindprozesse muessen dieses Modul importieren
    me = importlib.import_module("test_synthetisch")
    cfg = rs.load_config(_small_config(tmp, "cfg_crash"))
    out = tmp / "out_crash"
    X = rs.prepare_design(cfg, out)
    tasks = [t for t in rs.build_tasks(cfg, X, out) if t["point"] < 6]          # 12 Laeufe
    real_rep = rs.run_replicate
    rs.run_replicate = me.crash_worker
    try:
        prog = rs.Progress(out, len(tasks), 0, 2)
        rs.run_pool(tasks, 2, prog, defaultdict(int))
    finally:
        rs.run_replicate = real_rep
    status = {(t["point"], t["rep"]): (read_meta(Path(t["out_path"])) or {}).get("status") for t in tasks}
    crashed = [k for k, s in status.items() if s == "crashed"]
    assert crashed == [(CRASH_POINT, 1)], status
    assert all(s == "ok" for k, s in status.items() if k != (CRASH_POINT, 1)), status
    assert prog.n_done == len(tasks), (prog.n_done, len(tasks))
    print("Szenario Absturz-Isolation OK (Verursacher 'crashed', alle anderen ok)")


if __name__ == "__main__":
    main()
