"""
Sensitivitaetsstudie rechnen (Remote-Rechner, Spyder: einfach F5).

Ablauf
------
1. Config laden (Standard: sa_config_gesamt.py, siehe SPYDER_ARGS unten).
2. Saltelli-Stichprobe erzeugen (SALib, wie im Betreuer-Skript, aktuelle API
   ``SALib.sample.sobol``) und als design.npz / design.csv im Ergebnisordner ablegen.
   Dazu kommt der Basispunkt (mehr Replikate, Referenz fuer Verschiebung + Rauschen).
3. Jeder Lauf = ein Parameterpunkt x ein Seed. ALLE Punkte laufen mit denselben
   Seeds (Common Random Numbers) -> Unterschiede kommen von den Parametern.
4. Laeufe parallel (ProcessPoolExecutor). Jeder fertige Lauf landet sofort als
   eigene .npz in <Ergebnisordner>/runs/. Nach einem Abbruch einfach neu starten:
   fertige Laeufe werden erkannt und uebersprungen.
5. Fortschritt: eine Zeile pro fertigem Lauf in der Konsole UND in progress.log.

Aufruf
------
Spyder:   F5 (Argumente aus SPYDER_ARGS)
Konsole:  python run_sensitivity.py --config sa_config_gesamt.py [--workers 6]
Weitere Schalter:
    --dry-run       nur Plan + Laufzeitschaetzung ausgeben, nichts rechnen
    --benchmark K   K Replikate des Basispunkts parallel rechnen und daraus die
                    Gesamtdauer hochrechnen (Ergebnisse landen in <Ergebnis>/benchmark/)
    --max-tasks K   nur die naechsten K offenen Laeufe (zum Testen)
    --out DIR       anderer Ergebnisordner (zum Testen)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import runpy
import sys
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# Die Laeufe rechnen in eigenen Prozessen; numba soll dort EINEN Thread nehmen,
# sonst kaempfen 6 Prozesse x 12 Threads um 12 Kerne (siehe README, Parallelisierung).
# Muss gesetzt sein, BEVOR ein Kindprozess numba importiert (Kinder erben os.environ).
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

from sa_core import BASE_ID, _collect, check_params, read_meta, replicate_seeds, run_replicate  # noqa: E402

# Spyder (F5, keine Kommandozeilen-Argumente) -> diese Argumente
SPYDER_ARGS = ["--config", "sa_config_gesamt.py"]


# ===========================================================================
#  Config + Design
# ===========================================================================
def load_config(path: Path) -> dict:
    cfg = runpy.run_path(str(path))
    need = ["STUDY_NAME", "BASE_PARAMS", "SA_PARAMETERS", "N_SALTELLI", "CALC_SECOND_ORDER",
            "N_REPLICATES", "BASE_REPLICATES", "SAMPLE_SEED", "T_SAVE", "T_FULL", "BIN_DT",
            "PROBE_DT", "TIME_LIMIT_FACTOR", "WORKERS", "RESULT_DIR"]
    missing = [k for k in need if k not in cfg]
    if missing:
        raise KeyError(f"Config {path.name} fehlt: {missing}")
    check_params(cfg["BASE_PARAMS"])
    # Tippfehler-Schutz: ein "direct"-Ziel, das der Adapter nicht liest, waere still wirkungslos
    from sa_core import REQUIRED_KEYS
    for spec in cfg["SA_PARAMETERS"]:
        if spec.get("kind", "direct") == "direct" and spec["target"] not in REQUIRED_KEYS:
            raise KeyError(f"{path.name}: Parameter {spec['name']!r} zielt auf {spec['target']!r}, "
                           f"das liest der Adapter nicht (Tippfehler?).")
        lo, hi = map(float, spec["bounds"])
        if not lo < hi:
            raise ValueError(f"{path.name}: Parameter {spec['name']!r} hat bounds {spec['bounds']} (unten >= oben).")
    # Der Basispunkt muss den Basisfall wiedergeben (sonst passen die 'base'-Werte nicht mehr
    # zu BASE_PARAMS, z. B. nachdem dort etwas geaendert wurde)
    p, _ = to_params(cfg, base_row(cfg), exact_base=False)      # den echten Rechenweg pruefen
    off = [k for k, v in cfg["BASE_PARAMS"].items()
           if isinstance(v, (int, float)) and not isinstance(v, bool)
           and abs(float(p[k]) - float(v)) > 1e-9 * abs(float(v))]       # Basis 0 -> muss exakt 0 sein
    if off:
        raise ValueError(f"{path.name}: Der Basispunkt (Werte 'base') weicht von BASE_PARAMS ab bei {off}.")
    # Ein "direct"-Ziel, das danach eine umgerechnete Art ueberschreibt (z. B. AGG_DW_MAX durch
    # agg_dw_rel) oder ein zweiter Eintrag mit demselben Ziel, waere still wirkungslos
    # -> an beiden Bereichs-Enden pruefen, ob der gezogene Wert ankommt
    specs = cfg["SA_PARAMETERS"]
    for end in (0, 1):
        row = np.array([float(s["bounds"][end]) for s in specs])
        p_end, _ = to_params(cfg, row, exact_base=False)
        for spec, x in zip(specs, row):
            if spec.get("kind", "direct") != "direct":
                continue
            want = 10.0 ** x if spec.get("log_scale", False) else x
            if not np.isclose(float(p_end[spec["target"]]), want, rtol=1e-12, atol=0.0):
                raise ValueError(f"{path.name}: Parameter {spec['name']!r} kommt nicht an, "
                                 f"{spec['target']!r} wird von einem anderen Eintrag ueberschrieben.")
    return cfg


def salib_problem(cfg: dict) -> dict:
    """SALib-Problem; log-Parameter werden im Exponenten (log10) gezogen, wie im Optimierer."""
    return {
        "num_vars": len(cfg["SA_PARAMETERS"]),
        "names": [p["name"] for p in cfg["SA_PARAMETERS"]],
        "bounds": [list(map(float, p["bounds"])) for p in cfg["SA_PARAMETERS"]],
    }


def design_matrix(cfg: dict) -> np.ndarray:
    """Saltelli-Stichprobe (Zeilen = Punkte) im Zieh-Raum (log10 fuer log-Parameter)."""
    from SALib.sample import sobol as sobol_sample
    return sobol_sample.sample(salib_problem(cfg), int(cfg["N_SALTELLI"]),
                               calc_second_order=bool(cfg["CALC_SECOND_ORDER"]),
                               scramble=True, seed=int(cfg["SAMPLE_SEED"]))


def base_row(cfg: dict) -> np.ndarray:
    """Der Basispunkt im Zieh-Raum (aus BASE_VALUES der Config)."""
    return np.array([float(p["base"]) for p in cfg["SA_PARAMETERS"]], dtype=float)


def to_params(cfg: dict, row: np.ndarray, exact_base: bool = True) -> tuple[dict, dict]:
    """Zeile im Zieh-Raum -> (vollstaendige pop_params, dict der variierten Groessen in phys. Einheiten).

    Parameter-Arten (Schluessel "kind" in SA_PARAMETERS):
        "direct"          Wert (bzw. 10**Wert bei log_scale) direkt auf den pop_params-Schluessel
        "break_rate_ref"  Bruchrate eines Referenz-Granulats: S_ref = BREAK_P1 * V_ref^BREAK_P2
                          (V_ref aus d_ref). BREAK_P1 wird daraus NACH allen direkten
                          Parametern bestimmt, damit BREAK_P2 nur die Groessenabhaengigkeit
                          dreht und nicht zugleich das Niveau um Dekaden verschiebt.
        "agg_rate_ref"    Agglomerations-Vorfaktor fuer zwei Partikel mit Radius r_ref:
                          a_ref = EKE_CORR_BETA * r_ref^(-1.5 * AGG_A)  (kinetic_blend bei
                          r1 = r2 = r_ref, ohne die von AGG_A unabhaengigen Faktoren).
                          EKE_CORR_BETA wird daraus NACH AGG_A bestimmt. Grund: im Fit sind
                          beide entlang genau dieser Linie gekoppelt (r_ref = 17 um,
                          Korrelation -0.999); unabhaengig gezogen laege fast jeder Punkt
                          dekadenweit neben dem Tal -> Laufzeit-Explosion.
        "droplet_box"     Tropfendurchmesser [m]. Ist er groesser als d_box_ref, waechst die Box
                          (CONTROL_VOLUME, INITIAL_WEIGHT, VOLUMETRIC_FLOW_RATE) mit
                          (d/d_box_ref)^3, damit die Zahl der Tropfen in der Box nicht unter
                          den Basiswert faellt (Tropfen und Box "nicht zu nah", Vorgabe Eric).
                          Die Boxgroesse selbst aendert die Verteilung nicht (Kalibrierung).
        "agg_dw_rel"      AGG_DW_MAX relativ zum Startgewicht: AGG_DW_MAX = Wert * INITIAL_WEIGHT
                          (nach einer evtl. Box-Vergroesserung). So bleibt die Aufloesung
                          (Kollisionen je Event / Gewicht) gleich, wenn die Box waechst; mit
                          absolutem dW wuerde jede groessere Box die Eventzahl vervielfachen
                          (gemessen: dW 1 + Box x3.4 -> ~80x langsamer).
        "porosity_mass_fixed"  Porositaet der Primaerpartikel eps0 bei FESTER Feststoffmasse
                          (Vorgabe Eric): Durchmesser und Anzahl bleiben, der Adapter setzt den
                          Feststoff auf (1 - eps0), und die effektive Dichte waechst mit
                          STOKES_RHO_SOLID = rho_Basis * (1 - eps0_Basis) / (1 - eps0).
                          rho * V_fest bleibt so exakt die Basis-Masse. Hintergrund: Die reale
                          Porositaet (~0.8) steckt bisher ganz in rho_eff = 0.2 rho_s; eps0
                          verschiebt einen Teil davon in echten Porenraum. Die Dichte geht im
                          Solver nur in die Stokes-Masse ein (Kollisionsradius und Bruch
                          rechnen mit dem Huellvolumen).
    Reihenfolge: erst alle "direct", dann droplet_box, agg_dw_rel, agg_rate_ref, break_rate_ref,
    porosity_mass_fixed.
    """
    params = dict(cfg["BASE_PARAMS"])
    values = {}
    if exact_base and np.array_equal(np.asarray(row, dtype=float), base_row(cfg)):
        # Basispunkt: exakt die Basiswerte (ohne Rundung durch 10**log10 und Umrechnungen)
        for spec, x in zip(cfg["SA_PARAMETERS"], row):
            values[spec["name"]] = 10.0 ** float(x) if spec.get("log_scale", False) else float(x)
        return params, values
    deferred = []
    order = {"droplet_box": 0, "agg_dw_rel": 1, "agg_rate_ref": 2, "break_rate_ref": 3,
             "porosity_mass_fixed": 4}
    for spec, x in zip(cfg["SA_PARAMETERS"], row):
        kind = spec.get("kind", "direct")
        val = 10.0 ** float(x) if spec.get("log_scale", False) else float(x)
        values[spec["name"]] = val
        if kind == "direct":
            params[spec["target"]] = val
        elif kind in order:
            deferred.append((kind, spec, val))
        else:
            raise ValueError(f"unbekannte Parameter-Art {kind!r} bei {spec['name']}")
    deferred.sort(key=lambda d: order[d[0]])
    for kind, spec, val in deferred:
        if kind == "break_rate_ref":
            v_ref = np.pi / 6.0 * float(spec["d_ref"]) ** 3
            params["BREAK_P1"] = float(val / v_ref ** float(params["BREAK_P2"]))
            values["BREAK_P1"] = params["BREAK_P1"]
        elif kind == "agg_rate_ref":
            params["EKE_CORR_BETA"] = float(val * float(spec["r_ref"]) ** (1.5 * float(params["AGG_A"])))
            values["EKE_CORR_BETA"] = params["EKE_CORR_BETA"]
        elif kind == "droplet_box":
            params["DROPLET_DIAMETER"] = float(val)
            d_ref = float(spec["d_box_ref"])
            # bis d_ref (inkl. Rundung von 10**log10) bleibt die Box exakt die Basis-Box
            grow = 1.0 if val <= d_ref * (1.0 + 1e-9) else (float(val) / d_ref) ** 3
            # AGG_DW_MAX waechst mit (gleiche relative Aufloesung); ein agg_dw_rel-Parameter
            # ueberschreibt das danach ohnehin.
            for key in ("CONTROL_VOLUME", "INITIAL_WEIGHT", "VOLUMETRIC_FLOW_RATE", "AGG_DW_MAX"):
                params[key] = float(params[key]) * grow
            values["BOX_FACTOR"] = grow
        elif kind == "agg_dw_rel":
            params["AGG_DW_MAX"] = float(val * float(params["INITIAL_WEIGHT"]))
            values["AGG_DW_MAX"] = params["AGG_DW_MAX"]
        elif kind == "porosity_mass_fixed":
            if not 0.0 <= val < 1.0:
                raise ValueError(f"{spec['name']}: eps0 muss in [0, 1) liegen, ist {val}")
            base = cfg["BASE_PARAMS"]
            rho_mass = float(base["STOKES_RHO_SOLID"]) * (1.0 - float(base["INITIAL_POROSITY"]))
            params["INITIAL_POROSITY"] = float(val)
            params["STOKES_RHO_SOLID"] = rho_mass / (1.0 - float(val))
            values["STOKES_RHO_SOLID"] = params["STOKES_RHO_SOLID"]
    return params, values


def config_fingerprint(cfg: dict) -> str:
    """Fingerabdruck alles dessen, was die Laeufe bestimmt (fuer das sichere Fortsetzen).

    N_SALTELLI gehoert bewusst NICHT dazu: die Saltelli-Stichprobe ist praefix-stabil
    (das Design mit 2N enthaelt das mit N als erste Bloecke), N darf also spaeter erhoeht
    werden, ohne dass gerechnete Laeufe ungueltig werden.
    """
    keys = ["BASE_PARAMS", "SA_PARAMETERS", "CALC_SECOND_ORDER", "SAMPLE_SEED",
            "T_SAVE", "T_FULL", "BIN_DT", "PROBE_DT"]
    blob = json.dumps({k: cfg[k] for k in keys}, sort_keys=True, default=lambda o: np.asarray(o).tolist())
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def prepare_design(cfg: dict, out_dir: Path, write: bool = True) -> np.ndarray:
    """Design erzeugen bzw. beim Fortsetzen pruefen, dass es dasselbe ist.

    Erlaubt ist nur eine Verlaengerung (groesseres N_SALTELLI, sonst alles gleich): dann
    muss das alte Design genau der Anfang des neuen sein. write=False (--dry-run) prueft
    nur und schreibt nichts.
    """
    X = design_matrix(cfg)
    if X.shape[0] >= BASE_ID:
        raise SystemExit(f"Design mit {X.shape[0]} Punkten ist zu gross (Punktnummern sind 5-stellig, "
                         f"{BASE_ID} ist der Basispunkt). N_SALTELLI verkleinern.")
    fp = config_fingerprint(cfg)
    dfile = out_dir / "design.npz"
    if dfile.exists():
        with np.load(dfile, allow_pickle=False) as z:
            old_X, old_fp = z["X"], str(z["fingerprint"])
        same = old_fp == fp and old_X.shape == X.shape and np.array_equal(old_X, X)
        extend = (old_fp == fp and old_X.shape[0] < X.shape[0] and old_X.shape[1] == X.shape[1]
                  and np.array_equal(old_X, X[: old_X.shape[0]]))
        if same:
            return X
        if not extend:
            raise SystemExit(
                f"Im Ergebnisordner {out_dir} liegt ein ANDERES Design (andere Config"
                f"{' oder kleineres N' if old_fp == fp else ''}).\n"
                f"Entweder den alten Ordner umbenennen oder RESULT_DIR in der Config aendern.")
        print(f"Design wird verlaengert: {old_X.shape[0]} -> {X.shape[0]} Punkte "
              f"(gerechnete Laeufe bleiben gueltig).")
    if not write:
        return X
    out_dir.mkdir(parents=True, exist_ok=True)
    prob = salib_problem(cfg)
    tmp = out_dir / "_part_design.npz"                 # erst fertig schreiben, dann umbenennen
    np.savez(tmp, X=X, x_base=base_row(cfg), names=np.array(prob["names"]),
             bounds=np.array(prob["bounds"]),
             log_scale=np.array([bool(p.get("log_scale", False)) for p in cfg["SA_PARAMETERS"]]),
             N=int(cfg["N_SALTELLI"]), calc_second_order=bool(cfg["CALC_SECOND_ORDER"]),
             fingerprint=np.array(fp), sa_parameters_json=np.array(json.dumps(cfg["SA_PARAMETERS"])))
    os.replace(tmp, dfile)
    with open(out_dir / "design.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=";")
        names = list(to_params(cfg, X[0])[1])              # alle Spalten inkl. abgeleiteter Groessen
        w.writerow(["punkt"] + names)
        for label, row in [(BASE_ID, base_row(cfg))] + list(enumerate(X)):
            p, vals = to_params(cfg, row)
            # abgeleitete Groessen (BREAK_P1, EKE_CORR_BETA, AGG_DW_MAX, BOX_FACTOR) am Basispunkt
            # direkt aus den Parametern
            cells = [vals[n] if n in vals else (p[n] if n in p else 1.0) for n in names]
            w.writerow([f"{label} (Basis)" if label == BASE_ID else label] + [repr(c) for c in cells])
    return X


def build_tasks(cfg: dict, X: np.ndarray, out_dir: Path) -> list[dict]:
    """Alle Laeufe: Saltelli-Punkte (N_REPLICATES Seeds) + Basispunkt (BASE_REPLICATES Seeds).

    Reihenfolge: Basispunkt zuerst (Referenz frueh da), dann Punkt fuer Punkt.
    """
    n_rep, n_base = int(cfg["N_REPLICATES"]), int(cfg["BASE_REPLICATES"])
    seeds = replicate_seeds(int(cfg["BASE_PARAMS"]["MC_SEED"]), max(n_rep, n_base))
    common = {
        "t_save": np.asarray(cfg["T_SAVE"], dtype=float), "t_full": list(cfg["T_FULL"]),
        "bin_dt": float(cfg["BIN_DT"]), "probe_dt": float(cfg["PROBE_DT"]),
        "time_limit_s": 0.0,            # wird nach dem Basispunkt gesetzt (resolve_time_limit)
        "study": cfg["STUDY_NAME"],
    }
    rows = [(BASE_ID, base_row(cfg), n_base)] + [(i, X[i], n_rep) for i in range(len(X))]
    tasks = []
    for point, row, reps in rows:
        params, values = to_params(cfg, row)
        for r in range(reps):
            t = dict(common)
            t.update({"params": params, "sa_values": values, "point": int(point), "rep": r,
                      "n_rep_point": reps, "seed": seeds[r],
                      "out_path": str(out_dir / "runs" / f"p{point:05d}_r{r:02d}.npz")})
            tasks.append(t)
    return tasks


# ===========================================================================
#  Ausgabe
# ===========================================================================
def point_label(point: int) -> str:
    return "Basis     " if int(point) == BASE_ID else f"Punkt {int(point):5d}"


class Progress:
    """Fortschrittszeilen in Konsole + progress.log (von aussen pruefbar)."""

    def __init__(self, out_dir: Path, n_total: int, n_done_before: int, workers: int):
        self.log = out_dir / "progress.log"
        self.fail_log = out_dir / "failures.log"
        self.n_total = n_total
        self.n_done = n_done_before
        self.workers = workers
        self.t0 = time.perf_counter()
        self.walls: list[float] = []
        self.per_point = defaultdict(int)

    def say(self, text: str):
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {text}"
        print(line, flush=True)
        with open(self.log, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def done(self, res: dict, n_rep_point: int, n_done_point: int):
        self.n_done += 1
        self.walls.append(res["wall_s"])
        left = self.n_total - self.n_done
        elapsed = time.perf_counter() - self.t0
        rate = len(self.walls) / elapsed if elapsed > 0 else 0.0
        eta = left / rate / 3600 if rate > 0 else float("nan")
        status = "" if res["status"] == "ok" else f" [{res['status'].upper()}]"
        self.say(f"{point_label(res['point'])}: {n_done_point}/{n_rep_point} fertig{status} "
                 f"({res['wall_s']:.0f} s, {res['events']} Events) | gesamt {self.n_done}/{self.n_total} "
                 f"({100 * self.n_done / self.n_total:.1f} %) | Rest ~{eta:.1f} h")
        if n_done_point == n_rep_point:
            self.say(f"{point_label(res['point'])}: {n_rep_point}/{n_rep_point} abgeschlossen")
        if res["status"] != "ok":
            with open(self.fail_log, "a", encoding="utf-8") as fh:
                fh.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} punkt={res['point']} rep={res['rep']} "
                         f"status={res['status']} {res.get('error', '')}\n")


def estimate(cfg: dict, n_tasks: int, sec_per_run: float | None, workers: int) -> str:
    if not sec_per_run:
        return "keine Laufzeit bekannt (start_3_benchmark.py ausfuehren)"
    hours = n_tasks * sec_per_run / workers / 3600
    return f"{n_tasks} Laeufe x {sec_per_run:.0f} s / {workers} parallel = {hours:.1f} h"


# ===========================================================================
#  Rechnen
# ===========================================================================
def run_pool(tasks: list[dict], workers: int, prog: Progress, done_per_point: dict):
    """Laeufe parallel abarbeiten; hoechstens 2*workers gleichzeitig in der Warteschlange.

    Stirbt ein Arbeitsprozess hart (z. B. Speicher), ist der ganze Pool kaputt
    (BrokenProcessPool) und ALLE Laeufe darin scheitern, auch unschuldige. Sie werden in
    einem neuen Pool wiederholt. Wer dreimal in einem kaputten Pool war, laeuft am Ende
    allein (Isolation): Stirbt er auch allein, ist er der Verursacher und bekommt eine
    Datei mit Status 'crashed' (Block gilt als ausgefallen; erneut versuchen = Datei
    loeschen). Die Unschuldigen rechnen allein ganz normal.
    """
    from concurrent.futures.process import BrokenProcessPool

    pending = list(tasks)
    suspects = []                  # dreimal in einem kaputten Pool -> am Ende einzeln
    while pending:
        running = {}
        ex = ProcessPoolExecutor(max_workers=workers)
        broken = False
        retry = []                 # Auftraege des kaputten Pools (ohne Datei, NICHT gezaehlt)
        try:
            while (pending or running) and not broken:
                while pending and len(running) < 2 * workers:
                    t = pending.pop(0)
                    try:
                        running[ex.submit(run_replicate, t)] = t
                    except BrokenProcessPool:
                        broken = True
                        retry.append(t)
                        break
                if broken:
                    break
                finished, _ = wait(list(running), return_when=FIRST_COMPLETED)
                for fu in finished:
                    t = running.pop(fu)
                    try:
                        res = fu.result()
                    except BrokenProcessPool:
                        # Pool tot: ALLE laufenden/wartenden Futures scheitern gleichzeitig.
                        # Nicht als fertig zaehlen, sondern neu einreihen.
                        broken = True
                        retry.append(t)
                        continue
                    except Exception as exc:          # sonstiger Fehler ausserhalb run_replicate
                        res = write_crashed(t, repr(exc))
                    done_per_point[t["point"]] += 1
                    prog.done(res, t["n_rep_point"], done_per_point[t["point"]])
            if broken:
                # wer schon dreimal in einem abgestuerzten Pool war, laeuft am Ende allein
                requeue, n_susp = [], len(suspects)
                for t in retry + list(running.values()):
                    t["_crashes"] = t.get("_crashes", 0) + 1
                    if t["_crashes"] <= 2:
                        requeue.append(t)
                    else:
                        suspects.append(t)
                        prog.say(f"{point_label(t['point'])} Replikat {t['rep']}: dreimal in abgestuerztem Pool "
                                 f"-> wird am Ende einzeln gerechnet")
                pending = requeue + pending
                prog.say(f"WARNUNG: ein Arbeitsprozess ist abgestuerzt; neuer Pool, {len(requeue)} Laeufe "
                         f"werden neu gestartet"
                         + (f", {len(suspects) - n_susp} am Ende einzeln." if len(suspects) > n_susp else "."))
        except KeyboardInterrupt:
            prog.say("Abbruch angefordert. Noch nicht gestartete Laeufe werden verworfen; fertige Laeufe "
                     "sind gespeichert. Neustart (F5) setzt fort.")
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            ex.shutdown(wait=not broken, cancel_futures=True)

    # Isolation: jeder Verdaechtige allein in einem eigenen Prozess
    for t in suspects:
        ex1 = ProcessPoolExecutor(max_workers=1)
        dead = False
        try:
            res = ex1.submit(run_replicate, t).result()
        except BrokenProcessPool:
            dead = True
            res = write_crashed(t, "Arbeitsprozess stirbt auch allein (3x im Pool, 1x einzeln)")
        except Exception as exc:
            res = write_crashed(t, repr(exc))
        finally:
            ex1.shutdown(wait=not dead, cancel_futures=True)
        done_per_point[t["point"]] += 1
        prog.done(res, t["n_rep_point"], done_per_point[t["point"]])


def write_crashed(t: dict, error: str) -> dict:
    """Lauf, der sich nicht rechnen laesst: Meta-Datei mit Status 'crashed' schreiben.
    Damit gilt sein Block als ausgefallen (nicht 'offen'), und ein Neustart wiederholt ihn
    nicht endlos. Erneut versuchen = Datei loeschen."""
    res = {"point": t["point"], "rep": t["rep"], "status": "crashed", "wall_s": 0.0, "events": 0,
           "error": error}
    try:
        payload = _collect(t, None, None, "crashed", error, 0.0,
                           np.asarray(t["t_save"], dtype=float), t["t_full"])
        out = Path(t["out_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f"_part_{out.stem}_crashed.npz")
        np.savez_compressed(tmp, **payload)
        os.replace(tmp, out)
    except Exception as exc:                  # dann eben ohne Datei (naechster Start versucht es neu)
        res["error"] += f" | Datei nicht geschrieben: {exc!r}"
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="sa_config_gesamt.py")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--benchmark", type=int, default=0)
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true",
                    help="Umgebung pruefen: Importe, SALib, Config, Solver bauen + 200 Events, Code-Stand")
    ap.add_argument("--make-reference", action="store_true",
                    help="nur auf dem Laptop, mit --check: code_referenz.json (neu) schreiben")
    ap.add_argument("--benchmark-workers", type=int, nargs="+", default=None,
                    help="Durchsatz fuer mehrere Prozesszahlen vergleichen, z. B. 6 9 12")
    args = ap.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = HERE / cfg_path
    cfg = load_config(cfg_path)
    out_dir = Path(args.out) if args.out else Path(cfg["RESULT_DIR"])
    if not out_dir.is_absolute():
        out_dir = HERE / out_dir
    workers = int(args.workers or cfg["WORKERS"])

    if args.check:
        return check(cfg, make_reference=args.make_reference)
    if args.benchmark_workers:
        return benchmark_workers(cfg, out_dir, args.benchmark_workers)
    if args.benchmark:
        return benchmark(cfg, out_dir, args.benchmark, workers)

    X = prepare_design(cfg, out_dir, write=not args.dry_run)
    tasks = build_tasks(cfg, X, out_dir)
    base_id = BASE_ID
    factor = float(cfg["TIME_LIMIT_FACTOR"])
    metas = {t["out_path"]: read_meta(Path(t["out_path"])) for t in tasks}

    def is_done(t, limit):
        """Fertig = Datei da. Ausnahme: Timeout unter einem KLEINEREN Limit als dem jetzt
        gueltigen (z. B. TIME_LIMIT_FACTOR erhoeht; Faktor 0 = unbegrenzt) -> neu rechnen."""
        m = metas[t["out_path"]]
        if m is None:
            return False
        if m["status"] == "timeout":
            if factor <= 0:
                return False
            if limit > 0 and m.get("time_limit_s", 0.0) < 0.999 * limit:
                return False
        return True

    stored = stored_time_limit(out_dir, factor)
    done = [t for t in tasks if is_done(t, stored)]
    todo = [t for t in tasks if not is_done(t, stored)]
    n_retry = sum(1 for t in todo if metas[t["out_path"]] is not None)
    done_per_point = defaultdict(int)
    for t in done:
        done_per_point[t["point"]] += 1

    n_pts = len(X) + 1
    print(f"Studie {cfg['STUDY_NAME']}: {len(cfg['SA_PARAMETERS'])} Parameter, N = {cfg['N_SALTELLI']}, "
          f"2. Ordnung = {cfg['CALC_SECOND_ORDER']} -> {len(X)} Saltelli-Punkte + Basispunkt "
          f"= {n_pts} Punkte, {len(tasks)} Laeufe ({cfg['N_REPLICATES']} Seeds je Punkt, "
          f"Basis {cfg['BASE_REPLICATES']}).")
    print(f"Ergebnisordner: {out_dir}")
    print(f"Schon fertig: {len(done)}, offen: {len(todo)}"
          + (f" (davon {n_retry} Timeouts mit altem, kleinerem Limit)" if n_retry else "")
          + f". Parallel: {workers}. "
          f"Schaetzung: {estimate(cfg, len(todo), cfg.get('SEC_PER_RUN_ESTIMATE'), workers)}")
    if args.dry_run or not todo:
        return
    budget = args.max_tasks if args.max_tasks else None          # None = alles

    prog = Progress(out_dir, len(tasks), len(done), workers)
    prog.say(f"Start: {min(budget or len(todo), len(todo))} Laeufe, {workers} parallel, Config {cfg_path.name}")
    record_code_state(out_dir, prog)

    # 1) Basispunkt zuerst und OHNE Zeitlimit: Referenz + Laufzeit dieses Rechners
    base_todo = [t for t in todo if t["point"] == base_id]
    if budget is not None:
        base_todo = base_todo[:budget]
        budget -= len(base_todo)
    if base_todo:
        run_pool(base_todo, workers, prog, done_per_point)
    limit = resolve_time_limit(cfg, out_dir, tasks, base_id, prog)

    # 2) alle Saltelli-Punkte mit Zeitlimit; offene Laeufe gegen das JETZT gueltige Limit
    #    neu bestimmen (das Limit kann eben erst entstanden sein, dann kommen alte Timeouts
    #    dazu -> ohne --max-tasks wird hier nichts abgeschnitten)
    rest = [t for t in tasks if t["point"] != base_id and not is_done(t, limit)]
    if budget is not None:
        rest = rest[:max(budget, 0)]
    for t in rest:
        t["time_limit_s"] = limit
        if is_done(t, stored):          # war am Anfang als fertig gezaehlt -> Zaehler korrigieren
            prog.n_done -= 1
            done_per_point[t["point"]] -= 1
    if rest:
        run_pool(rest, workers, prog, done_per_point)
    prog.say("Fertig. Auswertung: analyze_sensitivity.py")


def record_code_state(out_dir: Path, prog: Progress) -> None:
    """Code-Stand im Ergebnisordner festhalten (code_stand.json vom ersten Start) und warnen,
    wenn er von code_referenz.json oder vom ersten Start abweicht. Bricht nie ab."""
    from sa_core import CODE_REFERENCE_FILE, code_fingerprint, compare_fingerprint
    try:
        fp = code_fingerprint()
    except Exception as exc:                       # nur Dokumentation, darf den Lauf nicht stoppen
        prog.say(f"WARNUNG: Code-Stand nicht bestimmbar ({exc!r}).")
        return

    def n_diff(ref: dict) -> int:
        return sum(len(v) for v in compare_fingerprint(ref, fp).values())

    ref_file = HERE / CODE_REFERENCE_FILE
    if not ref_file.exists():
        prog.say(f"WARNUNG: {CODE_REFERENCE_FILE} fehlt, Code-Stand gegen den Laptop nicht geprueft.")
    elif n_diff(json.loads(ref_file.read_text(encoding="utf-8"))["dateien"]):
        prog.say(f"WARNUNG: Simulationscode weicht von {CODE_REFERENCE_FILE} ab "
                 f"({n_diff(json.loads(ref_file.read_text(encoding='utf-8'))['dateien'])} Dateien). "
                 f"Details: --check. Ergebnisse sind dann nicht mit der Kalibrierung vergleichbar.")
    state = {"zeit": datetime.now().isoformat(timespec="seconds"), "dateien": fp}
    first = out_dir / "code_stand.json"
    if not first.exists():
        _write_text_atomic(first, json.dumps(state, indent=1))
    elif n_diff(json.loads(first.read_text(encoding="utf-8"))["dateien"]):
        later = out_dir / f"code_stand_{datetime.now():%Y%m%d_%H%M%S}.json"
        _write_text_atomic(later, json.dumps(state, indent=1))
        prog.say(f"WARNUNG: Der Simulationscode hat sich seit dem ersten Start geaendert "
                 f"(neuer Stand in {later.name}). Alte und neue Laeufe passen dann nicht zusammen.")


def _write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name("_part_" + path.name)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def stored_time_limit(out_dir: Path, factor: float) -> float:
    """Das Limit, das mit dem AKTUELLEN Faktor gilt (0, solange noch keins bestimmt ist)."""
    f = out_dir / "time_limit.json"
    if factor <= 0 or not f.exists():
        return 0.0
    return factor * float(json.loads(f.read_text(encoding="utf-8"))["base_median_wall_s"])


def resolve_time_limit(cfg: dict, out_dir: Path, tasks: list[dict], base_id: int, prog: Progress) -> float:
    """Zeitlimit je Lauf = TIME_LIMIT_FACTOR x Median-Laufzeit des Basispunkts auf DIESEM Rechner.

    Festgehalten in time_limit.json (Neustart benutzt dasselbe Limit), aber erst, wenn
    ALLE Basispunkt-Laeufe fertig sind. Wird TIME_LIMIT_FACTOR spaeter geaendert, wird das
    Limit mit dem neuen Faktor und dem gespeicherten Median neu gesetzt (und gemeldet).
    TIME_LIMIT_FACTOR = 0 -> kein Limit.
    """
    factor = float(cfg["TIME_LIMIT_FACTOR"])
    if factor <= 0:
        return 0.0
    f = out_dir / "time_limit.json"
    if f.exists():
        info = json.loads(f.read_text(encoding="utf-8"))
        if abs(float(info["factor"]) - factor) > 1e-12:
            info["time_limit_s"] = factor * float(info["base_median_wall_s"])
            prog.say(f"TIME_LIMIT_FACTOR geaendert ({info['factor']:g} -> {factor:g}): "
                     f"Zeitlimit jetzt {info['time_limit_s']:.0f} s")
            info["factor"] = factor
            _write_text_atomic(f, json.dumps(info, indent=1))
        return float(info["time_limit_s"])
    base_tasks = [t for t in tasks if t["point"] == base_id]
    metas = [read_meta(Path(t["out_path"])) for t in base_tasks]
    walls = [float(m["wall_s"]) for m in metas if m is not None and m["status"] == "ok"]
    if not walls:
        prog.say("WARNUNG: kein fertiger Basispunkt-Lauf -> in diesem Durchgang kein Zeitlimit")
        return 0.0
    med = float(np.median(walls))
    limit = factor * med
    if all(m is not None for m in metas):
        _write_text_atomic(f, json.dumps({"time_limit_s": limit, "factor": factor, "base_median_wall_s": med,
                                          "n_base_runs": len(walls)}, indent=1))
        prog.say(f"Zeitlimit je Lauf: {factor:g} x {med:.0f} s (Median Basispunkt, {len(walls)} Laeufe) "
                 f"= {limit:.0f} s (gespeichert in time_limit.json)")
    else:
        prog.say(f"Zeitlimit vorlaeufig {limit:.0f} s (Basispunkt noch nicht komplett, nicht gespeichert)")
    return limit


def check(cfg: dict, make_reference: bool = False):
    """Schnelltest der Umgebung (Sekunden): alles da, was die Studie braucht, und derselbe
    Simulationscode wie auf dem Laptop (code_referenz.json)?"""
    import importlib
    for mod in ("numpy", "pandas", "matplotlib", "openpyxl", "SALib", "wmcpbe",
                "pbe_optimizer.adapter.wet_granulation_adapter"):
        m = importlib.import_module(mod)
        print(f"  import {mod:48s} ok  {getattr(m, '__version__', '')}")
    X = design_matrix(cfg)
    print(f"  Design: {X.shape[0]} Punkte x {X.shape[1]} Parameter, Basispunkt {base_row(cfg)}")
    for row in (X[0], X[-1], base_row(cfg)):
        p, vals = to_params(cfg, row)
        print("   ", {k: f"{v:.4g}" for k, v in vals.items()})
    from sa_core import build_solver
    params, _ = to_params(cfg, base_row(cfg))
    t0 = time.perf_counter()
    s = build_solver(params, np.asarray(cfg["T_SAVE"], dtype=float), replicate_seeds(int(params["MC_SEED"]), 1)[0])
    s.solve(maxiter=200, max_particles=int(params["MAX_PARTICLES"]))
    run = {"t_nach_200_events": float(s._elapsed), "n_comp": int(s.a_tot)}
    print(f"  Solver gebaut + 200 Events in {time.perf_counter() - t0:.1f} s (inkl. numba-Laden), "
          f"t = {run['t_nach_200_events']:.6g} s, n_comp = {run['n_comp']}")

    # Gleicher Simulationscode wie auf dem Laptop, mit dem kalibriert wurde?
    from sa_core import CODE_REFERENCE_FILE, code_fingerprint, compare_fingerprint
    fp = code_fingerprint()
    ref_file = HERE / CODE_REFERENCE_FILE
    if make_reference:
        _write_text_atomic(ref_file, json.dumps({"erstellt": datetime.now().isoformat(timespec="seconds"),
                                                 "testlauf": run, "dateien": fp}, indent=1))
        print(f"  Referenz geschrieben: {ref_file.name} ({len(fp)} Dateien)")
        print("CHECK OK")
        return
    if not ref_file.exists():
        print(f"CHECK NICHT OK: {CODE_REFERENCE_FILE} fehlt (gehoert in den Ordner studies/sensitivity, "
              f"auf dem Laptop mit --check --make-reference erzeugt).")
        return
    ref = json.loads(ref_file.read_text(encoding="utf-8"))
    diff = compare_fingerprint(ref["dateien"], fp)
    if any(diff.values()):
        print("CHECK NICHT OK: Der Simulationscode hier ist NICHT derselbe wie beim Kalibrieren:")
        for what, names in diff.items():
            if names:
                print(f"  {what} ({len(names)}): " + ", ".join(names[:12]) + (" ..." if len(names) > 12 else ""))
        print("  -> mcpbe/ und pbe-optimizer/ auf den Stand des Laptops bringen (README, Schritt 0).")
        return
    print(f"  Code-Stand: identisch mit der Referenz ({len(fp)} Dateien, erstellt {ref['erstellt']})")
    r = ref["testlauf"]
    same = r == run
    print(f"  Testlauf: {'gleich' if same else 'ANDERS'} wie auf dem Laptop "
          f"(dort t = {r['t_nach_200_events']:.6g} s, n_comp = {r['n_comp']})"
          + ("" if same else " -> bei gleichem Code meist andere numpy/numba-Version; Ergebnisse "
                             "sind dann statistisch, aber nicht bitgleich vergleichbar"))
    print("CHECK OK")


def benchmark(cfg: dict, out_dir: Path, k: int, workers: int):
    """K Replikate des Basispunkts parallel -> Sekunden je Lauf auf DIESEM Rechner."""
    bdir = out_dir / "benchmark"
    params, values = to_params(cfg, base_row(cfg))
    seeds = replicate_seeds(int(params["MC_SEED"]), k)
    tasks = [{"params": params, "sa_values": values, "point": 0, "rep": r, "n_rep_point": k,
              "seed": seeds[r], "t_save": np.asarray(cfg["T_SAVE"], dtype=float),
              "t_full": list(cfg["T_FULL"]), "bin_dt": float(cfg["BIN_DT"]),
              "probe_dt": float(cfg["PROBE_DT"]), "time_limit_s": 0.0, "study": "benchmark",
              "out_path": str(bdir / f"bench_r{r:02d}.npz")} for r in range(k)]
    bdir.mkdir(parents=True, exist_ok=True)
    prog = Progress(bdir, k, 0, workers)
    t0 = time.perf_counter()
    run_pool(tasks, workers, prog, defaultdict(int))
    wall = time.perf_counter() - t0
    per_run = np.mean(prog.walls)
    X = design_matrix(cfg)
    n_tasks = len(X) * int(cfg["N_REPLICATES"]) + int(cfg["BASE_REPLICATES"])
    # Durchsatz: k Laeufe in 'wall' Sekunden mit 'workers' Prozessen. Ueber den ganzen
    # Parameterraum dauert ein Lauf im Mittel DESIGN_COST_FACTOR x so lang wie am Basispunkt
    # (Laptop-Mini-Studie der Gesamt-Config: 1.17).
    thru = k / wall
    hours = n_tasks / thru / 3600 * DESIGN_COST_FACTOR
    prog.say(f"Benchmark: {k} Laeufe, {workers} parallel, {wall:.0f} s gesamt, {per_run:.0f} s je Lauf "
             f"-> Durchsatz {thru * 3600:.0f} Laeufe/h am Basispunkt.")
    prog.say(f"Hochrechnung ganze Studie ({n_tasks} Laeufe, x{DESIGN_COST_FACTOR} fuer den Parameterraum): "
             f"~{hours:.1f} h. Die erste Haelfte der Bloecke (= N/2, schon auswertbar) nach ~{hours / 2:.1f} h.")


def benchmark_workers(cfg: dict, out_dir: Path, worker_list: list[int], n_runs: int = 36) -> int:
    """Durchsatz fuer mehrere Prozesszahlen (z. B. 6 Kerne / 12 Threads: 6, 9, 12) mit
    DENSELBEN n_runs Laeufen des Basispunkts. 36 = 6x6 = 9x4 = 12x3, also volle Wellen.
    Gibt die empfohlene Prozesszahl zurueck (die kleinste innerhalb von 5 % des Besten)."""
    params, values = to_params(cfg, base_row(cfg))
    seeds = replicate_seeds(int(params["MC_SEED"]), n_runs)

    def tasks_for(tag: str, k: int) -> list[dict]:
        bdir = out_dir / "benchmark" / tag
        bdir.mkdir(parents=True, exist_ok=True)
        return [{"params": params, "sa_values": values, "point": 0, "rep": r, "n_rep_point": k,
                 "seed": seeds[r], "t_save": np.asarray(cfg["T_SAVE"], dtype=float),
                 "t_full": list(cfg["T_FULL"]), "bin_dt": float(cfg["BIN_DT"]),
                 "probe_dt": float(cfg["PROBE_DT"]), "time_limit_s": 0.0, "study": "benchmark",
                 "out_path": str(bdir / f"bench_r{r:02d}.npz")} for r in range(k)]

    # Aufwaermen (numba-Cache fuellen), damit die erste Messung keine Kompilierzeit traegt
    warm_tasks = tasks_for("aufwaermen", 1)
    warm = Progress(out_dir / "benchmark" / "aufwaermen", 1, 0, 1)
    warm.say("Benchmark: Aufwaermlauf (1 Lauf) ...")
    run_pool(warm_tasks, 1, warm, defaultdict(int))
    rows = []
    for w in worker_list:
        tasks = tasks_for(f"w{w:02d}", n_runs)
        prog = Progress(out_dir / "benchmark" / f"w{w:02d}", n_runs, 0, w)
        prog.say(f"Benchmark mit {w} Prozessen: {n_runs} Laeufe ...")
        t0 = time.perf_counter()
        run_pool(tasks, w, prog, defaultdict(int))
        wall = time.perf_counter() - t0
        rows.append((w, wall, n_runs / wall * 3600, float(np.mean(prog.walls))))
    best = max(r[2] for r in rows)
    pick = min(r[0] for r in rows if r[2] >= 0.95 * best)
    X = design_matrix(cfg)
    n_tasks = len(X) * int(cfg["N_REPLICATES"]) + int(cfg["BASE_REPLICATES"])
    lines = ["", "Prozesse | Wandzeit | Laeufe/h | s je Lauf", "---------|----------|----------|----------"]
    for w, wall, thru, per in rows:
        lines.append(f"{w:8d} | {wall:6.0f} s | {thru:8.0f} | {per:7.1f}" + ("   <- Empfehlung" if w == pick else ""))
    thru_pick = next(r[2] for r in rows if r[0] == pick)
    hours = n_tasks / thru_pick * DESIGN_COST_FACTOR
    lines += ["", f"Empfehlung: WORKERS = {pick} in sa_gemeinsam.py eintragen "
                  f"(kleinste Prozesszahl innerhalb von 5 % des besten Durchsatzes).",
              f"Hochrechnung ganze Studie ({n_tasks} Laeufe, x{DESIGN_COST_FACTOR} fuer den Parameterraum): "
              f"~{hours:.1f} h, erste Haelfte (N/2, schon auswertbar) ~{hours / 2:.1f} h."]
    text = "\n".join(lines)
    print(text, flush=True)
    _write_text_atomic(out_dir / "benchmark" / "ergebnis.txt", text + "\n")
    return pick


# Mittlere Laufzeit ueber den Parameterraum / Laufzeit am Basispunkt. Stichprobe aus 128
# Punkten der endgueltigen Config (18 Parameter, Laptop): Mittel 1.9x, Median 1.3x.
DESIGN_COST_FACTOR = 1.9


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else SPYDER_ARGS)
