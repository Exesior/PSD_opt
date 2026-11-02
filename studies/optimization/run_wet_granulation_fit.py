"""
Runner fuer den AJ30-Nassgranulations-Fit.

Startet die Ray-Tune-Optimierung mit
    Config:  studies/optimization/opt_config_wet_granulation_full.py  (Standard; andere per --config <Datei>)
    Daten:   DATA_FILE aus der Config, passend zum Schalter COST_MODE:
               q3_sieve_mae -> studies/optimization/data/AJ_Q3_sieve.xlsx (Blatt "Q3_sieve")
               xq_mse       -> studies/optimization/data/AJ30_xQ.xlsx     (Blatt "x_Q3")
und schreibt Ergebnisse nach RESULT_DIR aus der Config (fehlt er: studies/optimization/wet_granulation_opt_results)
    studies/optimization/wet_granulation_opt_results/<N>.sqlite         (Warm-Start-Historie; alle
                                                           checkpoint_interval Kandidaten
                                                           eine Zwischen-SQLite)
    studies/optimization/wet_granulation_opt_results/best_params.json   (bester Parametersatz)
    studies/optimization/wet_granulation_ray/<Datenname>/               (Ray-Tune-Rohdaten)

Stellschrauben (Laufzeit, Zeitlimit, Checkpoints, max_reuse) stehen ALLE in der
Config. Auswertung als Tabelle (CSV + Excel, eine Zeile je Kandidat mit status,
Replikaten, Rechenzeit, Worker, Fehlergrund):
    python "studies/optimization/export_wet_granulation_results.py" studies/optimization/wet_granulation_opt_results

HINWEIS ZUR LAUFZEIT
-------------------
Eine einzelne Simulationsauswertung sind viele Millionen MC-Events. Diese
Optimierung ist fuer einen leistungsstarken Rechner gedacht -- NICHT auf dem
Laptop komplett laufen lassen. Zum reinen Testen der Kette:  --check --n-iter 1

Aufruf (Terminal):
    python studies/optimization/run_wet_granulation_fit.py                 # frischer Lauf, n_iter aus Config
    python studies/optimization/run_wet_granulation_fit.py --n-iter 500
    python studies/optimization/run_wet_granulation_fit.py --check         # + 1 Direktauswertung als Sanity-Check
    python studies/optimization/run_wet_granulation_fit.py --resume 200    # Historie 200.sqlite fortsetzen

Die drei Kostenfunktions-Laeufe (Eric, 2026-10-07) -- je ein eigener Job, unabhaengig startbar,
jeder schreibt in seinen eigenen Ergebnisordner (RESULT_DIR der Config):
    python studies/optimization/run_wet_granulation_fit.py --config "opt_config_wet_granulation_kf1_aj30.py"
    python studies/optimization/run_wet_granulation_fit.py --config "opt_config_wet_granulation_kf2_nosampling.py"
    python studies/optimization/run_wet_granulation_fit.py --config "opt_config_wet_granulation_kf3_nosampling_logw.py"
    python studies/optimization/run_wet_granulation_fit.py --config "opt_config_wet_granulation_kf4_nosampling_interp.py"   (KF4, 2026-10-14)
Vorher einmal die Datendateien erzeugen:  python studies/optimization/dataprep/prepare_kf_varianten.py
(KF4 zusaetzlich: python studies/optimization/dataprep/prepare_kf4_interp.py)
(die drei xlsx liegen bereits in studies/optimization/data/). Fortsetzen nach Abbruch: zusaetzlich --resume <N>.

Aufruf (Spyder, KEIN Terminal):
    Diese Datei oeffnen -> unten SPYDER_ARGS setzen -> F5 (Run file).
    Aus einem Terminal haben echte CLI-Argumente immer Vorrang.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

import ray

from pbe_optimizer.optimizer.opt_base import OptBase


TRIALS_DIR = Path(__file__).resolve().parent
# Standardwerte; DATA_FILE und (optional) RESULT_DIR kommen aus der gewaehlten Config.
RESULT_DIR_DEFAULT = TRIALS_DIR / "wet_granulation_opt_results"
CONFIG_DEFAULT = "opt_config_wet_granulation_full.py"

_HELPER_KEYS = {"actor_wait", "wait_time", "max_reuse", "__exp_paths", "__known_params"}

# ===========================================================================
#  SPYDER (kein Terminal): hier einstellen, dann F5 (Run file).
#  Wird NUR benutzt, wenn das Skript ohne Kommandozeilen-Argumente laeuft.
#  Aus einem Terminal haben echte CLI-Argumente immer Vorrang.
#     Smoke-Test (zuerst):        ["--check", "--n-iter", "2"]
#     Voller Lauf (CMA-ES):       []
#     Historie 500 fortsetzen:    ["--resume", "500"]
#     andere Config:              ["--config", "opt_config_wet_granulation_kf1_aj30.py"]
# ===========================================================================
SPYDER_ARGS: list[str] = []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=None,
                    help="Anzahl Optimierer-Iterationen (Default: aus der Config).")
    ap.add_argument("--check", action="store_true",
                    help="Vor der Optimierung eine einzelne Direktauswertung rechnen (teuer).")
    ap.add_argument("--resume", type=int, default=None, metavar="N",
                    help="Historie <N>.sqlite fortsetzen (n_iter_prev = N).")
    ap.add_argument("--config", default=CONFIG_DEFAULT,
                    help=f"Config-Datei in studies/optimization/ (Default: {CONFIG_DEFAULT}).")
    argv = sys.argv[1:] if len(sys.argv) > 1 else SPYDER_ARGS
    args = ap.parse_args(argv)
    if not sys.argv[1:]:
        print(f"[Spyder-Modus] keine CLI-Argumente -> SPYDER_ARGS = {SPYDER_ARGS}", flush=True)

    CONFIG_FILE = TRIALS_DIR / args.config
    print(f"Config: {CONFIG_FILE.name}", flush=True)
    if not CONFIG_FILE.is_file():
        raise FileNotFoundError(CONFIG_FILE)
    cfg_ns = runpy.run_path(str(CONFIG_FILE))
    DATA_FILE = Path(cfg_ns["DATA_FILE"])
    RESULT_DIR = Path(cfg_ns.get("RESULT_DIR", RESULT_DIR_DEFAULT))
    print(f"Daten : {DATA_FILE.name}", flush=True)
    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"{DATA_FILE} fehlt -- zuerst studies/optimization/dataprep/prepare_aj_q3_sieve.py "
            f"(COST_MODE q3_sieve_mae; KF1/KF2/KF3: danach prepare_kf_varianten.py) "
            f"bzw. prepare_aj30_xq.py (xq_mse) laufen lassen."
        )
    print(f"Ergebnis: {RESULT_DIR}", flush=True)
    RESULT_DIR.mkdir(exist_ok=True)

    # log_to_driver=True: Warnungen/Fehler der Ray-Worker landen in der Ausgabe
    # dieses Prozesses (= im Job-Log). Beim Lauf 1600 (False) waren sie unsichtbar.
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=True)
    try:
        opt = OptBase(config_path=str(CONFIG_FILE), data_path=str(DATA_FILE.parent))
        opt.core.result_dir = str(RESULT_DIR)

        if args.n_iter is not None:
            opt.core.n_iter = int(args.n_iter)

        if args.resume is not None:
            opt.core.n_iter_prev = int(args.resume)
            opt.core.resume_unfinished = True
        else:
            opt.core.n_iter_prev = 0
            opt.core.resume_unfinished = False

        if args.check:
            print("Direktauswertung (pop_params-Defaults) ...", flush=True)
            direct_delta, _ = opt.calc_PSD_delta(opt.pop_params, str(DATA_FILE))
            print(f"  direkte Kosten = {direct_delta:.6e}"
                  f"   (10.0 = Simulation gescheitert, 10.1 = Zeitlimit)   "
                  f"status = {opt.core.p.eval_info.get('status')}", flush=True)

        n_save = int(opt.core.n_iter) + int(getattr(opt.core, "n_iter_prev", 0))
        print(f"Starte Optimierung: n_iter={opt.core.n_iter}, "
              f"resume={opt.core.resume_unfinished} -> {RESULT_DIR / f'{n_save}.sqlite'}",
              flush=True)

        result = opt.find_opt_kernels(method="delta", data_names=DATA_FILE.name)

        best = {k: v for k, v in result["opt_params"].items() if k not in _HELPER_KEYS}
        out = {
            "opt_score": result["opt_score"],
            "data_name": result["file_path"],
            "n_iter": opt.core.n_iter,
            "n_iter_prev": getattr(opt.core, "n_iter_prev", 0),
            "best_params": best,
        }
        best_path = RESULT_DIR / "best_params.json"
        best_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

        print("\nFERTIG")
        print(f"  bester Score : {result['opt_score']:.6e}")
        print(f"  Parameter    : {best_path}")
        print(f"  Historie     : {RESULT_DIR / f'{n_save}.sqlite'}")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
