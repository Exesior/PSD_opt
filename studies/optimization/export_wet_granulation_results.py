"""
SQLite-Ergebnis eines Nassgranulations-Fits -> Tabelle (CSV + Excel) + Kurzuebersicht.

Eine Zeile pro Kandidat in Reihenfolge der Fertigstellung (= id; Spalte "Reihenfolge"):
    Reihenfolge, id, finished_at, worker_pid, reuse_num, status, n_rep_ok,
    n_rep_total, eval_seconds, score, <je gefittetem Parameter eine Spalte>,
    failure_reason

status (siehe WetGranulationAdapter.solve):
    ok               alle Replikate fertig                       score = MSE
    partial          einige Replikate gescheitert, Mittel Rest   score = MSE
    timeout_partial  Zeitlimit, >= 1 Replikat fertig             score = MSE
    timeout          Zeitlimit, kein Replikat fertig             score = 10.1
    failed           alle Replikate gescheitert (z. B. MAX_PARTICLES)  score = 10
    setup_error      Programmfehler vor der Simulation           score = 10
Dateien von vor dieser Erweiterung (nur 4 Spalten) haben leere Zusatzspalten.

Genauigkeit: CSV mit repr(float) = verlustfrei (Master). Excel wird mit
repr() statt openpyxls "%.16g" geschrieben; beides wird bitweise zurueckgeprueft.

Aufruf:
    python "studies/optimization/export_wet_granulation_results.py" studies/optimization/wet_granulation_opt_results
    python "studies/optimization/export_wet_granulation_results.py" pfad/zu/1600.sqlite
Ohne Argument: studies/optimization/wet_granulation_opt_results (die SQLite mit dem groessten N).
Ausgabe: <name>_table.csv und <name>_table.xlsx neben der SQLite.
"""

from __future__ import annotations

import json
import math
import sqlite3
import struct
import sys
from pathlib import Path

import pandas as pd

TRIALS_DIR = Path(__file__).resolve().parent
DEFAULT_DIR = TRIALS_DIR / "wet_granulation_opt_results"

HELPER_KEYS = {"actor_wait", "wait_time", "max_reuse", "__exp_paths", "__known_params"}
EXTRA_COLUMNS = ["finished_at", "worker_pid", "reuse_num", "status", "n_rep_ok",
                 "n_rep_total", "eval_seconds"]


def resolve_db(arg: str | None) -> Path:
    """A .sqlite file, or the <N>.sqlite with the largest N in a directory."""
    path = Path(arg) if arg else DEFAULT_DIR
    if path.is_dir():
        files = [p for p in path.glob("*.sqlite") if p.stem.isdigit()]
        if not files:
            raise FileNotFoundError(f"keine <N>.sqlite in {path}")
        return max(files, key=lambda p: int(p.stem))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_results(db_path: Path) -> pd.DataFrame:
    """One row per candidate in id order (= completion order in files written by
    opt_base_ray since the status columns exist; a resumed run keeps its history first)."""
    con = sqlite3.connect(str(db_path))
    try:
        have = {row[1] for row in con.execute("PRAGMA table_info(warm_params)")}
        cols = ["id"] + EXTRA_COLUMNS + ["score", "param_json", "failure_reason"]
        select = ", ".join(c if c in have else f"NULL AS {c}" for c in cols)
        frame = pd.read_sql_query(f"SELECT {select} FROM warm_params ORDER BY id", con)
    finally:
        con.close()

    params = pd.DataFrame([{k: v for k, v in json.loads(p).items() if k not in HELPER_KEYS}
                           for p in frame.pop("param_json")])
    reason = frame.pop("failure_reason")
    frame = pd.concat([frame, params, reason], axis=1)
    frame.insert(0, "Reihenfolge", range(1, len(frame) + 1))
    return frame


def summary(frame: pd.DataFrame) -> str:
    lines = [f"{len(frame)} Kandidaten"]
    if frame["status"].notna().any():
        counts = frame["status"].fillna("(leer)").value_counts()
        lines.append("status: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    else:
        lines.append("status: (Datei ohne Status-Spalten) score 10: "
                     f"{int((frame['score'] == 10.0).sum())}")
    valid = frame[frame["score"] < 10.0]
    if len(valid):
        best = valid.loc[valid["score"].idxmin()]
        lines.append(f"bester Score {float(best['score'])!r} (id {int(best['id'])}, "
                     f"Reihenfolge {int(best['Reihenfolge'])})")
    return "\n".join(lines)


def _bits(x) -> bytes | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else struct.pack("<d", x)


def _float_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if pd.api.types.is_float_dtype(frame[c])]


def write_tables(frame: pd.DataFrame, db_path: Path) -> tuple[Path, Path]:
    csv_path = db_path.with_name(f"{db_path.stem}_table.csv")
    xlsx_path = db_path.with_name(f"{db_path.stem}_table.xlsx")
    frame.to_csv(csv_path, index=False, float_format=None)   # str(float) == repr: verlustfrei

    import openpyxl.cell._writer as _xw
    original = _xw.safe_string
    _xw.safe_string = lambda v: repr(v) if isinstance(v, float) and math.isfinite(v) else original(v)
    try:
        frame.to_excel(xlsx_path, index=False, engine="openpyxl")
    finally:
        _xw.safe_string = original

    # bitweiser Round-Trip-Check aller Float-Spalten
    back_csv = pd.read_csv(csv_path, float_precision="round_trip")
    back_xlsx = pd.read_excel(xlsx_path, engine="openpyxl")
    for name, back in (("CSV", back_csv), ("Excel", back_xlsx)):
        for col in _float_columns(frame):
            for a, b in zip(frame[col], back[col]):
                if _bits(a) != _bits(b):
                    raise AssertionError(f"{name}: Spalte {col} nicht bitgleich ({a!r} != {b!r})")
    return csv_path, xlsx_path


def main(argv: list[str]) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    db_path = resolve_db(argv[0] if argv else None)
    frame = load_results(db_path)
    csv_path, xlsx_path = write_tables(frame, db_path)
    print(f"SQLite : {db_path}")
    print(summary(frame))
    print(f"CSV    : {csv_path}")
    print(f"Excel  : {xlsx_path}   (bitweise gegen die SQLite geprueft)")


if __name__ == "__main__":
    main(sys.argv[1:])
