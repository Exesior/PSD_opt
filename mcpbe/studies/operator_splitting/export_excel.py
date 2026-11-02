"""Alle Ergebnisse der Splitting-Diagnose als Excel exportieren (verlustfrei).

Liest jeden Ordner unter results/ (per_iteration.npz bzw. .csv, summary.json,
pop.json) sowie results/rate_scaling.csv und schreibt in den Zielordner:

    Uebersicht.xlsx
        README          Erklaerung aller Spalten, Einheiten, Hinweise
        Laeufe          eine Zeile pro Lauf (Kennzahlen, Event-Raten, Flips, ...)
        Rate_Skalierung Start-Event-Rate gegen f = Vc/V_mixer und N_comp
        Zeitfenster     10 Zeitfenster je Lauf (Langformat, Spalte Lauf)
        Parameter       alle Simulationsparameter je Lauf (Zeilen = Parameter)
    Laeufe/<Lauf>.xlsx
        pro_Iteration   eine Zeile pro Iteration, alle Kennzahlen + R = 1/dt
        Zusammenfassung summary.json flach (Schluessel / Wert)

Genauigkeit: openpyxl schreibt Floats sonst mit "%.16g" (letzte Stelle kann
kippen). Hier wird im eigenen Prozess repr() eingesetzt und JEDE Zahl jeder
Tabelle nach dem Schreiben bitweise gegen die Quelle geprueft.

Aufruf (aus dem Ordner `mcpbe`):
    python -m studies.operator_splitting.export_excel "C:\\Studium\\Master\\MA\\Daten\\Operator_Splitting_Diagnose"
"""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# Laeufe, die VOR der theta-Korrektur (v2) gerechnet und nicht wiederholt wurden.
_THETA_V1 = ("granulation_", "wet_granulation_production_n300_", "wet_granulation_best_1070_n50000000_")

COLUMN_DOC = [
    ("iter", "-", "Iterationsnummer (0-basiert); eine Iteration = ein MC-Versuch + ein kontinuierlicher Schritt"),
    ("t", "s", "Simulationszeit des Events t_n"),
    ("dt", "s", "Intervall t_n - t_(n-1), ueber das der kontinuierliche Schritt nachgezogen wird"),
    ("R_Events_pro_s", "1/s", "Event-Rate = 1/dt (Rechen-Events = MC-Versuche pro Sekunde)"),
    ("event", "-", "0 = kein Event, 1 = Agglomerationsversuch, 2 = Bruch"),
    ("dW_agg", "-", "Paketgroesse des Agglomerations-Events (0 = abgelehnt)"),
    ("dW_break", "-", "Paketgroesse des Bruch-Events"),
    ("a_tot", "-", "Anzahl Rechenpartikel"),
    ("theta_max", "-", "max_i k_eff,i * dt, aus der Porositaetsaenderung rueckgerechnet (Kennzahl 1)"),
    ("theta_mean_m", "-", "massengewichteter Mittelwert von k_eff,i * dt"),
    ("k_eff_max", "1/s", "theta_max / dt"),
    ("deps_mean_m / deps_max", "-", "|Aenderung Porositaet| im Schritt: massengewichtet / Maximum"),
    ("dS_mean_m / dS_max", "-", "|Aenderung Saettigung| im Schritt (Internalisierung + Kompression)"),
    ("dX_rel_mean_m / dX_rel_max", "-", "|relative Aenderung Durchmesser| im Schritt"),
    ("A_agg_pre / A_agg_post", "-", "Summe Agglomerations-Propensity vor / nach dem kontinuierlichen Schritt"),
    ("A_break_pre / A_break_post", "1/s", "Summe Bruchrate vor / nach dem kontinuierlichen Schritt"),
    ("rate_pre / rate_post", "1/s", "Gesamt-Eventrate (mix) vor / nach dem kontinuierlichen Schritt"),
    ("p_agg_pre / p_agg_post", "-", "Wahrscheinlichkeit, dass Agglomeration statt Bruch gewuerfelt wird"),
    ("delta_agg / delta_break / delta_rate", "-", "relative Drift (post - pre) / pre (Kennzahl 2), mit Vorzeichen"),
    ("dp_agg", "-", "p_agg_post - p_agg_pre"),
    ("acc_checked", "-", "1 = in dieser Iteration wurde ein Stokes-Akzeptanztest ausgewertet"),
    ("acc_old / acc_new", "-", "Entscheidung mit Zustand t_(n-1) (real) / mit ueber dt vorgezogenem Zustand; -1 = kein Test"),
    ("flip", "-", "1 = Entscheidung waere anders ausgefallen (Kennzahl 3)"),
    ("shim_mismatch", "-", "Gegenprobe dt = 0; muss immer 0 sein"),
    ("thetaS_max", "-", "v3: max_i theta_S,i = ln(d_alt/d_neu) der Fluessigkeitsaufnahme im Intervall, d = min(L, V_p) - l_int (nur Internalisierung, ohne Kompression); 50 = Gleichgewicht im Intervall erreicht"),
    ("kS_max", "1/s", "v3: thetaS_max / dt, schnellste Fluessigkeitsaufnahme eines Partikels"),
    ("kS_mean_m / kS_mean_br", "1/s", "v3: k_S massengewichtet / nach Bruchrate (vor dem Schritt) gewichtet, nur Partikel, die noch Fluessigkeit aufnehmen"),
    ("n_S_active / n_S_capped", "-", "v3: Partikel, die noch Fluessigkeit aufnehmen / davon im Intervall ins Gleichgewicht gelaufen"),
    ("x_max", "m", "v3: groesster Partikeldurchmesser zu Beginn des Intervalls"),
    ("dbr_part_min / dbr_part_max", "-", "v3: kleinste / groesste relative Bruchraten-Aenderung EINES Partikels (S_nach - S_vor) / S_vor"),
    ("dbr_part_min_share / dbr_part_max_share", "-", "v3: Anteil dieses Partikels an der Gesamtbruchrate vor dem Schritt (Wahrscheinlichkeit, als Bruchpartikel gewaehlt zu werden)"),
    ("n_br_from_zero", "-", "v3: Partikel mit Bruchrate 0 vor und > 0 nach dem Schritt (relative Aenderung unendlich)"),
]

README_NOTES = [
    "Splitting-Diagnose WMCPBE (Operator Splitting: MC-Event, danach Internalisierung + Kompression ueber dt).",
    "Code: mcpbe/studies/operator_splitting/ (diagnose_splitting.py, production_scenario.py, scan_ncomp.py, rate_scaling.py, export_excel.py).",
    "Die Diagnose veraendert den Lauf nicht: Endzustand bitgleich zu einem Lauf ohne Diagnose (Spalte bitgleich_ohne_Diagnose, wo geprueft).",
    "Produktions-Szenario = Kopie von opt_config_wet_granulation_full.py + wet_granulation_adapter._build_solver (bitgleich zum Adapter geprueft).",
    "best_1070 = beste Parameter der Optimierung (Kandidat 1070, Score 0.028586); Werte aus Excel-Anzeige, EKE_CORR_BETA und BREAK_P1 nur 6 Stellen.",
    "N_comp-Scan (best_1070_ncompN): Paketgewicht W fest (= Config, 3.19), Vc und Flussrate proportional zu N_comp -> Event-Rate ~ N_comp; Dichten unveraendert.",
    "Seeds = die ersten 3 Replikat-Seeds des Adapters: SeedSequence(7).generate_state(10)[0..2].",
    "theta_Schaetzer v1 (aeltere Laeufe): Partikel direkt an eps_min (Abstand ~1e-10) konnten theta durch Rundung ueberhoehen. v2 laesst Abstand <= 1e-8 aus.",
    "Delta-Werte enthalten NUR den kontinuierlichen Schritt (Internalisierung + Kompression), nicht die Nukleation.",
    "Diagnose v3 (Spalte Diagnose_Version): zusaetzlich k_S je Partikel, x_max, Bruchraten-Aenderung je Partikel. Aeltere Laeufe haben diese Spalten nicht (leer).",
    "Alle Zahlen sind verlustfrei (repr) geschrieben und bitweise zurueckgeprueft; Excel ZEIGT nur 15 Stellen an.",
    "Die bench-Laeufe granulation_* haben kein pop.json (Szenarien aus mcpbe/tests/bench/scenarios.py).",
]


# ---------------------------------------------------------------------------
def _bits(x) -> bytes | None:
    if isinstance(x, (float, np.floating)):
        return struct.pack("<d", float(x))
    return None


def _write_exact(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    """XLSX mit repr()-Floats schreiben und jede Zahl bitweise zurueckpruefen."""
    import openpyxl.cell._writer as _xw
    original = _xw.safe_string
    _xw.safe_string = lambda v: repr(v) if isinstance(v, float) and math.isfinite(v) else original(v)
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            for name, df in sheets.items():
                df.to_excel(xw, sheet_name=name, index=False)
    finally:
        _xw.safe_string = original

    back = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    for name, df in sheets.items():
        b = back[name]
        if b.shape != df.shape:
            raise RuntimeError(f"{path.name}/{name}: Form {b.shape} != {df.shape}")
        for col in df.columns:
            src = df[col].to_numpy()
            dst = b[col].to_numpy()
            for x, y in zip(src, dst):
                if isinstance(x, (float, np.floating)):
                    if math.isnan(x):
                        if not (isinstance(y, float) and math.isnan(y)):
                            raise RuntimeError(f"{path.name}/{name}/{col}: NaN nicht erhalten ({y!r})")
                    elif _bits(x) != _bits(float(y)):
                        raise RuntimeError(f"{path.name}/{name}/{col}: {x!r} -> {y!r}")


def _flatten(d: Any, prefix: str = "") -> List[tuple]:
    out = []
    if isinstance(d, dict):
        for k, v in d.items():
            out += _flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out += _flatten(v, f"{prefix}[{i}]")
    else:
        out.append((prefix, d))
    return out


def _load_iterations(folder: Path) -> pd.DataFrame:
    npz = folder / "per_iteration.npz"
    if npz.exists():
        d = np.load(npz)
        df = pd.DataFrame({k: d[k] for k in d.files})
    else:  # Laeufe vor dem npz-Format: event als Text
        df = pd.read_csv(folder / "per_iteration.csv", float_precision="round_trip")
        if df["event"].dtype == object:
            df["event"] = df["event"].fillna("").map({"": 0, "agg": 1, "break": 2}).astype(float)
        if "shim_mismatch" not in df:
            df["shim_mismatch"] = np.nan
    df = df.astype(float)
    with np.errstate(divide="ignore"):
        df.insert(df.columns.get_loc("dt") + 1, "R_Events_pro_s", 1.0 / df["dt"].to_numpy())
    return df


def _run_row(folder: Path, df: pd.DataFrame, sm: Dict[str, Any], pop: Dict[str, Any] | None) -> Dict[str, Any]:
    run = sm.get("run", {})
    R = df["R_Events_pro_s"].to_numpy()
    n_edge = max(1, min(50, R.size // 20))
    ident = run.get("identical_to_undiagnosed")
    acc = sm["acceptance_flips"]
    theta_v = run.get("theta_estimator") or ("v1" if folder.name.startswith(_THETA_V1) else "v2")
    row = {
        "Lauf": folder.name,
        "Szenario": run.get("scenario", folder.name),
        "Seed": run.get("seed"),
        "N_comp_Start": pop.get("INITIAL_PARTICLES") if pop else None,
        "f_Vc_durch_Vmixer": (pop["CONTROL_VOLUME"] / 1e-3) if pop else None,
        "Vc_m3": pop.get("CONTROL_VOLUME") if pop else None,
        "W_init": pop.get("INITIAL_WEIGHT") if pop else None,
        "maxiter": run.get("maxiter"),
        "Iterationen": sm["n_iterations"],
        "t_Ende_s": sm["t_end_reached"],
        "Agg_Versuche": sm["n_agg_events"],
        "Bruch_Events": sm["n_break_events"],
        "R_Start_Median_pro_s": float(np.median(R[:n_edge])),
        "R_Ende_Median_pro_s": float(np.median(R[-n_edge:])),
        "R_min_pro_s": float(np.min(R)),
        "dt_max_s": sm["dt"]["max"],
    }
    for key in ("theta_max", "deps_max", "dS_max", "dX_rel_max", "abs_delta_agg",
                "abs_delta_break", "abs_delta_rate", "abs_dp_agg",
                "thetaS_max", "kS_max", "abs_dbr_part_min", "x_max"):
        q = sm.get(key, {})   # v3-Kennzahlen fehlen in aelteren Laeufen
        for stat in ("median", "p90", "p99", "max"):
            row[f"{key}_{stat}"] = q.get(stat)
    row.update({
        "theta_Schaetzer": theta_v,
        "Diagnose_Version": run.get("diag_version", "v2"),
        "Akz_Tests": acc["n_checked"],
        "Akz_akzeptiert": acc["n_accepted"],
        "Flips_gesamt": acc["flips_total"],
        "Flips_akz_zu_abg": acc["accepted_to_rejected"],
        "Flips_abg_zu_akz": acc["rejected_to_accepted"],
        "sqrt_N_akz": acc["sqrt_n_accepted"],
        "Flips_durch_sqrtN": acc["flips_over_sqrt_n"],
        "Gegenprobe_dt0_Abweichungen": acc.get("shim_mismatches_dt0"),
        "erw_Ereignistyp_Flips": sm["event_type_flips_expected"],
        "Zeitversatz_abs_s": sm["clock_shift_estimate"]["abs_s"],
        "Zeitversatz_signiert_s": sm["clock_shift_estimate"]["signed_s"],
        "bitgleich_ohne_Diagnose": (None if ident is None else bool(all(ident.values()))),
        "Wandzeit_mit_Diagnose_s": run.get("wall_s"),
        "Wandzeit_ohne_Diagnose_s": run.get("wall_ref_s"),
        "MAX_PARTICLES_erreicht": run.get("hit_particle_limit"),
    })
    for tol, d in sm["delta_rate_thresholds"].items():
        row[f"Anteil_Iter_abs_delta_rate_gt_{tol}"] = d["frac_iterations"]
    return row


def _cell(v):
    """Zahl bleibt Zahl (int/float), alles andere (bool, None, Text) wird Text."""
    if isinstance(v, bool) or v is None:
        return "" if v is None else str(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return v
    return str(v)


def _excel_safe(df: pd.DataFrame) -> pd.DataFrame:
    """None -> NaN fuer numerische Spalten, bool/None-Spalten als Text."""
    out = df.copy()
    for c in out.columns:
        col = out[c]
        if col.map(lambda v: isinstance(v, bool)).any():
            out[c] = col.map(lambda v: "" if v is None else str(v))
        elif col.map(lambda v: isinstance(v, (int, float, np.integer, np.floating)) or v is None).all():
            out[c] = pd.to_numeric(col, errors="coerce").astype(float)
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    target = Path(argv[0])
    (target / "Laeufe").mkdir(parents=True, exist_ok=True)

    folders = sorted(p for p in RESULTS.iterdir() if p.is_dir() and (p / "summary.json").exists())
    rows, bins, params = [], [], {}
    for folder in folders:
        sm = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        pop = json.loads((folder / "pop.json").read_text(encoding="utf-8")) if (folder / "pop.json").exists() else None
        if pop is None and folder.name.startswith("wet_granulation_"):
            pop = _pop_for(folder.name)
        df = _load_iterations(folder)
        rows.append(_run_row(folder, df, sm, pop))
        for b in sm["time_bins"]:
            bins.append({"Lauf": folder.name, **b})
        if pop is not None:
            params[folder.name] = pop

        flat = pd.DataFrame(_flatten(sm), columns=["Schluessel", "Wert"])
        flat["Wert"] = flat["Wert"].map(_cell)
        _write_exact(target / "Laeufe" / f"{folder.name}.xlsx",
                     {"pro_Iteration": df, "Zusammenfassung": flat})
        print(f"  {folder.name}: {len(df)} Iterationen", flush=True)

    readme = pd.DataFrame(
        [("Hinweis", "", n) for n in README_NOTES]
        + [("Spalte (Laeufe/*.xlsx, pro_Iteration)", u, f"{c}: {t}") for c, u, t in COLUMN_DOC],
        columns=["Art", "Einheit", "Text"])
    sheets = {
        "README": readme,
        "Laeufe": _excel_safe(pd.DataFrame(rows)),
        "Zeitfenster": _excel_safe(pd.DataFrame(bins)),
    }
    rs = RESULTS / "rate_scaling.csv"
    if rs.exists():
        sheets["Rate_Skalierung"] = pd.read_csv(rs, float_precision="round_trip")
    if params:
        keys = sorted({k for p in params.values() for k in p})
        ptab = pd.DataFrame({"Parameter": keys})
        for name, p in params.items():
            ptab[name] = [_cell(p.get(k)) if k in p else "" for k in keys]
        sheets["Parameter"] = ptab
    _write_exact(target / "Uebersicht.xlsx", sheets)
    print(f"Geschrieben: {target} ({len(folders)} Laeufe), alle Zahlen bitweise geprueft")
    return 0


def _pop_for(name: str) -> Dict[str, Any] | None:
    """Parameter fuer Produktions-Laeufe, die vor pop.json gerechnet wurden."""
    from . import production_scenario as prod
    if name.startswith(prod.NAME_BEST_1070 + "_n"):
        return prod.POP_BEST_1070
    if name.startswith(prod.NAME + "_n"):
        return prod.POP
    return None


if __name__ == "__main__":
    raise SystemExit(main())
