"""
KF4 (Eric, 2026-10-14): wie KF2 (AJ40 6 min -> 360 s, Mittel AJ26/29/31 10 min -> 600 s,
je Gewicht 1/2), aber verglichen wird gegen die GESAMTE Messlinie zwischen den Siebmaschen,
nicht nur an den 6 Maschen.

Messlinie: Q3 zwischen zwei benachbarten Maschen LINEAR IN log10(x) interpoliert (so wie die
gestrichelte Verbindung in den Q3-Grafiken mit log-x-Achse aussieht). Nur zwischen der
kleinsten (100 um) und der groessten Masche (2000 um) -- KEINE Extrapolation.

Umsetzung ohne Code-Aenderung am Adapter:
  * Die Linie wird an vielen "virtuellen Maschen" abgetastet. Jedes Intervall zwischen zwei
    echten Maschen bekommt Stuetzstellen gleichmaessig in log10(x), Anzahl ~ Intervallbreite.
    Die echten Maschen sind immer Stuetzstellen (der Fit geht nicht an ihnen vorbei).
  * Blatt "Maschengewicht": Trapezregel auf log10(x) ueber diese Stuetzstellen, Summe 1.
  -> J = sum_i w_i * sum_k m_k |Q3_sim(x_k, t_i) - Q3_lin(x_k, t_i)|
       = mittlere Flaeche zwischen Sim-Kurve und Messlinie ueber log10(x) in [100, 2000] um.
  Q3_sim(x_k) ist wie immer der Massenanteil mit Durchmesser < x_k (exakt, keine Glaettung).

Quelle: studies/optimization/data/KF2_NoSampling_Q3_sieve.xlsx (von prepare_kf_varianten.py), Werte 1:1.
Ergebnis: studies/optimization/data/KF4_NoSampling_Q3_interp.xlsx  (Blaetter "Q3_sieve", "Maschengewicht")

Aufruf:
    python studies/optimization/dataprep/prepare_kf4_interp.py
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
SOURCE = DATA / "KF2_NoSampling_Q3_sieve.xlsx"
OUT = DATA / "KF4_NoSampling_Q3_interp.xlsx"
N_TOTAL = 100          # Anzahl Teilintervalle ueber 100..2000 um (Stuetzstellen = N_TOTAL + 1 ungefaehr)


def mesh_columns(frame: pd.DataFrame) -> list:
    out = []
    for c in frame.columns:
        try:
            float(c)
            out.append(c)
        except (TypeError, ValueError):
            pass
    return out


def grid_with_meshes(mesh_um: np.ndarray, n_total: int) -> np.ndarray:
    """Stuetzstellen gleichmaessig in log10(x) je Masche-Intervall, echte Maschen inklusive."""
    lx = np.log10(mesh_um)
    width = lx[-1] - lx[0]
    pts = [lx[0]]
    for a, b in zip(lx[:-1], lx[1:]):
        n = max(1, int(round(n_total * (b - a) / width)))
        pts.extend(np.linspace(a, b, n + 1)[1:])
    g = 10.0 ** np.asarray(pts)
    for m in mesh_um:                                # echte Maschen exakt (keine Rundungsdrift)
        g[np.argmin(np.abs(g - m))] = m
    return g


def trapezoid_weights_log(grid_um: np.ndarray) -> np.ndarray:
    lx = np.log10(grid_um)
    w = np.zeros_like(lx)
    d = np.diff(lx)
    w[:-1] += 0.5 * d
    w[1:] += 0.5 * d
    return w / w.sum()


def write_exact(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    import openpyxl.cell._writer as _xw
    original = _xw.safe_string
    _xw.safe_string = lambda v: repr(v) if isinstance(v, float) and math.isfinite(v) else original(v)
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            for name, frame in sheets.items():
                frame.to_excel(xw, sheet_name=name, index=False)
    finally:
        _xw.safe_string = original
    for name, frame in sheets.items():
        back = pd.read_excel(path, sheet_name=name)
        if [float(c) if isinstance(c, (int, float)) else c for c in back.columns] != \
           [float(c) if isinstance(c, (int, float)) else c for c in frame.columns]:
            raise AssertionError(f"{path.name}/{name}: Spaltenkoepfe nicht identisch zurueckgelesen")
        for a_col, b_col in zip(frame.columns, back.columns):
            if pd.api.types.is_float_dtype(frame[a_col]):
                for a, b in zip(frame[a_col], back[b_col]):
                    if struct.pack("<d", float(a)) != struct.pack("<d", float(b)):
                        raise AssertionError(f"{path.name}/{name}: Spalte {a_col} nicht bitgleich")


def main() -> None:
    src = pd.read_excel(SOURCE, sheet_name="Q3_sieve")
    mesh = mesh_columns(src)
    mesh_um = np.array([float(c) for c in mesh])
    grid = grid_with_meshes(mesh_um, N_TOTAL)
    if not (np.all(np.diff(grid) > 0) and np.isclose(grid[0], mesh_um[0]) and np.isclose(grid[-1], mesh_um[-1])):
        raise AssertionError("Gitter kaputt")
    if not all(np.any(grid == m) for m in mesh_um):
        raise AssertionError("echte Maschen fehlen im Gitter")

    info_cols = [c for c in src.columns if c not in mesh]
    out = src[info_cols].copy()
    q_interp = np.vstack([np.interp(np.log10(grid), np.log10(mesh_um), row[mesh].to_numpy(float))
                          for _, row in src.iterrows()])
    # an den echten Maschen exakt die Messwerte (keine Interpolationsrundung)
    for j, m in enumerate(mesh_um):
        k = int(np.flatnonzero(grid == m)[0])
        q_interp[:, k] = src[mesh[j]].to_numpy(float)
    headers = [float(g) for g in grid]
    out = pd.concat([out, pd.DataFrame(q_interp, columns=headers)], axis=1)
    mw = pd.DataFrame([dict(zip(headers, trapezoid_weights_log(grid)))])

    write_exact(OUT, {"Q3_sieve": out, "Maschengewicht": mw})
    print(f"Quelle: {SOURCE.name}  ->  {OUT}")
    print(f"{len(grid)} Stuetzstellen von {grid[0]:g} bis {grid[-1]:g} um, echte Maschen enthalten: "
          f"{[float(m) for m in mesh_um]}")
    print("Zeilen:", out[["Reihe", "t_sim_s", "Gewicht"]].to_dict("records"))
    print("bitweise zurueckgeprueft")


if __name__ == "__main__":
    main()
