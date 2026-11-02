"""
Drei Kostenfunktionen zum Vergleich (Entscheidung Eric, 2026-10-06) -> drei Datendateien
fuer den wet_granulation-Adapter im Modus COST_MODE = "q3_sieve_mae".

Quelle: studies/optimization/data/AJ_Q3_sieve.xlsx (Blatt "Q3_sieve", erzeugt von prepare_aj_q3_sieve.py).
Die Werte werden 1:1 uebernommen (nur Auswahl, Gewichte, Mittelung).

    KF1  studies/optimization/data/KF1_AJ30_Q3_sieve.xlsx
         nur die beprobte Reihe AJ30 (S1..S9, 60..600 s), jede Probe Gewicht 1/9.
    KF2  studies/optimization/data/KF2_NoSampling_Q3_sieve.xlsx
         nur Versuche OHNE Probenahme:
           6 min  AJ40                                  -> 360 s, Gewicht 1/2
           10 min Mittelwert von AJ26, AJ29, AJ31       -> 600 s, Gewicht 1/2
         (Mittelwert je Masche, damit ein Fit "durch die Mitte" perfekt sein kann)
    KF3  studies/optimization/data/KF3_NoSampling_Q3_sieve_logw.xlsx
         Daten wie KF2, zusaetzlich Blatt "Maschengewicht": Gewicht je Masche aus der
         Trapezregel auf der log10-Groessenachse von 34 um (Primaerpartikel, dort
         Q3_sim = Q3_exp = 0) bis 2000 um:
             m_k ~ (log10 x_(k+1) - log10 x_(k-1)) / 2 ,  letzte Masche: halbes Intervall
         -> J ~ Flaeche zwischen simulierter und gemessener Q3-Kurve ueber log x.
         Rein aus der Maschengeometrie, NICHT aus Messwerten.

Genauigkeit: Excel wird mit repr() statt openpyxls "%.16g" geschrieben und bitweise
gegen die Quelle zurueckgeprueft (wie studies/optimization/export_wet_granulation_results.py).

Aufruf:
    python studies/optimization/dataprep/prepare_kf_varianten.py
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
SOURCE = DATA / "AJ_Q3_sieve.xlsx"
OUT_KF1 = DATA / "KF1_AJ30_Q3_sieve.xlsx"
OUT_KF2 = DATA / "KF2_NoSampling_Q3_sieve.xlsx"
OUT_KF3 = DATA / "KF3_NoSampling_Q3_sieve_logw.xlsx"

X_MIN_UM = 34.0            # Primaerpartikel-Durchmesser = INITIAL_DIAMETER
TEN_MIN = ("AJ26", "AJ29", "AJ31")
SIX_MIN = "AJ40"


def mesh_columns(frame: pd.DataFrame) -> list:
    out = []
    for c in frame.columns:
        try:
            float(c)
            out.append(c)
        except (TypeError, ValueError):
            pass
    return out


def log_trapezoid_weights(mesh_um: list[float], x_min_um: float) -> np.ndarray:
    """Trapezregel-Gewichte je Masche auf der log10-Achse, Summe 1."""
    lx = np.log10(np.r_[x_min_um, np.asarray(mesh_um, dtype=float)])
    w = np.empty(len(mesh_um))
    for k in range(1, len(lx)):
        upper = lx[k + 1] if k + 1 < len(lx) else lx[k]
        w[k - 1] = 0.5 * (upper - lx[k - 1])
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
        for col in frame.columns:
            if pd.api.types.is_float_dtype(frame[col]):
                for a, b in zip(frame[col], back[col]):
                    if struct.pack("<d", float(a)) != struct.pack("<d", float(b)):
                        raise AssertionError(f"{path.name}/{name}: Spalte {col} nicht bitgleich")


def main() -> None:
    src = pd.read_excel(SOURCE, sheet_name="Q3_sieve")
    mesh = mesh_columns(src)

    # KF1: AJ30 allein, je 1/9
    kf1 = src[src.Reihe == "AJ30"].sort_values("t_sim_s").reset_index(drop=True).copy()
    kf1["Gewicht"] = 1.0 / len(kf1)

    # KF2: AJ40 (6 min) + Mittel der 10-min-Versuche, je 1/2
    six = src[src.Reihe == SIX_MIN].copy()
    ten = src[src.Reihe.isin(TEN_MIN)]
    if len(six) != 1 or len(ten) != len(TEN_MIN) or ten.t_sim_s.nunique() != 1:
        raise ValueError("Quelle hat nicht die erwarteten Zeilen fuer AJ40 / AJ26+AJ29+AJ31.")
    mean_row = six.iloc[0].to_dict()       # Vorlage fuer Spalten/Typen, alles wird ueberschrieben
    mean_row.update({
        "Reihe": "+".join(TEN_MIN), "Gruppe": "10 min ohne Proben (Mittelwert)", "Probe": "Mittel",
        "t_sim_s": float(ten.t_sim_s.iloc[0]), "Sieb_g": float(ten.Sieb_g.mean()),
    })
    for c in mesh:
        mean_row[c] = float(ten[c].mean())
    kf2 = pd.concat([six, pd.DataFrame([mean_row])], ignore_index=True)[src.columns]
    kf2["Gewicht"] = 0.5

    # KF3: wie KF2 + Maschengewicht
    mesh_w = log_trapezoid_weights([float(c) for c in mesh], X_MIN_UM)
    mw = pd.DataFrame([dict(zip(mesh, mesh_w))])

    write_exact(OUT_KF1, {"Q3_sieve": kf1})
    write_exact(OUT_KF2, {"Q3_sieve": kf2})
    write_exact(OUT_KF3, {"Q3_sieve": kf2, "Maschengewicht": mw})

    pd.set_option("display.width", 200)
    print(f"Quelle: {SOURCE}")
    print(f"\nKF1 -> {OUT_KF1.name}\n{kf1[['Reihe', 'Probe', 't_sim_s', 'Gewicht'] + mesh].round(4).to_string(index=False)}")
    print(f"\nKF2 -> {OUT_KF2.name}\n{kf2[['Reihe', 'Probe', 't_sim_s', 'Gewicht'] + mesh].round(4).to_string(index=False)}")
    print(f"\nKF3 -> {OUT_KF3.name}  (Daten wie KF2)\n  Maschengewicht: "
          + ", ".join(f"{c} um: {w:.4f}" for c, w in zip(mesh, mesh_w)))
    print("\nalle drei Dateien bitweise zurueckgeprueft")


if __name__ == "__main__":
    main()
