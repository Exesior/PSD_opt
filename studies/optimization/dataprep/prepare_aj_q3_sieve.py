"""
Aufbereitung der Siebdaten ALLER Versuchsreihen -> Eingabeblatt fuer den
wet_granulation-Adapter im Modus COST_MODE = "q3_sieve_mae".

Kostenfunktion (Entscheidung Eric, 2026-10-02)
----------------------------------------------
* Vergleich in der Q3(x)-Ansicht, senkrecht an den Siebmaschen
  100/250/500/800/1400/2000 um (keine Interpolation).
* Betrag statt Quadrat: 1 % falsch einsortierte Masse -> 1 % Einfluss.
* Jede Versuchsreihe zaehlt gleich. Eine Reihe mit mehreren Proben teilt ihr
  Gewicht gleichmaessig auf ihre Proben auf (AJ30: 9 Proben -> je 1/9 des
  Reihengewichts).

    J = sum_i  Gewicht_i * (1/6) * sum_k |Q3_sim(x_k, t_i) - Q3_i(x_k)|

  J ist dimensionslos: der mittlere Massenanteil, der je Masche auf der
  falschen Seite liegt. Die Gewichte stehen als Spalte im Blatt und koennen
  dort von Hand geaendert werden (der Adapter prueft nur: Summe = 1).

Quellen (Rohsiebungen, Blatt "PSD Boehmite using Sieving_Exp")
--------------------------------------------------------------
C:\\Studium\\Master\\MA\\Daten\\MVM-IKFT Eirich-Mischer Data\\
    260713_PuralTH130_Sampling\\260713_PSD_SieveTower_AJ30.xlsx          AJ30, S1..S9
    2607_PuralTH130_No Sampling\\Time_10min_1\\..._AJ26.xlsx              10 min, Vollcharge
    2607_PuralTH130_No Sampling\\Time_10min_2\\..._AJ29.xlsx              10 min, Vollcharge
    2607_PuralTH130_No Sampling\\Time_10min_3\\..._AJ31.xlsx              10 min, Vollcharge
    2607_PuralTH130_No Sampling\\Time_6min\\..._AJ40.xlsx                 6 min, Vollcharge
Nicht dabei: Time_40min (AJ42), Time_60min (AJ43) -- ausserhalb der Sim-Zeit (600 s).

Q3 je Messung = Masse unterhalb der Masche / gesamte gesiebte Masse (Summe aller
Siebe inkl. Auffangschale). Fuer AJ30 ist das identisch zum Blatt "Q3" der
AJ30-Datei (wird beim Lauf geprueft).

Zeitachse (wie prepare_aj30_q3.py): Sim-t = Prozess-t - 90 s.
    AJ30 S1..S9 -> 60..600 s ; 10-min-Versuche (690 s) -> 600 s ; 6 min (450 s) -> 360 s

Ergebnis
--------
studies/optimization/data/AJ_Q3_sieve.xlsx, Blatt "Q3_sieve", eine Zeile je Messung:
    Reihe | Gruppe | Probe | t_sim_s | Gewicht | Sieb_g | 100 | 250 | 500 | 800 | 1400 | 2000
    (Spaltenkoepfe mit Zahl = Siebmasche in um, Werte = Q3 an dieser Masche, 0..1)

Aufruf:
    python studies/optimization/dataprep/prepare_aj_q3_sieve.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd


ROOT = Path(r"C:\Studium\Master\MA\Daten\MVM-IKFT Eirich-Mischer Data")
OUT = Path(__file__).resolve().parents[1] / "data" / "AJ_Q3_sieve.xlsx"
SHIFT_S = 90.0
MESH_UM = (100, 250, 500, 800, 1400, 2000)

# Versuchsreihen: (Kurzname, Gruppe, Datei relativ zu ROOT). Jede Reihe zaehlt gleich.
SERIES = [
    ("AJ30", "AJ30 Probenreihe", "260713_PuralTH130_Sampling/260713_PSD_SieveTower_AJ30.xlsx"),
    ("AJ26", "10 min ohne Proben", "2607_PuralTH130_No Sampling/Time_10min_1/260703_PSD_FullBatch_Sieving_AJ26.xlsx"),
    ("AJ29", "10 min ohne Proben", "2607_PuralTH130_No Sampling/Time_10min_2/260707_PSD_FullBatch_Sieving_AJ29.xlsx"),
    ("AJ31", "10 min ohne Proben", "2607_PuralTH130_No Sampling/Time_10min_3/260713_PSD_FullBatch_Sieving_AJ31.xlsx"),
    ("AJ40", "6 min ohne Proben", "2607_PuralTH130_No Sampling/Time_6min/260727_PSD_FullBatch_Sieving_AJ40.xlsx"),
]

_LABEL = re.compile(r"(S\d+)\s+at\s+(\d+)\s*s")


def read_sieve_masses(path: Path) -> list[tuple[str, float, dict[float, float]]]:
    """Rohblatt -> [(Probe, Prozesszeit [s], {Masche [um]: Masse auf dem Sieb [g]})].

    Masche 0 = Auffangschale (< 100 um). Masse auf Masche m = Klasse [m, naechste Masche).
    """
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = next(s for s in wb.worksheets if s.title.strip().startswith("PSD Boehmite"))
    samples, cur = [], None
    for row in ws.iter_rows(values_only=True):
        a, c, f = row[0], row[2], row[5]
        m = _LABEL.match(str(a).strip()) if isinstance(a, str) else None
        if m:
            cur = (m.group(1), float(m.group(2)), {})
            samples.append(cur)
        if cur is None or c is None or f is None:
            continue
        mesh = 0.0 if str(c).strip().lower() == "bottom pan" else float(c)
        mass = float(f)
        if mass < -1e-9:
            raise ValueError(f"{path.name} {cur[0]}: negative Siebmasse {mass} auf {c}")
        cur[2][mesh] = max(mass, 0.0)
    if not samples:
        raise ValueError(f"{path.name}: keine Probe gefunden")
    return samples


def q3_at_mesh(masses: dict[float, float]) -> np.ndarray:
    """Q3(x_k) = Masse unterhalb Masche x_k / gesamte gesiebte Masse."""
    meshes = np.array(sorted(masses))
    m = np.array([masses[k] for k in meshes])
    return np.array([m[meshes < x].sum() / m.sum() for x in MESH_UM])


def build_table() -> pd.DataFrame:
    rows = []
    w_series = 1.0 / len(SERIES)
    for key, group, rel in SERIES:
        samples = read_sieve_masses(ROOT / rel)
        for label, t_proc, masses in samples:
            row = {"Reihe": key, "Gruppe": group, "Probe": label, "t_sim_s": t_proc - SHIFT_S,
                   "Gewicht": w_series / len(samples), "Sieb_g": sum(masses.values())}
            row.update(dict(zip(MESH_UM, q3_at_mesh(masses))))
            rows.append(row)
    table = pd.DataFrame(rows).sort_values(["t_sim_s", "Reihe"]).reset_index(drop=True)
    if abs(table["Gewicht"].sum() - 1.0) > 1e-12:
        raise AssertionError("Gewichte summieren nicht zu 1")
    return table


def check_aj30(table: pd.DataFrame) -> float:
    """AJ30 aus den Rohmassen gegen das Blatt 'Q3' der AJ30-Datei."""
    raw = pd.read_excel(ROOT / SERIES[0][2], sheet_name="Q3", header=None)
    labels = [str(x).strip() for x in raw.iloc[1, 1:]]
    body = raw.iloc[3:, :]
    diam = pd.to_numeric(body.iloc[:, 0], errors="coerce")
    sheet = body[diam.isin(MESH_UM)].iloc[:, 1:].to_numpy(float)          # (6 Maschen, 9 Proben)
    mine = table[table.Reihe == "AJ30"].set_index("Probe").loc[labels, list(MESH_UM)]
    return float(np.max(np.abs(mine.to_numpy().T - sheet)))


def main() -> None:
    table = build_table()
    dev = check_aj30(table)
    if dev > 1e-9:
        raise AssertionError(f"AJ30: Rohmassen weichen vom Blatt Q3 ab ({dev:.2e})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        table.to_excel(xw, sheet_name="Q3_sieve", index=False)

    print(f"geschrieben: {OUT}")
    print(f"Kontrolle AJ30 gegen Blatt 'Q3': max. Abweichung {dev:.1e}")
    print(f"Gewicht je Sim-Zeit: {table.groupby('t_sim_s').Gewicht.sum().round(4).to_dict()}")
    print()
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
