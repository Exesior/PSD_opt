"""
Aufbereitung der Sieb-PSD (Q3) aus Versuch AJ30 -> Eingabeblatt fuer den
wet_granulation-Adapter.

Quelle
------
C:\\Studium\\Master\\MA\\Daten\\MVM-IKFT Eirich-Mischer Data\\
    260713_PuralTH130_Sampling\\260713_PSD_SieveTower_AJ30.xlsx
Blatt "Q3": kumulierte Massen-Unterkornverteilung (Q3, 0..1) an den
Siebmaschen 100/250/500/800/1400/2000 um, Proben S1..S9.

Zeitachse (Entscheidung Eric, 2026-09-08)
----------------------------------------
* Simulation startet mit der Binderzugabe -> Prozess-t=90 s == Sim-t=0.
  Alle Probenzeiten werden um SHIFT (=90 s) nach vorne geschoben.
* Ausgangspulver: monodispers, d0 = 34 um. Die Simulation baut die
  Startpopulation selbst daraus; hier wird nur die zugehoerige t=0-Spalte
  ergaenzt (Q3 einer Monodispersen: 0 unterhalb d0, 1 ab d0 -> auf diesem
  Sieb-Raster ueberall 1.0).
* Alle 9 Proben gehen in den Fit.

Probenzeitpunkte (Prozesszeit -> Sim-Zeit bei SHIFT=90):
    S1 150->60  S2 270->180  S3 330->240  S4 390->300  S5 450->360
    S6 510->420 S7 570->480  S8 630->540  S9 690->600

Ergebnis
--------
studies/optimization/data/AJ30_Q3.xlsx, Blatt "Q3":
    Spalte 0  : Diameter_mm   (streng steigend)
    Spalte 1  : "0"           (Startzustand, monodispers d0)
    Spalte 2..: eine Spalte je Probe, Kopfzeile = Sim-Zeit in SEKUNDEN,
                Werte = Q3-CDF (0..1, je Spalte nicht fallend)

Der Adapter liest genau dieses Blatt und waehlt je t_vec-Wert die passende
Spalte (t_vec muss also [0, 60, 180, ...] sein).

Aufruf:
    python studies/optimization/dataprep/prepare_aj30_q3.py
    python studies/optimization/dataprep/prepare_aj30_q3.py --shift 90 --d0-um 34
    python studies/optimization/dataprep/prepare_aj30_q3.py --process-time   # keine Verschiebung
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(
    r"C:\Studium\Master\MA\Daten\MVM-IKFT Eirich-Mischer Data"
    r"\260713_PuralTH130_Sampling\260713_PSD_SieveTower_AJ30.xlsx"
)
OUT = Path(__file__).resolve().parents[1] / "data" / "AJ30_Q3.xlsx"

# Probenlabel -> Prozesszeit [s]  (Blatt "Dauer")
SAMPLE_TIME_S = {
    "S1": 150, "S2": 270, "S3": 330, "S4": 390, "S5": 450,
    "S6": 510, "S7": 570, "S8": 630, "S9": 690,
}


def read_q3(src: Path) -> pd.DataFrame:
    """Rohes Q3-Blatt -> DataFrame (index = Durchmesser [mm], Spalten = Prozesszeit [s])."""
    raw = pd.read_excel(src, sheet_name="Q3", header=None)

    sample_labels = [str(x).strip() for x in raw.iloc[1, 1:].tolist()]
    times_s = [SAMPLE_TIME_S[lbl] for lbl in sample_labels]

    body = raw.iloc[3:, :].copy()
    diam_um = pd.to_numeric(body.iloc[:, 0], errors="coerce")
    keep = diam_um.notna() & np.isfinite(diam_um)          # 'inf'-Zeile faellt weg
    diam_um = diam_um[keep].to_numpy(dtype=float)
    values = body.iloc[:, 1:].to_numpy(dtype=float)[keep.to_numpy()]

    df = pd.DataFrame(values, index=diam_um / 1000.0, columns=times_s)
    df.index.name = "Diameter_mm"

    df = df.sort_index()
    if not np.all(np.diff(df.index.to_numpy()) > 0):
        raise ValueError("Durchmesser nicht streng steigend nach Sortierung.")

    # Q3 je Spalte auf [0,1] und monoton nicht fallend (Messrauschen glaetten)
    df = df.clip(lower=0.0, upper=1.0)
    df = df.cummax(axis=0)
    return df


def monodisperse_q3(diam_mm: np.ndarray, d0_mm: float) -> np.ndarray:
    """Q3-CDF einer monodispersen Verteilung bei d0 auf dem gegebenen Raster."""
    return (diam_mm >= d0_mm).astype(float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift", type=float, default=90.0,
                    help="Prozesszeit-Offset [s], der als Sim-t=0 gilt (Default 90).")
    ap.add_argument("--process-time", action="store_true",
                    help="Keine Verschiebung; Original-Prozesszeiten behalten.")
    ap.add_argument("--d0-um", type=float, default=34.0,
                    help="Startdurchmesser des Ausgangspulvers [um] (Default 34).")
    args = ap.parse_args()

    if not SRC.is_file():
        raise FileNotFoundError(SRC)

    df = read_q3(SRC)

    shift = 0.0 if args.process_time else float(args.shift)
    df.columns = [int(round(c - shift)) for c in df.columns]

    # t=0-Startspalte (monodispers) voranstellen
    diam_mm = df.index.to_numpy(dtype=float)
    df.insert(0, 0, monodisperse_q3(diam_mm, args.d0_um / 1000.0))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        df.reset_index().to_excel(xw, sheet_name="Q3", index=False)

    print(f"geschrieben: {OUT}")
    print(f"Durchmesser [mm]: {diam_mm}")
    print(f"Zeitspalten [s] (Sim-Zeit): {list(df.columns)}")
    print(f"Startdurchmesser d0: {args.d0_um} um   Offset: {shift} s")
    print()
    print(df.round(4).to_string())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for c in df.columns:
            ax.plot(diam_mm * 1000.0, df[c].to_numpy(),
                    marker="o", label=f"{c} s")
        ax.set_xscale("log")
        ax.set_xlabel("Durchmesser [um]")
        ax.set_ylabel("Q3 (kum. Unterkorn-Massenanteil)")
        ax.set_title("AJ30 Sieb-Q3 ueber die Sim-Zeit")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        png = OUT.with_suffix(".png")
        fig.savefig(png, dpi=130)
        print(f"\nPlot: {png}")
    except Exception as exc:  # pragma: no cover
        print(f"(Plot uebersprungen: {exc})")


if __name__ == "__main__":
    main()
