"""
Aufbereitung der Sieb-PSD aus Versuch AJ30  ->  Eingabeblatt fuer den
wet_granulation-Adapter, im Format  x(Quantil, t)  (Betreuer-Konvention).

Quelle
------
C:\\Studium\\Master\\MA\\Daten\\MVM-IKFT Eirich-Mischer Data\\
    260713_PuralTH130_Sampling\\260713_PSD_SieveTower_AJ30.xlsx
Blatt "Q3": kumulierte Massen-Unterkornverteilung Q3(x) an den Siebmaschen
100/250/500/800/1400/2000 um, Proben S1..S9.

Was passiert
-----------
Q3(x) wird pro Zeitpunkt INVERTIERT: fuer ein festes Quantil-Raster
Q = [0.00, 0.05, ..., 1.00] wird der Durchmesser x abgelesen, bei dem Q3
diesen Wert erreicht  (linear, wie re_calc_distribution im Framework:
np.interp(Q, Q3(x), x)).

Stuetzstellen der Interpolation = die GEMESSENEN (Q3, x)-Punkte:
    - unten fest (Eric, 2026-09-09):  (x = d0 = 34 um, Q3 = 0)  -> alles unter
      der feinsten Masche (100 um) wird linear zwischen 34 um und 100 um
      verteilt; x(Q=0) = 34 um. Grund: das Sieb loest < 100 um nicht auf.
    - Mitte:  die gemessenen Siebpunkte, bis zur ERSTEN Masche x* mit Q3 = 1.0.
    - oben (Eric, 2026-09-10):  x(Q=1) = x*  (nicht mehr pauschal 2000 um!).
      Siebzeilen hinter x* sind redundant (Q3 bleibt 1.0) und wuerden x(Q=1)
      faelschlich auf 2000 um ziehen. Bsp S8: Q3(800 um)=1.0 -> x(Q=1)=800 um.
      Erreicht Q3 die 1.0 nie (S1, Schwanz > 2 mm): x(Q=1) = 2000 um.

Zeitachse (Entscheidung Eric):
    Sim-t=0 == Prozess-t=90 s (Start Binderzugabe). Alle Probenzeiten um
    SHIFT (=90 s) nach vorne. t=0-Spalte = monodispers d0.
        S1..S9 (Sim): 60,180,240,300,360,420,480,540,600 s

Ergebnis
--------
studies/optimization/data/AJ30_xQ.xlsx, Blatt "x_Q3":
    Spalte 0  : Q            (Quantil-Wert, 0.00 .. 1.00, Schrittweite 0.05)
    Spalte 1  : "0"          (Startzustand, x = d0 fuer alle Q)
    Spalte 2..: eine Spalte je Probe, Kopf = Sim-Zeit [s], Werte = x [mm]

Der Adapter liest genau dieses Blatt; opt.t_vec muss zu den Spaltenkoepfen
passen ([0, 60, 180, ...]).

Aufruf:
    python studies/optimization/dataprep/prepare_aj30_xq.py
    python studies/optimization/dataprep/prepare_aj30_xq.py --step 0.1
    python studies/optimization/dataprep/prepare_aj30_xq.py --shift 90 --d0-um 34
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
OUT = Path(__file__).resolve().parents[1] / "data" / "AJ30_xQ.xlsx"

SAMPLE_TIME_S = {
    "S1": 150, "S2": 270, "S3": 330, "S4": 390, "S5": 450,
    "S6": 510, "S7": 570, "S8": 630, "S9": 690,
}
DEFAULT_STEP = 0.05                       # Q-Raster 0.00, 0.05, ..., 1.00


def read_q3_of_x(src: Path) -> pd.DataFrame:
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
    df = df.clip(lower=0.0, upper=1.0).cummax(axis=0)       # Q3 monoton in x
    return df


def invert_to_xq(q3_of_x: pd.DataFrame, q_grid: np.ndarray, d0_mm: float) -> pd.DataFrame:
    """Q3(x) -> x(Q): pro Zeitspalte den Durchmesser an den Quantilen q_grid.

    Stuetzstellen fuer die lineare Interpolation je Zeitspalte = die
    GEMESSENEN (Q3, x)-Punkte, plus unten fest (x = d0, Q3 = 0):

        (0, d0)  (Q3(100), 100)  (Q3(250), 250)  ...  (Q3(x*), x*)

    x* = kleinste Siebmasche, bei der Q3 = 1.0 erreicht ist. Die Zeilen
    DAHINTER (Q3 bleibt 1.0) sind redundant und werden weggelassen -- sonst
    landet x(Q=1) faelschlich auf der groebsten Masche (2000 um) statt auf x*.
    Beispiel S8: Q3(800 um)=1.0 -> x(Q=1)=800 um, nicht 2000 um.
    Erreicht Q3 die 1.0 nie (z.B. S1: Schwanz > 2 mm), bleibt x(Q=1)=2000 um.
    x(Q=0) = d0 ; dazwischen linear.
    """
    sieve_mm = q3_of_x.index.to_numpy(dtype=float)          # [0.1, 0.25, ..., 2.0] mm
    out = {}
    for t in q3_of_x.columns:
        q3 = q3_of_x[t].to_numpy(dtype=float)
        hit = np.flatnonzero(q3 >= 1.0 - 1e-9)              # erste Masche mit Q3 = 1
        keep = (hit[0] + 1) if hit.size else len(q3)        # diese Masche noch mitnehmen
        q3_nodes = np.concatenate(([0.0], q3[:keep]))
        x_nodes = np.concatenate(([d0_mm], sieve_mm[:keep]))
        xq = np.interp(q_grid, q3_nodes, x_nodes)
        xq[q_grid <= 0.0] = d0_mm
        xq[q_grid >= 1.0] = float(x_nodes[-1])              # = x* (bzw. 2000 um, wenn Q3 nie 1 erreicht)
        out[t] = xq
    df = pd.DataFrame(out, index=q_grid)
    df.index.name = "Q"
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift", type=float, default=90.0,
                    help="Prozesszeit-Offset [s] als Sim-t=0 (Default 90).")
    ap.add_argument("--process-time", action="store_true",
                    help="Keine Verschiebung; Original-Prozesszeiten behalten.")
    ap.add_argument("--d0-um", type=float, default=34.0,
                    help="Startdurchmesser (monodispers) [um] (Default 34). "
                         "Zugleich die untere Interpolations-Stuetzstelle (Q3=0).")
    ap.add_argument("--step", type=float, default=DEFAULT_STEP,
                    help=f"Schrittweite des Q-Rasters auf [0, 1] (Default {DEFAULT_STEP}).")
    args = ap.parse_args()

    if not SRC.is_file():
        raise FileNotFoundError(SRC)

    if not (0.0 < args.step <= 0.5):
        raise ValueError("--step muss in (0, 0.5] liegen.")
    n_steps = int(round(1.0 / args.step))
    q_grid = np.round(np.linspace(0.0, 1.0, n_steps + 1), 6)   # [0.00, step, ..., 1.00]

    q3_of_x = read_q3_of_x(SRC)
    shift = 0.0 if args.process_time else float(args.shift)
    q3_of_x.columns = [int(round(c - shift)) for c in q3_of_x.columns]

    xq = invert_to_xq(q3_of_x, q_grid, args.d0_um / 1000.0)

    # t=0-Spalte: monodispers -> x(Q) = d0 fuer jedes Quantil
    xq.insert(0, 0, args.d0_um / 1000.0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        xq.reset_index().to_excel(xw, sheet_name="x_Q3", index=False)

    print(f"\ngeschrieben: {OUT}")
    print(f"Quantil-Raster Q : {list(q_grid)}")
    print(f"Zeitspalten [s]  : {list(xq.columns)}")
    print(f"d0 = {args.d0_um} um   Offset = {shift} s\n")
    print("x(Q, t)  [mm]:")
    print(xq.round(4).to_string())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        times = list(xq.columns)
        cmap = plt.get_cmap("viridis")
        norm = Normalize(vmin=0.0, vmax=1.0)
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.8))

        # links: x(Q) ueber die Zeit, Farbe = Quantil
        for q in q_grid:
            axL.plot(times, xq.loc[q].to_numpy() * 1000.0, "-o", ms=3,
                     color=cmap(norm(q)))
        axL.set_xlabel("Sim-Zeit [s]")
        axL.set_ylabel("Durchmesser x [um]")
        axL.set_yscale("log")
        axL.set_title("x(Quantil) ueber die Zeit")
        axL.grid(True, which="both", alpha=0.3)
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axL)
        cb.set_label("Q3-Quantil")

        # rechts: x(Q) je Zeit  --  Q auf x, Durchmesser auf y, Farbe = Zeit
        tnorm = Normalize(vmin=min(times), vmax=max(times))
        for t in times:
            axR.plot(q_grid, xq[t].to_numpy() * 1000.0, "-o", ms=3,
                     color=plt.get_cmap("plasma")(tnorm(t)))
        axR.set_yscale("log")
        axR.set_xlabel("Q3-Quantil")
        axR.set_ylabel("Durchmesser x(Q) [um]")
        axR.set_title("x(Q) je Zeit")
        axR.grid(True, which="both", alpha=0.3)
        cb2 = fig.colorbar(ScalarMappable(norm=tnorm, cmap=plt.get_cmap("plasma")), ax=axR)
        cb2.set_label("Sim-Zeit [s]")

        fig.suptitle(f"AJ30  aufbereitete Fit-Daten  (Q-Schritt {args.step:g}, d0 {args.d0_um:g} um)")
        fig.tight_layout()
        png = OUT.with_suffix(".png")
        fig.savefig(png, dpi=130)
        print(f"\nPlot: {png}")
    except Exception as exc:  # pragma: no cover
        print(f"(Plot uebersprungen: {exc})")


if __name__ == "__main__":
    main()
