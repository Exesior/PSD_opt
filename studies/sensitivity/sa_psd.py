"""
Verteilungs-Mathematik der Sensitivitaetsstudie (reine numpy-Funktionen, kein Solver).

Konvention wie im Fit-Adapter (``WetGranulationAdapter._q3_at_meshes_from_snapshot``):
    Durchmesser  x = (6 V_dry / pi)^(1/3)      (Huellvolumen V_dry = V_flat[-1])
    Masse        m = W * V_solid               (V_solid = V_flat[0]; Dichte kuerzt sich)
    Q3(x)        = Massenanteil aller Partikel mit Durchmesser < x  (Sieb-Logik)

Alle Funktionen nehmen EIN Replikat (eine Partikelliste) oder bereits berechnete
Q3-Kurven auf einem gemeinsamen Raster.
"""

from __future__ import annotations

import numpy as np

# numpy >= 2.0 heisst es trapezoid, davor trapz (Remote-Rechner evtl. aelter)
_trapz = getattr(np, "trapezoid", None) or np.trapz

# Siebmaschen der Versuche (KF3-Datenblatt "Q3_sieve") [um]
MESHES_UM = np.array([100.0, 250.0, 500.0, 800.0, 1400.0, 2000.0])

# Feines log-Raster fuer gespeicherte Q3-Kurven und Flaechenmasse [um]
# 10 um .. 5 mm, 241 Punkte = 90 Punkte pro Dekade -> Treppen der Partikelliste
# werden fein genug abgetastet (Rasterfehler << MC-Rauschen).
GRID_UM = np.logspace(1.0, np.log10(5000.0), 241)


def diameters_and_mass(v_flat: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Durchmesser [um] und Massen-Proxy (W*V_solid) der gueltigen Partikel, nach Groesse sortiert."""
    v_flat = np.asarray(v_flat, dtype=float)
    d_um = np.cbrt(6.0 * v_flat[-1] / np.pi) * 1e6
    mass = np.asarray(weights, dtype=float) * v_flat[0]
    ok = np.isfinite(d_um) & np.isfinite(mass) & (d_um > 0.0) & (mass > 0.0)
    d_um, mass = d_um[ok], mass[ok]
    order = np.argsort(d_um)
    return d_um[order], mass[order]


def q3_at(d_um_sorted: np.ndarray, mass_sorted: np.ndarray, x_um: np.ndarray) -> np.ndarray:
    """Q3 an den Stellen ``x_um``: Massenanteil mit Durchmesser < x (exakt wie der Fit-Adapter)."""
    x_um = np.asarray(x_um, dtype=float)
    if d_um_sorted.size == 0:
        return np.full(x_um.shape, np.nan)
    cum = np.cumsum(mass_sorted)
    cum = cum / cum[-1]
    n_below = np.searchsorted(d_um_sorted, x_um, side="left")
    return np.where(n_below > 0, cum[np.maximum(n_below - 1, 0)], 0.0)


def quantile_diameter(d_um_sorted: np.ndarray, mass_sorted: np.ndarray, q: float) -> float:
    """Durchmesser [um], unter dem der Massenanteil q liegt (wie np.interp im Fit-Adapter)."""
    if d_um_sorted.size == 0:
        return float("nan")
    cum = np.cumsum(mass_sorted)
    cum = cum / cum[-1]
    return float(np.interp(q, cum, d_um_sorted))


def q3_curve_quantile(q3_grid: np.ndarray, q: float, grid_um: np.ndarray = GRID_UM) -> float:
    """Quantil-Durchmesser [um] aus einer (gemittelten) Q3-Kurve auf dem Raster, Interpolation in log x."""
    q3_grid = np.maximum.accumulate(np.clip(np.asarray(q3_grid, dtype=float), 0.0, 1.0))
    if not np.isfinite(q3_grid).all():
        return float("nan")
    if q <= q3_grid[0]:
        return float(grid_um[0])
    if q >= q3_grid[-1]:
        return float(grid_um[-1])
    # erste Rasterstelle, an der Q3 >= q
    k = int(np.searchsorted(q3_grid, q, side="left"))
    q_lo, q_hi = q3_grid[k - 1], q3_grid[k]
    lx_lo, lx_hi = np.log10(grid_um[k - 1]), np.log10(grid_um[k])
    if q_hi <= q_lo:
        return float(grid_um[k])
    return float(10 ** (lx_lo + (q - q_lo) / (q_hi - q_lo) * (lx_hi - lx_lo)))


def log_shift_decades(q3_a: np.ndarray, q3_b: np.ndarray, grid_um: np.ndarray = GRID_UM) -> tuple[float, float]:
    """Abstand zweier Q3-Kurven auf der log10-Groessenachse (Wasserstein-1 in log10 x).

    Rueckgabe (signiert, betrag) in DEKADEN, ``a`` = Referenz, ``b`` = verglichene Kurve:
        signiert  = Integral (Q3_a - Q3_b) d log10 x   > 0  heisst: b ist GROEBER als a
                    (gleiche Richtung wie log10 d50; = Differenz der massengemittelten
                     log10-Durchmesser, solange beide Kurven innerhalb des Rasters bleiben)
        betrag    = Integral |Q3_a - Q3_b| d log10 x   = Flaeche zwischen den Kurven
    Anschaulich: 0.1 Dekaden = Faktor 10^0.1 = 1.26 -> ~26 % Groessenverschiebung.
    """
    lx = np.log10(np.asarray(grid_um, dtype=float))
    diff = np.asarray(q3_a, dtype=float) - np.asarray(q3_b, dtype=float)
    signed = float(_trapz(diff, lx))
    absolute = float(_trapz(np.abs(diff), lx))
    return signed, absolute
