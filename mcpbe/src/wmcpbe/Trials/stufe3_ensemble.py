"""Stufe 3, entscheidende Messung: Bias oder Rauschen?

Ein einzelner MC-Lauf kann nicht zwischen einem systematischen Fehler (Bias) und
statistischem Rauschen unterscheiden. Deshalb wird hier ueber ein Ensemble von
Seeds gemittelt und der VORZEICHENBEHAFTETE relative Fehler betrachtet:

    e(t) = ( n_MC(t) - n_analytisch(t) ) / n_analytisch(t)

  * Rauschen  mittelt sich gegen 0, der Mittelwert liegt innerhalb des
    Standardfehlers sem = std/sqrt(R).
  * Bias      bleibt als systematische Abweichung stehen und waechst mit der Zeit
    (Paper Sec. 3.4: der Bias akkumuliert).

Verglichen werden zwei Schemata, die sich AUSSCHLIESSLICH in der Paar-Kappung
unterscheiden (der W_i-Vorfaktor ist in beiden vorhanden):

    korrigiert   :  R_i = W_i * sum_j W_j beta / min(delta_i, delta_j)
    unkorrigiert :  R_i = W_i * sum_j W_j beta / delta_i
"""

from __future__ import annotations

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_SRC  = _os.path.abspath(_os.path.join(_HERE, '..', '..'))          # mcpbe/src
_PKG  = _os.path.abspath(_os.path.join(_HERE, '..', '..', '..'))    # mcpbe
for _p in (_SRC, _PKG, _HERE):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import sys
import numpy as np


import stufe3_bias as B  # noqa: E402
from wmcpbe.mcpbe import MCPBESolver  # noqa: E402

BETA = B.BETA
A0 = 400
W0 = 80.0
DW = 50.0
N_SAVE = 11
REPEATS = 12
MAXITER = 400000


def build(seed, t_end):
    t_vec = np.linspace(0.0, t_end, N_SAVE)
    s = MCPBESolver(
        dim=1, t_vec=t_vec, verbose=False, load_attr=False, init=False, seed=seed,
        agg_kernel_name="constant", agg_kernel_params={"corr_beta": BETA},
    )
    s.process_type = "agglomeration"
    s.c = np.full(1, 1e-2)
    s.x = np.full(1, 1e-6)
    s.PGV = np.full(1, "mono")
    s.SIG = np.full(1, 0.1)
    s.a0 = A0
    s.alpha_prim = np.ones(1)
    s.maybe_double_control_volume = False
    s.SIZEEVAL = 0
    s.agg_dW_max = DW
    s.agg_dW_min = DW
    s.agg_dW_mode = "const"
    s.recon_enable = False
    s._initialize_particles(W_init=np.full(A0, W0))
    s._init_lmc()
    s._initialize_samplers()
    return s


def signed_error(s, n0, t_end):
    t = np.asarray(s.t_vec)
    n = B.n_phys_series(s)
    k = min(len(n), len(t))
    ref = n0 / (1.0 + BETA * n0 * t[:k] / 2.0)
    return t[:k], (n[:k] - ref) / ref


def main():
    probe = build(1, 1.0)
    n0 = float(np.sum(probe.W[: probe.a_tot])) / float(probe.Vc)
    t_end = 4.0 / (BETA * n0)
    print(f"n0 = {n0:.4e} 1/m^3   dW = {DW:g}   W0 = {W0:g}   "
          f"Rechenpartikel = {A0}   Wiederholungen = {REPEATS}")
    print(f"t_end = {t_end:.4e} s\n")

    series = {}
    for label, patcher in [("korrigiert", None),
                           ("unkorrigiert", B.patch_no_pairdelta)]:
        errs = []
        for r in range(REPEATS):
            s = build(1000 + r, t_end)
            if patcher is not None:
                patcher(s)
                s._refresh_samplers_after_agg()
            nn0 = float(np.sum(s.W[: s.a_tot])) / float(s.Vc)
            s.solve(maxiter=MAXITER)
            t, e = signed_error(s, nn0, t_end)
            errs.append(e)
        m = min(len(e) for e in errs)
        arr = np.stack([e[:m] for e in errs], axis=0)
        series[label] = (t[:m], arr)

    print("Vorzeichenbehafteter relativer Fehler in n_phys, Mittel +/- Standardfehler")
    print(f"{'t/t_end':>8s} | {'korrigiert':>26s} | {'unkorrigiert':>26s} | {'Verhaeltnis':>11s}")
    print("-" * 82)
    tt = series["korrigiert"][0]
    for i in range(len(tt)):
        row = f"{tt[i]/t_end:8.2f} |"
        vals = {}
        for label in ("korrigiert", "unkorrigiert"):
            a = series[label][1][:, i]
            mean = float(a.mean())
            sem = float(a.std(ddof=1) / np.sqrt(a.shape[0])) if a.shape[0] > 1 else 0.0
            vals[label] = (mean, sem)
            sig = "*" if abs(mean) > 2.0 * sem and sem > 0 else " "
            row += f" {mean:+.3e} +/- {sem:.1e}{sig} |"
        r = (abs(vals['unkorrigiert'][0]) / abs(vals['korrigiert'][0])
             if vals['korrigiert'][0] != 0 else float('inf'))
        row += f" {r:10.1f}x"
        print(row)
    print("\n* = Mittelwert liegt ausserhalb von 2 Standardfehlern -> systematisch, "
          "nicht Rauschen")

    print("\nZusammenfassung am Endzeitpunkt:")
    for label in ("korrigiert", "unkorrigiert"):
        a = series[label][1][:, -1]
        mean, sem = float(a.mean()), float(a.std(ddof=1) / np.sqrt(a.shape[0]))
        verdict = "SYSTEMATISCH" if abs(mean) > 2 * sem else "mit Rauschen vertraeglich"
        print(f"  {label:13s}: {mean:+.4e} +/- {sem:.2e}   -> {verdict}")


if __name__ == "__main__":
    main()
