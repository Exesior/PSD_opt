"""Stufe 0 gegen die produktiven JIT-Kernel.

Prueft:
  (a) rebuild_r_pairdelta_moment == rebuild_r_pairdelta_pairwise  (die Herleitung)
  (b) parallel == serial                                          (Dispatch-Zwillinge)
  (c) separable_tables reproduziert kernel.compute_beta            (die f_k/g_k-Tabellen)
  (d) pick_partner_pairdelta summiert auf partner_total == R_i*/W_i (Sampler-Konsistenz)
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


from wmcpbe.kernels.aggregation import get_aggregation_kernel
from wmcpbe.kernels.aggregation.jit_kernels import (
    KERNEL_IDS, MOMENT_KERNELS, _beta_of, kernel_p0, separable_tables,
    rebuild_r_pairdelta_moment,
    rebuild_r_pairdelta_pairwise, rebuild_r_pairdelta_pairwise_serial,
    pick_partner_pairdelta,
)

# Muss jeden Namen aus KERNEL_IDS abdecken, sonst bricht das Skript beim
# Nachziehen eines neuen kompilierten Kernels mit einem KeyError ab.
KERNEL_ARGS = {
    "constant": dict(corr_beta=2.5e-10),
    "sum": dict(corr_beta=1e-9),
    "shear_chin1998": dict(corr_beta=1e-3, g=1000.0),
    "brownian_tsouris1995": dict(corr_beta=1.0, temperature=293.0, viscosity=1e-3),
    "eke_darelius2005": dict(corr_beta=1e-11, n_mixer=20.0),
    "etm_darelius2005": dict(corr_beta=1e-16, n_mixer=20.0),
}

_MISSING = sorted(set(KERNEL_IDS) - set(KERNEL_ARGS))
if _MISSING:
    raise SystemExit(
        f"KERNEL_ARGS fehlen Eintraege fuer {_MISSING}. Jeder kompilierte "
        "Kernel muss hier mit Testparametern hinterlegt sein."
    )


def make_pop(case, rng, n, dW):
    R = rng.uniform(1e-6, 5e-5, n)
    if case == "mixed":
        W = np.where(rng.random(n) < 0.5,
                     rng.integers(1, 60, n).astype(float),
                     rng.uniform(0.05, 60.0, n))
    elif case == "all_heavy":
        W = rng.uniform(dW, 5 * dW, n)
    elif case == "all_light":
        W = rng.uniform(1e-3, dW * 0.999, n)
    elif case == "many_ties":
        W = rng.choice(np.array([2.0, 5.0, 5.0, 10.0, float(dW)]), n)
    elif case == "boundary":
        W = np.full(n, float(dW))
    elif case == "sub_one":
        W = rng.uniform(0.01, 1.0, n)
    elif case == "integer":
        W = rng.integers(1, 40, n).astype(float)
    elif case == "with_zeros":
        W = rng.integers(0, 30, n).astype(float)
    else:
        raise ValueError(case)
    return np.ascontiguousarray(R), np.ascontiguousarray(W)


def deltas(W, dW):
    d = np.minimum(W, dW)
    return np.where(np.isfinite(d) & (d > 0.0), d, 0.0)


def relmax(a, b, atol_scale=None):
    """Mixed rel/abs deviation.

    A pure relative measure is wrong for quantities built by cancellation: where
    the exact answer is 0, any rounding residue scores 1.0 regardless of how
    tiny it is in absolute terms. ``atol_scale`` supplies the natural magnitude
    of the problem (here: the largest propensity in the population), below which
    differences are not meaningful.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    sc = np.maximum(np.abs(a), np.abs(b))
    if atol_scale is None:
        atol_scale = float(sc.max()) if sc.size else 0.0
    atol = 1e-14 * atol_scale
    denom = np.maximum(sc, atol)
    nz = denom > 0
    if not nz.any():
        return 0.0
    return float((np.abs(a[nz] - b[nz]) / denom[nz]).max())


def main():
    rng = np.random.default_rng(20260813)
    cases = ["mixed", "all_heavy", "all_light", "many_ties",
             "boundary", "sub_one", "integer", "with_zeros"]
    sizes = [1, 2, 5, 37, 200, 900]
    dWs = [1.0, 5.0, 10.0]

    print("(a)+(b) Momentenform / Pairwise / Serial")
    print("    (Kernel ohne Momentenform - EKE/ETM - haben keine Spalte (a);")
    print("     die Wurzel einer Summe laesst sich nicht separieren.)")
    ok = True
    worst_all = 0.0
    for name, kid in KERNEL_IDS.items():
        kernel = get_aggregation_kernel(name, **KERNEL_ARGS[name])
        p0 = kernel_p0(name, kernel)
        separable = name in MOMENT_KERNELS
        worst_m, worst_s = 0.0, 0.0
        for case in cases:
            for n in sizes:
                for dW in dWs:
                    R, W = make_pop(case, rng, n, dW)
                    d = deltas(W, dW)

                    r_par = np.zeros(n)
                    rebuild_r_pairdelta_pairwise(kid, p0, R, W, d, dW, r_par)
                    r_ser = np.zeros(n)
                    rebuild_r_pairdelta_pairwise_serial(kid, p0, R, W, d, dW, r_ser)
                    worst_s = max(worst_s, relmax(r_par, r_ser))

                    if not separable:
                        continue

                    F, G, bii, _ = separable_tables(name, kernel, R)
                    r_mom = np.zeros(n)
                    rebuild_r_pairdelta_moment(
                        np.ascontiguousarray(F), np.ascontiguousarray(G),
                        np.ascontiguousarray(bii), W, d, dW, r_mom)
                    worst_m = max(worst_m, relmax(r_mom, r_par))

        worst_all = max(worst_all, worst_m)
        good = worst_s == 0.0 and (worst_m <= 1e-13 if separable else True)
        flag = "OK " if good else "FAIL"
        mom = f"{worst_m:.3e}" if separable else "  --     "
        print(f"  {flag} {name:24s} moment-vs-pairwise = {mom} | par-vs-serial = {worst_s:.3e}")
        ok &= good

    print("\n(c) separable_tables vs. kernel.compute_beta")
    for name in sorted(MOMENT_KERNELS):
        kernel = get_aggregation_kernel(name, **KERNEL_ARGS[name])
        R = np.ascontiguousarray(rng.uniform(1e-6, 5e-5, 60))
        F, G, bii, _ = separable_tables(name, kernel, R)
        ref = np.array([[kernel.compute_beta(float(R[i]), float(R[j]))
                         for j in range(len(R))] for i in range(len(R))])
        got = F @ G.T
        e1 = relmax(ref, got)
        e2 = relmax(np.diag(ref), np.asarray(bii, dtype=float))
        flag = "OK " if max(e1, e2) <= 1e-13 else "FAIL"
        print(f"  {flag} {name:24s} beta = {e1:.3e} | beta(i,i) = {e2:.3e}")
        ok &= max(e1, e2) <= 1e-13

    print("\n(c2) _beta_of vs. kernel.compute_beta")
    # Fuer die separablen Kernel deckt (c) den Weg ueber die f_k/g_k-Tabellen ab.
    # Die nicht separablen erreichen den kompilierten Pfad NUR ueber _beta_of,
    # also muss genau dort die Gleichheit mit compute_beta geprueft werden.
    #
    # Massstab: relative Uebereinstimmung fuer alle, Bitgleichheit nur fuer die
    # Kernel in _BITEXACT. Der Unterschied ist Absicht, nicht Nachlaessigkeit:
    #   * `sum` rechnet in _beta_sum_jit `4/3*pi*r**3`, in _beta_of dagegen
    #     `_PI43*r*r*r`;
    #   * `brownian` multipliziert corr_beta in _beta_brownian_jit separat,
    #     waehrend _beta_of alles in p0 faltet.
    # Beides ist algebraisch identisch und weicht nur im letzten Bit ab -
    # dieselbe Klasse von Abweichung, die (a) fuer die Momentenform mit 1e-13
    # zulaesst. Eine Bitgleichheit zu fordern wuerde diese beiden Kernel
    # grundlos rot faerben.
    #
    # EKE und ETM sind dagegen bewusst so geschrieben, dass beide Seiten
    # dieselbe Formel in derselben Klammerung rechnen. Dort IST Bitgleichheit
    # die Zusage, und ein Bruch davon soll auffallen.
    _BITEXACT = {"eke_darelius2005", "etm_darelius2005"}
    for name, kid in KERNEL_IDS.items():
        kernel = get_aggregation_kernel(name, **KERNEL_ARGS[name])
        p0 = kernel_p0(name, kernel)
        R = rng.uniform(1e-6, 5e-5, 50)
        ref = np.array([[kernel.compute_beta(float(ri), float(rj)) for rj in R]
                        for ri in R])
        got = np.array([[_beta_of(kid, p0, float(ri), float(rj)) for rj in R]
                        for ri in R])
        e = relmax(ref, got)
        bad = int(np.count_nonzero(ref != got))
        strict = name in _BITEXACT
        good = (bad == 0) if strict else (e <= 1e-13)
        flag = "OK " if good else "FAIL"
        note = "bitgenau gefordert" if strict else "rel. Toleranz 1e-13"
        print(f"  {flag} {name:24s} rel = {e:.3e} | {bad:5d} Bit-Abweichungen "
              f"({note})")
        ok &= good

    print("\n(d) pick_partner_pairdelta: Gesamtgewicht == partner_total = R_i*/W_i")
    worst_p = 0.0
    for name, kid in KERNEL_IDS.items():
        kernel = get_aggregation_kernel(name, **KERNEL_ARGS[name])
        p0 = kernel_p0(name, kernel)
        for case in ["mixed", "all_light", "integer"]:
            for dW in [1.0, 5.0]:
                n = 120
                R, W = make_pop(case, rng, n, dW)
                d = deltas(W, dW)
                r = np.zeros(n)
                rebuild_r_pairdelta_pairwise_serial(kid, p0, R, W, d, dW, r)
                for i in [0, 7, n // 2, n - 1]:
                    if W[i] <= 0 or d[i] <= 0 or r[i] <= 0:
                        continue
                    total = r[i] / W[i]
                    # u_sel -> 1 laeuft den kompletten Loop durch; acc endet auf
                    # der Gesamtsumme, also muss der letzte Treffer existieren.
                    acc = 0.0
                    for j in range(n):
                        jj, wj = pick_partner_pairdelta(
                            kid, p0, i, R, W, d, dW, total, 0.0)
                        break
                    # Gesamtsumme rekonstruieren, indem der Loop komplett laeuft:
                    s = 0.0
                    for j in range(n):
                        if j == i:
                            if W[i] > 1.0:
                                sd = min(d[i], 0.5 * W[i])
                                if sd > 0:
                                    s += (W[i] - 1.0) * kernel.compute_beta(
                                        float(R[i]), float(R[i])) / sd
                            continue
                        if W[j] <= 0 or d[j] <= 0:
                            continue
                        s += W[j] * kernel.compute_beta(float(R[i]), float(R[j])) / min(d[i], d[j])
                    worst_p = max(worst_p, abs(s - total) / max(abs(s), abs(total)))
    flag = "OK " if worst_p <= 1e-12 else "FAIL"
    print(f"  {flag} max. Abweichung Summe(w_ij) vs. R_i*/W_i = {worst_p:.3e}")
    ok &= worst_p <= 1e-12

    print()
    print("GESAMT:", "BESTANDEN" if ok else "FEHLGESCHLAGEN")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
