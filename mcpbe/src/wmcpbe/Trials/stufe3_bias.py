"""Stage 3 physics regression: does the bias correction act in the right direction?

Set up as in paper Sec. 3.4: pure agglomeration with a CONSTANT kernel, because
that is the case with a closed analytic solution. Three variants are compared
against the same analytic reference:

  corrected    : the current form, R_i* = W_i * sum_j W_j beta / min(dW_i,dW_j),
                 partner ~ W_j beta / dW_ij                    (paper Eq. 36/40)
  uncorrected  : the older form,   r_i  = (sum_j W_j beta) / dW_i,
                 partner ~ W_j                                 (biased)
  dW=1         : control -- at batch size 1 there is no batch bias by
                 construction, so both schemes must coincide.

Analytic solution (Smoluchowski, constant kernel, pure agglomeration)::

    dn/dt = -(1/2) * beta * n^2      ->      n(t) = n0 / (1 + beta*n0*t/2)

``n`` is the PHYSICAL number density, in the solver ``n = sum(W)/Vc``.

The uncorrected form is reproduced by monkeypatching, so no repository file is
modified and nothing has to be checked out from history. That also keeps the
comparison free of other pending changes in the working tree.
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


from wmcpbe.mcpbe import MCPBESolver  # noqa: E402

BETA = 1e-3          # konstanter Kernel [m^3/s]
A0 = 400             # Rechenpartikel
W0 = 400.0           # Startgewicht je Rechenpartikel
N_SAVE = 11


# ---------------------------------------------------------------------------
# Nachbildung des alten, unkorrigierten Schemas
# ---------------------------------------------------------------------------
def patch_uncorrected(solver):
    """Alte Propensity + alte Partnerwahl per Monkeypatch."""

    def old_raw(R, W, delta, dW_const, out):
        a = out.shape[0]
        beta = solver.kernel_manager.agg_kernel.corr_beta
        alpha = solver._alpha_scalar()
        b = beta * alpha
        sum_W = float(np.sum(W))
        for i in range(a):
            Wi = float(W[i])
            di = float(delta[i])
            if di <= 0.0 or Wi <= 0.0:
                out[i] = 0.0
                continue
            # alt: sum_j W_j*beta mit Selbstterm (W_i-1), danach EINMAL / delta_i
            s = (sum_W - Wi) * b
            if Wi > 1.0:
                s += (Wi - 1.0) * b
            out[i] = max(0.0, s / di)

    def old_pick(i, R, W, delta, dW_const, partner_total, u_sel):
        # alt: j proportional zu rohem W_j, beta nur als Ja/Nein-Filter
        a = len(W)
        total_W = float(np.sum(W))
        if total_W <= 0.0:
            return -1, 0.0
        j = int(np.searchsorted(np.cumsum(W), u_sel * total_W, side="right"))
        j = min(j, a - 1)
        beta = solver.kernel_manager.agg_kernel.corr_beta * solver._alpha_scalar()
        if beta <= 0.0:
            return -1, 0.0
        if i == j:
            pick_w = max(0.0, float(W[i]) - 1.0) * beta
        else:
            pick_w = float(W[j]) * beta
        if pick_w <= 0.0:
            return -1, 0.0
        return j, pick_w

    solver._compute_raw_propensities = old_raw
    solver._pick_partner_kernel = old_pick
    # Der kompilierte Schnellpfad wuerde die Patches umgehen -> abschalten.
    solver._compiled_kernel_spec = lambda: None


# ---------------------------------------------------------------------------
def build(dW_max, t_end, seed=20260813):
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
    s.SIZEEVAL = 0                      # keine Groessen-Akzeptanz: reines Schema
    s.agg_dW_max = float(dW_max)
    s.agg_dW_min = float(dW_max)
    s.agg_dW_mode = "const"
    s.recon_enable = False
    # Set the start weights through the intended W_init path. Overwriting s.W
    # afterwards would leave the snapshots created in _initialize_particles
    # (W0, W_save[0]) sitting on the old values.
    s._initialize_particles(W_init=np.full(A0, W0))
    s._init_lmc()
    s._initialize_samplers()
    assert abs(float(np.sum(s.W_save[0])) - A0 * W0) < 1e-6, "W_init nicht uebernommen"
    return s


def n_phys_series(s):
    """Physikalische Anzahldichte zu jedem Speicherzeitpunkt."""
    vc = getattr(s, "Vc_save", None)
    out = []
    for k, W in enumerate(s.W_save):
        v = float(vc[k]) if vc is not None and k < len(np.atleast_1d(vc)) else float(s.Vc)
        out.append(float(np.sum(W)) / v)
    return np.asarray(out)


def analytic(n0, t):
    return n0 / (1.0 + BETA * n0 * t / 2.0)


def run(dW_max, uncorrected, t_end, maxiter):
    s = build(dW_max, t_end)
    if uncorrected:
        patch_uncorrected(s)
        s._refresh_samplers_after_agg()   # Propensities mit dem alten Schema neu aufbauen
    n0 = float(np.sum(s.W[: s.a_tot])) / float(s.Vc)
    s.solve(maxiter=maxiter)
    return s, n0


def report(label, s, n0, t_used):
    n = n_phys_series(s)
    k = min(len(n), len(t_used))
    t = np.asarray(t_used[:k])
    n = n[:k]
    ref = analytic(n0, t)
    rel = np.abs(n - ref) / ref
    return t, n, ref, rel


def patch_no_pairdelta(solver):
    """Nur die Paar-Kappung abschalten, W_i-Vorfaktor beibehalten.

    R_i = W_i * sum_j W_j beta / delta_i     (statt / min(delta_i, delta_j))

    Isoliert damit genau den Batch-Bias aus Paper Gl. 33/36 - unabhaengig vom
    separaten Defekt des fehlenden W_i-Vorfaktors.
    """

    def raw(R, W, delta, dW_const, out):
        a = out.shape[0]
        b = solver.kernel_manager.agg_kernel.corr_beta * solver._alpha_scalar()
        sum_W = float(np.sum(W))
        for i in range(a):
            Wi = float(W[i])
            di = float(delta[i])
            if di <= 0.0 or Wi <= 0.0:
                out[i] = 0.0
                continue
            s = (sum_W - Wi) * b
            if Wi > 1.0:
                s += (Wi - 1.0) * b
            out[i] = max(0.0, Wi * s / di)

    def pick(i, R, W, delta, dW_const, partner_total, u_sel):
        a = len(W)
        di = float(delta[i])
        if di <= 0.0 or partner_total <= 0.0:
            return -1, 0.0
        b = solver.kernel_manager.agg_kernel.corr_beta * solver._alpha_scalar()
        thresh = u_sel * partner_total
        acc = 0.0
        last_j, last_w = -1, 0.0
        for j in range(a):
            if j == i:
                Wi = float(W[i])
                if Wi <= 1.0:
                    continue
                wij = (Wi - 1.0) * b / di
            else:
                Wj = float(W[j])
                if Wj <= 0.0 or float(delta[j]) <= 0.0:
                    continue
                wij = Wj * b / di
            if wij <= 0.0:
                continue
            acc += wij
            last_j, last_w = j, wij
            if acc > thresh:
                return j, wij
        return (last_j, last_w) if last_j >= 0 else (-1, 0.0)

    solver._compute_raw_propensities = raw
    solver._pick_partner_kernel = pick
    solver._compiled_kernel_spec = lambda: None


def run_variant(dW_max, patcher, t_end, maxiter, w0):
    global W0
    saved = W0
    W0 = w0
    try:
        s = build(dW_max, t_end)
    finally:
        W0 = saved
    if patcher is not None:
        patcher(s)
        s._refresh_samplers_after_agg()
    n0 = float(np.sum(s.W[: s.a_tot])) / float(s.Vc)
    s.solve(maxiter=maxiter)
    return s, n0


def experiment_b(t_end, maxiter):
    """Batch bias in isolation.

    Start weight just above dW, so that "light" particles (W_i < dW) appear
    immediately and min(delta_i, delta_j) actually bites.
    """
    print("\n\n=== Experiment B: Batch-Bias isoliert (W_i-Vorfaktor in BEIDEN Varianten) ===")
    print("Startgewicht 80, dW = 50 -> Kinder bekommen W=50, Eltern fallen auf 30 (leicht).\n")
    print(f"{'Variante':26s} {'Endfehler':>11s} {'max Fehler':>11s} {'phys. Koll.':>12s}")
    print("-" * 64)
    out = {}
    for label, patcher in [("pair-delta korrigiert", None),
                           ("pair-delta abgeschaltet", patch_no_pairdelta)]:
        s, n0 = run_variant(50, patcher, t_end, maxiter, w0=80.0)
        t = np.asarray(s.t_vec)
        n = n_phys_series(s)
        k = min(len(n), len(t))
        rel = np.abs(n[:k] - analytic(n0, t[:k])) / analytic(n0, t[:k])
        ev = float(getattr(s, "real_agg_events", float("nan")))
        out[label] = rel
        print(f"{label:26s} {rel[-1]:11.3e} {rel.max():11.3e} {ev:12.0f}")
    print("\nrelativer Fehler ueber die Zeit:")
    print(f"{'t/t_end':>8s} {'korrigiert':>14s} {'abgeschaltet':>14s}")
    ks = min(len(v) for v in out.values())
    tt = np.asarray(build(50, t_end).t_vec)
    for i in range(ks):
        print(f"{tt[i]/t_end:8.2f} {out['pair-delta korrigiert'][i]:14.3e} "
              f"{out['pair-delta abgeschaltet'][i]:14.3e}")


def main():
    # Zeithorizont so waehlen, dass n auf ~1/3 faellt: t_half-ish = 2/(beta*n0)
    s_probe = build(10, 1.0)
    n0 = float(np.sum(s_probe.W[: s_probe.a_tot])) / float(s_probe.Vc)
    t_end = 4.0 / (BETA * n0)
    print(f"n0 = {n0:.4e} 1/m^3   beta = {BETA:g}   t_end = {t_end:.4e} s")
    print(f"Rechenpartikel = {A0}, Startgewicht = {W0:g} -> N_phys = {A0*W0:g}\n")

    maxiter = 200000
    variants = [
        ("dW=1  corrected  ", 1, False),
        ("dW=1  uncorrected", 1, True),
        ("dW=20 corrected  ", 20, False),
        ("dW=20 uncorrected", 20, True),
        ("dW=50 corrected  ", 50, False),
        ("dW=50 uncorrected", 50, True),
    ]

    print(f"{'Variante':20s} {'Endfehler':>11s} {'max Fehler':>11s} {'Events':>9s}")
    print("-" * 56)
    results = {}
    for label, dW, unc in variants:
        s, n0v = run(dW, unc, t_end, maxiter)
        t_used = np.asarray(s.t_vec)
        t, n, ref, rel = report(label, s, n0v, t_used)
        ev = float(getattr(s, "real_agg_events", float("nan")))
        results[label] = (t, n, ref, rel)
        print(f"{label:20s} {rel[-1]:11.3e} {rel.max():11.3e} {ev:9.0f}")

    print("\nVerlauf des relativen Fehlers in n_phys(t):")
    hdr = "t/t_end   " + "".join(f"{lab.split()[0]+lab.split()[1][:4]:>13s}" for lab, _, _ in variants)
    print(hdr)
    t_ref = results[variants[0][0]][0]
    for idx in range(len(t_ref)):
        row = f"{t_ref[idx]/t_end:7.2f}   "
        for lab, _, _ in variants:
            row += f"{results[lab][3][idx]:13.3e}"
        print(row)

    experiment_b(t_end, maxiter)


if __name__ == "__main__":
    main()
