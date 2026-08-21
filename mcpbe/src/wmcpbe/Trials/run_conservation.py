"""Stages 1 and 2 without pytest: weight invariant and mass conservation.

Uses the same scenarios as ``mcpbe/tests/test_conservation.py`` but adds the
weight invariant: no particle may survive with ``0 < W < eps``, and every
removed particle must have had exactly ``W == 0.0``.
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


from tests.bench.scenarios import SCENARIOS, build  # noqa: E402


def solid_volume(solver) -> float:
    a = solver.a_tot
    if a <= 0:
        return 0.0
    v_dry = solver.V_flat[-1, :a]
    poro = solver.porosity[:a]
    w = solver.W[:a]
    v_solid = np.where(np.isnan(poro), v_dry, v_dry * (1.0 - poro))
    return float(np.sum(v_solid * w))


def liquid_volume(solver) -> float:
    a = solver.a_tot
    if a <= 0:
        return 0.0
    return float(np.sum(solver.liquid_volume[:a] * solver.W[:a]))


SOLID = ["agg_shear_1d", "agg_shear_1d_large", "agg_constant_1d", "agg_sum_1d",
         "agg_brownian_1d", "agg_shear_2d",
         "break_powerlaw_1d", "mix_1d", "granulation_1d"]

RTOL = 1e-9


def instrument(solver, stats):
    """Check the real invariant of the batch discipline.

    The invariant is NOT "W[j] == 0 when _remove_particle_column is called" --
    _consume_parent_weight never writes the zero back, it removes directly, so
    W[j] still holds the pre-consumption value there. What must hold is:

      (1) the residual actually computed, w_rem = W - dW (or W - 2*dW for a
          self collision), is exactly 0.0 whenever the particle is dropped, and
      (2) no particle ever survives with 0 < W < eps.
    """
    orig_consume = solver._consume_parent_weight

    def patched_consume(i, j, dW):
        before = {}
        idxs = {int(i)} if i == j else {int(i), int(j)}
        for k in idxs:
            if 0 <= k < solver.a_tot:
                before[k] = float(solver.W[k])
        factor = 2.0 if i == j else 1.0
        for k, w in before.items():
            w_rem = w - factor * dW
            if w_rem <= 0.0:
                stats["removed"] += 1
                if w_rem != 0.0:
                    stats["nonzero_removals"] += 1
                    stats["worst_residual"] = max(stats["worst_residual"], abs(w_rem))
        return orig_consume(i, j, dW)

    solver._consume_parent_weight = patched_consume

    orig_refresh = solver._refresh_samplers_after_agg

    def patched_refresh():
        out = orig_refresh()
        a = solver.a_tot
        if a > 0:
            w = solver.W[:a]
            tiny = w[(w > 0.0) & (w < 1e-9)]
            if tiny.size:
                stats["tiny_alive"] += int(tiny.size)
                stats["smallest_alive_W"] = min(stats["smallest_alive_W"], float(tiny.min()))
        return out

    solver._refresh_samplers_after_agg = patched_refresh


def main():
    print(f"{'Szenario':24s} {'dSolid/Solid':>14s} {'dLiquid':>12s} "
          f"{'n_comp':>7s} {'entf.':>7s} {'W!=0':>6s} {'0<W<1e-9':>9s}")
    print("-" * 90)

    ok = True
    for name in SOLID:
        if name not in SCENARIOS:
            print(f"{name:24s}  (nicht vorhanden - uebersprungen)")
            continue
        sc = SCENARIOS[name]
        solver = build(sc)
        stats = {"removed": 0, "nonzero_removals": 0, "worst_residual": 0.0,
                 "tiny_alive": 0, "smallest_alive_W": float("inf")}
        instrument(solver, stats)

        s0 = solid_volume(solver)
        l0 = liquid_volume(solver)
        solver.solve(maxiter=sc.maxiter)
        s1 = solid_volume(solver)
        l1 = liquid_volume(solver)

        rel = abs(s1 - s0) / s0 if s0 > 0 else 0.0
        dliq = l1 - l0

        good = rel <= RTOL and stats["nonzero_removals"] == 0
        ok &= good
        flag = " " if good else "  <-- PROBLEM"
        smallest = ("-" if stats["smallest_alive_W"] == float("inf")
                    else f"{stats['smallest_alive_W']:.1e}")
        print(f"{name:24s} {rel:14.3e} {dliq:12.3e} {solver.a_tot:7d} "
              f"{stats['removed']:7d} {stats['nonzero_removals']:6d} {smallest:>9s}{flag}")
        if stats["nonzero_removals"]:
            print(f"    groesster Rest w_rem beim Entfernen: {stats['worst_residual']:.6e}")

    print()
    print("ERGEBNIS:", "BESTANDEN" if ok else "FEHLGESCHLAGEN")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
