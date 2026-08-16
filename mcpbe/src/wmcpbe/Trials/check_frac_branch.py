"""Ist der Fraktional-Tropfen-Zweig durch meinen Umbau tot?

In _finalize_remaining_liquid, Zweig n_droplets_exact < 1.0, wird
_distribute_one_droplet_with_dW mit max_physical_droplets = n_droplets_exact < 1
aufgerufen. Meine Aenderung verwirft den Event, sobald dW > max_physical_droplets/vc_scale.
Da dW = min(batch_size, W_i) und beide typisch >= 1 sind, muesste das IMMER greifen.

Hier wird das direkt nachgemessen statt argumentiert.
"""
from __future__ import annotations

import os
import sys

import numpy as np

PKG = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
for p in (os.path.join(PKG, "src"), PKG):
    if p not in sys.path:
        sys.path.insert(0, p)

from tests.bench.scenarios import SCENARIOS, build  # noqa: E402


def main():
    sc = SCENARIOS["granulation_1d"]
    s = build(sc)
    nuc = s.nucleation
    # ein paar Events laufen lassen, damit ein realistischer Zustand vorliegt
    s.solve(maxiter=200)

    a = s.a_tot
    print(f"a_tot = {a}, W: min={s.W[:a].min():.4g} max={s.W[:a].max():.4g}")
    print(f"batch_size = {nuc.config.batch_size}")
    vc_scale = nuc._Vc_reference / s.Vc
    print(f"vc_scale = {vc_scale:.6g}")
    v_droplet = nuc.config.droplet_volume
    print(f"v_droplet = {v_droplet:.6e}\n")

    print("Direkter Aufruf mit fraktionalem Cap (wie im Fraktional-Zweig):")
    for frac in (0.999, 0.75, 0.5, 0.1, 0.01):
        # Zustand pro Versuch neu, damit Seiteneffekte sich nicht summieren
        s2 = build(sc)
        s2.solve(maxiter=200)
        n2 = s2.nucleation
        before = float(np.sum(s2.W[: s2.a_tot]))
        dW, v_eff = n2._distribute_one_droplet_with_dW(
            v_droplet=v_droplet, max_physical_droplets=frac
        )
        after = float(np.sum(s2.W[: s2.a_tot]))
        verdict = "VERWORFEN" if dW <= 0 else f"dW={dW:.6g}"
        print(f"   max_physical_droplets={frac:6.3f} -> {verdict:>14s}"
              f"   (Summe W {before:.4g} -> {after:.4g})")

    print("\nZum Vergleich, Cap >= 1 (Integer-Zweig):")
    for frac in (1.0, 5.0, 50.0):
        s2 = build(sc)
        s2.solve(maxiter=200)
        n2 = s2.nucleation
        dW, v_eff = n2._distribute_one_droplet_with_dW(
            v_droplet=v_droplet, max_physical_droplets=frac
        )
        verdict = "VERWORFEN" if dW <= 0 else f"dW={dW:.6g}"
        print(f"   max_physical_droplets={frac:6.3f} -> {verdict:>14s}")


if __name__ == "__main__":
    main()
