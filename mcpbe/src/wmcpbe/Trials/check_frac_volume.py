"""Liefert der reparierte Fraktional-Zweig exakt das gewuenschte Volumen?

Soll:  abgegebene physikalische Fluessigkeit == max_physical_droplets * v_droplet
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


def liquid_total(s):
    a = s.a_tot
    return float(np.sum(s.liquid_volume[:a] * s.W[:a]))


def main():
    sc = SCENARIOS["granulation_1d"]
    v_droplet = None
    print(f"{'cap':>8s} {'soll [m^3]':>14s} {'ist [m^3]':>14s} {'rel. Fehler':>13s} {'dW':>6s}")
    print("-" * 60)
    worst = 0.0
    for frac in (0.999, 0.75, 0.5, 0.1, 0.01, 1.0, 3.0):
        s = build(sc)
        s.solve(maxiter=200)
        nuc = s.nucleation
        if v_droplet is None:
            v_droplet = nuc.config.droplet_volume
        vc_scale = nuc._Vc_reference / s.Vc

        before = liquid_total(s)
        dW = nuc._distribute_one_droplet_with_dW(
            v_droplet=v_droplet, max_physical_droplets=frac
        )
        after = liquid_total(s)

        ist = after - before
        # Bei cap >= dW*vc_scale wird nicht skaliert -> volle Tropfenmenge je Partikel
        soll = min(frac, dW * vc_scale) * v_droplet
        rel = abs(ist - soll) / soll if soll > 0 else 0.0

        # `ist` ist eine Differenz zweier grosser Summen (Gesamtfluessigkeit ~ before).
        # Deren Auauschloeschungsrauschen liegt bei ~eps*before, relativ zum winzigen
        # Sollwert also bei eps*before/soll. Ein reines rel<1e-12 waere hier kein
        # Aussage ueber den Code, sondern ueber die Messmethode.
        noise = 8.0 * np.finfo(float).eps * max(before, abs(after)) / soll if soll > 0 else 0.0
        ok = rel <= max(1e-12, noise)
        worst = max(worst, rel / max(noise, 1e-30))
        print(f"{frac:8.3f} {soll:14.6e} {ist:14.6e} {rel:13.3e} {dW:6.3g}"
              f"   Rauschgrenze {noise:.1e} {'OK' if ok else 'FAIL'}")

    print()
    print(f"groesstes Verhaeltnis Fehler/Rauschgrenze: {worst:.3f}  (<= 1 heisst: im Rauschen)")
    print("ERGEBNIS:", "BESTANDEN" if worst <= 1.0 else "FEHLGESCHLAGEN")
    return 0 if worst <= 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
