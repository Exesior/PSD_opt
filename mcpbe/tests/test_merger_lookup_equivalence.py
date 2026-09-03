"""``merger_lookup`` in {scan, hash, hash_lazy} must stay physically equivalent.

The knob (``mcpbe_base._MERGER_LOOKUP_MODES``) only changes HOW a duplicate
particle is found, never the matching *rule*. But it is NOT bit-for-bit neutral:
when several particles are within tolerance, ``_find_linear_scan`` returns the
lowest index while ``_find_via_hash`` returns whatever the Python set yields, so
a different particle gets the weight and the RNG path diverges from there. The
runs stay statistically equivalent -- mass exact, merge rate and PSD within ~1 %
-- which is what lets ``"scan"`` be the (faster, deterministic) default.

Asserted here:
  * ``hash`` and ``hash_lazy`` ARE bit-identical (hash_lazy only drops a
    redundant reindex pass).
  * ``scan`` conserves solid mass exactly and agrees with ``hash`` on merge
    rate / a_tot / mean diameter to a few percent.

See ``mcpbe/docs/historical/Merger_Lookup_Strategy_2026-09.md``.

Run standalone::

    python -m tests.test_merger_lookup_equivalence

or under pytest::

    pytest tests/test_merger_lookup_equivalence.py -v
"""

from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
import pytest

from .bench.scenarios import SCENARIOS, Scenario, build

MODES = ("scan", "hash", "hash_lazy")


def _fingerprint(solver) -> str:
    """SHA-256 over the exact float64 bytes of the final population state."""
    a = solver.a_tot
    h = hashlib.sha256()
    for arr in (
        solver.V_flat[:, :a],
        solver.W[:a],
        solver.liquid_volume[:a],
        np.nan_to_num(solver.porosity[:a], nan=-7.0),  # NaN = Vollkoerper
        solver.saturation[:a],
    ):
        x = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
        h.update(str(x.shape).encode("utf-8"))
        h.update(x.tobytes())
    return h.hexdigest()


def _solid_volume(solver) -> float:
    a = solver.a_tot
    poro = solver.porosity[:a]
    v_dry = solver.V_flat[-1, :a]
    v_solid = np.where(np.isnan(poro), v_dry, v_dry * (1.0 - np.nan_to_num(poro)))
    return float(np.sum(solver.W[:a] * v_solid))


def _run(scenario: Scenario, mode: str) -> dict:
    sc = dataclasses.replace(
        scenario,
        name=f"{scenario.name}__{mode}",
        solver_attrs={**scenario.solver_attrs, "merger_lookup": mode},
    )
    solver = build(sc)
    solid0 = _solid_volume(solver)
    solver.solve(maxiter=sc.maxiter)
    merger = solver._particle_merger
    a = solver.a_tot
    stats = merger.get_statistics() if merger is not None else {"merges": 0, "merge_rate": 0.0}
    return {
        "hash": _fingerprint(solver),
        "events": int(solver._iter_count),
        "a_tot": a,
        "merges": stats["merges"],
        "merge_rate": stats["merge_rate"],
        "use_hash": None if merger is None else merger.use_hash_index,
        "solid_err": abs(_solid_volume(solver) - solid0) / solid0,
        "d_mean": float(np.average(solver.X[:a] * 1e6, weights=solver.W[:a])),
    }


# granulation_1d: agg + nucleation + continuous processes -- the merger-heavy
# path. Run it longer than the benchmark (more events = a firmer guard), and
# add a mix variant so breakage-fragment merging is covered too.
_GRAN = dataclasses.replace(SCENARIOS["granulation_1d"], t_end=40.0, maxiter=600)
_GRAN_BREAK = dataclasses.replace(
    _GRAN,
    name="granulation_break_1d",
    process_type="mix",
    break_kernel_name="power_law",
    break_kernel_params={"p1": 1e-2, "p2": 1.0, "g": 1000.0, "breakrval": 1},
    solver_attrs={"BREAKFVAL": 2, "break_dW_max": 50.0},
)

CASES = [_GRAN, _GRAN_BREAK]


@pytest.mark.parametrize("scenario", CASES, ids=lambda s: s.name)
def test_merger_lookup_modes_are_equivalent(scenario):
    results = {mode: _run(scenario, mode) for mode in MODES}

    # the three modes really are different mechanisms
    assert results["scan"]["use_hash"] is False
    assert results["hash"]["use_hash"] is True
    assert results["hash_lazy"]["use_hash"] is True

    # the merger actually did work here, otherwise the test proves nothing
    assert results["scan"]["merges"] > 0

    # hash and hash_lazy differ only by a redundant reindex pass -> bit-identical
    assert results["hash_lazy"]["hash"] == results["hash"]["hash"], (
        f"{scenario.name}: hash_lazy is not bit-identical to hash"
    )

    # solid mass is conserved exactly in every mode (V_solid is added, not merged)
    for mode, r in results.items():
        assert r["solid_err"] < 1e-9, f"{scenario.name}/{mode}: solid mass drift {r['solid_err']:.2e}"

    # scan vs hash: statistically equivalent, not bit-identical
    s, h = results["scan"], results["hash"]
    assert abs(s["merge_rate"] - h["merge_rate"]) < 0.02, (
        f"{scenario.name}: merge rate {s['merge_rate']:.3f} (scan) vs "
        f"{h['merge_rate']:.3f} (hash)"
    )
    assert abs(s["a_tot"] - h["a_tot"]) <= max(5, 0.05 * h["a_tot"]), (
        f"{scenario.name}: a_tot {s['a_tot']} (scan) vs {h['a_tot']} (hash)"
    )
    assert abs(s["d_mean"] - h["d_mean"]) / h["d_mean"] < 0.03, (
        f"{scenario.name}: d_mean {s['d_mean']:.3f} (scan) vs {h['d_mean']:.3f} (hash)"
    )


def main() -> bool:
    ok = True
    for scenario in CASES:
        print(f"\n{scenario.name}")
        results = {mode: _run(scenario, mode) for mode in MODES}
        for mode, r in results.items():
            print(f"  {mode:<10} events={r['events']:<5} a_tot={r['a_tot']:<5} "
                  f"merge%={r['merge_rate']*100:5.2f}  d_mean={r['d_mean']:.4f}um  "
                  f"solid_err={r['solid_err']:.1e}  {r['hash'][:12]}")
        s, h = results["scan"], results["hash"]
        bit_ident = s["hash"] == h["hash"]
        stat_ok = (
            results["hash_lazy"]["hash"] == h["hash"]
            and all(r["solid_err"] < 1e-9 for r in results.values())
            and abs(s["merge_rate"] - h["merge_rate"]) < 0.02
            and abs(s["d_mean"] - h["d_mean"]) / h["d_mean"] < 0.03
        )
        print(f"  -> scan vs hash: {'bit-identical' if bit_ident else 'statistically equivalent'}"
              f"; checks {'PASS' if stat_ok else 'FAIL'}")
        ok = ok and stat_ok
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if main() else 1)
