"""Bit-exact regression fingerprints for the WMCPBE solver.

.. note::
   ``tests/golden_reference.OUTDATED.json`` (2026-07-31) is stale -- it no longer
   matches any scenario after the mixer-speed integration, F-07 and later work.
   Record a fresh baseline on a known-good state before relying on ``check``.

Usage
-----
Record a baseline (run this on the *unmodified* code first)::

    python -m tests.bench.golden_reference record --out baseline.json

Check the current code against a baseline::

    python -m tests.bench.golden_reference check --ref baseline.json

The fingerprint is a SHA-256 over the exact float64 bytes of every saved
snapshot plus the derived moments, so any change in the RNG stream or in the
arithmetic shows up immediately. A refactor that is meant to be
behaviour-preserving must leave every fingerprint untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from typing import Any, Dict, List

import numpy as np

from .scenarios import SCENARIOS, make_solver, warmup


def _hash_arrays(chunks: List[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for arr in chunks:
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
        digest.update(str(a.shape).encode("utf-8"))
        digest.update(a.tobytes())
    return digest.hexdigest()


def fingerprint(name: str) -> Dict[str, Any]:
    """Run one scenario and return its fingerprint + timing."""
    solver = make_solver(name)
    scenario = SCENARIOS[name]

    t0 = time.perf_counter()
    solver.solve(maxiter=scenario.maxiter)
    elapsed = time.perf_counter() - t0

    chunks: List[np.ndarray] = []
    for snap_name in (
        "V_save",
        "W_save",
        "liquid_volume_save",
        "porosity_save",
        "saturation_save",
    ):
        for snap in getattr(solver, snap_name, []):
            # NaN is a legitimate porosity value ("Vollkoerper"); map it to a
            # fixed sentinel so the hash stays stable across NaN payloads.
            chunks.append(np.nan_to_num(np.asarray(snap, dtype=np.float64), nan=-7.0))
    chunks.append(np.asarray(solver.Vc_save, dtype=np.float64))
    chunks.append(np.asarray(solver.t_right, dtype=np.float64))

    # Final live state. Snapshots alone are not enough because the event budget
    # (Scenario.maxiter) can stop the run before every save time is reached.
    a = solver.a_tot
    W = solver.W[:a]
    v_dry = solver.V_flat[-1, :a]
    poro = solver.porosity[:a]
    chunks.append(np.ascontiguousarray(solver.V_flat[:, :a]))
    chunks.append(W)
    chunks.append(solver.liquid_volume[:a])
    chunks.append(np.nan_to_num(poro, nan=-7.0))
    chunks.append(solver.saturation[:a])

    v_solid = np.where(np.isnan(poro), v_dry, v_dry * (1.0 - np.nan_to_num(poro)))

    return {
        "scenario": name,
        "hash": _hash_arrays(chunks),
        "events": int(solver._iter_count),
        "a_tot": int(a),
        "sim_time": float(solver._elapsed),
        "sum_W": float(np.sum(W)),
        "solid_volume": float(np.sum(W * v_solid)),
        "liquid_volume": float(np.sum(W * solver.liquid_volume[:a])),
        "seconds": elapsed,
    }


def run_all(names: List[str]) -> List[Dict[str, Any]]:
    out = []
    for name in names:
        print(f"  running {name} ...", end="", flush=True)
        rec = fingerprint(name)
        print(f" {rec['seconds']:8.2f}s  events={rec['events']:>7d}  {rec['hash'][:16]}")
        out.append(rec)
    return out


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("record", "check"))
    parser.add_argument("--out", default="golden_reference.json")
    parser.add_argument("--ref", default="golden_reference.json")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Restrict to these scenario names (default: all).",
    )
    args = parser.parse_args(argv)

    names = args.only or list(SCENARIOS)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        parser.error(f"unknown scenario(s): {unknown}")

    print("[golden] warming up JIT ...", flush=True)
    warmup()
    print(f"[golden] {args.mode}: {len(names)} scenario(s)")
    records = run_all(names)

    if args.mode == "record":
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2)
        print(f"[golden] wrote {args.out}")
        return 0

    with open(args.ref, "r", encoding="utf-8") as fh:
        ref = {r["scenario"]: r for r in json.load(fh)}

    failures = []
    print()
    print(f"{'scenario':<24} {'status':<8} {'ref [s]':>9} {'now [s]':>9} {'speedup':>8}")
    print("-" * 64)
    for rec in records:
        name = rec["scenario"]
        base = ref.get(name)
        if base is None:
            print(f"{name:<24} {'NO-REF':<8}")
            continue
        ok = base["hash"] == rec["hash"]
        speedup = base["seconds"] / rec["seconds"] if rec["seconds"] > 0 else float("nan")
        status = "OK" if ok else "MISMATCH"
        print(
            f"{name:<24} {status:<8} {base['seconds']:9.2f} {rec['seconds']:9.2f} {speedup:7.2f}x"
        )
        if not ok:
            failures.append((name, base, rec))

    if failures:
        print("\n[golden] FINGERPRINT MISMATCHES")
        for name, base, rec in failures:
            print(f"  {name}:")
            for key in ("events", "a_tot", "sum_W", "solid_volume", "liquid_volume"):
                if base[key] != rec[key]:
                    print(f"    {key}: {base[key]!r} -> {rec[key]!r}")
        return 1

    print("\n[golden] all fingerprints match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
