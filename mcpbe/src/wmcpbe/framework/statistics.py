"""Statistics across the repeats.

A single Monte Carlo run is a sample, not a result. Only the mean over several
independent runs -- together with a statement of how certain that mean is --
supports a claim.

The three quantities, and what separates them:

``std`` (standard deviation)
    How far do the *individual runs* scatter? Stays roughly the same as
    repeats are added -- it is a property of the model.
``sem`` (standard error, ``std/sqrt(N)``)
    How uncertain is the *mean*? Shrinks as repeats are added. This is the one
    that belongs on an error bar.
``ci95``
    Approximated as ``1.96 * sem``. For small N (below roughly 30) that is
    optimistic; for ten repeats Student's factor is closer to 2.26.

Convergence check
-----------------
If the runs really are independent, the standard error must fall as
``1/sqrt(N)`` -- a straight line of slope -0.5 on a log-log plot. Noticeably
less than that means the runs are correlated (the usual cause: a shared random
generator), and every error bar would be too small.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def stack_series(records: Sequence[Dict[str, Any]], key: str) -> np.ndarray:
    """Collect time series ``key`` from all repeats into an ``(N, T)`` matrix.

    Series of differing length are truncated to the shortest. That happens when
    a run ended early -- better to truncate visibly than to pad with zeros.
    """
    series = [np.asarray(r["series"][key], dtype=float) for r in records if key in r["series"]]
    if not series:
        return np.zeros((0, 0), dtype=float)
    n_t = min(len(s) for s in series)
    return np.stack([s[:n_t] for s in series], axis=0)


def summarize_series(records: Sequence[Dict[str, Any]], key: str) -> Dict[str, np.ndarray]:
    """Mean, scatter and uncertainty of one time series across all runs."""
    stacked = stack_series(records, key)
    if stacked.size == 0:
        return {}

    n = stacked.shape[0]
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0, ddof=1) if n > 1 else np.zeros(stacked.shape[1])
        p05 = np.nanpercentile(stacked, 5, axis=0)
        p95 = np.nanpercentile(stacked, 95, axis=0)
    sem = std / np.sqrt(n) if n > 1 else np.zeros_like(std)

    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95": 1.96 * sem,
        "min": np.nanmin(stacked, axis=0),
        "max": np.nanmax(stacked, axis=0),
        "p05": p05,
        "p95": p95,
        "n": np.full(stacked.shape[1], n, dtype=float),
    }


def summarize_scalar(records: Sequence[Dict[str, Any]], key: str) -> Dict[str, float]:
    """Mean, scatter and uncertainty of one scalar across all runs."""
    values = np.asarray(
        [float(r["scalars"][key]) for r in records if key in r["scalars"]], dtype=float
    )
    finite = values[np.isfinite(values)]
    n = int(finite.size)
    if n == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "sem": np.nan,
                "min": np.nan, "max": np.nan}
    std = float(np.std(finite, ddof=1)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": float(np.mean(finite)),
        "std": std,
        "sem": std / np.sqrt(n) if n > 1 else 0.0,
        "ci95": 1.96 * std / np.sqrt(n) if n > 1 else 0.0,
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def convergence_check(records: Sequence[Dict[str, Any]], key: str, at_index: int = -1) -> Dict[str, Any]:
    """Does the standard error fall like ``1/sqrt(N)``?

    Method: for N = 2, 3, ... compute the standard error from the first N runs
    and plot it against N on log-log axes. The slope should sit near -0.5.

    Limits of the statement: with ten repeats the slope is itself noisy.
    Anything between roughly -0.8 and -0.2 is unremarkable. A value near zero,
    on the other hand, means additional runs do not make the mean any more
    certain -- so they are not independent.
    """
    stacked = stack_series(records, key)
    if stacked.size == 0 or stacked.shape[0] < 3:
        return {"ok": None, "reason": "too few repeats to say anything",
                "slope": np.nan, "n_values": [], "sem_values": []}

    column = stacked[:, at_index]
    column = column[np.isfinite(column)]
    if column.size < 3:
        return {"ok": None, "reason": "too few finite values",
                "slope": np.nan, "n_values": [], "sem_values": []}

    n_values: List[int] = []
    sem_values: List[float] = []
    for n in range(2, column.size + 1):
        subset = column[:n]
        sem = float(np.std(subset, ddof=1) / np.sqrt(n))
        if sem > 0.0:
            n_values.append(n)
            sem_values.append(sem)

    if len(n_values) < 3:
        # Subsets with zero scatter are skipped. With few repeats and a
        # quantised quantity (n_phys only moves in multiples of the packet size
        # dW) too few points are left to fit a slope. That is a lack of data,
        # not a finding.
        return {
            "ok": None,
            "reason": (
                f"only {len(n_values)} usable subsets -- a slope needs at least 3 "
                "(more repeats required)"
            ),
            "slope": np.nan,
            "n_values": n_values,
            "sem_values": sem_values,
        }

    slope = float(np.polyfit(np.log(n_values), np.log(sem_values), 1)[0])
    return {
        "ok": bool(-0.85 <= slope <= -0.15),
        "slope": slope,
        "expected_slope": -0.5,
        "n_values": n_values,
        "sem_values": sem_values,
        "reason": "",
    }


def seeds_are_distinct(records: Sequence[Dict[str, Any]], key: str = "n_comp") -> Dict[str, Any]:
    """Do the seeds actually produce different runs?

    If every repeat is identical, the seed has no effect -- the typical cause
    being a submodule holding a frozen random generator. Any statistics would
    then be worthless even though everything ran without error.
    """
    stacked = stack_series(records, key)
    if stacked.shape[0] < 2:
        return {"ok": None, "reason": "only one repeat", "n_unique": stacked.shape[0]}
    unique_rows = np.unique(stacked, axis=0)
    n_unique = int(unique_rows.shape[0])
    return {
        "ok": n_unique == stacked.shape[0],
        "n_unique": n_unique,
        "n_total": int(stacked.shape[0]),
        "reason": "" if n_unique == stacked.shape[0] else "at least two runs are identical",
    }


def conservation_check(
    records: Sequence[Dict[str, Any]],
    key: str = "solid_conservation_error",
    tolerance: float = 1e-9,
) -> Dict[str, Any]:
    """Is a conserved quantity conserved across every seed?

    Mass must not depend on chance. If this error scatters over the seeds, that
    is a strong signal of a state-dependent defect -- exactly the kind that
    looks like noise in a single run.
    """
    stats = summarize_scalar(records, key)
    worst = max(abs(stats["min"]), abs(stats["max"])) if stats["n"] > 0 else np.nan
    return {
        "ok": bool(np.isfinite(worst) and worst <= tolerance),
        "worst_abs": float(worst),
        "tolerance": float(tolerance),
        **stats,
    }
