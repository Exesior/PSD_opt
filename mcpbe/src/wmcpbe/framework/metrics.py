"""Reducing a solved solver to a compact set of numbers.

These functions run **in the worker process**, right after ``solve()``. That is
deliberate: a finished solver carries the complete particle history with it
(2000 particles over 501 time points reach hundreds of megabytes quickly), and
anything crossing a process boundary has to be serialised. Sending the solver
back would spend exactly the time that parallelising just saved.

What travels back is only what the evaluation needs: a few series of length T
and a handful of scalars -- a few kilobytes.

Two kinds of mean
-----------------
Porosity, saturation and diameter are computed both ways:

``*_mean``
    Unweighted mean over the computational particles. Matches what
    ``Trials/test_sensitivity.py`` writes today, kept so existing CSVs stay
    comparable.
``*_wmean``
    Weighted by the DSMC weight ``W``. This is the physically correct one: a
    computational particle with W=600 stands for 600 real particles and must
    not count the same as one with W=1.

Use the ``*_wmean`` columns for any physical statement.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


# Series returned per repeat. This order is the column order in the CSV files.
SERIES_KEYS = (
    "n_comp",              # number of computational particles [-]
    "n_phys",              # physical number density [1/m^3]
    "M0",                  # zeroth moment (sum of weights)
    "M1_m3",               # first moment (total dry volume) [m^3]
    "M2_m6",               # second moment [m^6]
    "porosity_mean",       # unweighted
    "porosity_wmean",      # W-weighted
    "saturation_mean",
    "saturation_wmean",
    "diameter_mean_m",     # unweighted, from V_dry
    "diameter_wmean_m",    # W-weighted
    "liquid_in_system_m3", # liquid held in particles [m^3]
    "solid_volume_m3",     # solid volume [m^3] -- must stay constant
    "Vc_m3",               # control volume [m^3]
    "agg_events",          # cumulative, dW-weighted agglomeration events
    "break_events",        # cumulative, dW-weighted breakage events
)


def _safe_mean(values: np.ndarray) -> float:
    """Mean over the finite entries; 0.0 if none are finite."""
    finite = np.isfinite(values)
    if not np.any(finite):
        return 0.0
    return float(np.mean(values[finite]))


def _safe_wmean(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean over the finite entries."""
    finite = np.isfinite(values) & np.isfinite(weights)
    if not np.any(finite):
        return 0.0
    w = weights[finite]
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        return 0.0
    return float(np.sum(values[finite] * w) / w_sum)


def _volume_to_diameter(volume: np.ndarray) -> np.ndarray:
    """Sphere-equivalent diameter from a volume."""
    v = np.clip(np.asarray(volume, dtype=float), 0.0, None)
    return np.cbrt(6.0 * v / np.pi)


def extract_series(solver: Any) -> Dict[str, np.ndarray]:
    """Build the time series from the solver's ``*_save`` lists.

    Length follows the shortest list present. If a run ended early the series
    are correspondingly shorter -- which shows up in the statistics, rather
    than being padded with zeros and hidden.
    """
    n_t = min(
        len(solver.V_save),
        len(solver.W_save),
        len(solver.Vc_save),
        len(solver.porosity_save),
        len(solver.saturation_save),
        len(solver.liquid_volume_save),
        len(solver.t_vec),
    )

    out: Dict[str, np.ndarray] = {key: np.zeros(n_t, dtype=float) for key in SERIES_KEYS}
    out["t_s"] = np.asarray(solver.t_vec[:n_t], dtype=float)

    agg_save = getattr(solver, "real_agg_events_save", None)
    break_save = getattr(solver, "real_break_events_save", None)

    for it in range(n_t):
        v_dry = np.asarray(solver.V_save[it][-1, :], dtype=float)
        weights = np.asarray(solver.W_save[it], dtype=float)
        poro = np.asarray(solver.porosity_save[it], dtype=float)
        sat = np.asarray(solver.saturation_save[it], dtype=float)
        liq = np.asarray(solver.liquid_volume_save[it], dtype=float)
        Vc = float(solver.Vc_save[it])

        # Solid volume: particles with no defined porosity count as Vollkoerper
        # (same convention as test_powerlaw_rumpf_full.py).
        valid_poro = np.isfinite(poro)
        v_solid = np.where(valid_poro, v_dry * (1.0 - np.where(valid_poro, poro, 0.0)), v_dry)

        diameters = _volume_to_diameter(v_dry)

        out["n_comp"][it] = float(len(weights))
        out["M0"][it] = float(np.sum(weights))
        out["M1_m3"][it] = float(np.sum(weights * v_dry))
        out["M2_m6"][it] = float(np.sum(weights * v_dry ** 2))
        out["n_phys"][it] = out["M0"][it] / Vc if Vc > 0.0 else np.nan
        out["porosity_mean"][it] = _safe_mean(poro)
        out["porosity_wmean"][it] = _safe_wmean(poro, weights)
        out["saturation_mean"][it] = _safe_mean(sat)
        out["saturation_wmean"][it] = _safe_wmean(sat, weights)
        out["diameter_mean_m"][it] = _safe_mean(diameters)
        out["diameter_wmean_m"][it] = _safe_wmean(diameters, weights)
        out["liquid_in_system_m3"][it] = float(np.sum(liq * weights))
        out["solid_volume_m3"][it] = float(np.sum(v_solid * weights))
        out["Vc_m3"][it] = Vc
        out["agg_events"][it] = float(agg_save[it]) if agg_save is not None and it < len(agg_save) else np.nan
        out["break_events"][it] = float(break_save[it]) if break_save is not None and it < len(break_save) else np.nan

    return out


def extract_scalars(solver: Any, series: Dict[str, np.ndarray], params: Dict[str, Any]) -> Dict[str, float]:
    """Whole-run figures, one number per repeat.

    Includes the two conservation errors. They are the acceptance criterion:
    if they scatter across seeds, something is wrong -- mass must not depend on
    chance.
    """
    from .builder import expected_droplet_count, expected_liquid_volume

    # The balances are computed from the solver's LIVE end state, not from the
    # last snapshot. The loop keeps running past the final save point until the
    # next event, so the snapshot at t_vec[-1] sits slightly before the true
    # end. Mixing the two instants makes a perfect balance look like a loss of
    # several percent.
    a_tot = int(solver.a_tot)
    w_live = np.asarray(solver.W[:a_tot], dtype=float)
    v_dry_live = np.asarray(solver.V_flat[-1, :a_tot], dtype=float)
    poro_live = np.asarray(solver.porosity[:a_tot], dtype=float)
    liq_live = np.asarray(solver.liquid_volume[:a_tot], dtype=float)

    valid_poro = np.isfinite(poro_live)
    v_solid_live = np.where(
        valid_poro, v_dry_live * (1.0 - np.where(valid_poro, poro_live, 0.0)), v_dry_live
    )

    solid = series["solid_volume_m3"]
    solid0 = float(solid[0]) if len(solid) else np.nan
    solid_end = float(np.sum(v_solid_live * w_live))
    solid_error = (solid_end - solid0) / solid0 if solid0 not in (0.0, np.nan) else np.nan

    liquid_expected = expected_liquid_volume(params)
    liquid_end = float(np.sum(liq_live * w_live))

    droplets_expected = expected_droplet_count(params)
    droplets_actual = np.nan
    liquid_added = np.nan
    nucleation = getattr(solver, "nucleation", None)
    if nucleation is not None and hasattr(nucleation, "get_statistics"):
        try:
            stats = nucleation.get_statistics()
            droplets_actual = float(stats.get("droplets_added_total", np.nan))
            liquid_added = float(stats.get("liquid_volume_added_total", np.nan))
        except Exception:
            droplets_actual = np.nan
            liquid_added = np.nan

    # Does the liquid that was actually added stay in the system? Checked
    # against the ADDED amount on purpose, not against the amount expected for
    # the full window -- otherwise every run that ends before nucleation is
    # finished would fail, although nothing was lost.
    if np.isfinite(liquid_added) and liquid_added > 0.0:
        liquid_error = (liquid_end - liquid_added) / liquid_added
    else:
        liquid_error = np.nan

    # Dosing: was the planned amount added at all? Only meaningful if the
    # simulation covers the whole nucleation window.
    covers_window = float(params.get("t_total", 0.0)) >= float(params.get("nucleation_duration", 0.0))
    if covers_window and liquid_expected > 0.0 and np.isfinite(liquid_added):
        liquid_dose_error = (liquid_added - liquid_expected) / liquid_expected
    else:
        liquid_dose_error = np.nan

    return {
        "n_comp_final": float(series["n_comp"][-1]) if len(series["n_comp"]) else np.nan,
        "n_phys_final": float(series["n_phys"][-1]) if len(series["n_phys"]) else np.nan,
        "diameter_wmean_final_m": float(series["diameter_wmean_m"][-1]) if len(series["diameter_wmean_m"]) else np.nan,
        "porosity_wmean_final": float(series["porosity_wmean"][-1]) if len(series["porosity_wmean"]) else np.nan,
        "saturation_wmean_final": float(series["saturation_wmean"][-1]) if len(series["saturation_wmean"]) else np.nan,
        "solid_volume_initial_m3": solid0,
        "solid_volume_final_m3": solid_end,
        "solid_conservation_error": float(solid_error),
        "liquid_expected_m3": float(liquid_expected),
        "liquid_added_m3": float(liquid_added),
        "liquid_final_m3": liquid_end,
        "liquid_conservation_error": float(liquid_error),
        "liquid_dose_error": float(liquid_dose_error),
        "droplets_expected": float(droplets_expected),
        "droplets_actual": float(droplets_actual),
        "agg_events_total": float(getattr(solver, "real_agg_events", np.nan)),
        "break_events_total": float(getattr(solver, "real_break_events", np.nan)),
    }


def reduce_solver(solver: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """The whole reduction: solver in, small data packet out."""
    series = extract_series(solver)
    scalars = extract_scalars(solver, series, params)
    return {"series": series, "scalars": scalars}
