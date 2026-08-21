"""Event clock and batch-size bookkeeping for the weighted MC-PBE solver.

Every Monte Carlo step needs two numbers, and both are derived here:

``delta_i`` -- how much weight one event may consume from particle ``i``
    ``delta_i = min(dW_const, W_i)``, with ``dW_const`` the per-process cap
    (``agg_dW_max``, ``break_dW_max``). ``delta`` is part of the rate, not a
    clamp applied afterwards: the bias-corrected propensity divides by
    ``min(delta_i, delta_j)``. Raising a cap therefore changes the sampling
    distribution, not just the step size.

``dt`` -- how much simulated time passes before the next event
    Gillespie-style, from the summed propensity of the whole population.

Time-step conversions
---------------------
*Breakage* is a single-particle process; its propensity array already holds
rates in 1/s, so ``dt = 1 / sum``.

*Agglomeration* is a pair process, so the conversion carries two extra
factors::

    dt_agg = 2 * Vc * (a_tot - 1) / (a_tot * sum_prop)

``Vc`` turns the pair count into a concentration-based rate, the ``2``
compensates for every unordered pair being counted twice by the double sum,
and ``(a_tot-1)/a_tot`` removes the diagonal ``i == j`` from it.
:func:`agg_rate_from_sum_prop` is the exact reciprocal.

Two switches change the clock
-----------------------------
``exp_time_step`` (default ``False``)
    Off, ``dt`` is the mean waiting time. On, it is multiplied by ``-ln(u)``,
    i.e. a real exponential draw.

``sum_prop_pair`` (default ``False``)
    Off, ``dt`` uses the propensity from before the event. On, the logarithmic
    mean of before and after -- a midpoint correction for the fact that the
    event changes the propensity it was sampled from.

Layout
------
Free functions first: pure, numbers in and numbers out, no solver attribute is
read. ``MCPBETimeHelper`` below binds them to live solver state (``a_tot``,
``Vc``, ``W``, ``_r_agg``, ``_break_rate``) and adds the strategy builders that
``solve()`` uses.

Former complications and the derivation of the current form:
``mcpbe/docs/historical/Bias_Correction_und_Gewichtsdisziplin.md``.
"""

from __future__ import annotations

import math

import numpy as np


# ===========================================================================
# Layer 1 -- pure helpers
# ===========================================================================

def ensure_delta_array(solver, attr_name: str) -> np.ndarray:
    """Return ``solver.<attr_name>``, allocating it if missing or too short.

    First-allocation fallback only. Once the array exists,
    ``MCPBEBase._ensure_capacity_for`` grows it with the other per-particle
    arrays and copies the contents across.
    """
    arr = getattr(solver, attr_name, None)
    cap = int(getattr(solver, "_cap", 0))
    if arr is None or arr.shape[0] < cap:
        # Minimum size, so an empty solver still has a valid array to index.
        arr = np.zeros(max(8, cap), dtype=float)
        setattr(solver, attr_name, arr)
    return arr


def prepare_process_delta_config(
    solver,
    *,
    dW_attr_name: str,
    cache_attr_name: str,
    default_value: float,
) -> float:
    """Read a process's batch cap, sanitise it, and cache it on the solver.

    The cap is user-settable (``agg_dW_max`` / ``break_dW_max``) but read on
    every propensity rebuild, hence the cache under ``cache_attr_name``.
    Non-finite or non-positive values fall back to ``default_value``.
    """
    dW_const = float(getattr(solver, dW_attr_name, default_value))
    if (not np.isfinite(dW_const)) or dW_const <= 0.0:
        dW_const = float(default_value)
    setattr(solver, cache_attr_name, dW_const)
    return dW_const


def delta_from_weights(W: np.ndarray, dW_const: float) -> np.ndarray:
    """Vectorised ``delta_i = min(dW_const, W_i)``, zeroed where ``W`` is unusable.

    ``W > 0.0`` is an exact test to catch edge cases.

    Former complications and the derivation:
    ``mcpbe/docs/historical/Bias_Correction_und_Gewichtsdisziplin.md``.
    """
    delta = np.minimum(np.asarray(W, dtype=float), float(dW_const))
    delta = np.where(np.isfinite(delta) & (delta > 0.0), delta, 0.0)
    return delta


def update_delta_single(
    solver,
    i: int,
    *,
    attr_name: str,
    dW_const: float,
) -> float:
    """Recompute ``delta`` for the single particle ``i`` and store it.

    O(1) counterpart to :func:`delta_from_weights`, for use after an event has
    changed one or two weights. Out-of-range indices return ``0.0``; callers
    may hold an index that a swap-with-last removal has invalidated.
    """
    arr = ensure_delta_array(solver, attr_name)
    a_tot = int(getattr(solver, "a_tot", 0))
    if i < 0 or i >= a_tot:
        return 0.0
    Wi = float(solver.W[i])
    delta = min(float(dW_const), Wi) if (np.isfinite(Wi) and Wi > 0.0) else 0.0
    arr[i] = delta
    return float(delta)


# ---------------------------------------------------------------------------
# Time-step formulas
# ---------------------------------------------------------------------------

def log_mean_positive(x: float, y: float) -> float:
    """Logarithmic mean ``(y - x) / ln(y / x)``; ``NaN`` unless both are positive.

    Equal inputs return the arithmetic mean, which is the limit of the
    singular closed form.
    """
    if (not np.isfinite(x)) or (not np.isfinite(y)) or x <= 0.0 or y <= 0.0:
        return float("nan")
    if np.isclose(x, y, rtol=1e-12, atol=0.0):
        return 0.5 * (x + y)
    return (y - x) / math.log(y / x)


def dt_agg_from_sum_prop(a_tot: int, Vc: float, sum_prop: float) -> float:
    """Agglomeration waiting time from the summed pair propensity.

    ``inf`` means the process cannot fire; ``solve()`` handles that through the
    normal timer comparison.
    """
    if a_tot < 2 or sum_prop <= 0.0:
        return float("inf")
    return 2.0 * float(Vc) * (a_tot - 1) / (a_tot * sum_prop)


def dt_agg_from_sum_prop_pair(a_tot: int, Vc: float, sum_prop_before: float, sum_prop_after: float) -> float:
    """:func:`dt_agg_from_sum_prop` using the log-mean of before/after propensity."""
    if a_tot < 2:
        return float("inf")
    prop_eff = log_mean_positive(float(sum_prop_before), float(sum_prop_after))
    # NaN here means one of the two propensities was non-positive, i.e. the
    # population died out on this event.
    if (not np.isfinite(prop_eff)) or prop_eff <= 0.0:
        return float("inf")
    return 2.0 * float(Vc) * (a_tot - 1) / (a_tot * prop_eff)


def dt_break_from_sum_prop(sum_prop: float) -> float:
    """Breakage waiting time: the reciprocal of the summed rate."""
    if sum_prop <= 0.0:
        return float("inf")
    return 1.0 / sum_prop


def dt_break_from_sum_prop_pair(sum_prop_before: float, sum_prop_after: float) -> float:
    """:func:`dt_break_from_sum_prop` using the log-mean of before/after rate."""
    prop_eff = log_mean_positive(float(sum_prop_before), float(sum_prop_after))
    if (not np.isfinite(prop_eff)) or prop_eff <= 0.0:
        return float("inf")
    return 1.0 / prop_eff


def agg_rate_from_sum_prop(a_tot: int, Vc: float, sum_prop: float) -> float:
    """Agglomeration rate [1/s], the reciprocal of :func:`dt_agg_from_sum_prop`.

    ``mix`` mode needs the rate, because it compares agglomeration against
    breakage on a common scale before drawing which of the two fires.
    """
    if a_tot < 2 or sum_prop <= 0.0:
        return 0.0
    return float(a_tot) * float(sum_prop) / (2.0 * float(Vc) * (float(a_tot) - 1.0))


def mix_total_rate_from_sum_prop(a_tot: int, Vc: float, sum_prop_agg: float, sum_prop_break: float) -> float:
    """Combined event rate in ``mix`` mode.

    The breakage sum is already a rate, the agglomeration sum is not -- hence
    the conversion on one side only.
    """
    return agg_rate_from_sum_prop(a_tot, Vc, sum_prop_agg) + max(float(sum_prop_break), 0.0)


# ===========================================================================
# Layer 2 -- solver-bound mixin
# ===========================================================================

class MCPBETimeHelper:
    """Mixin providing the event clock to :class:`MCPBEBase`.

    The ``_build_*_dt_strategy`` methods are what ``solve()`` uses: they
    resolve the ``exp_time_step`` / ``sum_prop_pair`` switches once, before the
    loop starts, and return two closures. The per-event hot path then calls a
    closure instead of re-checking both flags on every iteration.
    """

    def _draw_time_multiplier(self) -> float:
        """``1.0``, or an exponential draw ``-ln(u)`` if ``exp_time_step`` is set.

        ``u`` is floored at 1e-300 so a drawn zero cannot produce ``inf``.
        """
        if not bool(getattr(self, "exp_time_step", False)):
            return 1.0
        rng = getattr(self, "_rng", None)
        if rng is None:
            return 1.0
        u = max(float(rng.random()), 1e-300)
        return -math.log(u)

    def _build_agg_dt_strategy(self):
        """Return ``(initial_dt, event_dt)`` closures for agglomeration.

        ``initial_dt`` seeds the timer before the loop, ``event_dt`` advances
        it after each event. Both take propensity sums rather than the solver,
        so ``solve()`` decides when they are read.
        """
        use_pair = bool(getattr(self, "sum_prop_pair", False))

        def initial_dt(sum_prop: float) -> float:
            return self._dt_agg_from_sum_prop(float(sum_prop)) * self._draw_time_multiplier()

        if use_pair:
            def event_dt(sum_prop_before: float, sum_prop_after: float) -> float:
                return self._dt_agg_from_sum_prop_pair(
                    float(sum_prop_before), float(sum_prop_after)
                ) * self._draw_time_multiplier()
        else:
            # Same signature either way, so solve() keeps one call site.
            def event_dt(sum_prop_before: float, _sum_prop_after: float) -> float:
                return self._dt_agg_from_sum_prop(float(sum_prop_before)) * self._draw_time_multiplier()

        return initial_dt, event_dt

    def _build_break_dt_strategy(self):
        """Return ``(initial_dt, event_dt)`` closures for breakage."""
        use_pair = bool(getattr(self, "sum_prop_pair", False))

        def initial_dt(sum_prop: float) -> float:
            return self._dt_break_from_sum_prop(float(sum_prop)) * self._draw_time_multiplier()

        if use_pair:
            def event_dt(sum_prop_before: float, sum_prop_after: float) -> float:
                return self._dt_break_from_sum_prop_pair(
                    float(sum_prop_before), float(sum_prop_after)
                ) * self._draw_time_multiplier()
        else:
            def event_dt(sum_prop_before: float, _sum_prop_after: float) -> float:
                return self._dt_break_from_sum_prop(float(sum_prop_before)) * self._draw_time_multiplier()

        return initial_dt, event_dt

    def _build_mix_dt_strategy(self):
        """Return ``(initial_dt, event_dt)`` closures for ``mix`` mode.

        These take an already-combined rate from
        :meth:`_mix_total_rate_from_sum_prop`, not a propensity sum.
        """
        use_pair = bool(getattr(self, "sum_prop_pair", False))

        def initial_dt(total_rate: float) -> float:
            if total_rate <= 0.0:
                return float("inf")
            return (1.0 / float(total_rate)) * self._draw_time_multiplier()

        if use_pair:
            def event_dt(total_rate_before: float, total_rate_after: float) -> float:
                prop_eff = self._log_mean_positive(float(total_rate_before), float(total_rate_after))
                if (not np.isfinite(prop_eff)) or prop_eff <= 0.0:
                    return float("inf")
                return (1.0 / prop_eff) * self._draw_time_multiplier()
        else:
            def event_dt(total_rate_before: float, _total_rate_after: float) -> float:
                if total_rate_before <= 0.0:
                    return float("inf")
                return (1.0 / float(total_rate_before)) * self._draw_time_multiplier()

        return initial_dt, event_dt

    # -----------------------------------------------------------------------
    # Batch caps
    # -----------------------------------------------------------------------

    def _prepare_agg_delta_config(self) -> float:
        """Allocate ``_delta_agg`` and cache the agglomeration batch cap (default ``1.0``)."""
        ensure_delta_array(self, "_delta_agg")
        return prepare_process_delta_config(
            self,
            dW_attr_name="agg_dW_max",
            cache_attr_name="_agg_dW_const",
            default_value=1.0,
        )

    def _prepare_break_delta_config(self) -> float:
        """Allocate ``_delta_break`` and cache the breakage batch cap (default ``50.0``).

        Intentionally asymmetric to agglomeration's ``1.0``. See
        ``mcpbe/docs/historical/Audit_2026-08-17.md``, B-06.
        """
        ensure_delta_array(self, "_delta_break")
        return prepare_process_delta_config(
            self,
            dW_attr_name="break_dW_max",
            cache_attr_name="_break_dW_const",
            default_value=50.0,
        )

    def _delta_from_weights(self, W: np.ndarray, *, dW_const: float) -> np.ndarray:
        return delta_from_weights(W, dW_const)

    def _update_delta_single(self, i: int, *, attr_name: str, dW_const: float) -> float:
        return update_delta_single(self, i, attr_name=attr_name, dW_const=dW_const)

    @staticmethod
    def _log_mean_positive(x: float, y: float) -> float:
        return log_mean_positive(x, y)

    # -----------------------------------------------------------------------
    # Solver-bound time steps
    #
    # Wrappers that pull `a_tot` and `Vc` off the instance. They take the
    # propensity sum as an argument because `solve()` already holds it, from
    # the Fenwick sampler's cached total.
    # -----------------------------------------------------------------------

    def _dt_agg_from_sum_prop(self, sum_prop: float) -> float:
        return dt_agg_from_sum_prop(int(self.a_tot), float(self.Vc), float(sum_prop))

    def _dt_agg_from_sum_prop_pair(self, sum_prop_before: float, sum_prop_after: float) -> float:
        return dt_agg_from_sum_prop_pair(
            int(self.a_tot), float(self.Vc), float(sum_prop_before), float(sum_prop_after)
        )

    def _dt_break_from_sum_prop(self, sum_prop: float) -> float:
        return dt_break_from_sum_prop(float(sum_prop))

    def _dt_break_from_sum_prop_pair(self, sum_prop_before: float, sum_prop_after: float) -> float:
        return dt_break_from_sum_prop_pair(float(sum_prop_before), float(sum_prop_after))

    def _agg_rate_from_sum_prop(self, sum_prop: float) -> float:
        return agg_rate_from_sum_prop(int(self.a_tot), float(self.Vc), float(sum_prop))

    def _mix_total_rate_from_sum_prop(self, sum_prop_agg: float, sum_prop_break: float) -> float:
        return mix_total_rate_from_sum_prop(
            int(self.a_tot), float(self.Vc), float(sum_prop_agg), float(sum_prop_break)
        )
