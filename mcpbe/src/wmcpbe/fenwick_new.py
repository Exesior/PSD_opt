"""Fenwick tree (binary indexed tree) sampler for dynamic non-negative weights.

Used to draw a particle index with probability proportional to its propensity
in O(log n), and to keep that distribution up to date in O(log n) per weight
change.

Performance notes
-----------------
The solver rebuilds the whole tree after every accepted event, so this class is
on the hot path. Two things matter:

* **No reallocation.** Weights and tree nodes live in geometrically grown
  buffers; the logical size ``n`` is tracked separately. ``rebuild``, ``append``
  and ``remove`` therefore never allocate in steady state.
* **Cached total.** ``total()`` is queried several times per event (time-step
  update, packet sizing, mix-mode branching) with no mutation in between. The
  prefix sum is computed once and invalidated on every mutation, so the cached
  value is always bit-identical to a fresh computation.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from pbe_core.func.jit_mcpbe import (
    nb_fenwick_add,
    nb_fenwick_build,
    nb_fenwick_prefix_sum_1based,
    nb_fenwick_remove_swap_last,
    nb_fenwick_update,
)


@njit(cache=True)
def _prefix_sum_search(tree: np.ndarray, n: int, high_bit: int, s: float) -> int:
    """Largest 0-based index whose prefix sum is still <= ``s``.

    Walks the implicit binary-lifting table of the Fenwick tree.
    """
    i = 0
    bit = high_bit
    while bit:
        nxt = i + bit
        if nxt <= n and tree[nxt] <= s:
            s -= tree[nxt]
            i = nxt
        bit >>= 1
    return i


class FenwickSampler:
    """Sample indices proportionally to non-negative weights.

    API:
        ``total()``                 sum of all weights
        ``size()``                  number of elements
        ``update(idx, weight)``     set one weight, O(log n)
        ``append(weight)``          add an element at the end, O(log n)
        ``remove(idx)``             drop an element (swap-with-last), O(log n)
        ``rebuild(weights)``        replace all weights, O(n)
        ``sample(rng)``             draw an index, O(log n)
        ``asarray()``               copy of the current weights
    """

    __slots__ = ("_n", "_cap", "_tree_buf", "_w_buf", "_total")

    def __init__(self, weights: np.ndarray):
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1:
            raise ValueError("weights must be a 1D array")
        self._n = 0
        self._cap = 0
        self._tree_buf = np.zeros(1, dtype=float)
        self._w_buf = np.zeros(0, dtype=float)
        self._total = None
        self.rebuild(w)

    # ------------------------------------------------------------------
    # Views onto the logically active region
    # ------------------------------------------------------------------
    @property
    def _tree(self) -> np.ndarray:
        return self._tree_buf[: self._n + 1]

    @property
    def _w(self) -> np.ndarray:
        return self._w_buf[: self._n]

    def _reserve(self, n: int) -> None:
        """Ensure the buffers can hold ``n`` weights, preserving existing data."""
        if self._cap >= n:
            return
        new_cap = max(int(n * 1.25) + 8, 16)
        tree = np.zeros(new_cap + 1, dtype=float)
        w = np.zeros(new_cap, dtype=float)
        if self._n:
            tree[: self._n + 1] = self._tree_buf[: self._n + 1]
            w[: self._n] = self._w_buf[: self._n]
        self._tree_buf = tree
        self._w_buf = w
        self._cap = new_cap

    # ------------------------------------------------------------------
    # Basic queries
    # ------------------------------------------------------------------
    def size(self) -> int:
        return self._n

    def total(self) -> float:
        if self._total is None:
            self._total = (
                float(nb_fenwick_prefix_sum_1based(self._tree_buf, self._n))
                if self._n > 0
                else 0.0
            )
        return self._total

    def asarray(self) -> np.ndarray:
        return self._w_buf[: self._n].copy()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def rebuild(self, weights: np.ndarray) -> None:
        """Replace every weight. O(n), allocation-free once warmed up."""
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1:
            raise ValueError("weights must be a 1D array")
        n = int(w.size)
        self._reserve(n)
        self._n = n
        self._tree_buf[: n + 1] = 0.0
        if n:
            self._w_buf[:n] = w
            nb_fenwick_build(self._tree_buf, n, self._w_buf)
        self._total = None

    def update(self, idx: int, new_weight: float) -> None:
        if idx < 0 or idx >= self._n:
            raise IndexError("FenwickSampler.update: idx out of range")
        new_w = float(new_weight)
        if new_w < 0.0:
            new_w = 0.0
        nb_fenwick_update(self._tree_buf, self._n, self._w_buf, idx, new_w)
        self._total = None

    def append(self, weight: float) -> None:
        """Append one element, updating only the affected tree node."""
        w = max(0.0, float(weight))
        n = self._n
        new_n = n + 1
        self._reserve(new_n)

        # Fenwick node at 1-based index ``new_n`` covers (l - 1, new_n].
        lowbit = new_n & -new_n
        left = new_n - lowbit + 1
        sum_right = float(nb_fenwick_prefix_sum_1based(self._tree_buf, n))
        sum_left = float(nb_fenwick_prefix_sum_1based(self._tree_buf, left - 1))

        self._w_buf[n] = w
        self._tree_buf[new_n] = (sum_right - sum_left) + w
        self._n = new_n
        self._total = None

    def remove(self, idx: int) -> None:
        """Remove one element using swap-with-last semantics."""
        n = self._n
        if n == 0:
            raise ValueError("FenwickSampler.remove on empty tree")
        if idx < 0 or idx >= n:
            raise IndexError("FenwickSampler.remove: idx out of range")

        nb_fenwick_remove_swap_last(self._tree_buf, self._w_buf, n, idx)
        self._n = n - 1
        self._total = None

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def prefix_sum_search(self, s: float) -> int:
        """Largest index whose prefix sum is <= ``s`` (0-based)."""
        if self._n == 0:
            raise ValueError("FenwickSampler.prefix_sum_search on empty tree")
        if s < 0.0 or s >= self.total():
            raise ValueError("s must be in [0,total)")
        high_bit = 1 << (self._n.bit_length() - 1)
        return int(_prefix_sum_search(self._tree_buf, self._n, high_bit, s))

    def sample(self, rng) -> int:
        """Draw an index with probability proportional to its weight."""
        if self._n == 0:
            raise ValueError("FenwickSampler.sample: empty or non-positive total")
        t = self.total()
        if t <= 0.0:
            raise ValueError("FenwickSampler.sample: empty or non-positive total")
        return self.prefix_sum_search(float(rng.random()) * t)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._n

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"FenwickSampler(n={self._n}, total={self.total():.6g})"


def rebuild_sampler(sampler: FenwickSampler | None, weights: np.ndarray) -> FenwickSampler:
    """Refill ``sampler`` in place, or build a new one if there is none.

    Preferred over ``FenwickSampler(weights)`` on the hot path: reusing the
    instance keeps the internal buffers and avoids two allocations per event.
    """
    if sampler is None:
        return FenwickSampler(weights)
    sampler.rebuild(weights)
    return sampler
