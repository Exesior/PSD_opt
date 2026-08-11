# Post-processing utilities
from __future__ import annotations

from typing import Tuple, List, Optional, Sequence, Any

import numpy as np
import math
import warnings


# =============================================================================
# Particle Property Container for coupled multi-property analysis
# =============================================================================

class ParticlePropertyContainer:
    """
    Efficient container for time-resolved particle properties.
    
    Stores all particle properties in aligned NumPy arrays for fast
    correlated analysis (e.g., porosity vs. volume, saturation vs. size).
    
    Memory-efficient design:
    - Uses structured NumPy arrays (not Python objects)
    - Pads variable-length particle lists with NaN/0
    - Provides convenient property-based indexing
    
    Usage
    -----
    >>> props = solver.get_all_properties()
    >>> 
    >>> # Access by property name
    >>> poro_at_t5 = props.porosity[5, :]      # All particles at t_idx=5
    >>> vol_over_time = props.V_dry[:, 42]     # One particle over time
    >>> 
    >>> # Get scatter data for plotting
    >>> x, y, w = props.scatter_data(t_idx=5, x="V_dry", y="porosity")
    >>> plt.scatter(x, y, s=w*10, alpha=0.5)
    >>> 
    >>> # Filter by condition
    >>> large_porous = props.filter(t_idx=5, condition="V_dry > 1e-12 and porosity > 0.3")
    >>> 
    >>> # Time series of subset
    >>> sat_series = props.time_series(particle_mask, "saturation")
    """
    
    def __init__(
        self,
        t_vec: np.ndarray,
        V_save: List[np.ndarray],
        W_save: List[np.ndarray],
        Vc_save: Optional[np.ndarray] = None,
        liquid_volume_save: Optional[List[np.ndarray]] = None,
        porosity_save: Optional[List[np.ndarray]] = None,
        saturation_save: Optional[List[np.ndarray]] = None,
        X_save: Optional[List[np.ndarray]] = None,
    ):
        """
        Initialize container from solver save arrays.
        
        Parameters
        ----------
        t_vec : np.ndarray, shape (T,)
            Time points
        V_save : list of np.ndarray
            V_flat snapshots, each shape (dim+1, n_part[t])
        W_save : list of np.ndarray
            Weight snapshots, each shape (n_part[t],)
        Vc_save : np.ndarray, optional
            Control volume over time
        liquid_volume_save : list of np.ndarray, optional
            Internal liquid volume per particle
        porosity_save : list of np.ndarray, optional
            Porosity per particle (NaN for Vollkörper)
        saturation_save : list of np.ndarray, optional
            Saturation per particle
        X_save : list of np.ndarray, optional
            Diameter per particle (computed from V if not provided)
        """
        self._t_vec = np.asarray(t_vec, dtype=float)
        self._T = len(self._t_vec)
        self._Vc_save = np.asarray(Vc_save, dtype=float) if Vc_save is not None else None
        
        # Find maximum particle count across all times (for padding)
        self._N_max = max(arr.shape[1] for arr in V_save) if V_save else 0
        
        # Allocate padded arrays (T, N_max)
        dim = V_save[0].shape[0] - 1 if V_save else 0
        self._V_dry = np.full((self._T, self._N_max), np.nan, dtype=float)
        self._V_solid = np.full((self._T, self._N_max), np.nan, dtype=float)
        self._W = np.zeros((self._T, self._N_max), dtype=float)
        self._X = np.full((self._T, self._N_max), np.nan, dtype=float)
        
        if liquid_volume_save is not None:
            self._liquid_volume = np.full((self._T, self._N_max), np.nan, dtype=float)
        else:
            self._liquid_volume = None
        
        if porosity_save is not None:
            self._porosity = np.full((self._T, self._N_max), np.nan, dtype=float)
        else:
            self._porosity = None
        
        if saturation_save is not None:
            self._saturation = np.full((self._T, self._N_max), np.nan, dtype=float)
        else:
            self._saturation = None
        
        # Fill arrays from snapshots
        for t_idx, (V_snap, W_snap) in enumerate(zip(V_save, W_save)):
            n_part = V_snap.shape[1]
            
            # V_dry (last row of V_flat)
            self._V_dry[t_idx, :n_part] = V_snap[-1, :]
            
            # W
            self._W[t_idx, :n_part] = W_snap
            
            # X (diameter) - compute if not provided
            if X_save is not None:
                self._X[t_idx, :n_part] = X_save[t_idx]
            else:
                self._X[t_idx, :n_part] = self._vol2diam(V_snap[-1, :])
            
            # V_solid = V_dry × (1 - porosity)
            if porosity_save is not None and t_idx < len(porosity_save):
                poro = porosity_save[t_idx]
                has_poro = ~np.isnan(poro)
                v_dry = V_snap[-1, :]
                v_solid = np.zeros_like(v_dry)
                v_solid[has_poro] = v_dry[has_poro] * (1.0 - poro[has_poro])
                v_solid[~has_poro] = v_dry[~has_poro]  # Vollkörper
                self._V_solid[t_idx, :n_part] = v_solid
                self._porosity[t_idx, :n_part] = poro
            else:
                self._V_solid[t_idx, :n_part] = V_snap[-1, :]  # Assume Vollkörper
            
            # Liquid volume
            if liquid_volume_save is not None and t_idx < len(liquid_volume_save):
                self._liquid_volume[t_idx, :n_part] = liquid_volume_save[t_idx]
            
            # Saturation
            if saturation_save is not None and t_idx < len(saturation_save):
                self._saturation[t_idx, :n_part] = saturation_save[t_idx]
        
        # Store active particle count per time
        self._n_active = np.array([arr.shape[1] for arr in V_save], dtype=int)
    
    @staticmethod
    def _vol2diam(V: np.ndarray) -> np.ndarray:
        """Convert volume to diameter assuming spherical particles."""
        return (6.0 * V / np.pi) ** (1.0 / 3.0)
    
    # -------------------------------------------------------------------------
    # Property accessors (read-only views)
    # -------------------------------------------------------------------------
    
    @property
    def t_vec(self) -> np.ndarray:
        """Time axis [s]."""
        return self._t_vec.copy()
    
    @property
    def T(self) -> int:
        """Number of time points."""
        return self._T
    
    @property
    def N_max(self) -> int:
        """Maximum particle count across all times."""
        return self._N_max
    
    @property
    def V_dry(self) -> np.ndarray:
        """Dry particle volume [m³], shape (T, N_max). Padded with NaN."""
        return self._V_dry.copy()
    
    @property
    def V_solid(self) -> np.ndarray:
        """Solid volume [m³], shape (T, N_max). Padded with NaN."""
        return self._V_solid.copy()
    
    @property
    def W(self) -> np.ndarray:
        """Computational weight, shape (T, N_max). Padded with 0."""
        return self._W.copy()
    
    @property
    def X(self) -> np.ndarray:
        """Particle diameter [m], shape (T, N_max). Padded with NaN."""
        return self._X.copy()
    
    @property
    def liquid_volume(self) -> Optional[np.ndarray]:
        """Internal liquid volume [m³], shape (T, N_max). None if not available."""
        return self._liquid_volume.copy() if self._liquid_volume is not None else None
    
    @property
    def porosity(self) -> Optional[np.ndarray]:
        """Porosity [0-1], shape (T, N_max). None if not available."""
        return self._porosity.copy() if self._porosity is not None else None
    
    @property
    def saturation(self) -> Optional[np.ndarray]:
        """Saturation [0-1], shape (T, N_max). None if not available."""
        return self._saturation.copy() if self._saturation is not None else None
    
    @property
    def Vc(self) -> Optional[np.ndarray]:
        """Control volume [m³] over time. None if not available."""
        return self._Vc_save.copy() if self._Vc_save is not None else None
    
    def n_active(self, t_idx: int) -> int:
        """Number of active (non-padded) particles at time t_idx."""
        return int(self._n_active[t_idx])
    
    # -------------------------------------------------------------------------
    # Data extraction methods
    # -------------------------------------------------------------------------
    
    def scatter_data(
        self,
        t_idx: int,
        x: str,
        y: str,
        w: Optional[str] = "W",
        active_only: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Extract paired property data for scatter plotting.
        
        Parameters
        ----------
        t_idx : int
            Time index
        x : str
            X-axis property: "V_dry", "V_solid", "X", "porosity", "saturation", "liquid_volume"
        y : str
            Y-axis property (same options as x)
        w : str, optional
            Weight/scatter size property (default: "W"). Set None for no weights.
        active_only : bool, default=True
            If True, return only active particles (exclude padded NaN/zeros)
        
        Returns
        -------
        x_vals : np.ndarray
            X-axis values
        y_vals : np.ndarray
            Y-axis values
        w_vals : np.ndarray or None
            Weight values (if w is specified)
        
        Example
        -------
        >>> x, y, w = props.scatter_data(t_idx=5, x="X", y="porosity")
        >>> plt.scatter(x*1e6, y, s=w*10, alpha=0.5)
        >>> plt.xlabel("Diameter [µm]")
        >>> plt.ylabel("Porosity")
        """
        prop_map = {
            "V_dry": self._V_dry,
            "V_solid": self._V_solid,
            "X": self._X,
            "W": self._W,
            "liquid_volume": self._liquid_volume,
            "porosity": self._porosity,
            "saturation": self._saturation,
        }
        
        if x not in prop_map:
            raise ValueError(f"Unknown x='{x}'. Available: {list(prop_map.keys())}")
        if y not in prop_map:
            raise ValueError(f"Unknown y='{y}'. Available: {list(prop_map.keys())}")
        if w is not None and w not in prop_map:
            raise ValueError(f"Unknown w='{w}'. Available: {list(prop_map.keys())}")
        
        arr_x = prop_map[x]
        arr_y = prop_map[y]
        arr_w = prop_map[w] if w is not None else None
        
        if arr_x is None or arr_y is None:
            missing = "x" if arr_x is None else "y"
            raise RuntimeError(f"{missing} property '{x if arr_x is None else y}' not available.")
        
        x_vals = arr_x[t_idx, :].copy()
        y_vals = arr_y[t_idx, :].copy()
        w_vals = arr_w[t_idx, :].copy() if arr_w is not None else None
        
        if active_only:
            n_act = self.n_active(t_idx)
            x_vals = x_vals[:n_act]
            y_vals = y_vals[:n_act]
            if w_vals is not None:
                w_vals = w_vals[:n_act]
        
        return x_vals, y_vals, w_vals
    
    def filter(
        self,
        t_idx: int,
        condition: str,
    ) -> np.ndarray:
        """
        Get boolean mask of particles matching a condition at time t_idx.
        
        Parameters
        ----------
        t_idx : int
            Time index
        condition : str
            Condition string using property names and operators.
            Example: "V_dry > 1e-12 and porosity > 0.3"
        
        Returns
        -------
        mask : np.ndarray, dtype=bool
            Boolean mask for active particles at t_idx
        
        Example
        -------
        >>> mask = props.filter(t_idx=5, condition="X > 100e-6 and saturation < 0.5")
        >>> print(f"Found {np.sum(mask)} large, dry particles")
        """
        n_act = self.n_active(t_idx)
        
        # Build namespace for eval
        ns = {
            "V_dry": self._V_dry[t_idx, :n_act],
            "V_solid": self._V_solid[t_idx, :n_act],
            "X": self._X[t_idx, :n_act],
            "W": self._W[t_idx, :n_act],
        }
        if self._liquid_volume is not None:
            ns["liquid_volume"] = self._liquid_volume[t_idx, :n_act]
        if self._porosity is not None:
            ns["porosity"] = self._porosity[t_idx, :n_act]
        if self._saturation is not None:
            ns["saturation"] = self._saturation[t_idx, :n_act]
        
        mask = eval(condition, {"__builtins__": {}}, ns)
        return np.asarray(mask, dtype=bool)
    
    def time_series(
        self,
        particle_mask: np.ndarray,
        property_name: str,
        aggregation: str = "mean",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract time series of a property for a subset of particles.
        
        Parameters
        ----------
        particle_mask : np.ndarray, shape (N_max,) or (N_initial,)
            Boolean mask selecting particles of interest
        property_name : str
            Property to track: "V_dry", "porosity", "saturation", etc.
        aggregation : {"mean", "sum", "weighted_mean"}
            How to aggregate across selected particles
        
        Returns
        -------
        values : np.ndarray, shape (T,)
            Aggregated property value over time
        t_vec : np.ndarray, shape (T,)
            Time points
        
        Example
        -------
        >>> # Track average porosity of initially large particles
        >>> init_mask = props.X[0, :] > 500e-6
        >>> poro_series, t = props.time_series(init_mask, "porosity", "mean")
        >>> plt.plot(t, poro_series)
        """
        prop_map = {
            "V_dry": self._V_dry,
            "V_solid": self._V_solid,
            "X": self._X,
            "W": self._W,
            "liquid_volume": self._liquid_volume,
            "porosity": self._porosity,
            "saturation": self._saturation,
        }
        
        if property_name not in prop_map:
            raise ValueError(f"Unknown property='{property_name}'. Available: {list(prop_map.keys())}")
        
        arr = prop_map[property_name]
        if arr is None:
            raise RuntimeError(f"Property '{property_name}' not available.")
        
        # Ensure mask matches array dimensions
        if len(particle_mask) != self._N_max:
            if len(particle_mask) == self._n_active[0]:
                # Extend mask with False for padded positions
                extended = np.zeros(self._N_max, dtype=bool)
                extended[:len(particle_mask)] = particle_mask
                particle_mask = extended
            else:
                raise ValueError(f"Mask length {len(particle_mask)} doesn't match N_max={self._N_max}")
        
        values = np.zeros(self._T, dtype=float)
        
        for t_idx in range(self._T):
            n_act = self._n_active[t_idx]
            active_mask = np.zeros(self._N_max, dtype=bool)
            active_mask[:n_act] = True
            
            combined = particle_mask & active_mask
            if not np.any(combined):
                values[t_idx] = np.nan
                continue
            
            subset = arr[t_idx, combined]
            valid = np.isfinite(subset)
            
            if not np.any(valid):
                values[t_idx] = np.nan
                continue
            
            if aggregation == "mean":
                values[t_idx] = np.mean(subset[valid])
            elif aggregation == "sum":
                values[t_idx] = np.nansum(subset)
            elif aggregation == "weighted_mean":
                weights = self._W[t_idx, combined][valid]
                vals = subset[valid]
                if np.sum(weights) > 0:
                    values[t_idx] = np.sum(vals * weights) / np.sum(weights)
                else:
                    values[t_idx] = np.nan
            else:
                raise ValueError(f"Unknown aggregation='{aggregation}'.")
        
        return values, self._t_vec.copy()
    
    def plot_correlation(
        self,
        t_idx: int,
        x: str,
        y: str,
        color_by: Optional[str] = None,
        ax=None,
        **scatter_kwargs,
    ):
        """
        Quick visualization of property correlations.
        
        Parameters
        ----------
        t_idx : int
            Time index
        x, y : str
            Property names for axes
        color_by : str, optional
            Color points by this property (creates colormap scatter)
        ax : matplotlib.axes.Axes, optional
            Axes to plot on (creates new if None)
        scatter_kwargs : dict
            Additional kwargs passed to plt.scatter()
        
        Returns
        -------
        fig, ax : matplotlib figure and axes
        
        Example
        -------
        >>> props.plot_correlation(t_idx=5, x="X", y="porosity", color_by="saturation")
        >>> plt.show()
        """
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        
        x_vals, y_vals, w_vals = self.scatter_data(t_idx, x, y, w="W")
        
        # Create figure/axes if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure
        
        # Default styling
        defaults = {
            "s": w_vals * 50 if w_vals is not None else 20,
            "alpha": 0.6,
            "edgecolors": "none",
        }
        defaults.update(scatter_kwargs)
        
        if color_by is not None:
            _, c_vals, _ = self.scatter_data(t_idx, x=color_by, y=y, w=None)
            sc = ax.scatter(x_vals, y_vals, c=c_vals, cmap="viridis", **defaults)
            plt.colorbar(sc, ax=ax, label=color_by)
        else:
            ax.scatter(x_vals, y_vals, **defaults)
        
        # Labels
        x_unit = " [m]" if x in ["V_dry", "V_solid", "liquid_volume"] else " [m]" if x == "X" else ""
        y_unit = " [m]" if y in ["V_dry", "V_solid", "liquid_volume"] else " [m]" if y == "X" else ""
        ax.set_xlabel(f"{x}{x_unit}")
        ax.set_ylabel(f"{y}{y_unit}")
        ax.set_title(f"t = {self._t_vec[t_idx]:.3f} s")
        ax.grid(True, alpha=0.3)
        
        return fig, ax


class MCPBEPost:
    """Post-processing mixin: compute time-resolved moments µ(i,j,t) and PSD.
    
    Extended to support additional particle properties:
    - liquid_volume: Internal liquid volume per particle [m³]
    - porosity: Void fraction [0-1] or NaN for Vollkörper
    - saturation: Liquid saturation S = V_liq_intern / V_pore [0-1]
    """

    def calc_moments_over_time(
        self, max_i: int = 2, max_j: int = 2, normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mu, t_vec) where mu[i,j,t] are mixed moments over components.
    
        Weighted version:
            mu = sum_k W_k * V1_k^i * V3_k^j / Vc(t)   (dim>1)
            mu = sum_k W_k * V_k^i / Vc(t)            (dim==1)
        If no W_save exists, fall back to W=1 (backward compatible).
        """
        T = min(len(self.V_save), len(self.Vc_save), len(self.t_vec))
        mu = np.zeros((max_i + 1, max_j + 1, T), dtype=float)
    
        has_W = hasattr(self, "W_save") and self.W_save is not None and len(self.W_save) >= T
    
        for t in range(T):
            Vc = float(self.Vc_save[t]) if (normalize and self.Vc_save) else 1.0
            V_snap = np.asarray(self.V_save[t], dtype=float)
    
            if has_W:
                W = np.asarray(self.W_save[t], dtype=float)
            else:
                W = np.ones(V_snap.shape[1], dtype=float)
    
            if self.dim == 1:
                V = np.asarray(V_snap[0, :], dtype=float)
                for i in range(max_i + 1):
                    mu[i, 0, t] = float(np.sum(W * np.power(V, i))) / Vc
            else:
                V1 = np.asarray(V_snap[0, :], dtype=float)
                V3 = np.asarray(V_snap[1, :], dtype=float)
                for i in range(max_i + 1):
                    Vi = np.power(V1, i)
                    for j in range(max_j + 1):
                        mu[i, j, t] = float(np.sum(W * Vi * np.power(V3, j))) / Vc
    
        return mu, self.t_vec[:T]


    # ------------------------------------------------------------------
    # PSD helpers (single realization)
    # ------------------------------------------------------------------
    def _compute_psd_cdf_from_snapshot(
        self,
        V_snap: np.ndarray,
        psd_basis: str = "volume",
        W_snap: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Build an empirical CDF from a particle snapshot (weighted).
    
        Parameters
        ----------
        V_snap : ndarray, shape (dim+1, a)
            Snapshot of particle volumes at a given time (last row is total volume).
        psd_basis : {"number", "volume"}
            Weighting basis for the CDF.
        W_snap : ndarray, shape (a,), optional
            Statistical/number weight for each compute particle (how many real particles it represents).
            If None, assumes W=1 (backward compatible).
    
        Returns
        -------
        (x_sorted, Q_sorted) or None
        """
        if V_snap.size == 0 or V_snap.shape[1] == 0:
            return None
    
        V_tot = np.asarray(V_snap[-1, :], dtype=float)  # total volume per particle
    
        if W_snap is None:
            W = np.ones_like(V_tot, dtype=float)
        else:
            W = np.asarray(W_snap, dtype=float)
            if W.shape[0] != V_tot.shape[0]:
                raise ValueError("W_snap length must match number of particles in V_snap")
    
        # CDF weight
        if psd_basis == "number":
            w = W
        elif psd_basis == "volume":
            w = W * V_tot
        else:
            raise ValueError(f"psd_basis must be 'number' or 'volume', got {psd_basis!r}")
    
        # convert to diameter
        x = self._vol2diam(V_tot)
    
        mask = (w > 0.0) & (x > 0.0) & np.isfinite(w) & np.isfinite(x)
        if not np.any(mask):
            return None
    
        x = x[mask]
        w = w[mask]
    
        idx = np.argsort(x)
        x_sorted = x[idx]
        w_sorted = w[idx]
        w_cum = np.cumsum(w_sorted)
        total = float(w_cum[-1])
        if total <= 0.0:
            return None
    
        Q_sorted = w_cum / total
        return x_sorted, Q_sorted


    @staticmethod
    def _eval_Q_of_x(
        x_sorted: np.ndarray,
        Q_sorted: np.ndarray,
        x_query: np.ndarray,
    ) -> np.ndarray:
        """Evaluate Q(x) on a given diameter grid using a step-wise empirical CDF."""
        xq = np.asarray(x_query, dtype=float)
        Qq = np.zeros_like(xq, dtype=float)
        idx = np.searchsorted(x_sorted, xq, side="right") - 1
        Qq[idx < 0] = 0.0
        valid = idx >= 0
        if np.any(valid):
            idx_clipped = np.clip(idx[valid], 0, len(Q_sorted) - 1)
            Qq[valid] = Q_sorted[idx_clipped]
        return Qq

    @staticmethod
    def _eval_x_of_Q(
        x_sorted: np.ndarray,
        Q_sorted: np.ndarray,
        Q_query: np.ndarray,
    ) -> np.ndarray:
        """Evaluate x(Q) (quantile function) on a given Q grid using the empirical CDF."""
        Qq = np.asarray(Q_query, dtype=float)
        xq = np.zeros_like(Qq, dtype=float)
        idx = np.searchsorted(Q_sorted, Qq, side="left")
        idx = np.clip(idx, 0, len(Q_sorted) - 1)
        xq = x_sorted[idx]
        return xq

    def compute_psd_cdf_over_time(
        self,
        psd_basis: str = "volume",
        time_scheme: str = "right",
    ) -> Tuple[List[Optional[Tuple[np.ndarray, np.ndarray]]], np.ndarray]:
        """Compute empirical PSD CDF at all saved times for a single realization (weighted)."""
        time_scheme = str(time_scheme).lower()
        if time_scheme not in ("right", "left", "interp", "nearest"):
            raise ValueError(
                f"time_scheme must be one of 'right', 'left', 'interp', 'nearest', "
                f"got {time_scheme!r}."
            )
    
        T_base = min(len(self.V_save), len(self.t_vec))
        if T_base == 0:
            return [], np.asarray([], dtype=float)
    
        use_left_side = time_scheme in ("left", "interp", "nearest")
        if use_left_side:
            if not hasattr(self, "V_save_left") or not hasattr(self, "t_left") or not hasattr(self, "t_right"):
                raise RuntimeError(
                    "time_scheme uses left/right information but V_save_left / t_left / t_right "
                    "are not available."
                )
            T = min(T_base, len(self.V_save_left), len(self.t_left), len(self.t_right))
        else:
            T = T_base
    
        if T == 0:
            return [], np.asarray([], dtype=float)
    
        t_vec = np.asarray(self.t_vec[:T], dtype=float)
    
        has_W_right = hasattr(self, "W_save") and self.W_save is not None and len(self.W_save) >= T
        has_W_left = hasattr(self, "W_save_left") and self.W_save_left is not None and len(self.W_save_left) >= T
    
        def _cdf_from_side(idx: int, side: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
            if side == "right":
                V_snap = self.V_save[idx]
                W_snap = self.W_save[idx] if has_W_right else None
            elif side == "left":
                V_snap = self.V_save_left[idx]
                W_snap = self.W_save_left[idx] if has_W_left else None
            else:
                raise ValueError(f"Unknown side {side!r} in _cdf_from_side.")
            return self._compute_psd_cdf_from_snapshot(V_snap, psd_basis=psd_basis, W_snap=W_snap)
    
        cdf_list: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
    
        for t in range(T):
            if time_scheme == "right":
                cdf_list.append(_cdf_from_side(t, "right"))
                continue
    
            if time_scheme == "left":
                cdf_list.append(_cdf_from_side(t, "left"))
                continue
    
            if time_scheme == "nearest":
                tl = float(self.t_left[t])
                tr = float(self.t_right[t])
                target = float(t_vec[t])
                side = "left" if abs(target - tl) <= abs(tr - target) else "right"
                cdf_list.append(_cdf_from_side(t, side))
                continue
    
            # interp
            # compute left/right CDF and interpolate in time
            cL = _cdf_from_side(t, "left")
            cR = _cdf_from_side(t, "right")
            if cL is None and cR is None:
                cdf_list.append(None)
                continue
            if cL is None:
                cdf_list.append(cR)
                continue
            if cR is None:
                cdf_list.append(cL)
                continue
    
            xL, QL = cL
            xR, QR = cR
    
            x_union = np.unique(np.concatenate([xL, xR]))
            QL_u = self._eval_Q_of_x(xL, QL, x_union)
            QR_u = self._eval_Q_of_x(xR, QR, x_union)
    
            tl = float(self.t_left[t])
            tr = float(self.t_right[t])
            target = float(t_vec[t])
            if tr <= tl + 1e-15:
                alpha = 1.0
            else:
                alpha = float((target - tl) / (tr - tl))
                alpha = float(np.clip(alpha, 0.0, 1.0))
    
            Q_interp = (1.0 - alpha) * QL_u + alpha * QR_u
            cdf_list.append((x_union, Q_interp))
    
        return cdf_list, t_vec

    # ------------------------------------------------------------------
    # Additional particle property accessors (NEW)
    # ------------------------------------------------------------------
    def get_liquid_volume_over_time(
        self,
        aggregation: str = "mean",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract liquid volume statistics over time.
        
        Parameters
        ----------
        aggregation : {"mean", "sum", "total_weighted"}
            How to aggregate per-particle data:
            - "mean": Average liquid volume per particle [m³]
            - "sum": Sum of liquid_volume × W (total represented volume)
            - "total_weighted": Sum(liquid_volume × W) / Vc (intensive)
        
        Returns
        -------
        stat : np.ndarray, shape (T,)
            Aggregated statistic over time
        t_vec : np.ndarray, shape (T,)
            Time points
        """
        if not hasattr(self, 'liquid_volume_save') or self.liquid_volume_save is None:
            raise RuntimeError(
                "liquid_volume_save not available. "
                "Ensure solver was initialized with proper save arrays."
            )
        
        T = min(len(self.liquid_volume_save), len(self.t_vec))
        t_vec = self.t_vec[:T]
        stat = np.zeros(T, dtype=float)
        
        has_W = hasattr(self, "W_save") and self.W_save is not None and len(self.W_save) >= T
        
        for t in range(T):
            lv_snap = np.asarray(self.liquid_volume_save[t], dtype=float)
            n_part = lv_snap.shape[0]
            
            if has_W:
                W = np.asarray(self.W_save[t], dtype=float)
            else:
                W = np.ones(n_part, dtype=float)
            
            if aggregation == "mean":
                stat[t] = np.mean(lv_snap)
            elif aggregation == "sum":
                stat[t] = np.sum(lv_snap * W)
            elif aggregation == "total_weighted":
                Vc = float(self.Vc_save[t]) if hasattr(self, 'Vc_save') else 1.0
                stat[t] = np.sum(lv_snap * W) / Vc
            else:
                raise ValueError(f"Unknown aggregation='{aggregation}'. Use 'mean', 'sum', or 'total_weighted'.")
        
        return stat, t_vec

    def get_porosity_over_time(
        self,
        aggregation: str = "mean",
        exclude_nan: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract porosity statistics over time.
        
        Parameters
        ----------
        aggregation : {"mean", "min", "max", "std"}
            How to aggregate per-particle data:
            - "mean": Average porosity (optionally excluding NaN)
            - "min": Minimum porosity
            - "max": Maximum porosity
            - "std": Standard deviation
        exclude_nan : bool, default=True
            If True, exclude NaN values (Vollkörper) from statistics
        
        Returns
        -------
        stat : np.ndarray, shape (T,)
            Aggregated statistic over time
        t_vec : np.ndarray, shape (T,)
            Time points
        """
        if not hasattr(self, 'porosity_save') or self.porosity_save is None:
            raise RuntimeError(
                "porosity_save not available. "
                "Ensure solver was initialized with proper save arrays."
            )
        
        T = min(len(self.porosity_save), len(self.t_vec))
        t_vec = self.t_vec[:T]
        stat = np.zeros(T, dtype=float)
        
        for t in range(T):
            poro_snap = np.asarray(self.porosity_save[t], dtype=float)
            
            if exclude_nan:
                valid = ~np.isnan(poro_snap)
                if not np.any(valid):
                    stat[t] = np.nan
                    continue
                poro_snap = poro_snap[valid]
            
            if aggregation == "mean":
                stat[t] = np.mean(poro_snap)
            elif aggregation == "min":
                stat[t] = np.min(poro_snap)
            elif aggregation == "max":
                stat[t] = np.max(poro_snap)
            elif aggregation == "std":
                stat[t] = np.std(poro_snap)
            else:
                raise ValueError(f"Unknown aggregation='{aggregation}'. Use 'mean', 'min', 'max', or 'std'.")
        
        return stat, t_vec

    def get_saturation_over_time(
        self,
        aggregation: str = "mean",
        exclude_nan: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract saturation statistics over time.
        
        Parameters
        ----------
        aggregation : {"mean", "min", "max", "std", "sum_weighted"}
            How to aggregate per-particle data
        exclude_nan : bool, default=True
            If True, exclude NaN values from statistics
        
        Returns
        -------
        stat : np.ndarray, shape (T,)
            Aggregated statistic over time
        t_vec : np.ndarray, shape (T,)
            Time points
        """
        if not hasattr(self, 'saturation_save') or self.saturation_save is None:
            raise RuntimeError(
                "saturation_save not available. "
                "Ensure solver was initialized with proper save arrays."
            )
        
        T = min(len(self.saturation_save), len(self.t_vec))
        t_vec = self.t_vec[:T]
        stat = np.zeros(T, dtype=float)
        
        has_W = hasattr(self, "W_save") and self.W_save is not None and len(self.W_save) >= T
        
        for t in range(T):
            sat_snap = np.asarray(self.saturation_save[t], dtype=float)
            
            if exclude_nan:
                valid = ~np.isnan(sat_snap)
                if not np.any(valid):
                    stat[t] = np.nan
                    continue
                sat_snap = sat_snap[valid]
                if has_W:
                    W = np.asarray(self.W_save[t], dtype=float)[valid]
            
            if aggregation == "mean":
                stat[t] = np.mean(sat_snap)
            elif aggregation == "min":
                stat[t] = np.min(sat_snap)
            elif aggregation == "max":
                stat[t] = np.max(sat_snap)
            elif aggregation == "std":
                stat[t] = np.std(sat_snap)
            elif aggregation == "sum_weighted" and has_W:
                stat[t] = np.sum(sat_snap * W)
            else:
                agg_name = "sum_weighted" if aggregation == "sum_weighted" else aggregation
                raise ValueError(f"Unknown aggregation='{agg_name}'. Use 'mean', 'min', 'max', 'std', or 'sum_weighted'.")
        
        return stat, t_vec

    def get_full_snapshot(
        self,
        t_idx: int,
        include_derived: bool = True,
    ) -> dict:
        """Get complete particle state snapshot at a specific time.
        
        Parameters
        ----------
        t_idx : int
            Time index (0 to T-1)
        include_derived : bool, default=True
            If True, compute derived quantities (V_solid, V_pore, etc.)
        
        Returns
        -------
        snapshot : dict
            Contains all particle properties at time t_idx:
            - "t": Time [s]
            - "V_flat": Particle volumes (dim+1, n_part)
            - "W": Computational weights
            - "X": Diameters [m]
            - "liquid_volume": Internal liquid [m³]
            - "porosity": Void fraction [0-1] or NaN
            - "saturation": Liquid saturation [0-1]
            - "Vc": Control volume [m³]
            - Optional derived: "V_solid", "V_pore", "V_liq_external"
        """
        if not hasattr(self, 'V_save') or self.V_save is None:
            raise RuntimeError("V_save not available.")
        
        T = min(len(self.V_save), len(self.t_vec))
        if t_idx < 0 or t_idx >= T:
            raise IndexError(f"t_idx={t_idx} out of range [0, {T-1}]")
        
        snapshot = {
            "t": float(self.t_vec[t_idx]),
            "V_flat": self.V_save[t_idx].copy(),
            "Vc": float(self.Vc_save[t_idx]) if hasattr(self, 'Vc_save') else None,
        }
        
        # Optional arrays
        for attr in ["W_save", "X_save", "liquid_volume_save", "porosity_save", "saturation_save"]:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                data = getattr(self, attr)
                if t_idx < len(data):
                    key = attr.replace("_save", "")
                    snapshot[key] = data[t_idx].copy()
        
        # Derived quantities (if requested and data available)
        if include_derived and "V_flat" in snapshot and "porosity" in snapshot:
            V_flat = snapshot["V_flat"]
            poro = snapshot["porosity"]
            
            # V_solid = V_dry × (1 - porosity)
            V_dry = V_flat[-1, :]
            has_poro = ~np.isnan(poro)
            V_solid = np.zeros_like(V_dry)
            V_solid[has_poro] = V_dry[has_poro] * (1.0 - poro[has_poro])
            V_solid[~has_poro] = V_dry[~has_poro]  # Vollkörper
            snapshot["V_solid"] = V_solid
            
            # V_pore = V_dry × porosity
            V_pore = np.zeros_like(V_dry)
            V_pore[has_poro] = V_dry[has_poro] * poro[has_poro]
            snapshot["V_pore"] = V_pore
            
            # V_liq_external (if liquid_volume available)
            if "liquid_volume" in snapshot:
                liq_vol = snapshot["liquid_volume"]
                snapshot["V_liq_external"] = np.maximum(liq_vol - V_pore, 0.0)
        
        return snapshot

    # ------------------------------------------------------------------
    # Coupled multi-property analysis (NEW)
    # ------------------------------------------------------------------
    def get_all_properties(self) -> ParticlePropertyContainer:
        """
        Extract all particle properties into a coupled container for correlated analysis.
        
        Returns a ParticlePropertyContainer that provides efficient access to:
        - All particle properties aligned by (time, particle_index)
        - Scatter data extraction for plotting correlations
        - Filtering and time-series analysis of particle subsets
        
        Returns
        -------
        props : ParticlePropertyContainer
            Container with all particle properties
        
        Example
        -------
        >>> props = solver.get_all_properties()
        >>> 
        >>> # Scatter plot: porosity vs. volume
        >>> x, y, w = props.scatter_data(t_idx=5, x="V_dry", y="porosity")
        >>> plt.scatter(x*1e6, y, s=w*10, alpha=0.5)
        >>> 
        >>> # Filter: large, wet particles
        >>> mask = props.filter(t_idx=5, condition="X > 200e-6 and saturation > 0.5")
        >>> print(f"Found {np.sum(mask)} particles")
        >>> 
        >>> # Time series: track porosity of initially small particles
        >>> small_init = props.X[0, :] < 100e-6
        >>> poro_series, t = props.time_series(small_init, "porosity")
        >>> plt.plot(t, poro_series)
        
        See Also
        --------
        ParticlePropertyContainer : Full API documentation
        scatter_data : Extract paired properties for plotting
        filter : Select particles by condition
        time_series : Track property evolution of subsets
        """
        # Check required arrays
        if not hasattr(self, 'V_save') or self.V_save is None:
            raise RuntimeError("V_save not available.")
        if not hasattr(self, 'W_save') or self.W_save is None:
            raise RuntimeError("W_save not available.")
        
        return ParticlePropertyContainer(
            t_vec=self.t_vec,
            V_save=self.V_save,
            W_save=self.W_save,
            Vc_save=getattr(self, 'Vc_save', None),
            liquid_volume_save=getattr(self, 'liquid_volume_save', None),
            porosity_save=getattr(self, 'porosity_save', None),
            saturation_save=getattr(self, 'saturation_save', None),
            X_save=getattr(self, 'X_save', None),  # Optional, computed if missing
        )

    def _invert_cdf_monotone(self, x_axis: np.ndarray, Q_vals: np.ndarray, q: float = 0.5) -> float:
        """Invert a (nearly) monotone CDF to find x at probability q.
    
        Enforces monotonicity (cummax) to reduce numerical noise, then uses
        linear interpolation in (Q, x). Returns NaN if q is outside range.
        """
        x_axis = np.asarray(x_axis, dtype=float)
        Q_vals = np.asarray(Q_vals, dtype=float)
        if x_axis.size == 0 or Q_vals.size == 0 or x_axis.size != Q_vals.size:
            return float("nan")
    
        mask = np.isfinite(x_axis) & np.isfinite(Q_vals)
        if not np.any(mask):
            return float("nan")
    
        x = x_axis[mask]
        Q = Q_vals[mask]
    
        order = np.argsort(x)
        x = x[order]
        Q = Q[order]
    
        # enforce monotone non-decreasing
        Q = np.maximum.accumulate(Q)
    
        if Q[0] > q or Q[-1] < q:
            return float("nan")
    
        k = int(np.searchsorted(Q, q, side="left"))
        if k <= 0:
            return float(x[0])
        if k >= Q.size:
            return float(x[-1])
    
        q0, q1 = float(Q[k - 1]), float(Q[k])
        x0, x1 = float(x[k - 1]), float(x[k])
        if q1 <= q0 + 1e-15:
            return float(x1)
        return float(x0 + (q - q0) * (x1 - x0) / (q1 - q0))

    def _filter_and_renormalize_cdf_by_xmin(
        self,
        x_sorted: np.ndarray,
        Q_sorted: np.ndarray,
        x_min: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Filter an empirical CDF (x_sorted, Q_sorted) by removing x < x_min,
        then re-normalize so that:
          - Q(x_min) = 0
          - Q(max)   = 1

        Returns None if nothing remains.
        """
        x_sorted = np.asarray(x_sorted, dtype=float).ravel()
        Q_sorted = np.asarray(Q_sorted, dtype=float).ravel()
        x_min = float(x_min)

        if x_sorted.size == 0 or Q_sorted.size == 0 or x_sorted.size != Q_sorted.size:
            return None

        # enforce monotone & bounds (numerical safety)
        Q_sorted = np.clip(Q_sorted, 0.0, 1.0)
        Q_sorted = np.maximum.accumulate(Q_sorted)

        # if x_min is below support -> nothing to do
        if x_min <= float(x_sorted[0]):
            return x_sorted, Q_sorted

        # if x_min is above support -> everything removed
        if x_min >= float(x_sorted[-1]):
            return None

        # Q at cutoff (stepwise: right-continuous convention consistent with _eval_Q_of_x)
        # For x<x_sorted[0], Q=0. Here x_min within support.
        j = int(np.searchsorted(x_sorted, x_min, side="right") - 1)
        j = max(j, 0)
        Q_cut = float(Q_sorted[j])

        denom = 1.0 - Q_cut
        if denom <= 0.0:
            return None

        # keep points with x >= x_min
        k0 = int(np.searchsorted(x_sorted, x_min, side="left"))
        x_tail = x_sorted[k0:]
        Q_tail = Q_sorted[k0:]

        # ensure x_min included as first point (use Q_cut at x_min)
        x_new = np.concatenate(([x_min], x_tail))
        Q_new_raw = np.concatenate(([Q_cut], Q_tail))

        # shift & renormalize: Q' = (Q - Q_cut)/(1 - Q_cut)
        Q_new = (Q_new_raw - Q_cut) / denom
        Q_new = np.clip(Q_new, 0.0, 1.0)
        Q_new = np.maximum.accumulate(Q_new)
        Q_new[0] = 0.0
        Q_new[-1] = 1.0

        return x_new, Q_new

    # ------------------------------------------------------------------
    # PSD aggregation over repeats
    # ------------------------------------------------------------------
    def aggregate_psd_repeats(
        self,
        cdf_repeats: Sequence[Sequence[Optional[Tuple[np.ndarray, np.ndarray]]]],
        t_vec: np.ndarray,
        psd_basis: str = "volume",
        psd_x_grid: Optional[np.ndarray] = None,
        psd_Q_grid: Optional[np.ndarray] = None,
    ) -> dict:
        """Aggregate PSD CDFs over multiple repeats into an averaged PSD on t_vec.

        Parameters
        ----------
        cdf_repeats : sequence of length N
            Each element is a list (length T) of (x_sorted, Q_sorted) or None.
        t_vec : ndarray, shape (T,)
            Time grid aligned with each cdf list.
        psd_basis : {"number", "volume"}
            Basis used when CDFs were computed.
        psd_x_grid : ndarray, optional
            If provided, the PSD is returned as Q(x) evaluated on this grid.
        psd_Q_grid : ndarray, optional
            If provided (and psd_x_grid is None), the PSD is returned as x(Q)
            evaluated on this Q grid.

        Returns
        -------
        psd_info : dict
            See docstring of MCPBEBase.solve_repeats for the exact structure.
        """
        # parse mode and grid choice
        psd_mode: Optional[str] = None   # "Q_of_x" or "x_of_Q"
        auto_x_grid = False

        x_grid_user: Optional[np.ndarray]
        Q_grid_user: Optional[np.ndarray]

        if psd_x_grid is not None and psd_Q_grid is not None:
            warnings.warn(
                "Both psd_x_grid and psd_Q_grid are provided; psd_x_grid will be used and Q(x) will be computed.",
                RuntimeWarning,
            )
            psd_mode = "Q_of_x"
            x_grid_user = np.asarray(psd_x_grid, dtype=float)
            Q_grid_user = None
        elif psd_x_grid is not None:
            psd_mode = "Q_of_x"
            x_grid_user = np.asarray(psd_x_grid, dtype=float)
            Q_grid_user = None
        elif psd_Q_grid is not None:
            psd_mode = "x_of_Q"
            Q_grid_user = np.asarray(psd_Q_grid, dtype=float)
            x_grid_user = None
        else:
            # auto-generate x_grid and output Q(x)
            psd_mode = "Q_of_x"
            auto_x_grid = True
            x_grid_user = None
            Q_grid_user = None
            warnings.warn(
                "psd_enable=True but neither psd_x_grid nor psd_Q_grid is provided; "
                "a log-spaced x_grid will be generated automatically and Q(x) will be returned.",
                RuntimeWarning,
            )

        T = int(len(t_vec))
        x_50 = np.full(T, np.nan, dtype=float)
        psd_info: dict[str, Any] = {
            "basis": psd_basis,
            "mode": psd_mode,
            "t_vec": np.asarray(t_vec, dtype=float),
            "note": (
                "PSD computed at all saved times (aligned with t_vec) "
                "and averaged over all repeats."
            ),
        }

        if T == 0 or not cdf_repeats:
            # no data
            if psd_mode == "Q_of_x":
                psd_info["x_grid"] = None
                psd_info["Q_mean"] = None
            else:
                psd_info["Q_grid"] = None
                psd_info["x_mean"] = None
            return psd_info

        # ensure all repeats have the same number of time steps
        for cdf_list in cdf_repeats:
            if len(cdf_list) != T:
                raise RuntimeError(
                    f"All CDF lists must have length T={T}, got {len(cdf_list)}."
                )

        # aggregation branches
        if psd_mode == "Q_of_x":
            # ------------------------------------------------------------------
            # Q(x) mode
            # ------------------------------------------------------------------
            if auto_x_grid:
                # determine global min/max diameter per time, then build a common x_grid
                global_min_x = np.full(T, np.inf, dtype=float)
                global_max_x = np.zeros(T, dtype=float)

                for cdf_list in cdf_repeats:
                    for it, cdf in enumerate(cdf_list):
                        if cdf is None:
                            continue
                        x_sorted, _ = cdf
                        xmin = float(x_sorted[0])
                        xmax = float(x_sorted[-1])
                        if xmin < global_min_x[it]:
                            global_min_x[it] = xmin
                        if xmax > global_max_x[it]:
                            global_max_x[it] = xmax

                # global range over all times (conservative choice)
                finite_min = global_min_x[np.isfinite(global_min_x)]
                if finite_min.size == 0:
                    warnings.warn(
                        "auto_x_grid mode: no valid CDFs collected; psd_info will be empty.",
                        RuntimeWarning,
                    )
                    psd_info["x_grid"] = None
                    psd_info["Q_mean"] = None
                    psd_info["x_50"] = np.full(T, np.nan, dtype=float)
                    return psd_info

                xmin_global = float(np.min(finite_min))
                xmax_global = float(np.max(global_max_x))
                xmin_global = max(xmin_global, 1e-20)
                if xmax_global <= xmin_global:
                    xmax_global = xmin_global * 1.01

                x_grid = np.logspace(
                    math.log10(xmin_global),
                    math.log10(xmax_global),
                    num=200,
                    base=10.0,
                )
            else:
                if x_grid_user is None:
                    raise RuntimeError("x_grid_user is None in Q_of_x mode.")
                x_grid = x_grid_user

            # now accumulate Q(x) over repeats for each time
            M = x_grid.shape[0]
            Q_sum = np.zeros((T, M), dtype=float)
            Q_count = np.zeros(T, dtype=int)

            use_qx_filter = bool(getattr(self, "Qx_filter", False)) and (not auto_x_grid)
            x_min_user = None
            if use_qx_filter:
                x_min_user = float(np.min(x_grid))

            for cdf_list in cdf_repeats:
                for it, cdf in enumerate(cdf_list):
                    if cdf is None:
                        continue
                    x_sorted, Q_sorted = cdf

                    if use_qx_filter and x_min_user is not None:
                        cdf2 = self._filter_and_renormalize_cdf_by_xmin(
                            x_sorted,
                            Q_sorted,
                            x_min_user,
                        )
                        if cdf2 is None:
                            continue
                        x_sorted, Q_sorted = cdf2

                    Q_r = self._eval_Q_of_x(x_sorted, Q_sorted, x_grid)
                    Q_sum[it] += Q_r
                    Q_count[it] += 1

            Q_mean = np.empty_like(Q_sum)
            for it in range(T):
                if Q_count[it] > 0:
                    Q_mean[it] = Q_sum[it] / float(Q_count[it])
                else:
                    Q_mean[it] = np.nan

            psd_info["x_grid"] = x_grid
            psd_info["Q_mean"] = Q_mean.T
            
            for it in range(T):
                x_50[it] = self._invert_cdf_monotone(x_grid, Q_mean[it, :], q=0.5)
            psd_info["x_50"] = x_50

        elif psd_mode == "x_of_Q":
            # ------------------------------------------------------------------
            # x(Q) mode
            # ------------------------------------------------------------------
            if Q_grid_user is None:
                raise RuntimeError("Q_grid_user is None in x_of_Q mode.")
            Q_grid = Q_grid_user
            M = Q_grid.shape[0]
            x_sum = np.zeros((T, M), dtype=float)
            x_count = np.zeros(T, dtype=int)

            for cdf_list in cdf_repeats:
                for it, cdf in enumerate(cdf_list):
                    if cdf is None:
                        continue
                    x_sorted, Q_sorted = cdf
                    x_r = self._eval_x_of_Q(x_sorted, Q_sorted, Q_grid)
                    x_sum[it] += x_r
                    x_count[it] += 1

            x_mean = np.empty_like(x_sum)
            for it in range(T):
                if x_count[it] > 0:
                    x_mean[it] = x_sum[it] / float(x_count[it])
                else:
                    x_mean[it] = np.nan

            psd_info["Q_grid"] = Q_grid
            psd_info["x_mean"] = x_mean
            
            q = 0.5
            hit = np.where(np.isclose(Q_grid, q, rtol=0.0, atol=1e-12))[0]
            if hit.size > 0:
                j = int(hit[0])
                x_50 = x_mean[:, j].astype(float, copy=False)
            else:
                for it in range(T):
                    xq = np.asarray(x_mean[it], dtype=float)
                    mask = np.isfinite(Q_grid) & np.isfinite(xq)
                    if not np.any(mask):
                        x_50[it] = float("nan")
                        continue
                    Qm = Q_grid[mask]
                    xm = xq[mask]
                    order = np.argsort(Qm)
                    Qm = Qm[order]
                    xm = xm[order]
                    xm = np.maximum.accumulate(xm)
                    if Qm[0] > q or Qm[-1] < q:
                        x_50[it] = float("nan")
                    else:
                        x_50[it] = float(np.interp(q, Qm, xm))
            psd_info["x_50"] = x_50
        else:
            raise RuntimeError(f"Unknown psd_mode={psd_mode!r}.")

        return psd_info
