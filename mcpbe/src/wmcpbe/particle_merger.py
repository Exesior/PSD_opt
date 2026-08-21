"""
Shared infrastructure for particle merging to reduce n_comp.

This module provides a centralized, optimized mechanism for finding and merging
similar particles during breakage and agglomeration events. By reusing existing
computational particles with matching intensive properties (V_dry, liquid_volume,
porosity, saturation), the total number of computational particles (n_comp) can
be significantly reduced without affecting mass conservation.

Key Features
------------
- Hash-based index for O(1) average-case lookup (configurable)
- Tolerance-based matching for intensive properties (default: 0.0001% relative)
- Unified API for breakage fragments and agglomeration children
- Mass-conserving by design: only W is modified, never volumes or liquid

Performance Characteristics
---------------------------
- Direct overhead per check: ~100-150 CPU cycles (hash lookup + key computation)
- Break-even merge probability: ~84% for net cycle savings per event
- Long-term benefit: Reduced n_comp leads to O(n²) savings in propensity rebuilds

Mass Conservation Note
----------------------
When merging particles, ONLY the computational weight W is modified. All intensive
properties (V_dry, liquid_volume, porosity, saturation, V_solid components) remain
unchanged. This is physically correct because these properties are defined PER
PHYSICAL PARTICLE, not per computational particle.

Example
-------
```python
merger = ParticleMerger(solver, use_hash_index=True, tol_rel=1e-6)

# During breakage/agglomeration:
idx, was_merged = merger.find_or_create(
    V_solid_target=V_solid,
    V_dry_target=V_dry,
    liquid_target=liquid,
    poro_target=poro,
    sat_target=sat,
    weight_to_add=dW,
    component_sum=V_comp
)

if was_merged:
    # Existing particle found: W[idx] += dW (nothing else changed!)
    pass
else:
    # New particle created with all properties initialized
    pass
```

Author: WMCPBE Team
Created: 2025
"""

import numpy as np
from typing import Optional, Tuple, Dict, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .mcpbe_base import MCPBEBase


class ParticleMerger:
    """
    Centralized logic for finding and merging similar particles.
    
    This class manages a hash-based index for fast lookup of particles with
    similar intensive properties. When a new particle would be created (during
    breakage or agglomeration), this class first checks if an existing particle
    with matching properties exists. If so, only the weight W is increased;
    otherwise, a new particle is created.
    
    Attributes
    ----------
    solver : MCPBEBase
        Reference to the parent solver instance
    use_hash_index : bool
        Whether to use hash-based indexing (recommended for large n_comp)
    tol_rel : float
        Relative tolerance for property matching (default: 1e-6 = 0.0001%)
    tol_abs_liquid : float
        Absolute tolerance for liquid volume (handles near-zero values)
    bin_digits_volume : int
        Number of decimal digits for log-scale volume binning
    bin_digits_poro : int
        Number of decimal digits for porosity/saturation binning
    
    Notes
    -----
    The hash index groups particles by binned property values. Particles in
    the same bin are then checked with exact tolerance comparison to handle
    edge cases from binning approximation.
    """
    
    def __init__(self, 
                 solver: 'MCPBEBase',
                 use_hash_index: bool = True,
                 tol_rel: float = 1e-6,
                 tol_abs_liquid: float = 1e-30,
                 tol_abs_frac: float = 1e-6,
                 bin_digits_volume: int = 8,
                 bin_digits_poro: int = 4):
        """
        Initialize the particle merger.
        
        Parameters
        ----------
        solver : MCPBEBase
            Parent solver instance that holds particle arrays
        use_hash_index : bool, optional
            Enable hash-based O(1) lookup (default: True). Disable for very
            small systems (<100 particles) where linear scan is faster.
        tol_rel : float, optional
            Relative tolerance for matching intensive properties (default: 1e-6)
        tol_abs_liquid : float, optional
            Absolute tolerance for liquid volume comparisons (default: 1e-30)
        tol_abs_frac : float, optional
            ABSOLUTE tolerance for porosity and saturation (default: 1e-6).
            Both are dimensionless fractions in [0, 1], so a relative tolerance
            is the wrong norm for them: it is undefined at 0 (which is where
            poreless particles sit) and would call 0.0001 vs 0.0002 a 100 %
            mismatch while treating 0.80 vs 0.799992 as identical.
        bin_digits_volume : int, optional
            Precision for volume binning in hash keys (default: 8 digits)
        bin_digits_poro : int, optional
            Precision for porosity/saturation binning (default: 4 digits)
        """
        self.solver = solver
        self.use_hash_index = use_hash_index
        self.tol_rel = tol_rel
        self.tol_abs_liquid = tol_abs_liquid
        self.tol_abs_frac = tol_abs_frac
        self.bin_digits_volume = bin_digits_volume
        self.bin_digits_poro = bin_digits_poro
        
        # Hash index: key = (V_dry_bin, liquid_bin, poro_bin, sat_bin), value = Set[idx]
        # Only used if use_hash_index=True
        self._hash_index: Optional[Dict[tuple, Set[int]]] = {} if use_hash_index else None

        # Reverse map idx -> key the particle was REGISTERED under.
        #
        # Why this is not optional: the key is derived from V_dry, liquid,
        # porosity and saturation, and ContinuousProcessesHandler rewrites
        # exactly those for the whole population after every MC event.
        # Recomputing the key at removal time (which is what the code used to
        # do) then yields a *different* key than the one the entry sits under,
        # so the removal silently does nothing: the index fills up with stale
        # entries and lookups miss the particles that are actually there.
        # Measured at eps_0 = 0.6: merge rate 99.4 % -> 10.9 %, and 99.8 % of
        # all entries under a key that no longer matched the particle.
        # See mcpbe/docs/historical/Audit_2026-08-17.md, B-04.
        self._key_of: Dict[int, tuple] = {} if use_hash_index else None
        
        # Statistics (for debugging/profiling)
        self._stats_merges = 0
        self._stats_creates = 0
        self._stats_lookups = 0
    
    def _resolve_tolerances(self, liquid_target: float,
                            tol_rel_override: Optional[float],
                            liquid_scale_ref: Optional[float]) -> Tuple[float, float]:
        """
        Resolve the effective (relative, liquid-absolute) tolerances for a lookup.

        The relative tolerance applies to V_dry and is the caller's rule
        ``|target - candidate| / target <= tol``, i.e. normalised by the TARGET.

        For liquid the same relative rule applies, optionally capped against a
        reference scale (``liquid_scale_ref``, in practice the droplet volume).
        Rationale: a purely target-relative liquid tolerance grows with how much
        liquid a particle already carries, so late merges accept ever larger
        absolute liquid differences while the statistics still book a whole
        droplet. Capping at droplet scale bounds that; it can only tighten the
        tolerance, never loosen it, at the price of a lower merge hit rate.

        Returns
        -------
        (tol_rel_eff, liq_tol) : Tuple[float, float]
        """
        tol = self.tol_rel if tol_rel_override is None else float(tol_rel_override)

        liq_tol = tol * abs(liquid_target)
        if liquid_scale_ref is not None:
            liq_tol = min(liq_tol, tol * abs(liquid_scale_ref))
        # Floor so that dry particles (liquid == 0) still compare equal.
        liq_tol = max(liq_tol, self.tol_abs_liquid)

        return tol, liq_tol

    def find_or_create(self,
                       V_solid_target: np.ndarray,
                       V_dry_target: float,
                       liquid_target: float,
                       poro_target: Optional[float],
                       sat_target: Optional[float],
                       weight_to_add: float,
                       component_sum: Optional[np.ndarray] = None,
                       tol_rel_override: Optional[float] = None,
                       liquid_scale_ref: Optional[float] = None,
                       exclude_idx: Optional[int] = None,
                       force_linear_scan: bool = False) -> Tuple[int, bool]:
        """
        Find existing similar particle OR create new one.
        
        This is the main entry point for breakage and agglomeration code. It
        attempts to find an existing particle with matching intensive properties.
        If found, the weight is increased; otherwise, a new particle is created.
        
        Parameters
        ----------
        V_solid_target : np.ndarray
            Target solid volume vector (dim components). Used for matching.
        V_dry_target : float
            Target dry volume (V_dry = V_solid / (1 - poro)). Primary match key.
        liquid_target : float
            Target liquid volume [m^3]. Matched with relative + absolute tolerance.
        poro_target : float or None
            Target porosity (0-1, or NaN for non-porous). Can be None if not tracked.
        sat_target : float or None
            Target saturation (0-1, or NaN if not applicable). Can be None.
        weight_to_add : float
            Computational weight to add (dW). Added to existing W if merged.
        component_sum : np.ndarray, optional
            Pre-computed component sum for V_flat[:dim]. If None, computed from
            V_solid_target.
        tol_rel_override : float, optional
            Use this relative tolerance instead of ``self.tol_rel``. Nucleation
            passes ``NucleationConfig.similarity_tol``.
        liquid_scale_ref : float, optional
            Cap the liquid tolerance against this scale (droplet volume for
            nucleation). See :meth:`_resolve_tolerances`.
        exclude_idx : int, optional
            Never match this index. Nucleation passes the source particle: a
            self-match would do ``W[i] += dW`` followed by ``W[i] -= dW`` while
            ``liquid_volume[i]`` is never updated, silently losing the droplet.
        force_linear_scan : bool, optional
            Skip the hash index for this lookup. The hash bins liquid on a log
            scale, so a match that is within an absolute (droplet-capped)
            tolerance can still fall into a neighbouring bin and be missed.
            The linear scan also returns the lowest matching index, which keeps
            results reproducible; the hash path iterates a set.

        Returns
        -------
        (index, was_merged) : Tuple[int, bool]
            index : Index of the particle (existing if merged, new if created)
            was_merged : True if added weight to existing particle, False if created new
        
        Raises
        ------
        ValueError
            If weight_to_add <= 0 or V_dry_target <= 0
        
        Notes
        -----
        **Mass Conservation:**
        When was_merged=True, ONLY W[index] is modified. All intensive properties
        remain unchanged, which is physically correct since they represent per-
        physical-particle quantities.
        
        **Sampler Updates:**
        Caller is responsible for updating Fenwick samplers after this call:
        - If merged: Update weight-dependent samplers (break_rate, agg_propensity)
        - If created: Initialize samplers for new particle
        
        Examples
        --------
        >>> idx, merged = merger.find_or_create(
        ...     V_solid_target=V_frag,
        ...     V_dry_target=V_dry_frag,
        ...     liquid_target=liq_frag,
        ...     poro_target=poro_frag,
        ...     sat_target=sat_frag,
        ...     weight_to_add=dW,
        ...     component_sum=V_frag
        ... )
        >>> if merged:
        ...     solver._break_sampler.update(idx, new_rate)
        >>> else:
        ...     solver._break_rate[idx] = new_rate
        """
        if weight_to_add <= 0:
            raise ValueError(f"weight_to_add must be positive, got {weight_to_add}")
        if V_dry_target <= 0:
            raise ValueError(f"V_dry_target must be positive, got {V_dry_target}")
        
        self._stats_lookups += 1
        
        # === DEBUG MERGER: FIND_OR_CREATE ===
        solver = self.solver
        if getattr(solver, 'mcpbe_debug_mass', False) and getattr(solver, 'mcpbe_debug_merger', True) and self._stats_lookups % 10 == 0:
            print(f"\n[MERGER DEBUG] Lookup #{self._stats_lookups}")
            if poro_target is not None:
                poro_str = f"{poro_target:.4f}"
            else:
                poro_str = 'nan'
            if sat_target is not None:
                sat_str = f"{sat_target:.4f}"
            else:
                sat_str = 'nan'
            print(f"  Target: V_dry={V_dry_target:.6e}, liq={liquid_target:.6e}, poro={poro_str}, sat={sat_str}")
            print(f"  Tol_rel={self.tol_rel:.1e}, tol_abs_liq={self.tol_abs_liquid:.1e}")
# ====================================
        
        # Try to find match
        match_idx = self.find_similar(
            V_solid_target=V_solid_target,
            V_dry_target=V_dry_target,
            liquid_target=liquid_target,
            poro_target=poro_target,
            sat_target=sat_target,
            tol_rel_override=tol_rel_override,
            liquid_scale_ref=liquid_scale_ref,
            exclude_idx=exclude_idx,
            force_linear_scan=force_linear_scan
        )
        
        if match_idx >= 0:
            # MERGE: Just add weight, nothing else!
            self.solver.W[match_idx] += weight_to_add
            self._stats_merges += 1
            
            # === DEBUG MERGER: MERGED ===
            if getattr(solver, 'mcpbe_debug_mass', False) and getattr(solver, 'mcpbe_debug_merger', True):
                print(f"  -> MERGED at idx={match_idx}: W[{match_idx}] += {weight_to_add:.2f} -> {self.solver.W[match_idx]:.2f}")
# ====================================
            return match_idx, True
        
        # CREATE: Append new particle
        self._stats_creates += 1
        
        # === DEBUG MERGER: CREATING NEW ===
        if getattr(solver, 'mcpbe_debug_mass', False) and getattr(solver, 'mcpbe_debug_merger', True):
            print(f"  -> CREATING NEW particle")
# ====================================
        
        new_idx = self._create_new_particle(
            V_solid=V_solid_target,
            V_dry=V_dry_target,
            liquid=liquid_target,
            poro=poro_target,
            sat=sat_target,
            weight=weight_to_add,
            component_sum=component_sum
        )
        return new_idx, False
    
    def find_similar(self,
                     V_solid_target: np.ndarray,
                     V_dry_target: float,
                     liquid_target: float,
                     poro_target: Optional[float],
                     sat_target: Optional[float],
                     tol_rel_override: Optional[float] = None,
                     liquid_scale_ref: Optional[float] = None,
                     exclude_idx: Optional[int] = None,
                     force_linear_scan: bool = False) -> int:
        """
        Find index of similar particle, or -1 if not found.
        
        Uses hash index if enabled and populated, otherwise falls back to
        linear scan over all active particles.
        
        Parameters
        ----------
        V_solid_target : np.ndarray
            Target solid volume (for verification, not primary key)
        V_dry_target : float
            Target dry volume (primary matching criterion)
        liquid_target : float
            Target liquid volume
        poro_target : float or None
            Target porosity
        sat_target : float or None
            Target saturation
        
        Returns
        -------
        int
            Index of matching particle, or -1 if no match found
        
        Notes
        -----
        Matching rules (see :meth:`_resolve_tolerances`):
        - ``V_dry``: relative to the target, ``|a - b| <= tol_rel * |b|``
        - ``liquid``: relative to the target, optionally capped at
          ``tol_rel * liquid_scale_ref``, floored at ``tol_abs_liquid`` so dry
          particles still compare equal
        - ``porosity`` / ``saturation``: ABSOLUTE, ``|a - b| <= tol_abs_frac``
          (bounded fractions -- relative is undefined at 0)

        Uses hash index if enabled and populated, otherwise falls back to
        linear scan over all active particles.
        """
        # === DEBUG MERGER: FIND_SIMILAR ENTRY ===
        solver = self.solver
        if getattr(solver, 'mcpbe_debug_mass', False) and getattr(solver, 'mcpbe_debug_merger', True) and self._stats_lookups % 50 == 0:
            key = self._compute_hash_key(V_dry_target, liquid_target, poro_target, sat_target)
            candidates = self._hash_index.get(key, set()) if self.use_hash_index else []
            print(f"  find_similar: hash_enabled={self.use_hash_index}, candidates_in_bin={len(candidates)}")
# ====================================
        tol, liq_tol = self._resolve_tolerances(
            liquid_target, tol_rel_override, liquid_scale_ref
        )

        if self.use_hash_index and self._hash_index is not None and not force_linear_scan:
            return self._find_via_hash(V_dry_target, liquid_target, poro_target, sat_target,
                                       tol, liq_tol, exclude_idx)
        else:
            return self._find_linear_scan(V_solid_target, V_dry_target, liquid_target,
                                          poro_target, sat_target, tol, liq_tol, exclude_idx)

    def _find_via_hash(self, V_dry: float, liquid: float,
                       poro: Optional[float], sat: Optional[float],
                       tol: float, liq_tol: float,
                       exclude_idx: Optional[int] = None) -> int:
        """
        O(1) average lookup via hash index.
        
        Hash collisions are possible due to binning approximation, so candidates
        are verified with exact tolerance comparison.
        
        Parameters
        ----------
        V_dry : float
            Dry volume [m^3] for hash key computation
        liquid : float
            Liquid volume [m^3] for hash key computation
        poro : float or None
            Porosity for hash key computation
        sat : float or None
            Saturation for hash key computation
        
        Returns
        -------
        int
            Index of matching particle, or -1 if no match found
        """
        key = self._compute_hash_key(V_dry, liquid, poro, sat)
        candidates = self._hash_index.get(key, set())

        if not candidates:
            return -1

        # Verify with exact tolerance check (hash collisions possible due to binning)
        for idx in candidates:
            if exclude_idx is not None and idx == exclude_idx:
                continue
            if self._matches_exact(idx, V_dry, liquid, poro, sat, tol, liq_tol):
                return idx
        return -1
    
    def _find_linear_scan(self, V_solid_target, V_dry_target, liquid_target,
                          poro_target, sat_target, tol, liq_tol,
                          exclude_idx: Optional[int] = None) -> int:
        """
        Fallback: O(n) linear scan over active particles.
        
        This matches the behavior of the original _find_similar_particle in
        mcpbe_nucleation.py, but extended to support optional poro/sat matching.
        
        Parameters
        ----------
        V_solid_target : np.ndarray
            Target solid volume
        V_dry_target : float
            Target dry volume
        liquid_target : float
            Target liquid volume
        poro_target : float or None
            Target porosity
        sat_target : float or None
            Target saturation
        
        Returns
        -------
        int
            Index of first matching particle, or -1 if no match
        """
        solver = self.solver
        n_active = solver.a_tot
        if n_active <= 0:
            return -1
        
        V_tol = tol * abs(V_dry_target)
        frac_tol = self.tol_abs_frac

        # Vectorized checks (NumPy boolean masking)
        mask = np.abs(solver.V_flat[-1, :n_active] - V_dry_target) <= V_tol
        if exclude_idx is not None and 0 <= exclude_idx < n_active:
            mask[exclude_idx] = False
        if not mask.any():
            return -1

        mask &= np.abs(solver.liquid_volume[:n_active] - liquid_target) <= liq_tol
        if not mask.any():
            return -1

        if poro_target is not None:
            poro = solver.porosity[:n_active]
            # Match if both are NaN, or both are finite and within tolerance
            poro_match = (np.isnan(poro) & np.isnan(poro_target)) | \
                         (~np.isnan(poro) & ~np.isnan(poro_target) &
                          (np.abs(poro - poro_target) <= frac_tol))
            mask &= poro_match
            if not mask.any():
                return -1

        if sat_target is not None:
            sat = solver.saturation[:n_active]
            sat_match = (np.isnan(sat) & np.isnan(sat_target)) | \
                        (~np.isnan(sat) & ~np.isnan(sat_target) &
                         (np.abs(sat - sat_target) <= frac_tol))
            mask &= sat_match
            if not mask.any():
                return -1

        # Return first match (lowest index for determinism)
        matches = np.where(mask)[0]
        return int(matches[0]) if len(matches) > 0 else -1
    
    def _compute_hash_key(self, V_dry: float, liquid: float,
                          poro: Optional[float], sat: Optional[float]) -> tuple:
        """
        Compute hash key with binning for tolerance-based grouping.
        
        Uses log-scale binning for volumes (covers many orders of magnitude)
        and linear binning for porosity/saturation (bounded 0-1 range).
        
        Parameters
        ----------
        V_dry : float
            Dry volume [m^3]
        liquid : float
            Liquid volume [m^3]
        poro : float or None
            Porosity (0-1, or NaN)
        sat : float or None
            Saturation (0-1, or NaN)
        
        Returns
        -------
        tuple
            Hashable key: (V_bin, liq_bin, poro_bin, sat_bin)
        
        Notes
        -----
        Binning precision is controlled by bin_digits_volume and bin_digits_poro.
        Higher precision -> more bins -> lower collision rate but sparser index.
        """
        # Log-scale binning for volumes (covers many orders of magnitude)
        if V_dry > 0 and np.isfinite(V_dry):
            V_bin = round(np.log10(V_dry), self.bin_digits_volume)
        else:
            V_bin = -np.inf
        
        if liquid > self.tol_abs_liquid and np.isfinite(liquid):
            liq_bin = round(np.log10(liquid), self.bin_digits_volume)
        else:
            liq_bin = 0  # Dry particle
        
        # Linear binning for porosity/saturation (0-1 range)
        if poro is not None and np.isfinite(poro):
            poro_bin = round(poro, self.bin_digits_poro)
        else:
            poro_bin = None  # NaN or not tracked
        
        if sat is not None and np.isfinite(sat):
            sat_bin = round(sat, self.bin_digits_poro)
        else:
            sat_bin = None  # NaN or not tracked
        
        return (V_bin, liq_bin, poro_bin, sat_bin)
    
    def _matches_exact(self, idx: int, V_dry: float, liquid: float,
                       poro: Optional[float], sat: Optional[float],
                       tol: Optional[float] = None,
                       liq_tol: Optional[float] = None) -> bool:
        """
        Verify exact match within tolerances for a candidate index.
        
        Called after hash lookup to filter false positives from binning.
        
        Parameters
        ----------
        idx : int
            Index of candidate particle to verify
        V_dry : float
            Target dry volume
        liquid : float
            Target liquid volume
        poro : float or None
            Target porosity
        sat : float or None
            Target saturation
        
        Returns
        -------
        bool
            True if all properties match within tolerances
        """
        solver = self.solver
        if tol is None:
            tol = self.tol_rel
        if liq_tol is None:
            liq_tol = max(tol * abs(liquid), self.tol_abs_liquid)
        frac_tol = self.tol_abs_frac

        # === DEBUG MERGER: EXACT MATCH CHECK ===
        if getattr(solver, 'mcpbe_debug_mass', False) and getattr(solver, 'mcpbe_debug_merger', True):
            V_dry_idx = solver.V_flat[-1, idx]
            liq_idx = solver.liquid_volume[idx]
            poro_idx = solver.porosity[idx]
            sat_idx = solver.saturation[idx] if hasattr(solver, 'saturation') else np.nan
            
            delta_V = abs(V_dry_idx - V_dry)
            delta_liq = abs(liq_idx - liquid)
            rel_tol_V = tol * abs(V_dry)

            match_V = delta_V <= rel_tol_V
            match_liq = delta_liq <= liq_tol
            print(f"  Candidate idx={idx}: V_diff={delta_V:.3e}/{rel_tol_V:.3e}({'OK' if match_V else 'FAIL'}), liq_diff={delta_liq:.3e}/{liq_tol:.3e}({'OK' if match_liq else 'FAIL'})")
# ====================================

        # Check V_dry (relative to target)
        if abs(solver.V_flat[-1, idx] - V_dry) > tol * abs(V_dry):
            return False

        # Check liquid_volume (relative to target, optionally capped, with floor)
        if abs(solver.liquid_volume[idx] - liquid) > liq_tol:
            return False

        # Check porosity if applicable (ABSOLUTE tolerance - bounded fraction)
        if poro is not None:
            p = solver.porosity[idx]
            # Both NaN -> match; both finite -> check tolerance
            if np.isnan(p) and np.isnan(poro):
                pass  # Match
            elif np.isnan(p) or np.isnan(poro):
                return False  # Mismatch: one NaN, one not
            elif abs(p - poro) > frac_tol:
                return False

        # Check saturation if applicable (ABSOLUTE tolerance - bounded fraction)
        if sat is not None:
            s = solver.saturation[idx]
            if np.isnan(s) and np.isnan(sat):
                pass  # Match
            elif np.isnan(s) or np.isnan(sat):
                return False
            elif abs(s - sat) > frac_tol:
                return False
        
        # === DEBUG MERGER: MATCH RESULT ===
        if getattr(solver, 'mcpbe_debug_mass', False) and getattr(solver, 'mcpbe_debug_merger', True):
            print(f"  -> OVERALL: MATCH")
# ====================================
        return True
    
    def _create_new_particle(self, V_solid, V_dry, liquid, poro, sat, weight, component_sum):
        """
        Append new particle and initialize all properties.
        
        Parameters
        ----------
        V_solid : np.ndarray
            Solid volume vector (dim components)
        V_dry : float
            Dry volume (V_flat[-1])
        liquid : float
            Liquid volume
        poro : float or None
            Porosity (NaN if not applicable)
        sat : float or None
            Saturation (NaN if not applicable)
        weight : float
            Initial computational weight W
        component_sum : np.ndarray or None
            Pre-computed component vector for V_flat[:dim]
        
        Returns
        -------
        int
            Index of newly created particle
        """
        solver = self.solver
        
        # Determine component vector
        if component_sum is not None:
            comp = np.asarray(component_sum, dtype=float).copy()
        elif solver.dim == 1:
            comp = np.array([float(np.sum(V_solid))], dtype=float)
        else:
            # Multi-component: use V_solid directly (already distributed)
            comp = np.asarray(V_solid, dtype=float).copy()
        
        # Append column to all arrays
        solver._append_particle_column(comp)
        new_idx = solver.a_tot - 1
        
        # Initialize properties
        solver.W[new_idx] = weight
        # V_dry (not V_solid!) -- and X has to follow, see mcpbe_base.
        # set_particle_dry_volume. _append_particle_column could only set X from
        # the solid volume it was handed.
        solver.set_particle_dry_volume(new_idx, V_dry)
        
        if hasattr(solver, 'liquid_volume'):
            solver.liquid_volume[new_idx] = float(liquid)
        
        if hasattr(solver, 'porosity'):
            solver.porosity[new_idx] = float(poro) if poro is not None else np.nan
        
        if hasattr(solver, 'saturation'):
            solver.saturation[new_idx] = float(sat) if sat is not None else np.nan
        
        # Add to hash index
        self.register_particle(new_idx)

        # === DEBUG MERGER: CREATE NEW ===
        if getattr(solver, 'mcpbe_debug_mass', False) and getattr(solver, 'mcpbe_debug_merger', True):
            poro_val = solver.porosity[new_idx]
            sat_val = solver.saturation[new_idx] if hasattr(solver, 'saturation') else None
            poro_str = f"{poro_val:.4f}" if np.isfinite(poro_val) else 'N/A'
            sat_str = f"{sat_val:.4f}" if sat_val is not None and np.isfinite(sat_val) else 'N/A'
            print(f"\n[MERGER DEBUG] Created NEW particle at idx={new_idx}")
            print(f"  W={solver.W[new_idx]:.2f}")
            print(f"  V_dry={solver.V_flat[-1, new_idx]:.6e}")
            print(f"  V_solid={np.sum(solver.V_flat[:solver.dim, new_idx]):.6e}")
            print(f"  liquid={solver.liquid_volume[new_idx]:.6e}")
            print(f"  porosity={poro_str}")
            print(f"  saturation={sat_str}")
        # ====================================
        
        return new_idx
    
    def _update_hash_on_weight_change(self, idx: int):
        """
        Called when W[idx] changes but intensive properties stay same.
        
        No hash update needed: hash key depends only on intensive properties,
        not on weight. This method exists for API completeness and potential
        future extensions.
        
        Parameters
        ----------
        idx : int
            Index of particle whose weight changed
        """
        pass  # No-op: hash key doesn't depend on W
    
    def register_particle(self, idx: int) -> None:
        """
        Add an existing particle to the hash index, reading its current state.

        Use this for particles created outside :meth:`find_or_create` -- e.g.
        the child of a nucleation-driven manual agglomeration, which is appended
        directly because the surrounding code has to track its index through
        several parent removals. Without registration such a particle is
        invisible to every later lookup, so agglomeration and breakage would
        create duplicates instead of merging into it.

        Parameters
        ----------
        idx : int
            Index of the particle to register.
        """
        if not self.use_hash_index or self._hash_index is None:
            return

        solver = self.solver
        if idx < 0 or idx >= solver.a_tot:
            return

        V_dry = solver.V_flat[-1, idx]
        liquid = solver.liquid_volume[idx] if hasattr(solver, 'liquid_volume') else 0.0
        poro = solver.porosity[idx] if hasattr(solver, 'porosity') else np.nan
        sat = solver.saturation[idx] if hasattr(solver, 'saturation') else np.nan

        key = self._compute_hash_key(V_dry, liquid, poro, sat)
        # Drop any previous registration of this slot first: slots are recycled
        # by swap-with-last removal, so `idx` may still be listed under the key
        # of a completely unrelated particle.
        self._unlink(idx)
        if key not in self._hash_index:
            self._hash_index[key] = set()
        self._hash_index[key].add(idx)
        self._key_of[idx] = key

    def _unlink(self, idx: int) -> None:
        """Drop ``idx`` from the bucket it is currently registered under.

        Uses the recorded key, never a recomputed one -- see ``_key_of``.
        """
        if not self.use_hash_index or self._hash_index is None:
            return
        key = self._key_of.pop(idx, None)
        if key is None:
            return
        bucket = self._hash_index.get(key)
        if bucket is None:
            return
        bucket.discard(idx)
        if not bucket:
            del self._hash_index[key]

    def remove_from_hash_index(self, idx: int):
        """
        Remove a particle from the hash index (called when particle is deleted).

        Parameters
        ----------
        idx : int
            Index of particle to remove
        """
        self._unlink(idx)

    def notify_index_swap(self, old_idx: int, new_idx: int) -> None:
        """
        Re-key a hash index entry after swap-with-last particle removal.

        ``mcpbe_base.py::_remove_particle_column`` deletes a particle by moving
        whatever currently sits at the last slot (``old_idx``) into the freed
        slot (``new_idx``). The hash index stores indices, not particle
        identity, so without this call the entry for that particle would keep
        pointing at ``old_idx`` -- a slot that becomes free and gets reused by
        the next unrelated ``_append_particle_column`` call. A later lookup
        could then silently match against that unrelated particle and add
        weight to it instead of the intended one, corrupting mass conservation.

        Must be called BEFORE the array swap happens, while ``old_idx`` still
        holds the particle's current property values.

        Parameters
        ----------
        old_idx : int
            Index the particle occupied before the swap (the "last" slot).
        new_idx : int
            Index the particle occupies after the swap (the freed slot).
        """
        if not self.use_hash_index or self._hash_index is None:
            return

        # The slot `new_idx` is being freed: whatever used to live there is
        # gone, so its registration must go too, or it would survive as a
        # stale entry pointing at the particle that moves in.
        self._unlink(new_idx)

        # Re-key by the RECORDED key, not a recomputed one: the particle's
        # properties may have changed since registration (compression,
        # internalization), in which case a recomputed key would not find the
        # entry and the move would silently be lost. See B-04.
        key = self._key_of.pop(old_idx, None)
        if key is None:
            return
        bucket = self._hash_index.get(key)
        if bucket is None:
            return
        bucket.discard(old_idx)
        bucket.add(new_idx)
        self._key_of[new_idx] = key

    def rebuild_hash_index(self):
        """
        Rebuild hash index from scratch.
        
        Called after reconstruction (CAM/RS/2PM/QMX) or when enabling the
        merger mid-simulation.
        
        Notes
        -----
        This is O(n) where n = a_tot (number of active particles). Should be
        called sparingly, e.g., only after major restructuring events.
        """
        if not self.use_hash_index:
            return

        self._hash_index = {}
        self._key_of = {}
        solver = self.solver

        for idx in range(solver.a_tot):
            if solver.W[idx] <= 0:
                continue

            V_dry = solver.V_flat[-1, idx]
            liquid = solver.liquid_volume[idx] if hasattr(solver, 'liquid_volume') else 0.0
            poro = solver.porosity[idx] if hasattr(solver, 'porosity') else np.nan
            sat = solver.saturation[idx] if hasattr(solver, 'saturation') else np.nan

            key = self._compute_hash_key(V_dry, liquid, poro, sat)
            if key not in self._hash_index:
                self._hash_index[key] = set()
            self._hash_index[key].add(idx)
            self._key_of[idx] = key

    def reindex_all(self) -> int:
        """Re-key every registered particle whose hash key has drifted.

        Belt-and-braces counterpart to :meth:`reindex_particles`: instead of
        trusting each write site to report what it changed, this checks the
        whole active slice. Entries whose key is unchanged -- the vast majority
        -- cost one key computation and one dict lookup and are left alone, so
        no bucket is touched unless it has to be.

        Called once per MC event from ``solve()``. Disable with
        ``solver._merger_full_reindex = False`` if profiling shows it matters;
        the targeted ``reindex_particles`` calls from the continuous processes
        then still cover the bulk writers.

        Returns the number of particles that actually moved.
        """
        if not self.use_hash_index or self._hash_index is None:
            return 0
        if not self._key_of:
            return 0
        return self.reindex_particles(list(self._key_of.keys()))

    def reindex_particles(self, indices) -> int:
        """Re-key particles whose intensive properties have changed.

        The continuous processes (compression, internalization) rewrite
        ``V_dry``, ``porosity`` and ``saturation`` in bulk, which changes the
        hash key. Without this call the entries stay under their old key and
        the index degenerates -- see ``_key_of`` and B-04.

        Cost is O(k) for k changed particles, with a dict update each. That is
        the same order as the propensity rebuild that already runs once per
        MC event, and it only touches particles that actually moved: entries
        whose recomputed key is unchanged are skipped without touching the
        buckets.

        No-op when the hash index is disabled -- then ``find_similar`` does a
        linear scan and there is no index to maintain.
        """
        if not self.use_hash_index or self._hash_index is None:
            return 0

        solver = self.solver
        a_tot = solver.a_tot
        has_liquid = hasattr(solver, 'liquid_volume')
        has_poro = hasattr(solver, 'porosity')
        has_sat = hasattr(solver, 'saturation')
        moved = 0

        for idx in indices:
            idx = int(idx)
            if idx < 0 or idx >= a_tot:
                # Slot no longer active: drop the registration outright.
                self._unlink(idx)
                continue
            old_key = self._key_of.get(idx)
            if old_key is None:
                continue  # not registered -- nothing to move
            new_key = self._compute_hash_key(
                solver.V_flat[-1, idx],
                solver.liquid_volume[idx] if has_liquid else 0.0,
                solver.porosity[idx] if has_poro else np.nan,
                solver.saturation[idx] if has_sat else np.nan,
            )
            if new_key == old_key:
                continue
            bucket = self._hash_index.get(old_key)
            if bucket is not None:
                bucket.discard(idx)
                if not bucket:
                    del self._hash_index[old_key]
            self._hash_index.setdefault(new_key, set()).add(idx)
            self._key_of[idx] = new_key
            moved += 1
        return moved
    
    def get_statistics(self) -> dict:
        """
        Return statistics about merge operations.
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 'lookups': Total number of find_or_create calls
            - 'merges': Number of successful merges (reused existing particle)
            - 'creates': Number of new particles created
            - 'merge_rate': Fraction of lookups that resulted in merges
        """
        total = self._stats_merges + self._stats_creates
        merge_rate = self._stats_merges / total if total > 0 else 0.0
        
        return {
            'lookups': self._stats_lookups,
            'merges': self._stats_merges,
            'creates': self._stats_creates,
            'merge_rate': merge_rate,
        }
    
    def reset_statistics(self):
        """Reset all statistics counters to zero."""
        self._stats_lookups = 0
        self._stats_merges = 0
        self._stats_creates = 0
