"""
Cone Model Porosity Kernel.

Geometrisches Modell für Porenbildung/-verlust basierend auf Kontaktgeometrie
zwischen sphärischen Partikeln.

Physical Model:
    Wenn zwei Partikel kontaktieren, entsteht eine komplexe Geometrie zwischen ihnen:
    - Ein Kegelstumpf verbindet die beiden Kugeln
    - Zwei Halbkugeln schließen die Enden ab
    - Diese "Kegelpille" hat mehr Volumen als die Summe der Einzelkugeln
    
    Dieses zusätzliche Volumen ΔV repräsentiert neu eingeschlossenen Porenraum.
    
    Bei Agglomeration: ΔV wird zum Porenvolumen HINZUGEFÜGT (Porenwachstum)
    Bei Breakage: ΔV wird vom Porenvolumen ABGEZOGEN (Porenverlust durch neue Oberfläche)

Geometry:
    Kegelpille = Kegelstumpf + 2 Halbkugeln
    
    V_cone = (1/3) × π × (ri + rj) × (ri² + ri×rj + rj²)
    V_hemi = (2/3) × π × (ri³ + rj³)
    V_pill = V_cone + V_hemi
    
    ΔV = V_pill - (V_i + V_j)
    
    Für ri = rj = r:
        V_old = (8/3) × π × r³
        V_cone = 2 × π × r³
        V_hemi = (4/3) × π × r³
        V_pill = (10/3) × π × r³
        ΔV = (2/3) × π × r³  > 0

Layering (stark ungleiche Partikel):
    Die Kegelpille ist nur bei ähnlichen Radien eine echte Hülle beider Kugeln.
    Bei ungleichen Radien schneidet der gerade Kegelmantel in die große Kugel
    hinein und ΔV wird negativ (positiv nur für r_j/r_i > (3-√5)/2 ≈ 0.381966).

    Dort greift stattdessen ein Layering-Modell: das kleine Partikel sitzt auf
    der lokal ebenen Oberfläche des großen, der neue Porenraum ist der
    umschreibende Zylinder abzüglich der Halbkugel darin:

        ΔV_layering = π × r_min³ - (2/3) × π × r_min³ = (1/3) × π × r_min³

    Verwendet wird ΔV = max(ΔV_kegelpille, ΔV_layering); der Übergang liegt bei
    r_j/r_i = 0.403032 und ist stetig.

    LAYERING ist die Anlagerung feiner Partikel an ein deutlich größeres
    Granulat -- ein eigenständiger, wichtiger Wachstumsmechanismus der
    Granulation neben Nukleation, Koaleszenz und Bruch.

Parameters:
    k_agg: Shape correction factor für Agglomeration [dimensionless]
           Skaliert ΔV vor Addition zum Porenvolumen.
           Typisch: 0.1 - 2.0 (fitbar an Experimente)
           
    k_break: Shape correction factor für Breakage [dimensionless]
           Skaliert ΔV vor Subtraktion vom Porenvolumen.
           Typisch: 0.01 - 0.5 (kleiner wegen (n-1) Kegel in Reihe)
           
    Why two parameters?
        - Agglomeration: 1 Kegel zwischen 2 Partikeln → ΔV direkt anwendbar
        - Breakage: (n-1) Kegel zwischen n Fragmenten → ΔV summiert sich stark
          → Benötigt stärkeres Damping für physikalisch sinnvolle Ergebnisse

References:
    - Kegelstumpf-Volumen: Standard geometrische Formel
    - Anwendung auf Granulation: Neu in diesem Modell (fitbar an PSD-Daten)

Example:
    >>> kernel = ConeModelKernel(k_agg=0.5, k_break=0.1)
    
    # Agglomeration
    >>> V_dry_new, poro_new = kernel.compute_merged_porosity(
    ...     v_dry1=1e-18, poro1=0.4,
    ...     v_dry2=2e-18, poro2=0.5
    ... )
    
    # Breakage
    >>> poro_frags = kernel.compute_fragment_porosity(
    ...     parent_porosity=0.45,
    ...     fragment_volumes=[0.5e-18, 0.3e-18, 0.2e-18],
    ...     parent_volume=1.0e-18
    ... )
"""

import numpy as np
from ..base import PorosityGrowthKernel
from typing import List, Union


class ConeModelKernel(PorosityGrowthKernel):
    """
    Cone Model Porosity Kernel for geometric pore formation/loss.
    
    This kernel implements a physics-based model where pore volume changes
    due to the geometry of contacting spheres:
    
    Agglomeration (2 → 1):
        - Two spheres contact → cone frustum forms between them
        - Additional volume ΔV becomes new pore space
        - V_solid is conserved (mass conservation!)
        - V_pore increases by k_agg × ΔV
        
    Breakage (1 → n):
        - Parent sphere breaks into n fragments
        - (n-1) cone frustums form between adjacent fragments
        - Surface area increases → pore space decreases
        - V_solid is conserved (mass conservation!)
        - V_pore decreases by k_break × ΣΔV
    
    Key Design Principles:
        1. MASS CONSERVATION: V_solid is ALWAYS conserved
        2. INTENSIVE PROPERTIES: All volumes are per PHYSICAL particle
        3. POSITIVE PORES: V_pore ≥ 0 always (clamped)
        4. VALID POROSITY: 0 ≤ ε < 1 always (clamped)
    
    Parameters:
        k_agg: Shape correction factor for agglomeration (default: 1.0)
               Scales ΔV added to pore volume during merging.
               Physical interpretation: Fraction of geometric ΔV that 
               becomes accessible pore space.
               
        k_break: Shape correction factor for breakage (default: 1.0)
                 Scales ΔV subtracted from pore volume during fragmentation.
                 Physical interpretation: Accounts for (n-1) cones in series
                 and pore collapse at fresh fracture surfaces.
    
    Example:
        >>> kernel = ConeModelKernel(k_agg=1.0, k_break=1.0)
        
        # Equal-sized agglomeration
        >>> V_dry, poro = kernel.compute_merged_porosity(
        ...     v_dry1=1e-18, poro1=0.4,
        ...     v_dry2=1e-18, poro2=0.4
        ... )
        # Result: poro > 0.4 (pore growth from contact geometry)
        
        # Binary breakage
        >>> poros = kernel.compute_fragment_porosity(
        ...     parent_porosity=0.5,
        ...     fragment_volumes=[0.5e-18, 0.5e-18],
        ...     parent_volume=1.0e-18
        ... )
        # Result: poros < 0.5 (pore loss from new surface)
    """
    
    #: Upper bound on porosity. Kept identical to `poro_max` of the
    #: `powerlaw_rumpf` breakage kernel so that the two never disagree about
    #: where the physically meaningful range ends.
    PORO_MAX = 0.9999

    @property
    def name(self) -> str:
        return 'cone_model'

    def get_default_params(self) -> dict:
        return {
            'k_agg': 1.0,       # Shape correction for agglomeration
            'k_break': 1.0,     # Shape correction for breakage (smaller due to n-1 cones)
        }

    def get_optional_params(self) -> list:
        """Keys that are honoured when present but have no default.

        Both are read via ``in self.params`` / ``.get()`` rather than from
        `get_default_params`, because *absent* is a meaningful state here:

        - ``default_porosity``: seeds a nucleation porosity. Without it,
          `compute_nucleation_porosity` returns a poreless particle (0.0) and
          porosity arises purely geometrically at contact.
        - ``liquid_split_ratio``: read by `mcpbe_nucleation._get_liquid_split_
          ratio` for the internal/external split of a fresh droplet; only
          reachable when no liquid_internalization kernel is active.
        """
        return ['default_porosity', 'liquid_split_ratio']


    def __init__(self, **params):
        defaults = self.get_default_params()
        defaults.update(params)
        self.params = self.validate_params(defaults)
        
        # Cache parameters
        self.k_agg = float(self.params['k_agg'])
        self.k_break = float(self.params['k_break'])
    
    def validate_params(self, params: dict) -> dict:
        """Validate shape correction factors."""
        if params['k_agg'] < 0:
            raise ValueError("k_agg must be non-negative")
        if params['k_break'] < 0:
            raise ValueError("k_break must be non-negative")
        # Warn if k_break is large (can cause negative pores)
        if params['k_break'] > 1.0:
            import warnings
            warnings.warn(
                f"k_break={params['k_break']} is large. "
                "This may cause negative pore volumes for multi-fragment breakage. "
                "Recommended range: 0.01 - 0.5",
                UserWarning,
                stacklevel=2
            )
        return params
    
    # =====================================================================
    # Helper: Geometric calculations
    # =====================================================================
    
    @staticmethod
    def _radius_from_volume(V: float) -> float:
        """Calculate sphere radius from volume."""
        if V <= 0:
            return 0.0
        return (3.0 * V / (4.0 * np.pi)) ** (1.0 / 3.0)
    
    @staticmethod
    def _volume_from_radius(r: float) -> float:
        """Calculate sphere volume from radius."""
        return (4.0 / 3.0) * np.pi * r ** 3
    
    def _cone_pill_volume(self, ri: float, rj: float) -> float:
        """
        Calculate volume of cone-pill (cone frustum + 2 hemispheres).
        
        Geometry:
              ╭─────╮
             ╱       ╲
            │   ri    │◄── Cone frustum ──►│   rj    │
             ╲       ╱
              ╰─────╯
        
        V_cone = (1/3) × π × (ri + rj) × (ri² + ri×rj + rj²)
        V_hemi = (2/3) × π × (ri³ + rj³)
        V_pill = V_cone + V_hemi
        
        Args:
            ri: Radius of sphere i [m]
            rj: Radius of sphere j [m]
        
        Returns:
            V_pill: Volume of cone-pill [m³]
        """
        # Cone frustum volume
        V_cone = (1.0 / 3.0) * np.pi * (ri + rj) * (ri**2 + ri*rj + rj**2)
        
        # Two hemispheres (one on each end)
        V_hemi = (2.0 / 3.0) * np.pi * (ri**3 + rj**3)
        
        return V_cone + V_hemi
    
    @staticmethod
    def _delta_volume_layering(r_min: float) -> float:
        """ΔV for a small particle deposited on a much larger one (LAYERING).

        Geometry: the smaller sphere rests on a plane -- the surface of the far
        larger partner, which is locally flat at this size ratio. The newly
        enclosed void is the upright cylinder that circumscribes it (height
        r_min, radius r_min) minus the hemisphere sitting inside::

            V_cyl  = π × r_min³
            V_hemi = (2/3) × π × r_min³
            ΔV     = (1/3) × π × r_min³

        NOTE: this term represents **layering** -- the deposition of fines onto
        a much larger granule. Layering is a growth mechanism of granulation in
        its own right (alongside nucleation, coalescence and breakage) and is
        geometrically different from the coalescence of two comparably sized
        particles that the cone-pill model describes. Which of the two applies
        is decided by size ratio alone, see :meth:`_delta_volume_pair`.

        Args:
            r_min: Radius of the SMALLER sphere [m]

        Returns:
            ΔV: Enclosed void from layering [m³], ≥ 0
        """
        return (1.0 / 3.0) * np.pi * r_min ** 3

    def _delta_volume_pair(self, Vi: float, Vj: float) -> float:
        """
        Calculate ΔV for a pair of contacting spheres.

        Two competing geometries, the larger contribution wins::

            ΔV = max( ΔV_cone-pill , ΔV_layering )

        **Cone-pill** (coalescence of comparable spheres), ``V_pill - (Vi+Vj)``:
        only a true hull of both spheres when the radii are similar -- then the
        frustum degenerates into a cylinder and the pill becomes a capsule. For
        equal spheres this gives ΔV = (2/3)πr³ > 0.

        For unequal radii the straight frustum wall cuts *into* the larger
        sphere, so the term goes negative: it is positive only above a radius
        ratio of ``(3-√5)/2 ≈ 0.381966``. At r_j/r_i = 0.2 the pill (1.573e-10 m³
        for r_i = 350 µm) is smaller than the large sphere alone (1.796e-10 m³)
        -- it does not even contain it. A negative ΔV is a geometric artefact of
        the substitute shape, not a physical statement.

        **Layering** takes over there, see :meth:`_delta_volume_layering`.

        Why ``max`` and not "switch when ΔV < 0": the two curves intersect at
        r_j/r_i = 0.403032, so the maximum is *continuous* at the handover. A
        switch at the sign change (0.381966) would instead jump from 0 to
        +0.0186·π·r_i³.

        Args:
            Vi: Volume of sphere i [m³]
            Vj: Volume of sphere j [m³]

        Returns:
            ΔV: Additional volume from contact geometry [m³], always ≥ 0
        """
        ri = self._radius_from_volume(Vi)
        rj = self._radius_from_volume(Vj)

        V_pill = self._cone_pill_volume(ri, rj)
        V_old = Vi + Vj
        dV_cone = V_pill - V_old

        dV_layering = self._delta_volume_layering(rj if rj < ri else ri)

        return dV_cone if dV_cone > dV_layering else dV_layering
    
    # =====================================================================
    # Agglomeration: 2 parents → 1 child
    # =====================================================================
    
    def compute_merged_porosity(self,
                                 v_dry1: float, poro1: float,
                                 v_dry2: float, poro2: float,
                                 v_liq1: float = None,
                                 v_liq2: float = None,
                                 sat1: float = None,
                                 sat2: float = None,
                                 solver = None
                                 ) -> tuple[float, float]:
        """
        Compute merged dry volume and porosity after agglomeration.
        
        CRITICAL: Mass conservation via V_solid!
        
        Algorithm:
            1. Decompose parents: V_dry → V_solid + V_pore
            2. Merge solid: V_solid_new = V_solid_1 + V_solid_2 (CONSERVED!)
            3. Merge old pores: V_pore_old = V_pore_1 + V_pore_2
            4. Calculate geometric ΔV from cone-pill model
            5. Add new pores: V_pore_new = V_pore_old + k_agg × ΔV
            6. Recombine: V_dry_new = V_solid_new + V_pore_new
            7. Compute porosity: ε_new = V_pore_new / V_dry_new
        
        Args:
            v_dry1: Dry volume of particle 1 [m³] (= V_solid + V_pore)
            poro1: Porosity of particle 1 (NaN for legacy Vollkörper, 0.0 for poreless)
            v_dry2: Dry volume of particle 2 [m³]
            poro2: Porosity of particle 2 (NaN for legacy Vollkörper, 0.0 for poreless)
            v_liq1, v_liq2: Liquid volumes (not used in this model)
            sat1, sat2: Saturations (not used in this model)
            solver: Not used
        
        Returns:
            Tuple of (v_dry_merged, poro_merged)
            - v_dry_merged: New dry volume [m³]
            - poro_merged: New porosity [dimensionless]
            
        Note:
            - V_solid is ALWAYS conserved (mass conservation!)
            - All volumes are per PHYSICAL particle (intensive)
            - Poreless particles (poro=0.0) gain porosity through ΔV on first contact!
            - Legacy Vollkörper (NaN) are treated as poreless (0.0), the modern
              convention -- they do NOT stay NaN. Keeping NaN here was never
              actually possible: the clamp turned it into 1.0 ("pure void") and
              destroyed the particle's solid volume.
        """
        # ==========================================
        # Step 1: Validate inputs
        # ==========================================
        # GUARD: Check for inf/nan in volumes - this indicates a bug upstream!
        if np.isinf(v_dry1) or np.isnan(v_dry1):
            raise RuntimeError(
                f"Invalid v_dry1={v_dry1:.6e} passed to compute_merged_porosity! "
                f"This indicates V_flat[-1] was corrupted in a previous step."
            )
        if np.isinf(v_dry2) or np.isnan(v_dry2):
            raise RuntimeError(
                f"Invalid v_dry2={v_dry2:.6e} passed to compute_merged_porosity! "
                f"This indicates V_flat[-1] was corrupted in a previous step."
            )
        
        # Legacy "Vollkoerper" sentinel (NaN) means NO pores -> 0.0, the modern
        # convention used everywhere else (cf. mcpbe_break.py, which maps NaN
        # fragment porosities to 0.0 as well).
        #
        # This MUST happen before the clamp: `min(1.0, nan)` returns 1.0 in
        # Python, so `max(0.0, min(1.0, nan))` silently turned a pore-free body
        # into "pure void" -- V_solid = V_dry * (1 - 1.0) = 0, i.e. the whole
        # mass of that particle vanished.
        if np.isnan(poro1):
            poro1 = 0.0
        if np.isnan(poro2):
            poro2 = 0.0

        # Clamp porosity to valid range [0, 1]
        poro1 = max(0.0, min(1.0, poro1))
        poro2 = max(0.0, min(1.0, poro2))
        
        # GUARD: Check for inf/nan after decomposition
        V_solid_1 = v_dry1 * (1.0 - poro1)
        V_pore_1 = v_dry1 * poro1
        V_solid_2 = v_dry2 * (1.0 - poro2)
        V_pore_2 = v_dry2 * poro2
        
        if np.isinf(V_solid_1) or np.isnan(V_solid_1):
            raise RuntimeError(
                f"V_solid_1={V_solid_1:.6e} invalid! v_dry1={v_dry1:.6e}, poro1={poro1:.4f}"
            )
        if np.isinf(V_solid_2) or np.isnan(V_solid_2):
            raise RuntimeError(
                f"V_solid_2={V_solid_2:.6e} invalid! v_dry2={v_dry2:.6e}, poro2={poro2:.4f}"
            )
        
        # ==========================================
        # Step 2: Decompose into V_solid and V_pore
        # ==========================================
        # IMPORTANT: ε = V_pore / V_dry  →  V_pore = ε × V_dry
        #            V_solid = V_dry × (1 - ε)
        # 
        # This works for BOTH poro=0.0 (non-porous) AND poro>0.0 (porous)!
        # For poro=0.0: V_solid = V_dry × 1.0 = V_dry ✓, V_pore = 0.0 ✓
        
        # ==========================================
        # Step 3: Merge with MASS CONSERVATION
        # ==========================================
        # CRITICAL: V_solid is CONSERVED (this is the mass!)
        V_solid_new = V_solid_1 + V_solid_2
        
        # Old pore volume (additive from parents)
        V_pore_old = V_pore_1 + V_pore_2
        
        # GUARD: Check intermediate values
        if np.isinf(V_solid_new) or np.isnan(V_solid_new):
            raise RuntimeError(
                f"V_solid_new={V_solid_new:.6e} invalid! "
                f"V_solid_1={V_solid_1:.6e}, V_solid_2={V_solid_2:.6e}"
            )
        if np.isinf(V_pore_old) or np.isnan(V_pore_old):
            raise RuntimeError(
                f"V_pore_old={V_pore_old:.6e} invalid! "
                f"V_pore_1={V_pore_1:.6e}, V_pore_2={V_pore_2:.6e}"
            )
        
        # ==========================================
        # Step 4: Calculate geometric ΔV
        # ==========================================
        # Use V_dry for radius calculation (represents outer geometry)
        ΔV = self._delta_volume_pair(v_dry1, v_dry2)
        
        # GUARD: Check ΔV
        if np.isinf(ΔV) or np.isnan(ΔV):
            raise RuntimeError(
                f"ΔV={ΔV:.6e} invalid! v_dry1={v_dry1:.6e}, v_dry2={v_dry2:.6e}"
            )
        
        # ==========================================
        # Step 5: Add new pores from contact geometry
        # ==========================================
        # V_pore_new = V_pore_old + k_agg × ΔV
        # Note: k_agg=0.5 (default) dampens pore growth to prevent runaway
        delta_pore = self.k_agg * ΔV
        
        # GUARD: Prevent excessive pore addition
        if np.isinf(delta_pore) or np.isnan(delta_pore):
            raise RuntimeError(
                f"k_agg*ΔV={delta_pore:.6e} invalid! k_agg={self.k_agg}, ΔV={ΔV:.6e}"
            )
        
        V_pore_new = V_pore_old + delta_pore
        
        # GUARD: Check V_pore_new before clamping
        if np.isinf(V_pore_new) or np.isnan(V_pore_new):
            raise RuntimeError(
                f"V_pore_new={V_pore_new:.6e} invalid! "
                f"V_pore_old={V_pore_old:.6e}, k_agg*ΔV={delta_pore:.6e}"
            )
        
        # Ensure non-negative (numerical safety)
        V_pore_new = max(0.0, V_pore_new)
        
        # ==========================================
        # Step 6: Cap the porosity via V_pore, then recombine to V_dry
        # ==========================================
        # V_solid is THE conserved quantity. When the porosity would run past
        # PORO_MAX, cap V_pore and derive V_dry and eps from the capped value --
        # do NOT clamp eps on its own afterwards.
        #
        # Clamping eps alone (the pre-2026-08 form) left V_dry at its uncapped
        # value, so `V_dry * (1 - eps)` no longer matched the V_solid stored in
        # `V_flat[0]` by the caller: at a true eps of 0.99998 the derived solid
        # volume came out 5x too large. `mcpbe_continuous_processes._apply_
        # compression` then re-derives V_solid exactly that way and writes V_dry
        # back from it, which cemented the phantom mass permanently.
        #
        # Capping V_pore keeps `V_dry * (1 - eps) == V_solid_new` exact instead:
        # V_pore = V_solid * eps/(1-eps)  =>  V_dry = V_solid / (1-eps).
        if V_solid_new > 0.0:
            V_pore_cap = V_solid_new * (self.PORO_MAX / (1.0 - self.PORO_MAX))
            if V_pore_new > V_pore_cap:
                V_pore_new = V_pore_cap
            V_dry_new = V_solid_new + V_pore_new
            poro_new = V_pore_new / V_dry_new if V_dry_new > 0.0 else 0.0
        else:
            # No solid at all -- nothing to conserve, nothing to cap against.
            V_dry_new = V_pore_new
            poro_new = 0.0

        # GUARD: NaN check - if NaN occurs here, it's a bug!
        if np.isnan(poro_new):
            raise RuntimeError(
                f"NaN porosity generated in compute_merged_porosity! "
                f"V_pore_new={V_pore_new:.6e}, V_dry_new={V_dry_new:.6e}"
            )

        # Lower bound only; the upper one is already enforced via V_pore above.
        if poro_new < 0.0:
            poro_new = 0.0

        return V_dry_new, poro_new
    
    # =====================================================================
    # Breakage: 1 parent → n children
    # =====================================================================
    
    def compute_fragment_porosity(self,
                                   parent_porosity: float,
                                   fragment_volume: Union[float, List[float], np.ndarray],
                                   parent_volume: float,
                                   solver = None
                                   ) -> Union[float, List[float]]:
        """
        Compute porosity for fragments after breakage.
        
        CRITICAL: Mass conservation via V_solid!
        
        Physical Model:
            When a parent breaks into n fragments, (n-1) cone-pill geometries
            form between adjacent fragments. This represents NEW SURFACE AREA,
            which REDUCES accessible pore volume (pore collapse at fractures).
        
        Algorithm:
            1. Decompose parent: V_dry → V_solid + V_pore
            2. Distribute solid: V_solid_k = fragment_volume[k] (CONSERVED!)
            3. Calculate total ΔV from (n-1) cone-pill pairs
            4. Reduce pores: V_pore_new = V_pore_parent - k_break × ΣΔV
            5. Distribute pores proportionally: V_pore_k ∝ V_solid_k
            6. Compute fragment porosities: ε_k = V_pore_k / V_dry_k
        
        Args:
            parent_porosity: Porosity of parent particle (NaN for Vollkörper)
            fragment_volumes: List/array of fragment dry volumes [m³]
                             OR single fragment volume for backward compatibility
            parent_volume: Dry volume of parent particle [m³]
            solver: Not used
        
        Returns:
            Single porosity value (if fragment_volume is scalar)
            OR list of porosities (if fragment_volume is list/array)
            
        Note:
            - V_solid is ALWAYS conserved across all fragments
            - All volumes are per PHYSICAL particle (intensive)
            - Pore volume can only decrease: ΣΔV ≥ 0 now holds for every size
              ratio, because ΔV falls back to the layering term instead of
              going negative (see :meth:`_delta_volume_pair`). Before that,
              strongly uneven fragments produced ΔV < 0 and breakage *created*
              pore volume -- the opposite of the model's intent.
            - Clamped to prevent negative pores
        """
        # ==========================================
        # Backward compatibility: scalar input
        # ==========================================
        if isinstance(fragment_volume, (int, float)):
            result = self._compute_fragment_porosity_multi(
                parent_porosity, [fragment_volume], parent_volume, solver
            )
            return result[0] if result else parent_porosity
        
        # Convert array-like to list
        if isinstance(fragment_volume, np.ndarray):
            fragment_volume = fragment_volume.tolist()
        
        return self._compute_fragment_porosity_multi(
            parent_porosity, fragment_volume, parent_volume, solver
        )
    
    def _compute_fragment_porosity_multi(self,
                                          parent_porosity: float,
                                          fragment_volumes: List[float],
                                          parent_volume: float,
                                          solver = None
                                          ) -> List[float]:
        """
        Internal method for multi-fragment breakage.
        
        CRITICAL: fragment_volumes are interpreted as V_DRY (total volume)!
        This matches the breakage code which conserves total volume.
        We extract V_solid using parent porosity scaling, then calculate
        new pore distribution after ΔV loss from cone geometry.
        
        Algorithm:
            1. fragment_volumes = V_dry_frag (from breakage, sum ≈ parent V_dry)
            2. V_solid_frag = V_dry_frag × (1 - ε_parent) [initial estimate]
            3. Calculate ΔV from cone geometry using V_dry_frag
            4. V_pore_new = V_pore_parent - k_break × ΔV
            5. Distribute V_pore_new ∝ V_solid_frag
            6. V_dry_frag_new = V_solid_frag + V_pore_frag_new [EXACT!]
            7. ε_frag_new = V_pore_frag_new / V_dry_frag_new
        
        Mass conservation: Σ V_solid_frag = V_solid_parent (by construction)
        
        See compute_fragment_porosity for documentation.
        """
        n = len(fragment_volumes)
        
        # ==========================================
        # Guard: Invalid cases
        # ==========================================
        if n == 0:
            return []
        
        if n == 1:
            # No actual breakage → return parent porosity
            return [parent_porosity]
        
        if parent_volume <= 0:
            # Invalid parent → return zero porosity
            return [0.0] * n
        
        # ==========================================
        # Step 1: Validate and clamp parent porosity
        # ==========================================
        # Legacy "Vollkoerper" sentinel (NaN) means NO pores -> 0.0. Must happen
        # BEFORE the clamp: `min(1.0, nan)` is 1.0 in Python, which would turn a
        # pore-free parent into "pure void" and annihilate its solid volume.
        # (The NaN guard below used to sit *after* the clamp and could therefore
        # never fire.)
        if np.isnan(parent_porosity):
            parent_porosity = 0.0

        # Clamp to valid range [0, 1]
        parent_porosity = max(0.0, min(1.0, parent_porosity))
        
        # ==========================================
        # Step 2: Decompose parent into V_solid and V_pore
        # ==========================================
        # IMPORTANT: ε = V_pore / V_dry  →  V_pore = ε × V_dry
        #            V_solid = V_dry × (1 - ε)
        # NOTE: parent_volume is V_dry of parent
        # 
        # For poro=0.0: V_solid = parent_volume × 1.0 = parent_volume ✓
        #               V_pore = 0.0 ✓
        V_solid_parent = parent_volume * (1.0 - parent_porosity)
        V_pore_parent = parent_volume * parent_porosity
        
        # ==========================================
        # Step 3: Interpret fragment_volumes as V_dry (outer geometry)
        # ==========================================
        # IMPORTANT: fragment_volumes from breakage are TOTAL volumes (V_dry),
        # not V_solid! They sum to parent V_dry (conservation of total volume).
        # We need to extract V_solid using parent porosity assumption.
        #
        # Initial estimate: V_solid_frag ≈ V_dry_frag × (1 - ε_parent)
        # This assumes fragments initially inherit parent's solid fraction.
        # After ΔV correction, actual porosity will differ.
        
        V_dry_frag_sum = sum(fragment_volumes)  # Should ≈ parent_volume (V_dry)
        
        # Convert to V_solid using parent porosity scaling
        # V_solid_frag = V_dry_frag × (1 - ε_parent)
        fragment_solids = [
            V_dry_frag * (1.0 - parent_porosity)
            for V_dry_frag in fragment_volumes
        ]
        V_solid_frag_sum = sum(fragment_solids)  # Should ≈ V_solid_parent
        
        # ==========================================
        # Step 4: Use fragment V_dry for cone geometry calculation
        # ==========================================
        # fragment_volumes ARE V_dry (from breakage), so use them directly!
        frag_v_dry_initial = fragment_volumes
        
        # ==========================================
        # Step 5: Calculate total ΔV from (n-1) cone-pill pairs
        # ==========================================
        # Use fragment V_dry for cone geometry calculation
        # Model: Fragments form a linear chain with (n-1) contacts
        
        ΔV_total = 0.0
        for i in range(n - 1):
            Vi_dry = frag_v_dry_initial[i]
            Vj_dry = frag_v_dry_initial[i + 1]
            
            if Vi_dry > 0 and Vj_dry > 0:
                ΔV_pair = self._delta_volume_pair(Vi_dry, Vj_dry)
                ΔV_total += ΔV_pair
        
        # ==========================================
        # Step 6: Reduce pore volume
        # ==========================================
        # V_pore_new = V_pore_parent - k_break × ΔV_total
        V_pore_new = V_pore_parent - self.k_break * ΔV_total
        
        # Ensure non-negative (pores can't be negative!)
        V_pore_new = max(0.0, V_pore_new)
        
        # ==========================================
        # Step 7: Distribute pores proportionally to fragments
        # ==========================================
        # Strategy: V_pore_k ∝ V_solid_k (larger fragments get more pores)
        
        poro_list = []
        for V_solid_frag in fragment_solids:
            if V_solid_frag_sum > 0 and V_solid_frag > 0:
                # Proportional share of new pore volume
                V_pore_frag = V_pore_new * (V_solid_frag / V_solid_frag_sum)
                
                # V_dry_frag = V_solid_frag + V_pore_frag (EXACT!)
                V_dry_frag = V_solid_frag + V_pore_frag
                
                # ε_frag = V_pore_frag / V_dry_frag
                if V_dry_frag > 0:
                    poro_frag = V_pore_frag / V_dry_frag
                else:
                    poro_frag = 0.0
            else:
                poro_frag = 0.0
            
            # GUARD: NaN check
            if np.isnan(poro_frag):
                poro_frag = 0.0
            
            # Clamp to valid range [0, 1)
            # Safe to clamp eps directly here (unlike compute_merged_porosity):
            # the caller derives V_dry from V_solid and this eps
            # (mcpbe_break.py: V_dry_frag = V_solid_frag / (1 - frag_poro)), so
            # V_solid stays the anchor and no phantom mass can appear.
            poro_frag = max(0.0, min(self.PORO_MAX, poro_frag))
            poro_list.append(poro_frag)
        
        return poro_list
    
    # =====================================================================
    # Nucleation: Create first porous particles
    # =====================================================================
    
    def compute_nucleation_porosity(self,
                                     v_solid: float,
                                     v_liquid: float,
                                     nucleation_params: dict = None,
                                     solver = None
                                     ) -> tuple[float, float]:
        """
        Compute porosity for newly nucleated particle.
        
        For cone_model, nucleation starts with ZERO porosity (poro=0.0).
        Porosity is CREATED during agglomeration via ΔV from contact geometry.
        
        IMPORTANT: Unlike volume_mixing, this kernel does NOT hardcode porosity!
        It starts with poro=0.0 and porosity grows naturally through agglomeration
        (ΔV > 0 adds pore volume at contact points).
        
        CHANGE (vs. legacy): Returns poro=0.0 instead of NaN.
        - Legacy: Vollkörper marked with NaN (undefined porosity)
        - Modern: Poreless particle marked with 0.0 (defined, no pores)
        - Benefit: Unified handling, no special NaN checks needed
        
        Args:
            v_solid: Solid volume of nucleus [m³]
            v_liquid: Liquid volume associated with nucleus [m³]
            nucleation_params: Can override initial_porosity (optional)
                               - 'default_porosity': Initial porosity (default: 0.0)
            solver: Not used
        
        Returns:
            Tuple of (v_dry, porosity)
            - Returns (v_solid, 0.0) for poreless when default_porosity <= 0
            - Returns (v_dry, porosity) for porous particles when default_porosity > 0
        """
        # Priority: 1) nucleation_params, 2) self.params (config), 3) 0.0 (poreless!)
        default_porosity = 0.0  # DEFAULT: Start poreless (no pores)
        
        if nucleation_params and 'default_porosity' in nucleation_params:
            default_porosity = float(nucleation_params['default_porosity'])
        elif 'default_porosity' in self.params:
            default_porosity = float(self.params['default_porosity'])
        
        # If explicitly set to 0 or negative → poreless (poro=0.0)
        # MODERN: Use 0.0 instead of NaN for better compatibility!
        if default_porosity <= 0:
            return float(v_solid), 0.0  # Poreless ✅ (was: np.nan)
        
        # Clamp to valid range
        porosity = max(0.0, min(0.999, default_porosity))
        
        # Compute dry volume from solid volume and porosity
        # V_dry = V_solid / (1 - ε)
        v_dry = v_solid / (1.0 - porosity)
        
        return v_dry, porosity
