"""
Kernel Integration Module for WMCPBE Solver.

This module provides kernel initialization and delegation for the new
modular kernel architecture.

Usage:
    >>> from wmcpbe.kernel_integration import KernelManager
    >>> manager = KernelManager(
    ...     agg_kernel_name='shear_chin1998',
    ...     agg_kernel_params={'corr_beta': 1e-3, 'g': 1000},
    ... )
    >>> manager.initialize_kernels(solver)
"""

from __future__ import annotations

from typing import Optional, Any, Dict
import numpy as np


class KernelEvaluationError(RuntimeError):
    """
    Exception raised when a kernel evaluation fails.
    
    This error indicates that a physics kernel (aggregation, breakage, etc.)
    encountered invalid input or failed during computation. The simulation
    may continue with degraded physics (e.g., zero collision rate) but
    the user should investigate the root cause.
    
    Attributes:
        kernel_name: Name of the kernel that failed
        message: Detailed error description
        original_exception: The underlying exception (if any)
    
    Example:
        >>> try:
        ...     beta = kernel.compute_beta(r1=-1e-6, r2=2e-6)
        ... except KernelEvaluationError as e:
        ...     logger.warning(f"Kernel {e.kernel_name} failed: {e}")
    """
    
    def __init__(self, kernel_name: str, message: str, original_exception: Optional[Exception] = None):
        self.kernel_name = kernel_name
        self.original_exception = original_exception
        super().__init__(f"{kernel_name}: {message}")


class KernelManager:
    """
    Manages kernel instances for WMCPBE solver.
    
    Handles:
    - Kernel instantiation from configuration
    - Backward compatibility with COLEVAL/BREAKRVAL
    - Delegation of physics calculations to kernels
    
    Attributes:
        agg_kernel: Aggregation kernel instance
        break_kernel: Breakage kernel instance
        porosity_growth_kernel: Porosity growth kernel instance
        liquid_dist_kernel: Liquid distribution kernel instance
        agglomeration_acceptance_kernel: Agglomeration acceptance kernel instance
        porosity_compression_kernel: Porosity compression kernel (continuous process)
        liquid_internalization_kernel: Liquid internalization kernel (continuous process)
        liq_internalisation_agglomeration_kernel: Liquid internalization during agglomeration (event-based)
    """
    
    def __init__(
        self,
        # Aggregation kernel config
        agg_kernel_name: Optional[str] = None,
        agg_kernel_params: Optional[Dict[str, Any]] = None,
        # Breakage kernel config
        break_kernel_name: Optional[str] = None,
        break_kernel_params: Optional[Dict[str, Any]] = None,
        # Porosity growth kernel config
        porosity_growth_kernel_name: Optional[str] = None,
        porosity_growth_kernel_params: Optional[Dict[str, Any]] = None,
        # Liquid distribution kernel config
        liquid_dist_kernel_name: Optional[str] = None,
        liquid_dist_kernel_params: Optional[Dict[str, Any]] = None,
        # Agglomeration acceptance kernel config (NEW)
        agg_acceptance_kernel_name: Optional[str] = None,
        agg_acceptance_kernel_params: Optional[Dict[str, Any]] = None,
        # Continuous processes kernels (NEW)
        porosity_compression_kernel_name: Optional[str] = None,
        porosity_compression_kernel_params: Optional[Dict[str, Any]] = None,
        liquid_internalization_kernel_name: Optional[str] = None,
        liquid_internalization_kernel_params: Optional[Dict[str, Any]] = None,
        # Liquid internalization during agglomeration kernel (event-based, NEW)
        liq_internalisation_agglomeration_kernel_name: Optional[str] = None,
        liq_internalisation_agglomeration_kernel_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize kernel manager with configuration.
        
        Args:
            agg_kernel_name: Name of aggregation kernel (e.g., 'shear_chin1998')
            agg_kernel_params: Parameters for aggregation kernel
            break_kernel_name: Name of breakage kernel (e.g., 'power_law')
            break_kernel_params: Parameters for breakage kernel
            porosity_growth_kernel_name: Name of porosity growth kernel
            porosity_growth_kernel_params: Parameters for porosity growth kernel
            compression_kernel_name: Name of compression kernel
            compression_kernel_params: Parameters for compression kernel
            liquid_dist_kernel_name: Name of liquid distribution kernel
            liquid_dist_kernel_params: Parameters for liquid distribution kernel
        """
        # Store configuration
        self.agg_kernel_name = agg_kernel_name
        self.agg_kernel_params = agg_kernel_params or {}
        self.break_kernel_name = break_kernel_name
        self.break_kernel_params = break_kernel_params or {}
        self.porosity_growth_kernel_name = porosity_growth_kernel_name
        self.porosity_growth_kernel_params = porosity_growth_kernel_params or {}
        self.liquid_dist_kernel_name = liquid_dist_kernel_name
        self.liquid_dist_kernel_params = liquid_dist_kernel_params or {}
        # Agglomeration acceptance kernel config (NEW)
        self.agg_acceptance_kernel_name = agg_acceptance_kernel_name
        self.agg_acceptance_kernel_params = agg_acceptance_kernel_params or {}
        # Continuous processes kernels (NEW)
        self.porosity_compression_kernel_name = porosity_compression_kernel_name
        self.porosity_compression_kernel_params = porosity_compression_kernel_params or {}
        self.liquid_internalization_kernel_name = liquid_internalization_kernel_name
        self.liquid_internalization_kernel_params = liquid_internalization_kernel_params or {}
        # Liquid internalization during agglomeration kernel (event-based, NEW)
        self.liq_internalisation_agglomeration_kernel_name = liq_internalisation_agglomeration_kernel_name
        self.liq_internalisation_agglomeration_kernel_params = liq_internalisation_agglomeration_kernel_params or {}
        
        # Kernel instances (initialized later)
        self.agg_kernel = None
        self.break_kernel = None
        self.porosity_growth_kernel = None
        self.liquid_dist_kernel = None
        self.agglomeration_acceptance_kernel = None
        # Continuous processes kernels (NEW)
        self.porosity_compression_kernel = None
        self.liquid_internalization_kernel = None
        # Liquid internalization during agglomeration kernel (event-based, NEW)
        self.liq_internalisation_agglomeration_kernel = None
    
    def initialize_kernels(self, solver) -> None:
        """
        Initialize all kernels based on configuration.
        
        Priority:
        1. Explicit kernel name (e.g., agg_kernel_name='eke_darelius2005')
        2. Legacy COLEVAL/BREAKRVAL mapping
        3. Default kernels
        
        Args:
            solver: Reference to MCPBEBase solver instance
        """
        # Use absolute imports for compatibility with direct execution
        try:
            from wmcpbe.kernels.aggregation import get_aggregation_kernel
            from wmcpbe.kernels.breakage import get_breakage_kernel
            from wmcpbe.kernels.porosity_growth import get_porosity_growth_kernel
            from wmcpbe.kernels.liquid_distribution import get_liquid_distribution_kernel
        except ImportError:
            # Fallback to relative imports (package context)
            from .kernels.aggregation import get_aggregation_kernel
            from .kernels.breakage import get_breakage_kernel
            from .kernels.porosity_growth import get_porosity_growth_kernel
            from .kernels.liquid_distribution import get_liquid_distribution_kernel
        
        # ==========================================
        # Aggregation Kernel
        # ==========================================
        if self.agg_kernel_name is not None:
            # Explicit kernel selection - validate and instantiate
            self.agg_kernel = get_aggregation_kernel(
                self.agg_kernel_name,
                **self.agg_kernel_params
            )
            # Set solver attributes for backward compatibility
            # (legacy code reads CORR_BETA, G from solver, not kernel)
            _agg_defaults = self.agg_kernel.get_default_params()
            if 'corr_beta' in self.agg_kernel_params:
                solver.CORR_BETA = self.agg_kernel_params['corr_beta']
            # The aggregation kernel owns solver.G (the process shear rate).
            # Always set it -- falling back to the kernel default keeps
            # solver.G defined and consistent with what the kernel computes.
            _agg_g = self.agg_kernel_params.get('g', _agg_defaults.get('g'))
            if _agg_g is not None:
                solver.G = float(_agg_g)
        else:
            # No aggregation kernel selected - raise error (aggregation is essential!)
            from .kernels.aggregation import list_aggregation_kernels
            available = list_aggregation_kernels()
            raise ValueError(
                "No aggregation kernel specified. "
                "Aggregation kernel is required for simulation. "
                f"Available kernels: {available}"
            )
        
        # ==========================================
        # Breakage Kernel
        # ==========================================
        if self.break_kernel_name is not None:
            # Explicit kernel selection - validate and instantiate
            self.break_kernel = get_breakage_kernel(
                self.break_kernel_name,
                **self.break_kernel_params
            )
            # Set solver attributes for backward compatibility.
            # Fall back to the kernel's own defaults, not to literals, so the
            # solver mirror cannot drift away from what the kernel computes.
            _bk_defaults = self.break_kernel.get_default_params()
            solver.pl_P1 = float(self.break_kernel_params.get(
                'p1', _bk_defaults.get('p1', 3e-2)))
            solver.pl_P2 = float(self.break_kernel_params.get(
                'p2', _bk_defaults.get('p2', 1.0)))
            # NOTE: `solver.G` is NOT written here. G is a property of the
            # process (the mixer), not of the breakage model, and it is read by
            # other physics too (e.g. mcpbe_nucleation). Letting the breakage
            # kernel overwrite it meant the nucleation collision velocity
            # silently followed the breakage kernel's `g`. The aggregation
            # kernel owns `solver.G`; a disagreeing breakage `g` is surfaced as
            # a warning below instead of being applied behind the scenes.
            _break_g = self.break_kernel_params.get(
                'g', _bk_defaults.get('g'))
            _solver_g = getattr(solver, 'G', None)
            if (_break_g is not None and _solver_g is not None
                    and abs(float(_break_g) - float(_solver_g)) > 1e-12):
                import warnings
                warnings.warn(
                    f"Breakage kernel shear rate g={float(_break_g)} differs "
                    f"from the aggregation/solver shear rate G={float(_solver_g)}. "
                    f"Both describe the same mixer. The breakage rate uses its "
                    f"own g; solver.G (used e.g. by nucleation) is left at the "
                    f"aggregation value. Set both explicitly if they should match.",
                    UserWarning,
                    stacklevel=2
                )
            # NOTE: `solver.pl_v` / `solver.pl_q` are not written from the
            # kernel params either -- they belong to the breakage FUNCTION
            # (fragment size distribution) and the rate kernels no longer
            # accept them at all. To steer the breakage function, set
            # `solver.break_frag_v` / `break_frag_q` explicitly.
        else:
            # No breakage kernel selected - disable breakage
            # IMPORTANT: Keep pl_v, pl_q at valid defaults for _compute_frag_num()!
            # pl_P1=0 ensures zero breakage rate even if kernel is accidentally called
            self.break_kernel = None
            solver.pl_P1 = 0.0  # Zero rate
            solver.pl_P2 = 0.0
            solver.pl_v = 2.0   # Default value (required by _compute_frag_num)
            solver.pl_q = 1.0   # Default value (required by _compute_frag_num)
            # solver.G is NOT reset to 0.0 here: it is the process shear rate
            # owned by the aggregation kernel and read by other physics
            # (nucleation). Disabling breakage must not silently zero it.
        
        # ==========================================
        # Porosity Growth Kernel
        # ==========================================
        if self.porosity_growth_kernel_name is not None:
            # Explicit kernel selection
            self.porosity_growth_kernel = get_porosity_growth_kernel(
                self.porosity_growth_kernel_name,
                **self.porosity_growth_kernel_params
            )
        else:
            # Optional kernel - set to None if not specified
            self.porosity_growth_kernel = None
        
        # ==========================================
        # Compression Kernel (REMOVED)
        # ==========================================
        # Legacy compression_kernel_name parameter is deprecated.
        #
        # This used to read `hasattr(self, 'compression_kernel_name')`, which is
        # ALWAYS False -- the attribute is never set in __init__ and is not an
        # __init__ parameter either. The migration hint below could therefore
        # never reach anyone. It is now read off the solver, which is where a
        # config file would put it (config files assign arbitrary attributes,
        # see base_solver._load_attributes), so an old config actually gets the
        # explanation instead of running with compression silently disabled.
        # See mcpbe/docs/historical/Audit_2026-08-17.md, B-14.
        _legacy_compression = getattr(solver, 'compression_kernel_name', None)
        if _legacy_compression is not None:
            import warnings
            warnings.warn(
                "The 'compression_kernel_name' parameter and kernels/compression/ module are REMOVED. "
                "Use ContinuousProcessesHandler with porosity_compression kernel instead.\n"
                "\nMigration guide:\n"
                "  OLD: solver = MCPBESolver(compression_kernel_name='exponential_decay', ...)\n"
                "  NEW: solver.create_continuous_processes_handler(\n"
                "           enabled=True,\n"
                "           compression_rate=0.02,\n"
                "           min_porosity=0.3,\n"
                "       )\n"
                "\nSee docs: wmcpbe.kernels.continuous_processes",
                DeprecationWarning,
                stacklevel=2
            )
        
        # ==========================================
        # Liquid Distribution Kernel
        # ==========================================
        if self.liquid_dist_kernel_name is not None:
            # Explicit kernel selection
            self.liquid_dist_kernel = get_liquid_distribution_kernel(
                self.liquid_dist_kernel_name,
                **self.liquid_dist_kernel_params
            )
        else:
            # Optional kernel - set to None if not specified
            self.liquid_dist_kernel = None
        
        # ==========================================
        # Agglomeration Acceptance Kernel (NEW)
        # ==========================================
        if self.agg_acceptance_kernel_name is not None:
            # Explicit kernel selection
            try:
                from wmcpbe.kernels.agglomeration_acceptance import get_agglomeration_acceptance_kernel
            except ImportError:
                from .kernels.agglomeration_acceptance import get_agglomeration_acceptance_kernel
            self.agglomeration_acceptance_kernel = get_agglomeration_acceptance_kernel(
                self.agg_acceptance_kernel_name,
                **self.agg_acceptance_kernel_params
            )
        else:
            # Optional kernel - set to None if not specified (always accept)
            self.agglomeration_acceptance_kernel = None
        
        # ==========================================
        # Continuous Processes Kernels (NEW)
        # ==========================================
        # These replace the legacy compression kernel with a more flexible framework
        
        # Porosity Compression Kernel
        # Note: Legacy compression_kernel_name is deprecated - use continuous_processes instead
        # The configured name selects the kernel (was hard-wired to
        # 'porosity_compression' before the mixer-speed variants existed);
        # get_continuous_kernel rejects an unknown name loudly.
        if self.porosity_compression_kernel_name is not None:
            try:
                from wmcpbe.kernels.continuous_processes import get_continuous_kernel
            except ImportError:
                from .kernels.continuous_processes import get_continuous_kernel
            self.porosity_compression_kernel = get_continuous_kernel(
                self.porosity_compression_kernel_name,
                **self.porosity_compression_kernel_params
            )
        else:
            self.porosity_compression_kernel = None
        
        # Liquid Internalization Kernel
        if self.liquid_internalization_kernel_name is not None:
            try:
                from wmcpbe.kernels.continuous_processes import get_continuous_kernel
            except ImportError:
                from .kernels.continuous_processes import get_continuous_kernel
            self.liquid_internalization_kernel = get_continuous_kernel(
                'liquid_internalization',
                **self.liquid_internalization_kernel_params
            )
        else:
            self.liquid_internalization_kernel = None
        
        # Liquid Internalization during Agglomeration Kernel (event-based)
        if hasattr(self, 'liq_internalisation_agglomeration_kernel_name') and \
           self.liq_internalisation_agglomeration_kernel_name is not None:
            try:
                from wmcpbe.kernels.continuous_processes import get_continuous_kernel
            except ImportError:
                from .kernels.continuous_processes import get_continuous_kernel
            self.liq_internalisation_agglomeration_kernel = get_continuous_kernel(
                'liq_internalisation_agglomeration',
                **self.liq_internalisation_agglomeration_kernel_params
            )
        else:
            # Optional kernel - None means no internalization during agglomeration (fallback)
            self.liq_internalisation_agglomeration_kernel = None

        # ==========================================
        # Cross-kernel consistency: mixer speed
        # ==========================================
        # Every mixer-speed-driven kernel describes the SAME machine, so they
        # must agree on how fast it turns. Unlike the shear-rate check further
        # up (which only warns, because there `solver.G` has a documented
        # owner and precedence), two different `n_mixer` values have no
        # sensible interpretation at all: the run's collision frequency,
        # collision severity and breakage rate would belong to three different
        # mixers, and nothing in the results would reveal it. Hence a hard
        # error, raised once here where the whole kernel set is visible.
        try:
            from wmcpbe.kernels.mixer_speed import assert_consistent_mixer_speed
        except ImportError:
            from .kernels.mixer_speed import assert_consistent_mixer_speed

        assert_consistent_mixer_speed([
            ('aggregation', self.agg_kernel),
            ('breakage', self.break_kernel),
            ('agglomeration_acceptance', self.agglomeration_acceptance_kernel),
            ('porosity_growth', self.porosity_growth_kernel),
            ('porosity_compression', self.porosity_compression_kernel),
            ('liquid_internalization', self.liquid_internalization_kernel),
            ('liquid_distribution', self.liquid_dist_kernel),
        ])

    # ==========================================
    # Delegation methods
    # ==========================================
    
    def compute_beta(
        self,
        r1: float,
        r2: float,
        particle1_idx: Optional[int] = None,
        particle2_idx: Optional[int] = None,
        solver = None
    ) -> float:
        """Delegate beta calculation to aggregation kernel."""
        if self.agg_kernel is None:
            raise RuntimeError("Aggregation kernel not initialized")
        return self.agg_kernel.compute_beta(
            r1, r2,
            particle1_idx=particle1_idx,
            particle2_idx=particle2_idx,
            solver=solver
        )
    
    def compute_break_rate(
        self,
        v_particle: float,
        particle_idx: Optional[int] = None,
        solver = None
    ) -> float:
        """Delegate breakage rate calculation to breakage kernel."""
        if self.break_kernel is None:
            raise RuntimeError("Breakage kernel not initialized")
        return self.break_kernel.compute_rate(
            v_particle,
            particle_idx=particle_idx,
            solver=solver
        )
    
    def compute_merged_porosity(
        self,
        v_dry1: float,
        poro1: float,
        v_dry2: float,
        poro2: float,
        v_liq1: Optional[float] = None,
        v_liq2: Optional[float] = None,
        sat1: Optional[float] = None,
        sat2: Optional[float] = None,
        solver = None
    ) -> tuple[float, float]:
        """Delegate porosity mixing to porosity growth kernel."""
        if self.porosity_growth_kernel is None:
            raise RuntimeError("Porosity growth kernel not initialized")
        return self.porosity_growth_kernel.compute_merged_porosity(
            v_dry1, poro1, v_dry2, poro2,
            v_liq1=v_liq1, v_liq2=v_liq2,
            sat1=sat1, sat2=sat2,
            solver=solver
        )
    
    def compute_liquid_internalization(
        self,
        saturation: float,
        v_pore: float,
        l_total: float,
        dt: float,
        solver = None
    ) -> float:
        """Delegate liquid internalization to continuous process kernel."""
        if self.liquid_internalization_kernel is None:
            raise RuntimeError("Liquid internalization kernel not initialized")
        return self.liquid_internalization_kernel.compute(
            saturation, v_pore, l_total, dt,
            solver=solver
        )
    
    def compute_liquid_internalization_agglomeration(
        self,
        v_dry1: float, v_dry2: float,
        v_liq_ext1: float, v_liq_ext2: float,
        particle1_idx: Optional[int] = None,
        particle2_idx: Optional[int] = None,
        solver = None
    ) -> float:
        """
        Delegate liquid internalization during agglomeration to kernel.
        
        FALLBACK: If kernel is not configured, returns 0.0 (no internalization).
        This ensures backward compatibility with simulations that don't use
        this feature.
        
        Args:
            v_dry1, v_dry2: Dry volumes of parent particles [m³]
            v_liq_ext1, v_liq_ext2: External liquid volumes [m³]
            particle1_idx, particle2_idx: Particle indices (optional)
            solver: Solver reference (optional)
        
        Returns:
            l_e_to_i: Internalized liquid volume [m³], or 0.0 if kernel not configured
        """
        # FALLBACK: No kernel configured → no internalization
        if self.liq_internalisation_agglomeration_kernel is None:
            return 0.0
        
        return self.liq_internalisation_agglomeration_kernel.compute_internalization(
            v_dry1, v_dry2, v_liq_ext1, v_liq_ext2,
            particle1_idx=particle1_idx,
            particle2_idx=particle2_idx,
            solver=solver
        )

    def compute_liquid_externalization_breakage(
        self,
        v_pore_parent: float,
        v_pore_fragments_total: float,
        saturation_parent: float,
        particle_idx: Optional[int] = None,
        solver = None
    ) -> float:
        """
        Delegate liquid externalization during breakage to kernel.

        Reverse process of compute_liquid_internalization_agglomeration,
        served by the SAME kernel instance/slot (analogous to how
        porosity_growth_kernel handles both merged and fragment porosity).

        FALLBACK: If kernel is not configured, returns 0.0 (no
        externalization). This ensures backward compatibility with
        simulations that don't use this feature.

        Args:
            v_pore_parent: Pore volume of the parent particle before
                breakage [m³]
            v_pore_fragments_total: Summed pore volume of all fragments
                after breakage [m³]
            saturation_parent: Saturation of the parent particle before
                breakage
            particle_idx: Parent particle index (optional)
            solver: Solver reference (optional)

        Returns:
            V_liq_int_to_ext: Externalized liquid volume [m³], or 0.0 if
                               kernel not configured
        """
        # FALLBACK: No kernel configured → no externalization
        if self.liq_internalisation_agglomeration_kernel is None:
            return 0.0

        return self.liq_internalisation_agglomeration_kernel.compute_externalization(
            v_pore_parent, v_pore_fragments_total, saturation_parent,
            particle_idx=particle_idx,
            solver=solver
        )

    def select_liquid_target(
        self,
        solver,
        v_droplet: float,
        current_time: Optional[float] = None
    ) -> int:
        """Delegate particle selection to liquid distribution kernel."""
        if self.liquid_dist_kernel is None:
            raise RuntimeError("Liquid distribution kernel not initialized")
        return self.liquid_dist_kernel.select_target_particle(
            solver, v_droplet, current_time=current_time
        )
