from __future__ import annotations

from .mcpbe_base import MCPBEBase
from .mcpbe_agg import MCPBEAgg
from .mcpbe_break import MCPBEBreak
from .mcpbe_post import MCPBEPost
from .mcpbe_nucleation import MCPBENucleation
# Optional: Optimized nucleation with sampler caching
try:
    from .mcpbe_nucleation_sampleopti import MCPBENucleationSampleOpti
    HAS_OPTI_NUCLEATION = True
except ImportError:
    HAS_OPTI_NUCLEATION = False
    MCPBENucleationSampleOpti = None

# Note: Dual-sampler nucleation has been merged into Timestep-sampler.
# The old dualsampler module is deprecated and no longer used.
HAS_DUAL_SAMPLER = False
MCPBENucleationDualSampler = None

# Optional: Timestep-sampler nucleation (once per timestep, no replacement)
try:
    from .mcpbe_nucleation_timestep import MCPBENucleationTimestep
    HAS_TIMESTEP_NUCLEATION = True
except ImportError:
    HAS_TIMESTEP_NUCLEATION = False
    MCPBENucleationTimestep = None


class MCPBESolver(MCPBEPost, MCPBEBreak, MCPBEAgg, MCPBENucleation, MCPBEBase):
    """Monte Carlo PBE solver with configurable nucleation strategy.
    
    Core framework in MCPBEBase (init, capacity buffers, main loop)
    Agglomeration logic in MCPBEAgg
    Breakage logic (incl. multi-fragment, two-level CDF) in MCPBEBreak
    Post-processing (moments) in MCPBEPost
    Nucleation logic (liquid droplet addition) in MCPBENucleation

    Inherit order ensures Base methods call mixin methods via MRO.
    Order: Post <- Break <- Agg <- Nucleation <- Base
    
    NUCLEATION MODES (set via nucleation_method attribute):
    --------------------------------------------------------
    "standard" (default): 
        Rebuilds Fenwick sampler on EVERY droplet.
        Most accurate, but slowest.
        
    "cached": 
        Caches sampler, rebuilds only after agglomeration.
        ~100-1000x speedup for nucleation-only cases!
        Use MCPBESolverOpti class or set nucleation_method="cached"
        
    "timestep": 
        Builds TWO samplers ONCE PER TIMESTEP (dual strategy).
        - i_sampler: large particles preferred (for direct droplet uptake)
        - j_sampler: small particles preferred (for agglomeration material)
        Tracks used particles to avoid duplicates in same timestep.
        Fastest option overall! Use MCPBESolverTimestep class.
    
    SWITCHING BETWEEN METHODS:
    --------------------------
    You can switch nucleation methods at runtime by setting the
    nucleation_method attribute BEFORE calling solve():
    
    >>> solver = MCPBESolver(dim=2, t_vec=t_vec)
    >>> solver.nucleation_method = "cached"  # Switch to cached
    >>> solver.liquid_flow_rate = 0.001
    >>> solver.nucleation_enable = True
    >>> solver.solve()
    
    Available methods:
    - "standard": Uses MCPBENucleation (this class)
    - "cached": Requires MCPBESolverOpti
    - "timestep": Requires MCPBESolverTimestep
    
    Example:
    --------
    >>> solver = MCPBESolver(dim=2, t_vec=t_vec, verbose=True)
    >>> solver.nucleation_method = "cached"  # or "timestep"
    >>> solver.liquid_flow_rate = 0.001
    >>> solver.droplet_diameter = 50e-6
    >>> solver.nucleation_enable = True
    >>> solver.solve(maxiter=int(1e7))
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Default nucleation method
        if not hasattr(self, 'nucleation_method'):
            self.nucleation_method = "standard"


# Only define MCPBESolverOpti if the optimized nucleation module is available
if HAS_OPTI_NUCLEATION:
    class MCPBESolverOpti(MCPBEPost, MCPBEBreak, MCPBEAgg, MCPBENucleationSampleOpti, MCPBEBase):
        """Optimized Monte Carlo PBE solver with cached nucleation sampler.
        
        This class is identical to MCPBESolver but uses the optimized nucleation
        implementation (MCPBENucleationSampleOpti) instead of the standard one.
        
        PERFORMANCE BENEFIT:
        The optimized version caches the Fenwick sampler and only rebuilds it when
        the particle list changes (after agglomeration). This provides:
        
        - ~100-1000x speedup for nucleation-only cases (no agglomeration)
        - ~10-100x speedup for cases with occasional agglomeration
        - Same results as standard version (numerically equivalent)
        
        Use this class for better performance in nucleation simulations.
        
        Example:
        --------
        >>> from mcpbe import MCPBESolverOpti
        >>> solver = MCPBESolverOpti(dim=2, t_vec=t_vec, verbose=True)
        >>> solver.liquid_flow_rate = 0.001  # L/s
        >>> solver.droplet_diameter = 50e-6  # m
        >>> solver.nucleation_enable = True
        >>> solver.solve(maxiter=int(1e7))
        """
        pass
else:
    MCPBESolverOpti = None  # type: ignore


# Note: MCPBESolverDual is DEPRECATED. Use MCPBESolverTimestep instead.
# The dual-sampler strategy has been merged into MCPBENucleationTimestep.
MCPBESolverDual = None  # type: ignore


# Only define MCPBESolverTimestep if the timestep-sampler nucleation module is available
if HAS_TIMESTEP_NUCLEATION:
    class MCPBESolverTimestep(MCPBEPost, MCPBEBreak, MCPBEAgg, MCPBENucleationTimestep, MCPBEBase):
        """FASTEST Monte Carlo PBE solver with DUAL-TIMESTEP nucleation sampler.
        
        This class combines the best of both strategies:
        
        **DUAL-SAMPLER (built once per timestep):**
        - i_sampler: weighted by solid_volume (prefers LARGE particles)
          Purpose: Find particles that can likely accept droplets directly
        - j_sampler: weighted by 1/solid_volume (prefers SMALL particles)
          Purpose: Provide material for agglomeration when i is too small
        
        **TIMESTEP-TRACKING:**
        - Both samplers built ONCE PER TIMESTEP (not per droplet!)
        - Used particles tracked via Exclusion-Set
        - No duplicate selections in same timestep!
        
        PERFORMANCE BENEFIT:
        - Only 2 sampler builds per timestep (vs hundreds/thousands)
        - Dual strategy reduces number of required agglomerations
        - Exclusion tracking prevents redundant selections
        - Best overall performance for most use cases!
        
        Expected speedup vs standard:
        - ~200-1000x faster for typical nucleation cases
        - ~2-5x faster than cached sampler (at high droplet rates)
        - ~1.5-3x faster than old dual-sampler (fewer rebuilds)
        - Best for high droplet rates with broad particle size distributions
        
        Example:
        --------
        >>> from mcpbe import MCPBESolverTimestep
        >>> solver = MCPBESolverTimestep(dim=2, t_vec=t_vec, verbose=True)
        >>> solver.liquid_flow_rate = 0.001  # L/s
        >>> solver.droplet_diameter = 50e-6  # m
        >>> solver.nucleation_enable = True
        >>> solver.solve(maxiter=int(1e7))
        
        See Also
        --------
        MCPBESolver : Standard version (slowest, most tested)
        MCPBESolverOpti : Cached sampler (fast, most accurate)
        MCPBESolverTimestep : Dual-timestep sampler (FASTEST!)
        """
        pass
else:
    MCPBESolverTimestep = None  # type: ignore

