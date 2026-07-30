"""
Kernel Framework for WMCPBE Solver.

This package provides a flexible kernel architecture for physics models
in Monte Carlo PBE simulations. Kernels are bound to solver instances
at initialization time.

Structure:
    - base.py: Abstract base classes for all kernel types
    - aggregation/: Agglomeration/collision kernels
    - breakage/: Breakage/fragmentation kernels
    - porosity_growth/: Porosity creation/growth during events
    - compression/: Porosity reduction over time
    - liquid_distribution/: Droplet/particle selection for nucleation

Usage:
    >>> from wmcpbe.kernels.aggregation import get_aggregation_kernel
    >>> kernel = get_aggregation_kernel('shear_chin1998', corr_beta=1e-3, g=1000)
    >>> beta = kernel.compute_beta(r1=1e-6, r2=2e-6)

Architecture:
    Each kernel category has:
    - Base class defining the interface
    - Factory function get_<category>_kernel(name, **params)
    - List function list_<category>_kernels()
    
    Kernels are instantiated with parameters and bound to solver at init.
"""

from .base import (
    KernelBase,
    AggregationKernel,
    BreakageKernel,
    PorosityGrowthKernel,
    CompressionKernel,
    LiquidDistributionKernel,
    LiquidInternalizationKernel,
)

__all__ = [
    'KernelBase',
    'AggregationKernel',
    'BreakageKernel',
    'PorosityGrowthKernel',
    'CompressionKernel',
    'LiquidDistributionKernel',
    'LiquidInternalizationKernel',
]
