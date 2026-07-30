"""
Liquid Distribution Kernels for WMCPBE Solver.

These kernels determine which particles receive liquid droplets during
nucleation. Different selection strategies model different physical
mechanisms of droplet-particle collision.

Available kernels:
    - uniform_weighted: Weight-proportional selection (existing behavior)
    - surface_weighted: Surface-area-weighted selection (NEW)
    - saturation_preferential: Prefer unsaturated particles (NEW)

Usage:
    >>> from wmcpbe.kernels.liquid_distribution import get_liquid_distribution_kernel
    >>> kernel = get_liquid_distribution_kernel('uniform_weighted')
    >>> idx = kernel.select_target_particle(solver, v_droplet=1e-18)
"""

from typing import Type
from ..base import LiquidDistributionKernel

# Import all kernel implementations
from .uniform_weighted import UniformWeightedKernel
from .surface_weighted import SurfaceWeightedKernel
from .saturation_preferential import SaturationPreferentialKernel

# Registry of available kernels
LIQUID_DIST_KERNELS: dict[str, Type[LiquidDistributionKernel]] = {
    'uniform_weighted': UniformWeightedKernel,
    'surface_weighted': SurfaceWeightedKernel,
    'saturation_preferential': SaturationPreferentialKernel,
}


def get_liquid_distribution_kernel(name: str, **params) -> LiquidDistributionKernel:
    """
    Factory function to create liquid distribution kernel by name.
    
    Args:
        name: Kernel name (see list_liquid_distribution_kernels())
        **params: Kernel-specific parameters
    
    Returns:
        Instantiated LiquidDistributionKernel
    
    Raises:
        ValueError: If kernel name is not found
    """
    if name not in LIQUID_DIST_KERNELS:
        available = list(LIQUID_DIST_KERNELS.keys())
        raise ValueError(
            f"Unknown liquid distribution kernel '{name}'. "
            f"Available kernels: {available}"
        )
    return LIQUID_DIST_KERNELS[name](**params)


def list_liquid_distribution_kernels() -> list[str]:
    """List available liquid distribution kernels."""
    return list(LIQUID_DIST_KERNELS.keys())


__all__ = [
    'get_liquid_distribution_kernel',
    'list_liquid_distribution_kernels',
    'UniformWeightedKernel',
    'SurfaceWeightedKernel',
    'SaturationPreferentialKernel',
]
