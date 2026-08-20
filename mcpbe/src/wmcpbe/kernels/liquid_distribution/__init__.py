"""
Liquid Distribution Kernels for WMCPBE Solver.

These kernels determine which particles receive liquid droplets during
nucleation. Different selection strategies model different physical
mechanisms of droplet-particle collision.

Available kernels:
    - uniform_weighted: Weight-proportional selection

`surface_weighted` and `saturation_preferential` were removed as stale
experiments; neither was ever used by a production configuration.

Usage:
    >>> from wmcpbe.kernels.liquid_distribution import get_liquid_distribution_kernel
    >>> kernel = get_liquid_distribution_kernel('uniform_weighted')
    >>> idx = kernel.select_target_particle(solver, v_droplet=1e-18)
"""

from typing import Type
from ..base import LiquidDistributionKernel, reject_unknown_params

# Import all kernel implementations
from .uniform_weighted import UniformWeightedKernel

# Registry of available kernels
LIQUID_DIST_KERNELS: dict[str, Type[LiquidDistributionKernel]] = {
    'uniform_weighted': UniformWeightedKernel,
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
        ValueError: If the kernel name is not found, or if `params` contains a
                    name this kernel does not read (see `reject_unknown_params`)
    """
    if name not in LIQUID_DIST_KERNELS:
        available = list(LIQUID_DIST_KERNELS.keys())
        raise ValueError(
            f"Unknown liquid distribution kernel '{name}'. "
            f"Available kernels: {available}"
        )
    kernel = LIQUID_DIST_KERNELS[name](**params)
    reject_unknown_params(kernel, params)
    return kernel


def list_liquid_distribution_kernels() -> list[str]:
    """List available liquid distribution kernels."""
    return list(LIQUID_DIST_KERNELS.keys())


__all__ = [
    'get_liquid_distribution_kernel',
    'list_liquid_distribution_kernels',
    'UniformWeightedKernel',
]
