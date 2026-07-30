"""
Porosity Growth Kernels for WMCPBE Solver.

These kernels model porosity creation and growth during discrete events
(agglomeration, nucleation). They are called when particles merge or form.

Available kernels:
    - volume_mixing: Simple volume addition (existing behavior)
    - incomplete_mixing: Trapped pores from incomplete coalescence (NEW)
    - cone_model: Geometric pore formation from contact cones (NEW)

Usage:
    >>> from wmcpbe.kernels.porosity_growth import get_porosity_growth_kernel
    >>> kernel = get_porosity_growth_kernel('volume_mixing')
    >>> v_dry, poro = kernel.compute_merged_porosity(v1, p1, v2, p2)
"""

from typing import Type
from ..base import PorosityGrowthKernel

# Import all kernel implementations
from .volume_mixing import VolumeMixingKernel
from .incomplete_mixing import IncompleteMixingKernel
from .cone_model import ConeModelKernel

# Registry of available kernels
POROSITY_GROWTH_KERNELS: dict[str, Type[PorosityGrowthKernel]] = {
    'volume_mixing': VolumeMixingKernel,
    'incomplete_mixing': IncompleteMixingKernel,
    'cone_model': ConeModelKernel,
}


def get_porosity_growth_kernel(name: str, **params) -> PorosityGrowthKernel:
    """
    Factory function to create porosity growth kernel by name.
    
    Args:
        name: Kernel name (see list_porosity_growth_kernels())
        **params: Kernel-specific parameters
    
    Returns:
        Instantiated PorosityGrowthKernel
    
    Raises:
        ValueError: If kernel name is not found
    """
    if name not in POROSITY_GROWTH_KERNELS:
        available = list(POROSITY_GROWTH_KERNELS.keys())
        raise ValueError(
            f"Unknown porosity growth kernel '{name}'. "
            f"Available kernels: {available}"
        )
    return POROSITY_GROWTH_KERNELS[name](**params)


def list_porosity_growth_kernels() -> list[str]:
    """List available porosity growth kernels."""
    return list(POROSITY_GROWTH_KERNELS.keys())


__all__ = [
    'get_porosity_growth_kernel',
    'list_porosity_growth_kernels',
    'VolumeMixingKernel',
    'IncompleteMixingKernel',
    'ConeModelKernel',
]
