"""
Agglomeration Acceptance Kernels for WMCPBE Solver.

These kernels determine if a collision results in agglomeration or if particles
bounce off. Called AFTER partner selection, BEFORE agglomeration execution.

Available kernels:
    - stokes_krit: Stokes criterion (Braumann et al. 2007)
    - stokes_dynamik: Stokes criterion with mixer-speed-dependent collision
                      velocity (dry mixing)
    - fittable: Simple fit parameter (rng < u_acc)

Usage:
    >>> from wmcpbe.kernels.agglomeration_acceptance import get_agglomeration_acceptance_kernel
    >>> kernel = get_agglomeration_acceptance_kernel('stokes_krit',
    ...     U_coll=1.0,
    ...     binder_viscosity=0.1,
    ...     rho_solid=2500.0,
    ...     rho_liquid=1000.0,
    ...     h_a=500e-9
    ... )
    >>> accepted = kernel.accept_collision(r1, r2, v_dry1, v_dry2, solver=solver)
"""

from typing import Type

# Import base class
from ..base import AggAcceptanceKernel, reject_unknown_params

# Import all kernel implementations
from .stokes_krit import StokesKritKernel
from .stokes_dynamik import StokesDynamikKernel
from .fittable import FittableKernel

# Registry of available kernels
AGG_ACCEPTANCE_KERNELS: dict[str, Type[AggAcceptanceKernel]] = {
    'stokes_krit': StokesKritKernel,
    'stokes_dynamik': StokesDynamikKernel,
    'fittable': FittableKernel,
}


def get_agglomeration_acceptance_kernel(name: str, **params) -> AggAcceptanceKernel:
    """
    Factory function to create agglomeration acceptance kernel by name.
    
    Args:
        name: Kernel name (see list_agglomeration_acceptance_kernels())
        **params: Kernel-specific parameters
    
    Returns:
        Instantiated AggAcceptanceKernel
    
    Raises:
        ValueError: If the kernel name is not found, or if `params` contains a
                    name this kernel does not read (see `reject_unknown_params`)

    Example:
        >>> kernel = get_agglomeration_acceptance_kernel('stokes_krit',
        ...     U_coll=1.0, binder_viscosity=0.1)
    """
    if name not in AGG_ACCEPTANCE_KERNELS:
        available = list(AGG_ACCEPTANCE_KERNELS.keys())
        raise ValueError(
            f"Unknown agglomeration acceptance kernel '{name}'. "
            f"Available kernels: {available}"
        )
    kernel = AGG_ACCEPTANCE_KERNELS[name](**params)
    reject_unknown_params(kernel, params)
    return kernel


def list_agglomeration_acceptance_kernels() -> list[str]:
    """
    List available agglomeration acceptance kernels.
    
    Returns:
        List of kernel names
    """
    return list(AGG_ACCEPTANCE_KERNELS.keys())


__all__ = [
    'get_agglomeration_acceptance_kernel',
    'list_agglomeration_acceptance_kernels',
    'StokesKritKernel',
    'StokesDynamikKernel',
    'FittableKernel',
]
