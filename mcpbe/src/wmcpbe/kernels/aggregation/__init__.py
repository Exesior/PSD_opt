"""
Aggregation Kernels for WMCPBE Solver.

Available kernels:
    - shear_chin1998: Shear-induced collisions (Chin et al. 1998)
    - brownian_tsouris1995: Brownian diffusion (Tsouris et al. 1995)
    - constant: Constant collision frequency
    - sum: Sum kernel (for validation)
    - eke_darelius2005: Equipartition of Kinetic Energy, mixer-speed driven
                        (dry mixing, no shear rate needed)
    - etm_darelius2005: Equipartition of Translational Momentum (dry mixing)

Usage:
    >>> from wmcpbe.kernels.aggregation import get_aggregation_kernel
    >>> kernel = get_aggregation_kernel('shear_chin1998', corr_beta=1e-3, g=1000)
    >>> beta = kernel.compute_beta(r1=1e-6, r2=2e-6)
"""

from typing import Type
from ..base import AggregationKernel, reject_unknown_params

# Import all kernel implementations
from .shear_chin1998 import ShearChinKernel
from .brownian_tsouris1995 import BrownianKernel
from .constant import ConstantKernel
from .sum_kernel import SumKernel
from .eke_darelius2005 import EKEDareliusKernel
from .etm_darelius2005 import ETMDareliusKernel

# Registry of available kernels
AGG_KERNELS: dict[str, Type[AggregationKernel]] = {
    'shear_chin1998': ShearChinKernel,
    'brownian_tsouris1995': BrownianKernel,
    'constant': ConstantKernel,
    'sum': SumKernel,
    'eke_darelius2005': EKEDareliusKernel,
    'etm_darelius2005': ETMDareliusKernel,
}


def get_aggregation_kernel(name: str, **params) -> AggregationKernel:
    """
    Factory function to create aggregation kernel by name.
    
    Args:
        name: Kernel name (see list_aggregation_kernels())
        **params: Kernel-specific parameters
    
    Returns:
        Instantiated AggregationKernel
    
    Raises:
        ValueError: If the kernel name is not found, or if `params` contains a
                    name this kernel does not read (see `reject_unknown_params`)

    Example:
        >>> kernel = get_aggregation_kernel('shear_chin1998', corr_beta=1e-3, g=1000)
    """
    if name not in AGG_KERNELS:
        available = list(AGG_KERNELS.keys())
        raise ValueError(
            f"Unknown aggregation kernel '{name}'. "
            f"Available kernels: {available}"
        )
    kernel = AGG_KERNELS[name](**params)
    reject_unknown_params(kernel, params)
    return kernel


def list_aggregation_kernels() -> list[str]:
    """
    List available aggregation kernels.
    
    Returns:
        List of kernel names
    """
    return list(AGG_KERNELS.keys())


__all__ = [
    'get_aggregation_kernel',
    'list_aggregation_kernels',
    'ShearChinKernel',
    'BrownianKernel',
    'ConstantKernel',
    'SumKernel',
    'EKEDareliusKernel',
    'ETMDareliusKernel',
]
