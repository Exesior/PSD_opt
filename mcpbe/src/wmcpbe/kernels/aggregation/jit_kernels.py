"""
JIT-compiled batch functions for aggregation kernels.

These functions compute propensities for ALL particle pairs at once,
avoiding the overhead of millions of Python function calls.

Usage:
    >>> from .jit_kernels import rebuild_r_agg_shear
    >>> r = rebuild_r_agg_shear(corr_beta, g, R, W)  # Returns array of r_i
"""

import math
from numba import njit


@njit(cache=True)
def rebuild_r_agg_shear(corr_beta: float, g: float,
                        R: float, W: float) -> float:
    """
    Compute per-particle propensity rates r_i for shear kernel.
    
    r_i = Σⱼ Wⱼ × β(i,j) where β(i,j) = corr_beta × g × (rᵢ + rⱼ)³
    
    For self-agglomeration (i==j): uses (W[i] - 1) instead of W[i].
    
    Args:
        corr_beta: Collision efficiency factor
        g: Shear rate [1/s]
        R: Particle radii array [m]
        W: Particle weights array
    
    Returns:
        Total system propensity Σᵢ r_i
    """
    a = len(W)
    total = 0.0
    for i in range(a):
        ri = R[i]
        ri_sum = 0.0
        for j in range(a):
            rj = R[j]
            Wj = W[j]
            beta = corr_beta * g * (ri + rj)**3
            if i == j:
                # Self-agglomeration: (W[i] - 1) pairs
                if Wj > 1.0:
                    ri_sum += (Wj - 1.0) * beta
            else:
                ri_sum += Wj * beta
        total += ri_sum
    return total


@njit(cache=True)
def rebuild_r_array_shear(corr_beta: float, g: float,
                          R: float, W: float,
                          r_out: float) -> None:
    """
    Compute per-particle propensity rates r_i for shear kernel.
    
    Writes results to r_out array (must be pre-allocated with length len(W)).
    
    Args:
        corr_beta: Collision efficiency factor
        g: Shear rate [1/s]
        R: Particle radii array [m]
        W: Particle weights array
        r_out: Output array for r_i values
    """
    a = len(W)
    for i in range(a):
        ri = R[i]
        ri_sum = 0.0
        for j in range(a):
            rj = R[j]
            Wj = W[j]
            beta = corr_beta * g * (ri + rj)**3
            if i == j:
                # Self-agglomeration: (W[i] - 1) pairs
                if Wj > 1.0:
                    ri_sum += (Wj - 1.0) * beta
            else:
                ri_sum += Wj * beta
        r_out[i] = ri_sum


@njit(cache=True)
def rebuild_r_agg_brownian(corr_beta: float, kT: float, viscosity: float,
                           R: float, W: float) -> float:
    """
    Compute per-particle propensity rates r_i for Brownian kernel.
    
    r_i = Σⱼ Wⱼ × β(i,j) where β(i,j) = corr_beta × 2kT×(rᵢ+rⱼ)² / (3μ×rᵢ×rⱼ)
    
    Returns:
        Total system propensity Σᵢ r_i
    """
    a = len(W)
    total = 0.0
    for i in range(a):
        ri = R[i]
        ri_sum = 0.0
        if ri <= 0:
            continue
        for j in range(a):
            rj = R[j]
            if rj <= 0:
                continue
            Wj = W[j]
            
            # Brownian kernel
            numerator = 2.0 * kT * (ri + rj)**2
            denominator = 3.0 * viscosity * ri * rj
            if denominator <= 0:
                continue
            beta = corr_beta * numerator / denominator
            
            if i == j:
                if Wj > 1.0:
                    ri_sum += (Wj - 1.0) * beta
            else:
                ri_sum += Wj * beta
        total += ri_sum
    return total


@njit(cache=True)
def rebuild_r_array_brownian(corr_beta: float, kT: float, viscosity: float,
                             R: float, W: float,
                             r_out: float) -> None:
    """
    Compute per-particle propensity rates r_i for Brownian kernel.
    
    Writes results to r_out array (must be pre-allocated).
    """
    a = len(W)
    for i in range(a):
        ri = R[i]
        ri_sum = 0.0
        if ri <= 0:
            r_out[i] = 0.0
            continue
        for j in range(a):
            rj = R[j]
            if rj <= 0:
                continue
            Wj = W[j]
            
            numerator = 2.0 * kT * (ri + rj)**2
            denominator = 3.0 * viscosity * ri * rj
            if denominator <= 0:
                continue
            beta = corr_beta * numerator / denominator
            
            if i == j:
                if Wj > 1.0:
                    ri_sum += (Wj - 1.0) * beta
            else:
                ri_sum += Wj * beta
        r_out[i] = ri_sum


@njit(cache=True)
def rebuild_r_agg_sum(corr_beta: float, R: float, W: float) -> float:
    """
    Compute per-particle propensity rates r_i for sum kernel.
    
    r_i = Σⱼ Wⱼ × β(i,j) where β(i,j) = corr_beta × (4π/3) × (rᵢ³ + rⱼ³)
    
    Returns:
        Total system propensity Σᵢ r_i
    """
    a = len(W)
    total = 0.0
    pi_factor = 4.0 / 3.0 * math.pi
    
    for i in range(a):
        ri = R[i]
        ri_sum = 0.0
        vi = pi_factor * ri**3
        for j in range(a):
            rj = R[j]
            Wj = W[j]
            vj = pi_factor * rj**3
            
            beta = corr_beta * (vi + vj)
            
            if i == j:
                if Wj > 1.0:
                    ri_sum += (Wj - 1.0) * beta
            else:
                ri_sum += Wj * beta
        total += ri_sum
    return total


@njit(cache=True)
def rebuild_r_array_sum(corr_beta: float, R: float, W: float,
                        r_out: float) -> None:
    """
    Compute per-particle propensity rates r_i for sum kernel.
    
    Writes results to r_out array (must be pre-allocated).
    """
    a = len(W)
    pi_factor = 4.0 / 3.0 * math.pi
    
    for i in range(a):
        ri = R[i]
        ri_sum = 0.0
        vi = pi_factor * ri**3
        for j in range(a):
            rj = R[j]
            Wj = W[j]
            vj = pi_factor * rj**3
            
            beta = corr_beta * (vi + vj)
            
            if i == j:
                if Wj > 1.0:
                    ri_sum += (Wj - 1.0) * beta
            else:
                ri_sum += Wj * beta
        r_out[i] = ri_sum


@njit(cache=True)
def rebuild_r_array_constant(corr_beta: float, W: float,
                              r_out: float) -> None:
    """
    Compute per-particle propensity rates r_i for constant kernel.
    
    r_i = Σⱼ Wⱼ × β where β = corr_beta (constant)
    
    For self-agglomeration (i==j): uses (W[i] - 1) instead of W[i].
    
    Writes results to r_out array (must be pre-allocated).
    """
    a = len(W)
    
    # Sum of all weights
    sum_W = 0.0
    for j in range(a):
        sum_W += W[j]
    
    for i in range(a):
        Wi = W[i]
        # For self-agglomeration, we need (W[i] - 1) instead of W[i]
        if Wi > 1.0:
            r_out[i] = corr_beta * (sum_W - 1.0)
        else:
            r_out[i] = corr_beta * (sum_W - Wi)


@njit(cache=True)
def rebuild_r_array_general(R: float, W: float,
                            corr_beta: float, g: float,
                            kernel_type: int,
                            r_out: float) -> None:
    """
    General JIT function for multiple kernel types.
    
    kernel_type:
        1 = shear (β ∝ (r1+r2)³)
        2 = brownian (β ∝ (r1+r2)²/(r1×r2))
        3 = constant (β = const)
        4 = sum (β ∝ r1³+r2³)
    
    This is a fallback for when you don't want to call type-specific functions.
    For best performance, use the specialized functions above.
    """
    a = len(W)
    pi_factor = 4.0 / 3.0 * math.pi
    kT = 4.07e-21  # Approximate at 293K
    viscosity = 1e-3  # Water
    
    for i in range(a):
        ri = R[i]
        ri_sum = 0.0
        
        for j in range(a):
            rj = R[j]
            Wj = W[j]
            
            beta = 0.0
            
            if kernel_type == 1:  # Shear
                beta = corr_beta * g * (ri + rj)**3
            elif kernel_type == 2:  # Brownian
                if ri > 0 and rj > 0:
                    numerator = 2.0 * kT * (ri + rj)**2
                    denominator = 3.0 * viscosity * ri * rj
                    if denominator > 0:
                        beta = corr_beta * numerator / denominator
            elif kernel_type == 3:  # Constant
                beta = corr_beta
            elif kernel_type == 4:  # Sum
                vi = pi_factor * ri**3
                vj = pi_factor * rj**3
                beta = corr_beta * (vi + vj)
            
            if beta > 0:
                if i == j:
                    if Wj > 1.0:
                        ri_sum += (Wj - 1.0) * beta
                else:
                    ri_sum += Wj * beta
        
        r_out[i] = ri_sum
