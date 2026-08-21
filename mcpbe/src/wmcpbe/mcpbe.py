"""Assembly point of the weighted MC-PBE solver.

This module contains no physics. It composes :class:`MCPBESolver` out of the
mixins that do, and offers two factories for the handlers that cannot be
mixins.

Mixins vs. handlers
-------------------
**Mixins** (``MCPBEAgg``, ``MCPBEBreak``, ``MCPBEPost``,
``ReconstructionMixin``) become part of the solver through inheritance. Their
methods work on ``self.V_flat``, ``self.W`` and friends directly, and they run
*inside* the Monte Carlo event loop when an event is drawn.

**Handlers** (``NucleationHandler``, ``ContinuousProcessesHandler``) are
separate objects that hold a reference back to the solver. They model
continuous processes -- liquid addition, capillary internalisation, pore
compression -- which are not drawn events. ``solve()`` calls their ``.step()``
after each event with the elapsed ``dt``, so they run on their own time scale
(operator splitting).

Method resolution order
-----------------------
::

    MCPBESolver -> MCPBEPost -> MCPBEBreak -> MCPBEAgg -> MCPBEBase
                -> MCPBETimeHelper -> BaseSolver -> ReconstructionMixin

Post-processing comes first because it may override anything below it.
``MCPBETimeHelper`` and ``BaseSolver`` (from ``pbe-core``) enter through
``MCPBEBase``. Note that ``ReconstructionMixin`` resolves **last**, after
``BaseSolver`` -- on a name collision, ``BaseSolver`` wins.

The concepts used throughout (computational weight ``W``, ``n_comp`` vs.
``n_phys``, control volume ``Vc``, DSMC scaling) are explained once in the
project ``README.md``.
"""

from __future__ import annotations

import warnings

from .mcpbe_base import MCPBEBase
from .mcpbe_agg import MCPBEAgg
from .mcpbe_break import MCPBEBreak
from .mcpbe_post import MCPBEPost
from .mcpbe_nucleation import NucleationHandler, NucleationConfig
from .mcpbe_continuous_processes import (
    ContinuousProcessesHandler,
    ContinuousProcessesConfig,
)
from .reconstruction_mixin import ReconstructionMixin


class MCPBESolver(MCPBEPost, MCPBEBreak, MCPBEAgg, MCPBEBase, ReconstructionMixin):
    """Weighted DSMC Monte Carlo solver for population balance equations.

    Composed of:

    ==========================  ===============================================
    ``MCPBEBase``               particle state, control volume, main loop
    ``MCPBEAgg`` / ``MCPBEBreak``  agglomeration and breakage physics
    ``MCPBEPost``               moments, PSD, statistics
    ``ReconstructionMixin``     particle-count reduction (CAM, RS, 2PM, QMX)
    ==========================  ===============================================

    Attached separately via the two factories below:
    ``solver.nucleation`` and ``solver.continuous_processes``.
    """

    def create_nucleation_handler(self, **kwargs) -> NucleationHandler:
        """Attach a :class:`NucleationHandler` for liquid addition.

        Distributes liquid droplets onto particles, sampled uniformly over
        physical particles (i.e. weighted by ``W``).

        The addition duration can be given in two mutually exclusive ways:
        directly as ``liquid_addition_duration`` [s], or derived from
        ``target_wt_percent`` together with ``solid_mass_in_mixer`` and the two
        densities. Giving both is allowed only if they agree within 0.1 %.

        ``volumetric_flow_rate`` is a physical rate [m3/s] and is used as such;
        the handler applies the DSMC correction for a changing ``Vc`` itself.

        Args:
            **kwargs: Passed to :class:`NucleationConfig`, which validates them.

        Returns:
            The handler, also stored as ``self.nucleation``.

        Raises:
            ValueError: On an invalid or contradictory parameter combination.

        Example:
            >>> solver.create_nucleation_handler(
            ...     enabled=True,
            ...     volumetric_flow_rate=1e-9,       # 1 uL/s
            ...     droplet_diameter=1e-6,           # 1 um droplets
            ...     liquid_addition_duration=300.0,  # 5 minutes
            ... )
        """
        # Replacing a handler drops its accumulated statistics, which is almost
        # never intended -- hence the warning rather than a silent overwrite.
        if hasattr(self, 'nucleation') and self.nucleation is not None:
            warnings.warn(
                "NucleationHandler already exists and will be replaced. "
                "This may lead to loss of statistics. Consider checking "
                "if nucleation is already configured before calling this method.",
                UserWarning,
                stacklevel=2
            )

        self.nucleation = NucleationHandler(self, NucleationConfig(**kwargs))

        return self.nucleation

    def create_continuous_processes_handler(self, **kwargs) -> ContinuousProcessesHandler:
        """Attach a :class:`ContinuousProcessesHandler` for the non-event physics.

        Two processes, applied in this order per time step:

        1. **Liquid internalisation** -- capillary flow from the external film
           into the pores, ``r = k_int * l_ex * (v_pore - l_intern)``
           (Braumann et al. 2007). Total liquid per particle is conserved;
           saturation is updated to ``S = l_intern / v_pore``.
        2. **Porosity compression** -- exponential pore collapse under shear.
           Reduces pore volume and externalises liquid again if ``S > 1``.

        Args:
            **kwargs: Passed to :class:`ContinuousProcessesConfig` (``enabled``,
                ``k_int``, ``compression_enabled``, ``compression_rate``,
                ``min_porosity``).

        Returns:
            The handler, also stored as ``self.continuous_processes``.

        Example:
            >>> solver.create_continuous_processes_handler(
            ...     enabled=True,
            ...     k_int=1e12,             # 1/(m3 s)
            ...     compression_enabled=True,
            ...     compression_rate=0.02,  # 1/s
            ...     min_porosity=0.3,
            ... )

        Note:
            For validation against analytical solutions use
            ``process_type="compression"`` instead: the handler runs after each
            MC event, that mode runs on fixed time steps.
        """
        if hasattr(self, 'continuous_processes') and self.continuous_processes is not None:
            warnings.warn(
                "ContinuousProcessesHandler already exists and will be replaced. "
                "This may lead to loss of statistics. Consider checking "
                "if continuous processes are already configured before calling this method.",
                UserWarning,
                stacklevel=2
            )

        self.continuous_processes = ContinuousProcessesHandler(
            self, ContinuousProcessesConfig(**kwargs)
        )
        return self.continuous_processes
