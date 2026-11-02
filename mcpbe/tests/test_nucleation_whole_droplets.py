"""Nucleation places whole droplets only -- one nucleus per droplet.

Regression tests for the droplet-packet fix of 2026-09-30
(``mcpbe_nucleation.py::_distribute_one_droplet_with_dW``).

The rule (Eric, 2026-09-30)::

    Only whole droplets are placed. If the budget holds one droplet, exactly
    ONE nucleus is built and it receives the whole droplet -- not 3.19 nuclei
    with a third of a droplet each. The rest below one droplet at the end of
    the addition window is added as ONE smaller droplet.

Before the fix, the forced agglomeration that builds a nucleus ran with the
agglomeration batch cap ``agg_dW_max`` instead of the droplet packet. With
``W = 3.19`` per computational particle one droplet built 3.19 nuclei of ~69
primary particles each and spread itself over all of them (until 14.08.2026
the surplus nuclei stayed dry instead). Either way 3-5x too much powder ended
up in nuclei.

The set-up mirrors the AJ30 fit configuration (SIM_FRACTION = 1e-7): 34 um
primary particles with the non-integer weight 3.19, 150 um droplets (86x the
primary-particle volume, so every droplet triggers forced agglomeration),
``batch_size = 20`` and ``agg_dW_max = 5``.

Run::

    python -m pytest tests/test_nucleation_whole_droplets.py -v
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from .bench.scenarios import Scenario, build

D_PRIMARY = 34e-6                                   # [m]
D_DROPLET = 150e-6                                  # [m]
V_PRIMARY = np.pi / 6.0 * D_PRIMARY ** 3            # [m^3]
V_DROPLET = np.pi / 6.0 * D_DROPLET ** 3            # [m^3]
FLOW_RATE = 1.7667e-13                              # [m^3/s] (AJ30, f = 1e-7)
WINDOW = 60.0                                       # [s]
RTOL = 1e-12

#: AJ30-like wet granulation, as the fit adapter wires it (reduced to what the
#: nucleation path needs). Not part of ``SCENARIOS`` on purpose: the golden
#: reference does not cover it.
AJ30_NUCLEATION = Scenario(
    name="aj30_nucleation_whole_droplets",
    dim=1,
    process_type="mix",
    a0=500,
    t_end=WINDOW + 1.0,
    maxiter=5_000_000,
    seed=7,
    agg_kernel_name="eke_darelius2005",
    agg_kernel_params={"corr_beta": 8.3e-13, "n_mixer": 30.0},
    agg_acceptance_kernel_name="stokes_dynamik",
    agg_acceptance_kernel_params={
        "U_coll_ref": 0.436, "n_mixer": 30.0, "binder_viscosity": 0.1,
        "rho_solid": 600.0, "rho_liquid": 1000.0, "h_a": 500e-9,
    },
    porosity_growth_kernel_name="cone_model",
    porosity_growth_kernel_params={"k_agg": 0.85, "k_break": 0.78},
    liquid_internalization_kernel_name="liquid_internalization",
    liquid_internalization_kernel_params={"k_int": 4e11},
    liq_internalisation_agglomeration_kernel_name="liq_internalisation_agglomeration",
    solver_attrs={
        "agg_propensity_mode": "moment",
        "recon_enable": False,
        "agg_dW_min": 1.0,
        "agg_dW_max": 5.0,
        "SIZEEVAL": 0,
    },
    init_particles={
        "particle_diameter": D_PRIMARY,
        "initial_porosity": 0.0,
        "initial_particles": 500,
        "initial_weight": 3.1874,
        "control_volume": 1e-10,
    },
    granulation=True,
    nucleation_params=dict(
        enabled=True,
        volumetric_flow_rate=FLOW_RATE,
        droplet_diameter=D_DROPLET,
        liquid_addition_start=0.0,
        liquid_addition_duration=WINDOW,
        batch_size=20.0,
    ),
    continuous_params=dict(
        enabled=True, k_int=4e11, compression_enabled=True,
        compression_rate=4.8e-5, min_porosity=0.2,
    ),
)


def _state(solver):
    a = solver.a_tot
    W = solver.W[:a].copy()
    v_solid = solver.V_flat[0, :a].copy()
    v_dry = solver.V_flat[-1, :a].copy()
    liquid = solver.liquid_volume[:a].copy()
    return W, v_solid, v_dry, liquid


def _vc_scale(solver):
    return solver.nucleation._Vc_reference / solver.Vc


def _build(**nucleation_overrides):
    scenario = AJ30_NUCLEATION
    if nucleation_overrides:
        params = dict(scenario.nucleation_params)
        params.update(nucleation_overrides)
        scenario = dataclasses.replace(scenario, nucleation_params=params)
    return build(scenario)


def _place_one_droplet(solver):
    """Put exactly one droplet into the account and let the handler place it."""
    nuc = solver.nucleation
    nuc._liquid_remainder = 0.0
    nuc._distribute_liquid_volume(V_DROPLET)
    return nuc


@pytest.mark.parametrize("batch_size", [20.0, 1.0])
def test_one_droplet_builds_one_wet_nucleus(batch_size):
    """One droplet -> exactly one physical nucleus, wetted with the whole droplet.

    ``batch_size = 20`` is the capping case (before: 3.19 nuclei, each with
    1/3.19 of the droplet); ``batch_size = 1`` the dry-nucleus case (before:
    1 wet nucleus plus up to 4 dry ones built with agg_dW_max).
    """
    solver = _build(batch_size=batch_size)
    W0, vs0, _, L0 = _state(solver)
    solid0 = np.sum(W0 * vs0)

    nuc = _place_one_droplet(solver)
    W, v_solid, v_dry, liquid = _state(solver)
    vc_scale = _vc_scale(solver)

    nuclei = v_solid > 1.5 * V_PRIMARY          # everything bigger than a primary
    wet = liquid > 0.0

    # exactly one physical nucleus exists, and it is wet
    assert np.sum(W[nuclei]) * vc_scale == pytest.approx(1.0, rel=RTOL)
    assert not np.any(nuclei & ~wet), "dry nucleus built by the droplet"
    # every wetted particle carries the WHOLE droplet
    np.testing.assert_allclose(liquid[wet], V_DROPLET, rtol=RTOL)
    # the nucleus can hold the droplet (capacity rule) ...
    assert np.all(v_dry[wet] >= V_DROPLET)
    # ... and was not built bigger than needed: one merge less would not do
    assert np.all(v_solid[wet] - V_PRIMARY < V_DROPLET)

    # conservation
    assert np.sum(W * v_solid) == pytest.approx(solid0, rel=RTOL)
    assert np.sum(W * liquid) * vc_scale == pytest.approx(V_DROPLET, rel=RTOL)
    assert nuc._liquid_remainder == pytest.approx(0.0, abs=RTOL * V_DROPLET)


def test_rest_is_one_smaller_droplet():
    """The rest below one droplet is placed completely as ONE smaller droplet."""
    solver = _build()
    W0, vs0, _, _ = _state(solver)
    solid0 = np.sum(W0 * vs0)
    nuc = solver.nucleation
    v_rest = 0.62 * V_DROPLET
    nuc._liquid_remainder = v_rest

    nuc._distribute_remaining_liquid(at_time=WINDOW)
    W, v_solid, v_dry, liquid = _state(solver)
    vc_scale = _vc_scale(solver)
    wet = liquid > 0.0

    assert nuc._liquid_remainder == pytest.approx(0.0, abs=RTOL * V_DROPLET)
    # one physical particle (possibly spread over several packets) got it ...
    assert np.sum(W[wet]) * vc_scale == pytest.approx(1.0, rel=RTOL)
    # ... and each wetted particle holds the whole smaller droplet
    np.testing.assert_allclose(liquid[wet], v_rest, rtol=RTOL)
    assert np.all(v_dry[wet] >= v_rest)
    assert not np.any((v_solid > 1.5 * V_PRIMARY) & ~wet)
    assert np.sum(W * liquid) * vc_scale == pytest.approx(v_rest, rel=RTOL)
    assert np.sum(W * v_solid) == pytest.approx(solid0, rel=RTOL)


def test_addition_window_conserves_mass():
    """A full addition window: all liquid arrives, solid is conserved."""
    solver = _build()
    W0, vs0, _, _ = _state(solver)
    solid0 = np.sum(W0 * vs0)

    solver.solve(maxiter=AJ30_NUCLEATION.maxiter)
    W, v_solid, _, liquid = _state(solver)
    nuc = solver.nucleation

    liquid_expected = FLOW_RATE * WINDOW                # physical, in Vc
    assert np.sum(W * liquid) == pytest.approx(liquid_expected, rel=1e-9)
    assert nuc._liquid_remainder == pytest.approx(0.0, abs=1e-9 * V_DROPLET)
    assert np.sum(W * v_solid) == pytest.approx(solid0, rel=1e-9)
