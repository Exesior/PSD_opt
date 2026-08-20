"""
Single source of truth for the mixer speed and the DEM correlation exponents.

Several kernels are driven by the mixer speed rather than by a shear rate:

    eke_darelius2005        collision frequency
    etm_darelius2005        collision frequency
    stokes_dynamik          collision velocity (acceptance)
    powerlaw_rumpf_dynamic  breakage rate

They all read the SAME physical quantity - there is one mixer, turning at one
speed. If each kernel carried its own default, a setup that forgets to set
`n_mixer` on one of them would silently describe two different machines at
once, and nothing in the results would give that away. Hence: one constant
here, imported by all of them, plus :func:`assert_consistent_mixer_speed`,
which refuses a configuration in which two active kernels disagree.

Origin of the exponents
-----------------------
A DEM study of a comparable mixer was evaluated over eight operating points
(circumferential velocity 5-40 m/s; the 30 m/s point has no particle-particle
data and is excluded). Two power laws were fitted:

    collision frequency         ~ n^0.0995
    mean relative velocity      ~ n^0.2852

Only the EXPONENTS transfer to this system - the DEM's absolute numbers
(prefactors ~3e7 and ~0.08 m/s) describe a different machine and a different
powder. Each kernel absorbs its prefactor into its own fit parameter
(`corr_beta`, `p1`) or normalises against a reference speed (`stokes_dynamik`).

Because only exponents transfer, `n_mixer` is defined only up to a constant
factor: switching its unit (m/s, rpm, 1/s) rescales every kernel by a constant
that the fit parameters swallow. What carries meaning is the SHAPE of the
dependence, and for a power law that is unit-free. The DEM used m/s, so that
is the natural reading, and N_MIXER_DEFAULT sits in the middle of its range.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

#: Default mixer speed. Middle of the DEM range (5-40 m/s).
N_MIXER_DEFAULT = 20.0

#: Collision FREQUENCY exponent: how often granules meet.
#: The user-supplied Excel trendline reads 0.0995; a least-squares refit of the
#: same seven usable operating points gives 0.0975. The difference is well
#: inside the scatter, and the trendline value is kept as the reference.
C_FREQ = 0.0995

#: Collision VELOCITY exponent: how hard granules meet. Already a RELATIVE
#: velocity in the DEM output, which is what a collision is governed by - no
#: assumption about directional correlation is needed to use it.
C_VEL = 0.2852

#: Breakage exponent: stressing frequency x impact energy.
#:
#:     rate of breakage = (loadings per second) x P(break per loading)
#:
#: The first factor is the collision frequency, n^C_FREQ. The second grows with
#: the impact energy, which scales as v_rel^2, i.e. n^(2*C_VEL). Their product
#: is the exponent that replaces the shear rate G in the dynamic breakage
#: kernel.
#:
#: Note that no particle mass appears in this. That is not an omission - it
#: follows from the EKE picture used on the aggregation side. Under
#: equipartition of fluctuating kinetic energy every granule carries the same
#: E_0, so v' ~ m^(-1/2), and the impact energy of a PAIR is
#:
#:     E_impact = 1/2 * mu * v_rel^2
#:              = 1/2 * [m_i m_j/(m_i+m_j)] * 2 E_0 (m_i+m_j)/(m_i m_j)
#:              = E_0
#:
#: The mass cancels exactly. A mass-dependent impact energy would contradict
#: the very kernel the aggregation runs on. Should a mass dependence ever be
#: wanted, it belongs in a SEPARATE kernel with its own stated assumption
#: (equal momentum -> E ~ 1/m, or equal speed -> E ~ m), not as a switch here.
C_BREAK = C_FREQ + 2.0 * C_VEL  # = 0.6699


def collect_mixer_speeds(kernels: Iterable[Tuple[str, Any]]) -> Dict[str, float]:
    """Map ``label -> n_mixer`` for every kernel that carries a mixer speed.

    Args:
        kernels: ``(label, kernel_or_None)`` pairs. ``None`` entries and
                 kernels without an ``n_mixer`` attribute are skipped, so the
                 caller can pass its whole kernel set without filtering.

    Returns:
        Dictionary of the mixer speeds actually configured.
    """
    found: Dict[str, float] = {}
    for label, kernel in kernels:
        if kernel is None:
            continue
        n = getattr(kernel, "n_mixer", None)
        if n is None:
            continue
        found[f"{label}={getattr(kernel, 'name', '?')}"] = float(n)
    return found


def assert_consistent_mixer_speed(kernels: Iterable[Tuple[str, Any]]) -> None:
    """Raise if two active kernels are configured for different mixer speeds.

    Deliberately an error, not a warning. The comparable check for the shear
    rate (aggregation `g` vs. solver `G`, see ``kernel_integration``) only
    warns, because there the two values at least have separate owners and a
    documented precedence. Here they do not: there is one mixer, and two
    different speeds for it is a configuration mistake with no sensible
    interpretation. Letting it through would produce a run whose collision
    frequency and breakage rate describe different machines - plausible-looking
    numbers that mean nothing.

    Args:
        kernels: ``(label, kernel_or_None)`` pairs, as for
                 :func:`collect_mixer_speeds`.

    Raises:
        ValueError: If more than one distinct mixer speed is configured.
    """
    found = collect_mixer_speeds(kernels)
    distinct = sorted(set(found.values()))
    if len(distinct) <= 1:
        return

    detail = ", ".join(f"{k}: n_mixer={v}" for k, v in sorted(found.items()))
    raise ValueError(
        f"Conflicting mixer speeds across kernels: {detail}. "
        f"All mixer-speed-driven kernels describe the same machine and must "
        f"use the same n_mixer (found {distinct}). Set n_mixer explicitly and "
        f"identically on every one of them, or leave them all at the default "
        f"of {N_MIXER_DEFAULT}."
    )
