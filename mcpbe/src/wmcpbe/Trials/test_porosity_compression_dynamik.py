"""Validation for the mixer-speed porosity-compression kernels.

    porosity_compression_dynamik         k = rate * n_mixer**c_mixer
    porosity_compression_dynamik_rumpf   k = rate * n_mixer**c_mixer / sigma(eps,S)

Mirrors the scope of ``test_dry_mixer_kernels.py`` for these two kernels:
registration, formula and scaling, the c_mixer=0 fallback, scalar/array parity,
the sigma factorisation against ``powerlaw_rumpf``, the frozen-sigma
("Option C") integration property, loud failure, the n_mixer cross-check inside
a real solver, self-containment and plug-and-play portability.

Console output is German by design -- this is a working tool, not documentation.

Usage
-----
    python -m wmcpbe.Trials.test_porosity_compression_dynamik
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wmcpbe.kernels.continuous_processes import (  # noqa: E402
    get_continuous_kernel,
    list_continuous_kernels,
)
from wmcpbe.kernels.continuous_processes.porosity_compression import (  # noqa: E402
    PorosityCompressionKernel,
)
from wmcpbe.kernels.mixer_speed import (  # noqa: E402
    C_BREAK,
    N_MIXER_DEFAULT,
)

_FAILURES: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    """Eine Zusicherung protokollieren, ohne den Rest des Laufs abzubrechen."""
    if condition:
        print(f"  [ok]   {label}")
    else:
        print(f"  [FAIL] {label}" + (f"  -- {detail}" if detail else ""))
        _FAILURES.append(label)


def _poro_samples(seed: int = 0) -> np.ndarray:
    """Edge cases (NaN Vollkoerper, 0.0 poreless, eps_min, 1.0) plus a spread."""
    rng = np.random.default_rng(seed)
    edge = np.array([0.7, 0.5, 0.35, 0.3, 0.25, 0.2, 0.1, 0.0, 1.0, np.nan])
    return np.concatenate([edge, rng.random(40)])


class _FakeSolver:
    """Minimal solver surface the rumpf variant's compute_array reads."""

    def __init__(self, saturation: np.ndarray, x0: np.ndarray):
        self.saturation = np.asarray(saturation, dtype=float)
        self.X0 = np.asarray(x0, dtype=float)


# ---------------------------------------------------------------------------
# 1. Registrierung
# ---------------------------------------------------------------------------
def test_registry() -> None:
    print("\n1. Registrierung")
    reg = list_continuous_kernels()
    for name in ("porosity_compression_dynamik", "porosity_compression_dynamik_rumpf"):
        check(name in reg, f"{name} in CONTINUOUS_KERNELS", str(reg))
        k = get_continuous_kernel(name)
        check(k.name == name, f"{name}: Factory liefert passenden .name", k.name)
        check(k.category == "compression", f"{name}: category == 'compression'")

    for name in ("porosity_compression_dynamik", "porosity_compression_dynamik_rumpf"):
        try:
            get_continuous_kernel(name, n_mixxer=20.0)
            check(False, f"{name}: unbekannter Parameter wird abgelehnt", "kein ValueError")
        except ValueError as exc:
            check("n_mixer" in str(exc), f"{name}: Tippfehler 'n_mixxer' abgelehnt",
                  "Vorschlag 'n_mixer' fehlt")


# ---------------------------------------------------------------------------
# 2. Einfache Variante: Formel, Skalierung, Rueckfallebene
# ---------------------------------------------------------------------------
def test_dynamik_formula() -> None:
    print("\n2. porosity_compression_dynamik -- Formel und Skalierung")
    rate, emin, dt = 3e-3, 0.25, 7.0

    k = get_continuous_kernel("porosity_compression_dynamik",
                              rate=rate, min_porosity=emin, n_mixer=20.0)
    check(abs(k.k - rate * 20.0 ** C_BREAK) < 1e-18,
          "k == rate * n_mixer**c_mixer", f"{k.k} vs {rate * 20.0 ** C_BREAK}")

    for e0 in (0.6, 0.4, 0.3, 0.26):
        ana = emin + (e0 - emin) * np.exp(-k.k * dt)
        got = k.compute(e0, dt)
        check(abs(got - ana) < 1e-15, f"compute({e0}) == analytische Loesung",
              f"{got} vs {ana}")

    check(k.n_mixer == 20.0, "n_mixer-Attribut gesetzt (collect_mixer_speeds)")

    lo = get_continuous_kernel("porosity_compression_dynamik", rate=rate, n_mixer=10.0)
    hi = get_continuous_kernel("porosity_compression_dynamik", rate=rate, n_mixer=20.0)
    check(abs(hi.k / lo.k - 2.0 ** C_BREAK) < 1e-12,
          "Verdopplung von n_mixer gibt Faktor 2**C_BREAK", f"{hi.k / lo.k}")

    # Default rate: k ~ 0.02 (der porosity_compression-Default) bei n=20
    dflt = get_continuous_kernel("porosity_compression_dynamik", n_mixer=N_MIXER_DEFAULT)
    check(abs(dflt.k - 0.02) < 5e-4, "Default rate kalibriert auf k ~ 0.02 bei n=20",
          f"k = {dflt.k}")

    # c_mixer = 0  ->  bitgleich zu porosity_compression mit rate = rate
    dyn0 = get_continuous_kernel("porosity_compression_dynamik",
                                 rate=rate, min_porosity=emin, n_mixer=37.0, c_mixer=0.0)
    stat = PorosityCompressionKernel(rate=rate, min_porosity=emin)
    poro = _poro_samples()
    for d in (0.0, 1e-6, 0.5, 100.0):
        a = dyn0.compute_array(poro, d)
        b = stat.compute_array(poro, d)
        check(np.array_equal(np.nan_to_num(a, nan=-7.0), np.nan_to_num(b, nan=-7.0)),
              f"c_mixer=0 == porosity_compression (dt={d}, bitgenau)")


def test_dynamik_edges_and_parity() -> None:
    print("\n3. porosity_compression_dynamik -- Grenzfaelle und Skalar/Array-Paritaet")
    k = get_continuous_kernel("porosity_compression_dynamik",
                              rate=0.2, min_porosity=0.2, n_mixer=15.0)
    poro = _poro_samples(1)
    for dt in (0.0, 1e-6, 0.5, 100.0):
        arr = k.compute_array(poro, dt)
        loop = np.array([k.compute(float(p), dt) for p in poro])
        check(np.array_equal(np.nan_to_num(arr, nan=-7.0), np.nan_to_num(loop, nan=-7.0)),
              f"compute_array == Skalarschleife (dt={dt}, bitgenau)")

    check(np.isnan(k.compute(np.nan, 10.0)), "NaN (Vollkoerper) bleibt NaN")
    check(k.compute(0.2, 10.0) == 0.2, "eps == eps_min bleibt unveraendert")
    check(k.compute(0.15, 10.0) == 0.15, "eps < eps_min bleibt unveraendert (reduktiv)")
    check(k.compute(0.0, 10.0) == 0.0, "eps = 0 (porenlos) bleibt 0")
    got = k.compute(0.9, 10.0)
    check(0.2 <= got < 0.9, "eps = 0.9 wird reduziert, bleibt >= eps_min", str(got))

    for bad in ({"n_mixer": 0.0}, {"n_mixer": -3.0}, {"rate": -1.0},
                {"min_porosity": 1.5}, {"c_mixer": np.inf}):
        try:
            get_continuous_kernel("porosity_compression_dynamik", **bad)
            check(False, f"dynamik: {bad} wird abgelehnt", "kein ValueError")
        except ValueError:
            check(True, f"dynamik: {bad} wird abgelehnt")


# ---------------------------------------------------------------------------
# 4. Rumpf-Variante: sigma-Faktorisierung gegen powerlaw_rumpf
# ---------------------------------------------------------------------------
def test_rumpf_sigma() -> None:
    print("\n4. porosity_compression_dynamik_rumpf -- sigma == Rumpf-Modell")
    from wmcpbe.kernels.breakage.powerlaw_rumpf_dynamic import _sigma_jit

    kr = get_continuous_kernel("porosity_compression_dynamik_rumpf",
                               n_mixer=20.0, x_s=34e-6, k=2.5, alpha=1.15,
                               gamma=0.072, delta=0.0)
    x_s = 34e-6
    worst = 0.0
    for e in (0.15, 0.25, 0.4, 0.6, 0.85, 0.999):
        for S in (0.0, 0.1, 0.29, 0.3, 0.31, 0.5, 0.55, 0.79, 0.8, 0.81, 0.95, 1.0):
            mine = float(kr._sigma(np.array([e]), np.array([S]), x_s)[0])
            ref = _sigma_jit(min(e, kr.poro_max), S, x_s,
                             2.5, 1.15, 0.072, np.cos(0.0), 0.3, 0.8, 0.5)
            worst = max(worst, abs(mine / ref - 1.0))
    check(worst < 1e-12,
          "sigma(eps,S) == powerlaw_rumpf _sigma_jit ueber alle Regimes",
          f"max. rel. Abw. {worst:.2e}")

    # Faktorisierung sigma = phi(S) * (1-eps)/eps: phi haengt nicht von eps ab
    spread = 0.0
    for S in (0.0, 0.5, 0.9):
        phi = [float(kr._sigma(np.array([e]), np.array([S]), x_s)[0]) * e / (1 - e)
               for e in (0.2, 0.4, 0.7)]
        spread = max(spread, max(phi) / min(phi) - 1.0)
    check(spread < 1e-12, "sigma faktorisiert exakt als phi(S)*(1-eps)/eps",
          f"phi-Spread {spread:.2e}")


def test_rumpf_formula_and_optionC() -> None:
    print("\n5. porosity_compression_dynamik_rumpf -- Formel und Option-C-Integration")
    kr = get_continuous_kernel("porosity_compression_dynamik_rumpf",
                               rate=20.0, min_porosity=0.2, n_mixer=20.0,
                               x_s=34e-6, k=2.5, alpha=1.0, gamma=0.072, delta=0.0)
    rate_nc = 20.0 * 20.0 ** C_BREAK
    check(abs(kr.rate_nc - rate_nc) < 1e-9, "rate_nc == rate * n_mixer**c_mixer")

    # compute == eps_min + (eps-eps_min)*exp(-k_eff*dt) mit sigma bei eps eingefroren
    for e0, S in ((0.6, 0.0), (0.45, 0.5), (0.3001, 0.9)):
        sigma = float(kr._sigma(np.array([e0]), np.array([S]), 34e-6)[0])
        k_eff = rate_nc / sigma
        ana = 0.2 + (e0 - 0.2) * np.exp(-k_eff * 3.0)
        got = kr.compute(e0, 3.0, saturation=S)
        check(abs(got - ana) < 1e-14, f"compute({e0}, S={S}) == eingefrorene Exponentialloesung",
              f"{got} vs {ana}")

    # Option C gegen eine feine Referenzintegration der wahren DGL
    #   deps/dt = -B * eps/(1-eps) * (eps - eps_min),  B = rate_nc / phi(S)
    def reference(e0, S, T, n=400000):
        phi = float(kr._sigma(np.array([0.5]), np.array([S]), 34e-6)[0]) * 0.5 / 0.5
        B = rate_nc / phi
        e = e0
        dtf = T / n
        for _ in range(n):
            e = e - dtf * B * e / (1 - e) * (e - 0.2)
        return e

    # Frozen-sigma (exponential-Euler) error is O(k*dt); ~6e-4 in porosity at
    # dt = 1 s here, and k*dt ~ 1e-3 in a real run (dt ~ 0.1 s). The bound below
    # is deliberately generous -- the point is "small and controlled", not exact.
    for e0, S in ((0.6, 0.0), (0.5, 0.4), (0.35, 0.9)):
        for dt in (0.1, 1.0):
            ref = reference(e0, S, dt)
            got = kr.compute(e0, dt, saturation=S)
            check(abs(got - ref) < 1.5e-3 * dt + 1e-7,
                  f"Option C ~ exakte DGL (e0={e0}, S={S}, dt={dt})",
                  f"got {got:.6f} ref {ref:.6f} dabw {got - ref:+.2e}")
            check((got - ref) <= 1e-9,
                  f"Option C leicht ueberverdichtet (e0={e0}, S={S}, dt={dt})",
                  f"dabw {got - ref:+.2e}")

    # Unbedingte Stabilitaet: absurd grosses dt -> nie unter eps_min, monoton
    for e0 in (0.9, 0.5, 0.3005):
        got = kr.compute(e0, 1e6, saturation=0.5)
        check(0.2 <= got <= e0, f"dt=1e6: eps bleibt in [eps_min, eps_alt] (e0={e0})", str(got))

    # Nasser -> langsamer; dichter -> langsamer (sigma steigt, k_eff faellt)
    dry = kr.compute(0.5, 5.0, saturation=0.1)
    wet = kr.compute(0.5, 5.0, saturation=0.95)
    check(wet > dry, "nasseres Granulat verdichtet langsamer", f"dry {dry:.5f} wet {wet:.5f}")
    sig_loose = float(kr._sigma(np.array([0.6]), np.array([0.3]), 34e-6)[0])
    sig_dense = float(kr._sigma(np.array([0.35]), np.array([0.3]), 34e-6)[0])
    check(sig_dense > sig_loose,
          "dichteres Granulat ist fester -> kleinere Verdichtungsrate",
          f"sigma(0.6)={sig_loose:.3e}  sigma(0.35)={sig_dense:.3e}")

    # n_mixer-Skalierung bei festem (eps, S)
    lo = get_continuous_kernel("porosity_compression_dynamik_rumpf", rate=20.0,
                               n_mixer=10.0, x_s=34e-6, alpha=1.0)
    hi = get_continuous_kernel("porosity_compression_dynamik_rumpf", rate=20.0,
                               n_mixer=20.0, x_s=34e-6, alpha=1.0)
    # exp(-k_eff dt): vergleiche die effektiven Raten ueber -ln((e-emin)/(e0-emin))
    e0 = 0.6
    r_lo = -np.log((lo.compute(e0, 1.0, saturation=0.0) - 0.3) / (e0 - 0.3))
    r_hi = -np.log((hi.compute(e0, 1.0, saturation=0.0) - 0.3) / (e0 - 0.3))
    check(abs(r_hi / r_lo - 2.0 ** C_BREAK) < 1e-9,
          "k_eff skaliert mit n_mixer**C_BREAK", f"{r_hi / r_lo:.6f}")

    # c_mixer = 0: n_mixer wirkungslos, sigma-Widerstand bleibt aktiv
    a = get_continuous_kernel("porosity_compression_dynamik_rumpf", rate=20.0,
                              n_mixer=5.0, c_mixer=0.0, x_s=34e-6, alpha=1.0)
    b = get_continuous_kernel("porosity_compression_dynamik_rumpf", rate=20.0,
                              n_mixer=40.0, c_mixer=0.0, x_s=34e-6, alpha=1.0)
    check(a.compute(0.5, 3.0, saturation=0.2) == b.compute(0.5, 3.0, saturation=0.2),
          "c_mixer=0 macht n_mixer wirkungslos (sigma bleibt)")
    check(a.compute(0.5, 3.0, saturation=0.2) < 0.5, "sigma-Widerstand: Verdichtung findet statt")


def test_rumpf_parity_and_loud_failure() -> None:
    print("\n6. porosity_compression_dynamik_rumpf -- Paritaet und lautes Scheitern")
    kr = get_continuous_kernel("porosity_compression_dynamik_rumpf",
                               rate=25.0, min_porosity=0.2, n_mixer=20.0,
                               k=2.5, alpha=1.0, gamma=0.072, delta=0.0)

    poro = _poro_samples(2)
    rng = np.random.default_rng(3)
    sat = rng.uniform(0.0, 1.0, poro.shape)
    solver = _FakeSolver(sat, np.full(poro.shape, 34e-6))

    for dt in (0.0, 1e-6, 0.5, 100.0):
        arr = kr.compute_array(poro, dt, solver=solver)
        loop = np.array([kr.compute(float(p), dt, saturation=float(s), solver=solver)
                         for p, s in zip(poro, sat)])
        check(np.array_equal(np.nan_to_num(arr, nan=-7.0), np.nan_to_num(loop, nan=-7.0)),
              f"compute_array == Skalarschleife (dt={dt}, bitgenau)")

    # Sauter-Durchmesser aus X0 (monodispers == Durchmesser)
    kr2 = get_continuous_kernel("porosity_compression_dynamik_rumpf", n_mixer=20.0)
    _ = kr2.compute_array(np.array([0.5]), 1.0, solver=_FakeSolver(np.array([0.3]),
                                                                  np.array([50e-6])))
    check(abs(kr2._get_x_s(None) - 50e-6) < 1e-18, "x_s aus solver.X0 (mono) gecacht",
          str(kr2._get_x_s(None)))

    # Lautes Scheitern
    try:
        get_continuous_kernel("porosity_compression_dynamik_rumpf").compute_array(
            np.array([0.5]), 1.0)
        check(False, "compute_array ohne solver -> ValueError", "kein Fehler")
    except ValueError:
        check(True, "compute_array ohne solver -> ValueError")

    try:
        get_continuous_kernel("porosity_compression_dynamik_rumpf").compute(0.5, 1.0)
        check(False, "compute ohne x_s/solver -> ValueError", "kein Fehler")
    except ValueError:
        check(True, "compute ohne x_s/solver -> ValueError")

    for bad in ({"k": 3.0}, {"alpha": 0.5}, {"gamma": 0.0},
                {"delta": np.pi / 2}, {"n_mixer": 0.0}, {"poro_max": 1.0}):
        try:
            get_continuous_kernel("porosity_compression_dynamik_rumpf", x_s=34e-6, **bad)
            check(False, f"rumpf: {bad} wird abgelehnt", "kein ValueError")
        except ValueError:
            check(True, f"rumpf: {bad} wird abgelehnt")


# ---------------------------------------------------------------------------
# 7. Solver-Integration + n_mixer-Querabgleich
# ---------------------------------------------------------------------------
def _build_solver(compression_name: str, compression_params: dict, n_mixer_break=20.0):
    from wmcpbe import MCPBESolver

    n, d, poro0 = 300, 34e-6, 0.45
    solver = MCPBESolver(
        dim=1, t_vec=np.linspace(0.0, 3.0, 7), verbose=False, load_attr=False,
        init=False, seed=7,
        agg_kernel_name="eke_darelius2005",
        agg_kernel_params={"corr_beta": 5e-8, "n_mixer": 20.0},
        break_kernel_name="powerlaw_rumpf_dynamic",
        break_kernel_params={"p1": 1e12, "p2": 1.0, "n_mixer": n_mixer_break,
                             "breakrval": 4, "k": 2.5, "alpha": 1.0,
                             "gamma": 0.072, "delta": 0.0},
        porosity_growth_kernel_name="cone_model",
        porosity_compression_kernel_name=compression_name,
        porosity_compression_kernel_params=compression_params,
        liquid_internalization_kernel_name="liquid_internalization",
        liquid_internalization_kernel_params={"k_int": 1e11},
    )
    solver.process_type = "mix"
    solver.recon_enable = False

    # Monodisperse init through V_flat, as tests/bench/scenarios.py does it:
    # BOTH the solid row (V_flat[0]) and the dry row (V_flat[-1]) must be set.
    particle_volume = (4.0 / 3.0) * np.pi * (d / 2.0) ** 3
    V_flat = np.zeros((2, n), dtype=float)
    V_flat[0, :] = particle_volume * (1.0 - poro0)
    V_flat[-1, :] = particle_volume
    solver.Vc = 1.0
    solver._initialize_particles(init_Vc=False, V_flat=V_flat,
                                 W_init=np.full(n, 600.0, dtype=float))
    a = solver.a_tot
    solver.porosity[:a] = poro0
    solver.liquid_volume[:a] = 0.0
    solver.saturation[:a] = 0.0
    solver._init_lmc()
    solver._initialize_samplers()
    solver.create_nucleation_handler(enabled=True, volumetric_flow_rate=1e-11,
                                     droplet_diameter=20e-6,
                                     liquid_addition_duration=3.0, batch_size=20)
    solver.create_continuous_processes_handler(enabled=True, k_int=1e11,
                                               compression_enabled=True,
                                               compression_rate=0.05, min_porosity=0.2)
    return solver


def _solid(solver) -> float:
    a = solver.a_tot
    vd = solver.V_flat[-1, :a]
    po = solver.porosity[:a]
    vs = np.where(np.isnan(po), vd, vd * (1.0 - np.nan_to_num(po)))
    return float(np.sum(solver.W[:a] * vs))


def test_solver_integration() -> None:
    print("\n7. Solver: laeuft an, V_solid exakt erhalten, Kompression aktiv")
    cases = [
        ("porosity_compression_dynamik",
         {"rate": 0.013, "min_porosity": 0.2, "n_mixer": 20.0}),
        ("porosity_compression_dynamik_rumpf",
         {"rate": 100.0, "min_porosity": 0.2, "n_mixer": 20.0, "k": 2.5,
          "alpha": 1.0, "gamma": 0.072, "delta": 0.0}),
    ]
    for name, params in cases:
        try:
            solver = _build_solver(name, params)
        except Exception as exc:  # noqa: BLE001
            check(False, f"{name}: Solver-Bau ohne Fehler", f"{type(exc).__name__}: {exc}")
            continue
        km = solver.kernel_manager.porosity_compression_kernel
        check(km is not None and km.name == name,
              f"{name}: Kernel korrekt verdrahtet", str(getattr(km, "name", None)))

        s0 = _solid(solver)
        p0 = float(np.nanmax(solver.porosity[:solver.a_tot]))
        try:
            solver.solve(maxiter=120)
        except Exception as exc:  # noqa: BLE001
            check(False, f"{name}: solve() ohne Fehler", f"{type(exc).__name__}: {exc}")
            continue
        s1 = _solid(solver)
        a = solver.a_tot
        p1 = float(np.nanmax(solver.porosity[:a]))
        comp = solver.continuous_processes._particles_compressed_total

        check(abs(s1 - s0) / s0 < 1e-9, f"{name}: V_solid exakt erhalten",
              f"{s0:.6e} -> {s1:.6e}  rel {abs(s1 - s0) / s0:.2e}")
        check(comp > 0, f"{name}: Kompression hat stattgefunden", f"{comp} Partikel-Schritte")
        po = solver.porosity[:a]
        ok = np.all(np.isnan(po) | ((po >= 0.0) & (po < 1.0)))
        check(ok, f"{name}: Porositaet in [0,1) oder NaN")
        print(f"         (Info) max-Porositaet {p0:.3f} -> {p1:.3f} "
              f"(cone_model-Wachstum vs. Kompression)")


def test_mixer_speed_consistency() -> None:
    print("\n8. n_mixer-Querabgleich im Solver (harter Fehler)")
    from wmcpbe.kernels.mixer_speed import assert_consistent_mixer_speed

    ok_defaults = {
        "dynamik": get_continuous_kernel("porosity_compression_dynamik").n_mixer,
        "rumpf": get_continuous_kernel("porosity_compression_dynamik_rumpf").n_mixer,
    }
    check(set(ok_defaults.values()) == {N_MIXER_DEFAULT},
          f"beide Kernel teilen den Default n_mixer={N_MIXER_DEFAULT}", str(ok_defaults))

    k20 = get_continuous_kernel("porosity_compression_dynamik", n_mixer=20.0)
    k40 = get_continuous_kernel("porosity_compression_dynamik", n_mixer=40.0)
    try:
        assert_consistent_mixer_speed([("agg", k20), ("compression", k40)])
        check(False, "widerspruechliche Drehzahl -> ValueError", "kein Fehler")
    except ValueError as exc:
        check("20.0" in str(exc) and "40.0" in str(exc),
              "widerspruechliche Drehzahl -> ValueError mit beiden Werten")

    try:
        _build_solver("porosity_compression_dynamik",
                      {"rate": 0.013, "n_mixer": 40.0}, n_mixer_break=20.0)
        check(False, "Solver: Kompression n=40 vs Bruch n=20 -> ValueError", "kein Fehler")
    except ValueError as exc:
        check("mixer speed" in str(exc).lower(),
              "Solver: unterschiedliche n_mixer -> ValueError", str(exc)[:100])


# ---------------------------------------------------------------------------
# 9. Eigenstaendigkeit + Weitergabe
# ---------------------------------------------------------------------------
def test_self_containment() -> None:
    print("\n9. Eigenstaendigkeit (nur base + mixer_speed)")
    import ast
    import inspect

    from wmcpbe.kernels.continuous_processes.porosity_compression_dynamik import (
        PorosityCompressionDynamikKernel,
    )
    from wmcpbe.kernels.continuous_processes.porosity_compression_dynamik_rumpf import (
        PorosityCompressionDynamikRumpfKernel,
    )

    allowed = {"base", "mixer_speed"}
    for cls in (PorosityCompressionDynamikKernel, PorosityCompressionDynamikRumpfKernel):
        path = Path(inspect.getfile(cls))
        tree = ast.parse(path.read_text(encoding="utf-8"))
        bad = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level > 0:
                mod = (node.module or "").split(".")[0]
                if mod and mod not in allowed:
                    bad.append(f"from {'.' * node.level}{node.module}")
        check(not bad, f"{path.name}: keine Importe aus Nachbar-Kerneln", ", ".join(bad))
        base_ok = all(b.__module__.endswith("kernels.base") for b in cls.__bases__)
        check(base_ok, f"{cls.__name__}: erbt nur aus kernels.base",
              str([b.__name__ for b in cls.__bases__]))


def test_plug_and_play() -> None:
    print("\n10. Weitergabe: Kernel isoliert lauffaehig (nur base + mixer_speed)")
    import importlib
    import shutil
    import tempfile

    kernels_dir = Path(__file__).resolve().parents[1] / "kernels"
    cases = [
        ("continuous_processes/porosity_compression_dynamik.py",
         "PorosityCompressionDynamikKernel", False),
        ("continuous_processes/porosity_compression_dynamik_rumpf.py",
         "PorosityCompressionDynamikRumpfKernel", True),
    ]
    for rel, clsname, needs_x_s in cases:
        sub, fname = rel.split("/")
        tmp = Path(tempfile.mkdtemp(prefix="pnp_comp_"))
        try:
            pkg = tmp / "solo"
            (pkg / sub).mkdir(parents=True)
            (pkg / "__init__.py").write_text("", encoding="utf-8")
            (pkg / sub / "__init__.py").write_text("", encoding="utf-8")
            shutil.copy(kernels_dir / "base.py", pkg / "base.py")
            shutil.copy(kernels_dir / "mixer_speed.py", pkg / "mixer_speed.py")
            shutil.copy(kernels_dir / rel, pkg / sub / fname)

            sys.path.insert(0, str(tmp))
            try:
                for m in [m for m in sys.modules if m.startswith("solo")]:
                    del sys.modules[m]
                mod = importlib.import_module(f"solo.{sub}.{fname[:-3]}")
                kernel = getattr(mod, clsname)(**({"x_s": 34e-6} if needs_x_s else {}))
                out = kernel.compute(0.5, 1.0, saturation=0.3)
                check(0.0 <= out <= 0.5, f"{fname} laeuft ohne Nachbar-Kernel",
                      f"compute(0.5, 1.0) -> {out!r}")
            finally:
                sys.path.remove(str(tmp))
        except Exception as exc:  # noqa: BLE001
            check(False, f"{fname} laeuft ohne Nachbar-Kernel",
                  f"{type(exc).__name__}: {exc}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("=" * 72)
    print("VALIDIERUNG: drehzahlabhaengige Porositaets-Kompressionskernel")
    print("=" * 72)

    test_registry()
    test_dynamik_formula()
    test_dynamik_edges_and_parity()
    test_rumpf_sigma()
    test_rumpf_formula_and_optionC()
    test_rumpf_parity_and_loud_failure()
    test_solver_integration()
    test_mixer_speed_consistency()
    test_self_containment()
    test_plug_and_play()

    print("\n" + "=" * 72)
    if _FAILURES:
        print(f"{len(_FAILURES)} FEHLGESCHLAGEN:")
        for f in _FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("Alle Checks bestanden.")
    print("=" * 72)
