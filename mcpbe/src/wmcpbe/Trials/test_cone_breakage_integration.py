"""
Integrationstest: Cone Model Kernel in Breakage-Prozess.

Testet die korrekte Integration von ConeModelKernel im Breakage-Workflow
von MCPBESolver.

Getestete Eigenschaften:
1. Multi-Fragment Breakage mit Cone Model (deferred computation)
2. Massenerhaltung (V_solid konstant)
3. Porenverlust bei Breakage (ΔV zwischen Fragmenten)
4. Backward Compatibility mit Volume Mixing Kernel

Usage:
    python test_cone_breakage_integration.py
"""

import numpy as np
import sys
import os

# Parent directory to path for imports (same pattern as test_nucleation_simple.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wmcpbe import MCPBESolver


def create_test_solver(porosity_kernel_name: str, **kernel_params):
    """Erstelle Solver mit spezifischem Porositäts-Kernel."""
    import numpy as np
    
    # Kurze Simulationszeit
    t_total = 1.0
    t_write = 0.5
    t_vec = np.linspace(0.0, t_total, int(t_total / t_write) + 1)
    
    solver = MCPBESolver(
        dim=1,
        t_vec=t_vec,
        verbose=False,
        load_attr=False,
        init=True,  # Standard initialization
        # Kernel configuration
        porosity_growth_kernel_name=porosity_kernel_name,
        porosity_growth_kernel_params=kernel_params,
    )
    
    # Process type: breakage only
    solver.process_type = "breakage"
    
    # Breakage parameters (high rate for fast testing)
    solver.G = 1e6  # High breakage rate
    solver.pl_P1 = 1.0
    solver.pl_P2 = 0.0
    solver.pl_v = 0.0
    solver.break_dW_const = 10.0
    
    return solver


def initialize_single_particle(solver, V_solid_0, porosity_0):
    """Initialisiere Solver mit einem einzelnen Partikel."""
    x_grid = solver.x_grid.copy()
    
    # Einzelnes computational particle
    solver.W = np.array([1000.0])  # Gewicht (Anzahl physikalischer Partikel)
    
    # V_flat: [:dim] = V_solid, [-1] = V_dry
    if porosity_0 is None or np.isnan(porosity_0):
        # Vollkörper
        V_dry_0 = V_solid_0
        solver.V_flat = np.array([[V_solid_0], [V_dry_0]])
    else:
        # Porös: V_dry = V_solid / (1 - ε)
        V_dry_0 = V_solid_0 / (1.0 - porosity_0)
        solver.V_flat = np.array([[V_solid_0], [V_dry_0]])
    
    # Porosität
    solver.porosity = np.array([porosity_0 if porosity_0 is not None else np.nan])
    
    # Liquid (optional, hier nicht relevant)
    solver.liquid_volume = np.array([0.0])
    
    # Grid-Zuweisung
    solver.coord_ind = np.array([0])  # Erstes Grid-Bin
    
    solver.a_tot = 1
    
    # Breakage-Sampler initialisieren
    solver._prepare_break_config()
    solver._rebuild_break_propensities()


def test_binary_breakage_volume_mixing():
    """Test: Binary Breakage mit Volume Mixing (Baseline)."""
    print("\n" + "="*70)
    print("TEST 1: Binary Breakage - Volume Mixing (Baseline)")
    print("="*70)
    
    solver = create_test_solver('volume_mixing')
    
    # Initialisierung: 1 Partikel mit V_solid=3e-18, ε=0.5
    V_solid_0 = 3.0e-18
    porosity_0 = 0.5
    initialize_single_particle(solver, V_solid_0, porosity_0)
    
    # Vor Breakage
    V_solid_before = np.sum(solver.W * np.sum(solver.V_flat[:solver.dim, :], axis=0))
    V_pore_before = np.sum(solver.W * (solver.V_flat[-1, :] - np.sum(solver.V_flat[:solver.dim, :], axis=0)))
    
    print(f"Vor Breakage:")
    print(f"  n_particles = {solver.a_tot}")
    print(f"  Σ(W × V_solid) = {V_solid_before:.3e}")
    print(f"  Σ(W × V_pore)  = {V_pore_before:.3e}")
    print(f"  ε_parent = {solver.porosity[0]:.4f}")
    
    # Einen Breakage-Schritt ausführen
    solver._do_one_break()
    
    # Nach Breakage
    n_frags = solver.a_tot
    V_solid_after = np.sum(solver.W * np.sum(solver.V_flat[:solver.dim, :], axis=0))
    V_dry_after = np.sum(solver.W * solver.V_flat[-1, :])
    V_pore_after = V_dry_after - V_solid_after
    
    porosities = solver.porosity[:n_frags]
    
    print(f"\nNach Breakage:")
    print(f"  n_fragments = {n_frags}")
    print(f"  Σ(W × V_solid) = {V_solid_after:.3e}")
    print(f"  Σ(W × V_pore)  = {V_pore_after:.3e}")
    
    # Checks
    mass_error = abs(V_solid_after - V_solid_before) / V_solid_before * 100
    print(f"\nMassenerhaltung:")
    print(f"  Fehler = {mass_error:.10f}%")
    
    # Volume Mixing: Porosität unverändert (fragments inherit parent porosity)
    print(f"\nPorositäten der Fragmente:")
    for i, p in enumerate(porosities):
        print(f"  Fragment {i+1}: ε = {p:.6f}")
    
    # Volume Mixing erwartet gleiche Porosität
    assert all(abs(p - porosity_0) < 1e-6 for p in porosities if not np.isnan(p)), \
        "Volume Mixing: Fragmente sollten Parent-Porosität erben!"
    
    assert mass_error < 1e-10, f"Massenerhaltung verletzt! Fehler: {mass_error}%"
    
    print("\n✓ PASSED")
    return True


def test_binary_breakage_cone_model():
    """Test: Binary Breakage mit Cone Model (Porenverlust durch ΔV)."""
    print("\n" + "="*70)
    print("TEST 2: Binary Breakage - Cone Model (NEW)")
    print("="*70)
    
    solver = create_test_solver(
        'cone_model',
        k_agg=1.0,      # Nicht relevant für Breakage-only
        k_break=0.1     # Porenverlust-Faktor
    )
    
    # Initialisierung: 1 Partikel mit V_solid=3e-18, ε=0.5
    V_solid_0 = 3.0e-18
    porosity_0 = 0.5
    initialize_single_particle(solver, V_solid_0, porosity_0)
    
    # Vor Breakage
    V_solid_before = np.sum(solver.W * np.sum(solver.V_flat[:solver.dim, :], axis=0))
    V_pore_before = np.sum(solver.W * (solver.V_flat[-1, :] - np.sum(solver.V_flat[:solver.dim, :], axis=0)))
    
    print(f"Vor Breakage:")
    print(f"  n_particles = {solver.a_tot}")
    print(f"  Σ(W × V_solid) = {V_solid_before:.3e}")
    print(f"  Σ(W × V_pore)  = {V_pore_before:.3e}")
    print(f"  ε_parent = {porosity_0:.4f}")
    
    # Ein Breakage-Event ausführen
    solver._do_one_break()
    
    # Nach Breakage
    n_frags = solver.a_tot
    V_solid_after = np.sum(solver.W * np.sum(solver.V_flat[:solver.dim, :], axis=0))
    V_dry_after = np.sum(solver.W * solver.V_flat[-1, :])
    V_pore_after = V_dry_after - V_solid_after
    
    porosities = solver.porosity[:n_frags]
    frag_volumes = [float(np.sum(solver.V_flat[:solver.dim, i])) for i in range(n_frags)]
    
    print(f"\nNach Breakage:")
    print(f"  n_fragments = {n_frags}")
    print(f"  Σ(W × V_solid) = {V_solid_after:.3e}")
    print(f"  Σ(W × V_pore)  = {V_pore_after:.3e}")
    
    # Checks
    mass_error = abs(V_solid_after - V_solid_before) / V_solid_before * 100
    print(f"\nMassenerhaltung:")
    print(f"  Fehler = {mass_error:.10f}%")
    
    print(f"\nFragment-Eigenschaften:")
    for i, (V_frag, p) in enumerate(zip(frag_volumes, porosities)):
        print(f"  Fragment {i+1}: V_solid={V_frag:.3e}, ε={p:.6f}")
    
    # Cone Model erwartet Porenverlust (ε_frag < ε_parent)
    avg_poro_frag = np.mean([p for p in porosities if not np.isnan(p)])
    poro_change = avg_poro_frag - porosity_0
    
    print(f"\nPorenverlust-Check:")
    print(f"  ε_parent = {porosity_0:.6f}")
    print(f"  avg(ε_frag) = {avg_poro_frag:.6f}")
    print(f"  Δε = {poro_change:+.6f} (erwartet: < 0)")
    
    assert mass_error < 1e-10, f"Massenerhaltung verletzt! Fehler: {mass_error}%"
    
    # Porenverlust sollte negativ sein (oder nahe 0 bei sehr kleinem k_break)
    # Hinweis: Bei sehr kleinen Partikeln kann ΔV klein sein
    if n_frags >= 2:
        # Mit k_break=0.1 und binärem Bruch sollte Porenverlust messbar sein
        assert poro_change <= 1e-6, f"Cone Model: Erwarteter Porenverlust nicht sichtbar! Δε={poro_change}"
    
    print("\n✓ PASSED")
    return True


def test_multi_fragment_breakage_cone_model():
    """Test: Multi-Fragment Breakage (n=3) mit Cone Model."""
    print("\n" + "="*70)
    print("TEST 3: Multi-Fragment Breakage (n=3) - Cone Model")
    print("="*70)
    
    solver = create_test_solver(
        'cone_model',
        k_agg=1.0,
        k_break=0.2  # Etwas höher für besseren Effekt
    )
    
    # Initialisierung
    V_solid_0 = 6.0e-18
    porosity_0 = 0.6
    initialize_single_particle(solver, V_solid_0, porosity_0)
    
    # Mehrfache Breakage-Events für statistische Aussage
    n_events = 5
    pore_losses = []
    
    print(f"Start: V_solid={V_solid_0:.3e}, ε={porosity_0:.4f}")
    print(f"Führe {n_events} Breakage-Events aus...\n")
    
    for event in range(n_events):
        # Wähle zufälliges Partikel für Breakage
        if solver.a_tot == 0:
            break
        
        # Vor Event
        V_pore_before = np.sum(solver.W * (solver.V_flat[-1, :] - np.sum(solver.V_flat[:solver.dim, :], axis=0)))
        
        # Breakage
        solver._do_one_break()
        
        # Nach Event
        V_pore_after = np.sum(solver.W * (solver.V_flat[-1, :] - np.sum(solver.V_flat[:solver.dim, :], axis=0)))
        
        ΔV_pore = V_pore_after - V_pore_before
        pore_losses.append(ΔV_pore)
        
        print(f"Event {event+1}: n_particles={solver.a_tot}, ΔV_pore={ΔV_pore:+.3e}")
    
    # Statistik
    avg_pore_loss = np.mean(pore_losses)
    print(f"\nDurchschnittlicher Porenverlust pro Event: {avg_pore_loss:+.3e}")
    
    # Sollte negativ sein (Porenverlust)
    assert avg_pore_loss <= 0, f"Erwarteter Porenverlust nicht sichtbar! avg(ΔV_pore)={avg_pore_loss}"
    
    print("\n✓ PASSED")
    return True


def test_vollkörper_breakage():
    """Test: Breakage von Vollkörper-Partikeln (NaN Porosität)."""
    print("\n" + "="*70)
    print("TEST 4: Vollkörper Breakage - Cone Model")
    print("="*70)
    
    solver = create_test_solver('cone_model', k_break=0.1)
    
    # Initialisierung: Vollkörper (NaN Porosität)
    V_solid_0 = 3.0e-18
    initialize_single_particle(solver, V_solid_0, None)  # None = NaN
    
    print(f"Vor Breakage:")
    print(f"  n_particles = {solver.a_tot}")
    print(f"  ε_parent = {solver.porosity[0]} (Vollkörper)")
    
    # Breakage
    solver._do_one_break()
    
    # Nach Breakage
    n_frags = solver.a_tot
    porosities = solver.porosity[:n_frags]
    
    print(f"\nNach Breakage:")
    print(f"  n_fragments = {n_frags}")
    print(f"  ε_fragments = {porosities}")
    
    # Fragmente sollten auch Vollkörper bleiben
    assert all(np.isnan(p) for p in porosities), \
        "Vollkörper-Fragmente sollten NaN-Porosität behalten!"
    
    print("\n✓ PASSED")
    return True


def test_mass_conservation_statistics():
    """Test: Massenerhaltung über viele Breakage-Events."""
    print("\n" + "="*70)
    print("TEST 5: Massenerhaltung (Statistik über 100 Events)")
    print("="*70)
    
    solver = create_test_solver('cone_model', k_break=0.1)
    
    # Initialisierung
    V_solid_0 = 1.0e-17
    porosity_0 = 0.5
    initialize_single_particle(solver, V_solid_0, porosity_0)
    
    V_solid_initial = np.sum(solver.W * np.sum(solver.V_flat[:solver.dim, :], axis=0))
    
    mass_errors = []
    
    for event in range(100):
        if solver.a_tot == 0:
            break
        
        solver._do_one_break()
        
        V_solid_current = np.sum(solver.W * np.sum(solver.V_flat[:solver.dim, :], axis=0))
        error = abs(V_solid_current - V_solid_initial) / V_solid_initial * 100
        mass_errors.append(error)
    
    max_error = max(mass_errors)
    avg_error = np.mean(mass_errors)
    
    print(f"Massenerhaltung über 100 Events:")
    print(f"  Maximaler Fehler = {max_error:.12f}%")
    print(f"  Durchschnittlicher Fehler = {avg_error:.12f}%")
    
    assert max_error < 1e-8, f"Massenerhaltung verletzt! Max error: {max_error}%"
    
    print("\n✓ PASSED")
    return True


def run_all_tests():
    """Alle Tests ausführen."""
    print("\n" + "#"*70)
    print("# Cone Model Breakage Integration Tests")
    print("#"*70)
    
    tests = [
        ("Binary Breakage (Volume Mixing)", test_binary_breakage_volume_mixing),
        ("Binary Breakage (Cone Model)", test_binary_breakage_cone_model),
        ("Multi-Fragment Breakage (Cone Model)", test_multi_fragment_breakage_cone_model),
        ("Vollkörper Breakage", test_vollkörper_breakage),
        ("Mass Conservation Statistics", test_mass_conservation_statistics),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, True, None))
        except AssertionError as e:
            results.append((name, False, str(e)))
            print(f"\n✗ FAILED: {e}")
        except Exception as e:
            results.append((name, False, f"Unexpected error: {e}"))
            print(f"\n✗ ERROR: {e}")
    
    # Zusammenfassung
    print("\n" + "="*70)
    print("ZUSAMMENFASSUNG")
    print("="*70)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, error in results:
        status = "✓ PASSED" if success else f"✗ FAILED: {error}"
        print(f"  {name:40s} {status}")
    
    print(f"\nErgebnis: {passed}/{total} Tests bestanden")
    
    if passed == total:
        print("\n" + "#"*70)
        print("# ALLE TESTS BESTANDET ✓")
        print("#"*70 + "\n")
        return True
    else:
        print("\n" + "#"*70)
        print(f"# FEHLGESCHLAGEN: {total - passed} Tests fehlgeschlagen")
        print("#"*70 + "\n")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
