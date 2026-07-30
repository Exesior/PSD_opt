"""
EINFACHER LIQUID VOLUME DEBUG TEST

Schreibt detaillierte Logs ohne komplexes Patching.
"""

import numpy as np
import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wmcpbe import MCPBESolver
from wmcpbe.mcpbe_nucleation import NucleationConfig, NucleationHandler

# ============================================================================
# KONFIGURATION
# ============================================================================

PARTICLE_DIAMETER = 5e-5  # 50 µm
DROPLET_DIAMETER = 1e-5   # 10 µm
VOLUMETRIC_FLOW_RATE = 1e-11  # m³/s
DURATION = 2.0  # seconds
AGGLOMERATION_COEFFICIENT = 1e-8

N_PARTICLES_INITIAL = 100
T_TOTAL = 3.0

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")

# ============================================================================
# HELPER
# ============================================================================

def ensure_results_dir():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

def generate_filename(prefix="test", extension="txt"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(RESULTS_DIR, f"{prefix}_{timestamp}.{extension}")

def calc_liquid(solver):
    """Σ lv[k] × W[k] (OHNE /Vc!)"""
    total = 0.0
    for k in range(solver.a_tot):
        if solver.W[k] > 0 and hasattr(solver, 'liquid_volume'):
            total += solver.liquid_volume[k] * solver.W[k]
    return total

def analyze_dist(solver, v_d):
    """Analysiert liquid_volume Verteilung."""
    tol = 1e-25
    liq = solver.liquid_volume[:solver.a_tot]
    w = solver.W[:solver.a_tot]
    
    result = {'n_zero': 0, 'n_1x': 0, 'n_2x': 0, 'n_3x': 0, 'n_other': 0,
              'w_1x': 0.0, 'w_2x': 0.0, 'w_3x': 0.0}
    
    for i in range(solver.a_tot):
        if w[i] <= 0:
            continue
        if liq[i] < tol:
            result['n_zero'] += 1
        elif abs(liq[i] - v_d) < tol:
            result['n_1x'] += 1
            result['w_1x'] += w[i]
        elif abs(liq[i] - 2*v_d) < tol:
            result['n_2x'] += 1
            result['w_2x'] += w[i]
        elif abs(liq[i] - 3*v_d) < tol:
            result['n_3x'] += 1
            result['w_3x'] += w[i]
        else:
            result['n_other'] += 1
    
    return result

# ============================================================================
# MAIN
# ============================================================================

def run_test():
    ensure_results_dir()
    log_file = generate_filename("liquid_log", "txt")
    summary_file = generate_filename("liquid_summary", "txt")
    
    # Berechnungen
    particle_radius = PARTICLE_DIAMETER / 2.0
    particle_volume = (4.0 / 3.0) * np.pi * particle_radius ** 3
    
    droplet_radius = DROPLET_DIAMETER / 2.0
    droplet_volume = (4.0 / 3.0) * np.pi * droplet_radius ** 3
    
    expected_liquid = VOLUMETRIC_FLOW_RATE * DURATION
    expected_droplets = expected_liquid / droplet_volume
    
    # Solver Setup (wie test_nucleation_debug.py)
    t_write = 0.1
    t_vec = np.linspace(0.0, T_TOTAL, int(T_TOTAL / t_write) + 1)
    
    solver = MCPBESolver(dim=1, t_vec=t_vec, verbose=False, load_attr=False, init=False)
    solver.process_type = "agglomeration"
    solver.COLEVAL = 3
    solver.CORR_BETA = AGGLOMERATION_COEFFICIENT
    solver.recon_enable = False
    
    V_flat = np.zeros((2, N_PARTICLES_INITIAL), dtype=float)
    V_flat[0, :] = particle_volume
    V_flat[1, :] = particle_volume
    
    W_init = np.full(N_PARTICLES_INITIAL, 100.0, dtype=float)
    # Vc=1 für Nucleation-Tests (vermeidet Skalierungsprobleme)
    solver.Vc = 1.0
    
    solver._initialize_particles(init_Vc=False, V_flat=V_flat, W_init=W_init)
    solver._init_lmc()
    solver._initialize_samplers()
    
    # Nucleation
    nuc_config = NucleationConfig(
        enabled=True,
        volumetric_flow_rate=VOLUMETRIC_FLOW_RATE,
        droplet_diameter=DROPLET_DIAMETER,
        liquid_addition_start=0.0,
        liquid_addition_duration=DURATION,
    )
    solver.nucleation = NucleationHandler(solver, nuc_config)
    
    # Logging während Simulation
    cv_events = []
    agg_events = []
    liquid_history = []
    
    # Snapshot vor Simulation
    liquid_history.append({
        'time': 0.0,
        'events': 0,
        'a_tot': solver.a_tot,
        'Vc': solver.Vc,
        'liquid': calc_liquid(solver),
        'dist': analyze_dist(solver, droplet_volume),
    })
    
    # Original-Funktionen speichern
    orig_double_cv = solver._maybe_double_control_volume
    orig_do_agg = solver._do_one_agg
    
    # CV-Doubling wrappen
    def logged_double_cv(et, ic):
        before_liq = calc_liquid(solver)
        before_dist = analyze_dist(solver, droplet_volume)
        before_a = solver.a_tot
        before_Vc = solver.Vc
        
        orig_double_cv(et, ic)
        
        after_liq = calc_liquid(solver)
        after_dist = analyze_dist(solver, droplet_volume)
        
        if solver.a_tot != before_a:  # Tatsächlich gedoubelt
            cv_events.append({
                'time': et,
                'events': ic,
                'before_a': before_a,
                'after_a': solver.a_tot,
                'before_Vc': before_Vc,
                'after_Vc': solver.Vc,
                'before_liq': before_liq,
                'after_liq': after_liq,
                'delta_liq': after_liq - before_liq,
                'before_dist': before_dist,
                'after_dist': after_dist,
            })
    
    solver._maybe_double_control_volume = logged_double_cv
    
    # Agglomeration wrappen (nur jede 50te loggen für Performance)
    agg_count = [0]
    
    def logged_do_agg():
        agg_count[0] += 1
        
        before_liq = calc_liquid(solver)
        orig_do_agg()
        after_liq = calc_liquid(solver)
        
        delta = after_liq - before_liq
        
        # Logge erste 20 und dann jede 50ste
        if agg_count[0] <= 20 or agg_count[0] % 50 == 0 or abs(delta) > droplet_volume * 0.1:
            agg_events.append({
                'num': agg_count[0],
                'time': solver._elapsed,
                'before': before_liq,
                'after': after_liq,
                'delta': delta,
            })
    
    solver._do_one_agg = logged_do_agg
    
    # SIMULATION
    start_time = time.time()
    solver.solve(maxiter=int(1e9))
    elapsed = time.time() - start_time
    
    # Final snapshot
    liquid_history.append({
        'time': solver._elapsed,
        'events': solver._iter_count,
        'a_tot': solver.a_tot,
        'Vc': solver.Vc,
        'liquid': calc_liquid(solver),
        'dist': analyze_dist(solver, droplet_volume),
    })
    
    # ERGEBNISSE
    nuc_stats = solver.nucleation.get_statistics()
    actual_liquid = nuc_stats['liquid_volume_added_total']
    actual_droplets = nuc_stats['droplets_added_total']
    
    liquid_in_system = calc_liquid(solver)
    final_dist = analyze_dist(solver, droplet_volume)
    
    error_added = (actual_liquid - expected_liquid) / expected_liquid * 100
    error_system = (liquid_in_system - expected_liquid) / expected_liquid * 100
    
    # LOG FILE SCHREIBEN
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("LIQUID VOLUME DEBUG TEST\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("PARAMETER:\n")
        f.write(f"  Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"  Particle Diameter: {PARTICLE_DIAMETER*1e6:.1f} µm\n")
        f.write(f"  Droplet Diameter: {DROPLET_DIAMETER*1e6:.1f} µm\n")
        f.write(f"  Flow Rate: {VOLUMETRIC_FLOW_RATE:.6e} m³/s\n")
        f.write(f"  Duration: {DURATION:.1f} s\n")
        f.write(f"  CORR_BETA: {AGGLOMERATION_COEFFICIENT:.6e}\n")
        f.write(f"  Initial n_comp: {N_PARTICLES_INITIAL}\n")
        f.write(f"  Expected Liquid: {expected_liquid:.6e} m³\n")
        f.write(f"  Expected Droplets: {expected_droplets:.2e}\n")
        f.write("\n")
        
        f.write("=" * 100 + "\n")
        f.write("INITIAL STATE\n")
        f.write("=" * 100 + "\n")
        f.write(f"  a_tot: {liquid_history[0]['a_tot']}\n")
        f.write(f"  Vc: {liquid_history[0]['Vc']:.6e}\n")
        f.write(f"  Liquid: {liquid_history[0]['liquid']:.6e}\n")
        f.write("\n")
        
        f.write("=" * 100 + "\n")
        f.write("CV-DOUBLING EVENTS\n")
        f.write("=" * 100 + "\n")
        if cv_events:
            for i, evt in enumerate(cv_events):
                f.write(f"\nEvent #{i+1}:\n")
                f.write(f"  Time: {evt['time']:.6f} s (MC Event #{evt['events']})\n")
                f.write(f"  a_tot: {evt['before_a']} → {evt['after_a']}\n")
                f.write(f"  Vc: {evt['before_Vc']:.6e} → {evt['after_Vc']:.6e}\n")
                f.write(f"  Liquid: {evt['before_liq']:.6e} → {evt['after_liq']:.6e}\n")
                f.write(f"  Δ Liquid: {evt['delta_liq']:+.6e} ({evt['delta_liq']/evt['before_liq']*100:+.2f}%)\n")
                
                if abs(evt['delta_liq']) > 1e-20:
                    f.write(f"  ⚠️  WARNING: Liquid changed!\n")
        else:
            f.write("  Keine CV-Doubling Events.\n")
        f.write("\n")
        
        f.write("=" * 100 + "\n")
        f.write("AGGLOMERATION EVENTS (ausgewählte)\n")
        f.write("=" * 100 + "\n")
        if agg_events:
            f.write(f"  Total logged: {len(agg_events)} von {agg_count[0]}\n\n")
            for evt in agg_events[:20]:  # Erste 20
                f.write(f"  Agg #{evt['num']:4d} @ t={evt['time']:.6f}s: ")
                f.write(f"Δliq = {evt['delta']:+.6e} ({evt['delta']/evt['before']*100:+.4f}%)\n")
            
            if len(agg_events) > 20:
                f.write(f"\n  ... ({len(agg_events) - 20} weitere nicht gezeigt)\n")
            
            # Größte Änderung
            max_evt = max(agg_events, key=lambda x: abs(x['delta']))
            f.write(f"\n  Größte Änderung: Agg #{max_evt['num']} @ t={max_evt['time']:.6f}s\n")
            f.write(f"  Δ = {max_evt['delta']:.6e} ({max_evt['delta']/max_evt['before']*100:+.4f}%)\n")
        else:
            f.write("  Keine Agglomeration Events geloggt.\n")
        f.write("\n")
        
        f.write("=" * 100 + "\n")
        f.write("FINAL STATE\n")
        f.write("=" * 100 + "\n")
        f.write(f"  Simulation Time: {elapsed:.2f} s\n")
        f.write(f"  MC Events: {solver._iter_count}\n")
        f.write(f"  Final a_tot: {solver.a_tot}\n")
        f.write(f"  Final Vc: {solver.Vc:.6e}\n")
        f.write(f"  Final Liquid (in system): {liquid_in_system:.6e} m³\n")
        f.write("\n")
        
        f.write("Liquid Addition:\n")
        f.write(f"  Expected:     {expected_liquid:.6e} m³\n")
        f.write(f"  Added (stat): {actual_liquid:.6e} m³ ({error_added:+.4f}%)\n")
        f.write(f"  In system:    {liquid_in_system:.6e} m³ ({error_system:+.4f}%)\n")
        f.write("\n")
        
        f.write(f"Droplets: Expected={expected_droplets:.2e}, Actual={actual_droplets:.2e} ({actual_droplets/expected_droplets*100:.2f}%)\n")
        f.write("\n")
        
        f.write("Liquid Distribution:\n")
        f.write(f"  lv=0:      {final_dist['n_zero']} Partikel\n")
        f.write(f"  lv=1×v_d:  {final_dist['n_1x']} Partikel (ΣW={final_dist['w_1x']:.1f})\n")
        f.write(f"  lv=2×v_d:  {final_dist['n_2x']} Partikel (ΣW={final_dist['w_2x']:.1f})\n")
        f.write(f"  lv=3×v_d:  {final_dist['n_3x']} Partikel (ΣW={final_dist['w_3x']:.1f})\n")
        f.write(f"  andere:    {final_dist['n_other']} Partikel\n")
        f.write("\n")
        
        f.write("=" * 100 + "\n")
        f.write("FAZIT\n")
        f.write("=" * 100 + "\n\n")
        
        if abs(error_system) < 5:
            f.write("✓ TEST BESTANDEN: Error < 5%\n")
        else:
            f.write(f"✗ FEHLGESCHLAGEN: Error = {error_system:+.2f}%\n\n")
            
            if error_system > 100:
                f.write("DIAGNOSE: Viel zu viel Flüssigkeit!\n")
                f.write("Mögliche Ursachen:\n")
                f.write("  - CV-Doubling dupliziert liquid_volume falsch\n")
                f.write("  - Nucleation verteilt Tropfen mehrfach\n")
            elif error_system < -100:
                f.write("DIAGNOSE: Viel zu wenig Flüssigkeit!\n")
                f.write("Mögliche Ursachen:\n")
                f.write("  - Agglomeration verliert liquid_volume\n")
                f.write("  - Parent-Skalierung falsch\n")
            
            # Check CV-Doubling
            for evt in cv_events:
                if abs(evt['delta_liq']) > expected_liquid * 0.01:
                    f.write(f"\n⚠️  CV-Doubling #{cv_events.index(evt)+1}: Unerwartete Änderung {evt['delta_liq']:.6e}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("ENDE\n")
        f.write("=" * 100 + "\n")
    
    # SUMMARY FILE
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("LIQUID VOLUME TEST - SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Log: {os.path.basename(log_file)}\n\n")
        f.write(f"CORR_BETA: {AGGLOMERATION_COEFFICIENT:.1e}\n")
        f.write(f"Duration: {DURATION} s\n\n")
        f.write(f"Error (added):   {error_added:+.4f}%\n")
        f.write(f"Error (system):  {error_system:+.4f}%\n")
        f.write(f"CV-Doublings:    {len(cv_events)}\n")
        f.write(f"MC Events:       {solver._iter_count}\n\n")
        
        if abs(error_system) < 5:
            f.write("STATUS: ✓ BESTANDEN\n")
        else:
            f.write(f"STATUS: ✗ FEHLGESCHLAGEN ({error_system:+.2f}%)\n")
    
    # Console
    print("=" * 80)
    print("TEST ABGESCHLOSSEN")
    print("=" * 80)
    print(f"\nLog: {log_file}")
    print(f"Summary: {summary_file}")
    print(f"\nError (added):   {error_added:+.4f}%")
    print(f"Error (system):  {error_system:+.4f}%")
    print(f"CV-Doublings:    {len(cv_events)}")
    print(f"MC Events:       {solver._iter_count}")
    
    if abs(error_system) < 5:
        print("\n✓ TEST BESTANDEN")
    else:
        print(f"\n✗ FEHLGESCHLAGEN")
        print("→ Siehe Log-File!")
    
    return log_file, summary_file

if __name__ == "__main__":
    run_test()
