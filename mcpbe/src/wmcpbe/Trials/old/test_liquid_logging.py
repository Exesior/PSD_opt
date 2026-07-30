"""
DEBUG TEST MIT AUSFÜHRLICHEM LOGGING

Dieser Test protokolliert jeden Schritt der Liquid Volume Verteilung
und schreibt die Ergebnisse in eine Datei im Results-Ordner.
"""

import numpy as np
import sys
import os
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wmcpbe import MCPBESolver
from wmcpbe.mcpbe_nucleation import NucleationConfig, NucleationHandler

# ============================================================================
# KONFIGURATION
# ============================================================================

# Physikalische Parameter
PARTICLE_DIAMETER = 5e-5  # 50 µm
DROPLET_DIAMETER = 1e-5   # 10 µm
VOLUMETRIC_FLOW_RATE = 1e-11  # m³/s
DURATION = 2.0  # seconds
AGGLOMERATION_COEFFICIENT = 1e-8  # niedrige Agglomeration

# Solver-Parameter
N_PARTICLES_INITIAL = 100
DT_TARGET = 0.1  # s
T_TOTAL = 3.0  # s

# Output-Ordner
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")

# ============================================================================
# HELPER-FUNKTIONEN
# ============================================================================

def ensure_results_dir():
    """Erstellt den Results-Ordner falls er nicht existiert."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Erstelle Results-Ordner: {RESULTS_DIR}")

def generate_filename(prefix="test", extension="txt"):
    """Generiert einen eindeutigen Dateinamen mit Timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(RESULTS_DIR, f"{prefix}_{timestamp}.{extension}")

def calculate_liquid_in_system(solver):
    """Berechnet Gesamtflüssigkeit im System: Σ lv[k] × W[k] / Vc"""
    total = 0.0
    for k in range(solver.a_tot):
        if solver.W[k] > 0 and hasattr(solver, 'liquid_volume'):
            total += solver.liquid_volume[k] * solver.W[k] / solver.Vc
    return total

def calculate_total_weight(solver):
    """Berechnet Σ W[k] für alle aktiven Partikel."""
    return sum(solver.W[k] for k in range(solver.a_tot) if solver.W[k] > 0)

def analyze_liquid_distribution(solver, v_droplet):
    """Analysiert die Verteilung der liquid_volume Werte."""
    tol = 1e-25
    liq_arr = solver.liquid_volume[:solver.a_tot]
    w_arr = solver.W[:solver.a_tot]
    
    n_zero = sum(1 for lv in liq_arr if lv < tol)
    n_single = sum(1 for lv in liq_arr if abs(lv - v_droplet) < tol)
    n_double = sum(1 for lv in liq_arr if abs(lv - 2*v_droplet) < tol)
    n_triple = sum(1 for lv in liq_arr if abs(lv - 3*v_droplet) < tol)
    n_other = solver.a_tot - n_zero - n_single - n_double - n_triple
    
    # Gewichtete Beiträge
    w_single = sum(w_arr[i] for i in range(solver.a_tot) if abs(liq_arr[i] - v_droplet) < tol and w_arr[i] > 0)
    w_double = sum(w_arr[i] for i in range(solver.a_tot) if abs(liq_arr[i] - 2*v_droplet) < tol and w_arr[i] > 0)
    w_triple = sum(w_arr[i] for i in range(solver.a_tot) if abs(liq_arr[i] - 3*v_droplet) < tol and w_arr[i] > 0)
    
    return {
        'n_zero': n_zero,
        'n_single': n_single,
        'n_double': n_double,
        'n_triple': n_triple,
        'n_other': n_other,
        'w_single': w_single,
        'w_double': w_double,
        'w_triple': w_triple,
    }

# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def run_liquid_debug_test():
    """Führt den Debug-Test mit ausführlichem Logging durch."""
    
    ensure_results_dir()
    log_file = generate_filename("liquid_debug_log", "txt")
    summary_file = generate_filename("liquid_debug_summary", "txt")
    
    # Berechnungen
    particle_radius = PARTICLE_DIAMETER / 2.0
    particle_volume = (4.0 / 3.0) * np.pi * particle_radius ** 3
    
    droplet_radius = DROPLET_DIAMETER / 2.0
    droplet_volume = (4.0 / 3.0) * np.pi * droplet_radius ** 3
    
    expected_liquid_volume = VOLUMETRIC_FLOW_RATE * DURATION
    expected_droplet_count = expected_liquid_volume / droplet_volume
    
    # Log-File öffnen
    with open(log_file, 'w', encoding='utf-8') as f:
        
        # ====================================================================
        # HEADER MIT PARAMETERN
        # ====================================================================
        f.write("=" * 100 + "\n")
        f.write("LIQUID VOLUME DEBUG TEST - AUSFÜHRLICHES LOG\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("TEST-PARAMETER:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Particle Diameter: {PARTICLE_DIAMETER * 1e6:.1f} µm\n")
        f.write(f"Particle Volume: {particle_volume:.6e} m³\n")
        f.write(f"Droplet Diameter: {DROPLET_DIAMETER * 1e6:.1f} µm\n")
        f.write(f"Droplet Volume: {droplet_volume:.6e} m³\n")
        f.write(f"Volumetric Flow Rate: {VOLUMETRIC_FLOW_RATE:.6e} m³/s\n")
        f.write(f"Duration: {DURATION:.1f} s\n")
        f.write(f"Agglomeration Coefficient: {AGGLOMERATION_COEFFICIENT:.6e}\n")
        f.write(f"Initial n_comp: {N_PARTICLES_INITIAL}\n")
        f.write(f"dt_target: {DT_TARGET:.2f} s\n")
        f.write(f"t_total: {T_TOTAL:.1f} s\n")
        f.write("\n")
        
        f.write("ERWARTETE ERGEBNISSE:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total Liquid Volume: {expected_liquid_volume:.6e} m³\n")
        f.write(f"Total Droplets: {expected_droplet_count:.2e}\n")
        f.write("\n")
        
        f.write("=" * 100 + "\n")
        f.write("SIMULATIONSLAUF\n")
        f.write("=" * 100 + "\n\n")
        
        # ====================================================================
        # SOLVER INITIALISIERUNG (wie in test_nucleation_debug.py)
        # ====================================================================
        
        # Time vector
        t_write = 0.1
        t_vec = np.linspace(0.0, T_TOTAL, int(T_TOTAL / t_write) + 1)
        
        # Create solver without auto-initialization
        solver = MCPBESolver(
            dim=1,
            t_vec=t_vec,
            verbose=False,
            load_attr=False,
            init=False,
        )
        
        # Set process type and physics parameters
        solver.process_type = "agglomeration"
        solver.COLEVAL = 3  # Constant kernel
        solver.CORR_BETA = AGGLOMERATION_COEFFICIENT
        solver.recon_enable = False
        
        # Prepare particle state
        V_flat = np.zeros((2, N_PARTICLES_INITIAL), dtype=float)
        V_flat[0, :] = particle_volume
        V_flat[1, :] = particle_volume
        
        W_init = np.full(N_PARTICLES_INITIAL, 100.0, dtype=float)
        
        # Control volume
        n_phys_target = 1e6
        Vc_calc = np.sum(W_init) / n_phys_target
        solver.Vc = Vc_calc
        
        # Initialize solver
        solver._initialize_particles(
            init_Vc=False,
            V_flat=V_flat,
            W_init=W_init,
        )
        
        solver._init_lmc()
        solver._initialize_samplers()
        
        f.write("INITIAL STATE:\n")
        f.write(f"  a_tot: {solver.a_tot}\n")
        f.write(f"  Vc: {solver.Vc:.6e} m³\n")
        f.write(f"  Total W: {calculate_total_weight(solver):.2f}\n")
        f.write(f"  Physical particles: {calculate_total_weight(solver) / solver.Vc:.2e}\n")
        f.write(f"  Liquid in system: {calculate_liquid_in_system(solver):.6e} m³\n")
        f.write("\n")
        
        # ====================================================================
        # NUCLEATION KONFIGURIEREN
        # ====================================================================
        nuc_config = NucleationConfig(
            enabled=True,
            volumetric_flow_rate=VOLUMETRIC_FLOW_RATE,
            droplet_diameter=DROPLET_DIAMETER,
            liquid_addition_start=0.0,
            liquid_addition_duration=DURATION,
        )
        solver.nucleation = NucleationHandler(solver, nuc_config)
        
        f.write("NUCLEATION CONFIGURED:\n")
        f.write(f"  Enabled: True\n")
        f.write(f"  Window: [0.0s, {DURATION}s]\n")
        f.write("\n")
        
        # ====================================================================
        # TRACKING-VARIABLEN
        # ====================================================================
        event_counter = [0]
        cv_doubling_events = []
        agglomeration_events = []
        liquid_history = []
        
        original_double_cv = solver._maybe_double_control_volume
        original_do_agg = solver._do_one_agg
        
        liquid_added_manual = 0.0
        
        # ====================================================================
        # PATCH: CV-DOUBLING LOGGING
        # ====================================================================
        def logged_double_cv():
            old_a = solver.a_tot
            old_Vc = solver.Vc
            old_liq = calculate_liquid_in_system(solver)
            old_dist = analyze_liquid_distribution(solver, droplet_volume)
            
            result = original_double_cv()
            
            new_a = solver.a_tot
            new_Vc = solver.Vc
            new_liq = calculate_liquid_in_system(solver)
            new_dist = analyze_liquid_distribution(solver, droplet_volume)
            
            event = {
                'time': solver._elapsed_time,
                'events': event_counter[0],
                'old_a': old_a,
                'new_a': new_a,
                'old_Vc': old_Vc,
                'new_Vc': new_Vc,
                'old_liq': old_liq,
                'new_liq': new_liq,
                'delta_liq': new_liq - old_liq,
                'old_dist': old_dist,
                'new_dist': new_dist,
            }
            cv_doubling_events.append(event)
            
            f.write(f"\n{'='*80}\n")
            f.write(f"⚠️  CV-DOUBLING EVENT #{len(cv_doubling_events)}\n")
            f.write(f"{'='*80}\n")
            f.write(f"  Time: {event['time']:.6f} s (Event #{event['events']})\n")
            f.write(f"  a_tot: {event['old_a']} → {event['new_a']}\n")
            f.write(f"  Vc: {event['old_Vc']:.6e} → {event['new_Vc']:.6e}\n")
            f.write(f"  Liquid BEFORE: {event['old_liq']:.6e} m³\n")
            f.write(f"  Liquid AFTER:  {event['new_liq']:.6e} m³\n")
            f.write(f"  Δ Liquid: {event['delta_liq']:+.6e} m³ ({event['delta_liq']/event['old_liq']*100:+.2f}%)\n")
            f.write(f"\n  Distribution BEFORE:\n")
            f.write(f"    lv=0:     {event['old_dist']['n_zero']} Partikel\n")
            f.write(f"    lv=1×v_d: {event['old_dist']['n_single']} Partikel (ΣW={event['old_dist']['w_single']:.1f})\n")
            f.write(f"    lv=2×v_d: {event['old_dist']['n_double']} Partikel (ΣW={event['old_dist']['w_double']:.1f})\n")
            f.write(f"    lv=3×v_d: {event['old_dist']['n_triple']} Partikel (ΣW={event['old_dist']['w_triple']:.1f})\n")
            f.write(f"\n  Distribution AFTER:\n")
            f.write(f"    lv=0:     {event['new_dist']['n_zero']} Partikel\n")
            f.write(f"    lv=1×v_d: {event['new_dist']['n_single']} Partikel (ΣW={event['new_dist']['w_single']:.1f})\n")
            f.write(f"    lv=2×v_d: {event['new_dist']['n_double']} Partikel (ΣW={event['new_dist']['w_double']:.1f})\n")
            f.write(f"    lv=3×v_d: {event['new_dist']['n_triple']} Partikel (ΣW={event['new_dist']['w_triple']:.1f})\n")
            
            if abs(event['delta_liq']) > 1e-20:
                f.write(f"\n  ⚠️  WARNING: Liquid changed during CV-doubling!\n")
            
            return result
        
        solver._maybe_double_control_volume = logged_double_cv
        
        # ====================================================================
        # PATCH: AGGLOMERATION LOGGING
        # ====================================================================
        def logged_do_agg():
            if solver.a_tot < 2:
                return original_do_agg()
            
            # Vor Agg
            liq_before = calculate_liquid_in_system(solver)
            dist_before = analyze_liquid_distribution(solver, droplet_volume)
            
            result = original_do_agg()
            
            # Nach Agg
            liq_after = calculate_liquid_in_system(solver)
            dist_after = analyze_liquid_distribution(solver, droplet_volume)
            
            delta = liq_after - liq_before
            
            # Nur loggen wenn signifikante Änderung
            if abs(delta) > 1e-25 or len(agglomeration_events) < 20 or len(agglomeration_events) % 50 == 0:
                event = {
                    'event_num': event_counter[0],
                    'time': solver._elapsed_time,
                    'liq_before': liq_before,
                    'liq_after': liq_after,
                    'delta': delta,
                    'dist_before': dist_before,
                    'dist_after': dist_after,
                }
                agglomeration_events.append(event)
                
                if len(agglomeration_events) <= 20 or len(agglomeration_events) % 50 == 0:
                    f.write(f"\nAgg #{len(agglomeration_events):4d} @ t={event['time']:.6f}s: ")
                    f.write(f"Δliq = {event['delta']:+.6e} ({event['delta']/event['liq_before']*100:+.4f}%) ")
                    f.write(f"[{event['liq_before']:.6e} → {event['liq_after']:.6e}]\n")
                    
                    if abs(event['delta']) > droplet_volume * 0.1:
                        f.write(f"  ⚠️  LARGE CHANGE! Distribution: {event['dist_before']} → {event['dist_after']}\n")
            
            return result
        
        solver._do_one_agg = logged_do_agg
        
        # ====================================================================
        # SIMULATION DURCHFÜHREN
        # ====================================================================
        f.write("\n" + "=" * 100 + "\n")
        f.write("EVENT-LOG (ausgewählte Events)\n")
        f.write("=" * 100 + "\n")
        
        start_time = time.time()
        solver.solve(maxiter=int(1e9))
        elapsed = time.time() - start_time
        
        event_counter[0] = solver._iter_count
        
        f.write("\n")
        f.write("=" * 100 + "\n")
        f.write("SIMULATION ABGESCHLOSSEN\n")
        f.write("=" * 100 + "\n\n")
        
        # ====================================================================
        # ERGEBNISSE
        # ====================================================================
        nuc_stats = solver.nucleation.get_statistics()
        actual_liquid_added = nuc_stats['liquid_volume_added_total']
        actual_droplets = nuc_stats['droplets_added_total']
        
        liquid_in_system = calculate_liquid_in_system(solver)
        final_dist = analyze_liquid_distribution(solver, droplet_volume)
        
        error_added = (actual_liquid_added - expected_liquid_volume) / expected_liquid_volume * 100
        error_system = (liquid_in_system - expected_liquid_volume) / expected_liquid_volume * 100
        
        f.write("ERGEBNISSE:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Simulation Time: {elapsed:.2f} s\n")
        f.write(f"MC Events: {solver._iter_count}\n")
        f.write(f"Final a_tot: {solver.a_tot}\n")
        f.write(f"Final Vc: {solver.Vc:.6e}\n")
        f.write(f"Final Total W: {calculate_total_weight(solver):.2f}\n")
        f.write("\n")
        
        f.write("Liquid Addition:\n")
        f.write(f"  Expected:      {expected_liquid_volume:.6e} m³\n")
        f.write(f"  Added (stat):  {actual_liquid_added:.6e} m³ ({error_added:+.4f}%)\n")
        f.write(f"  In system:     {liquid_in_system:.6e} m³ ({error_system:+.4f}%)\n")
        f.write("\n")
        
        f.write(f"Droplets:\n")
        f.write(f"  Expected: {expected_droplet_count:.2e}\n")
        f.write(f"  Actual:   {actual_droplets:.2e} ({actual_droplets/expected_droplet_count*100:.2f}%)\n")
        f.write("\n")
        
        f.write(f"Final Liquid Distribution:\n")
        f.write(f"  lv=0:     {final_dist['n_zero']} Partikel\n")
        f.write(f"  lv=1×v_d: {final_dist['n_single']} Partikel (ΣW={final_dist['w_single']:.1f})\n")
        f.write(f"  lv=2×v_d: {final_dist['n_double']} Partikel (ΣW={final_dist['w_double']:.1f})\n")
        f.write(f"  lv=3×v_d: {final_dist['n_triple']} Partikel (ΣW={final_dist['w_triple']:.1f})\n")
        f.write(f"  andere:   {final_dist['n_other']} Partikel\n")
        f.write("\n")
        
        f.write("CV-DOUBLING EVENTS:\n")
        f.write(f"  Total: {len(cv_doubling_events)}\n")
        for i, evt in enumerate(cv_doubling_events):
            f.write(f"  #{i+1}: t={evt['time']:.4f}s, a={evt['old_a']}→{evt['new_a']}, ")
            f.write(f"Vc={evt['old_Vc']:.2e}→{evt['new_Vc']:.2e}, ")
            f.write(f"liq={evt['old_liq']:.6e}→{evt['new_liq']:.6e} ({evt['delta_liq']:+.2f}%)\n")
        f.write("\n")
        
        f.write("AGGLOMERATION EVENTS (Zusammenfassung):\n")
        f.write(f"  Total logged: {len(agglomeration_events)}\n")
        
        # Größte Änderungen finden
        if agglomeration_events:
            max_delta = max(agglomeration_events, key=lambda x: abs(x['delta']))
            f.write(f"  Largest Δliq: #{max_delta['event_num']} @ t={max_delta['time']:.4f}s, ")
            f.write(f"Δ={max_delta['delta']:.6e} ({max_delta['delta']/max_delta['liq_before']*100:+.4f}%)\n")
        f.write("\n")
        
        # ====================================================================
        # FAZIT
        # ====================================================================
        f.write("=" * 100 + "\n")
        f.write("FAZIT\n")
        f.write("=" * 100 + "\n\n")
        
        if abs(error_system) < 5:
            f.write("✓ TEST BESTANDEN: Error < 5%\n")
        else:
            f.write(f"✗ TEST FEHLGESCHLAGEN: Error = {error_system:+.2f}%\n\n")
            
            f.write("MÖGLICHE URSACHEN:\n")
            
            # Check CV-Doubling
            for evt in cv_doubling_events:
                if abs(evt['delta_liq']) > expected_liquid_volume * 0.01:
                    f.write(f"  ⚠️  CV-Doubling #{cv_doubling_events.index(evt)+1}: ")
                    f.write(f"Unerwartete Liquid-Änderung von {evt['delta_liq']:.6e} m³\n")
            
            # Check Agglomeration
            large_agg = [e for e in agglomeration_events if abs(e['delta']) > droplet_volume * 0.5]
            if large_agg:
                f.write(f"  ⚠️  {len(large_agg)} Agglomeration-Events mit großer Liquid-Änderung (>0.5×v_d)\n")
            
            # Check Nucleation
            if abs(error_added) > 5:
                f.write(f"  ⚠️  Nucleation Statistics fehlerhaft: {error_added:+.4f}%\n")
            
            if error_system > 100:
                f.write(f"\n  HINWEIS: Viel zu viel Flüssigkeit im System!\n")
                f.write(f"  → Wahrscheinlich wird liquid_volume bei CV-Doubling falsch dupliziert\n")
                f.write(f"  → ODER Nucleation verteilt Tropfen mehrfach\n")
            elif error_system < -100:
                f.write(f"\n  HINWEIS: Viel zu wenig Flüssigkeit im System!\n")
                f.write(f"  → Wahrscheinlich geht liquid_volume bei Agglomeration verloren\n")
                f.write(f"  → ODER Parent-Skalierung ist falsch\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("ENDE DES LOGS\n")
        f.write("=" * 100 + "\n")
    
    # ========================================================================
    # SUMMARY FILE (kurze Zusammenfassung)
    # ========================================================================
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("LIQUID VOLUME DEBUG TEST - ZUSAMMENFASSUNG\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Log-File: {os.path.basename(log_file)}\n\n")
        
        f.write("PARAMETER:\n")
        f.write(f"  CORR_BETA = {AGGLOMERATION_COEFFICIENT:.1e}\n")
        f.write(f"  Duration  = {DURATION} s\n")
        f.write("\n")
        
        f.write("ERGEBNISSE:\n")
        f.write(f"  Error (added):   {error_added:+.4f}%\n")
        f.write(f"  Error (system):  {error_system:+.4f}%\n")
        f.write(f"  CV-Doublings:    {len(cv_doubling_events)}\n")
        f.write(f"  MC Events:       {solver._iter_count}\n\n")
        
        if abs(error_system) < 5:
            f.write("STATUS: ✓ BESTANDEN\n")
        else:
            f.write(f"STATUS: ✗ FEHLGESCHLAGEN ({error_system:+.2f}%)\n")
    
    # Console Output
    print("=" * 80)
    print("TEST ABGESCHLOSSEN")
    print("=" * 80)
    print(f"\nLog-File: {log_file}")
    print(f"Summary:  {summary_file}")
    print(f"\nError (added):   {error_added:+.4f}%")
    print(f"Error (system):  {error_system:+.4f}%")
    print(f"CV-Doublings:    {len(cv_doubling_events)}")
    print(f"MC Events:       {solver._iter_count}")
    
    if abs(error_system) < 5:
        print("\n✓ TEST BESTANDEN")
    else:
        print(f"\n✗ TEST FEHLGESCHLAGEN")
        print("→ Siehe Log-File für Details!")
    
    return log_file, summary_file

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    run_liquid_debug_test()
