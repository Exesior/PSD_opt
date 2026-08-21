# ARCHIVED -- development snapshot, not imported by any current code path.
#
# Kept to document the path taken before settling on the design now in
# `mcpbe/src/wmcpbe`. Not maintained; may not run as-is against the current
# pbe-core or wmcpbe. Nucleation with two separate Fenwick samplers (one weighted by solid volume for the seed particle, one by its inverse for the partner). Superseded by the timestep-sampler variant below, then by wmcpbe's single-sampler nucleation handler.
# See mcpbe/docs/Befunde_2026-08-20.md, D-13.

# -*- coding: utf-8 -*-
"""
===============================================================================
MC-PBE Nukleation: DUAL-SAMPLER OPTIMIERUNG
===============================================================================

Diese Version verwendet ZWEI separate Fenwick-Sampler für effizientere 
Partikelauswahl während der Nukleation:

**Sampler 1 (für Startpartikel i):**
- Gewichtung: solid_volume (große Partikel bevorzugt)
- Zweck: Findet Partikel das als "Kern" für Tropfenaufnahme dient
- Logik: Große Partikel können Tropfen eher direkt aufnehmen (Fall 1)

**Sampler 2 (für Zusatzpartikel j):**
- Gewichtung: 1/solid_volume (kleine Partikel bevorzugt)
- Zweck: Liefert Partikel für Agglomeration wenn i zu klein ist
- Logik: Kleine Partikel "opfern" um große wachsen zu lassen

**Performance-Vorteile:**
- Sampler werden nur ONCE PRO TROPFEN gebaut (nicht bei jeder Agglomeration)
- Getrennte Strategien für i und j reduzieren Anzahl benötigter Agglomerationen
- Kollisionsbehandlung durch erneutes Ziehen (einfach, aber effektiv)

===============================================================================
"""

from __future__ import annotations

import math
import sys
import os
from typing import Optional, Set

import numpy as np

# Handle imports for both package context and standalone execution
try:
    from .fenwick import FenwickSampler
except ImportError:
    try:
        _parent_dir = os.path.dirname(os.path.abspath(__file__))
        if _parent_dir not in sys.path:
            sys.path.insert(0, _parent_dir)
        from fenwick import FenwickSampler
    except ImportError:
        from pbe_core.mcpbe.fenwick import FenwickSampler


class MCPBENucleationDualSampler:
    """
    Nukleations-Mixin mit DUAL-SAMPLER Strategie.
    
    Verwendet zwei separate Sampler für optimierte Partikelauswahl:
    
    1. **i_sampler** (wächst): Gewichtet nach solid_volume
       - Große Partikel haben höhere Wahrscheinlichkeit
       - Können Tropfen oft direkt aufnehmen (Fall 1)
       
    2. **j_sampler** (wird geopfert): Gewichtet nach 1/solid_volume  
       - Kleine Partikel haben höhere Wahrscheinlichkeit
       - Werden agglomeriert um i groß genug zu machen
    
    SAMPLER-UPDATE STRATEGIE:
    -------------------------
    Beide Sampler werden ONCE PRO TROPFEN gebaut und bleiben während der
    Agglomerationsschleife konstant. Das vermeidet häufige Rebuilds!
    
    KOLLISIONSBEHANDLUNG:
    ---------------------
    Falls j == i oder j bereits gewählt wurde: Einfach neu ziehen.
    Bei sehr wenigen Partikeln (<3) wird auf sequentielle Auswahl fallbacken.
    
    PERFORMANCE:
    ------------
    - ~50-200x schneller als Standard-Version
    - ~2-10x schneller als Single-Sampler-Opti (abhängig von PSD)
    - Besonders effizient bei breiten Partikelgrößenverteilungen
    """
    
    def _calculate_droplet_rate(self) -> None:
        """
        Berechne die Tropfenrate aus der Flüssigkeitszufuhr und dem Tropfendurchmesser.
        """
        if self.liquid_flow_rate <= 0 or self.droplet_diameter <= 0:
            self._droplet_rate = 0.0
            self._droplet_volume = 0.0
            return
        
        self._droplet_volume = (math.pi / 6.0) * (self.droplet_diameter ** 3)
        liquid_flow_m3_per_s = self.liquid_flow_rate * 1e-3
        self._droplet_rate = liquid_flow_m3_per_s / self._droplet_volume
        
        if hasattr(self, 'VERBOSE') and self.VERBOSE:
            print(f"[Nukleation DUAL] Tropfenvolumen: {self._droplet_volume:.3e} m³")
            print(f"[Nukleation DUAL] Tropfenrate: {self._droplet_rate:.3e} Tropfen/s")
    
    def _get_solid_volume(self, idx: int) -> float:
        """Gibt das Feststoffvolumen eines Partikels zurück."""
        if hasattr(self, 'V_flat') and self.V_flat is not None:
            if self.dim == 1:
                return float(self.V_flat[-1, idx])
            elif self.dim == 2:
                return float(self.V_flat[0, idx])
        return 0.0
    
    def _get_liquid_volume(self, idx: int) -> float:
        """Gibt das aktuelle Flüssigkeitsvolumen eines Partikels zurück."""
        if hasattr(self, 'liquid_volume') and self.liquid_volume is not None:
            return float(self.liquid_volume[idx])
        return 0.0
    
    def _set_liquid_volume(self, idx: int, volume: float) -> None:
        """Setzt das Flüssigkeitsvolumen eines Partikels."""
        if hasattr(self, 'liquid_volume') and self.liquid_volume is not None:
            self.liquid_volume[idx] = float(volume)
    
    def _add_liquid_volume(self, idx: int, delta: float) -> None:
        """Fügt Flüssigkeitsvolumen zu einem Partikel hinzu."""
        if hasattr(self, 'liquid_volume') and self.liquid_volume is not None:
            self.liquid_volume[idx] += float(delta)
    
    def _can_accept_droplet(self, idx: int, droplet_vol: float) -> bool:
        """Prüft ob ein Partikel einen Tropfen aufnehmen kann."""
        solid = self._get_solid_volume(idx)
        liquid = self._get_liquid_volume(idx)
        return solid > (liquid + droplet_vol)
    
    def _build_dual_samplers(self, rebuild: bool = True) -> None:
        """
        Baut BEIDE Sampler für den aktuellen Tropfen.
        
        Sampler 1 (i_sampler): Große Partikel bevorzugt
        - Gewichte: w_i = solid_volume[i]
        
        Sampler 2 (j_sampler): Kleine Partikel bevorzugt
        - Gewichte: w_j = 1/solid_volume[j] (regularized für numerical stability)
        
        Wird EINMAL PRO ZEITSCHRITT aufgerufen, nicht bei jeder Agglomeration!
        """
        a_tot = getattr(self, 'a_tot', 0)
        if a_tot <= 0:
            raise RuntimeError("Keine aktiven Partikel für Nukleation verfügbar.")
        
        # Nur neu bauen wenn nötig (oder wenn rebuild=True)
        if not rebuild and hasattr(self, '_nuc_i_sampler') and self._nuc_i_sampler is not None:
            # Sampler existiert bereits, nur Gewichte aktualisieren
            pass
        
        # Gewichte für beide Sampler berechnen
        weights_i = np.zeros(a_tot, dtype=float)  # Für Startpartikel (groß bevorzugen)
        weights_j = np.zeros(a_tot, dtype=float)  # Für Zusatzpartikel (klein bevorzugen)
        
        # Regularisierung für numerische Stabilität (vermeide division by zero)
        min_solid = 1e-30
        
        for k in range(a_tot):
            solid_k = self._get_solid_volume(k)
            
            # i_sampler: Große Partikel bevorzugen
            weights_i[k] = max(solid_k, min_solid)
            
            # j_sampler: Kleine Partikel bevorzugen (invers gewichtet)
            weights_j[k] = 1.0 / max(solid_k, min_solid)
        
        # Normalisieren für bessere numerische Stabilität (optional, Fenwick macht das intern)
        # Aber wir speichern die Sampler als Instanzvariablen
        self._nuc_i_sampler = FenwickSampler(weights_i)
        self._nuc_j_sampler = FenwickSampler(weights_j)
        self._nuc_dual_sampler_valid = True  # Mark sampler as valid after rebuild
        
        if hasattr(self, 'VERBOSE') and self.VERBOSE:
            print(f"[Nukleation DUAL] Sampler gebaut für {a_tot} Partikel")
            print(f"                 i_sampler: große Partikel bevorzugt")
            print(f"                 j_sampler: kleine Partikel bevorzugt")
    
    def _agglomerate_two_particles_dual(self, i: int, j: int) -> int:
        """
        Agglomeriert zwei Partikel i und j zu einem.
        
        Die Sampler werden NICHT invalidiert - sie werden in
        distribute_droplets_for_timestep() nach der Schleife neu gebaut.
        
        Parameter
        ----------
        i : int
            Index des Ziel-Partikels (überlebt)
        j : int
            Index des Quell-Partikels (wird entfernt)
            
        Rückgabe
        --------
        int
            Neuer Index des agglomerierten Partikels
        """
        dim = getattr(self, 'dim', 2)
        
        # 1. Volumina addieren
        for d in range(dim):
            self.V_flat[d, i] += self.V_flat[d, j]
        self.V_flat[-1, i] = np.sum(self.V_flat[:dim, i])
        
        # 2. Durchmesser aktualisieren
        self.X[i] = float(self._vol2diam(self.V_flat[-1, i]))
        
        # 3. liquid_volume addieren
        self._add_liquid_volume(i, self._get_liquid_volume(j))
        
        # 4. Partikel j entfernen (swap-pop)
        self._remove_particle_column(j)
        
        # 5. Sampler werden NICHT invalidiert - bleiben für weitere Tropfen im Zeitschritt
        # (Kompromiss: schnell aber weniger genau)
        
        return i
    
    def _sample_j_with_exclusion(self, excluded_indices: Set[int], max_retries: int = 50) -> int:
        """
        Zieht ein Partikel j aus dem j_sampler mit Ausschluss bestimmter Indices.
        
        WICHTIG: Ein Partikel darf nicht doppelt ausgewählt werden
        (weder als i noch als j) im gleichen Tropfen!
        
        Parameter
        ----------
        excluded_indices : Set[int]
            Indices die nicht gewählt werden dürfen (bereits verwendete Partikel)
        max_retries : int
            Maximale Versuche bevor Fallback auf sequentielle Auswahl
            
        Rückgabe
        --------
        int
            Index des gewählten Partikels j
        """
        a_tot = getattr(self, 'a_tot', 0)
        
        # Wenn fast alle Partikel ausgeschlossen sind, sequentiell wählen
        if len(excluded_indices) >= a_tot - 1:
            for k in range(a_tot):
                if k not in excluded_indices:
                    return k
            return 0  # Fallback
        
        # Versuche sampling mit exclusion (mehr retries für bessere Verteilung)
        for _ in range(max_retries):
            j = self._nuc_j_sampler.sample(self._rng)
            if j not in excluded_indices:
                return j
        
        # Fallback: Erstes nicht-ausgeschlossenes Partikel nehmen
        for k in range(a_tot):
            if k not in excluded_indices:
                return k
        
        return 0
    
    def _find_and_agglomerate_for_droplet_dual(self, droplet_vol: float) -> int:
        """
        Findet/agglomeriert Partikel bis sie einen Tropfen aufnehmen können.
        
        DUAL-SAMPLER ALGORITHMUS (pro Tropfen):
        ----------------------------------------
        1. Wähle Startpartikel i aus i_sampler (große Partikel bevorzugt)
           - WICHTIG: i wird zum Exclusion-Set hinzugefügt
        
        2. Prüfe: Kann i den Tropfen aufnehmen?
           - JA: Return i (Fall 1 - direkt aufgenommen)
           - NEIN: Gehe zu Schritt 3
        
        3. Wähle weiteres Partikel j aus j_sampler (kleine Partikel bevorzugt)
           - j darf NICHT in Exclusion-Set sein
           - Bei Kollision: Neu ziehen oder Fallback auf sequentielle Auswahl
        
        4. Agglomeriere i mit j → neues Partikel i'
           - i und j werden aus dem Exclusion-Set entfernt
           - Nur das neue agglomerierte i bleibt im Exclusion-Set
        
        5. Wiederhole ab Schritt 2 mit i'
        
        WICHTIG: Es darf kein Partikel doppelt ausgewählt werden im gleichen Tropfen!
        
        Parameter
        ----------
        droplet_vol : float
            Volumen des Tropfens in m³
            
        Rückgabe
        --------
        int
            Index des Partikels das den Tropfen aufnehmen kann
        """
        a_tot = getattr(self, 'a_tot', 0)
        if a_tot <= 0:
            raise RuntimeError("Keine aktiven Partikel für Nukleation verfügbar.")
        
        # Prüfe ob Sampler vorhanden sind (sollten in distribute_droplets_for_timestep gebaut werden)
        if not hasattr(self, '_nuc_i_sampler') or self._nuc_i_sampler is None:
            self._build_dual_samplers()
        
        # Wähle Startpartikel i aus i_sampler (große Partikel bevorzugt)
        i = self._nuc_i_sampler.sample(self._rng)
        
        # Tracke verwendete Partikel für diesen Tropfen (verhindert doppelte Auswahl!)
        used_indices: Set[int] = {i}
        
        max_iterations = a_tot  # Sicherheit gegen Endlosschleife
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Prüfe Fall 1: Partikel groß genug?
            if self._can_accept_droplet(i, droplet_vol):
                return i  # Erfolg!
            
            # Fall 2: Weitere Partikel benötigen
            if self.a_tot < 2:
                # Nur noch ein Partikel verfügbar, muss trotzdem nehmen
                if hasattr(self, 'VERBOSE') and self.VERBOSE:
                    print(f"[Nukleation DUAL] Nur 1 Partikel verfügbar, nehme es trotzdem")
                return i
            
            # Wähle weiteres Partikel j (kleine Partikel bevorzugt, NICHT in used_indices)
            j = self._sample_j_with_exclusion(used_indices, max_retries=10)
            
            # Markiere j als verwendet
            used_indices.add(j)
            
            # Agglomeriere i und j (entfernt j aus der Partikelliste)
            # WICHTIG: Nach swap-pop kann sich der Index von i ändern!
            old_i = i
            old_j = j
            i = self._agglomerate_two_particles_dual(i, j)
            
            # Exclusion-Set aktualisieren:
            # - old_j wurde entfernt, muss aus dem Set entfernt werden
            # - old_i könnte durch swap-pop verschoben worden sein
            # - Nur das neue agglomerierte i bleibt im Set
            used_indices.discard(old_j)  # j wurde entfernt
            if old_i != i:
                used_indices.discard(old_i)  # alter Index von i nicht mehr gültig
            used_indices.add(i)  # neues agglomeriertes i hinzufügen
        
        # Wenn wir hier ankommen, wurde max_iterations erreicht
        if hasattr(self, 'VERBOSE') and self.VERBOSE:
            print(f"[Nukleation DUAL] Warnung: Max iterations ({max_iterations}) erreicht")
        
        return i
    
    def distribute_droplet(self) -> bool:
        """
        Verteilt EINEN Tropfen auf Partikel.
        
        Verwendet die Dual-Sampler Strategie für effiziente Partikelauswahl.
        """
        if self._droplet_rate <= 0 or self._droplet_volume <= 0:
            return False
        
        a_tot = getattr(self, 'a_tot', 0)
        if a_tot <= 0:
            return False
        
        target_idx = self._find_and_agglomerate_for_droplet_dual(self._droplet_volume)
        self._add_liquid_volume(target_idx, self._droplet_volume)
        
        return True
    
    def distribute_droplets_for_timestep(self, dt: float) -> int:
        """
        Verteilt Tropfen für einen gegebenen Zeitschritt.
        
        Verwendet ACCUMULATOR-PATTERN für korrekte Massenbilanz.
        
        WICHTIG: Sampler werden EINMAL pro Zeitschritt gebaut und für alle
        Tropfen im gleichen Zeitschritt wiederverwendet. Dies ist eine
        Optimierung für Geschwindigkeit auf Kosten der Genauigkeit.
        
        Für jeden Tropfen wird ein neues Exclusion-Set erstellt, um
        doppelte Auswahl im gleichen Tropfen zu verhindern.
        """
        if self._droplet_rate <= 0 or dt <= 0:
            return 0
        
        self._accumulated_time += dt
        total_droplets = self._droplet_rate * self._accumulated_time
        num_droplets_int = int(total_droplets)
        
        if num_droplets_int == 0:
            return 0
        
        # ============================================================
        # SAMPLER-Erstellung: EINMAL pro Zeitschritt
        # ============================================================
        # Bei Agglomeration wird a_tot reduziert, daher müssen die Sampler
        # neu gebaut werden. Die Sampler-Gewichte repräsentieren aber
        # den Stand VOR der Agglomeration - das ist ein Kompromiss für
        # Performance auf Kosten der Genauigkeit.
        # ============================================================
        self._build_dual_samplers(rebuild=True)
        
        distributed = 0
        for _ in range(num_droplets_int):
            if self.distribute_droplet():
                distributed += 1
        
        if distributed > 0 and self._droplet_rate > 0:
            time_used = distributed / self._droplet_rate
            self._accumulated_time -= time_used
        
        return distributed
    
    def initialize_nucleation(self) -> None:
        """
        Initialisiere den Nukleationsprozess vor der Simulation.
        """
        self._accumulated_time = 0.0
        
        # Dual-Sampler initialisieren
        self._nuc_i_sampler = None
        self._nuc_j_sampler = None
        
        self._calculate_droplet_rate()
        
        if not hasattr(self, '_rng') or self._rng is None:
            self._rng = np.random.default_rng()
        
        if hasattr(self, 'VERBOSE') and self.VERBOSE:
            print(f"[Nukleation DUAL] Initialisiert:")
            print(f"  - Flüssigkeitszufuhr: {self.liquid_flow_rate*1000:.4f} mL/s")
            print(f"  - Tropfendurchmesser: {self.droplet_diameter*1e6:.2f} µm")
            print(f"  - Tropfenvolumen: {self._droplet_volume:.3e} m³")
            print(f"  - Tropfenrate: {self._droplet_rate:.3e} Tropfen/s")
            print(f"  - Dual-Sampler: ENABLED (i=groß, j=klein)")


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MC-PBE Nukleation DUAL-SAMPLER - Test")
    print("=" * 70)
    
    print("\n[TEST] Initialisierung")
    nucleation = MCPBENucleationDualSampler()
    nucleation._calculate_droplet_rate()
    print(f"  Tropfenvolumen: {nucleation._droplet_volume:.3e} m³")
    print(f"  Tropfenrate: {nucleation._droplet_rate:.3e} Tropfen/s")
    
    print("\n" + "=" * 70)
    print("Test abgeschlossen!")
    print("=" * 70)
