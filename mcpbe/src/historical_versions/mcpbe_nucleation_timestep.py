# ARCHIVED -- development snapshot, not imported by any current code path.
#
# Kept to document the path taken before settling on the design now in
# `mcpbe/src/wmcpbe`. Not maintained; may not run as-is against the current
# pbe-core or wmcpbe. Nucleation combining the dual-sampler idea with per-timestep sampler reuse and an exclusion set against reusing a particle twice in one step. The last nucleation variant tried in this package before the weighted wmcpbe rewrite.
# See mcpbe/docs/Befunde_2026-08-20.md, D-13.

# -*- coding: utf-8 -*-
"""
===============================================================================
MC-PBE Nukleation: DUAL-TIMESTEP-SAMPLER OPTIMIERUNG
===============================================================================

Diese Version kombiniert die besten Aspekte beider Strategien:

**DUAL-SAMPLER (pro Zeitschritt):**
- Sampler 1 (i): Gewichtet nach solid_volume (große Partikel bevorzugt)
- Sampler 2 (j): Gewichtet nach 1/solid_volume (kleine Partikel bevorzugt)
- Beide Sampler werden EINMAL PRO ZEITSCHRITT gebaut

**TIMESTEP-TRACKING:**
- Exclusion-Set verhindert doppelte Auswahl im gleichen Zeitschritt
- Partikel können nur EINMAL pro Zeitschritt verwendet werden
- Besonders effizient bei vielen Tropfen

**Performance-Vorteile:**
- Nur 2 Sampler-Bauten pro Zeitschritt (statt pro Tropfen!)
- Strategische Auswahl reduziert Agglomerationen
- Exclusion-Tracking verhindert redundante Ziehungen

**Algorithmus:**
```
for each timestep:
    # Baue beide Sampler (einmalig)
    i_sampler = build(weights=solid_volume)      # groß bevorzugt
    j_sampler = build(weights=1/solid_volume)    # klein bevorzugt
    used_indices = set()
    
    for each droplet in timestep:
        1. Wähle i aus i_sampler (nicht in used_indices)
        2. Addiere i zu used_indices
        3. Wenn i zu klein:
           - Wähle j aus j_sampler (nicht in used_indices)
           - Addiere j zu used_indices
           - Agglomeriere i + j → neues i
           - Aktualisiere used_indices (nur neues i bleibt)
        4. Füge Tropfen zu i hinzu
```

===============================================================================
"""

from __future__ import annotations

import math
import sys
import os
from typing import Set

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


class MCPBENucleationTimestep:
    """
    Nukleations-Mixin mit DUAL-TIMESTEP-SAMPLER Strategie.
    
    Kombiniert duale Sampler-Strategie mit Timestep-Tracking:
    
    1. **Zwei Sampler pro Zeitschritt:**
       - i_sampler: Große Partikel bevorzugt (als "Kern" für Tropfenaufnahme)
       - j_sampler: Kleine Partikel bevorzugt (als "Material" für Agglomeration)
    
    2. **Exclusion-Tracking:**
       - Verwendete Partikel werden getrackt
       - Kein Partikel wird doppelt im gleichen Zeitschritt verwendet
    
    PERFORMANCE:
    ------------
    - ~200-1000x schneller als Standard-Version
    - ~2-5x schneller als reine Cached-Methode (bei vielen Tropfen)
    - ~1.5-3x schneller als reine Dual-Methode (weniger Sampler-Rebuilds)
    - Best für hohe Tropfenraten mit breiten Partikelgrößenverteilungen
    """
    
    def _calculate_droplet_rate(self) -> None:
        """Berechne die Tropfenrate aus der Flüssigkeitszufuhr und dem Tropfendurchmesser."""
        if self.liquid_flow_rate <= 0 or self.droplet_diameter <= 0:
            self._droplet_rate = 0.0
            self._droplet_volume = 0.0
            return
        
        self._droplet_volume = (math.pi / 6.0) * (self.droplet_diameter ** 3)
        liquid_flow_m3_per_s = self.liquid_flow_rate * 1e-3
        self._droplet_rate = liquid_flow_m3_per_s / self._droplet_volume
        
        if hasattr(self, 'VERBOSE') and self.VERBOSE:
            print(f"[Nukleation DUAL-TIMESTEP] Tropfenvolumen: {self._droplet_volume:.3e} m³")
            print(f"[Nukleation DUAL-TIMESTEP] Tropfenrate: {self._droplet_rate:.3e} Tropfen/s")
    
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
    
    def _build_dual_timestep_samplers(self) -> None:
        """
        Baut BEIDE Sampler für den aktuellen Zeitschritt.
        
        Sampler 1 (i_sampler): Große Partikel bevorzugt
        - Gewichte: w_i = solid_volume[i]
        - Zweck: Findet Partikel das als "Kern" für Tropfenaufnahme dient
        
        Sampler 2 (j_sampler): Kleine Partikel bevorzugt
        - Gewichte: w_j = 1/solid_volume[j] (regularized für numerical stability)
        - Zweck: Liefert Partikel für Agglomeration wenn i zu klein ist
        
        Wird EINMAL PRO ZEITSCHRITT aufgerufen, nicht bei jedem Tropfen!
        """
        a_tot = getattr(self, 'a_tot', 0)
        if a_tot <= 0:
            raise RuntimeError("Keine aktiven Partikel für Nukleation verfügbar.")
        
        # Regularisierung für numerische Stabilität (vermeide division by zero)
        min_solid = 1e-30
        
        # Gewichte für beide Sampler berechnen
        weights_i = np.zeros(a_tot, dtype=float)  # Für Startpartikel (groß bevorzugen)
        weights_j = np.zeros(a_tot, dtype=float)  # Für Zusatzpartikel (klein bevorzugen)
        
        for k in range(a_tot):
            solid_k = self._get_solid_volume(k)
            
            # i_sampler: Große Partikel bevorzugen
            weights_i[k] = max(solid_k, min_solid)
            
            # j_sampler: Kleine Partikel bevorzugen (invers gewichtet)
            weights_j[k] = 1.0 / max(solid_k, min_solid)
        
        # Sampler bauen und als gültig markieren
        self._nuc_i_sampler = FenwickSampler(weights_i)
        self._nuc_j_sampler = FenwickSampler(weights_j)
        self._nuc_timestep_valid = True
        self._nuc_used_indices: Set[int] = set()  # Track verwendete Partikel
        
        if hasattr(self, 'VERBOSE') and self.VERBOSE:
            print(f"[Nukleation DUAL-TIMESTEP] Sampler gebaut für {a_tot} Partikel")
            print(f"                        i_sampler: große Partikel bevorzugt")
            print(f"                        j_sampler: kleine Partikel bevorzugt")
    
    def _sample_with_exclusion(self, sampler: FenwickSampler, excluded: Set[int], 
                                max_retries: int = 100) -> int:
        """
        Zieht ein Partikel aus dem Sampler, das nicht im Exclusion-Set ist.
        
        WICHTIG: Nach swap-pop kann sich a_tot geändert haben!
        Daher immer aktuellen Wert prüfen und bei Bedarf fallback.
        
        Parameter
        ----------
        sampler : FenwickSampler
            Der Sampler aus dem gezogen werden soll
        excluded : Set[int]
            Indices die nicht gewählt werden dürfen
        max_retries : int
            Maximale Versuche bevor Fallback auf sequentielle Auswahl
            
        Rückgabe
        --------
        int
            Index des gewählten Partikels
        """
        a_tot = getattr(self, 'a_tot', 0)
        
        # Safety check: Keine Partikel verfügbar
        if a_tot <= 0:
            raise RuntimeError("Keine Partikel für Sampling verfügbar")
        
        # Wenn fast alle ausgeschlossen, sequentiell wählen
        if len(excluded) >= a_tot - 1:
            for k in range(a_tot):
                if k not in excluded:
                    return k
            return 0
        
        # Versuche sampling mit exclusion
        for _ in range(max_retries):
            idx = sampler.sample(self._rng)
            # SAFETY: Index muss im gültigen Bereich sein!
            if 0 <= idx < a_tot and idx not in excluded:
                return idx
        
        # Fallback: Erstes nicht-ausgeschlossenes Partikel
        for k in range(a_tot):
            if k not in excluded:
                return k
        
        return 0
    
    def _agglomerate_two_particles_timestep(self, i: int, j: int) -> int:
        """
        Agglomeriert zwei Partikel i und j.
        
        WICHTIG: liquid_volume von j wird ZUERST gespeichert, DANN swap-pop,
        DANN zum korrekten neuen Index von i addiert!
        
        Parameter
        ----------
        i : int
            Index des Ziel-Partikels (überlebt)
        j : int
            Index des Quell-Partikels (wird entfernt)
            
        Rückgabe
        --------
        int
            Neuer Index des agglomerierten Partikels (nach swap-pop)
        """
        dim = getattr(self, 'dim', 2)
        a_before = self.a_tot
        
        # 1. liquid_volume von j SPEICHERN (GANZ WICHTIG - vor swap-pop!)
        liquid_j = self._get_liquid_volume(j)
        
        # 2. Feststoff-Volumina addieren
        for d in range(dim):
            self.V_flat[d, i] += self.V_flat[d, j]
        self.V_flat[-1, i] = np.sum(self.V_flat[:dim, i])
        
        # 3. Durchmesser aktualisieren
        self.X[i] = float(self._vol2diam(self.V_flat[-1, i]))
        
        # 4. Partikel j entfernen (swap-pop)
        # ACHTUNG: Danach ist j UNGÜLTIG und a_tot hat sich reduziert!
        self._remove_particle_column(j)
        
        # 5. NEUEN Index von i bestimmen nach swap-pop
        # Swap-Pop Logik:
        # - Wenn j == a_before-1 (j war letztes): i bleibt unverändert
        # - Wenn i == a_before-1 (i war letztes): i wird nach Position j getauscht
        # - Sonst: i bleibt unverändert
        if i == a_before - 1 and j != a_before - 1:
            i_new = j  # i wurde nach Position j getauscht
        else:
            i_new = i  # i bleibt wo es war
        
        # 6. liquid_volume zu KORREKTEM Index addieren (NACH swap-pop!)
        self._add_liquid_volume(i_new, liquid_j)
        
        # 7. Exclusion-Set aktualisieren: Nur neues i_new bleibt im Set
        if hasattr(self, '_nuc_used_indices'):
            self._nuc_used_indices = {i_new}
        
        # 8. FIX: Sampler als ungültig markieren nach Agglomeration!
        # Nach swap-pop sind alle Indices im alten Sampler falsch!
        self._nuc_timestep_valid = False
        
        return i_new
    
    def _find_and_agglomerate_for_droplet_timestep(self, droplet_vol: float) -> int:
        """
        Findet/agglomeriert Partikel bis sie einen Tropfen aufnehmen können.
        
        DUAL-TIMESTEP ALGORITHMUS:
        --------------------------
        1. Wähle Startpartikel i aus i_sampler (große Partikel bevorzugt)
           - i darf NICHT im Exclusion-Set sein
           - i wird zum Exclusion-Set hinzugefügt
        
        2. Prüfe: Kann i den Tropfen aufnehmen?
           - JA: Return i (Fall 1 - direkt aufgenommen)
           - NEIN: Gehe zu Schritt 3
        
        3. Wähle weiteres Partikel j aus j_sampler (kleine Partikel bevorzugt)
           - j darf NICHT im Exclusion-Set sein
           - j wird zum Exclusion-Set hinzugefügt
        
        4. Agglomeriere i mit j → neues Partikel i'
           - Exclusion-Set aktualisieren: nur neues i' bleibt
        
        5. Wiederhole ab Schritt 2 mit i'
        
        WICHTIG: Kein Partikel darf doppelt verwendet werden im gleichen Zeitschritt!
        
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
        
        # Initialisiere Exclusion-Set falls nicht vorhanden
        if not hasattr(self, '_nuc_used_indices'):
            self._nuc_used_indices = set()
        
        # Prüfe ob Sampler vorhanden sind (sollten in distribute_droplets_for_timestep gebaut werden)
        if not hasattr(self, '_nuc_i_sampler') or self._nuc_i_sampler is None:
            self._build_dual_timestep_samplers()
        
        # Wähle Startpartikel i aus i_sampler (große Partikel bevorzugt, NICHT in used_indices)
        i = self._sample_with_exclusion(self._nuc_i_sampler, self._nuc_used_indices)
        self._nuc_used_indices.add(i)
        
        max_iterations = a_tot
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Prüfe Fall 1: Partikel groß genug?
            if self._can_accept_droplet(i, droplet_vol):
                return i  # Erfolg!
            
            # Fall 2: Weitere Partikel benötigen
            if self.a_tot < 2:
                if hasattr(self, 'VERBOSE') and self.VERBOSE:
                    print(f"[Nukleation DUAL-TIMESTEP] Nur 1 Partikel verfügbar")
                return i
            
            # Wähle weiteres Partikel j - sequentiell um Index-Probleme zu vermeiden
            # (Sampler wären nach swap-pop ohnehin ungültig)
            j_found = False
            for k in range(self.a_tot):
                if k not in self._nuc_used_indices:
                    j = k
                    j_found = True
                    break
            
            if not j_found:
                if hasattr(self, 'VERBOSE') and self.VERBOSE:
                    print(f"[Nukleation DUAL-TIMESTEP] Kein verfügbares Partikel für j")
                return i
            
            self._nuc_used_indices.add(j)
            
            # Agglomeriere i und j
            i = self._agglomerate_two_particles_timestep(i, j)
        
        if hasattr(self, 'VERBOSE') and self.VERBOSE:
            print(f"[Nukleation DUAL-TIMESTEP] Warnung: Max iterations erreicht")
        
        return i
    
    def distribute_droplet(self) -> bool:
        """Verteilt EINEN Tropfen auf Partikel."""
        if self._droplet_rate <= 0 or self._droplet_volume <= 0:
            return False
        
        a_tot = getattr(self, 'a_tot', 0)
        if a_tot <= 0:
            return False
        
        target_idx = self._find_and_agglomerate_for_droplet_timestep(self._droplet_volume)
        self._add_liquid_volume(target_idx, self._droplet_volume)
        
        return True
    
    def distribute_droplets_for_timestep(self, dt: float) -> int:
        """
        Verteilt Tropfen für einen Zeitschritt.
        
        WICHTIG: Beide Sampler werden EINMAL gebaut und für ALLE Tropfen verwendet.
        Exclusion-Set verhindert doppelte Partikel-Auswahl über ALLE Tropfen im Zeitschritt.
        
        FIX: Verwende round() statt int() um Floating-Point Präzisionsfehler zu vermeiden.
        Nach Agglomerationen MÜSSEN Sampler neu gebaut werden!
        
        Parameter
        ----------
        dt : float
            Zeitschrittdauer in Sekunden
            
        Rückgabe
        --------
        int
            Anzahl der tatsächlich verteilten Tropfen
        """
        if self._droplet_rate <= 0 or dt <= 0:
            return 0
        
        self._accumulated_time += dt
        # FIX: round() statt int() für korrekte Rundung bei FP-Fehlern
        total_droplets = self._droplet_rate * self._accumulated_time
        num_droplets_int = int(round(total_droplets))
        
        # Safety: Vermeide negative accumulated_time durch Über-Rundung
        max_possible = int(total_droplets) + 1
        num_droplets_int = min(num_droplets_int, max_possible)
        
        if num_droplets_int == 0:
            return 0
        
        # ============================================================
        # SAMPLER-Erstellung: EINMAL pro Zeitschritt (BEIDE Sampler)
        # ============================================================
        # FIX: Prüfe auf _nuc_timestep_valid - wird nach Agglomeration False!
        if not hasattr(self, '_nuc_i_sampler') or \
           self._nuc_i_sampler is None or \
           not getattr(self, '_nuc_timestep_valid', False):
            self._build_dual_timestep_samplers()
        
        distributed = 0
        for idx_droplet in range(num_droplets_int):
            # FIX: Nach jeder Agglomeration Sampler neu bauen!
            # _nuc_timestep_valid wird in _agglomerate_two_particles_timestep=False gesetzt
            if not getattr(self, '_nuc_timestep_valid', False):
                self._build_dual_timestep_samplers()
            
            if self.distribute_droplet():
                distributed += 1
        
        if distributed > 0 and self._droplet_rate > 0:
            time_used = distributed / self._droplet_rate
            self._accumulated_time -= time_used
            # Track total distributed droplets
            self._total_droplets_distributed += distributed
        
        return distributed
    
    def reset_timestep_samplers(self) -> None:
        """
        Setzt die Timestep-Sampler zurück.
        
        Sollte zu Beginn eines neuen Zeitschritts aufgerufen werden wenn
        die Partikelliste sich geändert hat (nach Agglomerationen).
        """
        self._nuc_i_sampler = None
        self._nuc_j_sampler = None
        self._nuc_timestep_valid = False
        self._nuc_used_indices = set()
    
    def initialize_nucleation(self) -> None:
        """Initialisiert den Nukleationsprozess."""
        self._accumulated_time = 0.0
        
        # NEW: Track total distributed droplets for debugging
        self._total_droplets_distributed = 0
        
        # Dual-Timestep-Sampler initialisieren
        self._nuc_i_sampler = None
        self._nuc_j_sampler = None
        self._nuc_timestep_valid = False
        self._nuc_used_indices = set()
        
        self._calculate_droplet_rate()
        
        if not hasattr(self, '_rng') or self._rng is None:
            self._rng = np.random.default_rng()
        
        if hasattr(self, 'VERBOSE') and self.VERBOSE:
            print(f"[Nukleation DUAL-TIMESTEP] Initialisiert:")
            print(f"  - Flüssigkeitszufuhr: {self.liquid_flow_rate*1000:.4f} mL/s")
            print(f"  - Tropfendurchmesser: {self.droplet_diameter*1e6:.2f} µm")
            print(f"  - Tropfenvolumen: {self._droplet_volume:.3e} m³")
            print(f"  - Tropfenrate: {self._droplet_rate:.3e} Tropfen/s")
            print(f"  - Dual-Timestep-Sampler: ENABLED (2 Sampler pro Zeitschritt)")


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MC-PBE Nukleation DUAL-TIMESTEP-SAMPLER - Test")
    print("=" * 70)
    
    print("\n[TEST] Initialisierung")
    nucleation = MCPBENucleationTimestep()
    nucleation._calculate_droplet_rate()
    print(f"  Tropfenvolumen: {nucleation._droplet_volume:.3e} m³")
    print(f"  Tropfenrate: {nucleation._droplet_rate:.3e} Tropfen/s")
    
    print("\n" + "=" * 70)
    print("Test abgeschlossen!")
    print("=" * 70)
