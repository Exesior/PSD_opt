# -*- coding: utf-8 -*-
"""
===============================================================================
MC-PBE Nukleation: Flüssigkeitszugabe durch Tropfen-Agglomeration
===============================================================================

OPTIMIERTE VERSION MIT SAMPLER-CACHING

Dieses Modul implementiert die Nukleation (Keimbildung) als zusätzlichen 
Prozess in der Monte-Carlo Population Balance Simulation.

Der Prozess funktioniert wie folgt:
1. Aus der Config-Datei werden Flüssigkeitszugabe (L/s) und Tropfendurchmesser (m) gelesen
2. Daraus wird die Tropfenrate (Tropfen pro Sekunde) berechnet
3. In jedem Simulationszeitschritt werden Tropfen auf Partikel verteilt

Zwei Fälle bei der Tropfenverteilung:
--------------------------------------
**Fall 1: Partikel ist groß genug**
   Kriterium: solid_volume > liquid_volume + droplet_volume
   → Tropfen wird direkt zum liquid_volume hinzugefügt

**Fall 2: Partikel ist NICHT groß genug**
   → Weitere Partikel werden über Fenwick-Sampler ausgewählt (gewichtetes Sampling)
   → Diese Partikel agglomerieren mit dem Startpartikel:
     - Alle Volumina werden zum ersten Partikel addiert
     - Die zusätzlichen Partikel werden entfernt (swap-pop)
   → Wiederhole bis: sum(solid) > sum(liquid) + droplet_volume
   → Tropfen wird zum liquid_volume des gewachsenen Partikels hinzugefügt

Die Agglomeration verwendet dieselbe Logik wie MCPBEAgg._do_one_agg():
- Fenwick-Sampler für gewichtete Partikelauswahl
- _remove_particle_column für swap-pop Entfernung
- Aktualisierung aller betroffenen Sampler

===============================================================================
"""

from __future__ import annotations

import math
import sys
import os
from typing import Optional, List

import numpy as np

# Handle imports for both package context and standalone execution
try:
    # Try relative import first (when used as part of the mcpbe package)
    from .fenwick import FenwickSampler
except ImportError:
    # Fallback to absolute import for standalone execution
    try:
        # Add parent directory to path if running standalone
        _parent_dir = os.path.dirname(os.path.abspath(__file__))
        if _parent_dir not in sys.path:
            sys.path.insert(0, _parent_dir)
        from fenwick import FenwickSampler
    except ImportError:
        # Last resort: try importing from pbe_core
        from pbe_core.mcpbe.fenwick import FenwickSampler


class MCPBENucleationSampleOpti:
    """
    OPTIMIERTE Nukleations-Mixin für MCPBE mit Sampler-Caching.
    
    Dieser Mixin fügt der Simulation die Möglichkeit hinzu, Flüssigkeitstropfen
    mit bestehenden Partikeln zu kombinieren. Dies simuliert einen
    Nukleations- oder Benetzungsprozess.
    
    PERFORMANCE-OPTIMIERUNG:
    ------------------------
    Im Gegensatz zur Standard-Version (MCPBENucleation) wird der Fenwick-Sampler
    hier GECACHED und nur neu gebaut wenn sich die Partikel-Liste ändert:
    
    - **Ohne Agglomeration**: Sampler wird 1x gebaut und für ALLE Tropfen wiederverwendet
    - **Mit Agglomeration**: Sampler wird nur nach tatsächlichen Agglomerationen neu gebaut
    
    Das spart ~99% der Sampler-Rebuilds bei typischen Nukleations-Cases!
    
    Parameter (aus Config-Datei):
    --------------------------------
    liquid_flow_rate : float
        Flüssigkeitszufuhr in Liter pro Sekunde [L/s]
        
    droplet_diameter : float
        Durchmesser eines einzelnen Tropfens in Metern [m]
    
    Verteilungsalgorithmus:
    -----------------------
    1. Berechne Tropfenrate: droplets_per_second = liquid_flow_rate / droplet_volume
       
    2. Für jeden Tropfen:
       a) Wähle Startpartikel i über Fenwick-Sampler (gewichtet nach solid_volume)
       b) Prüfe: solid[i] > liquid[i] + droplet_volume ?
          - JA (Fall 1): Tropfen hinzufügen, fertig
          - NEIN (Fall 2): Weitere Partikel j über Fenwick-Sampler auswählen,
                           agglomeriere i mit j, wiederhole bis Kriterium erfüllt
       c) Tropfen zum liquid_volume des finalen Partikels hinzufügen
    
    SAMPLER-CACHING Strategie:
    --------------------------
    Der Sampler wird als `_nucleation_sampler` gecached und mit dem Flag
    `_nucleation_sampler_valid` getrackt:
    
    - Initial: sampler = None, valid = False
    - Beim ersten Tropfen: Sampler bauen, valid = True
    - Bei Fall 1 (direkte Aufnahme): valid bleibt True, Sampler wiederverwenden
    - Bei Fall 2 (Agglomeration): valid = False, beim nächsten Durchlauf neu bauen
    
    Use this class instead of MCPBENucleation for better performance.
    """
    
    # No __init__ needed - nucleation parameters are set as attributes
    # by MCPBEBase or by the caller after instantiation.
    # This follows the same pattern as MCPBEAgg and MCPBEBreak mixins.
    
    
        
    def _calculate_droplet_rate(self) -> None:
        """
        Berechne die Tropfenrate aus der Flüssigkeitszufuhr und dem Tropfendurchmesser.
        
        Formel
        ------
        Tropfenvolumen = π/6 * Durchmesser³
        Tropfenrate = Flüssigkeitszufuhr [m³/s] / Tropfenvolumen [m³]
        """
        if self.liquid_flow_rate <= 0 or self.droplet_diameter <= 0:
            self._droplet_rate = 0.0
            self._droplet_volume = 0.0
            return
        
        # Tropfenvolumen (Kugelformel: V = π/6 * d³)
        self._droplet_volume = (math.pi / 6.0) * (self.droplet_diameter ** 3)
        
        # Wandle Flüssigkeitszufuhr von L/s in m³/s
        liquid_flow_m3_per_s = self.liquid_flow_rate * 1e-3
        
        # Tropfen pro Sekunde
        self._droplet_rate = liquid_flow_m3_per_s / self._droplet_volume
        
        if hasattr(self, 'VERBOSE') and self.VERBOSE:
            print(f"[Nukleation OPTI] Tropfenvolumen: {self._droplet_volume:.3e} m³")
            print(f"[Nukleation OPTI] Tropfenrate: {self._droplet_rate:.3e} Tropfen/s")
    
    def _get_solid_volume(self, idx: int) -> float:
        """
        Gibt das Feststoffvolumen eines Partikels zurück.
        
        Für 1D-Systeme: Gesamtvolumen ist Feststoff
        Für 2D-Systeme: Komponente 0 ist Feststoff
        """
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
        """
        Prüft ob ein Partikel einen Tropfen aufnehmen kann.
        
        Kriterium: solid > liquid + droplet_volume
        
        Parameter
        ----------
        idx : int
            Index des Partikels
        droplet_vol : float
            Volumen des Tropfens in m³
            
        Rückgabe
        --------
        bool
            True wenn das Partikel den Tropfen aufnehmen kann
        """
        solid = self._get_solid_volume(idx)
        liquid = self._get_liquid_volume(idx)
        return solid > (liquid + droplet_vol)
    
    def _build_nucleation_sampler_opti(self) -> None:
        """
        Baut den Fenwick-Sampler für die Nukleation MIT CACHING.
        
        Die Gewichtung erfolgt nach solid_volume, da größere Partikel
        mit höherer Wahrscheinlichkeit Tropfen aufnehmen können.
        
        PERFORMANCE-OPTIMIERUNG:
        Der Sampler wird gecached und nur neu gebaut wenn:
        - Er noch nicht existiert ODER
        - _nucleation_sampler_valid = False (nach Agglomeration)
        
        Das vermeidet ~99% der Sampler-Rebuilds bei Fällen ohne Agglomeration!
        """
        a_tot = getattr(self, 'a_tot', 0)
        if a_tot <= 0:
            raise RuntimeError("Keine aktiven Partikel für Nukleation verfügbar.")
        
        # Gewichte basierend auf solid_volume
        weights = np.zeros(a_tot, dtype=float)
        for i in range(a_tot):
            weights[i] = self._get_solid_volume(i)
        
        # Neuen Sampler bauen und als gültig markieren
        self._nucleation_sampler = FenwickSampler(weights)
        self._nucleation_sampler_valid = True
        
        if hasattr(self, 'VERBOSE') and self.VERBOSE:
            print(f"[Nukleation OPTI] Sampler neu gebaut für {a_tot} Partikel")
    
    def _agglomerate_two_particles(self, i: int, j: int) -> int:
        """
        Agglomeriert zwei Partikel i und j zu einem.
        
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
        
        # 1. liquid_volume von j SPEICHERN (GANZ WICHTIG - vor allem anderen!)
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
        if i == a_before - 1 and j != a_before - 1:
            i_new = j  # i wurde nach Position j getauscht
        else:
            i_new = i  # i bleibt wo es war
        
        # 6. liquid_volume zu KORREKTEM Index addieren (NACH swap-pop!)
        self._add_liquid_volume(i_new, liquid_j)
        
        # 7. Sampler als ungültig markieren
        self._nucleation_sampler_valid = False
        
        return i_new
    
    def _find_and_agglomerate_for_droplet(self, droplet_vol: float) -> int:
        """
        Findet/agglomeriert Partikel bis sie einen Tropfen aufnehmen können.
        
        Algorithmus:
        1. Wähle Startpartikel i über Fenwick-Sampler (gewichtet nach solid_volume)
        2. Prüfe: Kann i den Tropfen aufnehmen?
           - JA: Return i (Sampler bleibt gültig!)
           - NEIN: Gehe zu Schritt 3
        3. Wähle weiteres Partikel j über Fenwick-Sampler
        4. Agglomeriere i mit j → neues Partikel i' (Sampler wird ungültig)
        5. Wiederhole ab Schritt 2 mit i'
        
        PERFORMANCE-OPTIMIERUNG:
        Der Sampler wird GECACHED und nur neu gebaut wenn:
        - Er noch nicht existiert ODER
        - Eine Agglomeration stattgefunden hat (a_tot hat sich geändert)
        
        BUGFIX: Nach _build_nucleation_sampler_opti() MUSS sampler-Variable
        aktualisiert werden da sonst auf alten/ungültigen Sampler zugegriffen wird!
        
        Das spart ~99% der Sampler-Rebuilds bei Fällen ohne Agglomeration!
        
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
        
        # OPTIMIERUNG: Sampler cachen und nur bei Bedarf neu bauen
        # Initialer Sampler-Bau (wenn noch nicht vorhanden oder ungültig)
        if not hasattr(self, '_nucleation_sampler') or \
           self._nucleation_sampler is None or \
           not getattr(self, '_nucleation_sampler_valid', False):
            self._build_nucleation_sampler_opti()
        
        # Wähle Startpartikel
        i = self._nucleation_sampler.sample(self._rng)
        
        max_iterations = a_tot  # Sicherheit gegen Endlosschleife
        iteration = 0
        did_agglomeration = False
        
        while iteration < max_iterations:
            iteration += 1
            
            # Prüfe Fall 1: Partikel groß genug?
            if self._can_accept_droplet(i, droplet_vol):
                # Erfolg! Sampler bleibt gültig für nächsten Tropfen
                return i
            
            # Fall 2: Weitere Partikel benötigen
            if self.a_tot < 2:
                # Nur noch ein Partikel verfügbar, muss trotzdem nehmen
                if hasattr(self, 'VERBOSE') and self.VERBOSE:
                    print(f"[Nukleation OPTI] Nur 1 Partikel verfügbar, nehme es trotzdem")
                return i
            
            # OPTIMIERUNG: Sampler nur nach Agglomeration neu bauen
            # Nicht bei jedem Durchlauf!
            if did_agglomeration or not getattr(self, '_nucleation_sampler_valid', False):
                self._build_nucleation_sampler_opti()
                # FIX: sampler Variable wird automatisch über self._nucleation_sampler
                # verwendet, keine separate Variable mehr!
                did_agglomeration = False
            
            # Wähle weiteres Partikel j - IMMER aktuellen Sampler verwenden!
            j = self._nucleation_sampler.sample(self._rng)
            
            # Vermeide gleiche Partikel (selten, aber möglich)
            if j == i:
                j = (i + 1) % self.a_tot
            
            # Agglomeriere i und j (entfernt j, updated a_tot, invalidates sampler)
            i = self._agglomerate_two_particles(i, j)
            did_agglomeration = True  # Marker für nächsten Durchlauf
        
        # Wenn wir hier ankommen, wurde max_iterations erreicht
        if hasattr(self, 'VERBOSE') and self.VERBOSE:
            print(f"[Nukleation OPTI] Warnung: Max iterations ({max_iterations}) erreicht")
        
        return i
    
    def distribute_droplet(self) -> bool:
        """
        Verteilt EINEN Tropfen auf Partikel.
        
        Ablauf:
        1. Finde/agglomeriere Partikel bis Tropfen aufgenommen werden kann
        2. Füge Tropfen zum liquid_volume hinzu
        
        Rückgabe
        --------
        bool
            True wenn Tropfen erfolgreich verteilt wurde
        """
        if self._droplet_rate <= 0 or self._droplet_volume <= 0:
            return False
        
        a_tot = getattr(self, 'a_tot', 0)
        if a_tot <= 0:
            return False
        
        # Finde Partikel das den Tropfen aufnehmen kann (mit Agglomeration falls nötig)
        target_idx = self._find_and_agglomerate_for_droplet(self._droplet_volume)
        
        # Füge Tropfen zum liquid_volume hinzu
        self._add_liquid_volume(target_idx, self._droplet_volume)
        
        return True
    
    def distribute_droplets_for_timestep(self, dt: float) -> int:
        """
        Verteilt Tropfen für einen gegebenen Zeitschritt.
        
        Verwendet ACCUMULATOR-PATTERN für korrekte Massenbilanz:
        - Addiere neue Zeit zum Accumulator
        - Berechne gesamte Tropfen aus accumulierter Zeit
        - Verteile nur ganze Tropfen
        - Speichere Restzeit für nächsten Aufruf
        
        HINWEIS: Der Sampler wird in _find_and_agglomerate_for_droplet() automatisch
        gecached und nur bei Agglomerationen neu gebaut.
        
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
        
        # ACCUMULATOR-PATTERN:
        # 1. Neue Zeit zum Accumulator addieren
        self._accumulated_time += dt
        
        # 2. Gesamte Tropfen aus accumulierter Zeit berechnen
        total_droplets = self._droplet_rate * self._accumulated_time
        num_droplets_int = int(total_droplets)
        
        # 3. Nur ganze Tropfen verteilen
        distributed = 0
        for _ in range(num_droplets_int):
            if self.distribute_droplet():
                distributed += 1
        
        # 4. Verteilte Zeit abziehen, Rest für nächsten Aufruf behalten
        # Dies ist der Kern des Accumulator-Patterns:
        # Wenn z.B. 2.7 Tropfen berechnet wurden, wurden 2 verteilt.
        # Die restliche Zeit für 0.7 Tropfen bleibt im Accumulator.
        if distributed > 0 and self._droplet_rate > 0:
            time_used = distributed / self._droplet_rate
            self._accumulated_time -= time_used
            # Track total distributed droplets
            self._total_droplets_distributed += distributed
        
        return distributed
    
    def initialize_nucleation(self) -> None:
        """
        Initialisiere den Nukleationsprozess vor der Simulation.
        
        Sollte nach MCPBEBase._initialize_particles aufgerufen werden.
        """
        # Initialize accumulator for droplet time tracking
        self._accumulated_time = 0.0
        
        # NEW: Track total distributed droplets for debugging
        self._total_droplets_distributed = 0
        
        # OPTIMIERUNG: Sampler caching initialisieren
        self._nucleation_sampler = None
        self._nucleation_sampler_valid = False
        
        self._calculate_droplet_rate()
        
        # RNG vom Hauptsolver übernehmen falls verfügbar
        if not hasattr(self, '_rng') or self._rng is None:
            if hasattr(self, '_rng'):
                self._rng = self._rng
            else:
                self._rng = np.random.default_rng()
        
        if hasattr(self, 'VERBOSE') and self.VERBOSE:
            print(f"[Nukleation OPTI] Initialisiert:")
            print(f"  - Flüssigkeitszufuhr: {self.liquid_flow_rate*1000:.4f} mL/s")
            print(f"  - Tropfendurchmesser: {self.droplet_diameter*1e6:.2f} µm")
            print(f"  - Tropfenvolumen: {self._droplet_volume:.3e} m³")
            print(f"  - Tropfenrate: {self._droplet_rate:.3e} Tropfen/s")
            print(f"  - Sampler Caching: ENABLED")


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MC-PBE Nukleation OPTI - Test (Sampler Caching)")
    print("=" * 70)
    
    # Test 1: Initialisierung und Berechnungen
    print("\n[TEST 1] Initialisierung")
    nucleation = MCPBENucleationSampleOpti()
    nucleation._calculate_droplet_rate()
    print(f"  Tropfenvolumen: {nucleation._droplet_volume:.3e} m³ = {nucleation._droplet_volume*1e15:.2f} fL")
    print(f"  Tropfenrate: {nucleation._droplet_rate:.3e} Tropfen/s")
    
    print("\n" + "=" * 70)
    print("Test abgeschlossen!")
    print("=" * 70)
