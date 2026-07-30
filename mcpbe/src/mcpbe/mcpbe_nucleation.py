# -*- coding: utf-8 -*-
"""
===============================================================================
MC-PBE Nukleation: Flüssigkeitszugabe durch Tropfen-Agglomeration
===============================================================================

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


class MCPBENucleation:
    """
    Nukleations-Mixin für MCPBE.
    
    Dieser Mixin fügt der Simulation die Möglichkeit hinzu, Flüssigkeitstropfen
    mit bestehenden Partikeln zu kombinieren. Dies simuliert einen
    Nukleations- oder Benetzungsprozess.
    
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
    
    SAMPLER-UPDATE (#4):
    --------------------
    Der Fenwick-Sampler wird bei JEDEM Tropfen neu gebaut da nach Agglomerationen
    sich a_tot reduziert und swap-pop die Indices verändert. Eine Optimierung
    (Sampler nur pro Zeitschritt) ist NICHT möglich ohne fundamentale Änderungen
    am Agglomerations-Algorithmus.
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
            print(f"[Nukleation] Tropfenvolumen: {self._droplet_volume:.3e} m³")
            print(f"[Nukleation] Tropfenrate: {self._droplet_rate:.3e} Tropfen/s")
    
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
    
    def _build_nucleation_sampler(self, force_rebuild: bool = False) -> None:
        """
        Baut/aktualisiert den Fenwick-Sampler für die Nukleation.
        
        Die Gewichtung erfolgt nach solid_volume, da größere Partikel
        mit höherer Wahrscheinlichkeit Tropfen aufnehmen können.
        
        Der Sampler wird als Instanzvariable `_nucleation_sampler` gecached
        und nur bei Bedarf neu gebaut (force_rebuild=True oder nicht vorhanden).
        
        Parameter
        ----------
        force_rebuild : bool
            Wenn True, wird der Sampler auch dann neu gebaut, wenn er bereits existiert.
            Dies ist nötig nach Agglomerationen, da sich a_tot und Gewichte ändern.
        """
        a_tot = getattr(self, 'a_tot', 0)
        if a_tot <= 0:
            raise RuntimeError("Keine aktiven Partikel für Nukleation verfügbar.")
        
        # Prüfen ob rebuild nötig ist
        if not force_rebuild and hasattr(self, '_nucleation_sampler') and self._nucleation_sampler is not None:
            # Sampler existiert bereits und kann wiederverwendet werden
            return
        
        # Gewichte basierend auf solid_volume
        weights = np.zeros(a_tot, dtype=float)
        for i in range(a_tot):
            weights[i] = self._get_solid_volume(i)
        
        # Neuen Sampler bauen und cachen
        self._nucleation_sampler = FenwickSampler(weights)
    
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
        
        return i_new
    
    def _find_and_agglomerate_for_droplet(self, droplet_vol: float) -> int:
        """
        Findet/agglomeriert Partikel bis sie einen Tropfen aufnehmen können.
        
        Algorithmus:
        1. Wähle Startpartikel i über Fenwick-Sampler (gewichtet nach solid_volume)
        2. Prüfe: Kann i den Tropfen aufnehmen?
           - JA: Return i
           - NEIN: Gehe zu Schritt 3
        3. Wähle weiteres Partikel j über Fenwick-Sampler
        4. Agglomeriere i mit j → neues Partikel i'
        5. Wiederhole ab Schritt 2 mit i'
        
        WICHTIG: Nach JEDER Agglomeration MUSS der Sampler neu gebaut werden,
        da sich a_tot reduziert und swap-pop die Indices verändert hat.
        
        SAMPLER-MODE (#4):
        - "per_timestep" (default): Sampler wird einmal PRO TROPFEN neu gebaut
                                    (nicht pro Zeitschritt wie fälschlich versucht).
                                    Dies ist notwendig da Agglomeration die Partikel-Liste ändert.
        - "per_droplet": Identisch zu "per_timestep" (aus Kompatibilitätsgründen).
        
        HINWEIS: Die geplante Optimierung (Sampler nur einmal pro Zeitschritt) ist NICHT
        möglich ohne fundamentale Änderungen am Agglomerations-Algorithmus. Der aktuelle
        Code baut den Sampler korrekt nach jeder Agglomeration neu.
        
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
        
        # Baue Sampler für Startpartikel-Auswahl
        # HINWEIS: Sampler muss nach JEDEM Tropfen neu gebaut werden da:
        # 1. Agglomeration reduziert a_tot
        # 2. swap-pop vertauscht Indices
        # Beides macht den gecachten Sampler ungültig!
        sampler = self._build_nucleation_sampler_single()
        
        # Wähle Startpartikel
        i = sampler.sample(self._rng)
        
        max_iterations = a_tot  # Sicherheit gegen Endlosschleife
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Prüfe Fall 1: Partikel groß genug?
            if self._can_accept_droplet(i, droplet_vol):
                return i
            
            # Fall 2: Weitere Partikel benötigen
            if self.a_tot < 2:
                # Nur noch ein Partikel verfügbar, muss trotzdem nehmen
                if hasattr(self, 'VERBOSE') and self.VERBOSE:
                    print(f"[Nukleation] Nur 1 Partikel verfügbar, nehme es trotzdem")
                return i
            
            # WICHTIG: Sampler nach Agglomeration NEU bauen!
            # a_tot hat sich reduziert und swap-pop hat Indices verändert
            sampler = self._build_nucleation_sampler_single()
            
            # Wähle weiteres Partikel j
            j = sampler.sample(self._rng)
            
            # Vermeide gleiche Partikel (selten, aber möglich)
            if j == i:
                sampler = self._build_nucleation_sampler_single()
                j = sampler.sample(self._rng)
                if j == i and self.a_tot > 1:
                    # Immer noch gleich? Nächstes Partikel nehmen
                    j = (i + 1) % self.a_tot
            
            # Agglomeriere i und j (entfernt j, updated a_tot)
            i = self._agglomerate_two_particles(i, j)
    
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
        
        WICHTIG: Verwende round() statt int() um Floating-Point Präzisionsfehler
        zu vermeiden (z.B. 0.999999 -> 0 statt 1).
        
        HINWEIS: Der Sampler wird in _find_and_agglomerate_for_droplet() bei Bedarf
        automatisch neu gebaut (nach jeder Agglomeration).
        
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
            # DEBUG: Wichtigste Diagnose-Information!
            if hasattr(self, 'VERBOSE') and self.VERBOSE:
                print(f"[NUC DEBUG] EARLY RETURN: rate={self._droplet_rate}, dt={dt}")
            return 0
        
        # ACCUMULATOR-PATTERN:
        # 1. Neue Zeit zum Accumulator addieren
        self._accumulated_time += dt
        
        # 2. Gesamte Tropfen aus accumulierter Zeit berechnen
        # FIX: Verwende round() für korrekte Rundung bei Floating-Point Fehlern
        # Beispiel: 152.79 * 0.006545 = 0.999999... -> round() = 1, int() = 0
        total_droplets = self._droplet_rate * self._accumulated_time
        num_droplets_int = int(round(total_droplets))
        
        # Safety: Stelle sicher dass wir nicht mehr verteilen als physikalisch möglich
        # (vermeide negative accumulated_time durch Über-Rundung)
        if num_droplets_int > 0:
            max_possible = int(total_droplets) + 1
            num_droplets_int = min(num_droplets_int, max_possible)
        
        # DEBUG output
        if hasattr(self, 'VERBOSE') and self.VERBOSE:
            print(f"[NUC DEBUG] dt={dt:.6f}s, accumulated={self._accumulated_time:.9f}s, "
                  f"total_droplets={total_droplets:.4f}, distributing={num_droplets_int}")
        
        # 3. Nur ganze Tropfen verteilen
        distributed = 0
        for _ in range(num_droplets_int):
            if self.distribute_droplet():
                distributed += 1
        
        # 4. Verteilte Zeit abziehen, Rest für nächsten Aufruf behalten
        if distributed > 0 and self._droplet_rate > 0:
            time_used = distributed / self._droplet_rate
            self._accumulated_time -= time_used
            # Track total distributed droplets
            self._total_droplets_distributed += distributed
            
            if hasattr(self, 'VERBOSE') and self.VERBOSE:
                print(f"[NUC DEBUG] Distributed {distributed} droplets, "
                      f"remaining accumulated={self._accumulated_time:.9f}s")
        
        return distributed
    
    def _build_nucleation_sampler_single(self) -> FenwickSampler:
        """
        Baut einen einzelnen Fenwick-Sampler für die Nukleation (ohne Caching).
        
        Wird im "per_droplet" Modus verwendet, wo der Sampler nach jedem Tropfen
        neu gebaut wird (alte Verhaltensweise für maximale Genauigkeit).
        
        Rückgabe
        --------
        FenwickSampler
            Neuer Sampler gewichtet nach solid_volume der aktiven Partikel
        """
        a_tot = getattr(self, 'a_tot', 0)
        if a_tot <= 0:
            raise RuntimeError("Keine aktiven Partikel für Nukleation verfügbar.")
        
        # Gewichte basierend auf solid_volume
        weights = np.zeros(a_tot, dtype=float)
        for i in range(a_tot):
            weights[i] = self._get_solid_volume(i)
        
        return FenwickSampler(weights)
    
    def initialize_nucleation(self) -> None:
        """
        Initialisiere den Nukleationsprozess vor der Simulation.
        
        Sollte nach MCPBEBase._initialize_particles aufgerufen werden.
        """
        # Initialize accumulator for droplet time tracking
        self._accumulated_time = 0.0
        
        # NEW: Track total distributed droplets for debugging
        self._total_droplets_distributed = 0
        
        # Initialize nucleation sampler cache (nur für "per_timestep" Modus)
        self._nucleation_sampler = None
        self._nucleation_sampler_needs_rebuild = False
        
        # Set default sampler mode if not already set
        if not hasattr(self, 'nucleation_sampler_mode'):
            self.nucleation_sampler_mode = "per_timestep"  # Optimized default
        
        self._calculate_droplet_rate()
        
        # RNG vom Hauptsolver übernehmen falls verfügbar
        if not hasattr(self, '_rng') or self._rng is None:
            if hasattr(self, '_rng'):
                self._rng = self._rng
            else:
                self._rng = np.random.default_rng()
        
        if hasattr(self, 'VERBOSE') and self.VERBOSE:
            print(f"[Nukleation] Initialisiert:")
            print(f"  - Flüssigkeitszufuhr: {self.liquid_flow_rate*1000:.4f} mL/s")
            print(f"  - Tropfendurchmesser: {self.droplet_diameter*1e6:.2f} µm")
            print(f"  - Tropfenvolumen: {self._droplet_volume:.3e} m³")
            print(f"  - Tropfenrate: {self._droplet_rate:.3e} Tropfen/s")


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MC-PBE Nukleation - Test (Fenwick-Sampler Agglomeration)")
    print("=" * 70)
    
    # Test 1: Initialisierung und Berechnungen
    print("\n[TEST 1] Initialisierung")
    nucleation = MCPBENucleation()
    nucleation._calculate_droplet_rate()
    print(f"  Tropfenvolumen: {nucleation._droplet_volume:.3e} m³ = {nucleation._droplet_volume*1e15:.2f} fL")
    print(f"  Tropfenrate: {nucleation._droplet_rate:.3e} Tropfen/s")
    
    # Test 2: Mock-Objekt mit Partikeln
    print("\n[TEST 2] Simulation mit Mock-Partikeln")
    
    class MockSolver:
        def __init__(self):
            self.dim = 1
            self.a_tot = 5
            # 5 Partikel mit unterschiedlichen Volumina (Feststoff)
            # Kleines Partikel (0.5 pL) braucht Agglomeration für großen Tropfen
            self.V_flat = np.array([
                [0.5e-12, 1e-12, 2e-12, 5e-12, 10e-12],  # Volumina
            ])
            self.liquid_volume = np.zeros(5)
            self.X = np.zeros(5)
            for i in range(5):
                self.X[i] = (6.0 * self.V_flat[-1, i] / np.pi) ** (1.0/3.0)
            self._rng = np.random.default_rng(42)
            self.VERBOSE = True
            
            # Dummy-Methoden für Kompatibilität
            self._r_agg = np.zeros(5)
            self._break_rate = np.zeros(5)
        
        def _vol2diam(self, V):
            return (6.0 * V / np.pi) ** (1.0/3.0)
        
        def _get_solid_volume(self, idx):
            return float(self.V_flat[-1, idx])
        
        def _get_liquid_volume(self, idx):
            return float(self.liquid_volume[idx])
        
        def _set_liquid_volume(self, idx, vol):
            self.liquid_volume[idx] = vol
        
        def _add_liquid_volume(self, idx, delta):
            self.liquid_volume[idx] += delta
        
        def _remove_particle_column(self, j):
            """Swap-pop wie in MCPBEBase"""
            a = self.a_tot
            if j < 0 or j >= a or a <= 0:
                return
            if a <= 1:
                self.a_tot = 0
                return
            
            last = a - 1
            if j != last:
                self.V_flat[:, [j, last]] = self.V_flat[:, [last, j]]
                self.X[j], self.X[last] = self.X[last], self.X[j]
                self.liquid_volume[j], self.liquid_volume[last] = \
                    self.liquid_volume[last], self.liquid_volume[j]
            
            self.a_tot = last
            self.V_flat[:, self.a_tot:self.a_tot + 1] = 0.0
            self.liquid_volume[self.a_tot:self.a_tot + 1] = 0.0
    
    mock = MockSolver()
    
    # Erweitere Nucleation um Mock-Methoden
    class TestNucleation(MCPBENucleation):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dim = 1
            self.a_tot = 5
            self.V_flat = None
            self.liquid_volume = None
            self.X = None
            self._r_agg = None
            self._break_rate = None
        
        def _vol2diam(self, V):
            return (6.0 * V / np.pi) ** (1.0/3.0)
    
    nuc = TestNucleation(liquid_flow_rate=0.001, droplet_diameter=50e-6)
    nuc._calculate_droplet_rate()
    
    # Verbinde Nucleation mit Mock-Solver - alle Attribute direkt zuweisen
    nuc.dim = mock.dim
    nuc.a_tot = mock.a_tot
    nuc.V_flat = mock.V_flat
    nuc.liquid_volume = mock.liquid_volume
    nuc.X = mock.X
    nuc._rng = mock._rng
    nuc._r_agg = mock._r_agg
    nuc._break_rate = mock._break_rate
    # Methode direkt zuweisen
    nuc._remove_particle_column = lambda j: mock._remove_particle_column(j)
    # Auch die helper Methoden für solid/liquid volume müssen auf mock zugreifen
    nuc._get_solid_volume = lambda idx: mock._get_solid_volume(idx)
    nuc._get_liquid_volume = lambda idx: mock._get_liquid_volume(idx)
    nuc._add_liquid_volume = lambda idx, d: mock._add_liquid_volume(idx, d)
    nuc._set_liquid_volume = lambda idx, v: mock._set_liquid_volume(idx, v)
    
    print(f"  Partikelvolumina (Feststoff): {mock.V_flat[-1] * 1e12} pL")
    print(f"  Tropfenvolumen: {nuc._droplet_volume * 1e12:.2f} pL")
    print(f"  → Tropfen ist größer als Partikel 0,1 → Agglomeration erforderlich!")
    print()
    
    # Simuliere 3 Tropfen
    print("  Verteilung von 3 Tropfen:")
    print("  " + "-" * 60)
    
    for i in range(3):
        particles_before = mock.a_tot
        solids_before = [mock._get_solid_volume(idx) for idx in range(mock.a_tot)]
        liquids_before = [mock._get_liquid_volume(idx) for idx in range(mock.a_tot)]
        
        # Verteile Tropfen
        success = nuc.distribute_droplet()
        
        particles_after = mock.a_tot
        n_aggregated = particles_before - particles_after
        
        if success:
            if n_aggregated > 0:
                print(f"  Tropfen {i+1}: {n_aggregated+1} Partikel → 1 Partikel (Agglomeration)")
            else:
                print(f"  Tropfen {i+1}: 1 Partikel (direkt aufgenommen)")
            print(f"           → {particles_after} Partikel verbleibend")
        else:
            print(f"  Tropfen {i+1}: FEHLGESCHLAGEN")
        
        print()
    
    print("  " + "-" * 60)
    print(f"\n  Finale Partikel: {mock.a_tot}")
    print(f"  Finale Volumina: {mock.V_flat[-1, :mock.a_tot] * 1e12} pL")
    print(f"  Finale Liquids:  {mock.liquid_volume[:mock.a_tot] * 1e12} pL")
    
    # Verifikation: Summe aller liquids sollte = n_droplets * droplet_volume sein
    total_liquid = np.sum(mock.liquid_volume[:mock.a_tot])
    expected_liquid = 3 * nuc._droplet_volume
    print(f"\n  Gesamtliquid: {total_liquid*1e12:.2f} pL")
    print(f"  Erwartet:     {expected_liquid*1e12:.2f} pL (3 Tropfen)")
    print(f"  Abweichung:   {abs(total_liquid - expected_liquid)*1e12:.4f} pL")
    
    print("\n" + "=" * 70)
    print("Test abgeschlossen!")
    print("=" * 70)
