# Debug-Plan: Masseerhaltung parametrischer Fehler

## Ziel
Systematische Identifikation von Masseverlust/-gewinn durch Debug-Prints an allen kritischen Stellen.

## Grundprinzipien der Masseerhaltung in WMCPBE

### 1. Volumen-Semantik
```
V_flat[:dim]    = V_solid (Feststoffvolumen, ERHALTEN bei allen Prozessen!)
V_flat[-1]      = V_dry (Trockenvolumen = V_solid + V_pore)
porosity        = ε = V_pore / V_dry
liquid_volume   = V_liq_total (intensiv, pro physical particle)
saturation      = S = V_liq_intern / V_pore
```

### 2. Erhaltungsgrößen
- **V_solid × W × Summe** = Gesamt-Feststoffmasse (MUSST konstant bleiben!)
- **V_liq_total × W × Summe** = Gesamt-Flüssigkeitsmasse (nur bei Nucleation steigend)
- **n_phys = Sum(W) / Vc** = repräsentierte physikalische Partikelanzahl

### 3. Intensive vs. extensive Größen
| Größe | Typ | Verhalten |
|-------|-----|-----------|
| W | extensiv | Skaliert mit Vc, ändert sich bei Events |
| V_flat | intensiv | Pro physical particle (bleibt gleich) |
| liquid_volume | intensiv | Pro physical particle |
| porosity | intensiv | Pro physical particle |
| saturation | intensiv | Pro physical particle |

---

## Debug-Punkte im Detail

### PUNKT 1: Initialisierung
**Datei:** `mcpbe_base.py`  
**Funktion:** `_initialize_particles()`  
**Zeile:** ~1100 (nach Speichern von V0, W0)

**Print vor dem Speichern:**
```python
# === DEBUG MASSEERHALTUNG: INITIALISIERUNG ===
if getattr(self, 'mcpbe_debug_mass', False):
    v_dry_init = self.V_flat[-1, :self.a_tot]
    poro_init = self.porosity[:self.a_tot]
    w_init = self.W[:self.a_tot]
    
    valid_poro = ~np.isnan(poro_init)
    v_solid_init = np.zeros_like(v_dry_init)
    v_solid_init[valid_poro] = v_dry_init[valid_poro] * (1.0 - poro_init[valid_poro])
    v_solid_init[~valid_poro] = v_dry_init[~valid_poro]
    
    solid_vol_init = np.sum(v_solid_init * w_init)
    liq_vol_init = np.sum(self.liquid_volume[:self.a_tot] * w_init)
    n_phys_init = np.sum(w_init) / self.Vc
    
    print(f"\n[DEBUG INIT] t=0s")
    print(f"  n_comp={self.a_tot}, n_phys={n_phys_init:.3e}, Vc={self.Vc:.3e}")
    print(f"  V_solid_total={solid_vol_init:.6e} m³ (SOLLTE KONSTANT BLEIBEN)")
    print(f"  V_liquid_total={liq_vol_init:.6e} m³")
    print(f"  porosity_mean={np.nanmean(poro_init):.4f}, saturation_mean={np.nanmean(self.saturation[:self.a_tot]):.4f}")
    print(f"  W_range=[{np.min(w_init):.2f}, {np.max(w_init):.2f}]")
# ===============================================
```

**Warum:** Referenzwert für alle späteren Vergleiche. Wenn hier schon falsch, dann ist das Problem in der Initialisierung.

---

### PUNKT 2: Agglomeration - Vor dem Merge
**Datei:** `mcpbe_agg.py`  
**Funktion:** `_merge_pair()`  
**Zeile:** Am Funktionsanfang (~Zeile 850)

**Print:**
```python
# === DEBUG AGG: VOR MERGE ===
print(f"\n[DEBUG AGG] Event #{self.real_agg_events+1:.0f}")
print(f"  Parents: i={i}, j={j}")
print(f"  W[i]={self.W[i]:.2f}, W[j]={self.W[j]:.2f}")
print(f"  V_dry[i]={self.V_flat[-1,i]:.6e}, V_dry[j]={self.V_flat[-1,j]:.6e}")
print(f"  poro[i]={self.porosity[i]:.4f}, poro[j]={self.porosity[j]:.4f}")

# Berechne V_solid der Parents
v_solid_i = self.V_flat[-1,i] * (1.0 - self.porosity[i]) if not np.isnan(self.porosity[i]) else self.V_flat[-1,i]
v_solid_j = self.V_flat[-1,j] * (1.0 - self.porosity[j]) if not np.isnan(self.porosity[j]) else self.V_flat[-1,j]
print(f"  V_solid[i]={v_solid_i:.6e}, V_solid[j]={v_solid_j:.6e}")
print(f"  V_solid_SUM={v_solid_i + v_solid_j:.6e} (SOLLTE = Child V_solid sein)")
print(f"  liq[i]={self.liquid_volume[i]:.6e}, liq[j]={self.liquid_volume[j]:.6e}")
print(f"  liq_SUM={self.liquid_volume[i] + self.liquid_volume[j]:.6e}")
# ================================
```

**Warum:** Zeigt die Eingangswerte vor der Verschmelzung. Kritisch für Nachvollziehbarkeit.

---

### PUNKT 3: Agglomeration - Nach dem Merge
**Datei:** `mcpbe_agg.py`  
**Funktion:** `_merge_pair()`  
**Zeile:** Kurz vor `return new_idx` (~Zeile 950)

**Print:**
```python
# === DEBUG AGG: NACH MERGE ===
print(f"  Child: idx={new_idx}, W={self.W[new_idx]:.2f}")
print(f"  V_dry[child]={self.V_flat[-1,new_idx]:.6e}")
print(f"  poro[child]={self.porosity[new_idx]:.4f}")

v_solid_child = self.V_flat[-1,new_idx] * (1.0 - self.porosity[new_idx]) if not np.isnan(self.porosity[new_idx]) else self.V_flat[-1,new_idx]
print(f"  V_solid[child]={v_solid_child:.6e}")
print(f"  ΔV_solid={v_solid_child - (v_solid_i + v_solid_j):.6e} (SOLLTE ≈ 0 sein)")

print(f"  liq[child]={self.liquid_volume[new_idx]:.6e}")
print(f"  Δliq={self.liquid_volume[new_idx] - (self.liquid_volume[i] + self.liquid_volume[j]):.6e}")
print(f"  sat[child]={self.saturation[new_idx]:.4f}")
# ==============================
```

**Warum:** Vergleich mit Punkt 2 zeigt Masseänderung durch Merge.

---

### PUNKT 4: Agglomeration - Nach Parent-Consumption
**Datei:** `mcpbe_agg.py`  
**Funktion:** `_consume_parent_weight()`  
**Zeile:** Am Funktionsende (~Zeile 1050)

**Print:**
```python
# === DEBUG AGG: NACH CONSUME ===
if hasattr(self, '_debug_agg_count'):
    self._debug_agg_count += 1
else:
    self._debug_agg_count = 1

if self._debug_agg_count <= 20:  # Nur erste 20 Events
    print(f"  After consume: a_tot={self.a_tot}")
    
    # Globale Massenbilanz
    v_dry = self.V_flat[-1, :self.a_tot]
    poro = self.porosity[:self.a_tot]
    w = self.W[:self.a_tot]
    valid = ~np.isnan(poro)
    v_solid = np.zeros_like(v_dry)
    v_solid[valid] = v_dry[valid] * (1.0 - poro[valid])
    v_solid[~valid] = v_dry[~valid]
    
    solid_total = np.sum(v_solid * w)
    liq_total = np.sum(self.liquid_volume[:self.a_tot] * w)
    n_phys = np.sum(w) / self.Vc
    
    print(f"  GLOBAL: V_solid_total={solid_total:.6e}, V_liq_total={liq_total:.6e}, n_phys={n_phys:.3e}")
# =================================
```

**Warum:** Zeigt globale Masse nach komplettem Agg-Event.

---

### PUNKT 5: Breakage - Vor Fragment-Erstellung
**Datei:** `mcpbe_break.py`  
**Funktion:** `_do_one_break()`  
**Zeile:** Nach Auswahl von Partikel k (~Zeile 750)

**Print:**
```python
# === DEBUG BREAK: VOR FRAGMENTIERUNG ===
if not hasattr(self, '_break_debug_counter'):
    self._break_debug_counter = 0
self._break_debug_counter += 1

if self._break_debug_counter <= 20:  # Nur erste 20 Events
    print(f"\n[DEBUG BREAK] Event #{self.real_break_events+1:.0f}")
    print(f"  Parent: k={k}")
    print(f"  W[k]={self.W[k]:.2f}")
    print(f"  V_dry[k]={self.V_flat[-1,k]:.6e}")
    print(f"  poro[k]={self.porosity[k]:.4f}")
    
    v_solid_k = self.V_flat[-1,k] * (1.0 - self.porosity[k]) if not np.isnan(self.porosity[k]) else self.V_flat[-1,k]
    print(f"  V_solid[k]={v_solid_k:.6e}")
    print(f"  liq[k]={self.liquid_volume[k]:.6e}")
# ====================================
```

**Warum:** Dokumentiert Parent-Zustand vor Breakage.

---

### PUNKT 6: Breakage - Nach Fragment-Anwendung
**Datei:** `mcpbe_break.py`  
**Funktion:** `_break_apply_and_maintain()`  
**Zeile:** Am Funktionsende (~Zeile 480)

**Print:**
```python
# === DEBUG BREAK: NACH ANWENDUNG ===
if getattr(self, '_break_debug_counter', 0) <= 20:
    # Summiere Fragment-Volumina
    v_solid_frags = 0.0
    liq_frags = 0.0
    
    # Hole Parent-Wert für Vergleich (falls noch existent)
    if k < self.a_tot:
        v_solid_parent_remaining = self.V_flat[-1,k] * (1.0 - self.porosity[k]) if not np.isnan(self.porosity[k]) else self.V_flat[-1,k]
        w_parent_remaining = self.W[k]
    else:
        v_solid_parent_remaining = 0.0
        w_parent_remaining = 0.0
    
    # Durchlaufe alle neuen Fragmente (letzte dW Partikel)
    for idx in range(max(0, self.a_tot - len(frags)), self.a_tot):
        v_solid_f = self.V_flat[-1,idx] * (1.0 - self.porosity[idx]) if not np.isnan(self.porosity[idx]) else self.V_flat[-1,idx]
        v_solid_frags += v_solid_f * self.W[idx]
        liq_frags += self.liquid_volume[idx] * self.W[idx]
    
    print(f"  Fragments: n={len(frags)}, V_solid_total={v_solid_frags:.6e}, liq_total={liq_frags:.6e}")
    print(f"  Parent remaining: W={w_parent_remaining:.2f}, V_solid={v_solid_parent_remaining * w_parent_remaining:.6e}")
    print(f"  ΔV_solid={(v_solid_frags + v_solid_parent_remaining*w_parent_remaining) - v_solid_k*self.W[k]:.6e}")
# ================================
```

**Warum:** Prüft ob Summe der Fragmente = Parent.

---

### PUNKT 6b: Breakage - Nach Merger-Aufruf
**Datei:** `mcpbe_break.py`  
**Funktion:** `_break_apply_and_maintain()`  
**Zeile:** Nach `find_or_create()` Aufruf (~Zeile 450)

**Print:**
```python
# === DEBUG BREAK: NACH MERGER ===
if getattr(self, '_break_debug_counter', 0) <= 20:
    print(f"  Merger result: idx={new_idx}, was_merged={was_merged}")
    if was_merged:
        print(f"    MERGED: W[{new_idx}] increased by dW={dW:.2f}")
        print(f"    Intensive properties UNCHANGED (correct!)")
    else:
        print(f"    CREATED NEW: W[{new_idx}]={self.W[new_idx]:.2f}")
        V_dry_new = self.V_flat[-1, new_idx]
        poro_new = self.porosity[new_idx] if hasattr(self, 'porosity') else np.nan
        V_solid_new = V_dry_new * (1.0 - poro_new) if not np.isnan(poro_new) else V_dry_new
        print(f"    V_dry={V_dry_new:.6e}, V_solid={V_solid_new:.6e}, liq={self.liquid_volume[new_idx]:.6e}")
# =============================
```

**Warum:** Zeigt ob Merger genutzt wurde und dokumentiert den Zustand nach Merge/Create.

---

### PUNKT 6c: ParticleMerger - find_similar()
**Datei:** `particle_merger.py`  
**Funktion:** `find_similar()`  
**Zeile:** Am Funktionsanfang (~Zeile 200)

**Print:**
```python
# === DEBUG MERGER: FIND_SIMILAR ===
solver = self.solver
if getattr(solver, 'mcpbe_debug', False) and self._stats_lookups % 10 == 0:
    print(f"\n[MERGER DEBUG] Lookup #{self._stats_lookups+1}")
    print(f"  Target: V_dry={V_dry_target:.6e}, liq={liquid_target:.6e}, poro={poro_target:.4f}, sat={sat_target:.4f}")
    print(f"  Tol_rel={self.tol_rel:.1e}, tol_abs_liq={self.tol_abs_liquid:.1e}")
    key = self._compute_hash_key(V_dry_target, liquid_target, poro_target, sat_target)
    candidates = self._hash_index.get(key, set()) if self.use_hash_index else []
    print(f"  Hash index enabled: {self.use_hash_index}, candidates: {len(candidates)}")
# ====================================
```

**Warum:** Dokumentiert wie viele Candidates gefunden wurden und mit welchen Toleranzen gesucht wird.

---

### PUNKT 6d: ParticleMerger - matches_exact()
**Datei:** `particle_merger.py`  
**Funktion:** `_matches_exact()`  
**Zeile:** Für jeden Candidate Check (~Zeile 280)

**Print:**
```python
# === DEBUG MERGER: EXACT MATCH CHECK ===
solver = self.solver
if getattr(solver, 'mcpbe_debug', False):
    V_dry_idx = solver.V_flat[-1, idx]
    liq_idx = solver.liquid_volume[idx]
    poro_idx = solver.porosity[idx] if hasattr(solver, 'porosity') else np.nan
    sat_idx = solver.saturation[idx] if hasattr(solver, 'saturation') else np.nan
    
    delta_V = abs(V_dry_idx - V_dry)
    delta_liq = abs(liq_idx - liquid)
    rel_tol_V = tol * abs(V_dry)
    liq_tol = max(tol * abs(liquid), self.tol_abs_liquid)
    
    match_V = delta_V <= rel_tol_V
    match_liq = delta_liq <= liq_tol
    match_poro = (np.isnan(poro_idx) and np.isnan(poro)) or (abs(poro_idx - poro) <= tol if np.isfinite(poro_idx) and np.isfinite(poro) else False)
    match_sat = (np.isnan(sat_idx) and np.isnan(sat)) or (abs(sat_idx - sat) <= tol if np.isfinite(sat_idx) and np.isfinite(sat) else False)
    
    print(f"  Candidate idx={idx}: V_dry_diff={delta_V:.3e}/{rel_tol_V:.3e} ({'✓' if match_V else '✗'}), liq_diff={delta_liq:.3e}/{liq_tol:.3e} ({'✓' if match_liq else '✗'})")
    print(f"    poro_match={'✓' if match_poro else '✗'}, sat_match={'✓' if match_sat else '✗'}")
    print(f"    OVERALL: {'MATCH' if all([match_V, match_liq, match_poro, match_sat]) else 'NO MATCH'}")
# ====================================
```

**Warum:** Zeigt exakt welche Property zum Match-Fail führt. Kritisch wenn Toleranzen zu streng/lasch sind.

---

### PUNKT 6e: ParticleMerger - _create_new_particle()
**Datei:** `particle_merger.py`  
**Funktion:** `_create_new_particle()`  
**Zeile:** Nach Initialisierung aller Properties (~Zeile 380)

**Print:**
```python
# === DEBUG MERGER: CREATE NEW ===
solver = self.solver
if getattr(solver, 'mcpbe_debug', False):
    print(f"\n[MERGER DEBUG] Created NEW particle at idx={new_idx}")
    print(f"  W={solver.W[new_idx]:.2f}")
    print(f"  V_dry={solver.V_flat[-1, new_idx]:.6e}")
    print(f"  V_solid={np.sum(solver.V_flat[:solver.dim, new_idx]):.6e}")
    print(f"  liquid={solver.liquid_volume[new_idx]:.6e}")
    poro_val = solver.porosity[new_idx] if hasattr(solver, 'porosity') else None
    sat_val = solver.saturation[new_idx] if hasattr(solver, 'saturation') else None
    print(f"  porosity={poro_val:.4f if poro_val is not None and np.isfinite(poro_val) else 'N/A'}")
    print(f"  saturation={sat_val:.4f if sat_val is not None and np.isfinite(sat_val) else 'N/A'}")
    print(f"  Added to hash index: {self.use_hash_index}")
# ====================================
```

**Warum:** Dokumentiert erstellten Partikelzustand. Wichtig für Vergleich mit Target-Werten.

---

### PUNKT 6f: ParticleMerger - Statistiken am Ende
**Datei:** Testdatei nach `solver.solve()`  
**Funktion:** `get_statistics()` aufrufen

**Print in Testdatei:**
```python
# Nach solver.solve()
print("\n[PARTICLE MERGER STATISTICS]")
if hasattr(solver, '_particle_merger') and solver._particle_merger is not None:
    stats = solver._particle_merger.get_statistics()
    print(f"  Lookups: {stats['lookups']:,}")
    print(f"  Merges:  {stats['merges']:,}")
    print(f"  Creates: {stats['creates']:,}")
    print(f"  Merge rate: {stats['merge_rate']*100:.1f}%")
    print(f"  → Higher merge rate = fewer particles created = better performance")
    print(f"  → BUT: Too high merge rate may indicate tolerances too loose!")
else:
    print("  Particle merger not enabled")
```

**Warum:** Merge-Rate zeigt ob Toleranzen sinnvoll eingestellt sind. Zu hoch (>95%) → Toleranzen zu lasch, zu niedrig (<50%) → viele neue Partikel.

---

### PUNKT 7a: Create/Update Nucleated Particle
**Datei:** `mcpbe_nucleation.py`  
**Funktion:** `_create_or_update_nucleated_particle()`  
**Zeile:** Nach dem Setzen aller Properties (~Zeile 1620)

**Print:**
```python
# === DEBUG NUC: CREATE/UPDATE PARTICLE ===
if getattr(self.solver, 'mcpbe_debug', False):
    solver = self.solver
    print(f"\n[DEBUG NUC-PARTICLE] Creating/updating particle at idx={new_idx}")
    print(f"  dW={dW:.2f}")
    print(f"  V_dry_target={v_dry:.6e}, V_dry_actual={solver.V_flat[-1, new_idx]:.6e}")
    print(f"  poro_target={poro:.4f}, poro_actual={solver.porosity[new_idx]:.4f}")
    
    V_solid_calc = v_dry * (1.0 - poro) if not np.isnan(poro) else v_dry
    V_solid_actual = solver.V_flat[-1, new_idx] * (1.0 - solver.porosity[new_idx]) if not np.isnan(solver.porosity[new_idx]) else solver.V_flat[-1, new_idx]
    print(f"  V_solid_calc={V_solid_calc:.6e}, V_solid_actual={V_solid_actual:.6e}")
    print(f"  liquid_target={liquid:.6e}, liquid_actual={solver.liquid_volume[new_idx]:.6e}")
    print(f"  saturation={saturation:.4f}")
# ========================================
```

**Warum:** Dokumentiert den tatsächlichen Zustand des neu erstellten/aktualisierten Partikels. Wichtig für Vergleich mit Target-Werten.

---

### PUNKT 7b: Manual Agglomeration - VOR Agg (Nucleation-spezifisch)
**Datei:** `mcpbe_nucleation.py`  
**Funktion:** `_manual_agglomerate_particles()`  
**Zeile:** Am Funktionsanfang (~Zeile 1700)

**Print:**
```python
# === DEBUG NUC-AGG: VOR MANUELLER AGG ===
if getattr(self.solver, 'mcpbe_debug', False):
    print(f"\n[DEBUG NUC-AGG] Manual agglomeration for V_dry collection")
    print(f"  Particles: i={i}, j={j}")
    print(f"  W[i]={self.solver.W[i]:.2f}, W[j]={self.solver.W[j]:.2f}")
    
    # V_solid von beiden Partikeln
    V_dry_i = float(self.solver.V_flat[-1, i])
    V_dry_j = float(self.solver.V_flat[-1, j])
    poro_i = self.solver.porosity[i] if hasattr(self.solver, 'porosity') else np.nan
    poro_j = self.solver.porosity[j] if hasattr(self.solver, 'porosity') else np.nan
    
    V_solid_i = V_dry_i * (1.0 - poro_i) if not np.isnan(poro_i) else V_dry_i
    V_solid_j = V_dry_j * (1.0 - poro_j) if not np.isnan(poro_j) else V_dry_j
    
    print(f"  V_dry[i]={V_dry_i:.6e}, V_dry[j]={V_dry_j:.6e}")
    print(f"  V_solid[i]={V_solid_i:.6e}, V_solid[j]={V_solid_j:.6e}")
    print(f"  V_solid_SUM={V_solid_i + V_solid_j:.6e} (MUSST = Child V_solid sein)")
    
    liq_i = float(self.solver.liquid_volume[i]) if hasattr(self.solver, 'liquid_volume') else 0.0
    liq_j = float(self.solver.liquid_volume[j]) if hasattr(self.solver, 'liquid_volume') else 0.0
    print(f"  liq[i]={liq_i:.6e}, liq[j]={liq_j:.6e}, SUM={liq_i + liq_j:.6e}")
# =========================================
```

**Warum:** Nucleation ruft manuelle Agglomeration auf um V_dry zu sammeln. Hier kann Masse verloren gehen wenn merging nicht korrekt ist.

---

### PUNKT 7c: Manual Agglomeration - NACH Agg (Nucleation-spezifisch)
**Datei:** `mcpbe_nucleation.py`  
**Funktion:** `_manual_agglomerate_particles()`  
**Zeile:** Kurz vor `return child_current_idx` (~Zeile 1900)

**Print:**
```python
# === DEBUG NUC-AGG: NACH MANUELLER AGG ===
if getattr(self.solver, 'mcpbe_debug', False):
    print(f"  Child: idx={child_current_idx}")
    print(f"  W[child]={self.solver.W[child_current_idx]:.2f}")
    
    V_dry_child = float(self.solver.V_flat[-1, child_current_idx])
    poro_child = self.solver.porosity[child_current_idx] if hasattr(self.solver, 'porosity') else np.nan
    V_solid_child = V_dry_child * (1.0 - poro_child) if not np.isnan(poro_child) else V_dry_child
    
    print(f"  V_dry[child]={V_dry_child:.6e}")
    print(f"  V_solid[child]={V_solid_child:.6e}")
    print(f"  ΔV_solid={V_solid_child - (V_solid_i + V_solid_j):.6e} (SOLLTE ≈ 0 sein)")
    
    liq_child = float(self.solver.liquid_volume[child_current_idx]) if hasattr(self.solver, 'liquid_volume') else 0.0
    print(f"  liq[child]={liq_child:.6e}")
    print(f"  Δliq={liq_child - (liq_i + liq_j):.6e}")
    print(f"  a_tot changed: {solver.a_tot} (parents removed?)")
# =========================================
```

**Warum:** Validiert dass manuelle Agglomeration Masse erhält. Kritisch weil hier ohne Propensity-Rebuild gearbeitet wird.

---

### PUNKT 8: Nucleation - Vor Tropfenverteilung
**Datei:** `mcpbe_nucleation.py`  
**Funktion:** `_distribute_liquid_volume()`  
**Zeile:** Am Funktionsanfang (~Zeile 950)

**Print:**
```python
# === DEBUG NUC: VOR VERTEILUNG ===
if self.config.debug or getattr(self, '_debug_force', False):
    solver = self.solver
    v_dry = solver.V_flat[-1, :solver.a_tot]
    poro = solver.porosity[:solver.a_tot]
    w = solver.W[:solver.a_tot]
    valid = ~np.isnan(poro)
    v_solid = np.zeros_like(v_dry)
    v_solid[valid] = v_dry[valid] * (1.0 - poro[valid])
    v_solid[~valid] = v_dry[~valid]
    
    solid_before = np.sum(v_solid * w)
    liq_before = np.sum(solver.liquid_volume[:solver.a_tot] * w)
    
    print(f"\n[DEBUG NUC] t={self._current_time:.4f}s")
    print(f"  Distributing droplets: n={len(droplet_sizes)}, total_volume={sum(droplet_sizes):.6e}")
    print(f"  BEFORE: V_solid_total={solid_before:.6e}, V_liq_total={liq_before:.6e}")
# ============================
```

**Warum:** Dokumentation vor Flüssigkeitszugabe.

---

### PUNKT 9: Nucleation - Nach Tropfenverteilung
**Datei:** `mcpbe_nucleation.py`  
**Funktion:** `_distribute_liquid_volume()`  
**Zeile:** Am Funktionsende (~Zeile 1050)

**Print:**
```python
# === DEBUG NUC: NACH VERTEILUNG ===
if self.config.debug or getattr(self, '_debug_force', False):
    solver = self.solver
    v_dry = solver.V_flat[-1, :solver.a_tot]
    poro = solver.porosity[:solver.a_tot]
    w = solver.W[:solver.a_tot]
    valid = ~np.isnan(poro)
    v_solid = np.zeros_like(v_dry)
    v_solid[valid] = v_dry[valid] * (1.0 - poro[valid])
    v_solid[~valid] = v_dry[~valid]
    
    solid_after = np.sum(v_solid * w)
    liq_after = np.sum(solver.liquid_volume[:solver.a_tot] * w)
    
    print(f"  AFTER:  V_solid={solid_after:.6e}, V_liq={liq_after:.6e}")
    print(f"  ΔV_solid={solid_after - solid_before:.6e} (SOLLTE = 0 sein!)")
    print(f"  ΔV_liq={liq_after - liq_before:.6e} (SOLLTE ≈ added volume)")
    print(f"  Statistics: droplets_added={self._droplets_added_total:.2e}, volume_added={self._liquid_volume_added_total:.6e}")
# ============================
```

**Warum:** Prüft ob nur Flüssigkeit hinzugefügt wurde (keine Feststoffänderung).

---

### PUNKT 10: Continuous Processes - Vor Kompression
**Datei:** `mcpbe_continuous_processes.py`  
**Funktion:** `_apply_compression()`  
**Zeile:** Nach `active` Maskenberechnung (~Zeile 230)

**Print:**
```python
# === DEBUG COMP: VOR KOMPRESSION ===
if getattr(self, '_debug_comp', False):
    solver = self.solver
    v_dry = solver.V_flat[-1, :solver.a_tot]
    poro = solver.porosity[:solver.a_tot]
    w = solver.W[:solver.a_tot]
    valid = ~np.isnan(poro)
    v_solid = np.zeros_like(v_dry)
    v_solid[valid] = v_dry[valid] * (1.0 - poro[valid])
    v_solid[~valid] = v_dry[~valid]
    
    solid_before = np.sum(v_solid * w)
    print(f"\n[DEBUG COMP] t={self._current_time:.4f}s, dt={dt:.4f}s")
    print(f"  Active particles: {np.sum(active)}")
    print(f"  poro_range=[{np.nanmin(poro):.4f}, {np.nanmax(poro):.4f}]")
    print(f"  BEFORE: V_solid_total={solid_before:.6e} (MUSST KONSTANT BLEIBEN!)")
# ================================
```

**Warum:** Kompression darf V_solid NICHT ändern!

---

### PUNKT 11: Continuous Processes - Nach Kompression
**Datei:** `mcpbe_continuous_processes.py`  
**Funktion:** `_apply_compression()`  
**Zeile:** Am Funktionsende (~Zeile 280)

**Print:**
```python
# === DEBUG COMP: NACH KOMPRESSION ===
if getattr(self, '_debug_comp', False):
    solver = self.solver
    v_dry = solver.V_flat[-1, :solver.a_tot]
    poro = solver.porosity[:solver.a_tot]
    w = solver.W[:solver.a_tot]
    valid = ~np.isnan(poro)
    v_solid = np.zeros_like(v_dry)
    v_solid[valid] = v_dry[valid] * (1.0 - poro[valid])
    v_solid[~valid] = v_dry[~valid]
    
    solid_after = np.sum(v_solid * w)
    liq_after = np.sum(solver.liquid_volume[:solver.a_tot] * w)
    
    print(f"  AFTER:  V_solid_total={solid_after:.6e}, V_liq_total={liq_after:.6e}")
    print(f"  ΔV_solid={solid_after - solid_before:.6e} (SOLLTE = 0 sein!)")
    print(f"  poro_range=[{np.nanmin(poro):.4f}, {np.nanmax(poro):.4f}]")
# ================================
```

**Warum:** Validiert dass Kompression nur V_dry ändert, nicht V_solid.

---

### PUNKT 12: Control Volume Doubling
**Datei:** `mcpbe_base.py`  
**Funktion:** `_maybe_double_control_volume()`  
**Zeile:** Nach Verdopplung (~Zeile 1550)

**Print:**
```python
# === DEBUG VC-DOUBLE ===
if self.VERBOSE or self.mcpbe_debug:
    v_dry = self.V_flat[-1, :self.a_tot]
    poro = self.porosity[:self.a_tot]
    w = self.W[:self.a_tot]
    valid = ~np.isnan(poro)
    v_solid = np.zeros_like(v_dry)
    v_solid[valid] = v_dry[valid] * (1.0 - poro[valid])
    v_solid[~valid] = v_dry[~valid]
    
    solid_total = np.sum(v_solid * w)
    liq_total = np.sum(self.liquid_volume[:self.a_tot] * w)
    n_phys_new = np.sum(w) / self.Vc
    
    print(f"\n[DEBUG VC-DOUBLE] t={elapsed_time:.4f}s")
    print(f"  Vc: {old_Vc:.3e} -> {self.Vc:.3e}")
    print(f"  a_tot: {old_a} -> {self.a_tot}")
    print(f"  n_phys: {old_a/old_Vc:.3e} -> {n_phys_new:.3e}")
    print(f"  V_solid_total={solid_total:.6e} (MUSST GLEICH BLEIBEN!)")
    print(f"  V_liq_total={liq_total:.6e}")
# =========================
```

**Warum:** Vc-Doubling darf intensive Größen nicht skalieren!

---

### PUNKT 13: Nach jedem Save-Point (globale Bilanz)
**Datei:** `mcpbe_base.py`  
**Funktion:** `solve()`  
**Zeile:** Im Save-Loop (~Zeile 1950)

**Print:**
```python
# === DEBUG SAVE-POINT MASSENBILANZ ===
if self.mcpbe_debug and next_save_idx < len(self.t_vec):
    v_dry = self.V_flat[:, :self.a_tot][-1, :]
    poro = self.porosity[:self.a_tot]
    w = self.W[:self.a_tot]
    valid = ~np.isnan(poro)
    v_solid = np.zeros_like(v_dry)
    v_solid[valid] = v_dry[valid] * (1.0 - poro[valid])
    v_solid[~valid] = v_dry[~valid]
    
    solid_total = np.sum(v_solid * w)
    liq_total = np.sum(self.liquid_volume[:self.a_tot] * w)
    n_phys = np.sum(w) / self.Vc
    
    # Vergleich mit Initialisierung
    if hasattr(self, 'V0'):
        v_solid_0 = self.V0[-1, :] * (1.0 - self.porosity0) if hasattr(self, 'porosity0') else self.V0[-1, :]
        solid_0 = np.sum(v_solid_0 * self.W0)
        error_solid = (solid_total - solid_0) / solid_0 * 100
    else:
        error_solid = 0.0
    
    print(f"\n[DEBUG SAVE] t={elapsed_time:.4f}s (save #{next_save_idx})")
    print(f"  n_comp={self.a_tot}, n_phys={n_phys:.3e}")
    print(f"  V_solid={solid_total:.6e}, Δ={error_solid:+.6f}% (SOLLTE = 0%)")
    print(f"  V_liq={liq_total:.6e}")
# ================================
```

**Warum:** Globale Übersicht über gesamten Simulationsverlauf.

---

### PUNKT 14: Nucleation - V_dry Sampling Loop
**Datei:** `mcpbe_nucleation.py`  
**Funktion:** `_distribute_one_droplet_with_dW()`  
**Zeile:** Im while-Loop nach Agglomeration (~Zeile 1400)

**Print:**
```python
# === DEBUG NUC: V_DRY SAMPLING LOOP ===
if getattr(self.solver, 'mcpbe_debug', False) and attempts % 10 == 0:
    print(f"\n[DEBUG NUC-V_DRY] Collecting V_dry for droplet...")
    print(f"  Attempt {attempts}/{max_attempts}")
    print(f"  Current particle: i={i}")
    print(f"  V_dry_sum={V_dry_sum:.6e}, new_liquid={new_liquid:.6e}")
    print(f"  Ratio V_dry/liq={V_dry_sum/new_liquid:.2f} (SOLLTE >= 1.0 sein)")
    print(f"  Still collecting: {V_dry_sum < new_liquid}")
# ========================================
```

**Warum:** Wenn V_dry < new_liquid, wird agglomeriert um mehr Feststoff zu sammeln. Dieser Loop kann viele Iterationen brauchen und dabei Masse verlieren.

---

## Zusammenfassung Debug-Punkte

| Kategorie | Debug-Punkte |
|-----------|--------------|
| Initialisierung | 1 (Punkt 1) |
| Agglomeration (normal) | 3 (Punkte 2-4) |
| Breakage | 3 (Punkte 5-6, 6b) |
| **ParticleMerger** | 4 (Punkte 6c, 6d, 6e, 6f) ← NEU |
| **Nucleation** | 6 (Punkte 7a, 7b, 7c, 8, 9, 14) |
| Continuous Processes | 2 (Punkte 10-11) |
| Vc-Doubling | 1 (Punkt 12) |
| Save-Points | 1 (Punkt 13) |
| **SUMME** | **21 Debug-Punkte** |

---

## Aktivierung der Debug-Prints

### Feinere Auffächerung über Flags

| Flag | Zweck | Betroffene Punkte |
|------|-------|-------------------|
| `solver.mcpbe_debug_mass` | **Hauptflag für Masseerhaltung** | ALLE Punkte 1-14 |
| `solver.mcpbe_debug_agg` | Nur Agglomeration | Punkte 2-4 |
| `solver.mcpbe_debug_break` | Nur Breakage | Punkte 5-6, 6b |
| `solver.mcpbe_debug_merger` | Nur ParticleMerger | Punkte 6c-6f |
| `solver.mcpbe_debug_nuc` | Nur Nucleation | Punkte 7a-7c, 8-9, 14 |
| `solver.mcpbe_debug_comp` | Nur Compression | Punkte 10-11 |
| `solver.mcpbe_debug_vc` | Nur Vc-Doubling | Punkt 12 |

### Beispiel-Aktivierung in Testdatei
```python
# Vor solver.solve() aufrufen:

# ALLE Debug-Prints aktivieren (empfohlen für Fehlersuche)
solver.mcpbe_debug_mass = True

# ODER selektiv:
solver.mcpbe_debug_mass = True      # Basis muss an sein
solver.mcpbe_debug_nuc = True       # Nur Nucleation detailliert
solver.mcpbe_debug_merger = True    # Nur Merger detailliert

# Limitierung der Ausgabe (erste N Events pro Kategorie)
solver._debug_max_events = 50  # Default: 20
```

### Implementierungs-Template in Code
```python
# In jeder Datei prüfen:
if getattr(self.solver, 'mcpbe_debug_mass', False):
    # Globaler Check: immer wenn Hauptflag an ist
    pass

# Oder kategorien-spezifisch:
if getattr(self.solver, 'mcpbe_debug_mass', False) and \
   getattr(self.solver, 'mcpbe_debug_nuc', True):  # True = immer wenn mass an
    # Nucleation-spezifische Prints
    pass

# Event-Limitierung:
max_events = getattr(self.solver, '_debug_max_events', 20)
if event_count < max_events:
    # Print nur wenn unter Limit
    pass
```

---

## Auswertung der Logs

### Erwartete Ergebnisse (korrekte Masseerhaltung):

| Prozess | ΔV_solid | ΔV_liquid |
|---------|----------|-----------|
| Agglomeration | ≈ 0 (±1e-15) | ≈ 0 |
| Breakage | ≈ 0 (±1e-15) | ≈ 0 |
| Nucleation | ≈ 0 (±1e-15) | = added_volume |
| Compression | ≈ 0 (±1e-15) | ≈ 0 |
| Vc-Doubling | ≈ 0 (±1e-15) | ≈ 0 |

### Fehlerindikatoren:

| Symptom | Mögliche Ursache |
|---------|------------------|
| ΔV_solid > 1e-10 bei Agg | `_merge_pair()` berechnet V_solid falsch |
| ΔV_solid > 1e-10 bei Breakage | Fragment-Volumina summieren ≠ Parent |
| ΔV_solid > 1e-10 bei Nucleation | `_manual_agglomerate_particles()` verliert Masse |
| ΔV_solid sprunghaft bei Save | Vc-Doubling skaliert intensive Größen |
| ΔV_liquid ≠ added_volume | `_liquid_remainder` geht verloren |
| n_phys ändert sich ohne Event | W wird fälschlich modifiziert |

---

## Nächste Schritte

1. **Implementierung:** Alle 21 Debug-Punkte mit konditionalen Prints versehen
2. **Test anpassen:** `solver.mcpbe_debug = True` in Testdatei setzen
3. **Log-Analyse:** Ersten signifikanten Massefehler identifizieren (>1e-10 relative Abweichung)
4. **Fix:** Spezifischen Fehler beheben an identifizierter Stelle
5. **Validierung:** Test erneut laufen lassen und Masseerhaltung bestätigen