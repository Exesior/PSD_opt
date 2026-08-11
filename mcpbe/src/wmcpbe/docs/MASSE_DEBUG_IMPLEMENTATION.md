# Debug-Implementierung: Masseerhaltung parametrischer Fehler

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
**Zeile:** ~1120 (nach Speichern von V0, W0)

**Code:**
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
    print(f"  V_solid_total={solid_vol_init:.6e} m^3 (SOLLTE KONSTANT BLEIBEN)")
    print(f"  V_liquid_total={liq_vol_init:.6e} m^3")
    print(f"  porosity_mean={np.nanmean(poro_init):.4f}, saturation_mean={np.nanmean(self.saturation[:self.a_tot]):.4f}")
    print(f"  W_range=[{np.min(w_init):.2f}, {np.max(w_init):.2f}]")
# ===============================================
```

**Warum:** Referenzwert für alle späteren Vergleiche. Wenn hier schon falsch, dann ist das Problem in der Initialisierung.

---

### PUNKT 2: Agglomeration - Vor dem Merge
**Datei:** `mcpbe_agg.py`  
**Funktion:** `_merge_pair()`  
**Zeile:** ~890 (am Funktionsanfang)

**Code:**
```python
# === DEBUG AGG: VOR MERGE ===
if getattr(self, 'mcpbe_debug_mass', False):
    max_events = getattr(self, '_debug_max_events', 20)
    if not hasattr(self, '_debug_agg_count'):
        self._debug_agg_count = 0
    if self._debug_agg_count < max_events:
        self._debug_agg_count += 1
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
**Zeile:** ~1050 (kurz vor `return new_idx`)

**Code:**
```python
# === DEBUG AGG: NACH MERGE ===
if getattr(self, 'mcpbe_debug_mass', False):
    max_events = getattr(self, '_debug_max_events', 20)
    if hasattr(self, '_debug_agg_count') and self._debug_agg_count <= max_events:
        print(f"  Child: idx={new_idx}, W={self.W[new_idx]:.2f}")
        print(f"  V_dry[child]={self.V_flat[-1,new_idx]:.6e}")
        print(f"  poro[child]={self.porosity[new_idx]:.4f}")
        
        v_solid_child = self.V_flat[-1,new_idx] * (1.0 - self.porosity[new_idx]) if not np.isnan(self.porosity[new_idx]) else self.V_flat[-1,new_idx]
        print(f"  V_solid[child]={v_solid_child:.6e}")
        print(f"  ΔV_solid={v_solid_child - V_solid_merged:.6e} (SOLLTE ≈ 0 sein)")
        
        print(f"  liq[child]={self.liquid_volume[new_idx]:.6e}")
        print(f"  Δliq={self.liquid_volume[new_idx] - liquid_merged:.6e}")
        print(f"  sat[child]={self.saturation[new_idx]:.4f}")
# ==============================
```

**Warum:** Vergleich mit Punkt 2 zeigt Masseänderung durch Merge.

---

### PUNKT 4: Agglomeration - Nach Parent-Consumption
**Datei:** `mcpbe_agg.py`  
**Funktion:** `_consume_parent_weight()`  
**Zeile:** ~1200 (am Funktionsende)

**Code:**
```python
# === DEBUG AGG: NACH CONSUME ===
if getattr(self, 'mcpbe_debug_mass', False):
    max_events = getattr(self, '_debug_max_events', 20)
    if hasattr(self, '_debug_agg_count') and self._debug_agg_count <= max_events:
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
**Funktion:** `_break_apply_and_maintain()`  
**Zeile:** ~565 (nach Auswahl von Partikel k)

**Code:**
```python
# === DEBUG BREAK: VOR FRAGMENTIERUNG ===
if getattr(self, 'mcpbe_debug_mass', False):
    max_events = getattr(self, '_debug_max_events', 20)
    if not hasattr(self, '_break_debug_counter'):
        self._break_debug_counter = 0
    if self._break_debug_counter < max_events:
        self._break_debug_counter += 1
        print(f"\n[DEBUG BREAK] Event #{self.real_break_events+1:.0f}")
        print(f"  Parent: k={k}")
        print(f"  W[k]={self.W[k]:.2f}, dW={dW:.2f}")
        print(f"  V_dry[k]={self.V_flat[-1,k]:.6e}")
        print(f"  poro[k]={self.porosity[k]:.4f}")
        
        v_solid_k = self.V_flat[-1,k] * (1.0 - self.porosity[k]) if not np.isnan(self.porosity[k]) else self.V_flat[-1,k]
        print(f"  V_solid[k]={v_solid_k:.6e}")
        print(f"  liq[k]={self.liquid_volume[k]:.6e}")
# ====================================
```

**Warum:** Dokumentiert Parent-Zustand vor Breakage.

---

### PUNKT 6b: Breakage - Nach Merger-Aufruf
**Datei:** `mcpbe_break.py`  
**Funktion:** `_break_apply_and_maintain()`  
**Zeile:** ~700 (nach `find_or_create()` Aufruf)

**Code:**
```python
# === DEBUG BREAK: NACH MERGER ===
if getattr(self, 'mcpbe_debug_mass', False):
    max_events = getattr(self, '_debug_max_events', 20)
    if hasattr(self, '_break_debug_counter') and self._break_debug_counter <= max_events:
        print(f"  Frag[{idx_frag}]: idx={new_idx}, was_merged={was_merged}")
        if was_merged:
            print(f"    MERGED: W[{new_idx}] increased by dW={dW:.2f}")
            print(f"    Intensive properties UNCHANGED (correct!)")
        else:
            print(f"    CREATED NEW: W[{new_idx}]={self.W[new_idx]:.2f}")
            V_dry_new = self.V_flat[-1, new_idx]
            poro_new = self.porosity[new_idx] if hasattr(self, 'porosity') else np.nan
            V_solid_new = V_dry_new * (1.0 - poro_new) if not np.isnan(poro_new) else V_dry_new
            print(f"    V_dry={V_dry_new:.6e}, V_solid={V_solid_new:.6e}, liq={self.liquid_volume[new_idx]:.6e}")
# ==============================
```

**Warum:** Zeigt ob Merger genutzt wurde und dokumentiert den Zustand nach Merge/Create.

---

### PUNKT 6: Breakage - Nach Fragment-Anwendung
**Datei:** `mcpbe_break.py`  
**Funktion:** `_break_apply_and_maintain()`  
**Zeile:** ~780 (am Funktionsende)

**Code:**
```python
# === DEBUG BREAK: NACH ANWENDUNG ===
if getattr(self, 'mcpbe_debug_mass', False):
    max_events = getattr(self, '_debug_max_events', 20)
    if hasattr(self, '_break_debug_counter') and self._break_debug_counter <= max_events:
        print(f"  After breakage: a_tot={self.a_tot}")
        print(f"  Parent remaining: W={w_parent_remaining:.2f}, V_solid={v_solid_parent_remaining * w_parent_remaining:.6e}")
# ================================
```

**Warum:** Zeigt globale Masse nach Breakage-Event.

---

### PUNKT 6c-6f: ParticleMerger
**Datei:** `particle_merger.py`

**find_or_create()** (~Zeile 180):
```python
# === DEBUG MERGER: FIND_OR_CREATE ===
solver = self.solver
if getattr(solver, 'mcpbe_debug_mass', False) and getattr(solver, 'mcpbe_debug_merger', True) and self._stats_lookups % 10 == 0:
    print(f"\n[MERGER DEBUG] Lookup #{self._stats_lookups}")
    print(f"  Target: V_dry={V_dry_target:.6e}, liq={liquid_target:.6e}, poro={poro_target:.4f if poro_target is not None else 'nan'}, sat={sat_target:.4f if sat_target is not None else 'nan'}")
    print(f"  Tol_rel={self.tol_rel:.1e}, tol_abs_liq={self.tol_abs_liquid:.1e}")
# ====================================
```

**find_similar()** (~Zeile 270):
```python
# === DEBUG MERGER: FIND_SIMILAR ENTRY ===
solver = self.solver
if getattr(solver, 'mcpbe_debug_mass', False) and getattr(solver, 'mcpbe_debug_merger', True) and self._stats_lookups % 50 == 0:
    key = self._compute_hash_key(V_dry_target, liquid_target, poro_target, sat_target)
    candidates = self._hash_index.get(key, set()) if self.use_hash_index else []
    print(f"  find_similar: hash_enabled={self.use_hash_index}, candidates_in_bin={len(candidates)}")
# ====================================
```

**matches_exact()** (~Zeile 530):
```python
# === DEBUG MERGER: EXACT MATCH CHECK ===
if getattr(solver, 'mcpbe_debug_mass', False) and getattr(solver, 'mcpbe_debug_merger', True):
    V_dry_idx = solver.V_flat[-1, idx]
    liq_idx = solver.liquid_volume[idx]
    poro_idx = solver.porosity[idx]
    sat_idx = solver.saturation[idx] if hasattr(solver, 'saturation') else np.nan
    
    delta_V = abs(V_dry_idx - V_dry)
    delta_liq = abs(liq_idx - liquid)
    rel_tol_V = tol * abs(V_dry)
    liq_tol_check = max(tol * abs(liquid), self.tol_abs_liquid)
    
    match_V = delta_V <= rel_tol_V
    match_liq = delta_liq <= liq_tol_check
    print(f"  Candidate idx={idx}: V_diff={delta_V:.3e}/{rel_tol_V:.3e}({'✓' if match_V else '✗'}), liq_diff={delta_liq:.3e}/{liq_tol_check:.3e}({'✓' if match_liq else '✗'})")
# ====================================
```

**create_new_particle()** (~Zeile 620):
```python
# === DEBUG MERGER: CREATE NEW ===
if getattr(solver, 'mcpbe_debug_mass', False) and getattr(solver, 'mcpbe_debug_merger', True):
    poro_val = solver.porosity[new_idx]
    sat_val = solver.saturation[new_idx] if hasattr(solver, 'saturation') else None
    print(f"\n[MERGER DEBUG] Created NEW particle at idx={new_idx}")
    print(f"  W={solver.W[new_idx]:.2f}")
    print(f"  V_dry={solver.V_flat[-1, new_idx]:.6e}")
    print(f"  V_solid={np.sum(solver.V_flat[:solver.dim, new_idx]):.6e}")
    print(f"  liquid={solver.liquid_volume[new_idx]:.6e}")
    print(f"  porosity={poro_val:.4f if np.isfinite(poro_val) else 'N/A'}")
    print(f"  saturation={sat_val:.4f if sat_val is not None and np.isfinite(sat_val) else 'N/A'}")
# ====================================
```

**Warum:** ParticleMerger wird in Breakage UND Nucleation verwendet - Fehler hier betrifft beide Prozesse.

---

### PUNKT 7a: Create/Update Nucleated Particle
**Datei:** `mcpbe_nucleation.py`  
**Funktion:** `_create_or_update_nucleated_particle()`  
**Zeile:** ~1680 (nach dem Setzen aller Properties)

**Code:**
```python
# === DEBUG NUC: CREATE/UPDATE PARTICLE ===
if getattr(solver, 'mcpbe_debug_mass', False) and getattr(solver, 'mcpbe_debug_nuc', True):
    print(f"\n[DEBUG NUC-PARTICLE] Created/updated particle at idx={new_idx}")
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
**Zeile:** ~1790 (am Funktionsanfang)

**Code:**
```python
# === DEBUG NUC-AGG: VOR MANUELLER AGG ===
if getattr(solver, 'mcpbe_debug_mass', False) and getattr(solver, 'mcpbe_debug_nuc', True):
    print(f"\n[DEBUG NUC-AGG] Manual agglomeration for V_dry collection")
    print(f"  Particles: i={i}, j={j}")
    print(f"  W[i]={solver.W[i]:.2f}, W[j]={solver.W[j]:.2f}")
    
    # V_solid von beiden Partikeln
    V_dry_i_dbg = float(solver.V_flat[-1, i])
    V_dry_j_dbg = float(solver.V_flat[-1, j])
    poro_i_dbg = solver.porosity[i] if hasattr(solver, 'porosity') else np.nan
    poro_j_dbg = solver.porosity[j] if hasattr(solver, 'porosity') else np.nan
    
    V_solid_i_dbg = V_dry_i_dbg * (1.0 - poro_i_dbg) if not np.isnan(poro_i_dbg) else V_dry_i_dbg
    V_solid_j_dbg = V_dry_j_dbg * (1.0 - poro_j_dbg) if not np.isnan(poro_j_dbg) else V_dry_j_dbg
    
    print(f"  V_dry[i]={V_dry_i_dbg:.6e}, V_dry[j]={V_dry_j_dbg:.6e}")
    print(f"  V_solid[i]={V_solid_i_dbg:.6e}, V_solid[j]={V_solid_j_dbg:.6e}")
    print(f"  V_solid_SUM={V_solid_i_dbg + V_solid_j_dbg:.6e} (MUSST = Child V_solid sein)")
    
    liq_i_dbg = float(solver.liquid_volume[i]) if hasattr(solver, 'liquid_volume') else 0.0
    liq_j_dbg = float(solver.liquid_volume[j]) if hasattr(solver, 'liquid_volume') else 0.0
    print(f"  liq[i]={liq_i_dbg:.6e}, liq[j]={liq_j_dbg:.6e}, SUM={liq_i_dbg + liq_j_dbg:.6e}")
# =========================================
```

**Warum:** Nucleation ruft manuelle Agglomeration auf um V_dry zu sammeln. Hier kann Masse verloren gehen wenn merging nicht korrekt ist.

---

### PUNKT 7c: Manual Agglomeration - NACH Agg (Nucleation-spezifisch)
**Datei:** `mcpbe_nucleation.py`  
**Funktion:** `_manual_agglomerate_particles()`  
**Zeile:** ~2030 (kurz vor `return child_current_idx`)

**Code:**
```python
# === DEBUG NUC-AGG: NACH MANUELLER AGG ===
if getattr(solver, 'mcpbe_debug_mass', False) and getattr(solver, 'mcpbe_debug_nuc', True):
    print(f"  Child: idx={child_current_idx}")
    print(f"  W[child]={solver.W[child_current_idx]:.2f}")
    
    V_dry_child = float(solver.V_flat[-1, child_current_idx])
    poro_child = solver.porosity[child_current_idx] if hasattr(solver, 'porosity') else np.nan
    V_solid_child = V_dry_child * (1.0 - poro_child) if not np.isnan(poro_child) else V_dry_child
    
    print(f"  V_dry[child]={V_dry_child:.6e}")
    print(f"  V_solid[child]={V_solid_child:.6e}")
    print(f"  ΔV_solid={V_solid_child - (V_solid_i_dbg + V_solid_j_dbg):.6e} (SOLLTE ≈ 0 sein)")
    
    liq_child = float(solver.liquid_volume[child_current_idx]) if hasattr(solver, 'liquid_volume') else 0.0
    print(f"  liq[child]={liq_child:.6e}")
    print(f"  Δliq={liq_child - (liq_i_dbg + liq_j_dbg):.6e}")
    print(f"  a_tot changed: {solver.a_tot} (parents removed?)")
# =========================================
```

**Warum:** Validiert dass manuelle Agglomeration Masse erhält. Kritisch weil hier ohne Propensity-Rebuild gearbeitet wird.

---

### PUNKT 8: Nucleation - Vor Tropfenverteilung
**Datei:** `mcpbe_nucleation.py`  
**Funktion:** `_distribute_liquid_volume()`  
**Zeile:** ~1170 (am Funktionsanfang)

**Code:**
```python
# === DEBUG NUC: VOR VERTEILUNG ===
if getattr(self.solver, 'mcpbe_debug_mass', False) and getattr(self.solver, 'mcpbe_debug_nuc', True):
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
    print(f"  Distributing droplets: volume={v_liquid_to_distribute:.6e} m^3")
    print(f"  BEFORE: V_solid_total={solid_before:.6e}, V_liq_total={liq_before:.6e}")
# ============================
```

**Warum:** Dokumentation vor Flüssigkeitszugabe.

---

### PUNKT 9: Nucleation - Nach Tropfenverteilung
**Datei:** `mcpbe_nucleation.py`  
**Funktion:** `_distribute_liquid_volume()`  
**Zeile:** ~1380 (am Funktionsende)

**Code:**
```python
# === DEBUG NUC: NACH VERTEILUNG ===
if getattr(self.solver, 'mcpbe_debug_mass', False) and getattr(self.solver, 'mcpbe_debug_nuc', True):
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
**Zeile:** ~310 (nach `active` Maskenberechnung)

**Code:**
```python
# === DEBUG COMP: VOR KOMPRESSION ===
if getattr(solver, 'mcpbe_debug_mass', False) and getattr(solver, 'mcpbe_debug_comp', True):
    v_dry = solver.V_flat[-1, :solver.a_tot]
    poro = solver.porosity[:solver.a_tot]
    w = solver.W[:solver.a_tot]
    valid = ~np.isnan(poro)
    v_solid = np.zeros_like(v_dry)
    v_solid[valid] = v_dry[valid] * (1.0 - poro[valid])
    v_solid[~valid] = v_dry[~valid]
    
    solid_before = np.sum(v_solid * w)
    print(f"\n[DEBUG COMP] t={getattr(solver, '_elapsed', 0.0):.4f}s, dt={dt:.4f}s")
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
**Zeile:** ~380 (am Funktionsende)

**Code:**
```python
# === DEBUG COMP: NACH KOMPRESSION ===
if getattr(solver, 'mcpbe_debug_mass', False) and getattr(solver, 'mcpbe_debug_comp', True):
    v_dry_after = solver.V_flat[-1, :solver.a_tot]
    poro_after = solver.porosity[:solver.a_tot]
    w_after = solver.W[:solver.a_tot]
    valid_after = ~np.isnan(poro_after)
    v_solid_after = np.zeros_like(v_dry_after)
    v_solid_after[valid_after] = v_dry_after[valid_after] * (1.0 - poro_after[valid_after])
    v_solid_after[~valid_after] = v_dry_after[~valid_after]
    
    solid_after = np.sum(v_solid_after * w_after)
    liq_after = np.sum(solver.liquid_volume[:solver.a_tot] * w_after)
    
    print(f"  AFTER:  V_solid_total={solid_after:.6e}, V_liq_total={liq_after:.6e}")
    print(f"  ΔV_solid={solid_after - solid_before:.6e} (SOLLTE = 0 sein!)")
    print(f"  poro_range=[{np.nanmin(poro_after):.4f}, {np.nanmax(poro_after):.4f}]")
# ================================
```

**Warum:** Validiert dass Kompression nur V_dry ändert, nicht V_solid.

---

### PUNKT 12: Control Volume Doubling
**Datei:** `mcpbe_base.py`  
**Funktion:** `_maybe_double_control_volume()`  
**Zeile:** ~1610 (nach Verdopplung)

**Code:**
```python
# === DEBUG VC-DOUBLE ===
if getattr(self, 'mcpbe_debug_mass', False):
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
**Zeile:** ~2185 (im Save-Loop)

**Code:**
```python
# === DEBUG SAVE-POINT MASSENBILANZ ===
if getattr(self, 'mcpbe_debug_mass', False):
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
        error_solid = (solid_total - solid_0) / solid_0 * 100 if solid_0 > 0 else 0.0
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

## Zusammenfassung Debug-Punkte

| Kategorie | Debug-Punkte | Dateien |
|-----------|--------------|---------|
| Initialisierung | 1 (Punkt 1) | mcpbe_base.py |
| Agglomeration | 3 (Punkte 2-4) | mcpbe_agg.py |
| Breakage | 3 (Punkte 5-6, 6b) | mcpbe_break.py |
| **ParticleMerger** | 4 (Punkte 6c-6f) | particle_merger.py |
| **Nucleation** | 5 (Punkte 7a-7c, 8-9) | mcpbe_nucleation.py |
| Continuous Processes | 2 (Punkte 10-11) | mcpbe_continuous_processes.py |
| Vc-Doubling | 1 (Punkt 12) | mcpbe_base.py |
| Save-Points | 1 (Punkt 13) | mcpbe_base.py |
| **SUMME** | **20 Debug-Punkte** | **6 Dateien** |

---

## Aktivierung der Debug-Prints

### Feinere Auffächerung über Flags

| Flag | Zweck | Betroffene Punkte |
|------|-------|-------------------|
| `solver.mcpbe_debug_mass` | **Hauptflag für Masseerhaltung** | ALLE Punkte 1-13 |
| `solver.mcpbe_debug_agg` | Nur Agglomeration | Punkte 2-4 |
| `solver.mcpbe_debug_break` | Nur Breakage | Punkte 5-6, 6b |
| `solver.mcpbe_debug_merger` | Nur ParticleMerger | Punkte 6c-6f |
| `solver.mcpbe_debug_nuc` | Nur Nucleation | Punkte 7a-7c, 8-9 |
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

1. ✅ **Implementierung:** Alle 20 Debug-Punkte mit konditionalen Prints versehen
2. ✅ **Test anpassen:** `solver.mcpbe_debug_mass = True` in Testdatei setzen
3. ⏳ **Log-Analyse:** Ersten signifikanten Massefehler identifizieren (>1e-10 relative Abweichung)
4. ⏳ **Fix:** Spezifischen Fehler beheben an identifizierter Stelle
5. ⏳ **Validierung:** Test erneut laufen lassen und Masseerhaltung bestätigen
