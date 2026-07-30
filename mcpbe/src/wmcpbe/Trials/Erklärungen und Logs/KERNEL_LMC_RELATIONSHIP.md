# 🔗 Beziehung: Kernel-Framework vs. LMC

**Datum:** 2026-07-13  
**Frage:** Wird LMC vom neuen Kernel-Framework genutzt?

---

## 📊 Kurze Antwort

**NEIN!** LMC und Kernel-Framework sind **unabhängig**:

| Aspekt | Kernel-Framework | LMC |
|--------|------------------|-----|
| **Zweck** | Berechnet **Breakage-RATE** S(V) | Berechnet **Fragment-VERTEILUNG** |
| **Ort im Code** | `kernels/breakage/power_law.py` | `mcpbe_break.py` |
| **Wann ausgeführt** | Bei `_break_rate_single()` | Bei `_build_fragments_stepwise()` |
| **Konfiguration** | `break_kernel_name`, `break_kernel_params` | `use_lmc_pre_model`, `frag_num` |
| **Beeinflusst RNG?** | ~1 call pro Rate-Berechnung | ~100-2000 calls pro Fragment-Generation |

---

## 🏗️ Architektur-Übersicht

```
┌─────────────────────────────────────────────────────────────┐
│                    MCPBEBase Solver                         │
│                                                             │
│  ┌───────────────────┐      ┌───────────────────────────┐  │
│  │ Kernel Framework  │      │   LMC (in mcpbe_break)    │  │
│  │                   │      │                           │  │
│  │ • break_kernel    │      │ • use_lmc_pre_model       │  │
│  │   → PowerLaw      │      │ • use_lmc_live            │  │
│  │   → compute_rate()│      │ • frag_num                │  │
│  │   → S(V) = Rate   │      │ • lmc_adapter             │  │
│  │                   │      │                           │  │
│  │ NEW: Ja           │      │ • _build_fragments_...    │  │
│  │ LEGACY: JIT       │      │ • _produce_one_frag_...   │  │
│  └───────────────────┘      └───────────────────────────┘  │
│           ↓                              ↓                  │
│    Rate wird berechnet            Fragmente erzeugt         │
│    (1 RNG call)                 (100-2000 RNG calls!)       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔍 Code-Analyse

### 1. Kernel-Framework (Rate Calculation)

**Datei:** `kernels/breakage/power_law.py`

```python
class PowerLawBreakageKernel(BreakageKernel):
    def compute_rate(self, v_particle: float, ...) -> float:
        """Berechnet NUR die Breakage-Rate S(V)."""
        
        if self.breakrval == 1:
            rate = self.p1  # Konstante Rate
        elif self.breakrval == 4:
            rate = self.p1 * (self.g ** self.p2) * (v_particle ** alpha)
        # ...
        
        return rate  # ← EINZIGER Output: Rate [1/s]
```

**Verwendung in `mcpbe_break.py`:**

```python
def _break_rate_single(self, i: int) -> float:
    """Berechnet propensity für Partikel i."""
    
    # Branch 0: Kernel Framework (NEW only)
    if hasattr(self, 'kernel_manager') and self.kernel_manager.break_kernel is not None:
        v_particle = float(self.V_flat[-1, i])
        Si = self.kernel_manager.break_kernel.compute_rate(v_particle, ...)
        # ← HIER wird Kernel aufgerufen!
        
    # Branch 1/2: Legacy JIT (BACKUP & NEW fallback)
    else:
        Si = _kb_br1_single(...)  # JIT function
    
    propensity = W[i] * Si / delta_i
    return propensity
```

**RNG-Verbrauch:** ~0-1 calls (nur für Rate, keine Fragmente!)

---

### 2. LMC (Fragment Distribution)

**Datei:** `mcpbe_break.py`

```python
def _build_fragments_stepwise(self, Vrem_k: np.ndarray) -> list[np.ndarray]:
    """Erzeugt Fragment-Liste nach Breakage-Event."""
    
    *_, pexp = self._get_break_tables_for_state(Vrem_k)
    
    # Entscheidung: LMC oder klassisch?
    if pexp is not None and getattr(self, "use_lmc_pre_model", False):
        # 🔴 LMC-PFAD
        p = float(pexp)  # Wert aus LMC-Tabelle (z.B. 1000!)
    else:
        # ✅ KLASSISCH
        p = float(getattr(self, "frag_num", 2))  # Default: 4
    
    n = stochastic_round(p)  # Anzahl Fragmente
    
    frags = []
    for _ in range(n - 1):  # ← Schleife über ALLE Fragmente!
        frag = self._produce_one_frag_from_remaining(Vrem)
        # Jede Iteration: 1-2 RNG calls!
        frags.append(frag)
        Vrem -= frag
    
    frags.append(Vrem)  # Letztes Fragment
    return frags
```

**RNG-Verbrauch:**
- Klassisch (n=4): 1 + 3×2 = **7 calls**
- LMC (n=1000): 1 + 999×2 = **~2000 calls** 🔴

---

### 3. Wo wird was aufgerufen?

**In `_do_one_break()`:**

```python
def _do_one_break(self):
    # Schritt 1: Wähle Partikel zum Brechen
    k = self._break_sampler.sample(self._rng)  # ← 1 RNG call
    
    # Schritt 2: Berechne dW (Packet-Größe)
    dW = self._compute_dW_packet(k)  # ← Kein RNG
    
    # Schritt 3: Erzeuge Fragmente 🔴
    status, frags = self._break_build_fragments(Vrem_k)
    #   ↓ ruft auf:
    #   _build_fragments_stepwise()
    #     ↓ ruft auf (n-1 mal):
    #     _produce_one_frag_from_remaining()  ← (n-1) × RNG calls!
    
    # Schritt 4: Füge Fragmente hinzu, update Propensities
    self._break_apply_and_maintain(k, frags, dW, Vrem_k)
    #   ↓ ruft auf (pro Fragment):
    #   _break_rate_single(new_idx)
    #     ↓ ruft auf:
    #     kernel.compute_rate()  ← Kein/Few RNG calls
```

---

## 🎯 Fazit

### Kernel-Framework:
- **NUR für Breakage-RATE** verantwortlich
- **KEIN Einfluss** auf Fragment-Anzahl
- Wird in `_break_rate_single()` aufgerufen
- **Unabhängig von LMC!**

### LMC:
- **NUR für Fragment-VERTEILUNG** verantwortlich
- Bestimmt **Anzahl Fragmente pro Break**
- Wird in `_build_fragments_stepwise()` aufgerufen
- **Unabhängig vom Kernel-Framework!**

---

## 🔧 Konsequenz für unseren Bug

### Hypothese bestätigt:

| Solver | Kernel-Framework | LMC-Konfiguration | Fragmente/Break |
|--------|------------------|-------------------|-----------------|
| **LEGACY** | ❌ Nein (JIT) | `use_lmc_pre_model=True` (?) | ~1000 🔴 |
| **NEW** | ✅ Ja (PowerLaw) | `use_lmc_pre_model=False` (Default) | 4 ✅ |

### Warum unterschiedliche LMC-Konfiguration?

**LEGACY:**
```python
legacy_solver = MCPBESolver(..., load_attr=False, ...)
# ABER: Vielleicht wurde use_lmc_pre_model woanders gesetzt?
# ODER: _init_lmc() setzt Default-Werte anders?
```

**NEW:**
```python
new_solver = MCPBESolver(..., load_attr=False, ...)
# use_lmc_pre_model = bool(getattr(self, "use_lmc_pre_model", False))
# = False (Default)
```

### Lösung:

Beide Solver müssen **gleiche LMC-Konfiguration** haben:

```python
# In BEIDEN Solvern VOR _init_lmc():
solver.use_lmc_pre_model = False
solver.use_lmc_live = False
solver.frag_num = 4  # Explizit setzen!
```

Oder besser: In Debug-Skript `debug_rng_location.py` wird Phase 1 genau das zeigen!

---

## 📝 Zusammenfassung

| Frage | Antwort |
|-------|---------|
| **Nutzt Kernel-Framework LMC?** | ❌ NEIN! Völlig unabhängig |
| **Wo ist LMC implementiert?** | `mcpbe_break.py` (nicht in `kernels/`) |
| **Was verursacht 2000 RNG calls?** | LMC mit ~1000 Fragmenten/Break |
| **Warum hat LEGACY LMC aktiv?** | Vermutlich durch Config-Loading oder Default |
| **Wie fixen?** | Beide Solver auf `use_lmc_pre_model=False` setzen |

---

**Nächster Schritt:** `debug_rng_location.py` ausführen um zu beweisen, dass LEGACY LMC verwendet!
