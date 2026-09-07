# Compression Handler Migration Guide

## ⚠️ Wichtige Änderung: Compression-Code wurde vollständig entfernt

Folgende Komponenten wurden **vollständig entfernt**:

| Komponente | Status |
|------------|--------|
| `kernels/compression/` | ❌ ENTFERNT (seit v2.1) |
| `mcpbe_compression.py` | ❌ ENTFERNT (seit v2.1) |
| `CompressionHandler` | ❌ ENTFERNT (seit v2.1) |
| `CompressionConfig` | ❌ ENTFERNT (seit v2.1) |
| `create_compression_handler()` | ❌ ENTFERNT (seit v2.1) |

Alle Kompressions-Funktionalitäten sind jetzt ausschließlich im Modul `wmcpbe.kernels.continuous_processes` verfügbar.

---

## 📋 Was wurde entfernt?

| Komponente | Status | Ersatz |
|------------|--------|--------|
| `kernels/compression/` | ❌ ENTFERNT | `kernels/continuous_processes/` |
| `get_compression_kernel()` | ❌ ENTFERNT | `get_continuous_kernel('porosity_compression')` |
| `ExponentialDecayKernel` | ❌ ENTFERNT | `PorosityCompressionKernel` |
| `StressCompactionKernel` | ❌ ENTFERNT | (noch nicht implementiert) |
| `mcpbe_compression.py` | ❌ ENTFERNT | `mcpbe_continuous_processes.py` |
| `CompressionHandler` | ❌ ENTFERNT | `ContinuousProcessesHandler` |
| `CompressionConfig` | ❌ ENTFERNT | `ContinuousProcessesConfig` |
| `create_compression_handler()` | ❌ ENTFERNT | `create_continuous_processes_handler()` | |

---

## 🔄 Migration: So aktualisieren Sie Ihren Code

### Beispiel 1: Solver mit Compression-Kernel (ALT)

```python
# ❌ NICHT MEHR VERFÜGBAR
solver = MCPBESolver(
    agg_kernel_name='shear_chin1998',
    agg_kernel_params={'corr_beta': 1e-3, 'g': 1000},
    compression_kernel_name='exponential_decay',  # REMOVED!
    compression_kernel_params={'rate': 0.02, 'min_porosity': 0.3},
)
```

### Beispiel 1: Mit ContinuousProcessesHandler (NEU)

```python
# ✅ EMPFOHLEN
solver = MCPBESolver(
    agg_kernel_name='shear_chin1998',
    agg_kernel_params={'corr_beta': 1e-3, 'g': 1000},
    # Kein compression_kernel_name mehr!
)

# ContinuousProcessesHandler erstellen
solver.create_continuous_processes_handler(
    enabled=True,
    compression_rate=0.02,    # 1/s
    min_porosity=0.3,         # Minimale Porosität
)
```

---

### Beispiel 2: CompressionHandler verwenden (ALT)

```python
# ❌ ENTFERNT - Funktioniert nicht mehr!
solver.create_compression_handler(
    enabled=True,
    rate=0.02,
    min_porosity=0.3,
)
# ImportError: cannot import name 'CompressionHandler'
```

### Beispiel 2: ContinuousProcessesHandler (NEU)

```python
# ✅ EMPFOHLEN
solver.create_continuous_processes_handler(
    enabled=True,
    compression_rate=0.02,    # Entspricht rate
    min_porosity=0.3,
    k_int=1e12,               # Optional: Liquid Internalization
)
```

---

## 🎯 Vorteile der Migration

| Feature | CompressionHandler | ContinuousProcessesHandler |
|---------|-------------------|----------------------------|
| Porositätskompression | ✅ | ✅ |
| Liquid Internalization | ❌ | ✅ |
| Performance (vectorized) | ❌ | ✅ |
| Kernel-Architektur | ❌ | ✅ |
| Zukunftssicher | ❌ | ✅ |

---

## 📦 Neue Kernel in `continuous_processes`

### 1. Porosity Compression Kernel

**Formel:** `ε(t) = ε_min + (ε₀ - ε_min) × exp(-rate × t)`

```python
from wmcpbe.kernels.continuous_processes import get_continuous_kernel

kernel = get_continuous_kernel(
    'porosity_compression',
    rate=0.02,          # 1/s
    min_porosity=0.3,   # [0, 1)
)

poro_new = kernel.compute(poro=0.5, dt=0.1)
```

**Drehzahlabhängige Varianten** (gleiche Formel, `rate` → `k`):

| Name | `k` |
|---|---|
| `porosity_compression_dynamik` | `rate · n_mixer^c_mixer` (Default `c_mixer = C_BREAK`; `c_mixer = 0` → identisch zu `porosity_compression`) |
| `porosity_compression_dynamik_rumpf` | `rate · n_mixer^c_mixer / σ(ε,S)` — durch die Rumpf-Festigkeit geteilt; `compute_array` braucht `solver=` (Sättigung, Sauter-Durchmesser) |

`n_mixer` muss zu jedem anderen drehzahlgetriebenen Kernel im Lauf passen
(`assert_consistent_mixer_speed`, harter Fehler).

### 2. Liquid Internalization Kernel

**Formel:** `dl_intern/dt = k_int × l_ex × (v_pore - l_intern)`

```python
kernel = get_continuous_kernel(
    'liquid_internalization',
    k_int=1e12,  # 1/(m³·s)
)

sat_new = kernel.compute(saturation=0.5, v_pore=1e-15, l_total=0.6e-15, dt=0.01)
```

### 3. Liquid Internalization Agglomeration Kernel

Poren-Einschluss bei Agglomeration (Braumann et al. 2007).

```python
kernel = get_continuous_kernel('liq_internalisation_agglomeration')

l_e_to_i = kernel.compute_internalization(
    v_dry1=1e-15, v_dry2=2e-15,
    v_liq_ext1=0.1e-15, v_liq_ext2=0.2e-15,
)
```

---

## 🔍 Häufige Fehler nach der Migration

### Fehler 1: Import von removed module

```python
# ❌ FEHLER
from wmcpbe.kernels.compression import get_compression_kernel

# ✅ KORREKT
from wmcpbe.kernels.continuous_processes import get_continuous_kernel
```

### Fehler 2: Alter Parameter im Solver

```python
# ❌ FEHLER
solver = MCPBESolver(compression_kernel_name='exponential_decay')

# ✅ KORREKT
solver = MCPBESolver()  # Ohne compression_kernel_name
solver.create_continuous_processes_handler(enabled=True, ...)
```

### Fehler 3: Deprecated Handler

```python
# ❌ FEHLER (aber funktioniert noch mit Warning)
solver.create_compression_handler(enabled=True, rate=0.02)

# ✅ KORREKT
solver.create_continuous_processes_handler(
    enabled=True,
    compression_rate=0.02,
)
```

---

## 📊 Vergleich: Alte vs. neue API

| Konzept | Alt (removed) | Neu (empfohlen) |
|---------|---------------|-----------------|
| **Modul** | `kernels/compression/` | `kernels/continuous_processes/` |
| **Factory** | `get_compression_kernel()` | `get_continuous_kernel()` |
| **Kernel-Name** | `'exponential_decay'` | `'porosity_compression'` |
| **Rate-Parameter** | `rate` | `compression_rate` |
| **Handler** | `CompressionHandler` | `ContinuousProcessesHandler` |
| **Erstellung** | `create_compression_handler()` | `create_continuous_processes_handler()` |

---

## 🛠️ Debugging-Tipps

### 1. Nach Upgrade-Fehler: Import nicht gefunden

```python
# ❌ FEHLER nach Upgrade auf v2.1
from wmcpbe import CompressionHandler
# ImportError: cannot import name 'CompressionHandler'

# ✅ LÖSUNG: Code auf ContinuousProcessesHandler umstellen
from wmcpbe import ContinuousProcessesHandler
```

### 2. Prüfen ob Handler existiert

```python
if hasattr(solver, 'continuous_processes') and solver.continuous_processes is not None:
    print("✅ ContinuousProcessesHandler aktiv")
else:
    print("❌ Kein Handler konfiguriert")
```

### 2. Statistik abrufen

```python
stats = solver.continuous_processes.get_statistics()
print(f"Kompressionsschritte: {stats['particles_compressed_total']}")
print(f"Internalisierungsschritte: {stats['particles_internalized_total']}")
```

### 3. Kernel direkt testen

```python
from wmcpbe.kernels.continuous_processes import get_continuous_kernel

kernel = get_continuous_kernel('porosity_compression', rate=0.02, min_porosity=0.3)
result = kernel.compute(porosity=0.5, dt=0.1)
print(f"Porosity nach 0.1s: {result:.3f}")
```

---

## 📚 Weiterführende Dokumentation

- **Kernel-Katalog**: `kernels/README.md` (seit 20.08.2026 der einzige; `KERNEL_UEBERSICHT.md` ist darin aufgegangen)
- **Quickstart**: `../../../docs/historical/QUICKSTART.md`
- **Fundamentals**: `../../../docs/historical/Fundamentals.md`
- **Continuous Processes Code**: `mcpbe_continuous_processes.py`

---

## ❓ FAQ

**F: Warum wurde das Modul entfernt?**  
A: Redundanz vermeiden, Konsistenz erhöhen, bessere Architektur für zeitkontinuierliche Prozesse.

**F: Funktioniert mein alter Code noch?**  
A: **Nein!** Seit v2.1 schlägt der Import fehl mit:
   ```
   ImportError: cannot import name 'CompressionHandler'
   ```
   Sie müssen Ihren Code auf `ContinuousProcessesHandler` umstellen.

**F: Ich habe ein Upgrade auf v2.1 gemacht und jetzt geht nichts mehr!**  
A: Folgen Sie diesem Migrations-Guide:
   1. Ersetzen Sie alle `create_compression_handler()` Aufrufe durch `create_continuous_processes_handler()`
   2. Ändern Sie `rate=` zu `compression_rate=`
   3. Entfernen Sie alle Importe von `CompressionHandler` / `CompressionConfig`

**F: Muss ich migrieren?**  
A: **Ja, zwingend erforderlich.** Der alte Code funktioniert nicht mehr.

**F: Verliere ich Funktionalität?**  
A: Nein, `ContinuousProcessesHandler` bietet alle Features + zusätzliche Funktionen (Liquid Internalization).

**F: Gibt es eine Backward-Compatibility-Schicht?**  
A: Nein. Ein vollständiger Break war notwendig für saubere Architektur.

---

*Stand: Juli 2026*  
*WMCPBE Development Team*
