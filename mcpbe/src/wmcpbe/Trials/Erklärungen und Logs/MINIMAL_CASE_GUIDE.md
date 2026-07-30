# 📦 Minimaler Case-Aufbau für WMCPBE Solver

## 🎯 Die absoluten Minimum-Parameter

Für eine **funktionierende** Agglomerations-Simulation brauchst du nur:

```python
from wmcpbe.mcpbe import MCPBESolver

solver = MCPBESolver(
    dim=1,           # 1D (nur Volumen) oder 2D (Volumen + Porosität)
    t_total=10.0,    # Simulationszeit [s]
    t_write=5,       # Anzahl Snapshots
    verbose=False,   # Keine Logs
    init=False,      # Manuelle Initialisierung (wichtig!)
    seed=42,         # Reproduzierbarkeit
)

# ──────────────────────────────────────────────
# PFLICHTANGABEN (6 Stück!)
# ──────────────────────────────────────────────

# 1️⃣ Partikelgröße (Durchmesser [m])
solver.x = np.array([3e-6])  # 3 µm Partikel

# 2️⃣ Verteilungstyp
solver.PGV = np.array(["mono"])  # "mono" | "norm" | "weibull"

# 3️⃣ Streuung (bei "mono" irrelevant)
solver.SIG = np.array([0.0])

# 4️⃣ Prozess-Typ
solver.process_type = "agglomeration"  # "agglomeration" | "breakage" | "mix"

# 5️⃣ Agglomerations-Kernel
solver.COLEVAL = 3  # 3 = konstanter Kernel

# 6️⃣ Agglomerations-Stärke
solver.CORR_BETA = 1e-9  # Höher = stärkere Agglomeration

# ──────────────────────────────────────────────
# INITIALISIERUNG
# ──────────────────────────────────────────────

solver._initialize_particles()  # ← Erstellt Partikel
solver._initialize_samplers()   # ← Baut Sampler für MC-Events

# ──────────────────────────────────────────────
# SIMULATION STARTEN
# ──────────────────────────────────────────────

solver.solve(maxiter=1000)  # Maximal 1000 Events

# ──────────────────────────────────────────────
# ERGEBNISSE AUSLESEN
# ──────────────────────────────────────────────

print(f"Partikel am Start: {solver.a_tot}")
print(f"Partikel am Ende: {solver.a_tot}")
print(f"Simulierte Zeit: {solver._elapsed:.3f} s")
```

---

## 📊 Vollständiges Minimal-Beispiel

```python
import numpy as np
from wmcpbe.mcpbe import MCPBESolver

# 1. Solver erstellen
solver = MCPBESolver(
    dim=1,
    t_total=10.0,
    t_write=5,
    verbose=False,
    init=False,  # Wichtig: Keine automatische Init!
    seed=42,
)

# 2. Minimale Parameter setzen (NUR 6!)
solver.x = np.array([3e-6])           # 3 µm Partikel
solver.PGV = np.array(["mono"])       # Monodispers
solver.SIG = np.array([0.0])          # Keine Streuung
solver.process_type = "agglomeration" # Nur Agglomeration
solver.COLEVAL = 3                    # Konstanter Kernel
solver.CORR_BETA = 1e-9               # Agglomerationsstärke

# 3. Initialisieren
solver._initialize_particles()
solver._initialize_samplers()

# 4. Simulieren
solver.solve(maxiter=1000)

# 5. Ergebnisse
print(f"✓ Simulation fertig nach {solver._elapsed:.3f}s")
print(f"✓ Partikel: {solver.a0} → {solver.a_tot}")
```

---

## 🔧 Optionale Parameter (für mehr Kontrolle)

### **Partikelanzahl explizit setzen**

Standardmäßig wird die Anzahl aus `c` (Konzentration) und `Vc` (Control Volume) berechnet:

```python
# Standard (kompliziert):
solver.c = np.array([1e8])  # Konzentration [1/m³]
solver.Vc = 1e-12           # Control Volume [m³]
# → a0 = c * Vc = 100 Partikel

# Einfacher: Direkt a0 nutzen mit solve_repeats()
results, _ = solver.solve_repeats(
    N=1,
    base_seed=42,
    init_Vc=False,
    Vc=1e-12,
    V_flat=initial_V_flat,  # Selbst erstellte Partikel
    W_init=initial_W,       # Gewichte dazu
)
```

### **Breakage hinzufügen**

```python
solver.process_type = "mix"  # Agglomeration + Breakage

# Breakage-Parameter (3 Stück)
solver.BREAKRVAL = 1        # Breakage-Rate
solver.BREAKFVAL = 2        # Fragmente pro Event (2 = binär)
solver.pl_P1 = 0.5          # Breakage-Wahrscheinlichkeit
```

### **Größenverteilung**

```python
# Monodispers (alle gleich groß)
solver.x = np.array([3e-6])
solver.PGV = np.array(["mono"])

# Normalverteilt
solver.x = np.array([3e-6])     # Mittelwert
solver.PGV = np.array(["norm"])
solver.SIG = np.array([0.3])    # 30% Standardabweichung

# Weibull-verteilt
solver.x = np.array([3e-6])
solver.PGV = np.array(["weibull"])
solver.SIG = np.array([1.5])    # Formparameter
```

### **Mehrere Sorten (dim > 1)**

```python
solver = MCPBESolver(dim=2, ...)  # 2 Sorten

solver.x = np.array([1e-6, 5e-6])      # 1 µm + 5 µm
solver.c = np.array([0.5e-3, 0.5e-3])  # Je 50%
solver.PGV = np.array(["mono", "mono"])
solver.SIG = np.array([0.0, 0.0])
```

---

## ⚙️ Parameter-Referenz (Kurzfassung)

| Kategorie 	| Parameter 	| Pflicht? | Default 		| Beschreibung |
|---------------|---------------|----------|--------------------|--------------|
| **Basis** 	| `dim`     	| ✅ 	   | 2      		| Dimension (1 oder 2) |
|          	| `t_total` 	| ✅      | 601     		| Simulationszeit [s] |
|           	| `t_write` 	| ✅      | 10      		| Snapshots |
|           	| `init`   	| ❌      | True    		| Auto-Init (False für manuell) |
| **Partikel**	| `x` 		| ✅ 	   | 1e-6 		| Durchmesser [m] |
| 		| `PGV` 	| ✅      | "mono" 		| Verteilung ("mono"/"norm"/"weibull") |
| 		| `SIG` 	| ✅      | 0.1 		| Streuung |
| **Prozess** 	| `process_type`| ✅	   | "agglomeration" 	| "agglomeration"/"breakage"/"mix" |
| **Agglo** 	| `COLEVAL` 	| ✅* 	   | - 			| Kernel-Typ (3 = konstant) |
| 		| `CORR_BETA` 	| ✅*     | - 			| Agglomerationsstärke |
| **Break** 	| `BREAKRVAL` 	| ✅*     | - 			| Breakage-Rate |
| 		| `BREAKFVAL` 	| ✅*     | - 			| Fragmente/Event |
| 		| `pl_P1` 	| ✅*     | - 			| Breakage-Wahrscheinlichkeit |

*nur wenn `process_type` es erfordert

---

## 🚨 Häufige Fehler

### ❌ **Fehler 1: `init=True` vergessen**

```python
# FALSCH:
solver = MCPBESolver(dim=1, ...)
solver.x = [...]  # Zu spät! Init lief schon mit Defaults

# RICHTIG:
solver = MCPBESolver(dim=1, ..., init=False)
solver.x = [...]  # Parameter setzen
solver._initialize_particles()  # Jetzt init
```

### ❌ **Fehler 2: Falsche Array-Längen**

```python
# FALSCH:
solver.dim = 1
solver.x = np.array([1e-6, 2e-6])  # 2 Werte für dim=1!

# RICHTIG:
solver.dim = 1
solver.x = np.array([1e-6])  # 1 Wert für dim=1
```

### ❌ **Fehler 3: process_type ohne Parameter**

```python
# FALSCH:
solver.process_type = "mix"
# Aber keine Breakage-Parameter gesetzt!

# RICHTIG:
solver.process_type = "mix"
solver.BREAKRVAL = 1
solver.BREAKFVAL = 2
solver.pl_P1 = 0.5
```

---

## 📈 Typische Use-Cases

### **Use Case 1: Reine Agglomeration**

```python
solver.process_type = "agglomeration"
solver.COLEVAL = 3
solver.CORR_BETA = 1e-9  # Anpassen für gewünschte Rate
```

### **Use Case 2: Reine Breakage**

```python
solver.process_type = "breakage"
solver.BREAKRVAL = 2
solver.BREAKFVAL = 2  # Binäre Fragmentierung
solver.pl_P1 = 1.0    # Immer brechen
```

### **Use Case 3: Gemischt (Granulation)**

```python
solver.process_type = "mix"
solver.COLEVAL = 3
solver.CORR_BETA = 1e-12  # Schwache Agglo
solver.BREAKRVAL = 1
solver.BREAKFVAL = 2
solver.pl_P1 = 0.3        # 30% Breakage-Wahrscheinlichkeit
```

### **Use Case 4: Mit Nucleation (Liquid Addition)**

```python
solver.process_type = "mix"
solver.COLEVAL = 3
solver.CORR_BETA = 1e-11
solver.BREAKRVAL = 1
solver.BREAKFVAL = 2
solver.pl_P1 = 0.1

solver._initialize_particles()
solver._initialize_samplers()

# Nucleation hinzufügen
solver.create_nucleation_handler(
    enabled=True,
    volumenstrom=1e-15,      # m³/s
    tropfen_durchmesser=1e-6, # m
    wasserzugabe_start=0.0,
    wasserzugabe_dauer=5.0,
)

solver.solve(maxiter=500)
```

---

## 🎯 Fazit: Das absolute Minimum

**6 Parameter + 3 Methodenaufrufe:**

```python
solver = MCPBESolver(dim=1, t_total=10, t_write=5, init=False, seed=42)
solver.x = np.array([3e-6])
solver.PGV = np.array(["mono"])
solver.SIG = np.array([0.0])
solver.process_type = "agglomeration"
solver.COLEVAL = 3
solver.CORR_BETA = 1e-9
solver._initialize_particles()
solver._initialize_samplers()
solver.solve(maxiter=1000)
```

**Das war's!** 🎉
