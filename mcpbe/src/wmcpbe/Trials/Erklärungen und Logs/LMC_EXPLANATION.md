# 🔬 Was ist LMC? (Lattice Monte Carlo)

**Datum:** 2026-07-13  
**Zweck:** Erklärung von LMC für die Breakage-Discrepancy-Analyse

---

## 📖 Grundkonzept

**LMC = Lattice Monte Carlo**

Eine **physikalisch detaillierte Methode** zur Vorhersage von **Fragmentverteilungen** beim Partikelbruch.

---

## 🎯 Problemstellung

Wenn ein Partikel bricht, entstehen mehrere Fragmente. Fragen:

1. **Wie viele Fragmente?** (2? 4? 10? 100?)
2. **Welche Größen haben die Fragmente?** (Alle gleich? Eine Verteilung?)
3. **Wie ist die Zusammensetzung?** (Bei multi-component particles)

### Einfache Antwort (klassisches PBE):
```python
frag_num = 4  # Immer genau 4 Fragmente
Jedes Fragment bekommt V_parent / 4  # Alle gleich groß
```

### Komplexe Antwort (LMC):
```python
# Physikalisch fundierte Verteilung basierend auf:
# - Parent-Partikelgröße
# - Materialparametern
# - Bruchmechanismus (Sprödbruch, Duktiler Bruch, etc.)
# - Energiebilanz

Anzahl Fragmente ~ Poisson-verteilt um Mittelwert 10-1000
Fragmentgrößen ~ Verteilung aus Lattice-Simulation
```

---

## 🔧 Wie funktioniert LMC?

### Schritt 1: Lattice-Repräsentation

Das Parent-Partikel wird als **3D-Gitter (Lattice)** diskretisiert:

```
Parent-Partikel (V = 1000 Einheiten):
┌─────────────────────────────────────┐
│ ████░░░░████░░░░████░░░░████░░░░   │  ← Phase A (40%)
│ ░░░░████░░░░████░░░░████░░░░████   │  ← Phase B (60%)
│ ... 1000 Gitterzellen insgesamt    │
└─────────────────────────────────────┘
```

### Schritt 2: Rissbildung simulieren

LMC simuliert physikalische **Crack-Propagation**:

```python
for step in range(1000):
    # Wähle zufällige Startzelle für Riss
    start_cell = rng.choice(lattice_cells)
    
    # Wachse Riss basierend auf:
    # - Lokaler Spannung
    # - Materialeigenschaften
    # - Bereits existierenden Rissen
    
    if crack_connects():  # Riss geht durch ganzes Partikel
        fragments.append(extract_fragment())
```

### Schritt 3: Statistik sammeln

Nach vielen Simulationen entsteht eine **Verteilung**:

```
Fragment Size Distribution:
  1% Volume  → 500 Fragmente (sehr klein)
  5% Volume  → 200 Fragmente
  10% Volume → 100 Fragmente
  25% Volume →  50 Fragmente
  50% Volume →  10 Fragmente (sehr groß)
```

---

## 💻 LMC im Code

### Zwei Implementierungsarten:

### Typ 1: **LMC Pre-Model** (`use_lmc_pre_model = True`)

**Offline-Berechnung:**
```python
# Einmalig (kann Stunden dauern!):
lmc_simulation.run(num_particles=10000, save_to_file='lmc_tables.h5')

# Zur Laufzeit (schnell!):
tables = load('lmc_tables.h5')
fragments = tables.sample(parent_volume)
```

**Code-Pfad:**
```python
def _build_fragments_stepwise(self, Vrem_k):
    *_, pexp = self._get_break_tables_for_state(Vrem_k)
    
    # Wenn use_lmc_pre_model=True:
    # pexp kommt aus LMC-Tabelle (kann 10-1000 sein!)
    p = float(pexp)  # z.B. pexp=500 → ~500 Fragmente!
    
    n = stochastic_round(p)  # ≈ 500
    
    for i in range(n-1):  # ← 499 Iterationen!
        frag = self._produce_one_frag_from_remaining(Vrem)
        # Jede Iteration: 1-2 RNG calls
        
    return frags  # Liste mit ~500 Fragmenten!
```

**RNG-Verbrauch pro Break:**
```
n_frags ≈ 500
RNG calls = 1 (stochastic rounding) + (n_frags-1) × 1-2 (CDF sampling)
          = 1 + 499 × 2
          = 999 RNG calls pro Breakage-Event!
```

---

### Typ 2: **LMC Live** (`use_lmc_live = True`)

**Online-Berechnung:**
```python
# Bei JEDER Breakage-Simulation:
def _break_build_fragments(Vrem_k):
    if use_lmc_live:
        frags, energy = lmc_live.sample_one_shot(Vrem_k, rng)
        # Führt echte Lattice-Simulation durch!
        # Sehr langsam, sehr viele RNG calls
        return frags
```

**RNG-Verbrauch pro Break:**
```
Lattice-Größe = 1000 Zellen
Simulationsschritte = 1000-10000
RNG calls ≈ 1000-10000 pro Breakage-Event!
```

---

## 📊 Vergleich: Klassisch vs. LMC

| Aspekt | Klassisch (`frag_num=4`) | LMC Pre-Model | LMC Live |
|--------|-------------------------|---------------|----------|
| **Fragmente pro Break** | 4 (fix) | 10-1000 (verteilt) | 10-1000 (verteilt) |
| **RNG calls pro Break** | ~5-10 | ~100-2000 | ~1000-10000 |
| **Physikalische Details** | Keine | Mittel | Sehr hoch |
| **Rechenzeit** | Nanosekunden | Mikrosekunden | Millisekunden |
| **Speicherbedarf** | Keiner | MB bis GB (Tabellen) | Keiner |
| **Anwendung** | Standard-PBE | Forschung | Detailed Studies |

---

## 🔍 LMC in deinem Code

### Konfigurationsparameter:

```python
# In mcpbe_base.py / _init_lmc():
self.use_lmc_pre_model = False  # ← Standard: AUS!
self.use_lmc_live = False       # ← Standard: AUS!
self.lmc_NO_FRAG = 4            # ← Nur relevant wenn LMC AN
self.frag_num = 4               # ← Klassischer Default
```

### Code-Pfade:

```python
# _build_fragments_stepwise() in mcpbe_break.py:
def _build_fragments_stepwise(self, Vrem_k):
    *_, pexp = self._get_break_tables_for_state(Vrem_k)
    
    # Entscheidung:
    if pexp is not None and self.use_lmc_pre_model:
        # LMC-PFAD 🔴
        p = float(pexp)  # Wert aus LMC-Tabelle (groß!)
    else:
        # KLASSISCHER PFAD ✅
        p = float(self.frag_num)  # Normalerweise 4
    
    # Restlicher Code gleich...
```

---

## 🚨 Warum LMC unser Problem erklärt

### Beobachtung:
```
LEGACY: 2003 RNG calls pro Break
NEW:      3 RNG calls pro Break
DIFFERENCE: 2000 calls
```

### Erklärung:

**LEGACY verwendet LMC Pre-Model:**
```python
use_lmc_pre_model = True  # Durch Config geladen!
frag_num aus Tabelle = 1000  # LMC erwartet ~1000 Fragmente

RNG calls = 1 + (1000-1) × 2 ≈ 2000 ✓
```

**NEW verwendet klassischen Pfad:**
```python
use_lmc_pre_model = False  # Default!
frag_num = 4  # Hardcoded default

RNG calls = 1 + (4-1) × 2 = 7 ≈ 3 (wenn CDF caching)
```

---

## ❓ Warum wäre LMC aktiv in Legacy?

### Mögliche Ursachen:

1. **Config-Loading:**
   ```python
   # In __init__():
   if load_attr:
       self._load_attributes(config_path)  # ← Lädt use_lmc_pre_model=True!
   ```

2. **Default-Werte in `_init_lmc()`:**
   ```python
   self.use_lmc_pre_model = bool(getattr(self, "use_lmc_pre_model", False))
   # Wenn use_lmc_pre_model Attribut existiert (aus Config), wird es True!
   ```

3. **Unterschied zwischen `load_attr=False` und Config-Pfad:**
   ```python
   # Debug-Skript setzt load_attr=False
   # ABER: Vielleicht gibt es globale Defaults woanders?
   ```

---

## 🎯 Nächste Schritte für Option A

### Debug-Skript `debug_rng_location.py`:

```python
# Wrappe jede relevante Methode mit Counter:

methods_to_trace = {
    '_get_break_tables_for_state': 'Welche CDF-Tabelle wird geladen?',
    '_build_fragments_stepwise': 'Wie viele Fragmente werden erzeugt?',
    '_produce_one_frag_from_remaining': 'Wie viele RNG calls pro Fragment?',
    '_break_sampler.sample': 'Wie viele RNG calls im Sampler?',
}

# Nach einem Break-Event:
print("RNG Consumption by Method:")
for method, count in counters.items():
    print(f"  {method}: {count} calls")

# Erwartetes Ergebnis wenn LMC aktiv:
#   _get_break_tables_for_state: 1 call
#   _build_fragments_stepwise: 1 call
#   _produce_one_frag_from_remaining: 1000 calls ← HIER!
#   _break_sampler.sample: 1 call
```

---

## 💡 Meine Hypothese

**LEGACY hat `use_lmc_pre_model = True`** (durch Config oder Default), was zu:
- ~1000 Fragmenten pro Break führt
- ~2000 RNG calls verursacht
- Anderer RNG-Sequenz als NEW (der nur 4 Fragmente hat)

**FIX:** Beide Solver auf gleiche LMC-Konfiguration setzen:
```python
solver.use_lmc_pre_model = False
solver.use_lmc_live = False
solver.frag_num = 4  # Explizit setzen!
```

Aber erst wollen wir es **beweisen** mit Debug-Logging!

---

## 📝 Zusammenfassung

| Frage | Antwort |
|-------|---------|
| **Was ist LMC?** | Physikalisch detaillierte Fragmentverteilungs-Simulation |
| **Warum viele RNG calls?** | LMC erzeugt 10-1000 Fragmente (vs. klassisch 4) |
| **Ist LMC aktiv in unseren Tests?** | VERDACHT: Ja in Legacy, Nein in NEW |
| **Sollte LMC aktiv sein?** | NEIN (für reine Breakage-Validation zu komplex) |
| **Wie beweisen?** | Debug-Logging in `_build_fragments_stepwise()` |

---

Bist du einverstanden mit dieser Analyse? Soll ich das `debug_rng_location.py` Skript erstellen?
