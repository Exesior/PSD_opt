# Control Volume Doubling Analyse

## Ziel von CV-Doubling

Wenn `a_tot` zu klein wird (wenige computational particles), wird das Control Volume verdoppelt um:
- Mehr computational particles zu haben (bessere Statistik)
- Aber: **Selbe PHYSIKALISCHE Teilchenzahl** beibehalten!

## Mathematik

### Vor Doubling:
```
n_comp = 100
W[k] = 100 für alle k
Vc = 0.01 m³
liquid_volume[k] = v_d für alle k

Physikalische Teilchen: Σ W[k] / Vc = 100 × 100 / 0.01 = 1.000.000
Gesamtflüssigkeit: Σ lv[k] × W[k] / Vc = 100 × v_d × 100 / 0.01 = 1.000.000 × v_d
```

### Nach Doubling (korrekt):
```
n_comp = 200 (verdoppelt!)
W[k] = 100 für alle k (UNVERÄNDERT pro computational particle!)
Vc = 0.02 m³ (verdoppelt!)
liquid_volume[k] = v_d für alle k (UNVERÄNDERT - intensive Größe!)

Physikalische Teilchen: Σ W[k] / Vc = 200 × 100 / 0.02 = 1.000.000 ✓
Gesamtflüssigkeit: Σ lv[k] × W[k] / Vc = 200 × v_d × 100 / 0.02 = 1.000.000 × v_d ✓
```

### WICHTIG: W wird NICHT verdoppelt!

Im Code steht:
```python
W_dup = np.concatenate((W_active, W_active))
```

Das ist **Duplizieren**, nicht **Multiplizieren**! 
- Vorher: W = [100, 100, ..., 100] (100 mal)
- Nachher: W = [100, 100, ..., 100, 100, ..., 100] (200 mal)

Jeder Eintrag ist immer noch 100, nicht 200!

## Meine Korrektur war RICHTIG!

```python
lv_new[:old_a] = lv_active           # Original values
lv_new[old_a:self.a_tot] = lv_active  # Duplicate for new slots
```

Das ergibt:
- Vorher: lv = [v_d, v_d, ..., v_d] (100 mal)
- Nachher: lv = [v_d, v_d, ..., v_d, v_d, ..., v_d] (200 mal)

## Warum dann +9827% Fehler?

### Hypothese: Agglomeration + CV-Doubling Interaktion

Stell dir vor:
1. Nucleation fügt Tropfen hinzu → viele Partikel mit lv=v_d
2. Agglomeration erzeugt Child mit lv=2×v_d
3. CV-Doubling tritt auf → Partikel werden dupliziert

**Problem:** Wenn ein Partikel mit lv=2×v_d dupliziert wird, haben wir plötzlich ZWEI Partikel mit je lv=2×v_d!

### Beispiel-Sequenz:

**t=0.1s:**
- 50 Partikel mit lv=v_d, W=100
- Gesamt: 50×v_d×100/0.01 = 500.000×v_d

**t=0.2s (nach einiger Agglomeration):**
- 40 Partikel mit lv=v_d, W=90
- 5 Partikel mit lv=2×v_d, W=10 (Children)
- Gesamt: 40×v_d×90/0.01 + 5×2×v_d×10/0.01 = 360.000×v_d + 100.000×v_d = 460.000×v_d

**t=0.3s (CV-Doubling!):**
- 80 Partikel mit lv=v_d, W=90
- 10 Partikel mit lv=2×v_d, W=10
- Vc = 0.02
- Gesamt: 80×v_d×90/0.02 + 10×2×v_d×10/0.02 = 360.000×v_d + 100.000×v_d = 460.000×v_d ✓

Das sieht korrekt aus!

### Alternative Hypothese: Nucleation nach CV-Doubling

Was wenn Nucleation WEITERMACHEN nach CV-Doubling?

**t=0.3s (nach CV-Doubling):**
- Vc = 0.02
- Nucleation verteilt neuen Tropfen mit lv=v_d an zufälliges Partikel k
- ABER: Welches Partikel? Und wie wird lv aktualisiert?

In `_distribute_one_droplet_with_dW`:
```python
self.solver.liquid_volume[idx] += droplet_volume
```

Das addiert v_d zum existierenden lv[k]. Wenn k schon lv=2×v_d hatte (von Agg), hat es jetzt lv=3×v_d.

**Aber das ist KORREKT!** Das Partikel hat jetzt 3 Tropfen gesammelt.

### Dritte Hypothese: _liquid_remainder Bug

In der Nucleation-Schleife:
```python
n_droplets_full = int(n_droplets_to_add)
liquid_to_add = n_droplets_full * self.config.droplet_volume
self._liquid_remainder += liquid_to_add

# ... distribute ...

self._liquid_remainder -= distributed_liquid
```

Wenn `_liquid_remainder` falsch akkumuliert wird, könnten zu viele Tropfen verteilt werden!

---

## Diagnose

Ich brauche Logging um zu sehen:
1. Wann tritt CV-Doubling auf?
2. Wie verändert sich Σ lv×W/Vc nach jedem Doubling?
3. Stimmt die Anzahl verteilter Tropfen mit der erwarteten überein?
