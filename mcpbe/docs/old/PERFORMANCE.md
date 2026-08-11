# WMCPBE — Performance-Analyse und Ergebnisse

**Ziel des Refactorings:** Simulationen schneller machen, ohne die Physik zu
verändern.

**Umgebung aller Messungen:** Windows 11, Python 3.11.9, numpy 2.3.5,
numba 0.63.1. Baseline ist Commit `7a35310`, gemessen aus einem separaten
Git-Worktree mit identischer Harness, damit „vorher“ und „nachher“ exakt
dasselbe messen.

---

## 1. Ergebnis in einer Zeile

| Modus | Gesamtlaufzeit Benchmark-Suite | Faktor | Ergebnisse bitgenau? |
|---|---|---|---|
| Baseline (`7a35310`) | 14.90 s | 1.0× | — |
| **`pairwise`** (Default) | **3.60 s** | **4.14×** | **ja** |
| `moment` (Opt-in) | 2.10 s | **7.11×** | nein (~1e-15 relativ) |

---

## 2. Wo die Zeit hinging

Alle Aussagen stammen aus `cProfile` **nach** JIT-Warmup — ohne Warmup
dominieren mehrere hundert Millisekunden LLVM-Kompilierzeit die Messung und die
Profile sind wertlos.

### 2.1 Agglomerationslauf (`agg_shear_1d_large`, n≈1900, 149 Ereignisse)

```
ncalls  tottime  percall  filename:lineno(function)
   145    0.549    0.004  jit_kernels.py:54(rebuild_r_array_shear)     88 % der Laufzeit
   150    0.012    0.000  mcpbe_agg.py:480(_do_one_agg)
   145    0.007    0.000  mcpbe_agg.py:329(_rebuild_all_propensities)
```

Ein einziger Aufruf dominiert: die O(n²)-Neuberechnung aller Propensities nach
jedem Ereignis.

### 2.2 Granulationslauf (`granulation_1d`, n≈400, 299 Ereignisse)

```
ncalls  tottime  filename:lineno(function)
   300    0.770   mcpbe_continuous_processes.py:260(_apply_compression)
  1514    0.388   mcpbe_nucleation.py:1082(_find_similar_particle)
   300    0.369   mcpbe_continuous_processes.py:165(_apply_internalization)
107686    0.194   mcpbe_continuous_processes.py:221(_internalization_fallback)
```

Hier ist das Profil völlig anders: keine O(n²)-Kernel, sondern **Python-Schleifen
über alle Partikel nach jedem Ereignis**. 107 686 Aufrufe von
`_internalization_fallback` = 299 Ereignisse × ~360 Partikel.

### 2.3 `liquid_bridge`-Kernel

533 ms **pro Ereignis** bei nur 150 Partikeln — der Kernel hatte keine
JIT-Batchfunktion und fiel auf `n²` Python-Aufrufe zurück.

---

## 3. Die Maßnahmen

### 3.1 O(n²) → O(n): geschlossene Momentenform (der große Hebel)

> Vollständige Herleitung, Zahlenbeispiel und Anleitung zum Erweitern auf neue
> Kernel: [`MOMENT_MODE.md`](MOMENT_MODE.md).

Alle vier eingebauten Aggregationskernel sind in `(r_i, r_j)` separabel. Damit
kollabiert die innere Summe auf wenige gewichtete Momente der Population:

```
S0 = Σ W_j        S1 = Σ W_j r_j        S2 = Σ W_j r_j²       S3 = Σ W_j r_j³
```

| Kernel | β(i,j) | Σ_j W_j β(i,j) |
|---|---|---|
| shear | `c·g·(r_i+r_j)³` | `c·g·(r_i³S0 + 3r_i²S1 + 3r_iS2 + S3)` |
| brownian | `c·2kT(r_i+r_j)²/(3μ r_i r_j)` | `K·(r_i·Σ(W_j/r_j) + 2S0 + Σ(W_j r_j)/r_i)` |
| sum | `c·(v_i+v_j)` | `c·(v_i·S0 + Σ W_j v_j)` |
| constant | `c` | `c·S0` (war bereits geschlossen) |

Der Selbstkollisionsterm wird exakt korrigiert, z. B. `β_ii = 8·c·g·r_i³` für
shear. Das ist **algebraisch exakt** — keine Näherung, keine Abschneidung.

Reine Kernelzeit, Scherkernel:

| n | pairwise | moment | Faktor | max. rel. Abweichung | max. ULP |
|---|---|---|---|---|---|
| 250 | 0.0380 ms | 0.0026 ms | 14× | 1.25e-15 | 8 |
| 500 | 0.0535 ms | 0.0020 ms | 27× | 2.01e-15 | 13 |
| 1000 | 0.1729 ms | 0.0031 ms | 55× | 3.49e-15 | 24 |
| 2000 | 0.6666 ms | 0.0066 ms | 100× | 5.91e-15 | 41 |
| 4000 | 2.9694 ms | 0.0108 ms | **275×** | 7.22e-15 | 63 |

Die O(n²)-Spalte vervierfacht sich pro Verdopplung von n, die O(n)-Spalte
verdoppelt sich — der Vorsprung wächst also mit der Problemgröße.

**Warum ist das nicht der Default?** Weil eine andere Summationsreihenfolge
andere Rundung bedeutet. Die Abweichung von wenigen Dutzend ULP ist genau das,
was man über n Summanden erwartet, aber sie kann über viele Monte-Carlo-Schritte
eine Auswahlentscheidung kippen und Trajektorien auseinanderlaufen lassen. Der
Default `pairwise` hält bestehende Validierungsläufe bitgenau reproduzierbar.

### 3.2 Parallelisierung der exakten Doppelschleife (bitgenau)

Die äußere Schleife läuft jetzt über `prange`. Weil `fastmath` **aus** ist, darf
LLVM die innere Gleitkomma-Reduktion nicht umordnen — jedes `r_out[i]` wird in
exakt derselben Reihenfolge akkumuliert wie zuvor. Gegen wortgleiche Kopien der
Originalfunktionen geprüft:

```
prange pairwise == serial reference, bit-for-bit: True
```

Unterhalb von `PARALLEL_MIN_N = 800` wird auf serielle Zwillinge umgeschaltet:
Numbas Thread-Dispatch kostet auf Windows einige zehn Mikrosekunden, was bei
kleinen Populationen mehr ist als die eigentliche Arbeit. Ohne diese Schwelle
waren kleine Szenarien nach dem Refactoring messbar *langsamer*.

### 3.3 JIT-Batch für `liquid_bridge`

Der Kernel ist nicht separabel, bleibt also O(n²) — läuft jetzt aber kompiliert
statt als `n²` Python-Aufrufe. **1053× schneller**, bitgenau identisch.
Kernel ohne Batch-Implementierung geben jetzt eine einmalige Warnung aus, statt
still langsam zu sein.

### 3.4 Vektorisierung der kontinuierlichen Prozesse

`compute_array()`-Zwillinge für Internalisierung und Kompression, dazu maskierte
numpy-Operationen statt Python-Schleifen in beiden Handlern und im
`_find_similar_particle`-Scan der Nucleation. Granulation: **12.9× schneller**,
bitgenau identisch.

### 3.5 Allokationen aus dem heißen Pfad entfernen

| Vorher | Nachher |
|---|---|
| `V_prev_active = V_flat[:, :a_tot].copy()` in **jeder** Iteration | nur wenn dieser Schritt einen Snapshot erzeugt — die Zeitschritt-Timer sind vor dem Ereignis bekannt, die Vorhersage ist also exakt |
| `(X[:a]*0.5).astype(float64)`, `W[:a].astype(float64)`, `V_flat[0,:a].astype(...)` pro Ereignis | Scratch-Puffer bzw. Views, keine Allokation |
| `np.mean(self.V0[-1,:])**2` pro Ereignis | einmal berechnet, gecacht (`V0` ist nach der Initialisierung konstant) |
| `FenwickSampler(...)` neu gebaut nach jedem Ereignis | `rebuild()` befüllt geometrisch wachsende Puffer in-place |
| `total()` berechnet die Präfixsumme bei jedem der 2–4 Aufrufe pro Ereignis neu | gecacht, invalidiert bei jeder Mutation |
| `_delta_agg[:a].astype(float64)` als Argument, das die Funktion nie liest | Parameter entfernt |
| 3× `hasattr()` pro Iteration für die Handler | einmal vor der Schleife aufgelöst |
| `_particle_arrays()` aus 9 `getattr` pro Partikelentfernung | gecacht, invalidiert bei Reallokation |

Alle diese Änderungen sind bitgenau: gleiche Werte, gleiche Reihenfolge.

---

## 4. Messergebnisse

Benchmark-Suite, `--repeats 3 --scale 4`, jeweils bestes von drei Läufen.
`scale 4` vervierfacht das Ereignisbudget jedes Szenarios, damit die Laufzeiten
groß genug für belastbare Messungen sind.

### 4.1 `pairwise` — Default, bitgenau

| Szenario | vorher [s] | nachher [s] | Faktor |
|---|---:|---:|---:|
| agg_shear_1d | 0.1152 | 0.0952 | 1.21× |
| agg_shear_1d_large | 2.1200 | 0.9842 | 2.15× |
| agg_constant_1d | 0.1830 | 0.0847 | 2.16× |
| agg_sum_1d | 0.2865 | 0.1865 | 1.54× |
| agg_brownian_1d | 0.0015 | 0.0007 | 2.12× |
| **agg_liquid_bridge_1d** | 4.8974 | 0.0047 | **1053×** |
| agg_shear_2d | 0.0920 | 0.0932 | 0.99× ¹ |
| break_powerlaw_1d | 1.1358 | 0.9386 | 1.21× |
| mix_1d | 1.0396 | 0.8206 | 1.27× |
| **granulation_1d** | 5.0331 | 0.3897 | **12.92×** |
| **GESAMT** | **14.9041** | **3.5981** | **4.14×** |

¹ `agg_shear_2d` führt nach dem Massenerhaltungs-Fix (F-01) eine andere
Trajektorie aus; die Laufzeiten sind daher nicht direkt vergleichbar.

### 4.2 `moment` — Opt-in, O(n)

| Szenario | vorher [s] | nachher [s] | Faktor |
|---|---:|---:|---:|
| agg_shear_1d | 0.1152 | 0.0485 | 2.37× |
| **agg_shear_1d_large** | 2.1200 | 0.1447 | **14.65×** |
| agg_constant_1d | 0.1830 | 0.0853 | 2.15× |
| agg_sum_1d | 0.2865 | 0.1336 | 2.14× |
| agg_brownian_1d | 0.0015 | 0.0003 | 5.47× |
| agg_liquid_bridge_1d | 4.8974 | 0.0046 | 1071× |
| agg_shear_2d | 0.0920 | 0.0494 | 1.86× |
| break_powerlaw_1d | 1.1358 | 0.9732 | 1.17× |
| mix_1d | 1.0396 | 0.2582 | 4.03× |
| granulation_1d | 5.0331 | 0.3984 | 12.63× |
| **GESAMT** | **14.9041** | **2.0961** | **7.11×** |

Der Unterschied zwischen den beiden Modi wächst mit der Partikelzahl: bei
`agg_shear_1d_large` (n≈1900) sind es 2.15× gegen 14.65×.

---

## 5. Sind die beiden Modi statistisch unterscheidbar?

Ensemble-Vergleich, 8 Seeds pro Modus, Szenario `agg_shear_1d`:

```
        pairwise mean    moment mean    |Δ|         MC-Streuung   |Δ|/σ
M0      2.576250e+02     2.576250e+02   0.000e+00   5.655e+00     0.000
M1      2.617994e-16     2.617994e-16   0.000e+00   3.727e-32     0.000
M2      3.280958e-34     3.280958e-34   0.000e+00   8.377e-36     0.000
```

In diesem Szenario sind die Ergebnisse sogar identisch — die
ULP-Unterschiede haben über die Lauflänge nie eine Auswahlentscheidung gekippt.
Das ist nicht allgemein garantiert; die Kennzahl `|Δ|/σ` ist der Maßstab: Werte
deutlich unter 1 bedeuten, dass die Moduswahl nicht von gewöhnlichem
Monte-Carlo-Seed-Rauschen zu unterscheiden ist.

Selbst nachprüfen:

```bash
python -m tests.bench.compare_propensity_modes --seeds 8
```

---

## 6. Empfehlung

1. **Jetzt:** `pairwise` bleibt Default. Bestehende Validierungen für `dim=1`
   bleiben bitgenau gültig, und die Suite ist trotzdem 4.1× schneller.
2. **Für Produktionsläufe:** `moment` einschalten, sobald die
   Ensemble-Prüfung für die eigenen Parameter durchgeführt wurde:

   ```python
   solver.agg_propensity_mode = "moment"
   ```

   Das ist der verbleibende große Hebel, besonders bei > 1000 Partikeln.
3. **Nächster Schritt, falls noch mehr Geschwindigkeit nötig ist:** Die
   Momentenform macht den Rebuild O(n). Der nächste Deckel ist der
   Fenwick-Neuaufbau (ebenfalls O(n)) pro Ereignis. Mit inkrementell gepflegten
   Momentensummen ließe sich das auf O(log n) drücken — das ist aber ein
   deutlich größerer Eingriff und war für diesen Umfang nicht nötig.

---

## 7. Benchmarks selbst ausführen

```bash
# einmalig
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ./pbe-core -e ./mcpbe pytest
cd mcpbe
```

```bash
python -m tests.bench.bench_wmcpbe --repeats 3
```

```bash
python -m tests.bench.bench_wmcpbe --repeats 3 --scale 4 --propensity-mode moment
```

```bash
python -m tests.bench.profile_wmcpbe granulation_1d --top 15
```

```bash
python -m tests.bench.compare_propensity_modes --seeds 8
```

Alle Szenarien sind über ein **festes Ereignisbudget** (`Scenario.maxiter`)
begrenzt, nicht über die simulierte Zeit. Die Wandzeit ist damit direkt
proportional zu den Kosten pro Monte-Carlo-Ereignis — genau der Größe, die diese
Arbeit optimiert — und die Suite bleibt unabhängig von den Kernelparametern in
einem vorhersagbaren Zeitrahmen (Gesamtlauf < 1 min bei `--scale 1`).
