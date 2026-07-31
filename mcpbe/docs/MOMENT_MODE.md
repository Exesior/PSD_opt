# Moment-Modus: O(n²) → O(n) für die Agglomerations-Propensity

`solver.agg_propensity_mode ∈ {"pairwise", "moment"}`

Dieses Dokument erklärt, **was** der Moment-Modus rechnet, **warum** er exakt
ist, **warum er trotzdem nicht Default ist**, und **wie** man ihn auf einen neuen
Kernel erweitert.

Kurzfassung für Eilige: der Moment-Modus wertet exakt dieselbe Summe aus wie
vorher, nur umsortiert — statt für jedes Partikel über alle Partner zu laufen,
werden vier Summen *einmal* gebildet und dann pro Partikel eingesetzt. Aus O(n²)
wird O(n). Bei n = 4000 ist der Kernel dadurch **275×** schneller.

---

## 1. Das Problem

Nach jedem akzeptierten Monte-Carlo-Ereignis braucht der Solver für jedes
Rechenpartikel *i* seine Ereignisneigung (Propensity)

```
r_i  =  Σ_{j ≠ i} W_j · β(i,j)  +  (W_i − 1) · β(i,i)      falls W_i > 1
```

Der zweite Term ist die **Selbstkollision**: ein Rechenpartikel mit Gewicht
`W_i` repräsentiert `W_i` identische physikalische Partikel, die auch
untereinander kollidieren können. Ein Partikel kann aber nicht mit sich selbst
kollidieren, daher `(W_i − 1)` statt `W_i`. Für `W_i ≤ 1` entfällt der Term
ganz.

Naiv ausgewertet ist das eine Doppelschleife: für jedes *i* über alle *j*.
**O(n²) pro Ereignis.** Genau das war der dominante Kostenpunkt des Solvers —
88 % der Laufzeit bei n ≈ 1900.

---

## 2. Die Idee: Separierbarkeit

Der entscheidende Punkt: Der Kollisionskernel `β(i,j)` hängt nur über wenige
einfache Terme von *i* und *j* ab. Lässt er sich als **endliche Summe von
Produkten** schreiben,

```
β(i,j)  =  Σ_{k=1}^{K}  f_k(i) · g_k(j)
```

dann darf man die Summationsreihenfolge vertauschen:

```
Σ_j W_j · β(i,j)  =  Σ_j W_j · Σ_k f_k(i)·g_k(j)
                  =  Σ_k f_k(i) · ( Σ_j W_j·g_k(j) )
                                    └──────┬──────┘
                                    hängt NICHT von i ab
```

Die inneren Klammern sind **gewichtete Momente der Population**. Man berechnet
sie *einmal* in O(n) und setzt sie dann für jedes *i* in O(K) ein:

| | Aufwand |
|---|---|
| Momente bilden | O(n) |
| alle r_i auswerten | O(K·n) |
| **gesamt** | **O(n)**, da K eine kleine Konstante ist |

Das ist keine Näherung. Es ist dasselbe Ergebnis, nur anders geklammert.

---

## 3. Herleitung pro Kernel

### 3.1 Scherkernel (Chin et al. 1998) — der Standardfall

```
β(i,j) = c · g · (r_i + r_j)³
```

Binomische Entwicklung:

```
(r_i + r_j)³ = r_i³ + 3·r_i²·r_j + 3·r_i·r_j² + r_j³
```

Also K = 4 mit

| k | f_k(i) | g_k(j) |
|---|---|---|
| 1 | r_i³ | 1 |
| 2 | 3·r_i² | r_j |
| 3 | 3·r_i | r_j² |
| 4 | 1 | r_j³ |

Die vier benötigten Momente:

```
S0 = Σ_j W_j          S1 = Σ_j W_j·r_j
S2 = Σ_j W_j·r_j²     S3 = Σ_j W_j·r_j³
```

und damit die **vollständige** Summe (inklusive j = i):

```
F_i = Σ_j W_j·β(i,j) = c·g·( r_i³·S0 + 3·r_i²·S1 + 3·r_i·S2 + S3 )
```

### 3.2 Die Selbstterm-Korrektur (der subtile Teil)

`F_i` enthält den Term `W_i · β(i,i)` — wir brauchen aber
`(W_i − 1)·β(i,i)` bzw. 0. Also einmal abziehen, einmal richtig addieren:

```
r_i = F_i − W_i·β(i,i) + selbst_i
             ↑ was zu viel drin ist   ↑ was hin soll

selbst_i = (W_i − 1)·β(i,i)   falls W_i > 1
         = 0                   sonst
```

Für den Scherkernel ist `β(i,i) = c·g·(2·r_i)³ = 8·c·g·r_i³`, also zusammengefasst:

```
r_i = c·g·( r_i³·S0 + 3r_i²·S1 + 3r_i·S2 + S3 )  +  8·c·g·r_i³ · ( selbst_w_i − W_i )

mit  selbst_w_i = W_i − 1   falls W_i > 1,  sonst 0
```

Genau das steht in `rebuild_r_array_shear_moment()`. Man beachte die
**Unstetigkeit bei W_i = 1**: knapp darunter fällt der Selbstterm komplett weg,
knapp darüber ist er ≈ 0. Beide Seiten sind numerisch geprüft (siehe §6).

### 3.3 Brownscher Kernel (Tsouris et al. 1995)

```
β(i,j) = c · 2·k_B·T·(r_i + r_j)² / (3·μ·r_i·r_j)
```

Mit der Abkürzung `K = 2·c·k_B·T / (3·μ)`:

```
(r_i + r_j)²        r_i² + 2·r_i·r_j + r_j²        r_i         r_j
────────────   =   ────────────────────────   =   ───  +  2 + ───
   r_i·r_j                  r_i·r_j                r_j         r_i
```

Also K = 3:

| k | f_k(i) | g_k(j) |
|---|---|---|
| 1 | r_i | 1/r_j |
| 2 | 2 | 1 |
| 3 | 1/r_i | r_j |

```
T_inv = Σ_j W_j/r_j     T0 = Σ_j W_j     T1 = Σ_j W_j·r_j

F_i = K · ( r_i·T_inv + 2·T0 + T1/r_i )
β(i,i) = K · (1 + 2 + 1) = 4K
```

Wichtig: die Momente laufen nur über Partikel mit `r_j > 0` — genau wie die
`continue`-Wächter der Doppelschleife. Bei `μ ≤ 0` überspringt die
Doppelschleife jedes Paar; der Moment-Pfad gibt dann konsistent 0 zurück.

### 3.4 Summenkernel

```
β(i,j) = c·(v_i + v_j),        v = (4π/3)·r³
```

K = 2, trivial:

```
S0 = Σ_j W_j     Sv = Σ_j W_j·v_j

F_i = c·( v_i·S0 + Sv )
β(i,i) = 2·c·v_i
```

### 3.5 Konstanter Kernel

```
β(i,j) = c        →     F_i = c·S0,     β(i,i) = c
```

War bereits im Originalcode geschlossen ausgewertet. `pairwise` und `moment`
sind hier **identisch** — es gibt buchstäblich keinen Unterschied.

### 3.6 Was *nicht* geht: `liquid_bridge`

```
β(i,j) = c·g·(r_i+r_j)³ · exp( −((s_i+s_j)/2 − s_opt)² / (2σ²) )
                        · ( 1 + α·ln(1 + (e_i+e_j)/v_ref(r_i,r_j)) )
```

Der Gauß-Term enthält beim Ausquadrieren ein Kreuzglied `s_i·s_j` im Exponenten,
also einen Faktor `exp(−s_i·s_j/σ²)`. Eine Exponentialfunktion eines *Produkts*
lässt sich nicht als **endliche** Summe separabler Terme schreiben (nur als
unendliche Reihe). Der Logarithmus mit einem von *beiden* Radien abhängigen
`v_ref` ist ebenfalls nicht separabel.

`liquid_bridge` bleibt daher O(n²) — läuft aber jetzt als JIT-Batchfunktion statt
als n² Python-Aufrufe und ist dadurch **1053×** schneller als vorher. Der
Moment-Modus greift bei diesem Kernel schlicht nicht (`MOMENT_KERNELS` enthält
ihn nicht); es wird automatisch der Paarpfad genutzt.

---

## 4. Ein konkretes Zahlenbeispiel

Drei Partikel, Scherkernel mit `c·g = 1`:

```
r = [1, 2, 3]        W = [10, 20, 30]
```

**Paarweise, Partikel i = 0:**

```
j=0:  W_0 > 1  →  (10−1)·(1+1)³ =  9·8   =  72
j=1:                    20·(1+2)³ = 20·27  = 540
j=2:                    30·(1+3)³ = 30·64  = 1920
                                    r_0    = 2532
```

**Über Momente:**

```
S0 = 10+20+30                     = 60
S1 = 10·1 + 20·2 + 30·3           = 140
S2 = 10·1 + 20·4 + 30·9           = 360
S3 = 10·1 + 20·8 + 30·27          = 980

F_0 = 1³·60 + 3·1²·140 + 3·1·360 + 980  =  60 + 420 + 1080 + 980  = 2540
Korrektur: 8·r_0³·(selbst_w − W_0) = 8·1·(9 − 10) = −8
r_0 = 2540 − 8 = 2532                                   ✓
```

Für n = 3 ist das offensichtlich kein Gewinn. Bei n = 4000 spart man
16 000 000 innere Iterationen pro Ereignis.

---

## 5. Warum ist das nicht der Default?

Weil **Gleitkomma-Addition nicht assoziativ ist**.

Beide Verfahren sind in exakter Arithmetik identisch, aber sie summieren in
unterschiedlicher Reihenfolge, und jede Reihenfolge rundet anders:

```
pairwise:  ((( W_1β_i1 + W_2β_i2 ) + W_3β_i3 ) + … )
moment:    c·g·( r_i³·S0 + 3r_i²·S1 + 3r_i·S2 + S3 )  + Korrektur
```

Der Unterschied liegt bei etwa `n·ε` — gemessen ~1e-15 relativ, also einige
Dutzend ULP. Physikalisch bedeutungslos. **Aber**: `r_i` geht in den
Fenwick-Sampler und in die Zeitschrittberechnung ein. In seltenen Fällen liegt
die Zufallszahl `u · total` genau an einer Bereichsgrenze, ein anderes Partikel
wird gezogen, und die Trajektorie läuft auseinander.

Das Ergebnis ist dann **statistisch äquivalent, aber nicht bitgleich**. Für einen
Monte-Carlo-Code ist das völlig in Ordnung — nur eben nicht geeignet, wenn man
einen aufgezeichneten Validierungslauf exakt reproduzieren will.

Deshalb: `pairwise` ist Default und bitgenau identisch zum Code vor dem
Refactoring; `moment` ist ein bewusster Schalter.

---

## 6. Messungen

### 6.1 Reine Kernelzeit (Scherkernel)

| n | pairwise | moment | Faktor | max. rel. Abw. | max. ULP |
|---:|---:|---:|---:|---:|---:|
| 250 | 0.0380 ms | 0.0026 ms | 14× | 1.25e-15 | 8 |
| 500 | 0.0535 ms | 0.0020 ms | 27× | 2.01e-15 | 13 |
| 1000 | 0.1729 ms | 0.0031 ms | 55× | 3.49e-15 | 24 |
| 2000 | 0.6666 ms | 0.0066 ms | 100× | 5.91e-15 | 41 |
| 4000 | 2.9694 ms | 0.0108 ms | **275×** | 7.22e-15 | 63 |

Die O(n²)-Spalte **vervierfacht** sich pro Verdopplung von n, die O(n)-Spalte
**verdoppelt** sich. Der Vorsprung wächst also mit der Problemgröße — das ist
der eigentliche Punkt, nicht der konstante Faktor.

Dass die ULP-Abweichung mit n wächst, ist ebenfalls erwartet: mehr Summanden,
mehr akkumulierte Rundung.

### 6.2 Gegen die reine Definition geprüft

Beide Implementierungen wurden gegen eine naive Referenz direkt aus der
Definition verglichen, inklusive der Selbstterm-Randfälle:

```
pairwise vs brute-force : max rel 2.27e-16
moment   vs brute-force : max rel 1.23e-15

  W=   0.5000000   reldev=0.00e+00      (Selbstterm entfällt)
  W=   1.0000000   reldev=4.51e-16      (Grenzfall, entfällt)
  W=   1.0000001   reldev=1.15e-16      (Grenzfall, aktiv)
  W=   0.9999999   reldev=4.53e-16
  W=   2.0000000   reldev=2.23e-16
  W=1000000.0000000 reldev=3.00e-16     (extremes Gewicht)
```

### 6.3 Ende-zu-Ende

| | Suite gesamt | `agg_shear_1d_large` (n≈1900) |
|---|---:|---:|
| Baseline `7a35310` | 14.90 s | 3.539 ms/Ereignis |
| `pairwise` | 3.60 s | 1.092 ms/Ereignis |
| `moment` | **2.10 s** | **0.263 ms/Ereignis** |

### 6.4 Sind die Modi statistisch unterscheidbar?

Ensemble über 8 Seeds, Szenario `agg_shear_1d`:

```
        pairwise mean    moment mean    |Δ|         MC-Streuung   |Δ|/σ
M0      2.576250e+02     2.576250e+02   0.000e+00   5.655e+00     0.000
M1      2.617994e-16     2.617994e-16   0.000e+00   3.727e-32     0.000
M2      3.280958e-34     3.280958e-34   0.000e+00   8.377e-36     0.000
```

In diesem Szenario sind die Ergebnisse sogar **identisch** — die ULP-Unterschiede
haben über die Lauflänge nie eine Auswahlentscheidung gekippt. Das ist nicht
allgemein garantiert. Die Kennzahl ist `|Δ|/σ`: Werte deutlich unter 1 heißen,
dass die Moduswahl im gewöhnlichen Seed-Rauschen untergeht.

Selbst nachrechnen:

```bash
python -m tests.bench.compare_propensity_modes --seeds 8
```

---

## 7. Benutzung

```python
solver = MCPBESolver(..., agg_kernel_name="shear_chin1998", ...)
solver.agg_propensity_mode = "moment"      # Default: "pairwise"
```

Der Schalter wirkt nur auf Kernel in `MOMENT_KERNELS`
(`shear_chin1998`, `brownian_tsouris1995`, `constant`, `sum`). Für alle anderen
wird still der Paarpfad genutzt — man kann ihn also gefahrlos global setzen.

Im Benchmark:

```bash
python -m tests.bench.bench_wmcpbe --repeats 3 --scale 4 --propensity-mode moment
```

### Entscheidungshilfe

| Situation | Empfehlung |
|---|---|
| Aufgezeichneten Validierungslauf exakt reproduzieren | `pairwise` |
| Produktionslauf, n ≳ 1000 | `moment` (nach einmaliger Ensemble-Prüfung) |
| Parameterstudie / Optimierung über viele Läufe | `moment` |
| n ≲ 300 | egal, der Rebuild dominiert dort ohnehin nicht |
| Eigener, nicht separabler Kernel | automatisch `pairwise` |

---

## 8. Einen neuen Kernel ergänzen

1. **Prüfen, ob β(i,j) separabel ist**, d. h. als endliche Summe
   `Σ_k f_k(i)·g_k(j)` darstellbar. Polynome in `(r_i + r_j)`, Summen `v_i + v_j`
   und Quotienten wie `r_i/r_j` sind es. Alles, wo ein Produkt `i·j` im Exponenten
   oder in einem Logarithmus steht, ist es nicht.

2. **Momente identifizieren** — pro `g_k` eine Summe `Σ_j W_j·g_k(j)`.

3. **`β(i,i)` bestimmen** und die Selbstterm-Korrektur
   `+ β(i,i)·(selbst_w_i − W_i)` anhängen. Das ist die Stelle, an der
   erfahrungsgemäß Fehler passieren.

4. Funktion `rebuild_r_array_<name>_moment()` in
   `src/wmcpbe/kernels/aggregation/jit_kernels.py` ergänzen, Namen zu
   `MOMENT_KERNELS` hinzufügen und in
   `MCPBEAgg._compute_raw_propensities()` einhängen.

5. **Verifizieren**: gegen die Paarvariante auf zufälligen Populationen
   vergleichen, inklusive `W < 1`, `W = 1`, `W ≫ 1` und `r = 0`. Die Abweichung
   muss in der Größenordnung `n·ε` bleiben — alles darüber ist ein Fehler in der
   Herleitung, typischerweise im Selbstterm.

---

## 9. Wo ist die nächste Grenze?

Mit dem Moment-Modus ist der Rebuild O(n). Die nächste Schranke ist der
Fenwick-Neuaufbau, ebenfalls O(n) pro Ereignis.

Weiter käme man mit **inkrementell gepflegten Momentensummen**: bei einem
Ereignis ändern sich nur wenige Gewichte, also ließen sich `S0…S3` in O(1)
nachführen. Allerdings ändert das *alle* `r_i` gleichzeitig, sodass man den
Sampler ohne einen zusätzlichen Trick (z. B. lazy/hierarchische Auswertung)
weiterhin komplett neu aufbauen müsste. Das ist ein deutlich größerer Eingriff
und war für den aktuellen Umfang nicht nötig.

---

## Verwandte Dokumente

- [`PERFORMANCE.md`](PERFORMANCE.md) — Gesamtbild der Performance-Arbeit
- [`REFACTORING_FINDINGS.md`](REFACTORING_FINDINGS.md) — F-04 (dieser Punkt), F-05 (`liquid_bridge`)
- [`../tests/README.md`](../tests/README.md) — Benchmarks und Regressionsprüfung ausführen
