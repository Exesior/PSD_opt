# Bias-Correction, Two-Stage-Sampler und exakte Gewichtsdisziplin

**Stand:** 2026-08-14
**Referenz-Implementierung:** `upstream/dev_monorepo` = Commit `02dfac6` (2026-08-10),
identisch mit github.com/pdhs-group/PSD_opt/tree/dev_monorepo
**Paper:** Ji, H. & Rhein, F.: *A Weighted Monte Carlo Solver for Multicomponent
Population Balance Equations with Dynamic Grid-Based Reconstruction* (Preprint, Elsevier)

---

## Inhalt

1. [Worum es geht](#1-worum-es-geht)
2. [Die drei Defekte](#2-die-drei-defekte)
3. [Grundlagen: Batch-Größe, effektive Batch-Größe, Bias](#3-grundlagen)
4. [**Die Herleitung der O(n log n)-Momentenform**](#4-die-herleitung) ← Kernstück
5. [Umbauplan Teil A: Propensity & Sampling](#5-umbauplan-teil-a)
6. [Umbauplan Teil B: Gewichtsdisziplin](#6-umbauplan-teil-b)
7. [Verifikation](#7-verifikation)
8. [Risiken und offene Punkte](#8-risiken)

---

## 1. Worum es geht

Der Agglomerations-Kernpfad in `dev_eric` weicht an drei zusammenhängenden Stellen von
der validierten Referenz-Implementierung ab. Alle drei hängen am selben Codepfad
(`mcpbe_agg.py` + `kernels/aggregation/jit_kernels.py`) und teilen dieselbe Datenbasis
(`_delta_agg`, `compute_beta`, `W`-Konsumption). Ein Teil-Umbau würde inkonsistente
Zwischenzustände erzeugen — deshalb werden sie zusammen erledigt.

**Was sich NICHT ändert:** Die gesamte Physik. β-Kernel, Akzeptanz-Kernel (`stokes_krit`),
Porositäts-Kernel (`cone_model`, `volume_mixing`), Flüssigkeits-Internalisierung,
`ParticleMerger`, Rekonstruktion. Es geht ausschließlich um die *Buchhaltung*:
Wie werden Ereignis-Raten berechnet, wie werden Paare gezogen, wie wird Gewicht verbucht.

---

## 2. Die drei Defekte

### D1 — Fehlende Pair-Delta-Bias-Correction

**Ist-Zustand** (`mcpbe_agg.py::_rebuild_all_propensities`, Zeilen 574–582):

```python
# in jit_kernels.py, z.B. rebuild_r_array_shear_moment:
r_out[i] = Σ_j Wj·β(i,j)              # kein ΔW, kein Wi-Vorfaktor

# zurück in _rebuild_all_propensities:
np.divide(r, delta, out=r, where=delta > 0.0)   # delta = min(dW_const, W_i)
```

Die Division passiert **nach** der Summe und benutzt nur `δ_i` — das eigene Delta von
Partikel `i`, unabhängig davon, wie viel Gewicht der Partner `j` überhaupt hat.

**Soll-Zustand** (Paper Gl. 36/37, upstream `nb_rebuild_ragg_weighted_pair_delta`):

```
R_i* = W_i · [ Σ_{j≠i} W_j·β(i,j)/ΔW_ij  +  (W_i−1)·β(i,i)/ΔW_ii ]
```

Es fehlen also **zwei** Dinge: der `W_i`-Vorfaktor und die *paarweise* Division durch
`ΔW_ij = min(δ_i, δ_j)`.

### D2 — Partnerauswahl ignoriert β

`mcpbe_agg.py::_pick_partner_kernel`, Zeilen 384–386:

```python
cumsum_W = self._agg_buffers(a)["cumsum"][:a]
np.cumsum(W, out=cumsum_W)
j = int(np.searchsorted(cumsum_W, u_sel * total_W, side="right"))
```

Der Partner `j` wird proportional zu **rohem `W_j`** gezogen. `β(i,j)` wird erst danach
berechnet und wirkt nur noch als Ja/Nein-Filter (`if beta_ij <= 0: return -1`).
Im Docstring derselben Funktion steht das bereits als *"known statistical defect"*.

**Soll** (Paper Gl. 17/40): `j ~ W_j·β(i,j)/ΔW_ij`, gezogen über einen echten
Akkumulations-Loop.

### D3 — Restgewichte und `W_MIN_ACTIVE`

`mcpbe_time_helper.py`, Zeile 58:

```python
W_MIN_ACTIVE = 1e-12
```

Diese Schwelle wurde eingeführt, weil Restgewichte (z.B. `W = 3e-14` nach vielen
Subtraktionen) über `r_i / δ_i` praktisch unendliche Propensities erzeugen und den
Zeitschritt sprengen.

**Upstream braucht diese Schwelle nicht** — dort wird durchgängig konsumiert:

```python
# upstream mcpbe_break.py::_break_apply_and_maintain
dW = self._compute_dW_packet(k)      # == δ_break[k] == min(ΔW_b, W_k)
w_rem = w_parent_old - dW
self.W[k] = w_rem
if w_rem > 0.0:
    ...
else:
    self._remove_particle_column(k)   # exakt 0 → weg
```

Die minimale Referenz des Betreuers (`mcpbe/script/weighted_pure_death_experiment.py`)
macht dasselbe:

```python
dW = float(delta_before[i])   # die konsumierte Batch IST delta_i
W[i] -= dW
```

**Das Prinzip:** Wenn `dW ≡ ΔW_i = min(ΔW, W_i)`, dann gilt:

- `W_i ≥ ΔW` → es wird `ΔW` abgezogen, Rest bleibt sauber
- `W_i < ΔW` → `ΔW_i = W_i` → **`W_i` wird exakt `0.0`**

Ein Partikel bekommt also nie einen Bruchteil-Rest. Der letzte Event auf einem Partikel
räumt es exakt leer. `> 0.0` als Löschkriterium genügt dann.

**Wo `dev_eric` davon abweicht:**

| Stelle | Problem |
|---|---|
| `mcpbe_agg.py:712-716` | Selbstkollision wird *abgelehnt* (`if Wi <= 2.0*delta_i: return 0.0`) statt gekappt ⇒ `2·dW < W_i` immer ⇒ nie exakt 0 |
| `mcpbe_nucleation.py:1413-1416` | `max_dW_allowed = max_physical_droplets / vc_scale` — ein kontinuierlicher Volumenquotient ⇒ fraktionales `dW` |
| `mcpbe_nucleation.py:1945-1950` | `dW = Wi / 2.0` bzw. `min(Wi, Wj)` ohne δ-Kappung |
| `reconstruction_mixin.py` | QMOM/RS-Gewichte inhärent fraktional — **bleibt bewusst so**, siehe §6.5 |

---

## 3. Grundlagen

### 3.1 Die Größen

| Symbol | Bedeutung | Im Code |
|---|---|---|
| `W_i` | Statistisches Gewicht: wie viele *physische* Partikel repräsentiert Rechenpartikel `i` | `self.W[i]` |
| `ΔW` | Vorgegebene Batch-Größe (Konstante) | `agg_dW_max` → `_agg_dW_const` |
| `δ_i` | **Effektive** Batch-Größe von `i` = `min(ΔW, W_i)` | `_delta_agg[i]` |
| `β(i,j)` | Kollisionsrate zwischen `i` und `j` [m³/s] | `kernel_manager.compute_beta` |
| `R_i*` | Korrigierte Partial-Propensity von `i` | `_r_agg[i]` |

### 3.2 Warum es überhaupt einen Bias gibt

Ein Monte-Carlo-Event soll nicht *eine* physische Kollision darstellen, sondern ein ganzes
Paket von `ΔW` Kollisionen — sonst müsste man bei realistischen Partikelzahlen Milliarden
Events rechnen.

Damit die *mittlere* Dynamik trotzdem stimmt, muss gelten (Paper Gl. 49):

```
λ_i · ΔW_i  =  W_i · r(x_i, y_i)
   ↑ numerische   ↑ tatsächlich       ↑ wahre physikalische Rate
   Ereignisrate    abgearbeitete Menge
```

Daraus folgt zwingend `λ_i = W_i·r/ΔW_i` (Paper Gl. 50).

**Das Problem:** `ΔW` kann nicht überall gleich sein. Wenn ein Partikel weniger Gewicht
hat als `ΔW`, würde ein volles Paket das Gewicht negativ machen. Also muss lokal gekappt
werden: `ΔW_i = min(ΔW, W_i)`.

Wird diese Kappung **nicht** in die Auswahlwahrscheinlichkeit zurückgespiegelt, stimmt
`λ·ΔW = W·r` genau für die Partikel nicht mehr, bei denen gekappt wurde — typischerweise
leichte, frisch entstandene oder fast aufgebrauchte Partikel. Das Paper zeigt in §3.4,
dass dieser Bias **90–100 % des Gesamtfehlers** ausmacht und sich über die Zeit
akkumuliert.

Bei Agglomeration ist es ein *Paar*-Prozess, also muss die Kappung beide Partner
berücksichtigen: `ΔW_ij = min(δ_i, δ_j)` (Paper Gl. 33).

### 3.3 Die exakte Zielformel

Aus dem Upstream-Code (`nb_rebuild_ragg_weighted_pair_delta`), Term für Term:

```
R_i* = W_i · S_i

S_i  =  Σ_{j≠i}  W_j·β(i,j) / ΔW_ij           (a) Fremd-Paare
      + [W_i > 1] · (W_i−1)·β(i,i) / ΔW_ii     (b) Selbstkollision

ΔW_ij = min(δ_i, δ_j)
ΔW_ii = min(δ_i, W_i/2)
δ_i   = min(ΔW, W_i)
```

Der Term (b) verdient eine Erklärung: `W_i·(W_i−1)` zählt **geordnete physische Paare**
innerhalb desselben Rechenpartikels. Wenn `i` genau `W_i` physische Partikel
repräsentiert, hat jedes davon `W_i − 1` mögliche Partner im selben Pool (nicht `W_i`,
denn ein Partikel kann nicht mit sich selbst kollidieren). Das ist ein reines
Abzählargument — es ergibt nur für **ganzzahliges `W_i`** Sinn. Genau daher kommt die
Integer-Semantik von `W`.

Das `W_i/2` in `ΔW_ii`: eine Selbstkollision verbraucht *zwei* physische Partikel aus
demselben Pool, also darf `dW` höchstens `W_i/2` sein.

---

## 4. Die Herleitung

**Ziel:** `R_i*` für alle `i` in besser als O(n²) berechnen.

### 4.1 Warum der bisherige Momenten-Trick scheitert

Der heutige O(n)-Trick (`rebuild_r_array_shear_moment`) beruht darauf, dass sich
`β(i,j)` in `i`- und `j`-Anteile zerlegen lässt. Beispiel Scherkernel:

```
(r_i + r_j)³ = r_i³·1 + 3r_i²·r_j + 3r_i·r_j² + 1·r_j³
```

Damit wird aus der Doppelsumme:

```
Σ_j W_j·β(i,j) = C·( r_i³·S₀ + 3r_i²·S₁ + 3r_i·S₂ + S₃ )
```

mit den **Momenten** `S₀ = Σ W_j`, `S₁ = Σ W_j r_j`, `S₂ = Σ W_j r_j²`, `S₃ = Σ W_j r_j³`.
Die Momente kosten einmal O(n), danach ist jedes `r_i` O(1) → insgesamt O(n).

**Der Bruch:** Sobald `1/min(δ_i, δ_j)` im Summanden steht, ist der Summand nicht mehr
in `i` und `j` faktorisierbar — `min()` koppelt beide Indizes. Deshalb hat der Upstream
gar keine Momenten-Variante, sondern nur den O(n²)-Doppelloop.

### 4.2 Schritt 1 — Die Schlüsselbeobachtung

`δ_i = min(ΔW, W_i)` mit **konstantem** `ΔW` kann nur zwei Werte annehmen. Die Population
zerfällt damit in genau zwei Klassen:

```
H := { j : W_j ≥ ΔW }      ⟹  δ_j = ΔW      (für alle j∈H derselbe konstante Wert!)
L := { j : 0 < W_j < ΔW }  ⟹  δ_j = W_j
```

(Partikel mit `W_j ≤ 0` haben `δ_j = 0` und werden komplett ausgeschlossen — sie
existieren nach dem Umbau aus §6 ohnehin nicht mehr.)

Das ist der ganze Trick. `min()` ist nur dann "böse", wenn beide Argumente frei variieren.
Hier ist eines der beiden Argumente in großen Teilen der Population eine **Konstante**.

### 4.3 Schritt 2 — Blockzerlegung von `ΔW_ij`

Wir gehen alle vier Kombinationen durch:

| `i` | `j` | `δ_i` | `δ_j` | `ΔW_ij = min(δ_i,δ_j)` | Begründung |
|---|---|---|---|---|---|
| H | H | `ΔW` | `ΔW` | **`ΔW`** | beide gleich |
| H | L | `ΔW` | `W_j` | **`W_j`** | `W_j < ΔW` per Definition von L |
| L | H | `W_i` | `ΔW` | **`W_i`** | `W_i < ΔW` per Definition von L |
| L | L | `W_i` | `W_j` | **`min(W_i,W_j)`** | echt gekoppelt |

**Drei von vier Blöcken sind separabel:**
- `H×H`: `ΔW` ist eine Konstante → lässt sich vor die Summe ziehen
- `H×L`: hängt nur von `j` ab → geht in die `j`-Momente ein
- `L×H`: hängt nur von `i` ab → lässt sich vor die Summe ziehen

Nur `L×L` braucht Extra-Arbeit.

### 4.4 Schritt 3 — Der `L×L`-Block: Aufspaltung am Sortierindex

Für ein festes `i ∈ L` betrachten wir:

```
Σ_{j∈L} W_j·β(i,j) / min(W_i, W_j)
```

Wir spalten die Summe danach auf, welches der beiden Gewichte kleiner ist:

```
= Σ_{j∈L, W_j ≤ W_i}  W_j·β(i,j)/W_j        (hier ist min = W_j)
+ Σ_{j∈L, W_j > W_i}  W_j·β(i,j)/W_i        (hier ist min = W_i)
```

**Im ersten Term kürzt sich `W_j/W_j` weg**, im zweiten lässt sich `1/W_i` vor die
Summe ziehen:

```
= Σ_{j∈L, W_j ≤ W_i} β(i,j)   +   (1/W_i) · Σ_{j∈L, W_j > W_i} W_j·β(i,j)
      └── Präfixsumme ──┘              └────── Suffixsumme ──────┘
```

Wenn man `L` **aufsteigend nach `W` sortiert**, sind das genau eine Präfix- und eine
Suffixsumme über die sortierte Liste. Beide werden einmal in O(n) vorberechnet, danach
kostet der Zugriff für jedes `i` nur O(1) (Position im sortierten Array).

> **Randfall Gleichstand:** Bei `W_j = W_i` gilt `min(W_i,W_j) = W_i = W_j`. Beide Zweige
> liefern denselben Wert (`W_j/W_j = 1` bzw. `W_j/W_i = 1`). Der Trennpunkt ist also
> **eindeutig** — es braucht keine Tie-Breaking-Regel. Im Code: `np.searchsorted(...,
> side="right")`, damit Gleichstände in den `≤`-Zweig fallen.

### 4.5 Schritt 4 — Separable Kernel formalisieren

Wir schreiben den Kernel allgemein als Summe von Produkten:

```
β(i,j) = Σ_{k=0}^{K-1} f_k(i) · g_k(j)
```

Für die vier heutigen Moment-Kernel sieht das so aus:

**`constant`:** `β = c`, also `K = 1`

| k | `f_k(i)` | `g_k(j)` |
|---|---|---|
| 0 | `c` | `1` |

`β(i,i) = c`

**`sum`:** `β = c·(v_i + v_j)` mit `v = (4/3)πr³`, also `K = 2`

| k | `f_k(i)` | `g_k(j)` |
|---|---|---|
| 0 | `c·v_i` | `1` |
| 1 | `c` | `v_j` |

`β(i,i) = c·(v_i + v_i) = 2c·v_i`

**`shear_chin1998`:** `β = C·(r_i+r_j)³` mit `C = corr_beta · g`

Binomische Entwicklung:
```
(r_i + r_j)³ = r_i³ + 3r_i²r_j + 3r_ir_j² + r_j³
```
also `K = 4`:

| k | `f_k(i)` | `g_k(j)` |
|---|---|---|
| 0 | `C·r_i³` | `1` |
| 1 | `3C·r_i²` | `r_j` |
| 2 | `3C·r_i` | `r_j²` |
| 3 | `C` | `r_j³` |

`β(i,i) = C·(2r_i)³ = 8C·r_i³`

**`brownian_tsouris1995`:** `β = K_B·(r_i+r_j)²/(r_i·r_j)` mit `K_B = 2·corr_beta·kT/(3μ)`

Umformung:
```
(r_i+r_j)²/(r_i·r_j) = (r_i² + 2r_ir_j + r_j²)/(r_i·r_j)
                     = r_i/r_j + 2 + r_j/r_i
```
also `K = 3`:

| k | `f_k(i)` | `g_k(j)` |
|---|---|---|
| 0 | `K_B·r_i` | `1/r_j` |
| 1 | `2K_B` | `1` |
| 2 | `K_B/r_i` | `r_j` |

`β(i,i) = K_B·(2r_i)²/r_i² = 4K_B`

**Nicht separabel: `liquid_bridge`.** Der Faktor
`exp(−((s_i+s_j)/2 − s_opt)²/(2σ²))` enthält beim Ausquadrieren einen Kreuzterm `s_i·s_j`
im Exponenten, und `log1p((V_i+V_j)/V_ref)` ist ebenfalls nicht faktorisierbar. Dieser
Kernel bleibt auf dem O(n²)-Pfad — genau wie heute schon.

### 4.6 Schritt 5 — Die Momente definieren

Wir brauchen über eine Teilmenge `S` **zwei** Momentenfamilien (heute existiert nur die
erste):

```
B_k(S) := Σ_{j∈S}  W_j · g_k(j)      # gewichtet   — entspricht dem heutigen S₀..S₃
A_k(S) := Σ_{j∈S}        g_k(j)      # ungewichtet — NEU, identische Kosten
```

Plus über das nach `W` sortierte `L`, abhängig von der Position von `i`:

```
A_k^≤(i) := Σ_{j∈L, W_j ≤ W_i}  g_k(j)          # Präfixsumme
B_k^>(i) := Σ_{j∈L, W_j > W_i}  W_j·g_k(j)      # Suffixsumme
```

Warum taucht die *ungewichtete* Familie `A_k` neu auf? Genau wegen der Kürzung aus
Schritt 3: dort verschwand das `W_j` aus dem Zähler.

### 4.7 Schritt 6 — Die Diagonale sauber behandeln

Es ist bequemer, zuerst die **volle** Summe inklusive `j = i` zu bilden und den
Diagonalterm anschließend abzuziehen. Definiere:

```
T_i := Σ_{alle j}  W_j·β(i,j) / min(δ_i, δ_j)
```

Für `j = i` gilt `min(δ_i, δ_i) = δ_i`, der Diagonalbeitrag ist also
`W_i·β(i,i)/δ_i`. Damit:

```
Σ_{j≠i} = T_i − W_i·β(i,i)/δ_i
```

Diese Korrektur ist **einheitlich für H und L** — man muss nicht fallunterscheiden.
Kurze Gegenprobe:

- `i ∈ H`: der Diagonalterm liegt im `H×H`-Block mit `ΔW_ii = ΔW = δ_i` ✓
- `i ∈ L`: der Diagonalterm liegt im `L×L`-Block, `≤`-Zweig (denn `W_i ≤ W_i`), Beitrag
  dort `β(i,i)`, und `W_i·β(i,i)/δ_i = W_i·β(i,i)/W_i = β(i,i)` ✓

Damit lautet die Gesamtformel:

```
S_i = T_i − W_i·β(i,i)/δ_i + [W_i>1]·(W_i−1)·β(i,i)/min(δ_i, W_i/2)
R_i* = W_i · S_i
```

### 4.8 Schritt 7 — `T_i` zusammensetzen

**Fall `i ∈ H`** (`δ_i = ΔW`):

```
T_i = Σ_{j∈H} W_j·β(i,j)/ΔW  +  Σ_{j∈L} W_j·β(i,j)/W_j
    = (1/ΔW)·Σ_{j∈H} W_j·β(i,j)  +  Σ_{j∈L} β(i,j)
```

Jetzt die separable Zerlegung einsetzen:

```
Σ_{j∈H} W_j·β(i,j) = Σ_{j∈H} W_j·Σ_k f_k(i)g_k(j)
                   = Σ_k f_k(i)·Σ_{j∈H} W_j g_k(j)
                   = Σ_k f_k(i)·B_k(H)

Σ_{j∈L} β(i,j)     = Σ_k f_k(i)·A_k(L)
```

Also:

```
T_i^{(H)} = (1/ΔW)·Σ_k f_k(i)·B_k(H)  +  Σ_k f_k(i)·A_k(L)
```

**Fall `i ∈ L`** (`δ_i = W_i`):

```
T_i = Σ_{j∈H} W_j·β(i,j)/W_i  +  Σ_{j∈L} W_j·β(i,j)/min(W_i,W_j)
```

Der erste Term: `min(W_i, ΔW) = W_i`, also `1/W_i` vorziehen.
Der zweite Term: Schritt 3 einsetzen. Ergebnis:

```
T_i^{(L)} = (1/W_i)·Σ_k f_k(i)·B_k(H)
          + Σ_k f_k(i)·A_k^≤(i)
          + (1/W_i)·Σ_k f_k(i)·B_k^>(i)
```

### 4.9 Schritt 8 — Die entscheidende Kürzung

Jetzt `R_i* = W_i · S_i` bilden. Für `i ∈ L` multiplizieren wir `T_i^{(L)}` mit `W_i`:

```
W_i · T_i^{(L)} = W_i·(1/W_i)·Σ_k f_k(i)·B_k(H)
                + W_i·Σ_k f_k(i)·A_k^≤(i)
                + W_i·(1/W_i)·Σ_k f_k(i)·B_k^>(i)

                = Σ_k f_k(i)·[ B_k(H) + B_k^>(i) ]   +   W_i·Σ_k f_k(i)·A_k^≤(i)
```

**Das `1/W_i` kürzt sich vollständig gegen den äußeren `W_i`-Faktor weg.**

Das ist mehr als Kosmetik: Für leichte Partikel kann `W_i` sehr klein werden, und eine
Division dadurch wäre numerisch heikel (genau das Problem, das heute `W_MIN_ACTIVE`
notdürftig abfängt). Nach der Kürzung ist die Formel für `i ∈ L` **divisionsfrei**.

Auch der Diagonal-Abzug wird für `i ∈ L` divisionsfrei:
```
W_i · W_i·β(i,i)/δ_i = W_i·W_i·β(i,i)/W_i = W_i·β(i,i)
```

Und der Selbstterm für `i ∈ L`, wo `min(W_i, W_i/2) = W_i/2`:
```
W_i·(W_i−1)·β(i,i)/(W_i/2) = 2·(W_i−1)·β(i,i)
```

Für `i ∈ H` steht im Nenner nur die Konstante `ΔW` bzw. `min(ΔW, W_i/2) ≥ min(ΔW, ΔW/2)`
— ebenfalls unkritisch.

### 4.10 Die Endformeln

**Für `i ∈ H` (`W_i ≥ ΔW`):**

```
R_i* = (W_i/ΔW)·Σ_k f_k(i)·B_k(H)
     + W_i·Σ_k f_k(i)·A_k(L)
     − W_i²·β(i,i)/ΔW
     + [W_i>1]·W_i·(W_i−1)·β(i,i)/min(ΔW, W_i/2)
```

**Für `i ∈ L` (`0 < W_i < ΔW`):**

```
R_i* = Σ_k f_k(i)·[ B_k(H) + B_k^>(i) ]
     + W_i·Σ_k f_k(i)·A_k^≤(i)
     − W_i·β(i,i)
     + [W_i>1]·2·(W_i−1)·β(i,i)
```

### 4.11 Nachrechenbares Zahlenbeispiel

Zum Selbstnachrechnen — bewusst mit dem `constant`-Kernel, damit die β-Werte trivial sind.

**Aufbau:** `ΔW = 10`, `β(i,j) ≡ c = 2` für alle Paare.

| Partikel | `W` | Klasse | `δ` |
|---|---|---|---|
| 1 | 25 | H (25 ≥ 10) | 10 |
| 2 | 4 | L (4 < 10) | 4 |
| 3 | 7 | L (7 < 10) | 7 |

Für `constant`: `K = 1`, `f_0(i) = c = 2`, `g_0(j) = 1`, `β(i,i) = 2`.

---

**A) Direkte Berechnung von `R_2*`** (Brute Force nach der Zielformel aus §3.3):

`i = 2`, `W_2 = 4`, `δ_2 = 4`.

- `j = 1`: `ΔW_21 = min(δ_2, δ_1) = min(4, 10) = 4`  →  `W_1·β/4 = 25·2/4 = 12.5`
- `j = 3`: `ΔW_23 = min(δ_2, δ_3) = min(4, 7) = 4`   →  `W_3·β/4 = 7·2/4 = 3.5`
- Selbstterm: `W_2 = 4 > 1`, `ΔW_22 = min(δ_2, W_2/2) = min(4, 2) = 2`
  →  `(4−1)·2/2 = 3`

```
S_2 = 12.5 + 3.5 + 3 = 19
R_2* = W_2·S_2 = 4·19 = 76
```

**B) Über die hergeleitete Momentenform:**

Mengen: `H = {1}`, `L = {2, 3}`. Sortiert nach `W`: `L = [2 (W=4), 3 (W=7)]`.

Momente:
- `B_0(H) = Σ_{j∈H} W_j·g_0(j) = 25·1 = 25`
- `A_0^≤(2) = Σ_{j∈L, W_j ≤ 4} 1 = |{2}| = 1`
- `B_0^>(2) = Σ_{j∈L, W_j > 4} W_j·1 = 7`

Einsetzen in die L-Endformel:

```
R_2* = f_0(2)·[ B_0(H) + B_0^>(2) ]  +  W_2·f_0(2)·A_0^≤(2)  −  W_2·β(2,2)  +  2·(W_2−1)·β(2,2)
     = 2·[ 25 + 7 ]                  +  4·2·1                −  4·2         +  2·3·2
     = 64                            +  8                    −  8           +  12
     = 76   ✓
```

---

**C) Gegenprobe für ein H-Partikel, `R_1*`:**

Direkt: `i = 1`, `W_1 = 25`, `δ_1 = 10`.
- `j = 2`: `min(10, 4) = 4`  →  `4·2/4 = 2`
- `j = 3`: `min(10, 7) = 7`  →  `7·2/7 = 2`
- Selbst: `ΔW_11 = min(10, 12.5) = 10`  →  `(25−1)·2/10 = 4.8`

```
S_1 = 2 + 2 + 4.8 = 8.8   →   R_1* = 25·8.8 = 220
```

Über die Momentenform (`A_0(L) = |L| = 2`, `B_0(H) = 25`):

```
R_1* = (W_1/ΔW)·f_0·B_0(H) + W_1·f_0·A_0(L) − W_1²·β/ΔW + W_1(W_1−1)·β/min(ΔW, W_1/2)
     = (25/10)·2·25        + 25·2·2          − 625·2/10   + 25·24·2/10
     = 125                 + 100             − 125        + 120
     = 220   ✓
```

### 4.12 Komplexität

| Schritt | Kosten |
|---|---|
| Partition H/L | O(n) |
| Globale Momente `B_k(H)`, `A_k(L)` | O(n·K) |
| **Sortierung von L nach W** | **O(n log n)** |
| Präfix-/Suffixsummen über sortiertes L | O(n·K) |
| Zusammenbau pro `i` (inkl. `searchsorted`) | O(n·(K + log n)) |

**Gesamt: O(n log n)** statt O(n²).

Einordnung bei `n = 4000`: `np.argsort` braucht ca. 10⁻⁴ s, der Pairwise-Pfad dagegen
1.6·10⁷ Kernel-Auswertungen. Gegenüber dem heutigen reinen O(n)-Moment-Mode ist mit einer
Einbuße um Faktor ~2–3 zu rechnen (der Sort), nicht um Faktor 275.

### 4.13 Numerische Randfälle

| Fall | Behandlung |
|---|---|
| `W_i = 0` | Partikel existiert nach §6 nicht mehr (wird exakt geleert und entfernt). Defensiv trotzdem `δ_i = 0 → r_i = 0`. |
| `W_i` gleich für mehrere `j` | Beide Zweige identisch, `searchsorted(side="right")` ✓ (§4.4) |
| `L` leer (alle schwer) | `A_k(L) = 0`, Präfix/Suffix entfallen — Formel bleibt gültig |
| `H` leer (alle leicht) | `B_k(H) = 0` — Formel bleibt gültig |
| `W_i ≤ 1` | Selbstterm entfällt (`[W_i>1]`), exakt wie upstream |
| `r_j ≤ 0` (brownian: `1/r_j`) | Wie heute schon per `continue`-Guard ausschließen; die Momente müssen dieselbe Teilmenge verwenden wie der Pairwise-Pfad |

### 4.14 Auslöschung an der Diagonale — der eine Stolperstein

Der Diagonalterm `j = i` wird erst in die Momente **hinein**gerechnet (Schritt 6) und
danach wieder **abgezogen**. Ist die exakte Antwort 0 — etwa bei einem einzelnen
Partikel ohne Partner, oder wenn alle Partner Gewicht 0 haben — bleibt statt einer
sauberen Null ein Rundungsrest der Größenordnung `eps · W_i · β_ii` stehen.

Gemessen: bei einem einzelnen leichten Partikel liefert der Pairwise-Pfad exakt `0.0`,
die Momentenform `9.9e-32`. In absoluten Zahlen ist das ~20 Größenordnungen unter einem
typischen β (~1e-11) und könnte vom Sampler nie gezogen werden — aber es ist unsauber.

**Behandlung im Code** (`rebuild_r_pairdelta_moment`): Der ausgelöschte Betrag `diag`
wird mitgeführt, und Ergebnisse unterhalb von `1e-14 · diag` werden auf 0 gesetzt.
Damit stimmen beide Pfade auch im Null-Fall exakt überein.

> **Konsequenz für Tests:** Ein rein *relatives* Fehlermaß ist für auslöschungsbehaftete
> Größen untauglich — wo die Referenz exakt 0 ist, ergibt jeder noch so winzige Rest
> 100 % Abweichung. Die Verifikation benutzt daher ein gemischtes Maß
> `|a−b| / max(|a|, |b|, 1e-14 · Skala)`.

### 4.15 Status dieser Herleitung und Verifikationsergebnis

Diese Herleitung stammt **nicht vom Betreuer**, sondern ist im Rahmen dieser Session
entstanden. Sie ist exakte Algebra ohne Näherung — aber im Gegensatz zum übernommenen
Upstream-Code war sie **nicht fremdgeprüft**. Deshalb wurde sie vor jedem
Produktionscode numerisch gegen den O(n²)-Pair-Delta-Pfad gestellt.

**Ergebnis (2026-08-14), 480 Konfigurationen je Kernel** — Populationen mit gemischten
ganzzahligen/fraktionalen Gewichten, rein schweren, rein leichten, vielen exakten
Gleichständen, `W ≤ 1`, `W = 0`, sowie `n ∈ {1, 2, 5, 37, 200, 900}` und
`ΔW ∈ {1, 5, 10}`:

| Prüfung | Kriterium | Ergebnis |
|---|---|---|
| Momentenform vs. Pairwise — `constant` | ≤ 1e-13 | **9.9e-15** |
| Momentenform vs. Pairwise — `sum` | ≤ 1e-13 | **4.2e-15** |
| Momentenform vs. Pairwise — `shear_chin1998` | ≤ 1e-13 | **3.8e-15** |
| Momentenform vs. Pairwise — `brownian_tsouris1995` | ≤ 1e-13 | **3.7e-15** |
| Parallel vs. Serial (Dispatch-Zwillinge) | bitgleich | **0.0e+00** |
| `separable_tables` vs. `kernel.compute_beta` | ≤ 1e-13 | **≤ 6.5e-16** |
| `pick_partner_pairdelta`: Σ w_ij == R_i*/W_i | ≤ 1e-12 | **6.0e-16** |
| Handrechen-Beispiel §4.11 | exakt | **220 / 76 ✓** |

Skripte: `verify_pairdelta_moment.py` (Prototyp, reines numpy) und
`verify_jit_kernels.py` (gegen die produktiven JIT-Kernel).

**Damit greift das Abbruchkriterium nicht** — die Momentenform bleibt. Wäre die
Verifikation gescheitert, wäre sie ersatzlos durch den O(n²)-Pfad ersetzt worden;
der restliche Umbau (§5, §6) wäre davon nicht betroffen gewesen.

---

## 5. Umbauplan Teil A

### 5.1 `kernels/aggregation/jit_kernels.py`

**Neu — generische korrigierte Pairwise-Referenz** (die Grundwahrheit, immer verfügbar):

Struktur 1:1 wie upstream `nb_rebuild_ragg_weighted_pair_delta`:
```
pair_delta = min(delta_i, delta_j)
s += W_j·β(i,j)/pair_delta
self_delta = min(delta_i, 0.5·W_i)
s += (W_i−1)·β(i,i)/self_delta      falls W_i > 1
r_out[i] = W_i·s
```

**Neu — pro separablem Kernel eine `*_pairdelta_moment`-Funktion** nach §4.10.
Gemeinsames Gerüst (Partition, Sort, Präfix/Suffix, `searchsorted`) in einem Helper, der
nur die `f_k`/`g_k`-Tabellen des jeweiligen Kernels bekommt.

**Entfernen — `rebuild_r_array_*_moment`** (die alte, unkorrigierte S₀..S₃-Form).
Sie berechnet eine nachweislich falsche Größe und darf nicht als Option bestehen bleiben.

**Umstellen — `rebuild_r_array_liquid_bridge{,_serial}`** auf Pair-Delta-Form
(bleibt O(n²), da nicht separabel).

### 5.2 `mcpbe_agg.py::_rebuild_all_propensities`

- `delta` weiterhin aus `_delta_from_weights`, aber Schwelle `W_MIN_ACTIVE` → `> 0.0`
- Die nachgelagerte Zeile `np.divide(r, delta, out=r, ...)` **entfällt ersatzlos** —
  die Division steckt jetzt paarweise in den Kernel-Funktionen
- `agg_propensity_mode`: Namen `"moment"` / `"pairwise"` bleiben, Inhalt ist korrigiert

### 5.3 `mcpbe_agg.py::_pick_partner_kernel` (D2)

Ersetzen durch Akkumulations-Loop analog `nb_pick_partner_weighted_pair_delta`:

```
partner_total = R_i* / W_i                        # aus _r_agg
thresh = u_sel · partner_total
für j:  w_ij = W_j·β(i,j)/min(δ_i,δ_j)            # bzw. Selbstterm bei j == i
        akkumulieren bis > thresh
```

Rückgabe `(j, pick_w)` wie bisher, damit `_do_one_agg` und `_compute_agg_dW` unverändert
weiterarbeiten.

**Beabsichtigter Nebeneffekt:** Die heutige Korrelation zwischen Partnerwahl und
SIZEEVAL-Akzeptanz über *dieselbe* Zufallszahl `u_sel` (in `mcpbe_agg.py:365-371` selbst
als bekannter Defekt dokumentiert) wird aufgelöst — SIZEEVAL bekommt eine eigene Ziehung.

### 5.4 Was unberührt bleibt

Der `agglomeration_acceptance_kernel` (`stokes_krit`) wird weiterhin **nach** der Paarwahl
in `_select_pair` aufgerufen. Das ist eine eigene Physik-Erweiterung ohne Upstream-Vorbild
und logisch unabhängig von der Bias-Correction.

---

## 6. Umbauplan Teil B

**Leitregel:** *Jeder* Pfad, der Gewicht konsumiert, konsumiert exakt
`ΔW_i = min(ΔW, W_i)`, bei Selbstkollision exakt `2·δ_ii` mit `δ_ii = min(δ_i, W_i/2)`.

Dann trifft der letzte Event immer exakt `0.0` — und zwar in IEEE-754 **fehlerfrei**,
weil `0.5·W_i` nur den Exponenten dekrementiert und `2·(0.5·W_i) = W_i` exakt gilt.

### 6.1 `mcpbe_time_helper.py`

`W_MIN_ACTIVE` **entfernen**; `delta_from_weights` und `update_delta_single` auf `> 0.0`
zurückstellen (= Upstream-Zustand). Alle Import-Stellen mitziehen
(`mcpbe_agg`, `mcpbe_break`, `mcpbe_nucleation`).

### 6.2 `mcpbe_agg.py::_compute_agg_dW` — Selbstkollision kappen statt ablehnen

```python
# vorher:
if i == j:
    if delta_i <= 0.0 or Wi <= 2.0 * delta_i:
        return 0.0
    dW = min(dW, delta_i)

# nachher:
if i == j:
    delta_ii = min(delta_i, 0.5 * Wi)
    if delta_ii <= 0.0:
        return 0.0
    dW = min(dW, delta_ii)
```

Damit gilt `2·dW ≤ W_i` mit Gleichheit im Grenzfall ⇒ `w_rem` wird exakt `0.0`.
In `_consume_parent_weight`: `> W_MIN_ACTIVE` → `> 0.0`.

### 6.3 `mcpbe_nucleation.py` — beide Sonderpfade angleichen

**(a) `_distribute_one_droplet_with_dW` — Deckelung über das Volumen, nicht das Gewicht.**

Der alte Cap `dW = min(dW, max_physical_droplets / vc_scale)` ist ein kontinuierlicher
Volumenquotient und war die Hauptquelle fraktionaler Gewichte.

> **Sackgasse, die ich zuerst genommen habe** — hier dokumentiert, damit sie niemand
> wiederholt: Der erste Versuch war, den Event zu *verwerfen*, sobald `dW` nicht mehr ins
> Restvolumen passt, und das Volumen in `_liquid_remainder` liegen zu lassen. Das ist
> falsch. Im Fraktional-Tropfen-Zweig von `_finalize_remaining_liquid`
> (`n_droplets_exact < 1.0`) gilt immer `max_physical_droplets < 1`, während
> `dW = min(batch_size, W_i) ≥ 1` ist. Die Bedingung greift dort also **immer** — der
> Zweig war komplett tot. Genau dieser Zweig existiert aber laut seinem eigenen
> Kommentar, um systematische Unter-Zugabe von Flüssigkeit zu vermeiden. Nachgemessen
> mit `check_frac_branch.py`: alle Caps von 0.999 bis 0.01 wurden verworfen.

Der Denkfehler dahinter: Der Deckel ist eine **Volumen**-Bedingung, keine
Gewichts-Bedingung. Abgegeben werden soll insgesamt
`max_physical_droplets · v_droplet` an physikalischer Flüssigkeit. Statt das Paket zu
verkleinern, wird deshalb die Menge **pro Partikel** verkleinert:

```
v_eff = v_droplet · max_dW_allowed / dW      ⟹      dW · vc_scale · v_eff
                                                    == max_physical_droplets · v_droplet
```

`dW` bleibt damit auf dem δ-Raster (keine Gewichtsreste), und die Flüssigkeitsmenge
stimmt exakt. Physikalisch ist das auch die richtige Lesart: ein Partikel bekommt
*weniger Flüssigkeit*, statt dass „ein Bruchteil eines Partikels einen vollen Tropfen"
bekommt. Die Aufnahmefähigkeit wurde weiter oben mit dem *größeren* `v_droplet` geprüft
und bleibt damit konservativ gültig.

Verifiziert mit `check_frac_volume.py`: abgegebenes Volumen == Sollwert für Caps von
0.01 bis 3.0, Abweichung bei 3 % der Auslöschungs-Rauschgrenze der Messung.

**(b) `_manual_agglomerate_particles` (Z. 1945–1950).**

```python
# vorher:
if i == j:
    dW = Wi / 2.0
else:
    dW = min(Wi, Wj)

# nachher:
if i == j:
    dW = min(delta_i, 0.5 * Wi)
else:
    dW = min(delta_i, delta_j)
```

Damit ist diese Parallel-Kopie deckungsgleich mit `_compute_agg_dW` — die in
`Fundamentals.md` §7.2 dokumentierte Copy-Paste-Divergenz wird geschlossen.
Alle `W[i] <= W_MIN_ACTIVE`-Checks (Z. 1252, 1459, 2088, 2114) → `<= 0.0`.

### 6.4 `mcpbe_break.py`

Konsumiert bereits korrekt `dW = _compute_dW_packet(k) = δ_break[k]`.
Nur `w_rem > W_MIN_ACTIVE` → `w_rem > 0.0` (Z. 784) und Import anpassen.

### 6.5 Rekonstruktion bleibt fraktional — bewusst

`reconstruction_mixin.py` (QMOM/2PM/RS/CAM) erzeugt weiterhin nicht-ganzzahlige Gewichte.
Das ist **kein Problem** für die Restgewichts-Frage: Die Leitregel räumt jedes Partikel
exakt leer, unabhängig davon, ob sein Gewicht ganzzahlig ist.

Eine Rundung würde die exakte M₁/M₂-Erhaltung der QMOM-Rekonstruktion zerstören, die laut
Paper §3.3 deren Hauptvorteil gegenüber CAM/RS ist.

**Fazit zur Ganzzahligkeit:** `W` ist ganzzahlig, solange Startgewichte und `ΔW`
ganzzahlig sind und keine Rekonstruktion lief. Danach fraktional — aber weiterhin
**restfrei**, und darauf kommt es an.

---

## 7. Verifikation — durchgeführt am 2026-08-14

**Gesamtergebnis: alle Stufen bestanden.** Der korrigierte Solver trifft die analytische
Lösung auf 0.1 %, die alte Fassung lag um 197 % daneben. Der isolierte Batch-Bias ist von
+7.3 % (21 σ, monoton wachsend) auf einen mit Rauschen verträglichen Rest gefallen. Unter
voller Nassgranulations-Physik bleibt die Feststoffmasse auf 2e-16 relativ erhalten und
die Flüssigkeitsbilanz exakt. Details unten.


### Stufe 0 — Herleitung absichern ✅ BESTANDEN

Siehe §4.15. 480 Konfigurationen je Kernel, größte Abweichung **9.9e-15** gegen ein
Kriterium von 1e-13. Zusätzlich geprüft: Parallel/Serial bitgleich, `separable_tables`
reproduziert `kernel.compute_beta` auf 6.5e-16, Partner-Sampler summiert auf `R_i*/W_i`.

### Stufe 1+2 — Gewichts-Invariante und Masseerhaltung ✅ BESTANDEN

Alle zehn Szenarien aus `tests/bench/scenarios.py`, gefahren mit `solve(maxiter=…)` und
einem Monkeypatch auf `_consume_parent_weight`:

| Szenario | ΔSolid/Solid | ΔLiquid | Entfernt | `w_rem ≠ 0` | `0 < W < 1e-9` |
|---|---|---|---|---|---|
| agg_shear_1d | 0.0 | 0.0 | 484 | **0** | keine |
| agg_shear_1d_large | 0.0 | 0.0 | 286 | **0** | keine |
| agg_constant_1d | 0.0 | 0.0 | 536 | **0** | keine |
| agg_sum_1d | 1.9e-16 | 0.0 | 488 | **0** | keine |
| agg_brownian_1d | 0.0 | 0.0 | 2 | **0** | keine |
| agg_liquid_bridge_1d | 0.0 | 0.0 | 16 | **0** | keine |
| agg_shear_2d | 0.0 | 0.0 | 406 | **0** | keine |
| break_powerlaw_1d | 0.0 | 0.0 | 0 | **0** | keine |
| mix_1d | 0.0 | 0.0 | 233 | **0** | keine |
| granulation_1d | 1.2e-16 | 1e-18 | 14 | **0** | keine |

> **Messfallen, die dabei auftraten** (beide waren Fehler im Prüfskript, nicht im Code):
> 1. `_consume_parent_weight` schreibt die Null nie zurück — es entfernt direkt. Beim
>    Aufruf von `_remove_particle_column` steht in `W[j]` also noch der *alte* Wert.
>    Die zu prüfende Größe ist der berechnete Rest `w_rem`, nicht `W[j]`.
> 2. Die Szenarien müssen mit `solve(maxiter=sc.maxiter)` gefahren werden. Ohne das
>    Limit läuft `agg_constant_1d` gegen ~2.9e18 Ereignisse (`t_end = 2e6`).

### Stufe 4 — Laufzeit ✅

Scherkernel, `ΔW = 5`, ganzzahlige Gewichte 1…50, ein Propensity-Rebuild:

| n | Moment [ms] | Pairwise [ms] | Faktor |
|---|---|---|---|
| 500 | 0.022 | 0.368 | 17× |
| 1000 | 0.049 | 1.539 | 32× |
| 2000 | 0.098 | 6.397 | 65× |
| 4000 | 0.208 | 21.575 | **104×** |

Die Momentenform skaliert wie hergeleitet (näherungsweise linear, der `log n`-Anteil des
Sortierens fällt kaum ins Gewicht). **Der Speedup bleibt also erhalten** — die Sorge aus
der Planungsphase, die Pair-Delta-Korrektur würde den O(n)-Vorteil kosten, hat sich durch
die Herleitung erledigt.

### Stufe 3 — Physik-Regression ✅ BESTANDEN

**Methodik.** Nicht „git vorher/nachher" — das würde diesen Umbau mit anderen, schon
vorher offenen Änderungen im Arbeitsverzeichnis vermischen. Stattdessen wie Paper §3.4:
reine Agglomeration mit **konstantem Kernel**, weil dafür eine geschlossene analytische
Lösung existiert, und die alten Schemata werden per Monkeypatch im *aktuellen* Code
nachgebildet. Damit ist der Vergleich frei von Störeinflüssen und hat eine absolute
Referenz — es lässt sich also sagen, welche Variante *richtiger* ist, nicht nur, dass
sie sich unterscheiden.

Analytische Referenz (Smoluchowski, konstanter Kernel, reine Agglomeration):

```
dn/dt = −½·β·n²        ⟹        n(t) = n₀ / (1 + β·n₀·t/2)
```

mit `n = ΣW/Vc` (physikalische Anzahldichte).

#### Experiment A — der fehlende `W_i`-Vorfaktor

400 Rechenpartikel, `W₀ = 400`, Fehler in `n_phys(t_end)` gegen die analytische Lösung:

| ΔW | korrigiert | alter Stand | phys. Kollisionen korrigiert / alt |
|---|---|---|---|
| 1 | **4.4e-5** | 1.97 | 106 669 / 1 586 |
| 20 | **1.0e-3** | 1.98 | 106 720 / 840 |
| 50 | **2.5e-3** | 1.98 | 106 800 / 850 |

Analytisch zu erwarten sind `160 000·(1 − 1/3) = 106 667` Kollisionen — der korrigierte
Stand trifft das auf 0.1 % genau, der alte um Faktor ~100 zu wenig.

**Wichtige Lesart:** Der alte Fehler ist bei ΔW = 1 (1.970) genauso groß wie bei ΔW = 50
(1.984). Er hängt also **nicht** von der Batchgröße ab und ist damit *nicht* der
Batch-Bias aus Paper Gl. 33, sondern der separate Defekt des fehlenden `W_i`-Vorfaktors:
die Propensity war um ~`W` zu klein, also der Zeitschritt um ~`W` zu groß.

#### Experiment B — der Batch-Bias isoliert (Ensemble, 12 Seeds)

Beide Varianten haben den `W_i`-Vorfaktor; sie unterscheiden sich **ausschließlich** in
`min(δ_i, δ_j)` gegenüber `δ_i`. `W₀ = 80`, `ΔW = 50`, damit sofort viele leichte
Partikel entstehen. Gezeigt ist der **vorzeichenbehaftete** Fehler, Mittel ± Standardfehler:

| t/t_end | korrigiert | pair-delta abgeschaltet |
|---|---|---|
| 0.2 | −2.9e-3 ± 1.3e-3 | **+1.24e-2** ± 1.1e-3 |
| 0.4 | −0.95e-3 ± 2.1e-3 | **+2.97e-2** ± 2.1e-3 |
| 0.6 | −4.2e-3 ± 2.0e-3 | **+4.48e-2** ± 2.4e-3 |
| 0.8 | −4.3e-3 ± 2.2e-3 | **+5.95e-2** ± 3.1e-3 |
| 1.0 | −2.3e-3 ± 2.8e-3 | **+7.30e-2** ± 3.5e-3 |

- **Unkorrigiert:** +7.3e-2 am Ende = **21 Standardfehler** von null entfernt, streng
  monoton wachsend (0.004 → 0.012 → 0.022 → 0.030 → 0.037 → 0.045 → 0.054 → 0.060 →
  0.067 → 0.073). Das ist ein akkumulierender systematischer Fehler — genau die Signatur
  aus Paper §3.4. Das positive Vorzeichen bedeutet: zu wenige Agglomerationen, das
  Schema unterabtastet Paare, bei denen der *Partner* der begrenzende ist.
- **Korrigiert:** −2.3e-3 ± 2.8e-3 am Ende, also innerhalb eines Standardfehlers von
  null, ohne monotonen Trend. **24× kleiner** als unkorrigiert.

**Ehrliche Einschränkung:** Der korrigierte Verlauf ist durchweg leicht negativ
(≈ −3e-3), das ist vermutlich kein reines Rauschen. Das passt zum Paper §3.5, das
ausdrücklich festhält, dass batch-weise Ausführung die Dynamik **auch nach** der
Bias-Korrektur noch stört („state-freezing error"). Bestätigt wird das durch Experiment A:
der Restfehler skaliert mit ΔW (4.4e-5 bei ΔW = 1 → 2.5e-3 bei ΔW = 50) und verschwindet
bei ΔW = 1. Es ist also die dokumentierte Batch-Störung, kein Rest-Bias in der Korrektur.
Praktische Konsequenz (Paper §3.6): `ΔW_agg` konservativ wählen — hier war ΔW = 50
bewusst groß gewählt, um den Effekt sichtbar zu machen.

### Stufe 5 — Volle Nassgranulation ✅ BESTANDEN

`Trials/test_powerlaw_rumpf_full.py` (2000 Startpartikel, `cone_model`, `stokes_krit`,
`powerlaw_rumpf`, Nucleation + kontinuierliche Prozesse, 20 s simuliert):

| Größe | Ergebnis |
|---|---|
| Feststoffmasse | 8.231811e-06 kg → 8.231811e-06 kg, Δ = **−1.69e-21 kg** (2e-16 relativ) |
| Flüssigkeit zugegeben | 5.000e-10 m³ erwartet, 5.000e-10 m³ zugegeben, **0.0000 %** |
| Flüssigkeit im System | 5.000e-10 m³, **0.0000 %** |
| Tropfenzahl | 7639.44 erwartet / 7639.44 tatsächlich, **100.00 %** |
| Agglomeration | 76 akzeptiert, 5467 von Stokes abgelehnt |
| Breakage | 100 Ereignisse |
| n_comp | 2 000 → 11 775 (n_phys 790 056) |
| Porosität | 0.800 → 0.824 (min/max 0.739 / 0.886) |
| Sättigung | 0.000 → 0.354 (min/max 0.000 / 0.709) |

Die Feststoffabweichung liegt bei 2e-16 relativ, also an der Maschinengenauigkeit.
Damit ist der Umbau auch unter voller Nassgranulations-Physik — inklusive des
Nucleation-Pfads, der als einziger Teil kein Upstream-Vorbild hat — sauber.

> **Präzisions-Hinweis:** Dieser Lauf belegt die Gesamtkonsistenz, isoliert aber
> *nicht* die Reparatur des Fraktional-Tropfen-Zweigs aus §6.3(a). Der dort betroffene
> Betrag liegt bei ≤ 1 Tropfenvolumen (5.2e-22 m³) gegen 5e-10 m³ Gesamtflüssigkeit,
> also ~1e-12 relativ — beide Fassungen zeigen hier 0.0000 %. Für diesen Nachweis sind
> `check_frac_branch.py` und `check_frac_volume.py` zuständig, die auf der nötigen
> Präzision messen.

### Noch offen

Nichts Blockierendes. Optional nachziehbar: `test_droplet_10um_stable.py` und
`test_droplet_20um_instable.py` (identische Physik, nur Tropfengröße variiert).

**Bekannte Regression:** `dim = 2` fällt jetzt auf den generischen O(n²)-Python-Pfad
zurück (Warnung wird einmalig ausgegeben), weil dort `alpha` paarabhängig ist und nicht
in `p0` gefaltet werden kann. Vorher lief 2D über die kompilierten Kernel. Für die
Nassgranulation (durchgehend `dim = 1`) ohne Bedeutung; falls 2D wieder wichtig wird,
ist die saubere Lösung, `alpha_ij` als zusätzliche separable Terme in die `f_k`/`g_k`-
Tabellen aufzunehmen (es ist eine Summe von Produkten aus i- und j-Anteilen, also
separabel — nur mit größerem K).

---

### Die Prüfskripte

Alle vier liegen in `mcpbe/src/wmcpbe/Trials/` und sind ohne Argumente ausführbar
(Pfade werden relativ zum Skript aufgelöst, `pytest` wird nicht gebraucht):

| Skript | Stufe | Was es prüft |
|---|---|---|
| `verify_jit_kernels.py` | 0 | Momentenform vs. O(n²)-Referenz, Parallel/Serial, `f_k`/`g_k`-Tabellen, Sampler-Konsistenz |
| `run_conservation.py` | 1+2 | Masseerhaltung und Gewichts-Invariante über alle zehn Szenarien |
| `stufe3_bias.py` | 3 | Fehler gegen die analytische Lösung, `W_i`-Defekt und Batch-Bias getrennt |
| `stufe3_ensemble.py` | 3 | Ensemble über 12 Seeds: Bias oder Rauschen (vorzeichenbehaftet) |
| `check_frac_branch.py` | 6.3a | Fraktional-Tropfen-Zweig lebt (kein Verwerfen bei Cap < 1) |
| `check_frac_volume.py` | 6.3a | Abgegebenes Flüssigkeitsvolumen == `max_physical_droplets · v_droplet` |

```bash
"C:/Users/ericb/anaconda3/envs/Masterarbeitv4/python.exe" -u mcpbe/src/wmcpbe/Trials/verify_jit_kernels.py
```

## 8. Risiken und Status

1. ~~**Die Momentenform ist ungeprüfte Eigenarbeit.**~~ → durch Stufe 0 erledigt (§4.15).
   Bleibt als Hinweis: Sie ist *meine* Herleitung, nicht die des Betreuers. Wer sie
   anzweifelt, hat mit `verify_jit_kernels.py` ein Skript zur Hand, das sie in Sekunden
   gegen die O(n²)-Referenz stellt.
2. **Jede Trajektorie ändert sich.** Der Partner wird jetzt nach `W_j β/ΔW_ij` gezogen
   statt nach `W_j` allein, und SIZEEVAL hat eine eigene Zufallszahl. Bestehende
   Referenzergebnisse (`Trials/Results/sensitivity*`, `tests/golden_reference.json`)
   sind nicht mehr bit-vergleichbar. Unvermeidbar und beabsichtigt — die alten Zahlen
   stammen aus dem verzerrten Sampler.
3. **Der Nucleation-Umbau (§6.3) berührt eigene Physik ohne Upstream-Vorbild.** Die
   Zusicherung „vom Betreuer geprüft" greift hier *nicht*. `granulation_1d` ist sauber
   durchgelaufen (ΔSolid = 1.2e-16), aber das ist ein kleines Szenario — die vollen
   Trials aus §7 „Noch offen" sollten nachgezogen werden.
4. **Historische Warnung** (`Fundamentals.md` §7.1): Genau dieser Codepfad hatte schon
   einmal einen Faktor-2-Massefehler durch einen Debug-Block, der Zustand mutierte.
   Die `mcpbe_debug_mass`-Blöcke in `_merge_pair` / `_consume_parent_weight` wurden
   beim Umbau nicht angefasst und bleiben seiteneffektfrei.
5. **`dim = 2` ist langsamer geworden** (generischer Python-Pfad statt kompiliert),
   siehe §7 „Bekannte Regression".

---

## Anhang: Betroffene Dateien

| Datei | Art des Eingriffs |
|---|---|
| `kernels/aggregation/jit_kernels.py` | groß — alte Moment-Kernel raus, Pair-Delta-Moment + generische Referenz rein |
| `mcpbe_agg.py` | mittel — `_rebuild_all_propensities`, `_compute_raw_propensities`, `_pick_partner_kernel`, `_compute_agg_dW`, `_consume_parent_weight` |
| `mcpbe_time_helper.py` | klein — `W_MIN_ACTIVE` raus, Schwellen auf `> 0.0` |
| `mcpbe_nucleation.py` | mittel — `_distribute_one_droplet_with_dW`, `_manual_agglomerate_particles`, 4 Cutoff-Stellen |
| `mcpbe_break.py` | minimal — 1 Cutoff-Stelle, Import |
| `mcpbe/docs/Fundamentals.md` | Doku nachziehen (§3.1, §7.2, neue Leitregel) |

**Nicht angefasst:** alle `kernels/*` außer `aggregation/jit_kernels.py`,
`particle_merger.py`, `reconstruction_mixin.py`, `mcpbe_continuous_processes.py`,
`mcpbe_post.py`, `mcpbe_base.py`.
