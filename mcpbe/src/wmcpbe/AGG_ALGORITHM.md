# Agglomeration-Algorithmus: Detaillierte Rechenschritte

Dieses Dokument beschreibt **jeden einzelnen Rechenschritt** in `mcpbe_agg.py` in der exakten Ausführungsreihenfolge.

---

## Überblick: Die Hauptmethode `_do_one_agg()`

```
┌─────────────────────────────────────────────────────────────────┐
│                    _do_one_agg()                                │
│                  (Ein Agglomeration-Event)                      │
├─────────────────────────────────────────────────────────────────┤
│  Phase 1: Vorbereitung                                          │
│  Phase 2: Erstes Partikel auswählen (Fenwick-Sampling)          │
│  Phase 3: Zweites Partikel auswählen + Akzeptanztest            │
│  Phase 4: Paketgröße ΔW berechnen                               │
│  Phase 5: Kind-Partikel erzeugen (Verschmelzung)                │
│  Phase 6: Eltern-Partikel reduzieren                            │
│  Phase 7: Propensities neu berechnen                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## PHASE 0: Vorbedingungen (`_rebuild_all_propensities` vor Aufruf)

Bevor `_do_one_agg()` aufgerufen wird, MÜSSEN die Propensities berechnet sein:

### Schritt 0.1: Delta-Werte berechnen

**Zweck:** Normalisierung für gewichtete Sampling-Verfahren

```python
a = self.a_tot  # Anzahl aktiver Partikel
W = self.W[:a]  # Gewichte aller Partikel
dW_const = agg_dW_max  # Maximale Paketgröße (z.B. 50.0)

# Delta für jedes Partikel: δᵢ = min(dW_const, Wᵢ)
delta = np.minimum(dW_const, W)
```

**Warum?** Delta begrenzt die maximale Änderung pro Event für numerische Stabilität.

---

### Schritt 0.2: Roh-Propensities rᵢ berechnen (JIT-optimiert)

**Formel:** 
```
rᵢ = Σⱼ Wⱼ × β(i,j)    für j ≠ i
rᵢ += (Wᵢ - 1) × β(i,i)  für Selbst-Agglomeration
```

**Für Shear-Kernel (JIT):**
```python
kernel_name = 'shear_chin1998'

if kernel_name == 'shear_chin1998':
    r = np.zeros(a, dtype=float)
    rebuild_r_array_shear(
        corr_beta=1e-3,      # Kollisionseffizienz
        g=1000.0,            # Scherrate [1/s]
        R=R,                 # Radien array
        W=W,                 # Gewichte array
        r=r                  # Output array
    )
```

**JIT-Funktion (im Hintergrund):**
```python
@njit(cache=True)
def rebuild_r_array_shear(corr_beta, g, R, W, r):
    """
    Berechne r[i] = Σ[j] W[j] × β(i,j) für alle i
    
    β(i,j) = corr_beta × g × (R[i] + R[j])³
    """
    n = len(R)
    for i in range(n):
        ri = 0.0
        r1 = R[i]
        for j in range(n):
            r2 = R[j]
            
            # Shear-Kernel Formel
            beta_ij = corr_beta * g * (r1 + r2)**3
            
            if i == j:
                # Selbst-Agglomeration: (W[i]-1) Paare
                ri += max(0.0, W[j] - 1) * beta_ij
            else:
                # Verschiedene Partikel: W[j] Paare
                ri += W[j] * beta_ij
        
        r[i] = ri
```

**Warum O(n²)?** Jedes Partikel kann mit jedem anderen kollidieren → vollständige Paarberechnung nötig.

---

### Schritt 0.3: Propensities normalisieren

```python
# Normierte Propensity: r_norm[i] = r[i] / delta[i]
r_normalized = np.divide(
    r,                    # Roh-Propensity
    delta,                # Normalisierungsfaktor
    out=np.zeros_like(r),
    where=delta > 0.0     # Vermeide Division durch Null
)

# Speichern für Sampler
self._r_agg[:a] = r_normalized
self._delta_agg[:a] = delta
```

**Warum normalisieren?** Das Delta-basierte Sampling ermöglicht größere Zeitschritte bei kleinen Gewichten.

---

### Schritt 0.4: Fenwick Tree aufbauen

```python
self._agg_sampler = FenwickSampler(self._r_agg[:a])
```

**Fenwick Tree Struktur:**
```
Gewichte: [r₀, r₁, r₂, r₃, r₄] = [0.1, 0.3, 0.2, 0.3, 0.1]

Binary Indexed Tree (1-basiert):
tree[1] = r₀                    = 0.1
tree[2] = r₀ + r₁               = 0.4
tree[3] = r₂                    = 0.2
tree[4] = r₀ + r₁ + r₂ + r₃     = 0.9
tree[5] = r₄                    = 0.1

Total = tree[4] + tree[5] = 1.0
```

**Vorteil:** Sampling in O(log n) statt O(n).

---

## PHASE 1: Vorbereitung in `_do_one_agg()`

### Schritt 1.1: Guard-Clause

```python
a = self.a_tot
if a < 2:
    self._last_agg_dW = 0.0
    return  # Keine Agglomeration möglich mit < 2 Partikeln
```

---

### Schritt 1.2: Arrays cachen (für Performance)

```python
R = (self.X[:a] * 0.5).astype(np.float64)  # Radien aus Durchmessern
W = self.W[:a].astype(np.float64)          # Gewichte
```

**Warum cachen?** Vermeidet wiederholtes Slicing in Schleifen.

---

### Schritt 1.3: Kollisionseffizienz α vorbereiten

**Für 1D (ein Komponent):**
```python
if self.dim == 1:
    alpha1d = float(self.alpha_prim if np.ndim(self.alpha_prim) == 0 
                    else np.mean(self.alpha_prim))
    # Typisch: alpha1d = 1.0 (jede Kollision führt zu Agglomeration)
```

**Für 2D (zwei Komponenten):**
```python
else:
    alpha1d = 1.0
    ap = np.asarray(self.alpha_prim, dtype=np.float64)
    alpha4 = ap if ap.size == 4 else np.ones(4, dtype=np.float64)
    # alpha4 = [α₀₀, α₀₁, α₁₀, α₁₁] für Komponentenkombinationen
```

---

### Schritt 1.4: Size-Evaluation Parameter laden

```python
SIZEEVAL = int(getattr(self, "SIZEEVAL", 1))      # 1 = aktiviert
X_SEL = float(getattr(self, "X_SEL", 0.31))       # Selektionsbreite
Y_SEL = float(getattr(self, "Y_SEL", 1.06))       # Gauß-σ relativ
Vmean2 = float(np.mean(self.V0[-1, :]) ** 2)      # Target-Volumen²
```

**Zweck:** Bevorzugt Kollisionen um ein Zielvolumen herum (optional).

---

### Schritt 1.5: Zufallszahlen generieren

```python
u_sel = float(self._rng.random())   # Für Partnerauswahl
u_acc = float(self._rng.random())   # Für Akzeptanztest
```

---

## PHASE 2: Erstes Partikel auswählen

### Schritt 2.1: Sampling aus Propensity-Verteilung

```python
i = self._agg_sampler.sample(self._rng)
```

**Ablauf im FenwickSampler:**

```python
# 1. Total Propensity berechnen
total = self._tree_total()  # = Σᵢ rᵢ

# 2. Zufälligen Punkt wählen
u = rng.random() * total    # z.B. u = 0.63 × total

# 3. Binary Lifting auf Fenwick Tree
i = 0
bit = 1 << (n.bit_length() - 1)  # Größte 2er-Potenz ≤ n

while bit:
    nxt = i + bit
    if nxt <= n and tree[nxt] <= s:
        s -= tree[nxt]
        i = nxt
    bit >>= 1

return i  # Ausgewählter Index
```

**Beispiel:**
```
r = [0.1, 0.3, 0.2, 0.3, 0.1], total = 1.0
u = 0.63

Binary Lifting:
  bit=4: tree[4]=0.9 > 0.63 → nicht nehmen
  bit=2: tree[2]=0.4 ≤ 0.63 → nehmen, s=0.63-0.4=0.23, i=2
  bit=1: tree[3]=0.2 ≤ 0.23 → nehmen, s=0.23-0.2=0.03, i=3
  
Ergebnis: i = 3 (Partikel mit r₃=0.3 wurde gewählt)
```

**Warum dieses Verfahren?** Partikel mit höherer Propensity werden häufiger gewählt → physikalisch korrekt.

---

## PHASE 3: Zweites Partikel auswählen (`_pick_partner_kernel`)

### Schritt 3.1: Gewichtete Partnerauswahl

```python
total_W = np.sum(W)              # Σⱼ Wⱼ
cumsum_W = np.cumsum(W)          # Kumulative Summe
target = u_sel * total_W         # Zielpunkt

j = int(np.searchsorted(cumsum_W, target, side='right'))
j = min(j, a - 1)                # Clamp an Array-Grenze
```

**Beispiel:**
```
W = [10, 20, 30, 40], total_W = 100
u_sel = 0.63 → target = 63

cumsum_W = [10, 30, 60, 100]
searchsorted(63) → Index 3 (weil 63 > 60)

Ergebnis: j = 3 (Partikel mit W₃=40)
```

**Warum proportional zu W?** Jedes PHYSIKALISCHE Partikel soll gleiche Chance haben.

---

### Schritt 3.2: Beta(i,j) berechnen (Kernel-Aufruf)

```python
r1 = float(R[i])
r2 = float(R[j])

beta_ij = self.kernel_manager.compute_beta(
    r1, r2,
    particle1_idx=i,
    particle2_idx=j,
    solver=self
)
```

**Delegation an konkreten Kernel (z.B. ShearChinKernel):**

```python
# In ShearChinKernel.compute_beta():
if r1 <= 0 or r2 <= 0:
    return 0.0

# JIT-aufgerufene Funktion
beta = _beta_shear_jit(corr_beta=1e-3, g=1000, r1=r1, r2=r2)
# = 1e-3 × 1000 × (r1 + r2)³

return max(0.0, float(beta))
```

**Beispielrechnung:**
```
r1 = 1.0e-6 m, r2 = 2.0e-6 m
beta = 1e-3 × 1000 × (1e-6 + 2e-6)³
     = 1 × (3e-6)³
     = 27e-18 m³/s
     = 2.7e-17 m³/s
```

---

### Schritt 3.3: Kollisionseffizienz α(i,j) berechnen

**Für 1D:**
```python
alpha_ij = alpha1d  # Konstant, z.B. 1.0
```

**Für 2D (Komponenten-Mixing):**
```python
Vi0 = V0[i]; Vi1 = V1[i]; Vti = Vi0 + Vi1
Vj0 = V0[j]; Vj1 = V1[j]; Vtj = Vj0 + Vj1

# Volumenanteile jeder Komponente
fi0 = Vi0 / Vti    # Anteil Komponente 0 in Partikel i
fi1 = Vi1 / Vti    # Anteil Komponente 1 in Partikel i
fj0 = Vj0 / Vtj    # Anteil Komponente 0 in Partikel j
fj1 = Vj1 / Vtj    # Anteil Komponente 1 in Partikel j

# Wahrscheinlichkeiten für Komponentenkombinationen
p0 = fi0 * fj0     # Beide Komponente 0
p1 = fi0 * fj1     # i:Komp0, j:Komp1
p2 = fi1 * fj0     # i:Komp1, j:Komp0
p3 = fi1 * fj1     # Beide Komponente 1

# Gemischte Effizienz
alpha_ij = p0*α₀₀ + p1*α₀₁ + p2*α₁₀ + p3*α₁₁
```

**Beispiel:**
```
Partikel i: V0=60%, V1=40% → fi0=0.6, fi1=0.4
Partikel j: V0=30%, V1=70% → fj0=0.3, fj1=0.7

p0 = 0.6×0.3 = 0.18
p1 = 0.6×0.7 = 0.42
p2 = 0.4×0.3 = 0.12
p3 = 0.4×0.7 = 0.28

alpha4 = [1.0, 0.5, 0.5, 0.8]  # Benutzerdefinierte Effizienzen

alpha_ij = 0.18×1.0 + 0.42×0.5 + 0.12×0.5 + 0.28×0.8
         = 0.18 + 0.21 + 0.06 + 0.224
         = 0.674
```

---

### Schritt 3.4: Effektive Beta berechnen

```python
beta_eff = beta_ij * alpha_ij
```

**Beispiel:**
```
beta_ij = 2.7e-17 m³/s
alpha_ij = 0.674

beta_eff = 2.7e-17 × 0.674 = 1.82e-17 m³/s
```

---

### Schritt 3.5: Paar-Propensity berechnen

```python
if i == j:
    # Selbst-Agglomeration: (W[i]-1) distinct pairs
    pick_w = max(0.0, float(W[i] - 1)) * beta_eff
else:
    # Verschiedene Partikel: W[j] Paare
    pick_w = float(W[j]) * beta_eff
```

**Beispiel (verschiedene Partikel):**
```
W[j] = 40
beta_eff = 1.82e-17 m³/s

pick_w = 40 × 1.82e-17 = 7.28e-16 m³/s
```

**Beispiel (Selbst-Agglomeration):**
```
W[i] = 50
beta_eff = 1.82e-17 m³/s

pick_w = (50 - 1) × 1.82e-17 = 49 × 1.82e-17 = 8.92e-16 m³/s
```

**Warum (W[i]-1)?** Ein Partikel kann nicht mit sich selbst kollidieren, nur mit anderen im gleichen Paket.

---

### Schritt 3.6: Size-Evaluation Test (optional)

```python
if SIZEEVAL != 0:
    # Volumina der Partikel
    if dim == 1:
        Vi = V0[i]
        Vj = V0[j]
    else:
        Vi = V0[i] + V1[i]
        Vj = V0[j] + V1[j]
    
    # Zielvolumen und Breite
    V_target = np.sqrt(Vmean2)           # z.B. 1e-18 m³
    sigma_V = V_target * Y_SEL           # z.B. 1e-18 × 1.06
    
    # Gauß'sche Gewichtung für beide Partikel
    size_factor_i = exp(-0.5 × ((Vi - V_target) / sigma_V)²)
    size_factor_j = exp(-0.5 × ((Vj - V_target) / sigma_V)²)
    size_factor = size_factor_i * size_factor_j
    
    # Akzeptanztest
    if u_sel > size_factor:
        return -1, 0.0  # Abgelehnt!
```

**Beispiel:**
```
V_target = 1e-18 m³
sigma_V = 1.06e-18 m³
Vi = 0.8e-18 m³, Vj = 1.2e-18 m³
u_sel = 0.63

size_factor_i = exp(-0.5 × ((0.8-1.0)/1.06)²) = exp(-0.5 × 0.036) = 0.982
size_factor_j = exp(-0.5 × ((1.2-1.0)/1.06)²) = exp(-0.5 × 0.036) = 0.982
size_factor = 0.982 × 0.982 = 0.964

Test: 0.63 ≤ 0.964? JA → akzeptiert
```

**Zweck:** Bevorzugt Kollisionen nahe dem Zielvolumen (für bestimmte Experimente).

---

### Schritt 3.7: Rückgabe

```python
return j, pick_w  # Partner-Index und Paar-Propensity
```

---

## PHASE 4: Paketgröße ΔW berechnen

### Schritt 4.1: Gesamt-Propensity holen

```python
sum_prop_before = float(self._agg_sampler.total())
# = Σᵢ rᵢ (Summe aller Propensities vor diesem Event)
```

---

### Schritt 4.2: Paar-Propensity berechnen

```python
Wi = float(self.W[i])
pair_prop = Wi * pick_w
```

**Bedeutung:**
```
pair_prop = W[i] × W[j] × β_eff    (für i ≠ j)
pair_prop = W[i] × (W[i]-1) × β_eff (für i = j)
```

**Beispiel:**
```
Wi = 50, pick_w = 7.28e-16

pair_prop = 50 × 7.28e-16 = 3.64e-14 m³/s
```

---

### Schritt 4.3: ΔW-Modus bestimmen

```python
mode = str(getattr(self, "agg_dW_mode", "const")).lower()
dW_max = float(getattr(self, "agg_dW_max", 1.0))
dW_min = float(getattr(self, "agg_dW_min", 1.0))

if mode == "const":
    dW = dW_max
else:
    # Relativer Beitrag dieses Paares zur Gesamt-Propensity
    f = pair_prop / sum_prop_before
    f = np.clip(f, 0.0, 1.0)
    
    alpha = float(getattr(self, "agg_dW_alpha", 100.0))
    
    if mode == "sqrt":
        dW = dW_min + alpha × sqrt(f) × (dW_max - dW_min)
    else:  # "linear"
        dW = dW_min + alpha × f × (dW_max - dW_min)
```

**Beispiel (const):**
```
dW_max = 50.0
dW = 50.0
```

**Beispiel (linear):**
```
sum_prop_before = 1.0e-12
pair_prop = 3.64e-14
f = 3.64e-14 / 1.0e-12 = 0.0364
alpha = 100.0

dW = 1.0 + 100 × 0.0364 × (50 - 1)
   = 1.0 + 3.64 × 49
   = 1.0 + 178.36
   = 179.36 → gecapped auf dW_max = 50.0
```

---

### Schritt 4.4: Delta-Limits anwenden

```python
# Update delta für betroffene Partikel
delta_i = self._update_delta_single(i, attr_name="_delta_agg", dW_const=dW_max)
delta_j = self._update_delta_single(j, attr_name="_delta_agg", dW_const=dW_max)

# Limits prüfen
if i == j:
    # Selbst-Agglomeration: braucht 2×ΔW vom selben Partikel
    if delta_i <= 0.0 or Wi <= 2.0 * delta_i:
        return 0.0
    dW = min(dW, delta_i)
else:
    # Verschiedene Partikel
    dW = min(dW, Wi, Wj, delta_i, delta_j)
```

**Beispiel:**
```
Wi = 50, Wj = 40, delta_i = 50, delta_j = 40, dW = 50

dW = min(50, 50, 40, 50, 40) = 40
```

---

## PHASE 5: Kind-Partikel erzeugen

### Schritt 5.1: Eltern-Volumina laden

```python
Vi_comp = self.V_flat[:self.dim, i].copy()  # Komponent-Volumen (V_solid für dim=1)
Vj_comp = self.V_flat[:self.dim, j].copy()
Vi_dry = self.V_flat[-1, i]                 # Trockenvolumen
Vj_dry = self.V_flat[-1, j]
```

---

### Schritt 5.2: Feststoffvolumina berechnen (Massenerhaltung!)

```python
if hasattr(self, "porosity"):
    poro_i = self.porosity[i]
    poro_j = self.porosity[j]
    
    if np.isnan(poro_i):  # Vollkörper
        V_solid_i = Vi_dry
    else:  # Porös
        V_solid_i = Vi_dry × (1.0 - poro_i)
    
    if np.isnan(poro_j):
        V_solid_j = Vj_dry
    else:
        V_solid_j = Vj_dry × (1.0 - poro_j)
else:
    V_solid_i = np.sum(Vi_comp)
    V_solid_j = np.sum(Vj_comp)
```

**Beispiel:**
```
Vi_dry = 1.0e-18 m³, poro_i = 0.4
Vj_dry = 2.0e-18 m³, poro_j = 0.5

V_solid_i = 1.0e-18 × (1 - 0.4) = 0.6e-18 m³
V_solid_j = 2.0e-18 × (1 - 0.5) = 1.0e-18 m³
```

---

### Schritt 5.3: Feststoffvolumina addieren (MASSE ERHALTEN!)

```python
V_solid_merged = V_solid_i + V_solid_j
# = 0.6e-18 + 1.0e-18 = 1.6e-18 m³
```

**Kritisch:** Dies ist der Kern der Massenerhaltung!

---

### Schritt 5.4: Neues Partikel anlegen

```python
Vnew_comp = Vi_comp + Vj_comp  # Für dim=1: V_solid_merged

self._append_particle_column(Vnew_comp)
new_idx = self.a_tot - 1
self.W[new_idx] = dW  # = 40 (aus Phase 4)
```

**Was `_append_particle_column` tut:**
```python
def _append_particle_column(self, V_new_comp):
    # Kapazität prüfen, ggf. erweitern
    if self.a_tot >= self._cap:
        self._ensure_capacity_for(extra=1)
    
    # Neue Spalte schreiben
    self.V_flat[:self.dim, self.a_tot] = V_new_comp
    self.V_flat[-1, self.a_tot] = np.sum(V_new_comp)  # Temporär!
    
    self.a_tot += 1
```

---

### Schritt 5.5: Porosität des Kind-Partikels berechnen

```python
# Kollisionsenergie schätzen
rho = 1000.0  # kg/m³
d_eff = (self.X[i] + self.X[j]) * 0.5  # Mittlerer Durchmesser
g = float(getattr(self, 'G', 1000.0))   # Scherrate
v_rel = g * d_eff                       # Relativgeschwindigkeit
v_particle = Vi_dry + Vj_dry            # Gesamtvolumen
m = rho * v_particle                    # Masse
E_coll = 0.5 * m * v_rel ** 2           # Kinetische Energie

# Porositäts-Kernel aufrufen
porosity_kernel = self.kernel_manager.porosity_growth_kernel
# (oder Default: volume_mixing)

V_dry_merged, poro_merged = porosity_kernel.compute_merged_porosity(
    v_dry1=Vi_dry, poro1=poro_i,
    v_dry2=Vj_dry, poro2=poro_j,
    v_liq1=lv_i, v_liq2=lv_j,
    sat1=sat_i, sat2=sat_j,
    collision_energy=E_coll,
    solver=self
)
```

**VolumeMixingKernel Berechnung:**
```python
# Eltern dekomponieren
if np.isnan(poro_i):
    v_solid1 = Vi_dry
    v_pore1 = 0.0
else:
    v_solid1 = Vi_dry × (1 - poro_i)  # = 0.6e-18
    v_pore1 = Vi_dry × poro_i          # = 0.4e-18

if np.isnan(poro_j):
    v_solid2 = Vj_dry
    v_pore2 = 0.0
else:
    v_solid2 = Vj_dry × (1 - poro_j)  # = 1.0e-18
    v_pore2 = Vj_dry × poro_j          # = 1.0e-18

# Volumina additiv mischen
v_solid_merged = v_solid1 + v_solid2  # = 1.6e-18
v_pore_merged = v_pore1 + v_pore2     # = 1.4e-18
v_dry_merged = v_solid_merged + v_pore_merged  # = 3.0e-18

# Neue Porosität
if v_pore_merged > 0:
    poro_merged = v_pore_merged / v_dry_merged
             = 1.4e-18 / 3.0e-18 = 0.467
```

---

### Schritt 5.6: V_flat aktualisieren (KRITISCH!)

```python
# V_flat[:dim] = V_solid (bleibt unverändert, war schon korrekt)
self.V_flat[:self.dim, new_idx] = V_solid_merged

# V_flat[-1] = V_dry (muss aktualisiert werden!)
self.V_flat[-1, new_idx] = V_dry_merged
```

**Vorher/Nachher:**
```
Vorher (temporär nach _append_particle_column):
  V_flat[:1, new_idx] = 1.6e-18  (V_solid)
  V_flat[-1, new_idx] = 1.6e-18  (falsch! War nur Summe der Komponenten)

Nachher (korrekt):
  V_flat[:1, new_idx] = 1.6e-18  (V_solid, unverändert)
  V_flat[-1, new_idx] = 3.0e-18  (V_dry, korrigiert!)
```

---

### Schritt 5.7: Flüssigkeit und Sättigung mergen

```python
# Porenvolumina der Eltern
V_pore_i = Vi_dry * poro_i if not np.isnan(poro_i) else 0.0
V_pore_j = Vj_dry * poro_j if not np.isnan(poro_j) else 0.0

# Interne Flüssigkeit: V_liq_int = V_pore × S
V_liq_int_i = V_pore_i * sat_i
V_liq_int_j = V_pore_j * sat_j

# Externe Flüssigkeit: V_liq_ext = V_liq_total - V_liq_int
V_liq_ext_i = liq_i_full - V_liq_int_i
V_liq_ext_j = liq_j_full - V_liq_int_j

# Addieren (direkte Addition, kein dW/W Scaling!)
if i == j:
    V_liq_int_merged = 2.0 * V_liq_int_i
    V_liq_ext_merged = 2.0 * V_liq_ext_i
else:
    V_liq_int_merged = V_liq_int_i + V_liq_int_j
    V_liq_ext_merged = V_liq_ext_i + V_liq_ext_j

# Neues Porenvolumen
V_pore_merged = V_dry_merged * poro_merged

# Neue Sättigung
if V_pore_merged > 0:
    sat_merged = V_liq_int_merged / V_pore_merged
else:
    sat_merged = 0.0  # Vollkörper

# Überschuss externalisieren (wenn S > 1)
if sat_merged > 1.0:
    excess = V_pore_merged * (sat_merged - 1.0)
    V_liq_int_merged = V_pore_merged  # Auf 100% cappen
    V_liq_ext_merged += excess
    sat_merged = 1.0

# Gesamtflüssigkeit
liq_merged = V_liq_int_merged + V_liq_ext_merged

# Speichern
self.saturation[new_idx] = sat_merged
self.liquid_volume[new_idx] = liq_merged
```

**Beispiel:**
```
Eltern i: V_pore_i = 0.4e-18, sat_i = 0.6, liq_i = 0.3e-18
  → V_liq_int_i = 0.4e-18 × 0.6 = 0.24e-18
  → V_liq_ext_i = 0.3e-18 - 0.24e-18 = 0.06e-18

Eltern j: V_pore_j = 1.0e-18, sat_j = 0.5, liq_j = 0.6e-18
  → V_liq_int_j = 1.0e-18 × 0.5 = 0.5e-18
  → V_liq_ext_j = 0.6e-18 - 0.5e-18 = 0.1e-18

Kind:
  V_liq_int_merged = 0.24e-18 + 0.5e-18 = 0.74e-18
  V_liq_ext_merged = 0.06e-18 + 0.1e-18 = 0.16e-18
  V_pore_merged = 3.0e-18 × 0.467 = 1.4e-18
  
  sat_merged = 0.74e-18 / 1.4e-18 = 0.529
  liq_merged = 0.74e-18 + 0.16e-18 = 0.9e-18
```

---

## PHASE 6: Eltern-Partikel reduzieren

### Schritt 6.1: Selbst-Agglomeration (i == j)

```python
if i == j:
    w_now = float(self.W[i])
    w_rem = w_now - 2.0 * dW
    
    if w_rem > 0.0:
        self.W[i] = w_rem
        # liquid_volume NICHT ändern! (intensive Eigenschaft)
    else:
        self._remove_particle_column(i)  # Swap-Pop entfernen
```

**Beispiel:**
```
Wi = 50, dW = 40
w_rem = 50 - 2×40 = 50 - 80 = -30 → Partikel entfernen!
```

---

### Schritt 6.2: Verschiedene Partikel (i ≠ j)

```python
for idx in sorted({int(i), int(j)}, reverse=True):
    w_now = float(self.W[idx])
    w_rem = w_now - dW
    
    if w_rem > 0.0:
        self.W[idx] = w_rem
    else:
        self._remove_particle_column(idx)
```

**Beispiel:**
```
Wi = 50, Wj = 40, dW = 40

Sortiert absteigend: [j=40, i=50] → [j, i]

1. idx = j:
   w_rem = 40 - 40 = 0 → entfernen!
   
2. idx = i:
   w_rem = 50 - 40 = 10 → W[i] = 10

Ergebnis: Partikel j entfernt, Partikel i reduziert auf W=10
```

---

### Schritt 6.3: Remove-Implementierung (Swap-Pop)

```python
def _remove_particle_column(self, idx):
    last = self.a_tot - 1
    
    if idx != last:
        # Letzte Spalte nach idx kopieren
        self.V_flat[:, idx] = self.V_flat[:, last]
        self.X[idx] = self.X[last]
        self.W[idx] = self.W[last]
        self.liquid_volume[idx] = self.liquid_volume[last]
        self.porosity[idx] = self.porosity[last]
        self.saturation[idx] = self.saturation[last]
    
    # Aktive Anzahl reduzieren
    self.a_tot -= 1
```

**Vorteil:** O(1) statt O(n) durch Vermeidung von Lücken.

---

## PHASE 7: Propensities neu berechnen

### Schritt 7.1: Vollständiger Rebuild

```python
self._rebuild_all_propensities()
```

**Warum komplett neu?** Jede Gewichtsänderung beeinflusst ALLE Propensities:
```
r[k] = Σⱼ W[j] × β(k,j)

Wenn sich W[i] oder W[j] ändert, ändert sich r[k] für ALLE k!
```

---

### Schritt 7.2: Fenwick Tree neu aufbauen

```python
self._agg_sampler = FenwickSampler(self._r_agg[:self.a_tot])
```

---

### Schritt 7.3: Breakage-Sampler aktualisieren (falls aktiv)

```python
if process_type in ("breakage", "mix"):
    self._calc_break_rates_full()
    self._break_sampler = FenwickSampler(self._break_rate[:self.a_tot])
```

---

## Zusammenfassung: Kompletter Datenfluss

```
┌──────────────────────────────────────────────────────────────────────┐
│                     _rebuild_all_propensities() (PREPARE)            │
│  1. delta[i] = min(dW_max, W[i])                                     │
│  2. r[i] = Σⱼ W[j] × β(i,j)  [O(n²) JIT]                             │
│  3. r_norm[i] = r[i] / delta[i]                                      │
│  4. Fenwick Tree aufbauen                                            │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│                      _do_one_agg() (EVENT)                           │
│                                                                      │
│  PHASE 1: Vorbereitung                                               │
│  ─────────────────────                                               │
│  • Arrays cachen (R, W, V0, V1)                                      │
│  • Parameter laden (alpha, SIZEEVAL, ...)                            │
│  • Zufallszahlen generieren (u_sel, u_acc)                           │
│                                                                      │
│  PHASE 2: Erstes Partikel                                            │
│  ────────────────────────────                                        │
│  • i = FenwickSampler.sample(rng)  [O(log n)]                        │
│    → Partikel mit P(i) ∝ r[i]                                        │
│                                                                      │
│  PHASE 3: Zweites Partikel                                           │
│  ────────────────────────────                                        │
│  • j = weighted_sample(W, u_sel)  [O(n)]                             │
│    → Partikel mit P(j) ∝ W[j]                                        │
│  • beta_ij = kernel.compute_beta(r1, r2, ...)                        │
│  • alpha_ij = component_mixing(...)                                  │
│  • beta_eff = beta_ij × alpha_ij                                     │
│  • pick_w = W[j] × beta_eff  (oder (W[i]-1) für i==j)                │
│  • SIZEEVAL-Test (optional)                                          │
│                                                                      │
│  PHASE 4: Paketgröße                                                 │
│  ────────────────────                                                │
│  • pair_prop = W[i] × pick_w                                         │
│  • dW = calc_agg_dW(mode, pair_prop, sum_prop)                       │
│  • dW = min(dW, Wi, Wj, delta_i, delta_j)                            │
│                                                                      │
│  PHASE 5: Kind erzeugen                                              │
│  ─────────────────────                                               │
│  • V_solid_i = V_dry_i × (1 - poro_i)                                │
│  • V_solid_j = V_dry_j × (1 - poro_j)                                │
│  • V_solid_merged = V_solid_i + V_solid_j  ← MASSE ERHALTEN!         │
│  • _append_particle_column(V_solid_merged)                           │
│  • W[new] = dW                                                       │
│  • V_dry_merged, poro_merged = kernel.compute_merged_porosity(...)   │
│  • V_flat[:dim, new] = V_solid_merged                                │
│  • V_flat[-1, new] = V_dry_merged  ← KORREKTUR!                      │
│  • liq_merged, sat_merged = merge_liquid(...)                        │
│                                                                      │
│  PHASE 6: Eltern reduzieren                                          │
│  ────────────────────────────                                        │
│  • if i==j: W[i] -= 2×dW                                             │
│  • else: W[i] -= dW, W[j] -= dW                                      │
│  • if W <= 0: _remove_particle_column()  [O(1) swap-pop]             │
│                                                                      │
│  PHASE 7: Aktualisierung                                             │
│  ─────────────────────                                               │
│  • _rebuild_all_propensities()  [O(n²)]                              │
│  • _agg_sampler = FenwickSampler(r_norm)                             │
│  • if breakage: _calc_break_rates_full()                             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Performance-Charakteristik

| Phase | Operation | Komplexität | Häufigkeit |
|-------|-----------|-------------|------------|
| 0 | `_rebuild_all_propensities` | O(n²) | Nach jedem Event |
| 2 | `_agg_sampler.sample` | O(log n) | Einmal pro Event |
| 3 | `_pick_partner_kernel` | O(n) | Einmal pro Event |
| 3.2 | `compute_beta` | O(1) | Einmal pro Partnerwahl |
| 5 | Kind erzeugen | O(1) | Einmal pro erfolgreichem Event |
| 6 | Eltern reduzieren | O(1) | Einmal pro erfolgreichem Event |
| 7 | `_rebuild_all_propensities` | O(n²) | Nach jedem Event |

**Engpass:** Phase 0 und 7 (O(n²)) dominieren bei großen Partikelanzahlen.

**Optimierung:** JIT-Compilation der inneren Schleife in `rebuild_r_array_*`.

---

## Numerische Stabilität

### Guard-Clauses gegen NaN/Inf:

```python
# Beta-Berechnung
if r1 <= 0 or r2 <= 0:
    return 0.0
if not np.isfinite(beta_ij) or beta_ij <= 0.0:
    return -1, 0.0

# Delta-Berechnung
delta = np.minimum(dW_const, W)
delta = np.maximum(delta, 0.0)

# Division mit Schutz
r_normalized = np.divide(r, delta, out=zeros, where=delta > 0.0)

# Porosität clampen
poro_merged = max(0.0, min(1.0, poro_merged))
```

---

## Physikalische Invarianten

### Massenerhaltung:
```python
V_solid_merged = V_solid_i + V_solid_j  # ✓ Erhalt
```

### Flüssigkeitserhaltung:
```python
liq_merged = liq_i + liq_j  # ✓ Erhalt (pro physikalischem Partikel)
```

### Gewichtserhaltung (DSMC):
```python
Σ W_after = Σ W_before  # ✓ Im Mittel erhalten
```

### Porositätsbereich:
```python
0 ≤ porosity ≤ 1  # ✓ Immer gültig
0 ≤ saturation ≤ 1  # ✓ Nach Externalisierung
```
