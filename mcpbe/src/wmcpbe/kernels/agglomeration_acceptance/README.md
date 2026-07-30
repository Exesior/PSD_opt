# Agglomeration Acceptance Kernels

## Übersicht

Diese Kernel bestimmen, ob eine Kollision zwischen zwei Partikeln zur Agglomeration führt oder ob die Partikel abprallen. Sie werden **NACH** der Partnerauswahl und **VOR** der Agglomerations-Durchführung aufgerufen.

## Verfügbare Kernel

| Kernel | Beschreibung | Parameter |
|--------|--------------|-----------|
| **stokes_krit** | Stokes-Kriterium nach Braumann et al. (2007) | `U_coll`, `binder_viscosity`, `rho_solid`, `rho_liquid`, `h_a` |
| **fittable** | Einfacher Fit-Parameter (probabilistisch) | `u_acc` |

## Verwendung

### Im Solver initialisieren

```python
from wmcpbe import MCPBESolver

solver = MCPBESolver(
    dim=1,
    t_vec=np.linspace(0, 4, 41),
    agg_kernel_name='constant',
    agg_kernel_params={'corr_beta': 1e-5},
    # NEU: Agglomeration Acceptance Kernel
    agg_acceptance_kernel_name='stokes_krit',
    agg_acceptance_kernel_params={
        'U_coll': 1.0,              # [m/s] Kollisionsgeschwindigkeit
        'binder_viscosity': 0.1,    # [Pa·s] Binder-Viskosität
        'rho_solid': 2500.0,        # [kg/m³] Feststoffdichte
        'rho_liquid': 1000.0,       # [kg/m³] Flüssigkeitsdichte
        'h_a': 500e-9,              # [m] Oberflächenrauheit
    },
)
```

### Ohne Acceptance-Kernel (Default)

Wenn kein `agg_acceptance_kernel_name` angegeben wird, werden **alle** Kollisionen akzeptiert (rückwärtskompatibel):

```python
solver = MCPBESolver(
    agg_kernel_name='shear_chin1998',
    # Kein agg_acceptance_kernel_name → immer akzeptieren
)
```

---

## Kernel: stokes_krit

### Physikalisches Modell

Implementiert das Stokes-Kriterium aus Braumann et al. (2007) für nasse Granulation:

- **St < St_crit**: Partikel agglomerieren (viskose Kräfte dominieren)
- **St > St_crit**: Partikel prallen ab (Trägheit dominiert)

### Formeln

**Stokes-Zahl (Gl. 8):**
```
St = (m_harm * U) / (3 * π * η * R_harm²)
```

**Kritische Stokes-Zahl (Gl. 10):**
```
St_crit = (1 + 1/e_coag) * ln(h / h_a)
```

**wobei:**
- `m_harm`: Harmonisches Mittel der Partikelmassen
- `R_harm`: Harmonisches Mittel der Partikelradien
- `U`: Kollisionsgeschwindigkeit [m/s]
- `η`: Binder-Viskosität [Pa·s]
- `e_coag`: Effektiver Restitutionskoeffizient = √(e_i × e_j)
- `e_i`: m_solid_i / m_total_i (da e_solid=1, e_liquid=0)
- `h`: Flüssigkeitsfilmdicke (aus externer Flüssigkeit)
- `h_a`: Minimale Filmdicke / Oberflächenrauheit [m]

### Parameter

| Parameter | Default | Einheit | Beschreibung |
|-----------|---------|---------|--------------|
| `U_coll` | 1.0 | m/s | Relative Kollisionsgeschwindigkeit |
| `binder_viscosity` | 0.1 | Pa·s | Viskosität des Binders |
| `rho_solid` | 2500.0 | kg/m³ | Feststoffdichte |
| `rho_liquid` | 1000.0 | kg/m³ | Flüssigkeitsdichte |
| `h_a` | 500e-9 | m | Minimale Filmdicke (Oberflächenrauheit) |

### Hinweise

- Benötigt Zugriff auf `solver.liquid_volume` für externe Flüssigkeit
- Benötigt Zugriff auf `solver.porosity` für Feststoffvolumen-Berechnung
- Bei fehlendem Solver-Zugriff: lehnt ab (False)
- Bei `h ≤ h_a`: lehnt ab (keine Agglomeration ohne Flüssigkeitsbrücke)

---

## Kernel: fittable

### Physikalisches Modell

Einfaches probabilistisches Modell ohne explizite Physik:

- Zufallszahl `u ~ Uniform(0, 1)` ziehen
- Akzeptieren wenn `u < u_acc`
- Ablehnen sonst

Effektive Agglomerationsrate: `β_eff = β × u_acc`

### Parameter

| Parameter | Default | Bereich | Beschreibung |
|-----------|---------|---------|--------------|
| `u_acc` | 1.0 | [0, 1] | Akzeptanzwahrscheinlichkeit |

### Verwendung

```python
solver = MCPBESolver(
    agg_acceptance_kernel_name='fittable',
    agg_acceptance_kernel_params={
        'u_acc': 0.8,  # 80% der Kollisionen führen zu Agglomeration
    },
)
```

### Hinweise

- Nützlich für Kalibrierung an experimentelle Daten
- Alle physikalischen Parameter werden ignoriert
- Verwendet `solver._rng` für reproduzierbare Zufallszahlen

---

## Eigene Kernel implementieren

Siehe `blueprint.py` für eine vollständige Vorlage.

Kurzfassung:

```python
from wmcpbe.kernels.base import AggAcceptanceKernel

class MyCustomKernel(AggAcceptanceKernel):
    
    @property
    def name(self) -> str:
        return 'my_custom_kernel'
    
    def get_default_params(self) -> dict:
        return {
            'param1': 1.0,
            'param2': 0.5,
        }
    
    def accept_collision(self, r1, r2, v_dry1, v_dry2, 
                         particle1_idx, particle2_idx, solver) -> bool:
        # Deine Physik hier
        return True  # oder False
```

Dann in `__init__.py` registrieren:

```python
from .my_custom_kernel import MyCustomKernel
AGG_ACCEPTANCE_KERNELS['my_custom_kernel'] = MyCustomKernel
```

---

## Performance-Optimierungen

Die Implementierung verwendet folgende Optimierungen:

1. **Konstanten vorberechnet**: `C_H = (3/(4π))^(1/3)` als Klassenattribut
2. **Frühe Abbrüche**: Bei `h ≤ h_a` sofort False (kein teurer Log-Aufruf)
3. **Harmonische Mittel effizient**: Direkte Berechnung ohne Overhead
4. **Solver-Zugriff minimiert**: Nur notwendige Attribute lesen

---

## Referenz

Braumann, A., Kraft, M., & Wagner, L. (2007). Modelling and validation of granulation with heterogeneous binder dispersion and chemical reaction. *Chemical Engineering Science*, 62(17), 4709-4720.
