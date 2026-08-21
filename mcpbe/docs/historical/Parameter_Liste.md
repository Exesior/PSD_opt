# Vollständige Parameterliste für test_powerlaw_rumpf_full.py

Diese Liste enthält ALLE Parameter, die im Test `test_powerlaw_rumpf_full.py` verwendet werden, 
inklusive hardgecodeter Werte aus den Kernel-Implementierungen.

================================================================================
## 1. TEST-KONFIGURATION (TestConfig Klasse)
================================================================================

### 1.1 Zeitliche Parameter
| Parameter 		| Einheit 	| Beschreibung 				| Typischer Wertebereich |
|-----------		|---------	|--------------				|------------------------|
| SEED      		| -       	| Random Seed für Reproduzierbarkeit 	| Integer ≥ 0 |
| T_TOTAL   		| s 		| Gesamte Simulationsdauer 		| > 0 |
| T_WRITE   		| s 		| Ausgabeintervall für Results 		|  0< ≤ T_TOTAL |
| NUCLEATION_DURATION 	| s 		| Dauer der Flüssigkeitszugabe 		|  0< ≤ T_TOTAL |

### 1.2 Partikeleigenschaften (Initial)
| Parameter 		| Einheit 	| Beschreibung 			| Typischer Wertebereich |
|-----------		|---------	|--------------			|------------------------|
| PARTICLE_DIAMETER 	| m 		| Anfangspartikeldurchmesser 	| 10e-6 bis 2000e-6 |
| PARTICLE_DENSITY 	| kg/m³ 	| Dichte des Feststoffmaterials | 1000 bis 5000 |
| INITIAL_POROSITY 	| - 		| Initiale Porosität 		| 0.0 (Vollkörper) bis < 1.0 |

### 1.3 Tropfeneigenschaften
| Parameter 		| Einheit 	| Beschreibung 		| Typischer Wertebereich |
|-----------		|---------	|--------------		|------------------------|
| DROPLET_DIAMETER 	| m 		| Tropfendurchmesser 	| 10e-6 bis 500e-6 |
| DROPLET_DENSITY 	| kg/m³ 	| Dichte der Flüssigkeit| 800 bis 1500 |

### 1.4 Prozessparameter
| Parameter 		| Einheit	| Beschreibung 					| Typischer Wertebereich |
|-----------		|-------	|--------------					|------------------------|
| VOLUMETRIC_FLOW_RATE 	| m³/s 		| Volumetrische Flussrate der Flüssigkeit 	| > 0 |
| AGG_COEFFICIENT 	| m³/s 		| Constant Kernel Koeffizient für Agglomeration | ≥ 0 |
| BATCH_SIZE 		| - 		| Anzahl identisch verteilter Tropfen pro Event | Integer ≥ 1 |

### 1.5 Breakage-Parameter (PowerLaw-Rumpf)
| Parameter 		| Einheit 	| Beschreibung 				| Typischer Wertebereich |
|-----------		|---------	|--------------				|------------------------|
| BREAKAGE_ENABLED 	| - 		| Breakage aktiviert 			| True/False |
| PL_P1 		| 1/s·Pa·m⁻³ᵅ 	| Pre-factor der Breakage-Rate 		| ≥ 0 |
| PL_P2 		| - 		| Shear-Exponent 			| ≥ 0 |
| G 			| 1/s 		| Scherrate 				| ≥ 0 |
| BREAKRVAL 		| - 		| Breakage-Modell-Variante 		| 1, 2, 3, 4, 5 |
| PL_V 			| - 		| Volumen-Exponent für breakrval=4 	| ≥ 0 |
| RUMPF_K		| - 		| Fitting-Parameter trockenes Regime 	| 2.2 bis 2.8 |
| RUMPF_ALPHA 		| - 		| Fitting-Parameter nasses Regime 	| 1.0 bis 1.33 |
| RUMPF_GAMMA 		| N/m 		| Oberflächenspannung 			| > 0 |
| RUMPF_DELTA 		| rad 		| Kontaktwinkel 			| 0 bis π/2 |

### 1.6 Compression-Parameter
| Parameter 		| Einheit 	| Beschreibung 				| Typischer Wertebereich |
|-----------		|---------	|--------------				|------------------------|
| COMPRESSION_ENABLED 	| - 		| Kompression aktiviert 		| True/False |
| COMPRESSION_RATE 	| 1/s 		| Porositäts-Zerfallsrate 		| ≥ 0 |
| MIN_POROSITY 		| - 		| Minimal erreichbare Porosität 	| 0 bis 1 |

### 1.7 Liquid Internalization-Parameter
| Parameter 		| Einheit 	| Beschreibung 					| Typischer Wertebereich |
|-----------		|---------	|--------------					|------------------------|
| LIQ_INTERN_ENABLED 	| - 		| Kontinuierliche Internalisierung aktiviert 	| True/False |
| LIQ_INTERN_RATE 	| 1/(m³·s) 	| Rate-Konstante für Internalisierung 		| > 0 |
| LIQ_INTERN_AGG_ENABLED| - 		| Internalisierung bei Agglomeration aktiviert 	| True/False |

### 1.8 Agglomeration Acceptance (Stokes-Kriterium)
| Parameter 		| Einheit 	| Beschreibung 				| Typischer Wertebereich |
|-----------		|---------	|--------------				|------------------------|
| STOKES_ENABLED 	| - 		| Stokes-Kriterium aktiviert 		| True/False |
| BINDER_VISCOSITY 	| Pa·s 		| Viskosität des Binders 		| > 0 |
| COLLISION_VELOCITY 	| m/s 		| Kollisionsgeschwindigkeit 		| > 0 |
| H_A 			| m 		| Minimale Filmdicke/Oberflächenrauheit | > 0 |

### 1.9 Numerische Einstellungen
| Parameter 		| Einheit 	| Beschreibung 					| Typischer Wertebereich |
|-----------		|---------	|--------------					|------------------------|
| INITIAL_PARTICLES 	| - 		| Anzahl initialer Computational Particles 	| Integer ≥ 1 |
| INITIAL_WEIGHT 	| - 		| Computational Weight pro Partikel 		| > 0 |
| CONTROL_VOLUME 	| m³ 		| Control Volume der DSMC-Simulation 		| > 0 |

================================================================================
## 2. HARDGE CODETE PARAMETER IN KERNELN
================================================================================

### 2.1 Stokes-Krit Kernel (stokes_krit.py)
| Parameter 		| Einheit 	| Beschreibung 				 | Wertebereich / Formel |
|-----------		|---------	|--------------				 |----------------------|
| E_SOLID 		| - 		| Restitutionskoeffizient für Feststoff  | 1.0 (fest) |
| E_LIQUID		| - 		| Restitutionskoeffizient für Flüssigkeit| 0.0 (fest) |
| C_H 			| - 		| Geometrische Konstante für Filmdicke 	 | (3/(4π))^(1/3) ≈ 0.62 |
| THREE_PI 		| - 		| Vorfaktor in Stokes-Zahl 		 | 3π (fest) |

**Berechnung von e_coag (hardgecodete Logik):**
```python
# Aus stokes_krit.py:
# e_i: m_solid_i / m_total_i (since e_solid=1, e_liquid=0)
e_i = m_solid_i / m_total_i  # Da e_solid=1, e_liquid=0
e_coag = sqrt(e_i * e_j)     # Geometrisches Mittel
```

### 2.2 Sättigungs-Regime-Grenzen (powerlaw_rumpf.py)
| Parameter 		| Beschreibung 				| Wert |
|-----------		|--------------				|------|
| SAT_DRY_THRESHOLD 	| S < diesem Wert: trockenes Regime 	| 0.3 (fest) |
| SAT_WET_THRESHOLD 	| S > diesem Wert: nasses Regime 	| 0.8 (fest) |
| SAT_TRANSITION_WIDTH 	| Breite des Übergangsregimes 		| 0.5 (fest) |

### 2.3 Cone Model Kernel (cone_model.py)
| Parameter 		| Einheit 	| Beschreibung 				| Wertebereich |
|-----------		|---------	|--------------				|--------------|
| K_AGG_DEFAULT 	| - 		| Shape correction für Agglomeration 	| ≥ 0 (Default: 1.0) |
| K_BREAK_DEFAULT 	| - 		| Shape correction für Breakage 	| ≥ 0 (Default: 1.0) |

### 2.4 Exponential Decay Kernel (exponential_decay.py)
| Parameter | Einheit | Beschreibung | Wertebereich |
|-----------|---------|--------------|--------------|
| RATE_DEFAULT | 1/s | Default Zerfallsrate | ≥ 0 (Default: 0.02) |
| MIN_POROSITY_DEFAULT | - | Default minimale Porosität | 0 bis 1 (Default: 0.3) |

### 2.5 Constant Kernel (constant.py)
| Parameter | Einheit | Beschreibung | Wertebereich |
|-----------|---------|--------------|--------------|
| CORR_BETA_DEFAULT | m³/s | Default Kollisionsfrequenz | ≥ 0 (Default: 1e-10) |

### 2.6 PowerLaw-Rumpf Kernel (powerlaw_rumpf.py)
| Parameter | Einheit | Beschreibung | Wertebereich |
|-----------|---------|--------------|--------------|
| P1_DEFAULT | 1/s·m⁻³ᵅ | Default Pre-factor | ≥ 0 (Default: 3e-2) |
| P2_DEFAULT | - | Default Shear-Exponent | ≥ 0 (Default: 1.0) |
| G_DEFAULT | 1/s | Default Scherrate | ≥ 0 (Default: 1000) |
| BREAKRVAL_DEFAULT | - | Default Breakage-Modell | 1, 2, 3, 4, 5 (Default: 4) |
| PL_V_DEFAULT | - | Default Volumen-Exponent | ≥ 0 (Default: 2.0) |
| K_DEFAULT | - | Default Rumpf k (dry) | 2.2 bis 2.8 (Default: 2.5) |
| ALPHA_DEFAULT | - | Default Rumpf α (wet) | 1.0 bis 1.33 (Default: 1.15) |
| GAMMA_DEFAULT | N/m | Default Oberflächenspannung | > 0 (Default: 0.072, Wasser) |
| DELTA_DEFAULT | rad | Default Kontaktwinkel | 0 bis π/2 (Default: 0.0) |

================================================================================
## 3. SOLVER-KONFIGURATION
================================================================================

### 3.1 Solver-Initialisierungsparameter
| Parameter | Beschreibung | Typische Werte |
|-----------|--------------|----------------|
| dim | Anzahl Komponenten (Dimensionalität) | 1 (1D), 2 (2D) |
| verbose | Ausführliche Ausgabe | True/False |
| load_attr | Attribute laden | True/False |
| init | Initialisierung durchführen | True/False |
| process_type | Prozess-Typ | "agg", "break", "mix" |
| recon_enable | Rekonstruktion aktivieren | True/False |
| mcpbe_debug | Debug-Modus | True/False |
| agg_propensity_mode | Propensity-Berechnungsmodus | "moment" (O(n)), "pairwise" (O(n²)) |

### 3.2 Kernel-Typen und ihre Parameter

**Aggregation Kernel:**
| Kernel-Name | Beschreibung | Wichtige Parameter |
|-------------|--------------|-------------------|
| constant | Größenunabhängige Kollisionsfrequenz | corr_beta |
| shear_chin1998 | Scherungsgetrieben | corr_beta, g |
| brownian_tsouris1995 | Brownsche Bewegung | Temperatur, Viskosität |
| sum | Summenkernel (r1 + r2) | corr_beta |
| liquid_bridge | Flüssigkeitsbrücken-modelliert | Mehrere Parameter |

**Agglomeration Acceptance Kernel:**
| Kernel-Name | Beschreibung | Wichtige Parameter |
|-------------|--------------|-------------------|
| stokes_krit | Stokes-Kriterium (Braumann et al. 2007) | U_coll, binder_viscosity, rho_solid, rho_liquid, h_a |

**Breakage Kernel:**
| Kernel-Name | Beschreibung | Wichtige Parameter |
|-------------|--------------|-------------------|
| powerlaw_rumpf | PowerLaw mit Rumpf-Strength-Modell | p1, p2, g, breakrval, pl_v, k, alpha, gamma, delta |
| power_law | Einfaches Potenzgesetz | p1, pl_v |
| mlp_model | MLP-neuronales Netz | Modell-Pfade |

**Porosity Growth Kernel:**
| Kernel-Name | Beschreibung | Wichtige Parameter |
|-------------|--------------|-------------------|
| cone_model | Kegelstumpf-Geometrie bei Kontakt | k_agg, k_break |
| volume_mixing | Einfache Volumenmittelung | - |
| incomplete_mixing | Unvollständige Mischung | Mischfaktor |

**Compression Kernel:**
| Kernel-Name | Beschreibung | Wichtige Parameter |
|-------------|--------------|-------------------|
| exponential_decay | Exponentieller Zerfall | rate, min_porosity |

**Liquid Internalization Kernel (continuous):**
| Kernel-Name | Beschreibung | Wichtige Parameter |
|-------------|--------------|-------------------|
| liquid_internalization | Braumann et al. 2007 | k_intern |

**Liquid Internalization during Agglomeration:**
| Kernel-Name | Beschreibung | Wichtige Parameter |
|-------------|--------------|-------------------|
| braumann_2007 | Kontaktporen-Einfang | - |

### 3.3 Nucleation Handler Parameter
| Parameter | Einheit | Beschreibung | Wertebereich |
|-----------|---------|--------------|--------------|
| enabled | - | Aktiviert | True/False |
| volumetric_flow_rate | m³/s | Flussrate | > 0 |
| droplet_diameter | m | Tropfengröße | > 0 |
| liquid_addition_start | s | Startzeit | ≥ 0 |
| liquid_addition_duration | s | Dauer | > 0 |
| batch_size | - | Tropfen pro Event | Integer ≥ 1 |

### 3.4 Compression Handler Parameter
| Parameter | Einheit | Beschreibung | Wertebereich |
|-----------|---------|--------------|--------------|
| enabled | - | Aktiviert | True/False |
| rate | 1/s | Kompressionsrate | ≥ 0 |
| min_porosity | - | Minimale Porosität | 0 bis 1 |

================================================================================
## 4. ABGELEITETE GRÖßEN (berechnet im Test)
================================================================================

### 4.1 Partikelgrößen (abgeleitet)
| Größe | Formel | Einheit | Beschreibung |
|-------|--------|---------|--------------|
| particle_radius | PARTICLE_DIAMETER / 2 | m | Partikelradius |
| particle_volume | (4/3) × π × r³ | m³ | Partikelvolumen (sphärisch) |

### 4.2 Tropfengrößen (abgeleitet)
| Größe | Formel | Einheit | Beschreibung |
|-------|--------|---------|--------------|
| droplet_radius | DROPLET_DIAMETER / 2 | m | Tropfenradius |
| droplet_volume | (4/3) × π × r³ | m³ | Tropfenvolumen (sphärisch) |

### 4.3 Erwartete Ergebnisse (abgeleitet)
| Größe | Formel | Einheit | Beschreibung |
|-------|--------|---------|--------------|
| expected_liquid | flow_rate × duration | m³ | Erwartetes Flüssigkeitsvolumen |
| expected_droplets | expected_liquid / droplet_volume | - | Erwartete Tropfenanzahl |

### 4.4 Breakage-Exponenten (abgeleitet)
| Größe | Formel | Beschreibung |
|-------|--------|--------------|
| alpha (Volumen) | pl_v / 3 | Exponent für V^alpha bei breakrval=4 |

================================================================================
## 5. PHYSIKALISCHE KONSTANTEN (implizit verwendet)
================================================================================

| Konstante | Symbol | Wert | Verwendung |
|-----------|--------|------|------------|
| Pi | π | math.pi ≈ 3.14159 | Volumenberechnung Kugel |
| Euler-Zahl | e | math.e ≈ 2.71828 | Exponentialzerfall |

================================================================================
## 6. ZUSAMMENFASSUNG DER FESTEN WERTE (HARDCODED)
================================================================================

**In stokes_krit.py (nicht konfigurierbar):**
| Konstante | Wert | Beschreibung |
|-----------|------|--------------|
| E_SOLID | 1.0 | Restitution Feststoff |
| E_LIQUID | 0.0 | Restitution Flüssigkeit |
| C_H | (3/(4π))^(1/3) | Geometriekonstante Filmberechnung |
| THREE_PI | 3π | Vorfaktor Stokes-Zahl |

**In powerlaw_rumpf.py (nicht konfigurierbar):**
| Konstante | Wert | Beschreibung |
|-----------|------|--------------|
| SAT_DRY_THRESHOLD | 0.3 | Grenze trocken/transitional |
| SAT_WET_THRESHOLD | 0.8 | Grenze transitional/nass |
| SAT_TRANSITION_WIDTH | 0.5 | Breite Übergangsregime |

**Kernel-Defaults (überschreibbar):**
| Kernel | Parameter | Default |
|--------|-----------|---------|
| cone_model | k_agg, k_break | 1.0 |
| exponential_decay | rate, min_porosity | 0.02 1/s, 0.3 |
| constant | corr_beta | 1e-10 m³/s |
| powerlaw_rumpf | k, alpha, gamma, delta | 2.5, 1.15, 0.072 N/m, 0.0 rad |

================================================================================
Ende der Parameterliste
================================================================================
