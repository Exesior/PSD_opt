"""
CRITICAL ANALYSE: Wo ist der Vc-Skalierungs-Bug?

Ziel: Verstehen warum Test-Formel ×100 zu viel berechnet.
"""

print("=" * 80)
print("SKALIERUNGS-ANALYSE SCHRITT FÜR SCHRITT")
print("=" * 80)

# ============================================================================
# PHYSIKALISCHE GRÖSSEN
# ============================================================================
print("\n1. PHYSIKALISCHE PARAMETER:")
print("-" * 50)

Vc = 1.0e-2  # m³ (Control Volume)
v_droplet = 5.236e-16  # m³ (Volumen EINES physikalischen Tropfens)
Q_vol = 1e-11  # m³/s (Volumenstrom)
duration = 2.0  # s

expected_total_liquid = Q_vol * duration  # = 2e-11 m³
print(f"Vc = {Vc:.6e} m³")
print(f"v_droplet = {v_droplet:.6e} m³")
print(f"Q_vol = {Q_vol:.6e} m³/s")
print(f"Erwartete Gesamtflüssigkeit: {expected_total_liquid:.6e} m³")

# ============================================================================
# NUCLEATION RATE BERECHNUNG
# ============================================================================
print("\n2. NUCLEATION RATE:")
print("-" * 50)

# Aus Code: rate = Q_vol / Vc
rate = Q_vol / Vc  # = 1e-9 1/s
print(f"rate = Q_vol / Vc = {Q_vol:.6e} / {Vc:.6e} = {rate:.6e} 1/s")

# Pro Zeitschritt dt:
dt = 0.1  # s
n_droplets_per_step = rate * dt * Vc  # = Q_vol * dt
print(f"\nPro Zeitschritt dt={dt}s:")
print(f"  n_droplets = rate × dt × Vc = {rate:.6e} × {dt} × {Vc:.6e} = {n_droplets_per_step:.2f}")
print(f"  ODER: Q_vol × dt = {Q_vol:.6e} × {dt} = {n_droplets_per_step:.2f}")
print(f"  → Vc kürzt sich raus! ✅")

# ============================================================================
# EINZELES NUCLEATION EVENT
# ============================================================================
print("\n3. EINZELNES NUCLEATION EVENT:")
print("-" * 50)

# Typisches dW aus Code: dW = W_i / 2, mit W_i = 100
W_i = 100
dW = W_i / 2  # = 50
print(f"W_i (Parent weight) = {W_i}")
print(f"dW = W_i / 2 = {dW}")

# vc_scale = Vc_ref / Vc = 1 (am Anfang kein Doubling)
vc_scale = 1.0
effective_dW = dW * vc_scale  # = 50
print(f"vc_scale = Vc_ref / Vc = {vc_scale}")
print(f"effective_dW = dW × vc_scale = {effective_dW}")

# Physikalische Bedeutung:
print(f"\nPhysikalische Bedeutung:")
print(f"  Das Event repräsentiert {effective_dW} PHYSIKALISCHE Tropfen")
print(f"  Gesamtflüssigkeit dieses Events: {effective_dW} × {v_droplet:.6e} = {effective_dW * v_droplet:.6e} m³")

# ============================================================================
# WAS PASSIERT IM CODE?
# ============================================================================
print("\n4. CODE-VERHALTEN:")
print("-" * 50)

# Statistik (in _distribute_liquid_at_time):
v_event_stat = v_droplet * effective_dW
print(f"Statistik trackt: v_event = v_droplet × effective_dW")
print(f"  = {v_droplet:.6e} × {effective_dW} = {v_event_stat:.6e} m³ ✅")

# Liquid Volume Zuweisung (in _create_nucleated_particle_copy):
liquid_assigned = v_droplet  # NICHT × effective_dW!
print(f"\nLiquid Volume zugewiesen: liquid_volume = v_droplet")
print(f"  = {liquid_assigned:.6e} m³")
print(f"  → ABER: Das Partikel hat W={dW} repräsentiert {dW} physikalische Partikel!")

# ============================================================================
# TEST-FORMEL
# ============================================================================
print("\n5. TEST-FORMEL:")
print("-" * 50)

# Formel: Σ liquid_volume[k] × W[k] / Vc
# Für EIN nucleated Partikel:
W_new = dW  # = 50
test_contrib = liquid_assigned * W_new / Vc
print(f"Beitrag EINES nucleated Partikels:")
print(f"  liquid_volume × W / Vc = {liquid_assigned:.6e} × {W_new} / {Vc:.6e}")
print(f"  = {test_contrib:.6e} m³")

# Vergleich mit tatsächlicher Flüssigkeit:
actual_liq_this_particle = v_droplet * effective_dW  # = 50 × v_droplet
print(f"\nTatsächliche Flüssigkeit dieses Ensembles:")
print(f"  {effective_dW} Partikel × {v_droplet:.6e} = {actual_liq_this_particle:.6e} m³")

# Faktor:
factor = test_contrib / actual_liq_this_particle
print(f"\nFAKTOR: Test / Tatsächlich = {factor:.1f}× ❌")

# ============================================================================
# PROBLEM IDENTIFIZIERT!
# ============================================================================
print("\n" + "=" * 80)
print("PROBLEM GEFUNDEN!")
print("=" * 80)

print(f"""
Die Test-Formel berechnet:
  liquid_volume × W / Vc = {liquid_assigned:.6e} × {W_new} / {Vc:.6e}
                        = {test_contrib:.6e} m³

Aber die tatsächliche Flüssigkeit ist:
  effective_dW × v_droplet = {effective_dW} × {v_droplet:.6e}
                          = {actual_liq_this_particle:.6e} m³

Der Faktor ist: {factor:.1f}× = 1/Vc = {1/Vc:.0f}

→ Die Test-Formel teilt FÄLSCHLICHERWEISE durch Vc!
→ Richtig wäre: liquid_volume × W (OHNE /Vc)

ODER ALTERNATIV:
→ liquid_volume MUSS mit effective_dW skaliert werden beim Setzen!
""")

# ============================================================================
# FIX-OPTIONEN
# ============================================================================
print("=" * 80)
print("FIX-OPTIONEN")
print("=" * 80)

print("""
OPTION 1: Test-Formel korrigieren (empfohlen für intensive Semantik)
-----------------------------------------------------------------------
ALT: v_liquid += solver.liquid_volume[k] * solver.W[k] / solver.Vc
NEU: v_liquid += solver.liquid_volume[k] * solver.W[k]

Begründung: liquid_volume ist PRO PARTIKEL, W ist ANZAHL Partikel.
            Produkt ist GESAMTE Flüssigkeit des Ensembles.
            KEINE Division durch Vc nötig!

OPTION 2: liquid_volume Zuweisung skalieren
-----------------------------------------------------------------------
ALT: solver.liquid_volume[new_idx] = v_droplet
NEU: solver.liquid_volume[new_idx] = v_droplet * effective_dW

Dann wäre liquid_volume EXTENSIV (Gesamtflüssigkeit des Ensembles).
Test-Formel mit /Vc wäre dann korrekt.

EMPFEHLUNG: OPTION 1 (beibehalten intensive Semantik)
""")

# ============================================================================
# VERIFIKATION
# ============================================================================
print("=" * 80)
print("VERIFIKATION MIT OPTON 1")
print("=" * 80)

# Mit korrigierter Formel (OHNE /Vc):
corrected_contrib = liquid_assigned * W_new
print(f"Korrigierter Beitrag EINES Partikels:")
print(f"  liquid_volume × W = {liquid_assigned:.6e} × {W_new}")
print(f"  = {corrected_contrib:.6e} m³")

print(f"\nVergleich mit tatsächlicher Flüssigkeit:")
print(f"  expected = {actual_liq_this_particle:.6e} m³")
print(f"  computed = {corrected_contrib:.6e} m³")
print(f"  Faktor: {corrected_contrib/actual_liq_this_particle:.4f} ✅")

# Hochskalieren auf alle Tropfen:
n_total_droplets = expected_total_liquid / v_droplet  # ≈ 38200
total_W_nucleated = n_total_droplets * dW  # Alle nucleated Partikel zusammen

total_liquid_corrected = n_total_droplets * corrected_contrib
print(f"\nHOCHSKALIERT AUF ALLE TROPFEN:")
print(f"  Anzahl Tropfen: ~{n_total_droplets:.0f}")
print(f"  Gesamtflüssigkeit (korrekt): {expected_total_liquid:.6e} m³")
print(f"  Gesamtflüssigkeit (Formel OHNE /Vc): {total_liquid_corrected:.6e} m³")
print(f"  Faktor: {total_liquid_corrected/expected_total_liquid:.4f} ✅")
