"""
Analyse: Ist Vc-Skalierung das Problem?

Hypothese: Die Test-Formel teilt DURCH Vc, aber Statistik trackt BEREITS korrekt.
"""

# Parameter aus Test
Vc = 1.0e-2  # m³
v_droplet = 5.236e-16  # m³
expected_liquid = 2.0e-11  # m³

# Aus Logfile:
n_droplets_actual = 3.81e+04  # Tatsächlich verteilt
stat_liquid = 1.992e-11  # Statistik (korrekt!)
system_liquid = 1.98e-9   # Test-Berechnung (falsch!)

print("=" * 80)
print("VC-SKALIERUNG ANALYSE")
print("=" * 80)

# Statistik berechnet:
# liquid_volume_added_total = Σ (v_droplet × effective_dW)
# where effective_dW = dW × (Vc_ref / Vc)

# Bei Vc_ref = Vc (kein Doubling):
# effective_dW = dW
# liquid_volume_added_total = Σ (v_droplet × dW)

# Das ist die GESAMTE physikalische Flüssigkeit! ✅
print(f"\nStatistik trackt: {stat_liquid:.6e} m³")
print(f"Erwartet:         {expected_liquid:.6e} m³")
print(f"Error:            {(stat_liquid-expected_liquid)/expected_liquid*100:+.4f}% ✅")

# Test-Berechnung:
# v_liquid_in_system = Σ (liquid_volume[k] × W[k] / Vc)

# Wenn liquid_volume[k] = v_droplet (pro Partikel), dann:
# Für EIN nucleated particle mit W=dW:
#   Beitrag = v_droplet × dW / Vc

# ABER: Statistik hat BEREITS v_droplet × dW gezählt!
# Test teilt nochmal durch Vc → FALSCH!

# Beispiel:
dW_typical = 50  # Typisches dW
n_nucleated = int(n_droplets_actual)  # Anzahl nucleated particles

# Jeder nucleated particle hat:
# liquid_volume = v_droplet
# W = dW_typical

# Test-Beitrag PER NUCLEATED PARTICLE:
contrib_per_particle = v_droplet * dW_typical / Vc

# Gesamter Test-Wert (wenn alle gleich wären):
test_total_approx = n_nucleated * contrib_per_particle

print(f"\n--- BEISPIELRECHNUNG ---")
print(f"Typisches dW: {dW_typical}")
print(f"Anzahl nucleated Partikel: ~{n_nucleated}")
print(f"\nBeitrag PER nucleated Partikel:")
print(f"  liquid_volume × W / Vc = {v_droplet:.6e} × {dW_typical} / {Vc:.6e}")
print(f"  = {contrib_per_particle:.6e} m³")

print(f"\nGesamt (alle Partikel):")
print(f"  {n_nucleated} × {contrib_per_particle:.6e} = {test_total_approx:.6e} m³")

# Vergleich mit tatsächlichem Test-Ergebnis
print(f"\nTatsächlicher Test-Wert: {system_liquid:.6e} m³")
print(f"Faktor zum Erwarteten:   {system_liquid/expected_liquid:.1f}×")

# Kritische Frage: Was ist ΣW für nucleated particles?
# Aus Logfile: ΣW für lv=1x,2x,3x = 390.6 + 1273.3 + 2296.4 = 3960.3
# PLUS "andere" mit unbekanntem ΣW

sum_W_known = 390.6 + 1273.3 + 2296.4
print(f"\nΣW (nucleated, bekannt): {sum_W_known:.1f}")

# Wenn ΣW_total ≈ 10000 (aus n_phys=1e6, Vc=0.01):
W_total = 1e6 * Vc  # = 10000
sum_W_other = W_total - sum_W_known

print(f"ΣW (total, geschätzt):   {W_total:.1f}")
print(f"ΣW ('andere'):           {sum_W_other:.1f}")

# Test-Formel: Σ (lv[k] × W[k] / Vc)
# = (1/Vc) × Σ (lv[k] × W[k])

# Für Partikel mit lv = n × v_droplet:
# Σ (lv × W) = v_droplet × Σ (n_i × W_i)

# Wo n_i = Anzahl Tropfen auf Partikel i

# Statistik: Σ (v_droplet × effective_dW) = v_droplet × Σ effective_dW
# = v_droplet × n_droplets_physical

# Test: (1/Vc) × v_droplet × Σ (n_i × W_i)

# Damit Test = Statistik:
# (1/Vc) × Σ (n_i × W_i) = Σ effective_dW

# ODER: Σ (n_i × W_i) = Vc × Σ effective_dW

# Das ist NUR wahr wenn n_i = 1 für alle! (Jeder Tropfen auf neuem Partikel)

print("\n" + "=" * 80)
print("SCHLUSSFOLGERUNG")
print("=" * 80)

# Wenn jeder Tropfen auf EINEM NEUEN Partikel landet:
# n_i = 1 für alle nucleated particles
# Σ (n_i × W_i) = Σ W_i = Total weight of nucleated particles

# Aber Statistik: Σ effective_dW = n_droplets_physical

# Für Konsistenz müsste gelten:
# Σ W_nucleated = Vc × n_droplets_physical

lhs = sum_W_known + sum_W_other  # ≈ W_total = 10000
rhs = Vc * n_droplets_actual

print(f"\nWenn jeder Tropfen auf neuem Partikel landet:")
print(f"  Σ W_nucleated = {lhs:.1f}")
print(f"  Vc × n_droplets = {Vc:.6e} × {n_droplets_actual:.2e} = {rhs:.1f}")

if abs(lhs - rhs) < rhs * 0.1:
    print(f"  → PASST! (Faktor {lhs/rhs:.2f})")
else:
    print(f"  → PASST NICHT! (Faktor {lhs/rhs:.2f})")
    print(f"\n⚠️  DISKREPANZ: ΣW ({lhs:.1f}) ≠ Vc×n_droplets ({rhs:.1f})")
    print(f"   Faktor: {lhs/rhs:.1f}×")
    
    # Das erklärt den Error!
    predicted_error = (lhs / rhs - 1) * 100
    print(f"\n→ Das erklärt den Test-Error von ~{predicted_error:+.0f}%!")
