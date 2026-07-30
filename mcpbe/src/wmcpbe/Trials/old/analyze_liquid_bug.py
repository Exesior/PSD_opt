"""
Analyse: Woher kommt der Faktor 100?

Berechnet detailliert die Flüssigkeitsbilanz aus den Log-Daten.
"""

# Daten aus dem Logfile
Vc = 1.0e-2  # m³
v_droplet = 5.236e-16  # m³ (10 µm Durchmesser)

expected_liquid = 2.0e-11  # m³
actual_in_system = 1.98e-9  # m³ (aus Test)

# Verteilung aus Logfile:
dist = {
    'lv=0':     {'n': 114,    'sum_W': 0},
    'lv=1x':    {'n': 822,    'sum_W': 390.6},
    'lv=2x':    {'n': 2786,   'sum_W': 1273.3},
    'lv=3x':    {'n': 5388,   'sum_W': 2296.4},
    'andere':   {'n': 17304,  'sum_W': None},  # Unbekannt!
}

# Gesamt-W berechnen (aus Console: n_phys=9.87e+05, Vc=0.01)
n_phys_total = 9.87e+05
W_total = n_phys_total * Vc  # = 9870

print("=" * 80)
print("FLÜSSIGKEITS-BILANZ ANALYSE")
print("=" * 80)

# Bekannte Kategorien
known_W = sum(d['sum_W'] for d in dist.values() if d['sum_W'] is not None)
print(f"\nΣW (bekannte Kategorien): {known_W:.1f}")
print(f"ΣW (total erwartet): {W_total:.1f}")

# ΣW für "andere" schätzen
other_W = W_total - known_W
dist['andere']['sum_W'] = other_W
print(f"ΣW (andere, geschätzt): {other_W:.1f}")

# Beitrag jeder Kategorie zur Gesamtflüssigkeit
print("\n" + "-" * 80)
print("BEITRAG ZUR GESAMTFLÜSSIGKEIT:")
print("-" * 80)

total_liq = 0.0
for key, data in dist.items():
    if key == 'lv=0':
        contrib = 0.0
        lv_factor = 0
    elif key == 'lv=1x':
        lv_factor = 1
        contrib = data['sum_W'] * lv_factor * v_droplet / Vc
    elif key == 'lv=2x':
        lv_factor = 2
        contrib = data['sum_W'] * lv_factor * v_droplet / Vc
    elif key == 'lv=3x':
        lv_factor = 3
        contrib = data['sum_W'] * lv_factor * v_droplet / Vc
    elif key == 'andere':
        # Durchschnittlichen lv_factor schätzen
        n_other = data['n']
        w_other = data['sum_W']
        w_avg_other = w_other / n_other if n_other > 0 else 0
        
        # Um total_liq zu erreichen, muss lv_factor sein:
        # other_W * lv_factor * v_d / Vc = remaining
        remaining = actual_in_system - total_liq
        if w_other > 0:
            required_lv_factor = remaining * Vc / (w_other * v_droplet)
        else:
            required_lv_factor = 0
        
        contrib = remaining
        lv_factor = required_lv_factor
    
    print(f"{key:10s}: n={data['n']:5d}, ΣW={data['sum_W']:8.1f}, " + 
          f"W_avg={data['sum_W']/data['n'] if data['n']>0 else 0:5.1f}, " +
          f"lv_factor={'%.1f'%lv_factor if key!='lv=0' else '0':>6s}, " +
          f"Beitrag={contrib:.6e} m³")
    
    if key != 'andere':
        total_liq += contrib

print("-" * 80)
print(f"SUMME: {total_liq:.6e} m³")
print(f"Erwartet: {expected_liquid:.6e} m³")
print(f"Faktor: {total_liq / expected_liquid:.1f}×")

# Erforderlicher durchschnittlicher lv_factor für "andere"
print("\n" + "=" * 80)
print("SCHLUSSFOLGERUNG:")
print("=" * 80)

# Wenn "andere" durchschnittlich lv=1x hätten:
if other_W > 0:
    liq_if_1x = other_W * 1 * v_droplet / Vc
    print(f"\nWenn 'andere' lv=1×v_d hätten: Beitrag = {liq_if_1x:.6e} m³")
    
    # Tatsächlicher Beitrag
    actual_other_contrib = actual_in_system - (total_liq - contrib if key=='andere' else total_liq)
    print(f"Tatsächlicher Beitrag: {actual_other_contrib:.6e} m³")
    
    # Erforderlicher Faktor
    required_factor = actual_other_contrib / liq_if_1x if liq_if_1x > 0 else 0
    print(f"Erforderlicher lv_factor: {required_factor:.1f}×")
    
    print(f"\n→ Die 'anderen' Partikel haben durchschnittlich lv ≈ {required_factor:.1f}×v_d!")
    print(f"→ Das bedeutet: Viele Tropfen landen auf DEMSELBEN Partikel!")
