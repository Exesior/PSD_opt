"""Debug-Script für Nukleations-Probleme"""
import numpy as np

# Testdaten aus Ihrem Script
droplet_diameter = 0.0005  # 0.5 mm
liquid_flow_rate = 0.00001  # 10 µL/s = 0.00001 L/s
t_total = 5.0

# Berechnungen
droplet_vol = (np.pi / 6.0) * (droplet_diameter ** 3)
liquid_flow_m3 = liquid_flow_rate * 1e-3  # L/s -> m³/s
droplet_rate = liquid_flow_m3 / droplet_vol
expected_droplets = int(droplet_rate * t_total)
expected_total_liquid = droplet_rate * t_total * droplet_vol

print(f'Tropfenvolumen: {droplet_vol*1e9:.4f} nL')
print(f'Tropfenrate: {droplet_rate:.2f} Tropfen/s')
print(f'Erwartete Tropfen: {expected_droplets}')
print(f'Erwartete Gesamtflüssigkeit: {expected_total_liquid*1e9:.4f} nL')

# Partikelgröße
particle_size = 27e-5  # 270 µm
particle_vol = (np.pi / 6.0) * (particle_size ** 3)
print(f'\nPartikelvolumen: {particle_vol*1e9:.4f} nL')
print(f'Verhältnis Tropfen/Partikel: {droplet_vol/particle_vol:.2f}')
print(f'Mindestpartikel für direkten Tropfen: {droplet_vol/particle_vol:.0f}x aggregieren')

# Floating-Point Test
dt = 1.0 / droplet_rate
print(f'\nFloating-Point Test:')
print(f'dt = 1/rate = {dt:.10f} s')
print(f'rate * dt = {droplet_rate * dt:.10f}')
print(f'int(rate * dt) = {int(droplet_rate * dt)}')
print(f'int(round(rate * dt)) = {int(round(droplet_rate * dt))}')

# Accumulator Test über mehrere Schritte
accumulated = 0.0
total_distributed = 0
for step in range(100):
    accumulated += dt
    num = int(round(droplet_rate * accumulated))
    if num > 0:
        total_distributed += num
        time_used = num / droplet_rate
        accumulated -= time_used

print(f'\nNach 100 Schritten mit round():')
print(f'Total distributed: {total_droplets}')
print(f'Erwartet: ~{int(droplet_rate * 100 * dt)}')
