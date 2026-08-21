# OFFEN: Auslösebedingung der Zwangs-Agglomeration in der Nucleation

**Vermerk vom 17.08.2026. Kein Bug, keine Änderung vorgenommen.** In erster Näherung
ist der aktuelle Stand passend; die Diskussion wurde bewusst vertagt. Dieses Dokument
hält den Stand fest, damit sie nicht verloren geht.

---

## 1. Worum es geht

`mcpbe_nucleation.py::_distribute_one_droplet_with_dW` enthält eine Schleife
([Z. 1427](../src/wmcpbe/mcpbe_nucleation.py)):

```python
V_dry_sum     = float(solver.V_flat[-1, i])
current_liquid = float(solver.liquid_volume[i])
new_liquid     = current_liquid + v_droplet

while V_dry_sum < new_liquid and attempts < max_attempts:
    j = self._select_particle_uniform_physical()
    child_idx = self._perform_nucleation_agglomeration(i, j)
```

Ein Tropfen trifft ein Partikel. Reicht dessen Trockenvolumen nicht aus, wird so lange
mit zufälligen Partnern verschmolzen, bis genug `V_dry` beisammen ist. Der
Porositäts-Kernel greift dabei regulär (geprüft: 118.938 Auslösungen, 118.938
`compute_merged_porosity`-Aufrufe).

## 2. Die Physik dahinter (Nutzer, 17.08.2026)

> Ein Tropfen fliegt aus der Düse in den Rührer. Auf seinem Weg trifft er Partikel. Weil
> der Tropfen flüssig und damit voll plastisch ist, **kann der Kontakt nicht abgelehnt
> werden**. Ist der Tropfen größer als das Partikel, reißt er es einfach fort und fliegt
> zum nächsten. Das geht so lange, bis der Tropfen nicht mehr groß genug ist —
> schwierige, aber nötige Annahme: bei `v_dry > v_liq`.

Damit ist geklärt, warum dieser Pfad das Stokes-Kriterium umgeht, und das ist **richtig
so**: Stokes entscheidet über die Dissipation einer viskosen Brücke zwischen zwei festen
Körpern. Auf einen freien, voll plastischen Tropfen ist es nicht anwendbar. Es ist
Immersions-Nukleation — der Tropfen ist die aufnehmende Phase.

Ergänzend (Nutzer): es gibt keine bevorzugte Agglomeration nasser Partikel; Stokes ist
reine *acceptance*. Ein Partikel, das ohnehin schon so nass ist, dass `v_dry` und
`v_liq` vergleichbar sind, würde durch weitere Flüssigkeit so nass, dass es
wahrscheinlicher agglomeriert. Dass das sofort passiert, ist nicht perfekt, aber
vertretbar.

## 3. Der offene Punkt

**Der Code prüft nicht das Tropfenvolumen, sondern die gesamte auf dem Partikel
angesammelte Flüssigkeit.** `new_liquid = current_liquid + v_droplet`, wobei
`current_liquid` alles ist, was das Partikel über frühere Tropfen bereits trägt.

Beide Lesarten fallen in genau einem Fall zusammen: frischer Tropfen auf trockenes
Partikel (`current_liquid = 0`), dann heißt die Bedingung `v_droplet > V_dry` — exakt
das beschriebene Modell.

Sobald das Partikel schon Flüssigkeit trägt, laufen sie auseinander. In der
Trial-Konfiguration ist das der Normalfall:

| | Volumen |
|---|---|
| Tropfen (15 µm) | 1.77e-15 m³ |
| Partikel (34 µm) | 2.06e-14 m³ |

Der Tropfen ist **zwölfmal kleiner** als das Partikel — er kann nie eines fortreißen.
Die gemessenen 118.938 Auslösungen stammen also nicht aus dem beschriebenen
Mechanismus, sondern aus akkumulierter Beladung: Partikel, die über viele Tropfen
hinweg nass geworden sind, bis die Summe ihr Trockenvolumen überstieg.

Zwei physikalisch verschiedene Vorgänge:

| | Auslöser | Bedeutung |
|---|---|---|
| aktuell | `V_dry_sum < current_liquid + v_droplet` | übersättigtes Granulat agglomeriert |
| beschrieben | `V_dry_sum < v_droplet` | fliegender Tropfen sammelt Partikel ein |

## 4. Wenn wir die Diskussion aufnehmen

Vorbereitend messbar, ohne Codeänderung:

- Anteil der Auslösungen mit `current_liquid = 0` (beschriebenes Modell) gegen die aus
  akkumulierter Beladung. Erwartung bei 15 µm auf 34 µm: **0 zu 118.938**.
- Häufigkeit in der *echten* Konfiguration. Die vorliegende Zahl stammt aus einer
  verkleinerten Sonde (300 statt 2000 Startpartikel bei unverändertem Volumenstrom) und
  ist dadurch überzeichnet.
- Verteilung der Schleifendurchläufe pro Tropfen. Erlaubt sind `3 × a_tot`; jeder
  Durchlauf addiert einen vollen Kegelpillen-Sprung von gemessen +0.078 bis +0.084 in
  der Porosität. Offen ist, ob ein *einzelner* Tropfen mehrere solcher Sprünge auslösen
  soll.

Zur Einordnung: der Fixpunkt der Abbildung ε → 0.8·ε + 0.2 bei ε = 1 ist laut Nutzer
physikalisch in Ordnung — poröse Partikel brechen entsprechend leichter. Dieser Punkt
ist **nicht** Teil der offenen Frage.

## 5. Warum es zählt

Dieser Pfad zählt nicht in `real_agg_events`. Wachstum entsteht dadurch an den Zählern
vorbei: im Parameter-Sweep stieg x50 bei zehnfacher Flüssigkeit auf 54–58 µm, während
`real_agg = 0` dastand. Wer Läufe auswertet, muss die beiden Wege getrennt lesen.
