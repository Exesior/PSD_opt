# Nucleation: Flüssigkeitsbilanz und Kontaktstellen-Internalisierung

**Stand 16.08.2026.** Zwei Divergenzen im Nucleations-Pfad gegenüber der regulären
Agglomeration, beide gemessen belegt und behoben.

Ausgangspunkt war die Beobachtung, dass die Porosität im Rumpf-Testlauf nicht
wächst. Die Untersuchung führte auf einen tieferliegenden Punkt: die Aufteilung
der Flüssigkeit in intern und extern wurde bei jedem Tropfen neu erfunden.

---

## 1. Die zugrunde liegende Physik

Ein Partikel trägt Flüssigkeit an zwei physikalisch verschiedenen Orten:

| | Ort | Steuert |
|---|---|---|
| **extern** | Film auf der Oberfläche | Stokes-Kriterium — ob eine Kollision haftet |
| **intern** | imbibiert im Porenraum, S = V_liq,int / V_pore | Rumpf-Festigkeit σ — ob das Granulat bricht |

Nur externe Flüssigkeit kann die viskose Brücke bilden, die Stoßenergie
dissipiert. Nur die interne Sättigung entscheidet über das Festigkeitsregime
(trocken / Übergang / kapillar), zwischen denen σ um den Faktor 6α/k springt.

**Daraus folgt die zentrale Eigenschaft: der intern/extern-Split ist eine
Zustandsgröße mit Gedächtnis.** Interne Flüssigkeit entsteht durch Kapillarsog
über die Zeit — beschrieben durch ein Ratengesetz (Braumann et al. 2007,
`liquid_internalization`). Sie ist kein fester Anteil der Gesamtmenge.

Ein frisch auftreffender Tropfen hat noch nichts durchdrungen. Er ist damit
**vollständig extern**. Die bereits imbibierte Flüssigkeit bleibt, wo sie ist.

---

## 2. Befund A — Sättigungs-Reset bei jedem Tropfen

### Was falsch war

`mcpbe_nucleation.py::_compute_saturation_for_liquid` wurde für **jeden** Tropfen
aufgerufen und bildete den intern/extern-Split jedes Mal neu aus der
Gesamtflüssigkeit:

```python
if has_internalization_kernel:
    return 0.0              # gesamte Flüssigkeit des Partikels gilt als extern
split_ratio = self._get_liquid_split_ratio(poro)
V_liq_int = min(split_ratio * v_liquid, V_pore)
return V_liq_int / V_pore
```

Bei aktivem Internalisierungs-Kernel — die Produktionskonfiguration — wurde also
**bedingungslos 0.0 zurückgegeben**. Ein Partikel, das der Kernel über Sekunden
hinweg bis S = 1.0 gefüllt hatte, fiel mit dem nächsten Tropfen auf S = 0 zurück.
Die interne Flüssigkeit wurde nicht gelöscht, aber als extern umdeklariert.

Die Internalisierung arbeitete damit dauerhaft gegen einen Reset an. Die Kopplung
Benetzung → Festigkeit → Bruch war an dieser Stelle aufgetrennt.

### Messung (Monkeypatch-Diagnose, kein Repo-File verändert)

Konfiguration `test_powerlaw_rumpf_full.py`, 34 µm Partikel, 10 µm Tropfen,
T = 12 s, Nucleation bis t = 10 s.

| | ε₀ = 0.0 | ε₀ = 0.8 |
|---|---|---|
| Aufrufe `_compute_saturation_for_liquid` | 286.967 | 287.741 |
| davon Rückgabe `0.0` | **286.967 (100 %)** | **287.741 (100 %)** |
| Partikel hatte S > 0, wurde auf 0 gesetzt | 6 | **102.587 (35,6 %)** |
| dabei intern → extern umdeklariert | 5.18e-15 m³ | **7.80e-11 m³** |
| größte vorgefundene Sättigung | 0.093 | **1.000** |

Bei ε₀ = 0.0 ist der Schaden maskiert: ohne Poren existiert kaum interne
Flüssigkeit, die verloren gehen könnte. Sobald Porosität da ist, betrifft es
jedes dritte Tropfen-Ereignis.

Sättigungsverlauf bei ε₀ = 0.0 (mean / max), Nucleation bis t = 10 s:

```
t = 0.01 s   0.000000 / 0.000000
t = 1.01 s   0.000000 / 0.000000
t = 3.62 s   0.000000 / 0.000000
t = 6.79 s   0.000000 / 0.000000
t = 9.96 s   0.002793 / 0.103055     <- erst zum Nucleations-Ende
```

Die Sättigung steigt erst, als der Tropfenstrom versiegt.

### Die Korrektur

`_compute_saturation_for_liquid` schreibt den Zustand jetzt **fort** statt ihn neu
zu bilden. Die Signatur bekommt dafür zwei zusätzliche Argumente: `src_idx`
(Quelle des bisherigen Zustands) und `v_added` (das gerade eintreffende Volumen).

```
V_int_alt = min(liquid_alt, V_pore_alt * S_alt)        # vorhandener Zustand

V_int_neu = V_int_alt                                   # Kernel aktiv:
                                                        # Tropfen komplett extern
V_int_neu = V_int_alt + split * v_added                 # kein Kernel: empirischer
                                                        # Split NUR auf den Zuwachs

V_int_neu = min(V_int_neu, V_pore_neu, v_liquid)        # Überschuss -> extern
S         = V_int_neu / V_pore_neu
```

Drei Punkte dazu:

1. **Der Tropfen geht vollständig nach extern**, wenn der Internalisierungs-Kernel
   aktiv ist. Der Kernel zieht ihn anschließend über die Zeit nach innen.
2. **Ohne Kernel** ist der empirische Split der einzige Mechanismus, der überhaupt
   Flüssigkeit nach innen bringt. Er greift deshalb auf den *Zuwachs*, nicht auf
   die Gesamtmenge — damit bleibt auch dieser Pfad inkrementell.
3. **Der Überschuss beim Schrumpfen des Porenraums wird externalisiert, nicht
   verworfen.** Die Flüssigkeitsbilanz ist so unantastbar wie die Massenbilanz.
   Identisch zu `_apply_compression` und `mcpbe_agg._merge_pair`.

Der Deckel `min(liquid_alt, V_pore_alt * S_alt)` beim Auslesen ist bewusst: `poro`,
`saturation` und `liquid_volume` werden an verschiedenen Stellen unabhängig
gesetzt und sind nicht garantiert zueinander konsistent.

**Geänderte Stellen:** `mcpbe_nucleation.py` — Methodendefinition sowie beide
Aufrufstellen (regulärer Tropfenpfad und Rest-Tropfen-Pfad `_liquid_remainder`).

---

## 3. Befund B — Kontaktstellen-Kernel wurde übersprungen

### Was falsch war

Beim Zusammenfügen zweier Partikel wird an der neu entstehenden Kontaktstelle
zusätzlich Flüssigkeit aus den Oberflächenfilmen in die frisch gebildete Brücke
gezogen. Das ist ein eigener, **ereignisbasierter** Vorgang beim Stoß — nicht
dasselbe wie die kontinuierliche Internalisierung über die Zeit. Modelliert wird
er von `liq_internalisation_agglomeration` (`braumann_2007`).

`mcpbe_agg.py::_merge_pair` ruft diesen Kernel auf:

```python
l_e_to_i = self.kernel_manager.compute_liquid_internalization_agglomeration(...)
V_liq_int_merged = V_liq_int_contrib + l_e_to_i
V_liq_ext_merged = V_liq_ext_contrib - l_e_to_i
```

`mcpbe_nucleation.py::_manual_agglomerate_particles` tat das **nicht**. Es addierte
interne und externe Flüssigkeit der Eltern getrennt und war damit fertig. Der
Kernel war in der Testkonfiguration aktiviert (`LIQ_INTERN_AGG_ENABLED = True`),
wurde bei jeder Nucleations-Agglomeration aber stillschweigend übergangen.

### Die Korrektur

Der Aufruf wird in `_manual_agglomerate_particles` nachgezogen, mit derselben
Unterlauf-Behandlung wie in `_merge_pair`: will der Kernel mehr internalisieren,
als außen vorhanden ist, geht alles nach innen und die externe Menge wird 0 —
es wird nichts erfunden.

---

## 4. Warum das im aktuellen Testlauf wenig sichtbar ist

Die Konfiguration `test_powerlaw_rumpf_full.py` steht auf `INITIAL_POROSITY = 0.0`.
Mit `cone_model` entsteht Porosität ausschließlich geometrisch bei Agglomeration
(`compute_nucleation_porosity` liefert bewusst 0.0). Ohne Poren gibt es keinen
Porenraum, in den Flüssigkeit internalisiert werden könnte — beide Befunde sind
damit weitgehend wirkungslos.

Gemessen über 195.013 Tropfen:

- 195.005 von 195.054 Geometrie-Aufrufen nehmen den Zweig „porenlos → Kernel"
  und bekommen ε = 0.0 zurück.
- Die erzwungene Agglomeration greift 24× (0,012 %). Auslöser ist
  `V_dry < liquid_volume + v_droplet`, also die *aufsummierte* Beladung; bei
  34 µm Partikeln und 10 µm Tropfen liegt V_dry um Faktor 39 darüber.
- Die 24 Kinder bekommen exakt ε = 0.2000 — entsprechend der Abbildung
  ε → 0.8·ε + 0.2, die das Kegelmodell für gleich große Partikel liefert.
- Sie tragen zusammen 0,012 % des Gesamtgewichts und werden regulär bei W = 0
  entfernt (49 Entfernungen, alle bei `W = 0.000e+00`).

**Konsequenz:** Solange `INITIAL_POROSITY = 0.0` steht, ist der Lauf gegen beide
Korrekturen praktisch blind. Für eine aussagekräftige Bewertung braucht es eine
porös startende Population.

> **Nachtrag 17.08.2026.** Dieser Abschnitt beschreibt einen überholten Stand:
> `test_powerlaw_rumpf_full.py` läuft inzwischen mit `INITIAL_POROSITY = 0.7`, die
> Blindheit ist also aufgehoben und beide Korrekturen wirken voll. Zwei Folgen davon:
>
> - `compute_nucleation_porosity` wird gar nicht mehr erreicht. Bei poröser Population
>   greift immer Fall 1 von `_resolve_nucleation_geometry` (`poro_old > 0` ⇒ Geometrie
>   unverändert); die dort beschriebene ε = 0.0 des `cone_model` ist toter Pfad, solange
>   kein porenloses Partikel entsteht.
> - Die Zwangs-Agglomeration ist **nicht mehr selten**. Die hier gemessenen 24 von
>   195.013 Tropfen (0,012 %) galten für 34-µm-Partikel mit 10-µm-Tropfen und trockener
>   Population. In einer verkleinerten Sonde mit ε₀ = 0.7, 15-µm-Tropfen und
>   unverändertem Volumenstrom feuerte sie 118.938-mal, während die reguläre
>   Agglomeration 0 Merges beitrug — der Auslöser `V_dry < new_liquid` wird zum
>   Normalbetrieb, sobald die zugeführte Flüssigkeit in die Größenordnung des gesamten
>   Trockenvolumens kommt. Die Sonde ist überzeichnet (300 statt 2000 Startpartikel bei
>   gleichem Volumenstrom); die Häufigkeit in der echten Konfiguration ist ungemessen.
>
> Die Zwangs-Agglomeration hat einen eigenen Vermerk:
> [`Offen_Zwangsagglomeration_Nucleation.md`](Offen_Zwangsagglomeration_Nucleation.md).
> Sie ist als Immersions-Nukleation physikalisch begründet und kein Defekt; offen ist
> allein, ob der Auslöser das Tropfenvolumen oder die akkumulierte Beladung sein soll.

---

## 4b. Verifikation

### Wirksamkeit (dieselbe Sonde vor und nach der Korrektur, ε₀ = 0.8)

| | vorher | nachher |
|---|---|---|
| Aufrufe `_compute_saturation_for_liquid` | 287.741 | 209.214 |
| davon Rückgabe `0.0` | **100 %** | **4,8 %** |
| Partikel hatte S > 0, wird auf 0 gesetzt | **102.587** | **0** |
| intern → extern umdeklariert | 7.80e-11 m³ | **0.00e+00 m³** |

Sättigungsverlauf während der Nucleation — vorher bis t ≈ 6,8 s konstant 0,
jetzt monoton wachsend:

```
t = 0.01 s   mean 0.000000   max 0.000000
t = 1.01 s   mean 0.000910   max 0.002919
t = 2.04 s   mean 0.003134   max 0.009234
t = 3.31 s   mean 0.007756   max 0.020334
t = 7.28 s   mean 0.029333   max 0.069842
```

Die Internalisierung baut sich jetzt auf, statt gegen einen Reset anzuarbeiten.

### Bilanzen

- **`tests/` — 38 passed** (15:52 min), keine Regression.
- Feststoffmasse: `✓ PASS (-0.0000 %)`.
- Flüssigkeit: `Added (stats)` und `In system` sind exakt gleich — es geht nichts
  verloren. Ein zwischenzeitlich beobachteter `-27,18 %`-Fehlbetrag stammte
  ausschließlich daraus, dass der Diagnoselauf auf `T_TOTAL = 6 s` verkürzt war,
  während `NUCLEATION_DURATION = 10 s` beträgt — es wurde also weniger *zugeführt*,
  nicht weniger *gefunden*.
- Laufzeitnachweis über 613 Prüfpunkte: keine nicht-endlichen Werte in den
  Physik-Arrays, `V_dry·(1−ε)` gegen `V_flat[0]` maximal 1.9e-16 relativ,
  `V_liq_int / V_pore ≤ 0.0698` — die Sättigungsgrenze wird eingehalten.

---

## 5. Invarianten, die dabei gelten müssen

- Σ V_solid · W konstant
- Σ V_liquid · W konstant, und V_int + V_ext = V_gesamt zu **jedem** Zeitpunkt
- V_dry · (1 − ε) = V_solid, exakt, für jedes Partikel, nach jeder Operation
- 0 ≤ S ≤ 1, 0 ≤ ε < 1
- Intensive Größen skalieren nie mit W; nur W ist extensiv

Und die Regel, aus der beide Befunde folgen:

> **Kein Prozess darf eine Zustandsgröße neu berechnen, die aus einem Ratengesetz
> stammt.** Er darf sie nur inkrementell verändern. Wer S aus dem Gesamtvolumen
> neu bestimmt, löscht die Historie der Internalisierung — und damit die Kopplung
> zwischen Benetzung, Festigkeit und Bruch.
