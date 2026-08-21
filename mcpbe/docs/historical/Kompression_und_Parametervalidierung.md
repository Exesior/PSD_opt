# Kompression wirkungslos, Kernel-Parameter stillschweigend verworfen

**Stand 17.08.2026.** Zwei unabhängige Befunde mit demselben Symptom: ein Parameter
steht in der Konfiguration, wird beim Lauf aber nicht wirksam. Beide gemessen belegt,
beide behoben.

Ausgangspunkt war die Beobachtung, dass in `test_powerlaw_rumpf_full.py` die minimale
Porosität nach 180 s exakt auf dem Startwert 0.7 stand, obwohl `COMPRESSION_ENABLED`
gesetzt war.

---

## 1. Befund A — `compute_array` der Kompression schrieb ins Leere

### Was falsch war

`kernels/continuous_processes/porosity_compression.py::compute_array` (Z. 173 alt):

```python
out[finite][can_compress] = updated
```

Das ist eine **verkettete Fancy-Index-Zuweisung**. Python wertet sie in zwei Schritten
aus: `out[finite]` ist Boolean-Indizierung und liefert in NumPy grundsätzlich eine
**Kopie**, nicht einen View; das anschließende `[can_compress] = updated` schreibt
folglich in diese Kopie, die am Ende der Anweisung verworfen wird. `out` bleibt
unverändert, die Funktion gibt ihre Eingabe zurück.

Der `ContinuousProcessesHandler` nutzt den Kernel, sobald einer registriert ist
(`mcpbe_continuous_processes.py:340-347`); der analytische Fallback direkt darunter ist
korrekt, wird in dieser Lage aber nie erreicht. `COMPRESSION_RATE` und `MIN_POROSITY`
waren damit in **jedem** Lauf mit registriertem Kompressions-Kernel folgenlos.

Zusätzlich betroffen war die Skalar-Parität: der Docstring behauptet Bitgleichheit zur
Schleife aus `compute` — geprüft wurde sie nie.

### Messung

Kernel-Ebene, `rate=0.2`, `min_porosity=0.2`, `dt=100 s`:

| Eingabe | erwartet (Schleife aus `compute`) | `compute_array` vorher |
|---|---|---|
| `[0.7, 0.7, 0.5, 0.0, nan]` | `[0.2, 0.2, 0.2, 0.0, nan]` | `[0.7, 0.7, 0.5, 0.0, nan]` |

Solverlauf mit einer Sonde um `compute_array` (3 s, 300 Startpartikel, sonst
Trial-Konfiguration, ε₀ = 0.7):

| | vorher | nachher |
|---|---|---|
| `compute_array`-Aufrufe | 8 | 8 |
| komprimierbare Partikel (kumuliert, ε > ε_min) | 4.934 | 10.984 |
| davon tatsächlich verändert | **0** | **10.984 (100 %)** |
| Porosität final min / mean / max | 0.843 / 0.877 / 0.908 | 0.776 / 0.835 / 0.875 |
| n_comp am Ende | 6.037 | 1.155 |

Der Kernel wurde also aufgerufen, mit sinnvollem `dt`, bekam tausende komprimierbare
Partikel vorgelegt — und gab in 100 % der Fälle die Eingabe zurück.

### Die Korrektur

Die Maske wird vor dem Schreiben auf die Positionen des vollen Arrays zurückgeführt,
und der Schreibzugriff ist genau eine Indizierung:

```python
compressible = np.zeros(poro.shape, dtype=bool)
compressible[finite] = can_compress
out[compressible] = updated
```

`can_compress` ist relativ zu `poro[finite]` indiziert und damit kürzer als `out` — es
direkt auf `out` anzuwenden träfe die falschen Partikel. Verifiziert: bitgleich zur
Schleife aus `compute` für `dt` ∈ {0, 1e-6, 0.5, 100} über 57 Stützstellen.

### Tragweite

**Alle bisherigen Ergebnisse mit aktiver Kompression sind hinfällig, die
Sensitivity-Suiten eingeschlossen.** Die Porosität geht über `powerlaw_rumpf` in die
Rumpf-Festigkeit und über `stokes_krit` in die Agglomerations-Akzeptanz ein; die
Trajektorie verschiebt sich nicht, sie wird eine andere (n_comp 6.037 → 1.155 im
Sondenlauf).

Bis zu diesem Fix hatte `ε` nur **einen** wirksamen Mechanismus, und der zeigt nach
oben: `cone_model` bildet bei gleich großen Partikeln ε → 0.8·ε + 0.2 ab, mit Fixpunkt
ε = 1.0. Ohne Gegenspieler läuft die Porosität monoton gegen die Hohlkugel.

---

## 2. Befund B — falscher Parametername wurde stillschweigend geschluckt

### Was falsch war

Jeder Kernel baut seinen Zustand nach demselben Muster:

```python
defaults = self.get_default_params()
defaults.update(params)
self.params = self.validate_params(defaults)
```

Ein falsch geschriebener Name überschreibt dadurch **nichts** — er landet als
zusätzlicher Dict-Eintrag, den nie jemand liest, und der Kernel läuft auf seinem
Default weiter. Der Lauf ist vollständig, die Ergebnisse sind plausibel, und der
Parameter scheint wirkungslos zu sein, egal welchen Wert man einsetzt.

Konkret: `liquid_internalization` liest `k_int`, mehrere Trial-Skripte übergaben
`k_intern`.

```
params      : {'k_int': 1000000000000.0, 'k_intern': 1e-12}
wirksames k : 1000000000000.0
```

Betroffene Aufrufstellen (Stand des Befunds): `Trials/test_sensitivity.py` und dessen
beide Kopien unter `Trials/Results/…`, `test_powerlaw_rumpf_with_analysis.py`,
`test_particle_merger_rumpf_full.py`, `test_initial_liquid_behavior.py`,
`test_droplet_10um_stable.py`, `test_droplet_20um_instable.py`. Alle liefen mit dem
Default 1e12 statt mit dem konfigurierten Wert.

**Quelle des Fehlers war die Dokumentation**: `kernels/README.md` nannte den Parameter
`k_intern` — inzwischen korrigiert. `docs/old/Parameter_Liste.md` enthält denselben
Fehler, bleibt aber als Archiv unverändert.

Eine systematische Prüfung aller Kernel des Trial-Setups ergab genau diese eine Stelle;
`shear_chin1998`, `stokes_krit`, `powerlaw_rumpf`, `cone_model`,
`porosity_compression` und `liq_internalisation_agglomeration` waren sauber.

### Die Korrektur

`kernels/base.py` bekommt zwei Bausteine:

- **`reject_unknown_params(kernel, supplied)`** — vergleicht die übergebenen Namen mit
  denen, die der Kernel liest, und wirft sonst `ValueError`. Über `difflib` wird der
  vermutlich gemeinte Name vorgeschlagen.
- **`KernelBase.get_optional_params()`** — für Parameter, die ein Kernel akzeptiert, die
  aber bewusst keinen Default haben, weil „nicht gesetzt" ein eigener Zustand ist.

Aufgerufen wird die Prüfung in allen sechs `get_*_kernel(...)`-Factories — der Stelle,
an der Konfiguration ins System eintritt (der Solver legt jeden Kernel darüber an).

```
ValueError: Unknown parameter(s) for kernel 'liquid_internalization':
'k_intern' (did you mean 'k_int'?). Accepted parameters: ['k_int']. ...
```

`cone_model` deklariert `default_porosity` und `liquid_split_ratio` als optional —
beide werden per `in self.params` / `.get()` gelesen und stehen nicht in den Defaults.
Ohne diese Deklaration hätte die neue Prüfung den Hebel blockiert, mit dem sich die
Nucleations-Porosität von außen setzen lässt. `cone_model` ist der einzige Kernel im
Repo mit diesem Muster.

**Regel für neue Kernel:** wer einen Parameter über `self.params.get(...)` einliest,
ohne ihn in `get_default_params()` zu führen, muss ihn in `get_optional_params()`
eintragen.

### Verifikation

- Alle 18 registrierten Kernel lassen sich weiterhin mit ihren eigenen Defaults bauen.
- Die optionalen `cone_model`-Parameter gehen durch.
- `k_intern` und ein synthetischer Tippfehler (`gama` statt `gamma`) werden mit
  korrektem Vorschlag abgelehnt.

---

## 3. Größenordnung von `k_int` — kein Bug, aber eine Falle

Der Kernel rechnet `α = k_int · (V_pore − V_liq,ges)` und `exp(α·dt)`. Für die
34-µm-Konfiguration (V_pore ≈ 1.4e-14 m³) gemessen, S₀ = 0.10, dt = 0.5 s:

| `k_int` | α·dt | S nach einem Schritt |
|---|---|---|
| 1e-12 | 5.8e-27 | 0.100000 |
| 1e0 | 5.8e-15 | 0.100000 |
| 1e8 | 5.8e-07 | 0.100000 |
| 1e12 | 5.8e-03 | 0.100646 |
| 1e14 | 5.8e-01 | 0.146716 |

Unterhalb von etwa 1e11 rundet `exp(α·dt)` auf exakt 1.0, und mit `exp_term = 1`
liefert die analytische Lösung algebraisch **exakt** den Eingangswert zurück:

```
l_neu = (V_pore·B − l_ges·A)/(B − A)  mit A = V_pore − l, B = l_ges − l
      = l·(l_ges − V_pore)/(l_ges − V_pore) = l
```

Das ist Stillstand, kein Rundungsrauschen. Faustregel: `k_int` so wählen, dass
`k_int · V_pore · t_prozess` in der Größenordnung 1 liegt. Die frühere Angabe
„1e6–1e10" in `kernels/README.md` war für µm-Granulate durchweg wirkungslos und ist
korrigiert.

---

## 4. Übertragbare Lehren

1. **Ein Wert, der exakt auf dem Startwert steht, ist ein Alarmsignal.** Physik liefert
   krumme Zahlen. Steht eine Größe nach 180 s Simulation bit-genau auf ihrem
   Anfangswert, hat kein Prozess sie angefasst — dann ist nicht der Parameter zu klein,
   dann findet die Operation nicht statt.
2. **NumPy-Regel:** stehen links vom `=` zwei Klammerpaare hintereinander und ist das
   erste eine Maske oder Indexliste, wird in eine Kopie geschrieben. `a[maske][x] = v`
   ist immer falsch, `a[maske] = v` immer richtig. Auffindbar per Textsuche nach `][`
   gefolgt von `=`; im Paket existiert derzeit keine weitere solche Stelle.
3. **Wo eine schnelle Variante eine langsame ersetzt, gehört ein Test dazwischen, der
   beide vergleicht** — die langsame ist die Referenz. Genau dieser Test fehlte und
   hätte Befund A am Tag seiner Entstehung gemeldet.
4. **Ein Tippfehler in einem Parameternamen muss laut scheitern.** Ein geschluckter Name
   erzeugt einen Lauf, der vollständig aussieht und dessen Konfiguration nicht der ist,
   mit der gerechnet wurde.
5. Der Validierungs-Check `|Δporo_mean| > 1e-6` in `test_powerlaw_rumpf_full.py` heißt
   „Porosity evolved" und hält, was er verspricht — er taugt aber nicht als Wächter für
   die Kompression, weil `cone_model` die geforderte Änderung allein liefert. Ein Test
   für die Kompression müsste ein Partikel verfolgen, das *nicht* agglomeriert, oder
   `process_type = "compression"` gegen die analytische Lösung fahren.

---

## 5. Offen, bewusst nicht geändert

- **Wert von `LIQ_INTERN_RATE`.** Der Schlüssel ist am 17.08.2026 in allen sechs aktiven
  Trial-Skripten auf `k_int` korrigiert; die beiden archivierten Kopien unter
  `Trials/Results/…` bleiben bewusst unverändert, weil sie dokumentieren, womit die
  abgelegten Ergebnisse gerechnet wurden. Damit wirken dort jetzt die konfigurierten
  Werte (1e-15 … 1e9) statt des Defaults 1e12 — alle liegen unter der Wirkungsschwelle
  von ~1e11, die Internalisierung ist in diesen Läufen also faktisch abgeschaltet. Der
  physikalisch gewollte Wert ist noch zu bestimmen.
- **`k_agg = 1.0` in `cone_model`.** Der gesamte geometrische ΔV wird zu Porenraum. Mit
  jetzt wirksamer Kompression stehen sich zwei echte Effekte gegenüber; ob das
  Gleichgewicht stimmt, ist eine Kalibrierfrage.
- **Die Zwangs-Agglomeration der Nucleation** (`V_dry < new_liquid`) ist als
  Immersions-Nukleation physikalisch begründet und in erster Näherung passend; offen
  ist nur, ob der Auslöser das Tropfenvolumen oder die akkumulierte Beladung sein soll.
  Eigener Vermerk:
  [`Offen_Zwangsagglomeration_Nucleation.md`](Offen_Zwangsagglomeration_Nucleation.md).
