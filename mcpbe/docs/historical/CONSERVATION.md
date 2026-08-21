# Massenerhaltung: Fest- und Flüssigphase

Verifikationsbericht für den `wmcpbe`-Solver nach dem Refactoring.

Reproduzieren:

```bash
cd mcpbe
python -m tests.test_conservation          # lesbarer Report
python -m pytest tests/test_conservation.py -v   # 38 Tests
```

---

## 1. Was geprüft wird

Der Solver speichert **alle** Partikeleigenschaften *partikelintensiv*, also pro
physikalischem Partikel. Das Gewicht `W` sagt, wie viele physikalische Partikel
ein Rechenpartikel repräsentiert. Die erhaltenen Größen sind daher **extensiv**
und tragen `W`:

```
Feststoff:      M_s = Σ_i  W_i · V_solid_i
Flüssigkeit:    M_l = Σ_i  W_i · liquid_volume_i
```

mit

```
V_solid_i = V_dry_i · (1 − porosity_i)        für poröse Partikel
V_solid_i = V_dry_i                            für Vollkörper (porosity = NaN)
```

Erwartungswerte:

| Prozess | Feststoff | Flüssigkeit |
|---|---|---|
| Agglomeration | konstant | konstant (nur Umverteilung) |
| Breakage | konstant | konstant (proportional auf Fragmente) |
| Kontinuierliche Prozesse | konstant | konstant (nur intern ↔ extern) |
| Nucleation | konstant | steigt exakt um die zugegebene Menge |
| Vc-Verdopplung (DSMC) | konstant | konstant |

Zusätzlich gibt es **zwei unabhängige Wege**, den Feststoff zu berechnen:

* **Trockenvolumen-Weg:** `Σ W_i · V_flat[-1,i] · (1 − porosity_i)`
* **Komponenten-Weg:** `Σ W_i · Σ_d V_flat[d,i]`

Beide müssen übereinstimmen. Genau diese Übereinstimmung war in 2D gebrochen
(siehe [F-01](REFACTORING_FINDINGS.md#f-01)).

---

## 2. Ergebnis über einen vollständigen Lauf

`python -m tests.test_conservation`, Stand nach dem Refactoring:

```
Szenario                 Feststoff vor  Feststoff nach   rel. Drift  Komponenten  Flüssig-Drift
------------------------------------------------------------------------------------------------
agg_shear_1d             2.617994e-16   2.617994e-16      1.883e-16    0.000e+00     0.000e+00
agg_shear_1d_large       1.047198e-15   1.047198e-15      1.883e-16    0.000e+00     0.000e+00
agg_constant_1d          2.617994e-16   2.617994e-16      0.000e+00    0.000e+00     0.000e+00
agg_sum_1d               2.617994e-16   2.617994e-16      1.883e-16    0.000e+00     0.000e+00
agg_shear_2d             2.617994e-16   2.617994e-16      1.883e-16    0.000e+00     0.000e+00
break_powerlaw_1d        2.617994e-16   2.617994e-16      1.883e-16    0.000e+00     0.000e+00
mix_1d                   2.617994e-16   2.617994e-16      1.883e-16    0.000e+00     0.000e+00
granulation_1d           2.094395e-16   2.094395e-16      3.531e-16    3.531e-16     2.022e-14
```

`1.9e-16` entspricht einem einzelnen ULP bei doppelter Genauigkeit — das ist
Maschinengenauigkeit, kein Leck.

---

## 3. Vorher/Nachher

| Prüfung | vorher | nachher |
|---|---|---|
| Feststoff, `dim=1`, alle Prozesse | `1.9e-16` ✅ | `1.9e-16` ✅ |
| Feststoff, `dim=2`, Trockenvolumen-Weg | `1.9e-16` ✅ | `1.9e-16` ✅ |
| **Feststoff, `dim=2`, Komponenten-Weg** | **`6.0e-01` ❌ (+60 %)** | **`0.0` ✅** |
| **Flüssigkeit, Granulation** | **`5.8e-03` ❌ (−0.58 %)** | **`2.0e-14` ✅** |
| Feststoff mit aktiver Rekonstruktion | `4.4e-03` ❌ | Fehler statt stiller Fehlrechnung ✅ |

Die drei ❌-Zeilen entsprechen den Befunden
[F-01](REFACTORING_FINDINGS.md#f-01), [F-03](REFACTORING_FINDINGS.md#f-03) und
[F-02](REFACTORING_FINDINGS.md#f-02).

---

## 4. Wie der Flüssigkeitsverlust lokalisiert wurde

Die Gesamtbilanz allein sagt nur *dass* etwas fehlt. Um zu bestimmen *wo*, wurde
`Σ W·liquid_volume` um jeden Mutationspunkt herum gemessen:

```
Hook                               Aufrufe      Σ Δ(W·liq)
_append_particle_column               1921    -5.197754e-33
_do_one_agg                            300     0.000000e+00
_remove_particle_column               1920    -9.793655e-33
continuous.step                        300     0.000000e+00
nucleation.step                        300     5.088607e-19
nucleation.finalize                      1     4.853797e-19
```

Agglomeration, Breakage, die kontinuierlichen Prozesse und die Spaltenoperationen
erhalten die Flüssigkeit **exakt**. Eine Verfeinerung innerhalb der Nucleation
zeigte dann:

```
Zweig   Aufrufe   Zustand           erwartet (dW·v_drop)   fehlend
new        1908   9.985029e-19      9.985029e-19           −2.0e-32  (exakt)
```

Der reguläre Tropfenpfad ist exakt. Die Differenz zwischen den 9.985e-19 aus dem
Tropfenpfad und den 9.942e-19 im Endzustand stammte damit aus dem
**Resttropfen-Pfad**, der das Kindpartikel mit *nur* dem Rest anlegte statt mit
„Elternflüssigkeit + Rest“ — siehe [F-03](REFACTORING_FINDINGS.md#f-03).

Der Vorteil dieser Vorgehensweise: der Fehler wird dem *einen* Codepfad
zugeordnet, statt eine Gesamtabweichung von 0.58 % zu interpretieren.

---

## 5. Testabdeckung

`tests/test_conservation.py` (38 Tests):

| Test | Prüft |
|---|---|
| `test_solid_mass_is_conserved[*]` | `Σ W·V_solid` über den ganzen Lauf, 8 Szenarien |
| `test_volume_representations_agree[*]` | Trockenvolumen-Weg == Komponenten-Weg (Regression für F-01) |
| `test_solid_mass_conserved_per_agglomeration_event` | jedes einzelne Agglomerationsereignis |
| `test_solid_mass_conserved_per_breakage_event` | jedes einzelne Breakage-Ereignis |
| `test_liquid_conserved_by_agglomeration` | Flüssigkeit pro Agglomerationsereignis, nasse Population |
| `test_liquid_conserved_by_breakage` | Flüssigkeit pro Breakage-Ereignis |
| `test_liquid_conserved_by_continuous_processes` | Internalisierung + Kompression verschieben nur intern ↔ extern |
| `test_saturation_never_implies_more_liquid_than_stored` | `V_pore·S ≤ liquid_volume` — sonst erfindet `get_V_liquid_external()` durch Clamping Flüssigkeit |
| `test_state_bounds[*]` | `W > 0`, `porosity ∈ [0,1)` oder NaN, `saturation ∈ [0,1]`, `V_dry > 0` |
| `test_sampler_totals_match_arrays[*]` | `FenwickSampler.total() == Σ` der Propensity-Arrays |

Die Toleranz ist `1e-9` relativ. Real gemessen wird `~1e-16`; die Lücke dazwischen
ist bewusst großzügig, damit der Test nicht bei harmlosen Rundungsänderungen
rot wird, aber jedes echte Leck (kleinste beobachtete Leckrate: `4e-3`) sicher
fängt.

---

## 6. Bekannte Einschränkungen

1. **Rekonstruktion + Granulation** — nicht erhaltend, wird jetzt mit einer
   klaren Fehlermeldung abgelehnt. Siehe [F-02](REFACTORING_FINDINGS.md#f-02).
   Für trockene Läufe uneingeschränkt nutzbar.
2. **DSMC-Vc-Verdopplung** ist in allen Benchmark-Szenarien deaktiviert
   (`maybe_double_control_volume = False`, dem Default entsprechend). Die
   Erhaltung über eine Vc-Verdopplung hinweg ist damit **nicht** durch Tests
   abgedeckt. Vor dem produktiven Einsatz mit `maybe_double_control_volume=True`
   sollte ein entsprechendes Szenario ergänzt werden.
3. **MLP-/LMC-Breakage-Pfade** (`use_lmc_pre_model`, `use_lmc_live`,
   `lmc_use_breakage_model`) sind nicht abgedeckt — sie benötigen externe
   Modelldateien, die nicht im Repository liegen.
