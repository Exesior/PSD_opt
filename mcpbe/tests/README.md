# WMCPBE — Tests, Benchmarks und Regressionsschutz

Alle Kommandos aus dem Verzeichnis `mcpbe/` ausführen.

## Einrichtung

```bash
python -m venv .venv
```

```bash
.venv\Scripts\python.exe -m pip install -e ./pbe-core -e ./mcpbe pytest
```

Python 3.11 ist die Referenzversion (`pyproject.toml`: `>=3.9,<3.12`);
getestet mit numpy 2.3.5, numba 0.63.1.

---

## Was es gibt

| Datei | Zweck |
|---|---|
| `bench/scenarios.py` | Zehn vollständig spezifizierte, geseedete Simulationen. Einzige Quelle für Solver-Konfigurationen in Tests und Benchmarks. |
| `bench/golden_reference.py` | Bitgenaue Regressions-Fingerprints (SHA-256 über alle float64-Bytes). |
| `bench/bench_wmcpbe.py` | Wandzeit-Benchmark mit Vorher/Nachher-Vergleich. |
| `bench/profile_wmcpbe.py` | `cProfile` für ein einzelnes Szenario. |
| `bench/compare_propensity_modes.py` | Genauigkeit, Skalierung und Ensemble-Äquivalenz von `pairwise` vs. `moment` (Hintergrund: [`../docs/historical/MOMENT_MODE.md`](../docs/historical/MOMENT_MODE.md)). |
| `test_conservation.py` | Massenerhaltung Fest-/Flüssigphase, Zustandsgrenzen, Sampler-Konsistenz. |
| `test_merger_lookup_equivalence.py` | `merger_lookup` in {scan, hash, hash_lazy} bleibt physikalisch äquivalent. |
| `golden_reference.json` | **Aktuelle Baseline**, aufgenommen auf `a4aa35a` + Referenzszenario. Zehn Szenarien, Gesamtlaufzeit ~190 s. |
| `golden_reference.OUTDATED.json` | Veraltet (Stand 2026-07-31), stimmt seit der Mixer-Speed-Integration mit keinem Szenario überein. Nur noch als historischer Diff-Anker. |

---

## Regression prüfen (das Wichtigste)

```bash
python -m tests.bench.golden_reference check --ref tests/golden_reference.json
```

> Die Baseline gilt für den Codestand, auf dem sie aufgenommen wurde. Nach einer
> Änderung, die numerische Ergebnisse **absichtlich** verschiebt, neu aufnehmen —
> auf einem als gut bekannten Stand:
> ```bash
> python -m tests.bench.golden_reference record --out tests/golden_reference.json
> ```

Jede Änderung, die numerische Ergebnisse *nicht* verändern soll, muss hier
`all fingerprints match` liefern. Der Fingerprint deckt jeden gespeicherten
Snapshot **und** den finalen Live-Zustand ab, also auch Läufe, die vom
Ereignisbudget statt von der Zeit begrenzt werden.

Referenz nach einer *beabsichtigten* Ergebnisänderung neu aufnehmen (gleiches
`record`-Kommando wie oben).

> Das ist ein bewusster Schritt. Vorher klären, **warum** sich Ergebnisse ändern,
> und den Grund in `../docs/historical/REFACTORING_FINDINGS.md` festhalten.

---

## Massenerhaltung

```bash
python -m tests.test_conservation
```

Liefert einen Report über Fest- und Flüssigphase je Szenario. Als pytest-Suite:

```bash
python -m pytest tests/test_conservation.py -v
```

---

## Performance

```bash
python -m tests.bench.bench_wmcpbe --repeats 3
```

```bash
python -m tests.bench.bench_wmcpbe --repeats 3 --scale 4 --propensity-mode moment
```

Vorher/Nachher-Vergleich über zwei JSON-Dateien:

```bash
python -m tests.bench.bench_wmcpbe --repeats 3 --out after.json --compare before.json
```

Hotspots eines einzelnen Szenarios:

```bash
python -m tests.bench.profile_wmcpbe granulation_1d --top 15
```

---

## Aufbau der Szenarien

Jedes Szenario ist über ein **festes Ereignisbudget** begrenzt
(`Scenario.maxiter`), nicht über die simulierte Zeit. Damit ist die Wandzeit
direkt proportional zu den Kosten pro Monte-Carlo-Ereignis, und die Suite bleibt
unabhängig von den Kernelparametern in einem vorhersagbaren Rahmen.

Zwei Szenarien dominieren die Wandzeit: `agg_shear_2d` (~110 s, 2D hat keinen
kompilierten Batch-Pfad) und `granulation_rumpf_dynamic_1d` (~76 s, EKE ist
nicht separierbar und laeuft daher über den O(n²)-Pfad). Alle übrigen zusammen
bleiben unter einer Sekunde; ein kompletter Golden-Reference-Durchlauf liegt bei
~190 s.

| Szenario | dim | Prozess | n₀ | Ereignisse | Deckt ab |
|---|---|---|---|---|---|
| `agg_shear_1d` | 1 | Agglomeration | 500 | 400 | Standard-Granulationskernel |
| `agg_shear_1d_large` | 1 | Agglomeration | 2000 | 150 | O(n²)-Skalierung |
| `agg_constant_1d` | 1 | Agglomeration | 500 | 400 | konstanter Kernel + JIT-Partnerwahl |
| `agg_sum_1d` | 1 | Agglomeration | 500 | 400 | Summenkernel |
| `agg_brownian_1d` | 1 | Agglomeration | 500 | 400 | Brownscher Kernel |
| `agg_shear_2d` | 2 | Agglomeration | 500 | 300 | Zweikomponentenpfad |
| `break_powerlaw_1d` | 1 | Breakage | 500 | 1500 | Bruchrate + Fragment-CDF |
| `mix_1d` | 1 | Mix | 500 | 300 | schlimmster Fall: beide Sampler pro Ereignis |
| `granulation_1d` | 1 | Agglomeration | 400 | 300 | + Nucleation, Internalisierung, Kompression |
| `granulation_rumpf_dynamic_1d` | 1 | Mix | 1000 | 1600 | **Produktionsfall**: EKE + stokes_dynamik + powerlaw_rumpf_dynamic + cone_model, mit allen Handlern. Einziges Szenario mit **Bruchereignis bei aktiven Handlern**. ~76 s |

`granulation_rumpf_dynamic_1d` spiegelt
[`Trials/test_powerlaw_rumpf_dynamic_full.py`](../src/wmcpbe/Trials/test_powerlaw_rumpf_dynamic_full.py)
(gleicher Seed, gleiche Kernelparameter, gleiche monodisperse Initialisierung).
Das grosse Ereignisbudget ist Absicht: das Rumpf-Festigkeitsmodell laesst
Partikel erst brechen, wenn sie poroes und nass geworden sind — das erste
Bruchereignis faellt auf Iteration ~495 (t ≈ 35 s). Ein kleineres Budget deckt
nur den Agglomerationspfad ab.

**Warum JIT-Warmup?** Ohne den `warmup()`-Aufruf zahlt der erste gemessene Lauf
mehrere hundert Millisekunden LLVM-Kompilierzeit und die Zahlen sind wertlos.
`golden_reference`, `bench_wmcpbe` und `profile_wmcpbe` rufen ihn automatisch auf.

---

## Neues Szenario hinzufügen

`SCENARIOS` in `bench/scenarios.py` erweitern. Es wird automatisch von
Benchmark, Profiler und Golden-Reference erfasst; für die Erhaltungstests den
Namen zusätzlich in `SOLID_SCENARIOS` (`test_conservation.py`) eintragen.
