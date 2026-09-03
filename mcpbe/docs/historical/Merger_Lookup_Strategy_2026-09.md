# Merger-Lookup: linearer Scan als Default

`solver.merger_lookup ∈ {"scan", "hash", "hash_lazy"}` — Default `"scan"`
(seit 2026-09-03). Definiert in `mcpbe_base._MERGER_LOOKUP_MODES`.

Kurzfassung: Der `ParticleMerger` prüft jedes neu erzeugte Partikel gegen den
Bestand und erhöht bei Treffer nur `W`. Die Suche lief bisher über einen
**Hash-Index** (O(1) je Lookup). Profiling zeigt: das ist im Nassgranulations-Fall
7–8× langsamer als ein simpler vektorisierter linearer Scan, weil der Hash-Key
jedes Event für die halbe Population neu berechnet werden muss. Der Default ist
jetzt der Scan; Hash bleibt über `merger_lookup="hash"` erreichbar.

---

## 1. Das Problem

Der Hash-Key eines Partikels:

```python
key = (round(log10(V_dry), 8), round(log10(liquid), 8), round(poro, 4), round(sat, 4))
```

`V_dry`, `porosity` und `saturation` schreiben die **kontinuierlichen Prozesse**
(Poren-Kompression + Flüssigkeits-Internalisierung) bei *jedem* MC-Event für
*fast jedes* Partikel neu. Der Index ist danach veraltet und wird neu aufgebaut:

- `reindex_all()` — 1× je Event aus `solve()` — läuft über **alle** Partikel,
  rechnet jeden Key neu, vergleicht mit dem gespeicherten
- `_reindex_merger()` — ~2× je Event aus Kompression / Internalisierung

### Profil: `powerlaw_rumpf_dynamic_full` (nucleation + agg + break + Kompression + Internalisierung), n_comp ≈ 2 700, 24 043 Events

| Modus | `solve()` Wall | Faktor |
|---|---:|---:|
| `"hash"` | **431 s** | 1,0× |
| `"hash_lazy"` (kein `reindex_all()`) | 169 s | 2,6× |
| `"scan"` | **55 s** | **7,8×** |
| Merger ganz aus | Lauf erstickt (n_comp 1 000 → 22 000+ in 20 s) | — |

`tottime`-Aufschlüsselung im Hash-Modus:

| Funktion | tottime | Aufrufe | Anteil |
|---|---:|---:|---:|
| `builtins.round` | 217 s | 18,6 Mio | **50 %** |
| `_compute_hash_key` | 87 s | 5,0 Mio (≈ 61 µs/Aufruf) | 20 % |
| `rebuild_r_pairdelta_pairwise` (O(n²)-EKE) | 62 s | 2 598 | 14 % |
| `reindex_particles` (Schleifenrumpf) | 34 s | 4 118 | 8 % |
| `_find_linear_scan` (Nucleation-Lookups) | 6 s | 39 489 | 1,5 % |

→ ~78 % der Laufzeit ist Hash-Index-Pflege.

Im `"scan"`-Modus: `round` / `_compute_hash_key` / `reindex_particles` = **0
Aufrufe**. `_find_linear_scan` übernimmt *alle* 40 715 Lookups für **4,0 s**
gesamt.

## 2. Warum O(1) hier gegen O(n) verliert

Die Intuition „Hash O(1) schlägt Scan O(n)" gilt für den **Lookup** — aber der
ist < 2 % der Kosten.

- **Ein `_find_via_hash`-Lookup**: 65 µs (echtes O(1), n-unabhängig).
- **Ein `_find_linear_scan`-Lookup**: ~100 µs bei n ≈ 2 700. Kein Python-Loop,
  sondern *ein* vektorisierter NumPy-Durchlauf: `np.abs(V_flat[-1,:n] - target)
  <= tol` verarbeitet alle n Elemente in einer SIMD-Operation (~3 µs Rechnung),
  der Rest ist fixer Aufruf-Overhead, den der Hash-Lookup auch zahlt.

Der Hash gewinnt beim Lookup also nur ~1,5×. Verloren wird an der **Pflege**:
`_compute_hash_key` sind ~10 skalare NumPy-Operationen mit Python-Dispatch;
`round(np.float64, 8)` ruft `np.float64.__round__` → intern `np.round` mit
0-d-Array → **11,7 µs** (ein `round(python_float, 8)` wäre ~0,05 µs — der
`np.log10`-Rückgabewert macht es ~200× teurer). Pro Partikel: `_compute_hash_key`
= 61 µs für **eines** gegen `_find_linear_scan` = ~100 µs für **alle 2 700**.

Der Hash-Index würde gewinnen bei **stabilen Keys** (reiner Agg/Bruch, keine
Kompression), bei **sehr großem n_comp** (> ~50 k, wo der O(1)-Lookup die
~100-µs-Fixkosten des Scans schlägt) oder wenn `_compute_hash_key`
**vektorisiert** wäre (alle Keys mit `np.round(np.log10(V_dry_array), 8)` auf
einmal statt skalar im Python-Loop). Keins davon trifft für die Nassgranulation
zu.

## 3. Verifikation: der Merger merged wirklich

Skepsis war: läuft der Merger nur „homöopathisch" mit? Drei unabhängige Checks
(keiner nutzt die Merger-eigenen Zähler):

1. **Unabhängiger Wrapper** um `find_or_create`: merges / creates exakt gleich
   wie `get_statistics()`.
2. **Spalten-Bilanz**: `n_comp_start + _append_particle_column − _remove_particle_column
   == n_comp_end` auf die Ganzzahl, in jedem Modus. Merger **an**: 2 340
   `_append_particle_column`-Aufrufe über den ganzen Lauf. Merger **aus**
   (gleicher Seed, nur bis t≈20 s): **39 976**. Der Merger hat ~37 600
   Spalten-Anlagen in reine `W += dW` verwandelt.
3. **Toleranz-Sweep**: `tol_rel` von 1e-5 auf 1e-10 (5 Größenordnungen) → n_comp
   2 723 → 2 729 (+6). Die gemergten Partikel sind bis ~10 Stellen identisch,
   keine verdeckte Vergröberung.

Der Merger ist szenarioabhängig: **97 %** der Lookups im Nassgranulations-Fall
kommen aus der Nucleation. Ein reiner Agg/Bruch-Lauf ohne Nucleation gibt dem
Merger nur ~1 200 Lookups über 24 000 Events — dort läuft er fast leer, was er
kostet, spielt keine Rolle.

## 4. `"scan"` vs `"hash"`: meist bitgleich, sonst statistisch äquivalent

Wenn **mehrere** Partikel innerhalb der Toleranz liegen, wählt
`_find_linear_scan` den **kleinsten Index**, `_find_via_hash` das, was die
Set-Iteration liefert. Wird ein *anderes* Partikel mit `dW` beschwert, divergiert
der RNG-Pfad ab da (Schmetterlingseffekt auf die Fenwick-Ziehung). Diese
Mehrfach-Treffer-Situation tritt in den meisten Läufen nicht auf — dann sind die
Modi bitgleich.

- Masse bleibt in jedem Modus **exakt** erhalten (`V_solid` wird addiert, nicht
  gemergt).
- Merge-Rate, n_comp, PSD-Momente stimmen auf **~1 %**.
- `"hash"` und `"hash_lazy"` sind untereinander **bitgleich** (hash_lazy lässt
  nur eine redundante Reindex-Runde weg).
- `"scan"` vs `"hash"`: für die meisten Konfigurationen bitgleich (8 von 10
  Benchmark-Szenarien, s. Tabelle unten). Divergenz nur bruch- oder
  lauflängenlastig, dann auf ~1–2 %.

`"scan"` ist damit auch die **deterministische** Wahl: sein Ergebnis hängt nicht
von der Dict/Set-Iterationsreihenfolge (und damit potenziell der Python-Version)
ab. Regressionswächter: `mcpbe/tests/test_merger_lookup_equivalence.py`.

### `"scan"` vs `"hash"` über die Benchmark-Szenarien (aktueller Code)

| Szenario | bitgleich | a_tot hash / scan |
|---|---|---|
| `agg_shear_1d` | ✓ | 120 / 120 |
| `agg_shear_1d_large` | ✓ | 1722 / 1722 |
| `agg_constant_1d` | ✓ | 85 / 85 |
| `agg_sum_1d` | ✓ | 119 / 119 |
| `agg_brownian_1d` | ✓ | 499 / 499 |
| `agg_shear_2d` | ✓ | 151 / 151 |
| `mix_1d` | ✓ | 444 / 444 |
| `granulation_1d` (t_end 4 s, 18 Events) | ✓ | 218 / 218 |
| `break_powerlaw_1d` | **✗** | 1897 / 1868 (−1,5 %) |
| `granulation_1d` (t_end 40 s, ~160 Events) | **✗** | 186 / 186, aber Events 160/158 |

8 von 10 sind bitgleich. Die Ausreißer sind bruch- bzw. lauflängenlastig: viele
fast identische Fragmente/Kinder → viele Mehrfach-Treffer → unterschiedliche
Wahl. `sum_W` und `solid_volume` bleiben in allen Fällen (praktisch) gleich.

## 5. Was mitgenommen wurde

- **Init-Order-Bug behoben:** `MCPBESolver(init=True)` baute den Merger im
  Konstruktor; `_ensure_particle_merger()` gab einen bestehenden unverändert
  zurück. Ein *nach* der Konstruktion gesetzter Flag (so machten es
  `framework/builder.py` und diverse Trials-Skripte) wurde ignoriert.
  `_ensure_particle_merger()` baut jetzt neu, wenn die Konfiguration abweicht —
  sicher, weil es nur aus Setup-Pfaden läuft, nie aus `solve()`.
- **Tote Flags:** `framework/builder.py` Config-Key `merger_use_hash_index`
  (jetzt `merger_lookup`) wirkte deshalb nie.
- **Tote Zeilen:** `Trials/test_particle_merger_comparison.py` und
  `test_particle_merger_rumpf_full.py` setzten `solver._use_merger_hash_index`
  (falscher Name) — auf `merger_lookup` umgestellt.

## 6. Umschalten

```python
solver.merger_lookup = "hash"        # vor _initialize_samplers()
```

`framework/`: Config-Key `"merger_lookup"` in den Params-Dict.
Benchmark: `solver_attrs={"merger_lookup": "hash"}` im `Scenario`.
Profiling: `PROFILE_MERGER=hash python -m wmcpbe.Trials.test_powerlaw_rumpf_dynamic_full_profiled`.
