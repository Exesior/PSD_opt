# WMCPBE Projekt – Fundamentals

Diese Datei ersetzt die vorherige Fundamentals.md (Stand 2025) mit einer ausführlicheren,
aus einer tiefen Code-Analyse (August 2026) gewonnenen Fassung. Ziel: Eine neue KI-Session
soll ohne erneute Vollexploration produktiv weiterarbeiten können.

## 0. Was ist WMCPBE?

Ein **gewichteter DSMC Monte-Carlo Population-Balance-Solver** zur Simulation von
**Nassgranulation** (wet granulation): trockene, poröse Primärpartikel werden mit
Flüssigkeitströpfchen benetzt (Nucleation), agglomerieren zu Granulaten, brechen ggf.
wieder auf (Breakage), und ihre Porosität/Sättigung entwickelt sich über die Zeit
(Porositätswachstum bei Kollision, Kompression, Flüssigkeits-Internalisierung).

- **Projekt-Root**: `C:\Users\ericb\Documents\GitHub\PSD_opt\`
- **Hauptpackage**: `C:\Users\ericb\Documents\GitHub\PSD_opt\mcpbe\`
- **Hauptcode (Fokus)**: `C:\Users\ericb\Documents\GitHub\PSD_opt\mcpbe\src\wmcpbe\`
- **Gemeinsame Basis**: `C:\Users\ericb\Documents\GitHub\PSD_opt\pbe-core\` (base_solver.py, jit_mcpbe.py)
- **Diese Docs**: `C:\Users\ericb\Documents\GitHub\PSD_opt\mcpbe\docs\` (projektweite Doku)
- **Modul-eigene Docs/Debug-Notizen**: `C:\Users\ericb\Documents\GitHub\PSD_opt\mcpbe\src\wmcpbe\docs\`
  (dort liegen u.a. die MASSE_*.md-Dokumente zur Masseerhaltungs-Debugsession)
- **Tests/Trials**: `C:\Users\ericb\Documents\GitHub\PSD_opt\mcpbe\src\wmcpbe\Trials\`
- **Debug-Logs von Testläufen**: `...\Trials\debug_logs\`

Es gibt auch `wmcpbe_granulation/` (granulationsspezifische Erweiterungen, nicht im
Fokus dieser Doku) und `mcpbe/` (nicht-gewichteter Solver, älter/einfacher).

**Environment zum Ausführen**: `C:\Users\ericb\anaconda3\envs\Masterarbeitv4\python.exe`
(system-`python` ist NICHT verfügbar; kein PATH-Eintrag, nur der MS-Store-Stub).
Ausführen z.B. so (aus `mcpbe/src`):
```
"C:\Users\ericb\anaconda3\envs\Masterarbeitv4\python.exe" -m wmcpbe.Trials.test_powerlaw_rumpf_full
```
Für saubere Unicode-Ausgabe (Δ, ³, etc.) in der Konsole: `PYTHONIOENCODING=utf-8` setzen,
sonst crasht `print()` auf Windows-cp1252-Konsolen mit `UnicodeEncodeError`.

---

## 1. Zentrale Architektur

### 1.1 Einstiegspunkt

Der Solver wird zusammengesetzt in **[mcpbe.py](../src/wmcpbe/mcpbe.py)**:

```python
class MCPBESolver(MCPBEPost, MCPBEBreak, MCPBEAgg, MCPBEBase, ReconstructionMixin):
```

MRO (Method Resolution Order), von "zuerst gefragt" zu "zuletzt":
**Post → Break → Agg → Base → Reconstruction**. Post-Processing muss zuletzt in der
Vererbungskette stehen (braucht Zugriff auf alles); Base enthält die fundamentale
Initialisierung und den Haupt-Solve-Loop.

`MCPBESolver` bietet außerdem `create_nucleation_handler(**kwargs)` und
`create_continuous_processes_handler(**kwargs)` als Factory-Methoden für die beiden
Handler-Objekte (Komposition statt Vererbung, siehe 1.3).

### 1.2 Datei-für-Datei-Überblick (`mcpbe/src/wmcpbe/`)

| Datei | Rolle |
|---|---|
| `mcpbe.py` | Zusammenbau der `MCPBESolver`-Klasse, Handler-Factories |
| `mcpbe_base.py` | Partikelarrays (`V_flat`, `W`, `X`, `porosity`, `liquid_volume`, `saturation`), Control-Volume-Management, **Haupt-Solve-Loop**, `_append_particle_column`, `_remove_particle_column`, `_initialize_particles`, `_initialize_samplers` |
| `mcpbe_agg.py` | Agglomerations-Physik: `_do_one_agg`, `_select_pair`, `_compute_agg_dW`, `_merge_pair`, `_consume_parent_weight`, `_refresh_samplers_after_agg`, Propensity-Rebuild (pairwise/moment) |
| `mcpbe_break.py` | Breakage-Physik: `_do_one_break`, Rate-Berechnung, Fragment-CDF/LMC-Sampling |
| `mcpbe_post.py` | Momente (µ(i,j,t)), PSD/CDF-Auswertung |
| `mcpbe_nucleation.py` | `NucleationHandler` + `NucleationConfig` (Flüssigkeitszugabe, siehe 3.1) |
| `mcpbe_continuous_processes.py` | `ContinuousProcessesHandler` + Config (Kompression + Liquid-Internalisierung, siehe 3.2). **Ersetzt das alte, inzwischen gelöschte `mcpbe_compression.py`.** |
| `reconstruction_mixin.py` | CAM/RS/2PM/QMX-Rekonstruktion (Partikelzahl-Reduktion bei > `recon_N_max`) |
| `particle_merger.py` | **`ParticleMerger`** – zentrale Dedup-Logik für neu erzeugte Partikel (Agglomeration + Breakage), s. Abschnitt 4 |
| `kernel_integration.py` | `KernelManager` – hält/dispatcht alle Physik-Kernel, s. Abschnitt 5 |
| `fenwick_new.py` | `FenwickSampler` – O(log n) gewichtetes Sampling (Binary Indexed Tree, Numba-JIT) |
| `mcpbe_time_helper.py` | Zeitschritt-/Event-Zeitmanagement |
| `lmc_adapter.py` | Lattice-Monte-Carlo-Adapter für Fragmentverteilungen (Breakage) |
| `mlp_breakage_adapter.py` | MLP-Modell für Breakage-Raten (ML-Alternative zu Kernel-Formeln) |
| `helpers.py` | Convenience-Funktionen: `setup_initial_particles`, `compute_moments`, `validate_mass_conservation` (die `create_*_solver`-Templates wurden am 20.08.2026 entfernt) |
| `kernels/` | Modulare Physik-Kernel, s. Abschnitt 5 |
| `Trials/` | Tests, Debug-Skripte, `debug_logs/` |
| `docs/` | Modul-eigene Doku inkl. Masseerhaltungs-Debug-Historie (MASSE_*.md) |

**Wichtig**: `mcpbe_compression.py` und `kernels/compression/` **existieren nicht mehr**
(im Zuge eines Refactors gelöscht, siehe `git log` / `git status` auf Branch `dev_eric`).
Kompression läuft ausschließlich über `ContinuousProcessesHandler` +
`kernels/continuous_processes/porosity_compression.py`. Falls alte Dokumente oder
Kommentare noch `CompressionHandler` erwähnen: veraltet, ignorieren.

### 1.3 Mixins vs. Handler (wichtig für Verständnis!)

- **Mixins** (`MCPBEAgg`, `MCPBEBreak`, `MCPBEPost`, `ReconstructionMixin`) sind über
  Mehrfachvererbung Teil von `MCPBESolver` selbst – ihre Methoden greifen direkt auf
  `self.V_flat`, `self.W` etc. zu und werden **innerhalb** der Haupt-Event-Loop
  (`solve()` in `mcpbe_base.py`) aufgerufen, wenn ein Agg-/Break-Event gewürfelt wird.
- **Handler** (`NucleationHandler`, `ContinuousProcessesHandler`) sind **komponierte**,
  eigenständige Objekte (`solver.nucleation`, `solver.continuous_processes`), die eine
  Referenz auf den Solver halten (`self.solver`) und **zusätzlich** nach jedem
  MC-Event via `.step(current_time, dt_event)` aufgerufen werden – sie laufen auf einer
  eigenen Zeitskala, nicht als "gewürfeltes" Event.

---

## 2. Kern-Datenmodell (KRITISCH für Masseerhaltung!)

### 2.1 `V_flat` – die wichtigste Datenstruktur

Shape `(dim+1, _cap)`. Für `dim=1` (Standardfall, monodispers bzgl. Komponenten):

```
V_flat[0, idx]  = V_solid   ->  KONSERVIERTE Größe! Ändert sich NIE außer durch
                                 tatsächliche Massenerzeugung/-vernichtung (die es
                                 physikalisch nicht geben darf).
V_flat[-1, idx] = V_dry     ->  V_solid + V_pore. KEINE Erhaltungsgröße – ändert
                                 sich zulässig mit der Porosität (Kompression,
                                 Porenwachstum bei Agglomeration, etc.)
porosity[idx]   = ε = V_pore / V_dry     (NaN = "Vollkörper", legacy; moderne
                                           Kernel liefern 0.0 statt NaN)
V_solid = V_dry * (1 - ε)   für NaN: V_solid = V_dry (kein Porenraum)
```

**Zwei äquivalente Darstellungen von V_solid existieren im Code:**
1. Direkt gespeichert: `V_flat[:dim, idx]` (bzw. `V_flat[0, idx]` bei dim=1)
2. Abgeleitet: `V_flat[-1, idx] * (1 - porosity[idx])`

Empirisch verifiziert (August 2026, per Diagnose-Skript über eine volle 100s-Simulation):
**Beide stimmen im aktuellen Code immer exakt überein** (Differenz < 1e-27 m³ bei
~2300 Partikeln) – d.h. es gibt hier aktuell KEINE Divergenz, obwohl die doppelte
Datenhaltung architektonisch fragil ist und bei künftigen Änderungen sorgfältig
synchron gehalten werden muss. Jede Stelle, die einen neuen Partikel erzeugt
(`_append_particle_column`, `_create_new_particle` in particle_merger.py,
`_create_or_update_nucleated_particle` in mcpbe_nucleation.py), MUSS beide
Repräsentationen konsistent setzen.

### 2.2 Weitere Partikel-Arrays (alle Länge `_cap`, aktiv sind Indizes `[0, a_tot)`)

| Array | Typ | Bedeutung |
|---|---|---|
| `W[idx]` | extensiv | Computational Weight = Anzahl physischer Partikel, die idx repräsentiert |
| `X[idx]` | intensiv | Durchmesser, abgeleitet aus `V_flat[-1, idx]` (V_dry, NICHT V_solid!) |
| `liquid_volume[idx]` | intensiv | Gesamtflüssigkeit pro physischem Partikel [m³] |
| `saturation[idx]` | intensiv | S = V_liq_intern / V_pore, ∈ [0,1] |

**Intensiv vs. extensiv**: Intensive Größen sind "pro physischem Partikel" und bleiben
bei Vererbung/Vc-Verdopplung/Kopieren unverändert. `W` ist die einzige extensive
Größe unter diesen. **Gesamtmasse eines Zustands = Σ (V_solid[idx] · W[idx])** über
alle aktiven `idx`.

### 2.3 Control Volume / DSMC

`Vc` (Control Volume) skaliert so, dass `Σ(W)/Vc` = physikalische Partikeldichte
konstant bleibt. `_maybe_double_control_volume()` in `mcpbe_base.py` verdoppelt `Vc`
und dupliziert Partikel (W wird kopiert, NICHT skaliert – siehe Fundamentals-Beispiel
im alten Dokument), wenn Partikelzahl unter 50% der Kapazität fällt.
`n_phys = Σ(W)/Vc`, `n_comp = a_tot`.

### 2.4 Partikel-Array-Verwaltung (Wachstum/Entfernung)

- **`_append_particle_column(frag_vols)`** (`mcpbe_base.py`): Hängt EIN neues Partikel
  an (Index `a_tot`, danach `a_tot += 1`). `frag_vols` sind die `dim`
  Solid-Komponenten. Setzt `V_flat[-1, idx]` initial auf `sum(frag_vols)` – der
  Aufrufer MUSS das danach i.d.R. mit dem echten `V_dry` überschreiben (Poren!).
- **`_remove_particle_column(j)`** (`mcpbe_base.py`): Entfernt Partikel `j` per
  **Swap-with-last**: Partikel an Index `a_tot-1` wird nach `j` kopiert, `a_tot -= 1`,
  freigewordener Slot genullt. **Alle Indizes ≥ `j` können sich dadurch verschieben**
  – Code, der einen Index über einen `_remove_particle_column`-Aufruf hinweg
  "festhält" (z.B. einen frisch erzeugten Kind-Partikel-Index), muss das aktiv tracken
  (siehe `child_current_idx`-Pattern in `mcpbe_nucleation.py::_manual_agglomerate_particles`,
  und `notify_index_swap` in `particle_merger.py`, s. Abschnitt 4).

---

## 3. Prozesse

### 3.1 Agglomeration (`mcpbe_agg.py`)

> **Stand 14.08.2026 — Bias-Correction umgesetzt.** Propensity und Partnerauswahl folgen
> jetzt der Pair-Delta-korrigierten Form aus Ji & Rhein (Gl. 33/36–41). Vollständige
> Herleitung, Verifikationsprotokoll und Umbaubeschreibung:
> [`Bias_Correction_und_Gewichtsdisziplin.md`](Bias_Correction_und_Gewichtsdisziplin.md).
> Das Wichtigste in Kürze:
> - `_r_agg[i]` hält jetzt `R_i* = W_i·Σ_j W_j·β(i,j)/min(ΔW_i,ΔW_j)`, **nicht** mehr
>   `Σ_j W_j·β(i,j)/ΔW_i`. Der `W_i`-Vorfaktor und die *paarweise* Division sind neu.
> - Partner `j` wird proportional zu `W_j·β(i,j)/ΔW_ij` gezogen, nicht mehr zu `W_j`.
> - `W_MIN_ACTIVE` existiert nicht mehr; das Löschkriterium ist wieder `W > 0.0`.

Ablauf pro Event (`_do_one_agg`):
1. `_select_pair(a)`: Zweistufiger Sampler — Partikel `i` via Fenwick-Sampler
   proportional zu `R_i*`, dann Partner `j` bedingt proportional zu
   `W_j·β(i,j)/min(ΔW_i,ΔW_j)`. Danach Größen-Akzeptanz (SIZEEVAL, **eigene**
   Zufallszahl) und physikalisches Akzeptanzkriterium
   (`agglomeration_acceptance_kernel`, z.B. Stokes).
   Bei Ablehnung: `None` zurück, Event ist ein No-Op.
2. `_compute_agg_dW(...)`: Paketgröße `dW` (wie viele physische Kollisionen dieses
   Event repräsentiert). **Selbstkollision (`i==j`) ist explizit erlaubt** und wird
   speziell behandelt: `dW` wird auf `δ_ii = min(δ_i, W_i/2)` **gekappt** (früher
   wurde der Event stattdessen abgelehnt). Dadurch gilt `2*dW <= W[i]` mit Gleichheit
   im Grenzfall, und der letzte Event räumt das Partikel exakt auf `0.0`.
3. `_merge_pair(i, j, dW)`: Erzeugt Kind-Partikel via `ParticleMerger.find_or_create`
   (Dedup, s. Abschnitt 4). `V_solid_child = V_solid_i + V_solid_j` (exakt, unabhängig
   vom Porositäts-Kernel). `V_dry_child`/`poro_child` kommen vom
   `porosity_growth_kernel`.
4. `_consume_parent_weight(i, j, dW)`: `W[i] -= dW`, `W[j] -= dW` (bzw. `W[i] -= 2*dW`
   bei `i==j`). Partikel mit `W <= 0` werden entfernt.
5. `_refresh_samplers_after_agg()`: Propensities neu berechnen (O(n) im
   `agg_propensity_mode = "moment"`, empfohlen; O(n²) im `"pairwise"`-Modus, nur für
   Referenz/kleine Systeme).

> **`SIZEEVAL` existiert, ist für WMCPBE-Nassgranulation aber nicht relevant
> (Stand 18.08.2026, Ruecksprache mit dem Betreuer).** Der Schalter steuert eine
> zusaetzliche, empirische Groessenabhaengigkeit der Kollisionseffizienz α und
> wurde vom Betreuer fuer Trockenagglomeration geschrieben, wo er eine
> Gleichgewichtsgroesse zwischen Wachstum und Scherabbruch modelliert (Ersatz
> fuer einen fehlenden expliziten Bruchmechanismus). WMCPBE hat mit
> `powerlaw_rumpf` (Bruch) und `stokes_krit` (Haftkriterium, Braumann 2007)
> bereits beide Effekte physikalisch abgedeckt -- der Filter ist hier ueberfluessig.
>
> **Zusaetzlich weicht die WMCPBE-Implementierung vom Projektstandard ab:**
> `pbe-core`/`dpbe`/`mcpbe` pruefen `SIZEEVAL == 1` (aus, Default) vs.
> `SIZEEVAL == 2` (Soos2007-Modell). `mcpbe_agg.py::_accept_sizeeval` prueft
> stattdessen `SIZEEVAL == 0`, mit einer Gauß-Glocke ueber dem Partikelvolumen
> statt der Soos2007-Formel -- der Filter ist damit bei `SIZEEVAL = 1` (dem
> Default!) unbeabsichtigt AKTIV. **Abschalten: `solver.SIZEEVAL = 0`** (nicht
> `1`, wie die Konvention nahelegen wuerde). `Trials/test_powerlaw_rumpf_full.py`
> setzt das seit dem 18.08.2026 explizit. Vollstaendige Herleitung inkl.
> Formelvergleich und gemessener Wirkung:
> [`Audit_2026-08-17.md`, Befund B-03](../docs/Audit_2026-08-17.md#b-03).

### 3.2 Breakage (`mcpbe_break.py`)

`_do_one_break()`: Partikel `k` proportional zu `W_k · S_k` wählen (`S` = Breakage-Rate
vom `break_kernel`, z.B. `power_law` oder `powerlaw_rumpf` – letzterer
porositäts-/sättigungsabhängig nach Rumpf-Festigkeitstheorie). Fragmentanzahl
stochastisch runden, Fragmente via CDF-Tabellen/LMC/analytisch samplen, je Fragment
`ParticleMerger.find_or_create` aufrufen, Elterngewicht um `dW` reduzieren.

### 3.3 Nucleation (`mcpbe_nucleation.py`, `NucleationHandler`)

Fügt Flüssigkeit als diskrete Tropfen hinzu, `.step()` wird nach jedem MC-Event
aufgerufen. Kern-Methoden:

- **`_distribute_liquid_volume(v_liquid_to_distribute)`**: Akkumuliert Volumen in
  `_liquid_remainder` (verhindert systematischen Verlust durch Integer-Diskretisierung
  der Tropfenzahl), verteilt ganze Tropfen über `_distribute_one_droplet_with_dW`, Rest
  als ein "Mini-Tropfen".
- **`_distribute_one_droplet_with_dW(v_droplet, ...)`**: Wählt Partikel `i` gewichtet
  nach `W` (uniform über physische Partikel). Wenn `V_dry[i] < v_liquid_neu`
  (Partikel kann die Flüssigkeit nicht aufnehmen), wird **manuell agglomeriert**
  (`_perform_nucleation_agglomeration` → `_manual_agglomerate_particles`), bis genug
  `V_dry` gesammelt ist. Danach: Porosität via `porosity_growth_kernel.
  compute_nucleation_porosity(v_solid, v_liquid, ...)`, Sättigung berechnen, entweder
  in existierendes ähnliches Partikel mergen (`_find_similar_particle`, **eigene,
  von `ParticleMerger` unabhängige Implementierung!** Toleranz
  `NucleationConfig.similarity_tol`, Default `1e-5`) oder neues Partikel via
  `_create_or_update_nucleated_particle` anlegen.
- **`_manual_agglomerate_particles(i, j)`**: Eigene Kopie der Agglomerations-Logik
  (NICHT `mcpbe_agg.py::_merge_pair`!), weil hier ohne Propensity-Rebuild gearbeitet
  werden muss (würde bei jedem Tropfen O(n²) kosten). Muss denselben Regeln folgen wie
  `mcpbe_agg.py` (insbesondere: `i==j` → `dW` auf `Wi/2` begrenzen – **hier lag am
  10.–11.08.2026 ein Bug**, siehe Abschnitt 7).

> ⚠ **Der ausgelassene Rebuild war bis 17.08.2026 unkompensiert.** Ihn *innerhalb* der
> Tropfenschleife wegzulassen ist richtig — nur hat ihn danach auch niemand auf
> Ereignisebene nachgeholt. Neu angelegte Partikel gingen deshalb mit Propensity 0 in
> `_r_agg`/`_break_rate` und konnten weder agglomerieren noch brechen. Bei
> `INITIAL_POROSITY = 0` wurde daraus eine Selbstblockade (kein Bruch ⇒ kein Refresh ⇒
> keine Agglomeration ⇒ kein Refresh), die 13 von 14 Parametervarianten auf exakt null
> Agglomerationen festnagelte. `solve()` frischt jetzt einmal pro MC-Event auf, und nur
> wenn die Nucleation tatsächlich etwas verändert hat — also O(n) pro Ereignis statt
> pro Tropfen. Details:
> [`Nucleation_Propensity_Blockade.md`](Nucleation_Propensity_Blockade.md).

**DSMC-Skalierung bei Nucleation**: `effective_dW = dW * (Vc_ref/Vc)` –
physikalische Tropfenzahl bleibt korrekt, auch wenn sich `Vc` durch
Vc-Doubling ändert.

### 3.4 Continuous Processes (`mcpbe_continuous_processes.py`, `ContinuousProcessesHandler`)

Pro `.step(current_time, dt_event)`:
1. **Liquid Internalization** (`_apply_internalization`): kapillargetriebenes
   Eindringen von externer in interne (Poren-)Flüssigkeit, Braumann et al. (2007):
   `dl_intern/dt = k_int · l_extern · (V_pore - l_intern)`.
2. **Porosity Compression** (`_apply_compression`): exponentieller Porositätsabfall
   `ε(t) = ε_min + (ε_0-ε_min)·exp(-rate·t)`. **Ändert `V_dry` und `porosity`, hält
   `V_solid` exakt konstant** (verifiziert: ΔV_solid = 0.0 exakt bei jedem Event, kein
   Rundungsfehler beobachtet). Externalisiert überschüssige Flüssigkeit, wenn `S > 1`.

> ⚠ **Bis 17.08.2026 war dieser Schritt wirkungslos.** `PorosityCompressionKernel.
> compute_array` schrieb sein Ergebnis über eine verkettete Zuweisung
> `out[finite][can_compress] = updated` — Boolean-Indizierung liefert in NumPy eine
> Kopie, die Zuweisung landete also in einem Wegwerf-Array. Der Handler nutzt den
> Kernel, sobald einer registriert ist (der korrekte analytische Fallback darunter wird
> dann nie erreicht), womit `COMPRESSION_RATE` und `MIN_POROSITY` in **jedem** Lauf mit
> registriertem Kompressions-Kernel folgenlos blieben. Details, Messung und Konsequenzen:
> [`Kompression_und_Parametervalidierung.md`](Kompression_und_Parametervalidierung.md).

---

## 4. `ParticleMerger` (`particle_merger.py`) – Dedup-Mechanismus

**Zweck** (Nutzer-Zitat, wichtig für Designverständnis): *"Der Merger war so gedacht,
dass keine redundanten Child-Partikel entstehen. Wenn Children entstehen, werden diese
einmal gegen die bereits existierenden Partikel getestet. Falls eines identisch ist,
wird kein neues n_comp erstellt, sondern das W dem bereits existierenden n_comp
zugewiesen."*

- **`find_or_create(V_solid_target, V_dry_target, liquid_target, poro_target,
  sat_target, weight_to_add, component_sum)`** → `(idx, was_merged)`. Wird von
  `mcpbe_agg.py::_merge_pair` und `mcpbe_break.py` (Fragment-Erzeugung) genutzt.
  **Nicht** von `mcpbe_nucleation.py` – die hat ihre eigene, separate
  `_find_similar_particle`-Implementierung (Code-Duplikation, s.o.).
- Matching ist **toleranzbasiert**, nicht exakt: `tol_rel` (Default `1e-6`) relativ auf
  V_dry/porosity/saturation, `tol_abs_liquid` (Default `1e-30`) absolut auf
  liquid_volume. Bei Match: **nur `W[idx] += weight_to_add`**, alle intensiven
  Properties bleiben unverändert (Designannahme: Ziel- und Bestandspartikel sind
  "nah genug" identisch).
- **Hash-Index** (`_hash_index: Dict[tuple, Set[int]]`) für O(1)-Lookup, binned auf
  log-Skala (`bin_digits_volume=8`) für Volumina, linear (`bin_digits_poro=4`) für
  Porosität/Sättigung. `rebuild_hash_index()` wird nur nach Rekonstruktion
  (CAM/RS/2PM/QMX) aufgerufen – sonst inkrementell gepflegt.
- **`notify_index_swap(old_idx, new_idx)`** (von mir am 11.08.2026 ergänzt, s.
  Abschnitt 7): wird zentral aus `mcpbe_base.py::_remove_particle_column` aufgerufen,
  BEVOR der Array-Swap passiert, und hält den Hash-Index synchron, wenn Swap-with-last
  ein Partikel auf einen neuen Index verschiebt. Ohne das würde der Hash-Index nach
  vielen Entfernungen auf falsche/wiederverwendete Slots zeigen.

---

## 5. Kernel-Framework (`kernel_integration.py` + `kernels/`)

`KernelManager` hält 8 Kernel-Slots (alle optional, `None` wenn nicht konfiguriert):

| Slot | Zweck | Verzeichnis | Beispiel-Implementierungen |
|---|---|---|---|
| `agg_kernel` | Kollisionsrate β(r1,r2) | `kernels/aggregation/` | `shear_chin1998`, `brownian_tsouris1995`, `constant`, `sum`, `eke_darelius2005`, `etm_darelius2005` |
| `break_kernel` | Breakage-Rate S(V) | `kernels/breakage/` | `power_law`, `powerlaw_rumpf` |
| `porosity_growth_kernel` | Porosität bei Merge/Nucleation/Breakage | `kernels/porosity_growth/` | `volume_mixing` (additiv, kein Pore-Collapse), `cone_model` (geometrisch, Kegelpillen-Modell), `incomplete_mixing` |
| `agglomeration_acceptance_kernel` | Physikalische Akzeptanz (Kollision "klebt"?) | `kernels/agglomeration_acceptance/` | `stokes_krit` (Braumann 2007), `fittable` |
| `liquid_dist_kernel` | Partikelauswahl für Nucleation | `kernels/liquid_distribution/` | `uniform_weighted` |
| `porosity_compression_kernel` | Kontinuierliche Kompression | `kernels/continuous_processes/` | `porosity_compression` |
| `liquid_internalization_kernel` | Kontinuierliche Internalisierung | `kernels/continuous_processes/` | `liquid_internalization` |
| `liq_internalisation_agglomeration_kernel` | Event-basierte Internalisierung bei Kollision | `kernels/continuous_processes/` | `liq_internalisation_agglomeration` |

Jeder Kernel-Typ hat eine `blueprint.py` (abstrakte Basisklasse/Contract) im jeweiligen
Unterverzeichnis. Kernel werden per Name + Params-Dict beim `MCPBESolver(...)`-
Konstruktor ausgewählt (`agg_kernel_name=...`, `agg_kernel_params={...}`, usw.).

**Parameternamen werden geprüft (seit 17.08.2026).** Jede `get_*_kernel(...)`-Factory
ruft `kernels/base.py::reject_unknown_params` und wirft einen `ValueError`, wenn ein
übergebener Name von diesem Kernel gar nicht gelesen wird — mit Vorschlag des
vermutlich gemeinten Namens. Vorher wurde ein Tippfehler stillschweigend geschluckt
(`get_default_params()` aktualisiert mit den Aufrufer-Werten: ein unbekannter Schlüssel
überschreibt nichts, er kommt nur zusätzlich ins Dict) und der Kernel lief auf seinem
Default weiter — genau so blieb `k_intern` statt `k_int` über mehrere Trial-Skripte
hinweg unbemerkt. Kernel mit optionalen, absichtlich default-losen Parametern
deklarieren diese in `get_optional_params()`; derzeit nur `cone_model`
(`default_porosity`, `liquid_split_ratio`). Siehe
[`Kompression_und_Parametervalidierung.md`](Kompression_und_Parametervalidierung.md).

### 5.1 Porositäts-Kernel im Detail (wichtig für Masse-Invarianten)

Beide Haupt-Kernel (`volume_mixing.py`, `cone_model.py`) implementieren dieselbe
Schnittstelle:
- `compute_merged_porosity(v_dry1, poro1, v_dry2, poro2, ...) -> (V_dry_neu, poro_neu)`:
  zerlegt Eltern in V_solid/V_pore, summiert V_solid **exakt**, berechnet neues
  V_pore (additiv bei `volume_mixing`; additiv + geometrisches ΔV bei `cone_model`,
  Kegelstumpf-Formel aus Kontaktgeometrie zweier Kugeln), rekombiniert zu V_dry_neu.
  Beide Kernel garantieren `V_dry_neu*(1-poro_neu) == V_solid_1+V_solid_2` exakt
  (abgesehen von Clamping in Extremfällen, das aber nicht als Ursache eines
  Masse-Bugs bestätigt wurde).
- `compute_nucleation_porosity(v_solid, v_liquid, ...) -> (V_dry, poro)`: `volume_mixing`
  startet neue Tröpfchen-Partikel mit `poro=0.4` (hartcodiert, damit überhaupt
  Porosität entstehen kann); `cone_model` startet mit `poro=0.0` (Porosität entsteht
  erst durch nachfolgende Agglomeration).
- `compute_fragment_porosity(...)`: Breakage, `cone_model` reduziert Porenvolumen um
  `k_break · ΣΔV` (Oberflächenzuwachs durch neue Bruchflächen).

---

## 6. Sonstige wichtige Bausteine

- **`FenwickSampler`** (`fenwick_new.py`): Binary Indexed Tree, Numba-JIT,
  `sample(rng)` (O(log n)), `update(idx, w)`, `append(w)`, `remove(idx)`,
  `total()`. Wird für Agglomerations-Propensities, Breakage-Raten und
  Nucleation-Gewichtsauswahl genutzt.
- **`ReconstructionMixin`**: Reduziert Partikelzahl bei `a_tot > recon_N_max`
  (Default-Trigger 4000) durch CAM (Cell Average Method) / RS (Resampling mit
  C-1-Korrektur) / 2PM (Two-Point-Methode) / QMX (Quantile-Mix, hybrid). Ruft danach
  `particle_merger.rebuild_hash_index()`.
- **`agg_propensity_mode`**: `"moment"` (O(n), Default/empfohlen, ~1e-15 Abweichung
  von `"pairwise"`) vs. `"pairwise"` (O(n²), bitgenau, nur Referenz/kleine Systeme).
  Details/Herleitung in `docs/MOMENT_MODE.md` (falls vorhanden).
- **Debug-Flags** (siehe auch Abschnitt 7 zu Vorsicht!): `solver.mcpbe_debug_mass`,
  `solver.mcpbe_debug_agg/_break/_merger/_nuc/_comp/_vc`, `solver._debug_max_events`
  (Default 20). Aktivieren detaillierte `print()`-Ausgaben an vielen Stellen im Code
  (mcpbe_agg.py, mcpbe_break.py, mcpbe_nucleation.py, mcpbe_continuous_processes.py,
  particle_merger.py, mcpbe_base.py).

---

## 7. Bekannte Fallstricke / Lessons Learned (Stand 11.08.2026)

Diese Erkenntnisse stammen aus einer intensiven Masseerhaltungs-Debug-Session und
sind als **Warnsignale für ähnliche zukünftige Bugs** gedacht – nicht als
Status-Update zum aktuellen Bugfix-Stand (siehe dafür `git log` und
`mcpbe/src/wmcpbe/docs/MASSE_*.md`).

1. **Debug-Print-Blöcke sind eine wiederkehrende Bug-Quelle.** In diesem Codebase
   wurden mehrfach Debug-Print-Blöcke (gated hinter `mcpbe_debug_mass`,
   `_debug_agg_count < N`, o.ä.) gefunden, die:
   - auf undefinierte Variablen zugriffen (`NameError`),
   - dupliziert vorlagen (zwei fast identische Blöcke hintereinander),
   - über einen **kaputten Zähler** (`_debug_agg_count`, der nur unter einer ANDEREN
     Debug-Flag inkrementiert wurde) fälschlich **immer** (nicht nur "erste 10 Events")
     auslösten und dabei mengenkritische Logik (Gewichtsreduktion) ein zweites Mal
     ausführten – ein Faktor-2-Massefehler bei jedem Agglomerationsereignis.
   **Lektion**: Bei Verdacht auf Masse-/Zählfehler IMMER prüfen, ob ein
   `if <debug_flag>: ... else: <eigentliche Logik>`-Muster sauber exklusiv ist
   (kein Fallthrough, keine doppelte Ausführung, kein toter/immer-wahrer Zweig).
   Debug-Code sollte NIEMALS Zustand mutieren, der nicht auch ohne die Flag mutiert
   würde – reine Seiteneffekt-freie Prints sind sicher, alles andere ist riskant.

2. **`i == j` (Selbstkollision) braucht überall dieselbe Sonderbehandlung**: `dW` muss
   auf `δ_ii = min(δ_i, W_i/2)` gekappt werden (ein Self-Collision-Event verbraucht
   zwei physische Partikel aus demselben Pool). `mcpbe_agg.py::_compute_agg_dW` und
   `mcpbe_nucleation.py::_manual_agglomerate_particles` sind zwei **separate
   Code-Kopien** derselben Idee und sind historisch mehrfach auseinandergelaufen.
   Seit 14.08.2026 sind beide auf `min(δ_i, 0.5*W_i)` bzw. `min(δ_i, δ_j)`
   angeglichen — bei Änderungen weiterhin **beide** anfassen.

3. **Swap-with-last-Entfernung (`_remove_particle_column`) verschiebt Indizes.** Jede
   Datenstruktur, die Partikel-Indizes "von außen" referenziert (Hash-Index in
   `particle_merger.py`, ein frisch erzeugter Kind-Index, der über mehrere
   Eltern-Entfernungen hinweg gebraucht wird), muss das aktiv nachverfolgen. Suche nach
   `child_current_idx`-Mustern als Vorbild.

4. **Es gibt zwei unabhängige "Finde-ähnliches-Partikel"-Implementierungen**
   (`particle_merger.py::ParticleMerger` für Agg/Break, `mcpbe_nucleation.py::
   NucleationHandler._find_similar_particle` für Nucleation). Bugfixes/Verbesserungen
   in einer Implementierung (z.B. `notify_index_swap`) werden NICHT automatisch auf die
   andere übertragen – beide Stellen im Auge behalten.

5. **Diagnose-Strategie, die sich bewährt hat**: Statt sich auf die eingebauten
   `mcpbe_debug_mass`-Prints zu verlassen (die oft durch `_debug_max_events` gekappt
   sind und damit nur die ersten N Events zeigen), lohnt sich ein externes,
   schreibgeschütztes Monkeypatch-Diagnose-Skript (Python, `sys.path.insert` auf
   `mcpbe/src`, Methoden der Zielklasse durch eine instrumentierte Wrapper-Funktion
   ersetzen, die vor/nach dem Original-Call die globale Masse
   `Σ(V_solid·W)` misst). Damit lässt sich der Fehler unkappt über die GESAMTE
   Simulation kategorienweise attribuieren (AGG/BREAK/NUC/COMP/VC-DOUBLE) und bis auf
   ein einzelnes Event herunterbrechen. Kein Repo-File wird dabei verändert.

6. **Nützliche Test-Konfiguration für Masse-Checks**: `Trials/test_powerlaw_rumpf_full.py`
   (2000 Partikel, 34µm, 80% Porosität, `cone_model`, `stokes_krit`, `powerlaw_rumpf`,
   volle Physik-Suite) plus `Trials/test_droplet_10um_stable.py` /
   `test_droplet_20um_instable.py` (identische Physik, nur Tropfengröße 10 vs. 20µm,
   mit `mcpbe_debug_mass=True` vorkonfiguriert – Vorsicht, das aktiviert u.U. sehr
   viel Print-Output und kann bei Bugs im Debug-Code selbst crashen, s. Punkt 1).
   `Trials/test_runner_mass_debug.py` regeneriert diese beiden Dateien automatisch aus
   `test_powerlaw_rumpf_full.py` (Template-Mechanismus, Regex-Ersetzung von
   `DROPLET_DIAMETER`) und schreibt Logs nach `Trials/debug_logs/`.

---

## 8. Glossar (Kurzreferenz)

| Begriff | Bedeutung |
|---|---|
| DSMC | Direct Simulation Monte Carlo |
| PBE | Population Balance Equation |
| PSD | Particle Size Distribution |
| Vc | Control Volume |
| W | Computational Weight (N_physical / N_computational) |
| n_comp / n_phys | Anzahl computational- bzw. physikalischer Partikel |
| V_solid | Feststoffvolumen – **Erhaltungsgröße** |
| V_dry | V_solid + V_pore – **keine** Erhaltungsgröße |
| dW | Paketgröße eines MC-Events (wie viele physische Partikel es repräsentiert) |
| Vollkörper | Partikel ohne Poren, historisch `porosity=NaN`, modern `porosity=0.0` |
| a_tot | Anzahl aktiver computational particles (Index-Grenze in allen Arrays) |
| _cap | Kapazität der vorallokierten Arrays (≥ a_tot, wächst dynamisch) |

---

*Erstellt: 2026-08-11, nach einer Debug-Session zur Masseerhaltung. Ersetzt die Version
von 2025. Für den konkreten Bugfix-Verlauf siehe Git-Historie und
`mcpbe/src/wmcpbe/docs/MASSE_*.md`.*
