# Analyse: Agglomerations- vs. Bruch-Kernel zwischen `dev_eric` und `dev_monorepo`

> ## ✅ STATUS 16.08.2026 — BEHOBEN
>
> Die Analyse unten wurde verifiziert und ist im Kern korrekt. Der Fund ist inzwischen
> behoben; dieses Dokument bleibt als Befund- und Ursachenprotokoll bestehen.
>
> **Ursache (in der Analyse noch offen): Verwechslung von `BREAKFVAL` mit `BREAKRVAL`.**
> `pl_v`/`pl_q` sind im Referenzcode die Parameter der *Fragmentgrößen-Verteilung*
> (`breakage_func_1d(x, y, v, q, BREAKFVAL)`), nicht der Rate. In `power_law.py`
> standen sie mit dem Kommentar *"Volume exponent for BREAKFVAL=4"* bzw.
> *"Exponent for BREAKFVAL=3 (MUST BE 1.0 FOR STABILITY!)"* — Letzterer ist wörtlich
> aus `jit_kernel_break.py` Z. 32-34 kopiert, wo er zur *Fragmentfunktion* gehört.
> Sie wurden dann aber in die *Ratenformel* verdrahtet. Das erklärt auch den
> erfundenen `BREAKRVAL=5`: `BREAKFVAL` hat fünf Zweige, `BREAKRVAL` nur vier — die
> Arität wurde mitkopiert.
>
> **Gold-Standard bestätigt:** `upstream/dev_monorepo:mcpbe/src/wmcpbe/mcpbe_break.py`
> (Z. 153-159) ruft `calc_break_rate_1d` direkt auf `V_flat[-1, :a]` auf, mit derselben
> Propensity `W_i·S_i/δ_i`. Volumen-Argument und Propensity stimmten schon vorher
> überein — abgewichen ist ausschließlich die Ratenformel.
>
> **Umsetzung:**
> - Neue gemeinsame Funktion `kernels/breakage/_base_rate.py` (Single Source of Truth),
>   von `power_law.py` **und** `powerlaw_rumpf.py` genutzt — die Duplikation aus 2.1
>   ist damit aufgelöst.
> - `breakrval` 1-4 rechnen bitgenau wie `calc_break_rate_1d`; `breakrval=5` gelöscht.
> - `pl_v`/`pl_q` werden von den Ratenkerneln mit erklärendem `ValueError` abgelehnt.
> - Verifiziert: `Trials/test_breakage_rate_coleval_parity.py` (jetzt ein echter
>   Regressionstest mit Exit-Code) sowie 3000 zufällige Parameterkombinationen —
>   relativer Fehler durchgehend `0.000e+00`.
>
> **Drei Zusatzfehler, die bei der Verifikation gefunden und mitbehoben wurden:**
> 1. `pl_q` war in `powerlaw_rumpf` wirkungslos (nie in `__init__` gecacht →
>    `getattr` lieferte immer 0.5). Erledigt sich mit dem Löschen von `breakrval=5`.
> 2. Der Breakage-Kernel überschrieb `solver.G` (`kernel_integration.py` Z. 205 nach
>    Z. 182), mit Fallback auf hartcodiert `1000`. `mcpbe_nucleation.py` liest
>    `solver.G` — die Nucleation lief also still mit der Scherrate des Bruch-Kernels.
>    Jetzt besitzt die Agglomeration `solver.G`; Abweichungen werden gewarnt.
> 3. Tote Validierung in `power_law.py` (`'pl_v' not in params` konnte nie wahr sein,
>    da `get_default_params()` den Schlüssel immer einfügte).
>
> **Folge für Kalibrierungen:** Die Rate sinkt bei 34 µm um Faktor ~3.6e4, und der
> Faktor ist größenabhängig (1.2e5 bei 10 µm, 4.1e3 bei 300 µm). Umrechnung bei
> gleichem Volumen: `P1_neu = P1_alt · V^(−1/3) ≈ P1_alt · 1.24 / d`. Ein einzelner
> `P1`-Wert reproduziert das alte Verhalten daher **nicht** über alle Größen —
> die Größenabhängigkeit ist echt anders. Bestehende Ergebnisse unter
> `Trials/Results/` sind entsprechend markiert (`README_PRE_FIX.md`).

**Erstellt:** 2026-08-15 (Analyse-Session, keine Code-Änderungen)
**Kontext:** `dev_eric` und `dev_monorepo` sind vor ca. 3 Monaten (07.05.2026, Commit `71fb8ab`)
auseinandergelaufen. `dev_eric` wurde seitdem in Richtung Nassgranulation weiterentwickelt
(Nucleation, Porosität, Sättigung) und dabei bewusst an das Verhalten von `dev_monorepo`
angenähert/portiert. Diese Analyse wurde bewusst **ohne Vorwissen über diese Annäherung**
erstellt (reiner Code-Vergleich beider Branches gegen die gemeinsame Basis `pbe-core/`), um
eine unvoreingenommene Einschätzung zu liefern, wo die Annäherung gelungen ist und wo nicht.

**Verglichene Stände:**
- `dev_eric`: lokaler Arbeitsstand (Branch `dev_eric`, Commit `341163d` + ungetrackte Änderungen)
- `dev_monorepo`: `upstream/dev_monorepo` (`pdhs-group/PSD_opt`, Commit `02dfac6`, aktuellster
  verfügbarer Stand — identisch mit `origin/dev_monorepo` bzgl. `mcpbe/` und `pbe-core/`, der
  eine zusätzliche Commit dort betrifft nur das unabhängige `breakage-rate-model/`-Paket)
- Gemeinsame Basis: `pbe-core/src/pbe_core/func/jit_kernel_agg.py` (seit Trennung von
  **keiner** Seite verändert) und `jit_kernel_break.py` (nur eine 7-zeilige, rein defensive
  Änderung auf `dev_eric`-Seite, keine Formeländerung)

---

## Executive Summary

| Bereich | Status | Handlungsbedarf |
|---|---|---|
| **Agglomeration** (Kollisionsrate β) | ✅ Alle 4 Standard-Kernel (shear/brownian/constant/sum) liefern **bitgenau dieselbe Formel** wie `dev_monorepo` | Keiner |
| **Agglomeration** (Paket-Größe `dW`, Bias-Korrektur) | ✅ Nahezu zeilengleicher Algorithmus, unabhängig konvergiert auf dieselbe "Pair-Delta"-Korrektur | Keiner |
| **Breakage** (Bruchraten-Formel S(V)) | ❌ **Nur `BREAKRVAL=1` stimmt überein.** `BREAKRVAL=2,3,4` liefern strukturell andere Formeln, `BREAKRVAL=5` existiert bei `dev_monorepo` gar nicht. **`BREAKRVAL=4` ist der produktiv genutzte Default-Wert** (`powerlaw_rumpf`-Kernel) | **Ja — das ist der zu behebende Fund** |
| Breakage-Codegerüst (CDF-Sampling, Fragment-Verteilung, LMC-Adapter) | ✅ Praktisch wortgleich, eindeutig gemeinsame Abstammung | Keiner |

---

## Teil 1: Agglomeration — das Referenzmuster (funktioniert korrekt)

### 1.1 Wie die Differenzierung funktioniert (COLEVAL)

`dev_monorepo`'s `mcpbe_agg.py::_beta()` ruft direkt die gemeinsame Funktion
`pbe_core.func.jit_kernel_agg.calc_beta(COLEVAL, CORR_BETA, G, R, i, j)` auf. `COLEVAL` ist
ein Integer-Parameter des Solvers, der zwischen vier fest codierten physikalischen Modellen
unterscheidet:

```python
# pbe-core/src/pbe_core/func/jit_kernel_agg.py, calc_beta(), Zeilen 271-283
if COLEVAL == 1:
    beta = CORR_BETA * G * (r1 + r2) ** 3                                    # Shear (Chin 1998)
elif COLEVAL == 2:
    beta = CORR_BETA * 2 * 1.38e-23 * 293 * (r1 + r2) ** 2 / (3e-3 * (r1 * r2))  # Brownian (Tsouris 1995)
elif COLEVAL == 3:
    beta = CORR_BETA                                                          # Constant
elif COLEVAL == 4:
    beta = CORR_BETA * 4 * math.pi * (r1 ** 3 + r2 ** 3) / 3                  # Sum-Kernel
else:
    beta = 0.0
```

`dev_eric` verwendet **kein `COLEVAL`-Integer**, sondern wählt das Modell über einen
**Namen-String** (`agg_kernel_name='shear_chin1998'` o.ä.), der auf eine eigene, unabhängige
Kernel-Klasse in `mcpbe/src/wmcpbe/kernels/aggregation/*.py` verweist. Jede dieser Klassen
hat ihre **eigene, separat implementierte** JIT-Formel — sie ruft `calc_beta` nicht auf.
Die Zuordnung ist bestätigt (per Anfrage geprüft, s.u.):

| `COLEVAL` (dev_monorepo) | Name-String (dev_eric) | Datei | Formel stimmt überein? |
|---|---|---|---|
| 1 | `shear_chin1998` | `kernels/aggregation/shear_chin1998.py` | ✅ Ja, exakt: `corr_beta·G·(r1+r2)³` |
| 2 | `brownian_tsouris1995` | `kernels/aggregation/brownian_tsouris1995.py` | ✅ Ja (bei Default-Parametern `viscosity=1e-3`, `temperature=293`; dev_eric parametrisiert kT/μ generisch statt hartcodiert, ergibt bei Defaults dieselbe Zahl) |
| 3 | `constant` | `kernels/aggregation/constant.py` | ✅ Ja, exakt: `β = corr_beta` |
| 4 | `sum` | `kernels/aggregation/sum_kernel.py` | ✅ Ja, exakt: `corr_beta·(V1+V2)` |

Zusätzlich dokumentiert `kernels/aggregation/jit_kernels.py` (Zeile 74) explizit:
*"Kernel ids for the compiled dispatch (mirrors the COLEVAL switch upstream)"* — die
Entwickler waren sich der COLEVAL-Entsprechung also bewusst und haben sie bei der
Aggregation korrekt gepflegt.

### 1.2 Paket-Größe (`dW`) und Bias-Korrektur

Zusätzlich zur Kollisionsrate selbst haben beide Branches **unabhängig voneinander**
dieselbe mathematische Korrektur für gewichtete DSMC-Partikel eingeführt: die Propensity
wird durch `min(dW_i, dW_j)` statt nur `dW_i` geteilt (paarweise statt einseitig).
`dev_eric` nennt das "Bias Correction (Ji & Rhein)" mit eigenem Herleitungsdokument
(`docs/Bias_Correction_und_Gewichtsdisziplin.md`), `dev_monorepo` nennt die Funktionen
`nb_rebuild_ragg_weighted_pair_delta` / `nb_pick_partner_weighted_pair_delta` — zwei
unabhängige Umsetzungen derselben Idee. Die `_compute_agg_dW()`-Funktionen beider Branches
sind bis auf Variablennamen praktisch identisch (gleiche Parameter: `agg_dW_mode`,
`agg_dW_max`, `agg_dW_min`, `agg_dW_alpha`, gleiche Selbstkollisions-Sonderbehandlung
`delta_ii = min(delta_i, 0.5·W_i)`).

**Fazit Agglomeration:** Kein Handlungsbedarf. Architektonisch getrennt (Plugin-Kernel vs.
direkter Funktionsaufruf), aber mathematisch für alle vier Standardmodelle nachweislich
identisch.

---

## Teil 2: Breakage — der zu behebende Fund

### 2.1 Wie es aussehen SOLLTE (Analogie zu COLEVAL)

Erwartungsgemäß sollte es ein Äquivalent zu `COLEVAL` geben: `BREAKRVAL` (Integer,
existiert auf beiden Seiten als Parametername) sollte konsistent auf dieselben vier/fünf
physikalischen Modelle abbilden, so wie es bei COLEVAL der Fall ist.

**Gemeinsame Basis** (`pbe-core/src/pbe_core/func/jit_kernel_break.py`,
`calc_break_rate_1d()`, Zeilen 350-365):

```python
@njit
def calc_break_rate_1d(V, pl_P1, pl_P2, G, BREAKRVAL, i):
    if BREAKRVAL == 1:
        B_R = pl_P1                          # konstant
    elif BREAKRVAL == 2:
        B_R = pl_P1 * V[i]                   # linear in V
    elif BREAKRVAL == 3:
        # Power Law Pandy and Spielmann --> See Jeldres2018 (28)
        B_R = pl_P1 * G * V[i] ** pl_P2       # G^1 · V^P2
    elif BREAKRVAL == 4:
        # Hypothetical formula considering volume fraction
        B_R = pl_P1 * G * V[i] ** pl_P2       # identisch zu BREAKRVAL=3 in 1D!
    return B_R
    # BREAKRVAL == 5: kein Zweig vorhanden -> B_R bleibt unassigned
```

`dev_monorepo`'s `mcpbe_break.py` ruft diese Funktion **direkt** auf
(`from pbe_core.func.jit_kernel_break import calc_break_rate_1d as _kb_br1_single`).

`dev_eric` hat **zwei unabhängige Reimplementierungen** derselben `BREAKRVAL`-Logik, die
NICHT die gemeinsame Funktion aufrufen:

1. `kernels/breakage/power_law.py`, `PowerLawBreakageKernel.compute_rate()` /
   `compute_rate_array()` (Zeilen 113-206) — der eigene Docstring behauptet explizit
   (Zeile 119): *"CRITICAL: This must match the legacy JIT implementation EXACTLY."*
2. `kernels/breakage/powerlaw_rumpf.py`, freie Funktion `_compute_base_rate_jit()`
   (Zeilen 128-155) — das ist der **produktiv genutzte** Kernel (`powerlaw_rumpf`,
   `breakrval=4` als Default, siehe `test_sensitivity.py` / `test_powerlaw_rumpf_full.py`).
   Diese Funktion ist eine **Kopie** von (1), unabhängig dupliziert.

```python
# BEIDE Dateien (power_law.py Zeilen 145-170, powerlaw_rumpf.py Zeilen 137-155), identisch:
if breakrval == 1:
    rate = p1                                        # konstant  -> stimmt mit COLEVAL-Aequivalent ueberein
elif breakrval == 2:
    rate = p1 * (v_particle ** (1.0/3.0))             # V^(1/3)   -> pbe_core: V^1  MISMATCH
elif breakrval == 3:
    rate = p1                                         # konstant  -> pbe_core: G*V^P2  MISMATCH
elif breakrval == 4:
    alpha = pl_v / 3.0
    rate = p1 * (g ** p2) * (v_particle ** alpha)     # G^P2 * V^(pl_v/3) -> pbe_core: G^1 * V^P2  MISMATCH
elif breakrval == 5:
    V_ref = 1e-18
    rate = p1 * (g ** p2) * ((v_particle/V_ref) ** pl_q)  # existiert bei pbe_core gar nicht
```

### 2.2 Der entscheidende Unterschied bei `BREAKRVAL=4` (Produktions-Default!)

| | `pbe_core` / `dev_monorepo` | `dev_eric` (`power_law.py` UND `powerlaw_rumpf.py`) |
|---|---|---|
| Formel | `P1 · G¹ · V^P2` | `P1 · G^P2 · V^(pl_v/3)` |
| Rolle von `P2` | Volumen-Exponent | Scher-Exponent |
| Rolle von `pl_v` | *(existiert nicht)* | Volumen-Exponent (÷3) |
| G-Abhängigkeit | immer linear (`G¹`) | `G^P2`, nur bei `P2=1` zufällig linear |

`P2` bedeutet auf den beiden Seiten physikalisch etwas **anderes** — auf der einen Seite
"wie stark hängt die Bruchrate vom Volumen ab", auf der anderen "wie stark hängt sie von der
Scherrate ab". Das ist keine reine Zahlendrift, sondern eine **strukturelle
Neuinterpretation der Parameter**.

Mit den in `test_sensitivity.py` verwendeten Default-Werten (`p2=1.0`, `pl_v=2.0`) ergibt
sich zufällig dieselbe `G`-Abhängigkeit (`G^1.0 = G`), aber eine **andere
Volumen-Abhängigkeit**: `dev_eric` rechnet `V^0.667`, `pbe_core`/`dev_monorepo` würde bei
gleichem Parameterverständnis `V^1.0` rechnen — unterschiedliche Größenabhängigkeit der
Bruchhäufigkeit, kein Rundungsfehler.

### 2.3 Empirischer Beleg (deterministisch, ohne Simulation)

Test-Skript: [`Trials/test_breakage_rate_coleval_parity.py`](../src/wmcpbe/Trials/test_breakage_rate_coleval_parity.py)
(neu angelegt, reine Diagnose, keine Änderung an bestehenden Dateien). Vergleicht
`pbe_core.calc_break_rate_1d` direkt gegen `PowerLawBreakageKernel.compute_rate()` und
`PowerLawRumpfBreakageKernel._compute_base_rate_powerlaw()` für 5 Partikelgrößen
(1-50 µm) und alle `BREAKRVAL`-Werte, bei `p1=3e-2, p2=1.0, G=1000, pl_v=2.0`:

| BREAKRVAL | Relativer Fehler | Bewertung |
|---|---|---|
| 1 | `0.000e+00` bei allen 5 Partikelgrößen | ✅ **Exakter Treffer** |
| 2 | `1.000e+00` (Größenordnungen auseinander: `1.57e-20`…`1.96e-15` vs. `2.4e-8`…`1.2e-6`) | ❌ Kompletter Mismatch |
| 3 | `1.000e+00` (`pbe_core` wächst mit V, `dev_eric` bleibt konstant `3.0e-2`) | ❌ Kompletter Mismatch |
| 4 | `1.000e+00` (**Produktions-Default!** `pbe_core`: `1.57e-17`…`1.96e-12`; `dev_eric`: `1.95e-11`…`4.87e-8`) | ❌ Kompletter Mismatch |
| 5 | `pbe_core` liefert durchgehend `0.0` (kein `BREAKRVAL=5`-Zweig vorhanden), `dev_eric` liefert Werte zwischen `~16` und `~2·10⁶` | ❌ Existiert bei `dev_monorepo` nicht |

Volle Rohdaten reproduzierbar via:
```bash
cd mcpbe/src && python -m wmcpbe.Trials.test_breakage_rate_coleval_parity
```

### 2.4 Was NICHT betroffen ist

Das restliche Bruch-Codegerüst ist **praktisch wortgleich** zwischen beiden Branches
(gleiche Docstrings, gleiche interne Variablennamen wie `_ONE_SHOT_ADAPTER_NAMES`,
`_produce_one_frag_from_remaining`, `_build_fragments_stepwise`, identische LMC-Adapter-
Dispatch-Logik) — eindeutig gemeinsame Code-Abstammung, seit der Trennung kaum verändert.
Betroffen ist ausschließlich die **Bruchraten-Formel S(V)** (`BREAKRVAL`-Zweige 2-5), nicht
die Fragmentgrößen-Verteilung (`BREAKFVAL`, in `jit_kernel_break.py::breakage_func_1d`),
die separat geprüft wurde und bis auf eine rein defensive Ergänzung (fehlender
`else: return 0.0`-Zweig bei `dev_monorepo`) übereinstimmt.

---

## Teil 3: Ansatzpunkte für einen Fix (keine Entscheidung getroffen)

Zur Auswahl, ohne Empfehlung — abhängig davon, ob die abweichende Parametrisierung bei
`dev_eric` beabsichtigt war (z.B. weil `pl_v` als eigener Freiheitsgrad für die
Rumpf-Kalibrierung gebraucht wird) oder ein unbeabsichtigter Drift ist:

- **Option A — Formel an `pbe_core` angleichen:** `power_law.py` und
  `powerlaw_rumpf.py::_compute_base_rate_jit` so ändern, dass sie
  `pbe_core.func.jit_kernel_break.calc_break_rate_1d` entweder direkt aufrufen oder deren
  Formel exakt nachbilden (P2 wieder als Volumen-Exponent, G linear, `pl_v` entfernen oder
  auf `breakrval` beschränken, bei dem es tatsächlich gebraucht wird).
  → Betrifft **zwei Stellen** (Code-Duplikation, siehe 2.1) — bei einem Fix beide anfassen,
  oder die Duplikation auflösen (`powerlaw_rumpf.py` sollte vermutlich die Funktion aus
  `power_law.py`/`pbe_core` wiederverwenden statt sie zu kopieren).
- **Option B — Abweichung ist beabsichtigt:** Falls `dev_eric`s Parametrisierung
  (separater `pl_v`-Freiheitsgrad, `G^P2` statt `G^1`) bewusst für die
  Nassgranulations-Kalibrierung gewählt wurde, dann zumindest den irreführenden Docstring
  in `power_law.py` Zeile 119 korrigieren (der aktuell fälschlich exakte Übereinstimmung
  behauptet) und `BREAKRVAL` in `dev_eric` ggf. umbenennen/dokumentieren, um Verwechslung
  mit dem `pbe_core`-Namensraum zu vermeiden.
- **Vor jedem Fix zu prüfen:** Ob bestehende Kalibrierungen/Ergebnisse in `Trials/`
  (insbesondere alles mit `powerlaw_rumpf`, Default `breakrval=4`) auf der aktuellen
  (abweichenden) Formel beruhen — ein Fix von Option A würde bestehende
  Simulationsergebnisse zahlenmäßig verändern, nicht nur "korrigieren".

---

## Teil 4: Testbarkeit — geht das trotz unterschiedlichem RNG-Verbrauch?

**Ja, für die Bruchraten-Formel selbst uneingeschränkt.** `S(V)` ist eine reine,
deterministische Funktion von `(V, P1, P2, G, BREAKRVAL[, pl_v, pl_q])` — da sind
**keine Zufallszahlen beteiligt**. Der Vergleich in Abschnitt 2.3 lief nie eine
DSMC-Simulation, sondern hat beide Formeln nur an denselben Eingabewerten ausgewertet.
Das umgeht das RNG-Problem komplett, weil es gar nicht erst auftritt.

**Nein, für eine vollständige simulierte Teilchengrößenverteilung (PSD) wäre es
problematisch**, wenn man einzelne Event-Trajektorien 1:1 vergleichen wollte: Beide Seiten
verbrauchen ihre Zufallszahlen in unterschiedlicher Reihenfolge (andere Sampler-
Implementierung, andere interne Aufrufreihenfolge), selbst bei identischer Formel und
gleichem Seed liefen die Simulationen dadurch auseinander. Für diesen Fall wäre ein
**statistischer** Vergleich nötig (Momente M0/M1/M2 über viele Wiederholungen, nicht
Event-für-Event) — das ist aber eine separate, sekundäre Frage. Für das aktuelle Problem
(Bruchraten-Formel) ist sie nicht relevant, weil Abschnitt 2.3 das Problem schon ganz ohne
Simulation eindeutig zeigt.

**Empfehlung:** `Trials/test_breakage_rate_coleval_parity.py` nach jedem Fix-Versuch erneut
laufen lassen. Der relative Fehler sollte für die reparierten `BREAKRVAL`-Werte auf
Maschinen-Präzision (`~1e-15` oder exakt `0.0`) fallen.

---

## Referenzierte Dateien

| Datei | Rolle |
|---|---|
| `pbe-core/src/pbe_core/func/jit_kernel_agg.py` | Gemeinsame Basis, Kollisionsraten-Formeln (`calc_beta`, `COLEVAL` 1-4) |
| `pbe-core/src/pbe_core/func/jit_kernel_break.py` | Gemeinsame Basis, Bruchraten-Formeln (`calc_break_rate_1d`, `BREAKRVAL` 1-4) und Fragmentverteilungs-Formeln (`breakage_func_1d`, `BREAKFVAL`) |
| `mcpbe/src/wmcpbe/mcpbe_agg.py` | dev_eric: Agglomerations-Orchestrierung (Kernel-Framework) |
| `mcpbe/src/wmcpbe/mcpbe_break.py` | dev_eric: Bruch-Orchestrierung (Kernel-Framework) |
| `mcpbe/src/wmcpbe/kernels/aggregation/*.py` | dev_eric: unabhängige Kollisionsraten-Reimplementierungen (alle 4 geprüft, stimmen überein) |
| `mcpbe/src/wmcpbe/kernels/breakage/power_law.py` | dev_eric: unabhängige Bruchraten-Reimplementierung (Basis-Modell) — **hier liegt der Fund** |
| `mcpbe/src/wmcpbe/kernels/breakage/powerlaw_rumpf.py` | dev_eric: produktiv genutzter Bruch-Kernel, dupliziert dieselbe abweichende Formel in `_compute_base_rate_jit()` — **hier liegt der Fund, produktionsrelevant** |
| `mcpbe/src/wmcpbe/Trials/test_breakage_rate_coleval_parity.py` | Neu: deterministischer Marker-Test (dieser Analyse-Sitzung), referenziert für künftige Fix-Verifikation |
