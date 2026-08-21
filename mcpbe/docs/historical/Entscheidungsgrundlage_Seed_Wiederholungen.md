# Entscheidungsgrundlage: Jetzt auf Seed-Wiederholungen umsteigen?

**Frage:** Sind alle Audit-Themen gelöst, ist der Code „produktionsreif", und was
spricht dafür/dagegen, jetzt `solve_repeats()` über mehrere Seeds laufen zu
lassen?

**Diese Datei ändert nichts am Code.** Sie fasst den Ist-Zustand zusammen und
stellt Pro/Contra gegenüber. Keine Empfehlung wird stillschweigend vorausgesetzt.

---

## Kurzantwort

**Nein, nicht alle Themen sind gelöst — aber die gefährlichste Kategorie ist es.**

Von den 24 dokumentierten Befunden (B-01 bis B-25, zwei Nummern übersprungen)
sind:

- **19 code-seitig behoben und verifiziert** — darunter ausnahmslos alle Befunde,
  die die **Massenerhaltung** verletzt haben (B-01, B-04, B-07, B-17) oder
  Partikel-Dedup/Indexpflege betrafen (B-05, B-09 bis B-16, B-18, B-25).
- **2 für die Referenzconfig entschieden**, aber nicht codebasisweit behoben
  (B-03 SIZEEVAL, B-06 dW-Balance) — beide sind Konfigurationsentscheidungen,
  keine Bugs mehr.
- **3 bewusst offen gelassen**, weil du gesagt hast, Parameterwerte sind vor dem
  Fit ohnehin hinfällig (B-02 Vc/Konzentration, B-08 k_int, B-20
  Bruch-Runaway) — das sind aber keine Kleinigkeiten, siehe unten.
- **4 Module nie im Detail geprüft** (Nucleation-Tiefenprüfung,
  `reconstruction_mixin.py`, ungenutzte Kernel, `mcpbe_post.py`).

„Produktionsreif" ist deshalb keine Ja/Nein-Antwort, sondern eine Frage der
**Reichweite**: für die eine, jetzt sauber konfigurierte Referenzconfig
(`test_powerlaw_rumpf_full.py`) ist die Grundlage solide. Für „ich starte jetzt
eine große Seed-Kampagne und vertraue den Zahlen" fehlen noch ein paar Dinge —
im Detail unten.

---

## Was tatsächlich gesichert ist

Das ist der Teil, der für Seed-Wiederholungen am meisten zählt, und er ist
substanziell:

**Jeder Bug, der Masse in einer seed-abhängigen, unvorhersehbaren Weise
verändert hätte, ist behoben und mit Vorher/Nachher-Zahlen belegt:**

| Befund | Was er verzerrt hätte | Messung nach Fix |
|---|---|---|
| B-01 | Kollisionsrate β um bis zu Faktor 3 zu klein, porositätsabhängig | 0 von 949 Partikeln inkonsistent (vorher 64) |
| B-17 | Feststoffmasse im Breakage-Fallback verloren, proportional zu ε | ΔV_solid = 0,000000 % (vorher −0,0014 %) |
| B-04 | Merger-Trefferquote bricht unter Kompression ein (99,4 % → 10,9 %) | 0,0 % veraltete Einträge (vorher bis 99,8 %) |
| B-07 | Kompression konnte bei ε→1 Masse erzeugen | Porosität und V_dry jetzt atomar gemeinsam geschrieben |
| B-18 | `real_break_events` zählte nach jedem Fehlversuch die letzte Paketgröße erneut | Reset + Accept/Reject-Zähler ergänzt |

Das ist die Kategorie, die für eine **statistische** Studie über Seeds am
gefährlichsten ist: Ein Bug, der *systematisch* falsch ist, verschiebt jeden
Seed gleich — man merkt es vielleicht. Ein Bug, der *zustandsabhängig* falsch
ist (wie B-01, dessen Fehler mit der Porosität wächst, oder B-04, dessen Effekt
von der zufälligen Kompressionshistorie abhängt), erzeugt **zusätzliches
Rauschen, das wie Seed-Varianz aussieht** — und genau das würde man mit einer
Mehrfach-Seed-Studie eigentlich messen wollen. Diese Konfundierung ist jetzt
weg.

---

## Was NICHT gesichert ist

### 1. Der physikalische Betriebspunkt der Referenzconfig ist unrealistisch (B-02)

`CONTROL_VOLUME = 1 m³` bei 2000 Partikeln × Gewicht 600 ergibt einen
Feststoffvolumenanteil von φ ≈ 6·10⁻⁹ — sechs Teile pro Milliarde, gegenüber
φ ≈ 0,3–0,6 bei einem echten Granulator. Das wurde **nicht geändert**, mit der
Begründung, dass ein Fit gegen Experimentaldaten die Werte ohnehin verschiebt.

Das stimmt für den *Zahlenwert*. Es stimmt nicht dafür, ob eine
Seed-Wiederholungsstudie an diesem Punkt aussagekräftig ist: Bei φ ≈ 6·10⁻⁹
passieren in 20 s nur ~150 Ereignisse insgesamt. Seed-Varianz bei so wenigen
Ereignissen ist **riesig relativ zum Signal** — man würde vor allem
Schrotrauschen vermessen, nicht die Physik.

### 2. B-20 (Bruch-Runaway) ist keine Petitesse

Bei `INITIAL_POROSITY = 0.6` (statt 0.0 in der aktuellen Referenzconfig)
kollabiert das System in einen Extremfall: **0 Agglomerationen, 119 678 Brüche**
in 4000 Ereignissen, Partikelzahl 200 → 11 824. Das ist kein Rand­effekt,
sondern zeigt, dass die drei gekoppelten Parameter (Vc, k_int, Bruchvorfaktor)
aktuell kein stabiles Gleichgewicht zwischen Wachstum und Bruch finden, sobald
man von ε₀=0 wegbewegt. Für eine Seed-Studie mit realistischerer Porosität
wäre das der Punkt, an dem einzelne Seeds numerisch explodieren könnten
(Partikelzahl-Wachstum ohne Grenze, siehe `max_particles`-Sicherung in
`solve()`).

### 3. SIZEEVAL/B-03 ist nur maskiert, nicht behoben

`solver.SIZEEVAL = 0` steht jetzt explizit in `test_powerlaw_rumpf_full.py`.
Der zugrundeliegende Bug in `mcpbe_agg.py::_accept_sizeeval` (invertierte
Schalter-Konvention) existiert weiterhin im Code. Für **diese eine Datei** ist
das kein Risiko mehr. Für jedes neue Skript, jede Kopie, jede künftige Session
(auch eine andere KI-Session, die das nicht liest) ist die Falle weiterhin
scharf — der Default ist nach wie vor „Filter versehentlich aktiv".

### 4. Vier Module wurden nie im Detail geprüft

- **`mcpbe_nucleation.py`** (2721 Zeilen): **inzwischen vollstaendig geprueft**
  (18.08.2026, siehe [Tiefenpruefung Nucleation](Audit_2026-08-17.md#n-nucleation)). Ergebnis: Massen- und
  Fluessigkeitsbilanz **exakt** bei eps0 = 0,0 / 0,4 / 0,8; `_liquid_remainder`
  geht vollstaendig auf. Vier Nebenbefunde (N-01..N-04), keiner davon ein
  Massen- oder Ergebnisfehler. Dieser Punkt ist damit **kein Argument mehr**
  gegen den naechsten Schritt.
- **`reconstruction_mixin.py`** (1532 Zeilen): **komplett ungeprüft**, weil
  `recon_enable = False` in der Referenzconfig — der Code läuft nie. Das ist
  bei Seed-Wiederholungen aber nicht garantiert so: mehr Ereignisse (längere
  Laufzeit, mehr Seeds als Einzelläufe) erhöhen die Wahrscheinlichkeit, dass
  irgendwann `a_tot > recon_N_max` erreicht wird. Wichtig dabei: beim
  Abgleich mit `upstream/dev_monorepo` kam heraus, dass **Reconstruction beim
  Betreuer selbst Flüssigkeit/Porosität/Sättigung nicht über den
  Resampling-Schritt trägt** (~0,4 % Feststoffverlust pro Aufruf, gemessen).
  In `dev_eric` gibt es dafür ein Sicherheitsnetz
  (`_assert_reconstruction_is_safe`), das den Aufruf bei nasser Population mit
  einem klaren Fehler verweigert, statt still Masse zu verlieren — aber dieses
  Sicherheitsnetz selbst wurde in diesem Audit nicht geprüft.
- **`mcpbe_post.py`**: nur die für B-01 relevanten Stellen (Neuberechnung von
  X aus V_dry) verifiziert, nicht die übrige Momenten-/PSD-Logik.
- **Ungenutzte Kernel**: `liquid_bridge`, `incomplete_mixing`, `volume_mixing`,
  `brownian_tsouris1995`, `sum_kernel`, `constant`, `power_law`, `fittable`,
  alle drei `liquid_distribution`-Kernel — keiner davon wurde geprüft, weil
  die Referenzconfig sie nicht nutzt. Für eine reine Wiederholungsstudie der
  *aktuellen* Kernelkombination ist das irrelevant; für Sensitivitätsstudien
  über Kernel-Varianten (falls geplant) nicht.

### 5. Vollständiger Lauf der finalen Konfiguration — inzwischen erledigt ✅

Während dieses Dokument entstand, ist der volle Referenzlauf (2000 Partikel,
20 s, finale SIZEEVAL=0/dW-Konfiguration) im Hintergrund durchgelaufen:

```
Laufzeit:                    47,35 s  (1137 MC-Ereignisse)
Liquid conservation:          PASS (+0,0000 %)
Solid mass conservation:      PASS (+0,0000 %)
Droplet accuracy:             PASS (100,00 %)
Agglomeration active:         PASS (6226 dW-gewichtete Ereignisse)
Stokes rejection active:      PASS (241 abgelehnt)
Breakage active:              PASS (14833 dW-gewichtete Ereignisse)
Porosity evolved:             PASS (|Δporo| = 0,003)
OVERALL:                      ALLE TESTS BESTANDEN
```

Das ist echte, positive Evidenz: die finale Konfiguration läuft ohne Fehler
durch, Masse (fest und flüssig) bleibt auf 0,0000 % erhalten, und beide
Prozesse (Agglomeration, Breakage) sind aktiv — keiner dominiert den anderen
vollständig wie im B-20-Extremfall. Dieser eine Lauf ersetzt keine
Seed-Statistik, aber er entkräftet den Einwand „ungetestete finale
Konfiguration“ als offenen Punkt.

### 6. Neun weitere Trial-Skripte tragen die Config-Fixes nicht

`test_droplet_10um_stable.py`, `test_droplet_20um_instable.py`,
`test_particle_merger_comparison.py`, `test_particle_merger_rumpf_full.py` und
weitere setzen weder `SIZEEVAL` noch `agg_dW_max` — sie laufen weiterhin mit der
ursprünglichen (SIZEEVAL-Bug-aktiv, dW-unbalanciert) Konfiguration. Die
**Code**-Fixes (Massenerhaltung etc.) gelten für sie automatisch mit, weil die
in der gemeinsamen Solver-Bibliothek liegen — nur die beiden
**Konfigurationsentscheidungen** (B-03, B-06) wurden nur in der einen
Referenzdatei nachgezogen.

---

## Argumente FÜR den nächsten Schritt jetzt

1. **Die für Seed-Studien gefährlichste Bugklasse ist geschlossen.**
   Zustandsabhängige Massenfehler (B-01, B-04, B-07, B-17), die als
   Pseudo-Seed-Varianz erschienen wären, sind behoben und mit Zahlen belegt.
   Genau das war die Voraussetzung, die vor jeder Seed-Studie erfüllt sein
   musste — sie ist es jetzt.
2. **Die bekannten Konfigurations-Confounds der Referenzconfig sind raus.**
   SIZEEVAL war unbeabsichtigt aktiv und hat B-06 um 25 % verzerrt — das ist
   jetzt korrigiert. Ohne diesen Schritt hätte man Seed-Varianz auf einer
   verzerrten Baseline gemessen.
3. **`solve_repeats()` ist eine reife, fertige API** (seriell oder parallel,
   optionale PSD-Aggregation, `dump_results` für große Ensembles) — keine
   fehlende Infrastruktur, die den nächsten Schritt aufhalten würde.
4. **Seed-Wiederholungen sind selbst eine Validierung.** Stürzt der Code über
   viele Seeds ab oder zeigt er absurde Ausreißer, ist das neue, nützliche
   Information — dafür muss nicht erst jedes Modul geprüft sein.
5. **Parameterwerte sind ohnehin vorläufig** (deine eigene Einschätzung zu
   B-02/B-08/B-20). Seed-Wiederholung prüft *numerische Stabilität eines
   Verfahrens*, nicht *physikalische Korrektheit eines Parametersatzes* — für
   Ersteres ist die aktuelle Grundlage tragfähig genug, um erste Erfahrungen
   zu sammeln.

## Argumente DAGEGEN / zum Zurückstellen

1. **Der aktuelle Betriebspunkt ist statistisch uninteressant.** Bei φ ≈
   6·10⁻⁹ passieren ~150 Ereignisse in 20 s — Seed-Varianz wäre dort vor allem
   Schrotrauschen. Eine Seed-Kampagne auf diesem Punkt liefert präzise
   Statistik über ein physikalisch nicht-repräsentatives Regime. Mindestens
   `Vc` (und ggf. `k_int`) auf eine plausible Größenordnung zu bringen, bevor
   man Rechenzeit in Wiederholungen investiert, wäre der günstigere nächste
   Schritt.
2. **B-20 zeigt einen Instabilitätsmodus**, der bei realistischerer
   Anfangsporosität (nicht ε₀=0) auftritt: Partikelzahl-Explosion,
   Nulldurchsatz bei Agglomeration. Läuft man Seeds bei einer Konfiguration,
   die in diesen Modus kippen kann, drohen einzelne Seeds mit
   `_hit_particle_limit` abzubrechen oder unbrauchbar lange zu laufen —
   uneinheitliche Ensembles.
3. **`reconstruction_mixin.py` ist ungeprüft**, und es gibt einen konkreten,
   vom Betreuer-Code geerbten Risikofall (Reconstruction verwirft
   Flüssigkeit/Porosität) mit einem ungeprüften Sicherheitsnetz davor. Sollte
   eine Seed-Kampagne (mehr Ereignisse als Einzelläufe) `recon_N_max`
   erreichen, träfe man auf Code, der in diesem Audit nicht angefasst wurde —
   entweder ein sauberer Fehlerabbruch (unschön, aber sicher) oder ein
   unentdeckter Bug.
4. **Kein abgeschlossener Lauf der finalen Konfiguration liegt vor** (siehe
   oben, Hintergrundlauf noch offen). Vor einer mehrfach wiederholten Kampagne
   wäre ein einzelner, vollständig durchgelaufener und mass-check-verifizierter
   Referenzlauf ein natürlicher, günstiger Zwischenschritt.
5. **Neun weitere Trial-Skripte laufen noch mit der alten, unbalancierten
   Konfiguration.** Falls die Seed-Studie nicht exakt `test_powerlaw_rumpf_full.py`
   verwendet, sondern eines der Geschwisterskripte, sind B-03/B-06 dort noch
   nicht behoben.
6. **Nucleation-Tiefenprüfung fehlt.** Bei 2721 Zeilen und einer bereits
   zweimal bestätigten Historie duplizierter, auseinanderlaufender Logik
   (`Fundamentals.md` §7, hier zusätzlich durch B-21 bestätigt) ist die
   Wahrscheinlichkeit ungleich null, dass dort noch etwas Ähnliches wie B-01
   oder B-17 liegt, das eine gezielte Suche (wie in diesem Audit) einfach noch
   nicht gefunden hat, weil niemand gezielt danach gesucht hat.

---

## Was diese Liste bewusst nicht tut

Sie sagt nicht „mach es" oder „mach es nicht". Sie sortiert die offenen Punkte
nach **Relevanz für genau diese Entscheidung**:

- **Blockierend im engeren Sinne ist nichts** — es gibt keinen bekannten Bug,
  der eine Seed-Studie technisch verhindert oder ihre Ergebnisse mit
  Sicherheit unbrauchbar macht.
- **Am ehesten wertmindernd** für eine Seed-Studie *jetzt* sind Punkt 1 und 2
  der Contra-Liste (unrealistischer Betriebspunkt, Instabilitätsmodus) — nicht
  weil sie Bugs wären, sondern weil sie die Frage aufwerfen, *was* man mit den
  Wiederholungen eigentlich misst.
- **Am ehesten überraschungsträchtig**, falls man die Studie größer/länger
  fährt, sind Punkt 3 und 6 (Reconstruction, Nucleation-Tiefe) — unbekanntes
  Terrain, kein bekannter Fehler, aber auch keine Entwarnung.
