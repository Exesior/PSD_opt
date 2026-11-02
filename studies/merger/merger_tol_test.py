"""
Merger-Toleranz-Test mit dem KF3-Setup (Eric, 2026-10-20).

Frage: Wie gut funktioniert der ParticleMerger? Er haelt die Zahl der Rechenpartikel
(n_comp) klein: ein neues Partikel, das einem vorhandenen "gleich genug" ist, erhoeht nur
dessen Gewicht W, statt eine eigene Zeile zu bekommen. Weil "gleich" nur innerhalb einer
Toleranz tau gilt, kann jeder solche Merge etwas Masse und Verteilung verfaelschen.

5 Einstellungen; alles andere ist exakt WetGranulationAdapter._build_solver mit der
KF3-Config und dem gefitteten KF3-Kandidaten aus kandidat.json:

    aus        Merger ganz aus (_enable_particle_merging = False)  -> Nulllinie
    tau_1e-2   alle Merger-Toleranzen = 1e-2
    tau_1e-4   alle Merger-Toleranzen = 1e-4
    tau_1e-6   alle Merger-Toleranzen = 1e-6
    tau_1e-8   alle Merger-Toleranzen = 1e-8

"Alle Merger-Toleranzen" (Eric: alle gleich) = die drei Stellen, an denen der Merger vergleicht:
    solver._fragment_merge_tol      Agglomeration + Bruch: V_dry und Fluessigkeit, relativ
    nucleation.config.similarity_tol
                                    Nucleation: V_dry und Fluessigkeit, relativ
                                    (Fluessigkeit zusaetzlich auf tau * Tropfenvolumen gedeckelt)
    ParticleMerger.tol_abs_frac     Porositaet und Saettigung, absolut (in allen drei Pfaden)
Im Fit galt 1e-6 / 1e-5 / 1e-6 -> tau_1e-6 weicht vom Fit nur in der Nucleation-Toleranz ab.

Replikate: die ersten REPLIKATE (5) der 10 festen Fit-Seeds; "aus" zunaechst nur der erste
Seed (REPLIKATE_AUS = 1, mit --rep-aus erhoehen). Fertige Replikate (npz vorhanden) werden
nie neu gerechnet. Kein Zeitlimit; MAX_PARTICLES der Config bleibt (status=max_particles).

Speicherpunkte alle 5 s (0..600 s). Ein Speicherpunkt ist ein reiner Schnappschuss, sobald
ein Event die Zeit ueberschreitet -> Zufallszahlen und Zeitschritte sind dieselben wie mit
dem t_vec der Config.

Gemessen je Replikat an jedem Speicherpunkt:
    n_comp                      Rechenpartikel
    M  = sum(W * V_solid) / V_c Feststoff je Kontrollvolumen (V_c wegen Vc-Verdopplung)
    dM = M / M(t=0) - 1         Feststoffmassen-Defekt
    Wand-/CPU-Zeit              seit Start von solve()
    Merger-Zaehler              Suchen / Treffer / neu angelegt (alle Pfade; dazu nur Bruch)
    Woher dM kommt (kumuliert)  Agglomeration aufgeteilt nach "Kind aufs eigene Elternteil
                                gemergt" / "auf ein anderes Partikel gemergt" / "neu angelegt"
                                (je Anzahl + Feststoffaenderung), dazu Feststoffaenderung in
                                Bruch-Events und Nucleation. Rest = dM minus diese Teile.
    Operationen (je Speicherpunkt kumuliert):
        Iterationen, sum n und sum n^2 ueber die Iterationen (jede Iteration baut die
        Agglomerations-Propensitaeten paarweise neu = ~n^2/2 Paar-Auswertungen), die
        gemessene Zeit in diesem Rebuild, die Merger-Suche (sum n je Suche = verglichene
        Partikel, plus gemessene Zeit) und sum (n + m)^2 mit m = bisherige Merger-Treffer
        = Schaetzung, wie gross sum n^2 OHNE Merger waere (Obergrenze: ein gemergtes
        Partikel waere ohne Merger eine eigene Spalte bis Laufende).
        Ersparnis [s] ~ Zeit Rebuild * (sum (n+m)^2 / sum n^2 - 1) = OBERGRENZE (gemergte Partikel
        wuerden ohne Merger oft spaeter wegagglomeriert; bei vielen Treffern, z. B. tau = 1e-2,
        sinnlos gross). Verlaesslicher: empirisch = Zeit Rebuild * (sum n^2 des "aus"-Laufs mit
        gleichem Seed / sum n^2 - 1). Netto = Ersparnis - Zeit Merger-Suche.
    Hintergrund: Faellt das Kind (i + j) innerhalb tau auf ein Elternteil, erhoeht der Merger
    dessen W um dW, danach zieht die Agglomeration dW von i und j ab -> die dW Partner-Partikel
    verschwinden samt Masse (gemessen im Smoke-Test, tau = 1e-2: -57 % Feststoff).
Bei 600 s, Q3 (Masse; Durchmesser aus V_dry, Sieb-Definition d < x) gegen die Nulllinie "aus":
    D_KF3 = sum_k m_k |Q3_a(x_k) - Q3_b(x_k)|  an den KF3-Siebmaschen x_k mit den
                                              KF3-Maschengewichten m_k (gleiche Einheit wie J)
    D_max = max_x |Q3_a(x) - Q3_b(x)|          auf feinem Raster 20..3000 um
    je als: Mittel der Replikate vs. Mittel "aus" | gleicher Seed vs. "aus" |
            Rauschen = Replikat vs. Replikat derselben Einstellung (was zwei unabhaengige
            MC-Laeufe ohnehin voneinander abweichen)

Aufruf (Terminal; Pfade von ueberall)
    python studies/merger/merger_tol_test.py sim --sqlite <PFAD> [--id N]   (einmal: Kandidat)
    python studies/merger/merger_tol_test.py sim --workers 6
    python studies/merger/merger_tol_test.py auswertung
Optionen:
    --workers N      parallele Prozesse (Default: Anzahl CPUs, hoechstens offene Replikate)
    --rep N          Replikate je tau (Default 5)
    --rep-aus N      Replikate fuer "aus" (Default 1)
    --sqlite PFAD    Kandidat aus dieser SQLite uebernehmen -> kandidat.json
    --id N           ... mit dieser id (ohne: kleinster Score)
    --smoke          Funktionstest: 60 statt 500 Rechenpartikel, 2 Replikate je tau,
                     eigener Ordner smoke/ (physikalisch NICHT aussagekraeftig)
    --unterordner N  Ergebnisse in den Unterordner N (eigenes npz/, Excel, Grafiken), z. B.
                     "seriell" fuer einen Lauf mit --workers 1 -- vorhandene Daten bleiben unberuehrt.
    --bruch-faktor F Numerische Variante mit viel Bruch: BREAK_P1 des Kandidaten x F
                     (z. B. 1e5). Eigener Unterordner bruch_x<F>/ mit eigenem npz/, Excel
                     und Grafiken -- die Hauptergebnisse werden nie ueberschrieben.
                     Beim KF3-Betriebspunkt bricht praktisch nichts; Smoke-Scan
                     (60 Partikel): x1e3 -> erste Brueche, x1e4 -> ~27, x1e5 -> ~190
                     Bruch-Events und n_comp 64 -> ~600 (ohne Merger).
Start per "Play" (Spyder, ohne Argumente): SPYDER_ARGS unten; erst sim, dann auswertung.
Geht der Parallelbetrieb in der IDE nicht, rechnet das Skript nacheinander weiter.
Rechenzeiten: parallele Prozesse teilen sich die Maschine -> die Wandzeit haengt von der
Auslastung ab; die CPU-Zeit ist der fairere Vergleich.

Ausgabe (dieser Ordner):
    kandidat.json                 Kandidat (Parameter exakt)
    npz/<Einstellung>_s<Seed>.npz Rohdaten je Replikat (alle Speicherpunkte)
    merger_tol_test.xlsx          Info, Uebersicht, Laeufe, Zeitreihen, Q3_600s,
                                  Partikel_<Einstellung>[_2, ...] = alle Rechenpartikel an
                                  jedem Speicherpunkt. Zahlen exakt (repr statt 16 Stellen),
                                  nach dem Schreiben bitweise zurueckgelesen.
    Zeitverlauf.png               n_comp und |dM| ueber der Zeit
    Q3_600s.png                   Q3(x) bei 600 s und Differenz zur Nulllinie
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import runpy
import sqlite3
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIG = REPO / "studies" / "optimization" / "opt_config_wet_granulation_kf3_nosampling_logw.py"
HELPER_KEYS = {"actor_wait", "wait_time", "max_reuse", "__exp_paths", "__known_params"}

SETTINGS = {  # Name -> tau (None = Merger aus)
    "aus": None,
    "tau_1e-2": 1e-2,
    "tau_1e-4": 1e-4,
    "tau_1e-6": 1e-6,
    "tau_1e-8": 1e-8,
}
REPLIKATE = 5
REPLIKATE_AUS = 1
DT_SAVE = 5.0                  # Speicherpunkte alle 5 s
PROGRESS_EVERY = 60.0          # Fortschrittszeile alle 60 s Simulationszeit
X_GRID_UM = np.geomspace(20.0, 3000.0, 800)
EXCEL_MAX_ROWS = 1_048_575     # Datenzeilen je Blatt (+ Kopfzeile = Excel-Grenze)
DM_FLOOR = 1e-17               # |dM| darunter (auch exakt 0) wird im Plot auf diese Linie gesetzt

# Farbe + Marker + Linienstil je Einstellung (nie Farbe allein). Farben: validierte
# kategoriale Palette (dataviz-Skill), "aus" als schwarze Referenz.
STYLE = {
    "aus":      ("#0b0b0b", "o", "-"),
    "tau_1e-2": ("#2a78d6", "s", "--"),
    "tau_1e-4": ("#eb6834", "^", "-."),
    "tau_1e-6": ("#1baf7a", "D", ":"),
    "tau_1e-8": ("#eda100", "v", (0, (6, 1.5, 1, 1.5, 1, 1.5))),
}


def out_dir(smoke: bool) -> Path:
    """Ordner mit dem Basis-Kandidaten (kandidat.json)."""
    return HERE / "smoke" if smoke else HERE


def run_dir(args) -> Path:
    """Ergebnisordner: Basisordner, ggf. Unterordner bruch_x<F>/ und/oder --unterordner."""
    out = out_dir(args.smoke)
    if args.bruch_faktor != 1.0:
        out = out / f"bruch_x{args.bruch_faktor:g}"
    if args.unterordner:
        out = out / args.unterordner
    return out


def run_candidate(args) -> dict:
    """Basis-Kandidat, bei --bruch-faktor mit BREAK_P1 x Faktor; im Ergebnisordner als
    kandidat.json abgelegt (und gegen einen dort vorhandenen geprueft)."""
    cand = candidate(args)
    if args.bruch_faktor != 1.0:
        cand = json.loads(json.dumps(cand))
        cand["params"]["BREAK_P1"] = cand["params"]["BREAK_P1"] * args.bruch_faktor
        cand["bruch_faktor"] = args.bruch_faktor
    out = run_dir(args)
    if out != out_dir(args.smoke):
        path = out / "kandidat.json"
        if path.exists():
            old = json.loads(path.read_text(encoding="utf-8"))
            if old["params"] != cand["params"]:
                raise RuntimeError(f"{path} haelt einen anderen Kandidaten -- Ordner wegraeumen oder umbenennen.")
        else:
            out.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cand, indent=2), encoding="utf-8")
    return cand


def time_grid() -> np.ndarray:
    return np.arange(0.0, 600.0 + 0.5 * DT_SAVE, DT_SAVE)


# ===========================================================================
#  Kandidat + Adapter + Merger-Einstellung
# ===========================================================================
def read_candidate(sqlite_path: Path, cid: int | None) -> dict:
    con = sqlite3.connect(Path(sqlite_path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        if cid is None:
            row = con.execute("SELECT id, score, param_json FROM warm_params ORDER BY score LIMIT 1").fetchone()
        else:
            row = con.execute("SELECT id, score, param_json FROM warm_params WHERE id = ?", (cid,)).fetchone()
    finally:
        con.close()
    if row is None:
        raise ValueError(f"id {cid} nicht in {sqlite_path}")
    params = {k: v for k, v in json.loads(row[2]).items() if k not in HELPER_KEYS}
    return {"sqlite": str(sqlite_path), "id": int(row[0]), "score": float(row[1]), "params": params}


def candidate(args) -> dict:
    """kandidat.json lesen; mit --sqlite einmalig anlegen (nie stillschweigend austauschen)."""
    out = out_dir(args.smoke)
    path = out / "kandidat.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if args.sqlite is None:
        if old is None:
            raise FileNotFoundError(f"{path} fehlt -- einmal mit --sqlite <PFAD> [--id N] starten.")
        return old
    new = read_candidate(Path(args.sqlite), args.id)
    if old is not None and old["params"] != new["params"]:
        raise RuntimeError(f"{path} haelt schon einen anderen Kandidaten (id {old['id']}). Fuer einen "
                           "neuen Kandidaten einen eigenen Ordner nehmen oder kandidat.json + npz/ wegraeumen.")
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(new, indent=2), encoding="utf-8")
    return new


def make_adapter(cand: dict, smoke: bool):
    from pbe_optimizer.adapter.wet_granulation_adapter import WetGranulationAdapter
    from pbe_optimizer.optimizer.opt_core import OptCore

    cfg = runpy.run_path(str(CONFIG))
    pop = dict(cfg["config"]["pop_params"])
    if smoke:  # gleiche Partikeldichte n = N * W / Vc, nur weniger Rechenpartikel
        factor = pop["INITIAL_PARTICLES"] / 60.0
        pop["INITIAL_PARTICLES"] = 60
        pop["INITIAL_WEIGHT"] = pop["INITIAL_WEIGHT"] * factor
    t_vec = time_grid()
    t_cfg = np.asarray(cfg["config"]["algo_params"]["t_vec"], dtype=float)
    if t_vec[-1] != t_cfg[-1] or not np.all(np.isin(t_cfg, t_vec)):
        raise ValueError(f"5-s-Raster passt nicht zum t_vec der Config {t_cfg.tolist()}.")
    core = OptCore()
    core.t_vec = t_vec
    core.dim = 1
    core.max_iter_time = 0.0
    ad = WetGranulationAdapter(opt=core, dim=1, t_vec=t_vec, load_attr=False)
    for k, v in {**pop, **cand["params"]}.items():
        setattr(ad, k, v)
    ad.get_all_data(Path(cfg["DATA_FILE"]))
    return ad, pop, t_vec


def apply_merger(solver, tau: float | None) -> dict:
    """Merger aus bzw. alle drei Toleranzen auf tau. Gibt die tatsaechlich gesetzten Werte zurueck."""
    if tau is None:
        solver._enable_particle_merging = False
        solver._ensure_particle_merger()           # -> _particle_merger = None (auch fuer Nucleation)
    else:
        solver._fragment_merge_tol = float(tau)    # Agglomeration + Bruch
        merger = solver._ensure_particle_merger()   # baut den Merger mit der neuen Toleranz neu
        merger.tol_abs_frac = float(tau)            # Porositaet + Saettigung
        solver.nucleation.config.similarity_tol = float(tau)   # Nucleation
    m = getattr(solver, "_particle_merger", None)
    return {"merger": m is not None,
            "tol_rel": None if m is None else float(m.tol_rel),
            "tol_abs_frac": None if m is None else float(m.tol_abs_frac),
            "nucleation_similarity_tol": None if m is None else float(solver.nucleation.config.similarity_tol)}


class MassTracker:
    """Woher kommt dM? Huellt _do_one_agg, _merge_pair, _do_one_break und nucleation.step
    der Solver-INSTANZ ein und misst die Feststoffaenderung M = sum(W * V_solid) / V_c
    vor/nach jedem Aufruf. Rein beobachtend (kein Eingriff in den Core, Verlauf unveraendert)."""

    KEYS = ("agg_eltern_n", "agg_eltern_dM", "agg_andere_n", "agg_andere_dM",
            "agg_neu_n", "agg_neu_dM", "bruch_dM", "nucleation_dM")

    def __init__(self, solver):
        self.s = solver
        self.c = dict.fromkeys(self.KEYS, 0.0)
        self._kind = None
        orig_pair, orig_agg = solver._merge_pair, solver._do_one_agg

        def merge_pair(i, j, dW):
            m = getattr(solver, "_particle_merger", None)
            n0 = m._stats_merges if m is not None else 0
            idx = orig_pair(i, j, dW)
            merged = m is not None and m._stats_merges > n0
            self._kind = "eltern" if merged and idx in (i, j) else ("andere" if merged else "neu")
            return idx

        def do_one_agg(*a, **k):
            self._kind = None                    # bleibt None, wenn die Kollision abgelehnt wird
            m0 = self.mass()
            r = orig_agg(*a, **k)
            if self._kind is not None:
                self.c[f"agg_{self._kind}_n"] += 1
                self.c[f"agg_{self._kind}_dM"] += self.mass() - m0
            return r

        solver._merge_pair = merge_pair
        solver._do_one_agg = do_one_agg
        self._wrap(solver, "_do_one_break", "bruch_dM")
        self._wrap(solver.nucleation, "step", "nucleation_dM")

    def mass(self) -> float:
        a = self.s.a_tot
        return float(np.dot(self.s.W[:a], self.s.V_flat[0, :a]) / self.s.Vc)

    def _wrap(self, obj, attr: str, key: str) -> None:
        orig = getattr(obj, attr)

        def f(*a, **k):
            m0 = self.mass()
            r = orig(*a, **k)
            self.c[key] += self.mass() - m0
            return r
        setattr(obj, attr, f)

    def row(self) -> list:
        return [self.c[k] for k in self.KEYS]


class OpCounter:
    """Zaehlt und misst die Arbeit: den paarweisen Propensitaets-Rebuild (einmal je Iteration,
    ~n^2/2 Paar-Auswertungen) und die Merger-Suche (vektorisierter Vergleich gegen alle n
    aktiven Partikel). Huellt _rebuild_all_propensities und ParticleMerger.find_similar der
    INSTANZ ein. Rein beobachtend (kein Core-Eingriff, Verlauf unveraendert)."""

    def __init__(self, solver):
        self.iters = 0
        self.sum_n = 0.0
        self.sum_n2 = 0.0
        self.sum_nm2 = 0.0          # sum (n + m)^2, m = bisherige Merger-Treffer
        self.t_rebuild = 0.0
        self.scan_n = 0.0
        self.t_scan = 0.0
        orig_rebuild = solver._rebuild_all_propensities

        def rebuild(*a, **k):
            n = float(solver.a_tot)
            m = getattr(solver, "_particle_merger", None)
            hits = float(m._stats_merges) if m is not None else 0.0
            self.iters += 1
            self.sum_n += n
            self.sum_n2 += n * n
            self.sum_nm2 += (n + hits) ** 2
            t0 = time.perf_counter()
            r = orig_rebuild(*a, **k)
            self.t_rebuild += time.perf_counter() - t0
            return r
        solver._rebuild_all_propensities = rebuild

        merger = getattr(solver, "_particle_merger", None)
        if merger is not None:
            orig_find = merger.find_similar

            def find_similar(*a, **k):
                self.scan_n += float(solver.a_tot)
                t0 = time.perf_counter()
                r = orig_find(*a, **k)
                self.t_scan += time.perf_counter() - t0
                return r
            merger.find_similar = find_similar

    def row(self) -> list:
        return [self.iters, self.sum_n, self.sum_n2, self.t_rebuild, self.scan_n, self.t_scan, self.sum_nm2]


class SaveHook(list):
    """Ersetzt solver.V_save. solve() haengt pro Speicherpunkt genau einmal an
    (mcpbe_base, Schnappschuss-Schleife); dabei werden Rechenzeit, Merger-Zaehler und
    der MassTracker-Stand notiert. Rein beobachtend -- der Simulationsverlauf bleibt unberuehrt."""

    def __init__(self, items, solver, tracker: MassTracker, ops: OpCounter, label: str, t_vec: np.ndarray):
        super().__init__(items)
        self.solver, self.tracker, self.ops, self.label, self.t_vec = solver, tracker, ops, label, t_vec
        self.t0 = self.c0 = 0.0
        self.rows = [self._row(0.0, 0.0)]          # Speicherpunkt t = 0 (vor solve)

    def start(self) -> None:
        self.t0, self.c0 = time.perf_counter(), time.process_time()

    def _row(self, wall: float, cpu: float) -> list:
        m = getattr(self.solver, "_particle_merger", None)
        b = getattr(self.solver, "_break_debug_stats", None) or {}
        return [wall, cpu,
                m._stats_lookups if m else 0, m._stats_merges if m else 0, m._stats_creates if m else 0,
                b.get("merged", 0), b.get("created_new", 0), *self.tracker.row(), *self.ops.row()]

    def append(self, item) -> None:
        super().append(item)
        wall, cpu = time.perf_counter() - self.t0, time.process_time() - self.c0
        self.rows.append(self._row(wall, cpu))
        i = len(self) - 1
        t = float(self.t_vec[i]) if i < len(self.t_vec) else float("nan")
        if i == len(self.t_vec) - 1 or t % PROGRESS_EVERY == 0.0:
            print(f"  [{self.label}] t = {t:5.0f} s   n_comp = {np.shape(item)[1]:6d}   "
                  f"Wand {wall:8.0f} s", flush=True)


STAMP_COLS = ["Wand [s]", "CPU [s]", "Merger Suchen", "Merger Treffer", "Merger neu",
              "Bruch Treffer", "Bruch neu",
              "Agg Kind aufs Elternteil gemergt", "dM Agg Eltern-Merge (rel.)",
              "Agg Kind auf anderes gemergt", "dM Agg andere Merges (rel.)",
              "Agg Kind neu angelegt", "dM Agg neu angelegt (rel.)",
              "dM Bruch-Events (rel.)", "dM Nucleation (rel.)",
              "Iterationen", "sum n (Iterationen)", "sum n^2 (Iterationen)", "Zeit Rebuild [s]",
              "Merger-Suche: verglichene Partikel", "Zeit Merger-Suche [s]",
              "sum (n+m)^2 (ohne Merger, geschaetzt)"]
DM_STAMP = (8, 10, 12, 13, 14)   # Spalten in stamps, die Feststoffaenderungen [m3/m3] sind -> / M(0)


def solid_per_vc(V: np.ndarray, W: np.ndarray, Vc: float) -> float:
    return float(np.sum(np.asarray(W, float) * np.asarray(V, float)[0]) / Vc)


# ===========================================================================
#  Ein Replikat (laeuft im Worker-Prozess)
# ===========================================================================
def run_replicate(task: dict) -> dict:
    import warnings
    warnings.simplefilter("ignore")
    out_file = Path(task["out_file"])
    ad, pop, t_vec = make_adapter(task["cand"], task["smoke"])
    solver = ad._build_solver(t_vec, seed=task["seed"])
    tol = apply_merger(solver, task["tau"])
    tracker = MassTracker(solver)
    ops = OpCounter(solver)
    hook = SaveHook(solver.V_save, solver, tracker, ops, f"{task['setting']} s{task['seed']}", t_vec)
    solver.V_save = hook
    hook.start()
    solver.solve(maxiter=int(pop["MAXITER"]), max_particles=int(pop["MAX_PARTICLES"]))
    wall, cpu = time.perf_counter() - hook.t0, time.process_time() - hook.c0

    n = len(solver.V_save)
    if getattr(solver, "_hit_particle_limit", False):
        status = "max_particles"
    elif n == len(t_vec):
        status = "ok"
    else:
        status = "unvollstaendig"
    data = {"setting": task["setting"], "tau": np.nan if task["tau"] is None else float(task["tau"]),
            "seed": int(task["seed"]), "status": status, "wall": wall, "cpu": cpu, "t_vec": t_vec,
            "n_saved": n, "cand": json.dumps(task["cand"]), "tol": json.dumps(tol),
            "Vc": np.asarray(solver.Vc_save[:n], dtype=float),
            "agg_events": np.asarray(solver.real_agg_events_save[:n], dtype=float),
            "break_events": np.asarray(solver.real_break_events_save[:n], dtype=float),
            "stamps": np.asarray(hook.rows[:n], dtype=float)}
    for i in range(n):
        data[f"V_{i}"] = np.asarray(solver.V_save[i], dtype=float)          # [V_solid, V_dry]
        data[f"W_{i}"] = np.asarray(solver.W_save[i], dtype=float)
        data[f"poro_{i}"] = np.asarray(solver.porosity_save[i], dtype=float)
        data[f"sat_{i}"] = np.asarray(solver.saturation_save[i], dtype=float)
        data[f"liq_{i}"] = np.asarray(solver.liquid_volume_save[i], dtype=float)
    m0 = solid_per_vc(data["V_0"], data["W_0"], data["Vc"][0])
    m1 = solid_per_vc(data[f"V_{n - 1}"], data[f"W_{n - 1}"], data["Vc"][n - 1])
    solver._close()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **data)
    os.replace(tmp, out_file)
    return {"setting": task["setting"], "seed": task["seed"], "status": status, "wall": wall, "cpu": cpu,
            "n_comp": int(np.shape(data[f"W_{n - 1}"])[0]), "t": float(t_vec[n - 1]), "dM": m1 / m0 - 1.0}


# ===========================================================================
#  Simulation
# ===========================================================================
def simulate(args) -> None:
    out = run_dir(args)
    cand = run_candidate(args)
    npz_dir = out / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    ad, _, _ = make_adapter(cand, args.smoke)
    seeds = ad._resolve_mc_seeds()
    n_rep, n_aus = (2, 1) if args.smoke else (args.rep, args.rep_aus)
    if max(n_rep, n_aus) > len(seeds):
        raise ValueError(f"hoechstens {len(seeds)} Replikate (Fit-Seeds).")

    tasks = []
    for name, tau in SETTINGS.items():          # "aus" zuerst: vermutlich der laengste Lauf
        for s in seeds[: (n_aus if tau is None else n_rep)]:
            f = npz_dir / f"{name}_s{s}.npz"
            if not f.exists():
                tasks.append({"setting": name, "tau": tau, "seed": int(s), "cand": cand,
                              "smoke": args.smoke, "out_file": str(f)})
    print(f"Kandidat id {cand['id']} (Score {cand['score']:.6g}) aus {cand['sqlite']}", flush=True)
    print(f"{len(tasks)} offene Replikate, {DT_SAVE:g}-s-Speicherraster, Ausgabe {out}", flush=True)
    if not tasks:
        return

    def report(r: dict) -> None:
        print(f"FERTIG {r['setting']:9s} Seed {r['seed']:>10}: {r['status']:14s} bis t = {r['t']:5.0f} s  "
              f"Wand {r['wall']:8.0f} s  CPU {r['cpu']:8.0f} s  n_comp {r['n_comp']:6d}  "
              f"dM {r['dM']:+.3e}", flush=True)

    workers = max(1, min(args.workers, len(tasks)))
    if workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(run_replicate, t) for t in tasks]
                for fut in as_completed(futures):
                    report(fut.result())
            return
        except Exception as exc:  # BrokenProcessPool, Pickling-/Import-Fehler im Kindprozess, ...
            # z. B. Spyder unter Windows ohne externes Terminal: Mehrprozess geht nicht ->
            # die noch fehlenden Replikate nacheinander rechnen (fertige npz bleiben erhalten).
            # Ein echter Fehler in der Simulation taucht dann unten erneut auf.
            print(f"  Parallelbetrieb nicht moeglich ({type(exc).__name__}: {exc}) "
                  f"-> rechne die restlichen Replikate nacheinander.", flush=True)
    for t in tasks:
        if not Path(t["out_file"]).exists():
            report(run_replicate(t))


# ===========================================================================
#  Auswertung: Kennzahlen, Excel, Grafiken
# ===========================================================================
def load_runs(npz_dir: Path, seeds: list) -> dict:
    """Einstellung -> Liste der Replikate (npz), in Fit-Seed-Reihenfolge."""
    runs = {}
    for name in SETTINGS:
        files = [npz_dir / f"{name}_s{s}.npz" for s in seeds]
        runs[name] = [np.load(f, allow_pickle=False) for f in files if f.exists()]
    return runs


def q3_curve(V: np.ndarray, W: np.ndarray, x_um: np.ndarray) -> np.ndarray:
    """Massenanteil mit Durchmesser < x (Sieb-Definition), Durchmesser aus V_dry."""
    d_um = np.cbrt(6.0 * V[-1] / np.pi) * 1e6
    mass = W * V[0]
    o = np.argsort(d_um)
    cum = np.r_[0.0, np.cumsum(mass[o])] / mass.sum()
    return cum[np.searchsorted(d_um[o], x_um, side="left")]


def evaluate(args) -> None:
    # Figure-Objekte statt pyplot: kein Backend-Wechsel (matplotlib.use("Agg") bliebe sonst in der
    # Spyder-Konsole haengen) und keine Aenderung der globalen rcParams.
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    out = run_dir(args)
    npz_dir = out / "npz"
    cand = json.loads((out / "kandidat.json").read_text(encoding="utf-8"))
    ad, _, t_vec = make_adapter(cand, args.smoke)
    seeds = ad._resolve_mc_seeds()
    runs = load_runs(npz_dir, seeds)
    for name, reps in runs.items():
        for z in reps:
            if json.loads(str(z["cand"]))["params"] != cand["params"]:
                raise RuntimeError(f"{name} s{int(z['seed'])}: npz mit anderem Kandidaten als kandidat.json.")
    i_end = len(t_vec) - 1
    mesh_w = (ad._mesh_weight if ad._mesh_weight is not None
              else np.full(len(ad._mesh_m), 1.0 / len(ad._mesh_m)))

    def complete(z) -> bool:
        return int(z["n_saved"]) == len(t_vec)

    def q3_mesh(z, i: int) -> np.ndarray:
        return ad._q3_at_meshes_from_snapshot(z[f"V_{i}"], z[f"W_{i}"], ad._mesh_m)

    def q3_fine(z, i: int) -> np.ndarray:
        return q3_curve(z[f"V_{i}"], z[f"W_{i}"], X_GRID_UM)

    def d_kf3(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(mesh_w, np.abs(a - b)))

    def d_max(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.max(np.abs(a - b)))

    def j_kf3(reps: list) -> float:
        """KF3-Kosten des Replikat-Mittels gegen das Experiment (wie im Adapter)."""
        if not reps:
            return float("nan")
        q3 = [np.vstack([q3_mesh(z, i) for i in range(len(t_vec))])[ad._obs_tidx] for z in reps]
        ad.data_mod = np.stack(q3).mean(axis=0)
        return float(ad.calc_delta_pop(None, ad._data_exp))

    def series(z) -> dict:
        n = int(z["n_saved"])
        Vc = z["Vc"]
        M = np.array([solid_per_vc(z[f"V_{i}"], z[f"W_{i}"], Vc[i]) for i in range(n)])
        return {"n": n, "n_comp": np.array([len(z[f"W_{i}"]) for i in range(n)]), "M": M,
                "dM": M / M[0] - 1.0}

    def stamps(z) -> np.ndarray:
        """stamps auf die aktuelle Spaltenzahl aufgefuellt (aeltere npz ohne Operationszaehler -> NaN)."""
        st = np.asarray(z["stamps"], dtype=float)
        if st.shape[1] < len(STAMP_COLS):
            st = np.hstack([st, np.full((st.shape[0], len(STAMP_COLS) - st.shape[1]), np.nan)])
        return st

    def stamp_row(z, s: dict, i: int) -> list:
        """stamps-Zeile i; die Feststoffaenderungen relativ zu M(t=0) wie dM."""
        row = stamps(z)[i].tolist()
        for c in DM_STAMP:
            row[c] = row[c] / s["M"][0]
        return row

    S = {name: [series(z) for z in reps] for name, reps in runs.items()}
    ok = {name: [z for z in reps if complete(z)] for name, reps in runs.items()}
    mesh_end = {name: [q3_mesh(z, i_end) for z in reps] for name, reps in ok.items()}
    fine_end = {name: [q3_fine(z, i_end) for z in reps] for name, reps in ok.items()}
    base_mesh = np.mean(mesh_end["aus"], axis=0) if mesh_end["aus"] else None
    base_fine = np.mean(fine_end["aus"], axis=0) if fine_end["aus"] else None
    base_by_seed = {int(z["seed"]): (q3_mesh(z, i_end), q3_fine(z, i_end)) for z in ok["aus"]}

    def nanmean(values) -> float:
        values = [v for v in values if v is not None and np.isfinite(v)]
        return float(np.mean(values)) if values else float("nan")

    # ---------------- Kennzahlen je Replikat ("Laeufe") und je Einstellung ("Uebersicht")
    OPS_HEADER = ["Iterationen", "sum n^2", "Zeit Rebuild [s]", "Merger-Suche: verglichene Partikel",
                  "Zeit Merger-Suche [s]", "sum (n+m)^2 ohne Merger (geschaetzt)",
                  "Ersparnis Rebuild Obergrenze [s]", "Netto Obergrenze [s]", "Merger-Suche Anteil CPU [%]",
                  "sum n^2 aus (gleicher Seed)", "Ersparnis Rebuild empirisch [s]", "Netto empirisch [s]"]
    # sum n^2 der "aus"-Laeufe je Seed: empirischer Vergleich (gleicher Seed, aber eigener MC-Pfad)
    sn2_aus = {int(z["seed"]): float(stamps(z)[-1][17]) for z in ok["aus"]}

    def ops_cols(st, cpu: float, seed: int, name: str) -> list:
        it, sn2, t_reb, scan, t_scan, snm2 = st[15], st[17], st[18], st[19], st[20], st[21]
        valid = np.isfinite(sn2) and sn2 > 0
        save = t_reb * (snm2 / sn2 - 1.0) if valid else float("nan")
        ref = sn2_aus.get(seed, float("nan")) if name != "aus" else float("nan")
        emp = t_reb * (ref / sn2 - 1.0) if valid and np.isfinite(ref) else float("nan")
        return [it, sn2, t_reb, scan, t_scan, snm2, save, save - t_scan, 100.0 * t_scan / cpu,
                ref, emp, emp - t_scan]

    laeufe = []
    for name, reps in runs.items():
        for z, s in zip(reps, S[name]):
            st = stamps(z)[-1]
            rel = stamp_row(z, s, s["n"] - 1)
            parts = [rel[c] for c in DM_STAMP]
            tol = json.loads(str(z["tol"]))
            seed = int(z["seed"])
            same = base_by_seed.get(seed) if complete(z) and name != "aus" else None
            laeufe.append([
                name, None if name == "aus" else SETTINGS[name], seed, str(z["status"]), s["n"],
                float(z["wall"]), float(z["cpu"]),
                tol["tol_rel"], tol["tol_abs_frac"], tol["nucleation_similarity_tol"],
                int(s["n_comp"][-1]), int(s["n_comp"].max()), float(s["dM"][-1]), float(np.abs(s["dM"]).max()),
                int(st[2]), int(st[3]), int(st[4]), int(st[5]), int(st[6]),
                float(z["agg_events"][-1]), float(z["break_events"][-1]),
                d_kf3(q3_mesh(z, i_end), same[0]) if same else None,
                d_max(q3_fine(z, i_end), same[1]) if same else None,
                j_kf3([z]) if complete(z) else None,
                int(st[7]), parts[0], int(st[9]), parts[1], int(st[11]), parts[2], parts[3], parts[4],
                float(s["dM"][-1]) - sum(parts),
                *ops_cols(st, float(z["cpu"]), seed, name),
            ])
    laeufe_header = ["Einstellung", "tau", "Seed", "Status", "Speicherpunkte", "Wand gesamt [s]", "CPU gesamt [s]",
                     "tol_rel (Agg/Bruch)", "tol_abs_frac (Poro/Saett.)", "similarity_tol (Nucleation)",
                     "n_comp Ende", "n_comp max", "dM Ende", "max |dM|",
                     "Merger Suchen", "Merger Treffer", "Merger neu", "Bruch Treffer", "Bruch neu",
                     "Agg-Events Ende", "Bruch-Events Ende",
                     "D_KF3 vs aus (gleicher Seed)", "D_max vs aus (gleicher Seed)", "J_KF3 (dieses Replikat)",
                     "Agg Kind aufs Elternteil gemergt", "dM Agg Eltern-Merge (rel.)",
                     "Agg Kind auf anderes gemergt", "dM Agg andere Merges (rel.)",
                     "Agg Kind neu angelegt", "dM Agg neu angelegt (rel.)",
                     "dM Bruch-Events (rel.)", "dM Nucleation (rel.)", "dM Rest (rel.)",
                     *OPS_HEADER]

    uebersicht = []
    for name, reps in runs.items():
        rows = [r for r in laeufe if r[0] == name]
        okr = [r for r in rows if r[3] == "ok"]
        pairs_mesh = [d_kf3(a, b) for a, b in itertools.combinations(mesh_end[name], 2)]
        pairs_fine = [d_max(a, b) for a, b in itertools.combinations(fine_end[name], 2)]
        mean_mesh = np.mean(mesh_end[name], axis=0) if mesh_end[name] else None
        mean_fine = np.mean(fine_end[name], axis=0) if fine_end[name] else None
        cmp_ok = name != "aus" and mean_mesh is not None and base_mesh is not None
        uebersicht.append([
            name, None if name == "aus" else SETTINGS[name], len(okr), len(rows),
            nanmean(r[5] for r in okr), nanmean(r[6] for r in okr),
            nanmean(r[10] for r in okr), max((r[11] for r in rows), default=None),
            nanmean(r[12] for r in okr), max((r[13] for r in rows), default=None),
            nanmean(r[15] for r in okr), nanmean(r[15] / r[14] for r in okr if r[14] > 0),
            d_kf3(mean_mesh, base_mesh) if cmp_ok else None,
            d_max(mean_fine, base_fine) if cmp_ok else None,
            nanmean(r[21] for r in okr), nanmean(r[22] for r in okr),
            nanmean(pairs_mesh), nanmean(pairs_fine), len(pairs_mesh),
            j_kf3(ok[name]),
            nanmean(r[24] for r in okr), nanmean(r[25] for r in okr), nanmean(r[32] for r in okr),
            *[nanmean(r[33 + k] for r in okr) for k in range(len(OPS_HEADER))],
        ])
    uebersicht_header = ["Einstellung", "tau", "Replikate ok", "Replikate gesamt",
                         "Wand Mittel [s]", "CPU Mittel [s]", "n_comp Ende Mittel", "n_comp max",
                         "dM Ende Mittel", "max |dM|", "Merger Treffer Mittel", "Trefferquote Mittel",
                         "D_KF3 Mittel vs Mittel aus", "D_max Mittel vs Mittel aus",
                         "D_KF3 gleicher Seed (Mittel)", "D_max gleicher Seed (Mittel)",
                         "D_KF3 Rauschen (Replikat vs Replikat)", "D_max Rauschen (Replikat vs Replikat)",
                         "Rauschen: Anzahl Paare", "J_KF3 (Mittel der Replikate)",
                         "Agg Eltern-Merges Mittel", "dM Agg Eltern-Merge Mittel (rel.)", "dM Rest Mittel (rel.)",
                         *[f"{h} Mittel" for h in OPS_HEADER]]

    # ---------------- Excel-Blaetter (Zeilen-Generatoren: einmal schreiben, einmal pruefen)
    def zeitreihen_rows():
        for name, reps in runs.items():
            for z, s in zip(reps, S[name]):
                for i in range(s["n"]):
                    yield [name, None if name == "aus" else SETTINGS[name], int(z["seed"]), str(z["status"]),
                           float(t_vec[i]), int(s["n_comp"][i]), float(z["Vc"][i]), float(s["M"][i]),
                           float(s["dM"][i]), float(z["agg_events"][i]), float(z["break_events"][i]),
                           *stamp_row(z, s, i)]
    zeitreihen_header = ["Einstellung", "tau", "Seed", "Status", "t [s]", "n_comp", "V_c [m3]",
                         "M = sum(W*V_solid)/V_c [m3/m3]", "dM = M/M(0) - 1", "Agg-Events (kum.)",
                         "Bruch-Events (kum.)", *STAMP_COLS]

    q3_cols = [("x [um]", X_GRID_UM)]
    for name in SETTINGS:
        if fine_end[name]:
            q3_cols.append((f"{name} Mittel", np.mean(fine_end[name], axis=0)))
            q3_cols += [(f"{name} s{int(z['seed'])}", q) for z, q in zip(ok[name], fine_end[name])]

    def q3_rows():
        for k in range(len(X_GRID_UM)):
            yield [float(col[k]) for _, col in q3_cols]

    tol_text = ("tau = alle Merger-Toleranzen gleich: _fragment_merge_tol (Agg+Bruch, V_dry + Fluessigkeit "
                "relativ), nucleation.similarity_tol (Nucleation; Fluessigkeit zusaetzlich <= tau*V_Tropfen), "
                "ParticleMerger.tol_abs_frac (Porositaet + Saettigung absolut). Fit: 1e-6 / 1e-5 / 1e-6.")
    info = [["Kandidat id", cand["id"]], ["Fit-Score J (KF3, 10 Replikate)", cand["score"]],
            ["Quelle", cand["sqlite"]], ["Config", str(CONFIG.name)], ["Toleranzen", tol_text],
            ["Nulllinie", "aus = _enable_particle_merging False (Agg, Bruch, Nucleation legen immer neu an)"],
            ["Speicherraster", f"alle {DT_SAVE:g} s, 0..{t_vec[-1]:g} s"],
            ["dM", "M = sum(W * V_solid) / V_c je Speicherpunkt; dM = M / M(t=0) - 1"],
            ["D_KF3", "sum_k m_k |Q3_a(x_k) - Q3_b(x_k)| an den KF3-Siebmaschen, KF3-Maschengewichte, t = 600 s"],
            ["D_max", "max_x |Q3_a(x) - Q3_b(x)|, x = 20..3000 um (800 Punkte, log), t = 600 s"],
            ["Rauschen", "Mittel ueber alle Replikat-Paare derselben Einstellung"],
            ["J_KF3", "KF3-Kosten gegen das Experiment (Mittel der Replikate, wie im Fit)"],
            ["Q3", "Masse = W * V_solid, Durchmesser aus V_dry, Sieb-Definition d < x"],
            ["Rechenzeit", "Wand und CPU seit Start von solve(); parallele Prozesse teilen sich die Maschine"],
            ["Eltern-Merge", "Agg-Kind (i + j) faellt innerhalb tau auf ein eigenes Elternteil -> dessen W += dW, "
                             "danach -dW von i und j: die dW Partner-Partikel verschwinden samt Masse"],
            ["dM Rest", "dM minus Agg-, Bruch- und Nucleation-Anteil = kontinuierliche Prozesse, "
                        "Vc-Verdopplung, Rundung"],
            ["BREAK_P1-Faktor (numerische Bruch-Variante)", float(cand.get("bruch_faktor", 1.0))],
            ["Smoke-Test", bool(args.smoke)], ["Erstellt", datetime.now().isoformat(timespec="seconds")]]
    info += [[f"Parameter {k}", v] for k, v in cand["params"].items()]

    sheets = [("Info", ["Feld", "Wert"], lambda: iter(info)),
              ("Uebersicht", uebersicht_header, lambda: iter(uebersicht)),
              ("Laeufe", laeufe_header, lambda: iter(laeufe)),
              ("Zeitreihen", zeitreihen_header, zeitreihen_rows),
              ("Q3_600s", [c for c, _ in q3_cols], q3_rows)]
    part_header = ["Seed", "t [s]", "Nr.", "d_aussen [um]", "V_solid [m3]", "V_dry [m3]", "W",
                   "Porositaet", "Saettigung", "Fluessigkeit [m3]"]
    for name, reps in runs.items():
        total = sum(int(sum(s["n_comp"])) for s in S[name])
        if total == 0:
            continue

        def part_rows(reps=reps):
            for z in reps:
                seed = int(z["seed"])
                for i in range(int(z["n_saved"])):
                    V, W = z[f"V_{i}"], z[f"W_{i}"]
                    d = np.cbrt(6.0 * V[1] / np.pi) * 1e6
                    cols = zip(d.tolist(), V[0].tolist(), V[1].tolist(), W.tolist(), z[f"poro_{i}"].tolist(),
                               z[f"sat_{i}"].tolist(), z[f"liq_{i}"].tolist())
                    t = float(t_vec[i])
                    for k, c in enumerate(cols, start=1):
                        yield [seed, t, k, *c]
        n_chunks = math.ceil(total / EXCEL_MAX_ROWS)
        for c in range(n_chunks):
            sheet = f"Partikel_{name}" + (f"_{c + 1}" if c else "")
            sheets.append((sheet, part_header,
                           lambda f=part_rows, c=c: itertools.islice(f(), c * EXCEL_MAX_ROWS,
                                                                     (c + 1) * EXCEL_MAX_ROWS)))

    xlsx = out / "merger_tol_test.xlsx"
    t0 = time.perf_counter()
    write_excel(xlsx, sheets)
    n_cells = check_excel(xlsx, sheets)
    print(f"Excel: {xlsx}  ({time.perf_counter() - t0:.0f} s, {n_cells} Zellen bitweise zurueckgelesen: gleich)")
    for row in uebersicht:
        print("  " + "  ".join(f"{h}={v:.4g}" if isinstance(v, float) else f"{h}={v}"
                               for h, v in zip(uebersicht_header, row)))

    # ---------------- Grafiken
    tag = (f"KF3-Kandidat id {cand['id']}"
           + (f", BREAK_P1 x {cand['bruch_faktor']:g}" if "bruch_faktor" in cand else "")
           + ("  [SMOKE-TEST]" if args.smoke else ""))

    def handle(name: str, extra: str = "") -> Line2D:
        color, marker, ls = STYLE[name]
        return Line2D([], [], color=color, marker=marker, ls=ls, lw=1.8, ms=7, label=f"{name}{extra}")

    def grid(*axs) -> None:
        for ax in axs:
            ax.grid(True, color="#e6e6e6", linewidth=0.6)
            ax.set_axisbelow(True)

    def padded(arrs: list) -> np.ndarray:
        out_ = np.full((len(arrs), len(t_vec)), np.nan)
        for r, a in enumerate(arrs):
            out_[r, : len(a)] = a
        return out_

    fig = Figure(figsize=(13, 5))
    axes = fig.subplots(1, 2)
    grid(*axes)
    handles = []
    for idx, (name, ser) in enumerate(S.items()):
        if not ser:
            continue
        color, marker, ls = STYLE[name]
        for ax, key in ((axes[0], "n_comp"), (axes[1], "dM")):
            y = padded([np.abs(s[key]) if key == "dM" else s[key] for s in ser])
            if key == "dM":
                y = np.where(np.isfinite(y), np.maximum(y, DM_FLOOR), np.nan)
            mean = np.nanmean(y, axis=0)
            if len(ser) > 1:
                ax.fill_between(t_vec, np.nanmin(y, axis=0), np.nanmax(y, axis=0), color=color, alpha=0.12, lw=0)
            ax.plot(t_vec, mean, color=color, ls=ls, lw=1.8, marker=marker, ms=7,
                    markevery=(2 * idx, 12), mec="white", mew=0.8)
        n_ok = sum(1 for z in runs[name] if complete(z))
        handles.append(handle(name, f"  ({n_ok}/{len(runs[name])} fertig)"))
    axes[0].set_yscale("log")
    axes[0].set_ylabel("n_comp (Rechenpartikel)")
    axes[1].set_yscale("log")
    axes[1].set_ylabel(f"|dM| = |M/M(0) - 1|   (Werte < {DM_FLOOR:g}, auch 0, auf {DM_FLOOR:g})")
    for ax in axes:
        ax.set_xlabel("t_sim [s]")
        ax.set_xlim(0, t_vec[-1])
    axes[0].set_title("Rechenpartikel", fontsize=10)
    axes[1].set_title("Feststoffmassen-Defekt", fontsize=10)
    fig.suptitle(f"Merger-Toleranz-Test: Mittel der Replikate, Band = min..max   ({tag})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.83, 0.95))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.835, 0.9), title="Einstellung", fontsize=9)
    fig.savefig(out / "Zeitverlauf.png", dpi=130, bbox_inches="tight")

    fig = Figure(figsize=(11, 8.5))
    ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    grid(ax1, ax2)
    handles = []
    for idx, name in enumerate(SETTINGS):
        if not fine_end[name]:
            continue
        color, marker, ls = STYLE[name]
        mean = np.mean(fine_end[name], axis=0)
        kw = dict(color=color, ls=ls, lw=1.8, marker=marker, ms=7, markevery=(16 * idx, 80),
                  mec="white", mew=0.8, drawstyle="steps-post")
        ax1.plot(X_GRID_UM, mean, **kw)
        if name != "aus" and base_fine is not None:
            ax2.plot(X_GRID_UM, mean - base_fine, **kw)
        handles.append(handle(name, f"  ({len(fine_end[name])} Replikat(e))"))
    ax2.axhline(0.0, color="#52514e", lw=0.8)
    ax1.set_xscale("log")
    ax1.set_xlim(X_GRID_UM[0], X_GRID_UM[-1])
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_ylabel("Q3(x) [-]  (Mittel der Replikate)")
    ax2.set_ylabel("Q3 - Q3(aus) [-]")
    ax2.set_xlabel("Durchmesser x [um] (log)")
    ax1.set_title(f"Q3 bei t = {t_vec[-1]:g} s", fontsize=10)
    ax2.set_title("Differenz zur Nulllinie (Merger aus)" if base_fine is not None
                  else "Differenz: Nulllinie (aus) noch nicht fertig", fontsize=10)
    fig.suptitle(f"Merger-Toleranz-Test ({tag})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.8, 0.96))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.805, 0.92), title="Einstellung", fontsize=9)
    fig.savefig(out / "Q3_600s.png", dpi=130, bbox_inches="tight")
    print(f"Grafiken: {out / 'Zeitverlauf.png'}, {out / 'Q3_600s.png'}")


# ===========================================================================
#  Excel exakt schreiben + bitweise pruefen
# ===========================================================================
def _cell(v):
    if v is None or isinstance(v, str):
        return v
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    v = float(v)
    return v if math.isfinite(v) else None      # NaN/inf -> leere Zelle


def write_excel(path: Path, sheets: list) -> None:
    """openpyxl schreibt Gleitkommazahlen sonst mit nur 16 Stellen (%.16g) -> repr()."""
    import openpyxl
    import openpyxl.cell._writer as xw
    original = xw.safe_string
    xw.safe_string = lambda x: repr(x) if isinstance(x, float) and math.isfinite(x) else original(x)
    try:
        wb = openpyxl.Workbook(write_only=True)
        for name, header, rows in sheets:
            ws = wb.create_sheet(name)
            ws.freeze_panes = "A2"
            ws.append(header)
            for r in rows():
                ws.append([_cell(v) for v in r])
        tmp = path.with_name(path.stem + ".tmp.xlsx")
        wb.save(tmp)
        os.replace(tmp, path)
    finally:
        xw.safe_string = original


def check_excel(path: Path, sheets: list) -> int:
    """Liest jede Zelle zurueck und vergleicht Gleitkommazahlen bitweise mit der Quelle."""
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True)
    n = 0
    try:
        for name, header, rows in sheets:
            back_rows = wb[name].iter_rows(values_only=True)
            if list(next(back_rows))[: len(header)] != list(header):
                raise RuntimeError(f"Excel-Pruefung: Kopfzeile von {name} weicht ab.")
            for r, (src, back) in enumerate(itertools.zip_longest(rows(), back_rows), start=2):
                if src is None or back is None:
                    raise RuntimeError(f"Excel-Pruefung: {name} hat eine andere Zeilenzahl (Zeile {r}).")
                src = [_cell(v) for v in src]
                back = list(back) + [None] * max(0, len(src) - len(back))
                if any(v is not None for v in back[len(src):]):
                    raise RuntimeError(f"Excel-Pruefung: {name} Zeile {r} hat ueberzaehlige Zellen.")
                for a, b in zip(src, back):
                    same = (struct.pack("<d", a) == struct.pack("<d", b)) if isinstance(a, float) and \
                        isinstance(b, float) else (a == b and type(a) is type(b))
                    if not same:
                        raise RuntimeError(f"Excel-Pruefung: {name} Zeile {r}: {a!r} != {b!r}")
                    n += 1
    finally:
        wb.close()
    return n


# ===========================================================================
#  SPYDER / "Play" (keine Kommandozeilen-Argumente): diese Liste wird benutzt.
#  Ohne Schritt-Angabe laufen nacheinander "sim" und "auswertung".
#  Der Kandidat steht in kandidat.json (einmal mit --sqlite angelegt).
#  Funktionstest zuerst: "--smoke" an die Liste anhaengen.
# ===========================================================================
# Lauf "seriell": alle 25 Replikate nacheinander (saubere Rechenzeiten + Operationszaehler),
# eigener Unterordner seriell/ -> die bisherigen Ergebnisse bleiben unberuehrt.
SPYDER_ARGS: list[str] = ["--workers", "1", "--unterordner", "seriell", "--rep-aus", "5"]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):   # Spyder-Konsole (TTYOutStream) kann das nicht
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("schritt", nargs="?", choices=["sim", "auswertung"], default=None,
                    help="ohne Angabe: erst sim, dann auswertung")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--rep", type=int, default=REPLIKATE)
    ap.add_argument("--rep-aus", type=int, default=REPLIKATE_AUS)
    ap.add_argument("--sqlite", default=None, help="absolut oder relativ zum Repo-Ordner")
    ap.add_argument("--id", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--unterordner", default="", help="Ergebnisse in diesen Unterordner")
    ap.add_argument("--bruch-faktor", type=float, default=1.0, help="BREAK_P1 x Faktor, eigener Unterordner")
    argv = sys.argv[1:] if len(sys.argv) > 1 else SPYDER_ARGS
    if not sys.argv[1:]:
        print(f"[Spyder/Play-Modus] keine Argumente -> SPYDER_ARGS = {SPYDER_ARGS}", flush=True)
    args = ap.parse_args(argv)
    if args.id is not None and args.sqlite is None:
        ap.error("--id braucht --sqlite")
    if args.sqlite and not Path(args.sqlite).is_absolute():
        args.sqlite = str(REPO / args.sqlite)
    if args.schritt in (None, "sim"):
        simulate(args)
    if args.schritt in (None, "auswertung"):
        evaluate(args)


if __name__ == "__main__":
    main()
