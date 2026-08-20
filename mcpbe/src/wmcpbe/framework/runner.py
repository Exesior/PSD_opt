"""N Wiederholungen desselben Setups mit unabhaengigen Seeds.

Vier Bausteine, die aus ``upstream/dev_monorepo`` uebernommen sind:

1. **Unabhaengige Seeds.** Nicht ``seed = base + i`` (benachbarte Seeds koennen
   korrelierte Zufallsstroeme erzeugen), sondern ``SeedSequence.spawn(N)``.
   Nur so sind die Wiederholungen statistisch wirklich unabhaengig -- und nur
   dann bedeutet ein Standardfehler das, was er zu bedeuten vorgibt.
2. **Reduktion im Worker.** Der Worker gibt Kennzahlen zurueck, nicht den
   Solver (siehe ``metrics.py``).
3. **Backpressure.** Es liegen nie mehr als ``workers`` Auftraege gleichzeitig
   in der Warteschlange. Bei vielen Wiederholungen bleibt der Speicherbedarf
   damit konstant statt mit N zu wachsen.
4. **Wiederaufsetzbarkeit.** Jede fertige Wiederholung schreibt sofort ihre
   eigene ``.npz``-Datei -- atomar (erst ``.tmp``, dann umbenennen), damit ein
   Abbruch nie eine halbe Datei hinterlaesst. Beim Neustart werden fertige
   Laeufe eingelesen und nur der Rest gerechnet.

Unterschied zur Vorlage des Betreuers: dort wird der Solver-Zustand per
``copy.deepcopy(self.__dict__)`` an den Worker geschickt. Hier wandert nur das
Rezept, und der Worker baut den Solver selbst -- Begruendung in ``builder.py``.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from . import bootstrap_paths, pin_threads
from .builder import resolve_params


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
def build_seeds(n_repeats: int, base_seed: int = 42) -> List[np.random.SeedSequence]:
    """Erzeuge N garantiert unabhaengige Seed-Sequenzen aus einem Basiswert.

    Derselbe ``base_seed`` liefert immer dieselben N Sequenzen -- die ganze
    Kampagne ist damit reproduzierbar, obwohl jeder einzelne Lauf zufaellig ist.
    """
    return list(np.random.SeedSequence(int(base_seed)).spawn(int(n_repeats)))


def seed_label(seed_seq: np.random.SeedSequence) -> str:
    """Kurzer, stabiler Name einer Seed-Sequenz fuer Dateinamen und CSV."""
    return "-".join(str(k) for k in seed_seq.spawn_key) or "root"


# ---------------------------------------------------------------------------
# Identitaet eines Laufs (fuer die Wiederaufnahme)
# ---------------------------------------------------------------------------
def params_fingerprint(params: Dict[str, Any]) -> str:
    """Kurzer Hash ueber die Parameterwerte.

    Aendert sich ein Parameter, aendert sich der Fingerabdruck -- alte
    Ergebnisse werden dann nicht faelschlich wiederverwendet.
    """
    resolved = resolve_params(params)
    payload = json.dumps(resolved, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def task_id(params: Dict[str, Any], index: int, seed_seq: np.random.SeedSequence) -> str:
    return f"{params_fingerprint(params)}_r{index:03d}_s{seed_label(seed_seq)}"


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _worker_init() -> None:
    """Laeuft einmal je Worker-Prozess, bevor der erste Auftrag kommt."""
    pin_threads()
    bootstrap_paths()


def run_single(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Ein kompletter Lauf. Laeuft im Worker-Prozess (oder seriell im Haupt).

    Wichtig: Modulebene, keine Methode. Windows startet Worker mit ``spawn``,
    dabei muss die Funktion ueber ihren Modulpfad auffindbar sein.
    """
    _worker_init()

    from .builder import build_reference_solver
    from .metrics import reduce_solver

    params = payload["params"]
    seed_seq = payload["seed"]
    index = int(payload["index"])

    # Zwei Zeitmessungen, weil sie Verschiedenes aussagen:
    #   wall_time_s -- verstrichene Uhrzeit. Bei mehreren Workern auf zu wenigen
    #                  Kernen enthaelt sie Wartezeit und ist fuer Vergleiche
    #                  zwischen Konfigurationen unbrauchbar.
    #   cpu_time_s  -- tatsaechlich verbrauchte Rechenzeit dieses Prozesses.
    #                  Bleibt unter Konkurrenz belastbar; das ist die Groesse
    #                  fuer "welche Einstellung ist teurer".
    t_wall = time.time()
    t_cpu = time.process_time()
    solver = build_reference_solver(params, seed=seed_seq, verbose=bool(payload.get("verbose", False)))
    solver.solve(maxiter=int(params.get("maxiter", int(1e9))))
    elapsed = time.time() - t_wall
    elapsed_cpu = time.process_time() - t_cpu

    reduced = reduce_solver(solver, params)
    reduced["scalars"]["wall_time_s"] = float(elapsed)
    reduced["scalars"]["cpu_time_s"] = float(elapsed_cpu)
    reduced["index"] = index
    reduced["seed_label"] = seed_label(seed_seq)
    reduced["task_id"] = str(payload["task_id"])

    npz_path = payload.get("npz_path")
    if npz_path:
        _save_record(reduced, npz_path)

    return reduced


# ---------------------------------------------------------------------------
# Persistenz
# ---------------------------------------------------------------------------
def _save_record(record: Dict[str, Any], npz_path: str) -> None:
    """Schreibe eine Wiederholung atomar auf die Platte.

    Erst in eine ``.tmp``-Datei, dann umbenennen: ``os.replace`` ist atomar,
    also existiert die Zieldatei entweder vollstaendig oder gar nicht. Ein
    Abbruch mitten im Schreiben kann so keine halbe Datei hinterlassen, die
    beim naechsten Start als "fertig" gelesen wuerde.
    """
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    payload: Dict[str, Any] = {}
    for key, value in record["series"].items():
        payload[f"series__{key}"] = np.asarray(value, dtype=float)
    for key, value in record["scalars"].items():
        payload[f"scalar__{key}"] = np.asarray(float(value), dtype=float)
    payload["meta__index"] = np.asarray(int(record["index"]), dtype=np.int64)
    payload["meta__seed_label"] = np.asarray(str(record["seed_label"]))
    payload["meta__task_id"] = np.asarray(str(record["task_id"]))

    # Die Endung ".npz" muss stehen bleiben: np.savez_compressed haengt sie
    # sonst selbst an und schreibt in eine andere Datei als erwartet.
    tmp_path = npz_path[: -len(".npz")] + ".tmp.npz"
    np.savez_compressed(tmp_path, **payload)
    os.replace(tmp_path, npz_path)


def _load_record(npz_path: str) -> Optional[Dict[str, Any]]:
    """Lies eine gespeicherte Wiederholung zurueck; ``None`` bei Defekt."""
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            series = {
                key[len("series__"):]: np.asarray(data[key], dtype=float)
                for key in data.files
                if key.startswith("series__")
            }
            scalars = {
                key[len("scalar__"):]: float(data[key])
                for key in data.files
                if key.startswith("scalar__")
            }
            return {
                "series": series,
                "scalars": scalars,
                "index": int(data["meta__index"]),
                "seed_label": str(data["meta__seed_label"]),
                "task_id": str(data["meta__task_id"]),
            }
    except Exception:
        return None


def _warn_about_foreign_fingerprints(output_dir: str, current: str, n_repeats: int) -> None:
    """Melde, wenn im Pool Ergebnisse mit anderem Fingerabdruck liegen.

    Ohne diesen Hinweis sieht ein vollstaendiger Neustart der Rechnung
    aus wie ein Fehler. Tatsaechlich ist es die richtige Reaktion: der
    Fingerabdruck deckt ALLE Parameter ab, also macht schon das Hinzufuegen
    eines neuen Schluessels zu ``REFERENCE_PARAMS`` die alten Ergebnisse
    formal ungueltig -- auch wenn der neue Wert dem bisherigen Verhalten
    entspricht. Lieber einmal zu viel rechnen als zwei Konfigurationen
    stillschweigend vermischen.
    """
    pool_dir = os.path.join(output_dir, "repeats")
    if not os.path.isdir(pool_dir):
        return
    others: Dict[str, int] = {}
    for name in os.listdir(pool_dir):
        if not name.endswith(".npz"):
            continue
        prefix = name.split("_r")[0]
        if prefix and prefix != current:
            others[prefix] = others.get(prefix, 0) + 1
    if not others:
        return
    listed = ", ".join(f"{key} ({count} Dateien)" for key, count in sorted(others.items()))
    print(
        f"[repeats] HINWEIS: im Pool liegen Ergebnisse mit anderem Parameter-Fingerabdruck "
        f"({listed}), aktuell ist '{current}'. Es wird deshalb komplett neu gerechnet "
        f"({n_repeats} Laeufe). Ursache ist eine geaenderte oder neu hinzugefuegte "
        f"Parameterangabe -- nicht ein Fehler im Cache.",
        flush=True,
    )


def _run_single_process_entry(payload: Dict[str, Any]) -> None:
    """Prozess-Einstiegspunkt fuer den Zeitlimit-Pfad.

    Gibt bewusst nichts ueber IPC zurueck: ``run_single`` speichert sein
    Ergebnis bereits selbst atomar als ``.npz`` (siehe ``_save_record``). Der
    Elternprozess erkennt Erfolg daran, dass genau diese Datei nach Prozessende
    existiert -- unabhaengig davon, ob der Prozess sauber durchgelaufen oder
    per Zeitlimit hart beendet wurde.
    """
    _worker_init()
    try:
        run_single(payload)
    except Exception:
        traceback.print_exc()


def _run_repeats_with_timeout(
    payloads: List[Dict[str, Any]],
    workers: int,
    max_wall_time_s: float,
    records: List[Optional[Dict[str, Any]]],
    done_count: int,
    n_repeats: int,
    on_progress: Optional[Callable[[Dict[str, Any], int, int], None]],
) -> int:
    """Wie der normale Parallelpfad, aber mit hartem Zeitlimit je Wiederholung.

    ``ProcessPoolExecutor`` eignet sich dafuer nicht: ``future.result(timeout=...)``
    gibt zwar die Kontrolle an den Aufrufer zurueck, die haengende Berechnung
    laeuft im Worker-Prozess des Pools unbegrenzt weiter und blockiert diesen
    Platz fuer den Rest der Kampagne -- bei 250 geplanten Laeufen wuerde ein
    einziger haengender Fall einen von wenigen parallelen Slots dauerhaft
    kosten. Deshalb bekommt hier jede Wiederholung einen eigenen
    ``multiprocessing.Process``, der bei Ueberschreitung hart beendet wird
    (``terminate()``, nach kurzer Frist ``kill()``) und den Platz danach frei
    gibt.

    Eine per Zeitlimit abgebrochene Wiederholung hinterlaesst keine ``.npz``
    (der letzte Schreibschritt ist atomar, siehe ``_save_record`` -- entweder
    ganz oder gar nicht). Sie taucht deshalb im Ergebnis einfach nicht auf,
    genau wie eine Wiederholung, die nie gestartet wurde. Bei ``resume=True``
    wird sie beim naechsten Aufruf automatisch erneut versucht.
    """
    ctx = mp.get_context("spawn")
    active: Dict[int, tuple] = {}  # index -> (process, payload, start_time)
    next_idx = 0

    def spawn(payload: Dict[str, Any]):
        proc = ctx.Process(target=_run_single_process_entry, args=(payload,), daemon=True)
        proc.start()
        return proc

    while next_idx < len(payloads) and len(active) < workers:
        payload = payloads[next_idx]
        active[payload["index"]] = (spawn(payload), payload, time.time())
        next_idx += 1

    while active:
        time.sleep(1.0)
        timed_out_now: List[int] = []
        finished_now: List[int] = []
        for idx, (proc, payload, start) in active.items():
            if not proc.is_alive():
                finished_now.append(idx)
            elif (time.time() - start) > max_wall_time_s:
                timed_out_now.append(idx)

        for idx in timed_out_now:
            proc, payload, start = active[idx]
            proc.terminate()
            proc.join(timeout=10)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=10)
            print(
                f"[repeats] Wiederholung {idx} hat das Zeitlimit von "
                f"{max_wall_time_s / 3600.0:.2f} h ueberschritten und wurde abgebrochen. "
                f"Wird bei einem spaeteren Aufruf mit resume=True erneut versucht.",
                flush=True,
            )
            finished_now.append(idx)

        for idx in finished_now:
            proc, payload, start = active.pop(idx)
            if not proc.is_alive():
                proc.join(timeout=5)
            record = _load_record(payload["npz_path"]) if payload["npz_path"] else None
            if record is not None:
                records[record["index"]] = record
                done_count += 1
                if on_progress is not None:
                    on_progress(record, done_count, n_repeats)
            elif idx not in timed_out_now:
                # Prozess ist beendet, aber keine .npz vorhanden: run_single
                # ist mit einem Fehler abgebrochen (siehe traceback oben in
                # der Ausgabe). Kein RuntimeError hier -- ein einzelner
                # fehlgeschlagener Fall soll bei einem grossen Sweep nicht die
                # gesamte Kampagne stoppen; die Luecke bleibt sichtbar, weil
                # dieser Index in `records` leer bleibt.
                print(f"[repeats] Wiederholung {idx} ist ohne Ergebnisdatei beendet "
                      f"(vermutlich Fehler im Lauf, siehe Ausgabe oben).", flush=True)

            if next_idx < len(payloads):
                new_payload = payloads[next_idx]
                active[new_payload["index"]] = (spawn(new_payload), new_payload, time.time())
                next_idx += 1

    return done_count


# ---------------------------------------------------------------------------
# Kampagne
# ---------------------------------------------------------------------------
def run_repeats(
    params: Optional[Dict[str, Any]] = None,
    n_repeats: int = 10,
    base_seed: int = 42,
    workers: int = 1,
    output_dir: Optional[str] = None,
    resume: bool = True,
    on_progress: Optional[Callable[[Dict[str, Any], int, int], None]] = None,
    warn_on_foreign_results: bool = False,
    max_wall_time_s: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Fuehre ``n_repeats`` Wiederholungen aus und gib die Ergebnisse zurueck.

    Parameters
    ----------
    params
        Rezept, siehe ``builder.REFERENCE_PARAMS``.
    n_repeats
        Anzahl unabhaengiger Wiederholungen (= Anzahl Seeds).
    base_seed
        Basiswert; bestimmt reproduzierbar alle N Seeds.
    workers
        1 = seriell (gut zum Debuggen, volle Fehlermeldungen).
        >1 = so viele Prozesse gleichzeitig.
    output_dir
        Ablage der ``.npz``-Dateien. ``None`` = nichts speichern.
    resume
        Bereits vorhandene, gueltige ``.npz``-Dateien wiederverwenden.
    on_progress
        Rueckruf ``(record, fertig, gesamt)`` nach jeder Wiederholung.
    warn_on_foreign_results
        Nur fuer Aufrufer, die **eine einzige** Konfiguration in einem eigenen
        Verzeichnis rechnen. Dort heisst ein fremder Fingerabdruck im Pool
        tatsaechlich "die Parameter haben sich geaendert", und der Hinweis
        erklaert, warum neu gerechnet wird.

        Eine Studie mit vielen Zellen teilt sich dagegen absichtlich einen
        Pool -- dort hat jede Zelle ihren eigenen Fingerabdruck, und der
        Hinweis waere fuer jede neue Zelle ein Fehlalarm. Deshalb standardmaessig
        aus.
    max_wall_time_s
        Hartes Zeitlimit je Wiederholung [s]. ``None`` (Default) = kein Limit,
        wie bisher. Ist gesetzt, bekommt jede Wiederholung einen eigenen
        Prozess statt einen geteilten Pool-Platz, damit ein haengender Lauf
        hart beendet werden kann, ohne einen Worker-Slot fuer den Rest der
        Kampagne zu blockieren. Erfordert ``output_dir`` (das Ergebnis eines
        rechtzeitig fertigen Laufs wird ueber die gespeicherte ``.npz``-Datei
        erkannt, nicht ueber einen Rueckgabewert -- ein per Zeitlimit
        beendeter Prozess kann keinen regulaeren Rueckgabewert mehr liefern).
        Ein abgebrochener Lauf fehlt anschliessend einfach im Ergebnis (wie
        einer, der nie gestartet wurde) und wird bei ``resume=True`` spaeter
        automatisch nachgeholt.

    Returns
    -------
    list of dict
        Ein Eintrag je Wiederholung, sortiert nach Index. Unvollstaendige
        Laeufe fehlen -- die Liste kann also kuerzer als ``n_repeats`` sein.
    """
    params = resolve_params(params)
    workers = max(1, int(workers))
    if max_wall_time_s is not None and not output_dir:
        raise ValueError(
            "max_wall_time_s erfordert output_dir: ein per Zeitlimit hart beendeter "
            "Prozess kann keinen Rueckgabewert mehr liefern, das Ergebnis wird stattdessen "
            "ueber die gespeicherte .npz-Datei erkannt."
        )
    seeds = build_seeds(n_repeats, base_seed)

    payloads: List[Dict[str, Any]] = []
    for index, seed_seq in enumerate(seeds):
        tid = task_id(params, index, seed_seq)
        payloads.append(
            {
                "index": index,
                "seed": seed_seq,
                "params": params,
                "task_id": tid,
                "npz_path": os.path.join(output_dir, "repeats", f"{tid}.npz") if output_dir else None,
                "verbose": False,
            }
        )

    records: List[Optional[Dict[str, Any]]] = [None] * n_repeats

    # --- Wiederaufnahme ---------------------------------------------------
    pending_payloads = payloads
    if resume and output_dir:
        recovered = 0
        still_open = []
        for payload in payloads:
            existing = _load_record(payload["npz_path"]) if payload["npz_path"] else None
            if existing is not None:
                records[payload["index"]] = existing
                recovered += 1
            else:
                still_open.append(payload)
        pending_payloads = still_open
        if recovered:
            print(f"[repeats] {recovered}/{n_repeats} Wiederholungen bereits vorhanden, "
                  f"es werden noch {len(pending_payloads)} gerechnet.", flush=True)
        elif pending_payloads and warn_on_foreign_results:
            _warn_about_foreign_fingerprints(output_dir, params_fingerprint(params), n_repeats)

    if not pending_payloads:
        return [r for r in records if r is not None]

    done_count = n_repeats - len(pending_payloads)

    # Threads schon im Elternprozess begrenzen: die Worker erben die
    # Umgebungsvariablen, und sie muessen vor dem numpy-Import wirken.
    pin_threads()

    if max_wall_time_s is not None:
        # Eigener Pfad fuer BEIDE Faelle (workers==1 und workers>1): siehe
        # Docstring von _run_repeats_with_timeout fuer die Begruendung, warum
        # ProcessPoolExecutor dafuer nicht reicht.
        done_count = _run_repeats_with_timeout(
            pending_payloads, workers, float(max_wall_time_s),
            records, done_count, n_repeats, on_progress,
        )
    elif workers == 1:
        for payload in pending_payloads:
            record = run_single(payload)
            records[record["index"]] = record
            done_count += 1
            if on_progress is not None:
                on_progress(record, done_count, n_repeats)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as executor:
            queue: Dict[Any, int] = {}
            next_idx = 0

            def _fill() -> None:
                nonlocal next_idx
                while next_idx < len(pending_payloads) and len(queue) < workers:
                    payload = pending_payloads[next_idx]
                    queue[executor.submit(run_single, payload)] = payload["index"]
                    next_idx += 1

            _fill()
            while queue:
                finished, _ = wait(tuple(queue.keys()), return_when=FIRST_COMPLETED)
                for future in finished:
                    index = queue.pop(future)
                    try:
                        record = future.result()
                    except Exception as exc:
                        for other in queue:
                            other.cancel()
                        raise RuntimeError(
                            f"[repeats] Wiederholung {index} ist fehlgeschlagen: {exc}\n"
                            + traceback.format_exc()
                        ) from exc
                    records[record["index"]] = record
                    done_count += 1
                    if on_progress is not None:
                        on_progress(record, done_count, n_repeats)
                _fill()

    return [r for r in records if r is not None]


def compare_serial_parallel(
    params: Optional[Dict[str, Any]] = None,
    n_repeats: int = 2,
    base_seed: int = 42,
    workers: int = 2,
) -> Dict[str, Any]:
    """Abnahmepruefung: liefert der Parallelbetrieb dieselben Zahlen?

    Bei gleichem Seed muss ein Lauf im Worker-Prozess bitgenau dasselbe
    ergeben wie im Hauptprozess. Weicht etwas ab, ist Zustand ueber die
    Prozessgrenze gewandert, der dort nicht hingehoert.

    Zusaetzlich wird geprueft, dass verschiedene Seeds *unterschiedliche*
    Ergebnisse liefern -- sonst wirkt der Seed gar nicht.
    """
    serial = run_repeats(params, n_repeats, base_seed, workers=1, output_dir=None, resume=False)
    parallel = run_repeats(params, n_repeats, base_seed, workers=workers, output_dir=None, resume=False)

    max_abs_diff = 0.0
    worst_key = ""
    for rec_s, rec_p in zip(serial, parallel):
        for key in rec_s["series"]:
            a = np.asarray(rec_s["series"][key], dtype=float)
            b = np.asarray(rec_p["series"][key], dtype=float)
            if a.shape != b.shape:
                return {
                    "identical": False,
                    "reason": f"unterschiedliche Laenge in '{key}': {a.shape} vs {b.shape}",
                    "seeds_differ": None,
                }
            both = np.isfinite(a) & np.isfinite(b)
            if np.any(both):
                diff = float(np.max(np.abs(a[both] - b[both])))
                if diff > max_abs_diff:
                    max_abs_diff, worst_key = diff, key

    seeds_differ = None
    if len(serial) > 1:
        first = np.asarray(serial[0]["series"]["n_comp"], dtype=float)
        second = np.asarray(serial[1]["series"]["n_comp"], dtype=float)
        seeds_differ = bool(first.shape != second.shape or np.any(first != second))

    return {
        "identical": max_abs_diff == 0.0,
        "max_abs_diff": max_abs_diff,
        "worst_key": worst_key,
        "seeds_differ": seeds_differ,
        "n_serial": len(serial),
        "n_parallel": len(parallel),
    }
