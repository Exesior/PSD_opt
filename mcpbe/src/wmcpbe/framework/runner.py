"""N repeats of the same setup with independent seeds.

Four building blocks, taken from ``upstream/dev_monorepo``:

1. **Independent seeds.** Not ``seed = base + i`` (neighbouring seeds can
   produce correlated streams) but ``SeedSequence.spawn(N)``. Only then are the
   repeats statistically independent -- and only then does a standard error
   mean what it claims to mean.
2. **Reduction in the worker.** The worker returns figures, not the solver
   (see ``metrics.py``).
3. **Backpressure.** Never more than ``workers`` jobs queued at once. With many
   repeats, memory stays constant instead of growing with N.
4. **Resumability.** Every finished repeat immediately writes its own ``.npz``
   file, atomically (write ``.tmp``, then rename), so an abort can never leave
   half a file behind. On restart, finished runs are read back and only the
   remainder is computed.

Difference to the supervisor's template: there the solver state is shipped to
the worker with ``copy.deepcopy(self.__dict__)``. Here only the recipe travels
and the worker builds the solver itself -- reasoning in ``builder.py``.
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
    """Create N guaranteed-independent seed sequences from one base value.

    The same ``base_seed`` always yields the same N sequences, so the whole
    campaign is reproducible even though each individual run is random.
    """
    return list(np.random.SeedSequence(int(base_seed)).spawn(int(n_repeats)))


def seed_label(seed_seq: np.random.SeedSequence) -> str:
    """Short, stable name for a seed sequence, for filenames and CSV."""
    return "-".join(str(k) for k in seed_seq.spawn_key) or "root"


# ---------------------------------------------------------------------------
# Identity of a run (for resuming)
# ---------------------------------------------------------------------------
def params_fingerprint(params: Dict[str, Any]) -> str:
    """Short hash over the parameter values.

    Change a parameter and the fingerprint changes -- old results are then not
    reused by mistake.
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
    """Runs once per worker process, before the first job arrives."""
    pin_threads()
    bootstrap_paths()


def run_single(payload: Dict[str, Any]) -> Dict[str, Any]:
    """One complete run. Executes in a worker process (or serially in the main one).

    Must stay at module level, not become a method: Windows starts workers with
    ``spawn``, which requires the function to be findable by its module path.
    """
    _worker_init()

    from .builder import build_reference_solver
    from .metrics import reduce_solver

    params = payload["params"]
    seed_seq = payload["seed"]
    index = int(payload["index"])

    # Two timings, because they say different things:
    #   wall_time_s -- elapsed clock time. With several workers on too few
    #                  cores it includes waiting, and is useless for comparing
    #                  configurations.
    #   cpu_time_s  -- CPU time this process actually consumed. Stays
    #                  meaningful under contention; this is the one that
    #                  answers "which setting costs more".
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
# Persistence
# ---------------------------------------------------------------------------
def _save_record(record: Dict[str, Any], npz_path: str) -> None:
    """Write one repeat to disk atomically.

    First into a ``.tmp`` file, then rename: ``os.replace`` is atomic, so the
    target file either exists complete or not at all. An abort mid-write can
    therefore not leave a half file that the next start would read as
    "finished".
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

    # The ".npz" suffix has to stay: otherwise np.savez_compressed appends it
    # itself and writes to a different file than intended.
    tmp_path = npz_path[: -len(".npz")] + ".tmp.npz"
    np.savez_compressed(tmp_path, **payload)
    os.replace(tmp_path, npz_path)


def _load_record(npz_path: str) -> Optional[Dict[str, Any]]:
    """Read one stored repeat back; ``None`` if it is unreadable."""
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
    """Report results in the pool that carry a different fingerprint.

    Without this note, a complete recomputation looks like a failure. It is in
    fact the correct response: the fingerprint covers ALL parameters, so merely
    adding a new key to ``REFERENCE_PARAMS`` formally invalidates the old
    results -- even when the new value matches the previous behaviour. Better
    to compute once too often than to silently mix two configurations.
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
    listed = ", ".join(f"{key} ({count} files)" for key, count in sorted(others.items()))
    print(
        f"[repeats] NOTE: the pool holds results with a different parameter fingerprint "
        f"({listed}); the current one is '{current}'. Everything is therefore recomputed "
        f"({n_repeats} runs). The cause is a changed or newly added parameter -- "
        f"not a broken cache.",
        flush=True,
    )


def _run_single_process_entry(payload: Dict[str, Any]) -> None:
    """Process entry point for the timeout path.

    Deliberately returns nothing over IPC: ``run_single`` already stores its
    result atomically as ``.npz`` (see ``_save_record``). The parent recognises
    success by that file existing after the process ends -- regardless of
    whether the process finished cleanly or was killed by the time limit.
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
    """Like the normal parallel path, but with a hard time limit per repeat.

    ``ProcessPoolExecutor`` is not suitable here: ``future.result(timeout=...)``
    does return control to the caller, but the hung computation keeps running
    in the pool's worker process indefinitely and blocks that slot for the rest
    of the campaign -- with 250 planned runs, a single hung case would
    permanently cost one of the few parallel slots. Each repeat therefore gets
    its own ``multiprocessing.Process`` here, which is killed on overrun
    (``terminate()``, then ``kill()`` after a short grace period) and frees the
    slot afterwards.

    A repeat aborted by the time limit leaves no ``.npz`` behind (the final
    write is atomic, see ``_save_record`` -- all or nothing). It therefore
    simply does not appear in the result, exactly like a repeat that was never
    started. With ``resume=True`` it is retried automatically on the next call.
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
                f"[repeats] repeat {idx} exceeded the time limit of "
                f"{max_wall_time_s / 3600.0:.2f} h and was aborted. "
                f"It will be retried on a later call with resume=True.",
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
                # Process ended but no .npz exists: run_single aborted with an
                # error (its traceback is above in the output). No RuntimeError
                # here -- a single failed case should not stop a large sweep,
                # and the gap stays visible because this index remains empty in
                # `records`.
                print(f"[repeats] repeat {idx} ended without a result file "
                      f"(probably an error in the run, see output above).", flush=True)

            if next_idx < len(payloads):
                new_payload = payloads[next_idx]
                active[new_payload["index"]] = (spawn(new_payload), new_payload, time.time())
                next_idx += 1

    return done_count


# ---------------------------------------------------------------------------
# Campaign
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
    """Run ``n_repeats`` repeats and return the results.

    Parameters
    ----------
    params
        Recipe, see ``builder.REFERENCE_PARAMS``.
    n_repeats
        Number of independent repeats (= number of seeds).
    base_seed
        Base value; reproducibly determines all N seeds.
    workers
        1 = serial (good for debugging, full error messages).
        >1 = that many processes at once.
    output_dir
        Where the ``.npz`` files go. ``None`` = store nothing.
    resume
        Reuse existing, valid ``.npz`` files.
    on_progress
        Callback ``(record, done, total)`` after each repeat.
    warn_on_foreign_results
        Only for callers that compute **a single** configuration in their own
        directory. There, a foreign fingerprint in the pool really does mean
        "the parameters changed", and the note explains why everything is being
        recomputed.

        A study with many cells deliberately shares one pool -- there each cell
        has its own fingerprint, and the note would be a false alarm for every
        new cell. Hence off by default.
    max_wall_time_s
        Hard time limit per repeat [s]. ``None`` (default) = no limit. When
        set, each repeat gets its own process instead of a shared pool slot, so
        a hung run can be killed without blocking a worker slot for the rest of
        the campaign. Requires ``output_dir``: the result of a run that
        finished in time is recognised through its stored ``.npz`` file, not
        through a return value -- a process killed by the time limit can no
        longer return one. An aborted run simply is missing from the result
        (like one that never started) and is picked up later with
        ``resume=True``.

    Returns
    -------
    list of dict
        One entry per repeat, ordered by index. Incomplete runs are missing, so
        the list may be shorter than ``n_repeats``.
    """
    params = resolve_params(params)
    workers = max(1, int(workers))
    if max_wall_time_s is not None and not output_dir:
        raise ValueError(
            "max_wall_time_s requires output_dir: a process killed by the time limit "
            "can no longer return a value, so the result is recognised through the "
            "stored .npz file instead."
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

    # --- resume -----------------------------------------------------------
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
            print(f"[repeats] {recovered}/{n_repeats} repeats already present, "
                  f"{len(pending_payloads)} still to compute.", flush=True)
        elif pending_payloads and warn_on_foreign_results:
            _warn_about_foreign_fingerprints(output_dir, params_fingerprint(params), n_repeats)

    if not pending_payloads:
        return [r for r in records if r is not None]

    done_count = n_repeats - len(pending_payloads)

    # Pin threads in the parent already: workers inherit the environment
    # variables, and they must take effect before numpy is imported.
    pin_threads()

    if max_wall_time_s is not None:
        # Separate path for BOTH cases (workers==1 and workers>1): see the
        # docstring of _run_repeats_with_timeout for why ProcessPoolExecutor
        # is not enough.
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
                            f"[repeats] repeat {index} failed: {exc}\n"
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
    """Acceptance check: does parallel execution produce the same numbers?

    With the same seed, a run in a worker process must give bit-for-bit the
    same result as one in the main process. Any deviation means state crossed
    the process boundary that should not have.

    Also checks that different seeds give *different* results -- otherwise the
    seed has no effect at all.
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
                    "reason": f"different length in '{key}': {a.shape} vs {b.shape}",
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
