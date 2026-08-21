"""Repeat framework for the WMCPBE Monte Carlo simulation.

This package is deliberately *additive*: it changes no existing file, and
deleting the folder leaves nothing behind anywhere else in the repository.

Layers, each aware only of the one below it:

``builder.py``
    Describes a run as a plain data packet (``dict``) and builds a fully
    configured solver from it. That packet -- not the solver -- is what gets
    sent to a worker process.
``metrics.py``
    Reduces an *already solved* solver to compact time series and scalars.
    Runs inside the worker, so only a few kilobytes travel between processes
    instead of complete particle histories.
``runner.py``
    Executes N repeats with independent seeds, serially or across processes.
``statistics.py``
    Mean, standard deviation and standard error over the repeats, plus a
    convergence check.
``run_reference_repeats.py``
    Executable script: one configuration, ten seeds.

The package lives inside ``wmcpbe/`` and is therefore importable as
``wmcpbe.framework``. Executable studies built on top of it live in
``wmcpbe/Trials/``.

See ``mcpbe/docs/historical/Parallelisierung_Wiederholungen.md``.
"""

from __future__ import annotations

import os
import sys


def bootstrap_paths() -> None:
    """Put ``mcpbe/src`` and ``pbe-core/src`` on the import path.

    Called in the worker process too: Windows starts workers with ``spawn``,
    i.e. as a fresh interpreter without the parent's ``sys.path``. Without this
    call the worker cannot find ``wmcpbe``.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    # framework -> wmcpbe -> src -> mcpbe -> repo
    mcpbe_src = os.path.abspath(os.path.join(here, "..", ".."))
    repo_root = os.path.abspath(os.path.join(mcpbe_src, "..", ".."))
    for path in (mcpbe_src, os.path.join(repo_root, "pbe-core", "src"), repo_root):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def pin_threads() -> None:
    """Limit BLAS and Numba to one thread each.

    Inside a worker process, multithreading is counterproductive: eight
    processes with eight threads each contend for eight cores and finish slower
    than a single serial run. It has to take effect before numpy/numba are
    imported, hence the environment variables.
    """
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")


bootstrap_paths()
