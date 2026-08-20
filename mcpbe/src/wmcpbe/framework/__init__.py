"""Wiederholungs-Framework fuer die WMCPBE-Monte-Carlo-Simulation.

Dieses Paket ist bewusst *additiv*: es aendert keine bestehende Datei. Es kann
komplett entfernt werden (Ordner loeschen), ohne dass am uebrigen Repository
etwas zurueckbleibt.

Aufbau (jede Schicht kennt nur die darunter):

``builder.py``
    Beschreibt einen Lauf als reines Datenpaket (``dict``) und baut daraus
    einen fertig konfigurierten Solver. Das ist die Einheit, die an einen
    Worker-Prozess geschickt wird -- nicht der Solver selbst.
``metrics.py``
    Reduziert einen *fertig gerechneten* Solver auf kompakte Zeitreihen und
    Kennzahlen. Laeuft im Worker, damit nur wenige Kilobyte statt kompletter
    Partikel-Historien zwischen den Prozessen wandern.
``runner.py``
    Fuehrt N Wiederholungen mit unabhaengigen Seeds aus, seriell oder auf
    mehreren Prozessen.
``statistics.py``
    Mittelwert / Standardabweichung / Standardfehler ueber die Wiederholungen
    plus Konvergenzpruefung.
``run_reference_repeats.py``
    Ausfuehrbares Skript: eine Konfiguration, zehn Seeds.

Das Paket liegt in ``wmcpbe/`` und ist damit als ``wmcpbe.framework``
importierbar. Ausfuehrbare Studien, die darauf aufsetzen, liegen in
``wmcpbe/Trials/``.

Siehe ``mcpbe/docs/Parallelisierung_Wiederholungen.md``.
"""

from __future__ import annotations

import os
import sys


def bootstrap_paths() -> None:
    """Lege ``mcpbe/src`` und ``pbe-core/src`` auf den Importpfad.

    Wird auch im Worker-Prozess aufgerufen: Windows startet Worker mit
    ``spawn``, also als frischen Python-Prozess ohne den ``sys.path`` des
    Elternprozesses. Ohne diesen Aufruf findet der Worker ``wmcpbe`` nicht.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    # framework -> wmcpbe -> src -> mcpbe -> repo
    mcpbe_src = os.path.abspath(os.path.join(here, "..", ".."))
    repo_root = os.path.abspath(os.path.join(mcpbe_src, "..", ".."))
    for path in (mcpbe_src, os.path.join(repo_root, "pbe-core", "src"), repo_root):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def pin_threads() -> None:
    """Begrenze BLAS-/Numba-Threads auf 1.

    In einem Worker-Prozess ist Mehrfach-Threading schaedlich: acht Prozesse
    mit je acht Threads konkurrieren um acht Kerne und sind zusammen langsamer
    als ein einzelner serieller Lauf. Muss vor dem Import von numpy/numba
    wirken, deshalb wird es als Umgebungsvariable gesetzt.
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
