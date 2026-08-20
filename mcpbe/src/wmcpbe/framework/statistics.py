"""Statistik ueber die Wiederholungen.

Ein einzelner Monte-Carlo-Lauf ist eine Stichprobe, kein Ergebnis. Erst der
Mittelwert ueber mehrere unabhaengige Laeufe -- zusammen mit einer Angabe,
wie sicher dieser Mittelwert ist -- laesst eine Aussage zu.

Die drei Groessen und was sie unterscheiden:

``std`` (Standardabweichung)
    Wie stark streuen die *einzelnen Laeufe*? Bleibt bei mehr Wiederholungen
    ungefaehr gleich gross -- sie ist eine Eigenschaft des Modells.
``sem`` (Standardfehler, ``std/sqrt(N)``)
    Wie unsicher ist der *Mittelwert*? Schrumpft mit mehr Wiederholungen. Das
    ist die Groesse, die als Fehlerbalken gehoert.
``ci95``
    Naeherung ``1.96 * sem``. Bei kleinem N (unter etwa 30) ist das optimistisch;
    fuer zehn Wiederholungen ist der Faktor nach Student eher 2.26.

Konvergenzpruefung
------------------
Sind die Laeufe wirklich unabhaengig, muss der Standardfehler mit ``1/sqrt(N)``
fallen -- im doppelt logarithmischen Bild also eine Gerade mit Steigung -0.5.
Kommt deutlich weniger heraus, sind die Laeufe korreliert (typischer Fall: ein
geteilter Zufallsgenerator), und alle Fehlerbalken waeren zu klein.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def stack_series(records: Sequence[Dict[str, Any]], key: str) -> np.ndarray:
    """Lege die Zeitreihe ``key`` aller Wiederholungen als (N, T)-Matrix ab.

    Unterschiedlich lange Reihen werden auf die kuerzeste gekuerzt. Das
    passiert, wenn ein Lauf vorzeitig endet -- besser sichtbar kuerzen als
    still mit Nullen auffuellen.
    """
    series = [np.asarray(r["series"][key], dtype=float) for r in records if key in r["series"]]
    if not series:
        return np.zeros((0, 0), dtype=float)
    n_t = min(len(s) for s in series)
    return np.stack([s[:n_t] for s in series], axis=0)


def summarize_series(records: Sequence[Dict[str, Any]], key: str) -> Dict[str, np.ndarray]:
    """Mittelwert, Streuung und Unsicherheit einer Zeitreihe ueber alle Laeufe."""
    stacked = stack_series(records, key)
    if stacked.size == 0:
        return {}

    n = stacked.shape[0]
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0, ddof=1) if n > 1 else np.zeros(stacked.shape[1])
        p05 = np.nanpercentile(stacked, 5, axis=0)
        p95 = np.nanpercentile(stacked, 95, axis=0)
    sem = std / np.sqrt(n) if n > 1 else np.zeros_like(std)

    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95": 1.96 * sem,
        "min": np.nanmin(stacked, axis=0),
        "max": np.nanmax(stacked, axis=0),
        "p05": p05,
        "p95": p95,
        "n": np.full(stacked.shape[1], n, dtype=float),
    }


def summarize_scalar(records: Sequence[Dict[str, Any]], key: str) -> Dict[str, float]:
    """Mittelwert, Streuung und Unsicherheit einer Kennzahl ueber alle Laeufe."""
    values = np.asarray(
        [float(r["scalars"][key]) for r in records if key in r["scalars"]], dtype=float
    )
    finite = values[np.isfinite(values)]
    n = int(finite.size)
    if n == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "sem": np.nan,
                "min": np.nan, "max": np.nan}
    std = float(np.std(finite, ddof=1)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": float(np.mean(finite)),
        "std": std,
        "sem": std / np.sqrt(n) if n > 1 else 0.0,
        "ci95": 1.96 * std / np.sqrt(n) if n > 1 else 0.0,
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def convergence_check(records: Sequence[Dict[str, Any]], key: str, at_index: int = -1) -> Dict[str, Any]:
    """Faellt der Standardfehler wie ``1/sqrt(N)``?

    Vorgehen: fuer N = 2, 3, ... wird der Standardfehler aus den ersten N
    Laeufen berechnet und gegen N doppelt logarithmisch aufgetragen. Die
    Steigung sollte nahe -0.5 liegen.

    Grenze der Aussagekraft: bei zehn Wiederholungen ist die Steigung selbst
    verrauscht. Alles zwischen etwa -0.8 und -0.2 ist unauffaellig; ein Wert
    nahe 0 dagegen bedeutet, dass zusaetzliche Laeufe den Mittelwert nicht
    sicherer machen -- dann sind sie nicht unabhaengig.
    """
    stacked = stack_series(records, key)
    if stacked.size == 0 or stacked.shape[0] < 3:
        return {"ok": None, "reason": "zu wenige Wiederholungen fuer eine Aussage",
                "slope": np.nan, "n_values": [], "sem_values": []}

    column = stacked[:, at_index]
    column = column[np.isfinite(column)]
    if column.size < 3:
        return {"ok": None, "reason": "zu wenige endliche Werte",
                "slope": np.nan, "n_values": [], "sem_values": []}

    n_values: List[int] = []
    sem_values: List[float] = []
    for n in range(2, column.size + 1):
        subset = column[:n]
        sem = float(np.std(subset, ddof=1) / np.sqrt(n))
        if sem > 0.0:
            n_values.append(n)
            sem_values.append(sem)

    if len(n_values) < 3:
        # Teilmengen mit Streuung null werden uebersprungen. Bei wenigen
        # Wiederholungen und einer quantisierten Groesse (n_phys aendert sich
        # nur in Vielfachen der Paketgroesse dW) bleiben dann zu wenige Punkte
        # fuer eine Steigung uebrig. Das ist kein Befund, sondern zu wenig Daten.
        return {
            "ok": None,
            "reason": (
                f"nur {len(n_values)} auswertbare Teilmengen -- fuer eine Steigung "
                "werden mindestens 3 gebraucht (mehr Wiederholungen noetig)"
            ),
            "slope": np.nan,
            "n_values": n_values,
            "sem_values": sem_values,
        }

    slope = float(np.polyfit(np.log(n_values), np.log(sem_values), 1)[0])
    return {
        "ok": bool(-0.85 <= slope <= -0.15),
        "slope": slope,
        "expected_slope": -0.5,
        "n_values": n_values,
        "sem_values": sem_values,
        "reason": "",
    }


def seeds_are_distinct(records: Sequence[Dict[str, Any]], key: str = "n_comp") -> Dict[str, Any]:
    """Liefern die Seeds tatsaechlich verschiedene Laeufe?

    Wenn alle Wiederholungen identisch sind, wirkt der Seed nicht -- typisch
    dafuer, dass ein Untermodul einen eingefrorenen Zufallsgenerator haelt.
    Dann waere jede Statistik wertlos, obwohl alles fehlerfrei durchlaeuft.
    """
    stacked = stack_series(records, key)
    if stacked.shape[0] < 2:
        return {"ok": None, "reason": "nur eine Wiederholung", "n_unique": stacked.shape[0]}
    unique_rows = np.unique(stacked, axis=0)
    n_unique = int(unique_rows.shape[0])
    return {
        "ok": n_unique == stacked.shape[0],
        "n_unique": n_unique,
        "n_total": int(stacked.shape[0]),
        "reason": "" if n_unique == stacked.shape[0] else "mindestens zwei Laeufe sind identisch",
    }


def conservation_check(
    records: Sequence[Dict[str, Any]],
    key: str = "solid_conservation_error",
    tolerance: float = 1e-9,
) -> Dict[str, Any]:
    """Bleibt eine Erhaltungsgroesse ueber alle Seeds erhalten?

    Masse darf nicht vom Zufall abhaengen. Streut dieser Fehler ueber die
    Seeds, ist das ein starkes Signal fuer einen zustandsabhaengigen Fehler --
    genau die Sorte, die in einem Einzellauf wie Rauschen aussieht.
    """
    stats = summarize_scalar(records, key)
    worst = max(abs(stats["min"]), abs(stats["max"])) if stats["n"] > 0 else np.nan
    return {
        "ok": bool(np.isfinite(worst) and worst <= tolerance),
        "worst_abs": float(worst),
        "tolerance": float(tolerance),
        **stats,
    }
