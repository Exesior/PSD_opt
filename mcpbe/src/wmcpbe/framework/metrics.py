"""Reduktion eines fertig gerechneten Solvers auf kompakte Kennzahlen.

Diese Funktionen laufen **im Worker-Prozess**, direkt nach ``solve()``. Das ist
Absicht: ein fertiger Solver traegt die komplette Partikel-Historie mit sich
(bei 2000 Partikeln und 501 Zeitpunkten schnell dreistellige Megabyte). Alles,
was zwischen zwei Prozessen wandert, muss serialisiert werden -- wuerde man den
Solver zurueckschicken, frisst die Serialisierung genau den Zeitgewinn wieder
auf, den die Parallelisierung gebracht hat.

Zurueck geht deshalb nur, was fuer die Auswertung gebraucht wird: ein paar
Zeitreihen der Laenge T und eine Handvoll Skalare -- wenige Kilobyte.

Hinweis zu Mittelwerten
-----------------------
Porositaet, Saettigung und Durchmesser werden in zwei Varianten berechnet:

``*_mean``
    ungewichtetes Mittel ueber die Rechenpartikel. Entspricht dem, was
    ``Trials/test_sensitivity.py`` heute schreibt -- fuer die Vergleichbarkeit
    mit vorhandenen CSVs beibehalten.
``*_wmean``
    mit dem DSMC-Gewicht ``W`` gewichtetes Mittel. Das ist die physikalisch
    richtige Groesse: ein Rechenpartikel mit W=600 steht fuer 600 reale
    Partikel und darf nicht so zaehlen wie eines mit W=1.

Fuer physikalische Aussagen die ``*_wmean``-Spalten verwenden.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


# Zeitreihen, die pro Wiederholung zurueckgegeben werden. Reihenfolge = Reihen-
# folge in den CSV-Dateien.
SERIES_KEYS = (
    "n_comp",              # Anzahl Rechenpartikel [-]
    "n_phys",              # physikalische Anzahldichte [1/m^3]
    "M0",                  # nulltes Moment (Summe der Gewichte)
    "M1_m3",               # erstes Moment (Gesamt-Trockenvolumen) [m^3]
    "M2_m6",               # zweites Moment [m^6]
    "porosity_mean",       # ungewichtet
    "porosity_wmean",      # W-gewichtet
    "saturation_mean",
    "saturation_wmean",
    "diameter_mean_m",     # ungewichtet, aus V_dry
    "diameter_wmean_m",    # W-gewichtet
    "liquid_in_system_m3", # Fluessigkeit in Partikeln [m^3]
    "solid_volume_m3",     # Feststoffvolumen [m^3] -- muss konstant bleiben
    "Vc_m3",               # Kontrollvolumen [m^3]
    "agg_events",          # kumulierte, dW-gewichtete Agglomerationsereignisse
    "break_events",        # kumulierte, dW-gewichtete Bruchereignisse
)


def _safe_mean(values: np.ndarray) -> float:
    """Mittelwert ueber die endlichen Eintraege; 0.0 wenn keiner endlich ist."""
    finite = np.isfinite(values)
    if not np.any(finite):
        return 0.0
    return float(np.mean(values[finite]))


def _safe_wmean(values: np.ndarray, weights: np.ndarray) -> float:
    """Gewichteter Mittelwert ueber die endlichen Eintraege."""
    finite = np.isfinite(values) & np.isfinite(weights)
    if not np.any(finite):
        return 0.0
    w = weights[finite]
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        return 0.0
    return float(np.sum(values[finite] * w) / w_sum)


def _volume_to_diameter(volume: np.ndarray) -> np.ndarray:
    """Kugelaequivalenter Durchmesser aus dem Volumen."""
    v = np.clip(np.asarray(volume, dtype=float), 0.0, None)
    return np.cbrt(6.0 * v / np.pi)


def extract_series(solver: Any) -> Dict[str, np.ndarray]:
    """Baue die Zeitreihen aus den ``*_save``-Listen des Solvers.

    Die Laenge richtet sich nach der kuerzesten vorhandenen Liste. Bricht ein
    Lauf vorzeitig ab, sind die Reihen entsprechend kuerzer -- das faellt in
    der Statistik auf, statt still mit Nullen aufgefuellt zu werden.
    """
    n_t = min(
        len(solver.V_save),
        len(solver.W_save),
        len(solver.Vc_save),
        len(solver.porosity_save),
        len(solver.saturation_save),
        len(solver.liquid_volume_save),
        len(solver.t_vec),
    )

    out: Dict[str, np.ndarray] = {key: np.zeros(n_t, dtype=float) for key in SERIES_KEYS}
    out["t_s"] = np.asarray(solver.t_vec[:n_t], dtype=float)

    agg_save = getattr(solver, "real_agg_events_save", None)
    break_save = getattr(solver, "real_break_events_save", None)

    for it in range(n_t):
        v_dry = np.asarray(solver.V_save[it][-1, :], dtype=float)
        weights = np.asarray(solver.W_save[it], dtype=float)
        poro = np.asarray(solver.porosity_save[it], dtype=float)
        sat = np.asarray(solver.saturation_save[it], dtype=float)
        liq = np.asarray(solver.liquid_volume_save[it], dtype=float)
        Vc = float(solver.Vc_save[it])

        # Feststoffvolumen: Partikel ohne definierte Porositaet gelten als
        # Vollkoerper (gleiche Konvention wie test_powerlaw_rumpf_full.py).
        valid_poro = np.isfinite(poro)
        v_solid = np.where(valid_poro, v_dry * (1.0 - np.where(valid_poro, poro, 0.0)), v_dry)

        diameters = _volume_to_diameter(v_dry)

        out["n_comp"][it] = float(len(weights))
        out["M0"][it] = float(np.sum(weights))
        out["M1_m3"][it] = float(np.sum(weights * v_dry))
        out["M2_m6"][it] = float(np.sum(weights * v_dry ** 2))
        out["n_phys"][it] = out["M0"][it] / Vc if Vc > 0.0 else np.nan
        out["porosity_mean"][it] = _safe_mean(poro)
        out["porosity_wmean"][it] = _safe_wmean(poro, weights)
        out["saturation_mean"][it] = _safe_mean(sat)
        out["saturation_wmean"][it] = _safe_wmean(sat, weights)
        out["diameter_mean_m"][it] = _safe_mean(diameters)
        out["diameter_wmean_m"][it] = _safe_wmean(diameters, weights)
        out["liquid_in_system_m3"][it] = float(np.sum(liq * weights))
        out["solid_volume_m3"][it] = float(np.sum(v_solid * weights))
        out["Vc_m3"][it] = Vc
        out["agg_events"][it] = float(agg_save[it]) if agg_save is not None and it < len(agg_save) else np.nan
        out["break_events"][it] = float(break_save[it]) if break_save is not None and it < len(break_save) else np.nan

    return out


def extract_scalars(solver: Any, series: Dict[str, np.ndarray], params: Dict[str, Any]) -> Dict[str, float]:
    """Kennzahlen fuer den Gesamtlauf (eine Zahl je Wiederholung).

    Enthaelt insbesondere die beiden Erhaltungsgroessen. Sie sind das
    Abnahmekriterium: streuen sie ueber die Seeds, stimmt etwas nicht --
    Masse darf nicht vom Zufall abhaengen.
    """
    from .builder import expected_droplet_count, expected_liquid_volume

    # WICHTIG: die Bilanzen werden aus dem LIVE-Endzustand des Solvers
    # gerechnet, nicht aus dem letzten Snapshot. Die Schleife laeuft nach dem
    # letzten Speicherzeitpunkt noch bis zum naechsten Ereignis weiter -- der
    # Snapshot bei t_vec[-1] liegt also etwas frueher als das Ende. Mischt man
    # beide Zeitpunkte, sieht eine perfekte Bilanz wie ein Verlust von einigen
    # Prozent aus. (Genau so gemessen: Snapshot 1.1807e-09 gegen live
    # 1.2310e-09 -- die Differenz war reine Zeitverschiebung, kein Verlust.)
    a_tot = int(solver.a_tot)
    w_live = np.asarray(solver.W[:a_tot], dtype=float)
    v_dry_live = np.asarray(solver.V_flat[-1, :a_tot], dtype=float)
    poro_live = np.asarray(solver.porosity[:a_tot], dtype=float)
    liq_live = np.asarray(solver.liquid_volume[:a_tot], dtype=float)

    valid_poro = np.isfinite(poro_live)
    v_solid_live = np.where(
        valid_poro, v_dry_live * (1.0 - np.where(valid_poro, poro_live, 0.0)), v_dry_live
    )

    solid = series["solid_volume_m3"]
    solid0 = float(solid[0]) if len(solid) else np.nan
    solid_end = float(np.sum(v_solid_live * w_live))
    solid_error = (solid_end - solid0) / solid0 if solid0 not in (0.0, np.nan) else np.nan

    liquid_expected = expected_liquid_volume(params)
    liquid_end = float(np.sum(liq_live * w_live))

    droplets_expected = expected_droplet_count(params)
    droplets_actual = np.nan
    liquid_added = np.nan
    nucleation = getattr(solver, "nucleation", None)
    if nucleation is not None and hasattr(nucleation, "get_statistics"):
        try:
            stats = nucleation.get_statistics()
            droplets_actual = float(stats.get("droplets_added_total", np.nan))
            liquid_added = float(stats.get("liquid_volume_added_total", np.nan))
        except Exception:
            droplets_actual = np.nan
            liquid_added = np.nan

    # Bilanz: bleibt die tatsaechlich zugefuehrte Fluessigkeit im System?
    # Bewusst gegen die ZUGEFUEHRTE Menge geprueft, nicht gegen die fuer das
    # volle Zeitfenster erwartete. Sonst schlaegt jeder Lauf fehl, der vor dem
    # Ende der Nukleationsdauer endet -- obwohl nichts verloren gegangen ist.
    if np.isfinite(liquid_added) and liquid_added > 0.0:
        liquid_error = (liquid_end - liquid_added) / liquid_added
    else:
        liquid_error = np.nan

    # Dosierung: wurde ueberhaupt die geplante Menge zugefuehrt? Nur sinnvoll,
    # wenn die Simulation das Nukleationsfenster vollstaendig abdeckt.
    covers_window = float(params.get("t_total", 0.0)) >= float(params.get("nucleation_duration", 0.0))
    if covers_window and liquid_expected > 0.0 and np.isfinite(liquid_added):
        liquid_dose_error = (liquid_added - liquid_expected) / liquid_expected
    else:
        liquid_dose_error = np.nan

    return {
        "n_comp_final": float(series["n_comp"][-1]) if len(series["n_comp"]) else np.nan,
        "n_phys_final": float(series["n_phys"][-1]) if len(series["n_phys"]) else np.nan,
        "diameter_wmean_final_m": float(series["diameter_wmean_m"][-1]) if len(series["diameter_wmean_m"]) else np.nan,
        "porosity_wmean_final": float(series["porosity_wmean"][-1]) if len(series["porosity_wmean"]) else np.nan,
        "saturation_wmean_final": float(series["saturation_wmean"][-1]) if len(series["saturation_wmean"]) else np.nan,
        "solid_volume_initial_m3": solid0,
        "solid_volume_final_m3": solid_end,
        "solid_conservation_error": float(solid_error),
        "liquid_expected_m3": float(liquid_expected),
        "liquid_added_m3": float(liquid_added),
        "liquid_final_m3": liquid_end,
        "liquid_conservation_error": float(liquid_error),
        "liquid_dose_error": float(liquid_dose_error),
        "droplets_expected": float(droplets_expected),
        "droplets_actual": float(droplets_actual),
        "agg_events_total": float(getattr(solver, "real_agg_events", np.nan)),
        "break_events_total": float(getattr(solver, "real_break_events", np.nan)),
    }


def reduce_solver(solver: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Kompletter Reduktionsschritt: Solver rein, kleines Datenpaket raus."""
    series = extract_series(solver)
    scalars = extract_scalars(solver, series, params)
    return {"series": series, "scalars": scalars}
