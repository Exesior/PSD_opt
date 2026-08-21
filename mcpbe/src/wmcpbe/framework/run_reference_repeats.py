"""One configuration, ten seeds -- the data set the evaluation is built on.

Runs the reference configuration from ``Trials/test_powerlaw_rumpf_full.py``
several times with independent seeds and writes four CSV files:

``timeseries_per_seed.csv``
    Long format, one row per (seed, time point). This is the file to evaluate
    from -- any aggregate can be rebuilt from it.
``timeseries_summary.csv``
    Wide format, one row per time point, with ``_mean`` / ``_std`` / ``_sem``
    per quantity. Directly plottable (mean as a line, ``_sem`` as a band).
``scalars_per_seed.csv`` / ``scalars_summary.csv``
    One number per run (final values, balance errors, runtime) and the
    statistics over them.

Console output of this script is German by design -- it is a working tool, not
documentation.

Usage (from ``mcpbe/src``)::

    python -m wmcpbe.framework.run_reference_repeats --repeats 10 --workers 6

Smoke test (short, additionally checks serial against parallel)::

    python -m wmcpbe.framework.run_reference_repeats --smoke

All options via ``--help``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Sequence

import numpy as np

# Supports both "python -m framework.run_reference_repeats" and running the
# file directly.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from framework import bootstrap_paths, pin_threads
    from framework.builder import REFERENCE_PARAMS, resolve_params
    from framework.metrics import SERIES_KEYS
    from framework.runner import compare_serial_parallel, run_repeats
    from framework import statistics as stats_mod
else:
    from . import bootstrap_paths, pin_threads
    from .builder import REFERENCE_PARAMS, resolve_params
    from .metrics import SERIES_KEYS
    from .runner import compare_serial_parallel, run_repeats
    from . import statistics as stats_mod

bootstrap_paths()

# Results stay under Trials/Results so that runs computed before the package
# moved remain reusable.
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Trials", "Results", "repeats",
)

SCALAR_KEYS = (
    "n_comp_final",
    "n_phys_final",
    "diameter_wmean_final_m",
    "porosity_wmean_final",
    "saturation_wmean_final",
    "solid_volume_initial_m3",
    "solid_volume_final_m3",
    "solid_conservation_error",
    "liquid_expected_m3",
    "liquid_added_m3",
    "liquid_final_m3",
    "liquid_conservation_error",
    "liquid_dose_error",
    "droplets_expected",
    "droplets_actual",
    "agg_events_total",
    "break_events_total",
    "wall_time_s",
    "cpu_time_s",
)


# ---------------------------------------------------------------------------
# CSV-Ausgabe
# ---------------------------------------------------------------------------
def write_timeseries_per_seed(records: Sequence[Dict[str, Any]], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["repeat_index", "seed_label", "t_s", *SERIES_KEYS])
        for record in records:
            series = record["series"]
            n_t = len(series["t_s"])
            for it in range(n_t):
                writer.writerow(
                    [record["index"], record["seed_label"], f"{series['t_s'][it]:.6g}"]
                    + [f"{series[key][it]:.10g}" for key in SERIES_KEYS]
                )


def write_timeseries_summary(records: Sequence[Dict[str, Any]], path: str) -> None:
    time_axis = stats_mod.stack_series(records, "t_s")
    if time_axis.size == 0:
        return
    t_ref = time_axis[0]

    summaries = {key: stats_mod.summarize_series(records, key) for key in SERIES_KEYS}
    columns: List[str] = []
    for key in SERIES_KEYS:
        columns.extend([f"{key}_mean", f"{key}_std", f"{key}_sem"])

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t_s", "n_repeats", *columns])
        for it in range(len(t_ref)):
            row: List[Any] = [f"{t_ref[it]:.6g}", len(records)]
            for key in SERIES_KEYS:
                summary = summaries[key]
                if not summary or it >= len(summary["mean"]):
                    row.extend(["", "", ""])
                    continue
                row.extend(
                    [
                        f"{summary['mean'][it]:.10g}",
                        f"{summary['std'][it]:.10g}",
                        f"{summary['sem'][it]:.10g}",
                    ]
                )
            writer.writerow(row)


def write_scalars(records: Sequence[Dict[str, Any]], per_seed_path: str, summary_path: str) -> None:
    with open(per_seed_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["repeat_index", "seed_label", *SCALAR_KEYS])
        for record in records:
            writer.writerow(
                [record["index"], record["seed_label"]]
                + [f"{record['scalars'].get(key, float('nan')):.10g}" for key in SCALAR_KEYS]
            )

    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "n", "mean", "std", "sem", "ci95", "min", "max"])
        for key in SCALAR_KEYS:
            summary = stats_mod.summarize_scalar(records, key)
            writer.writerow(
                [
                    key,
                    summary["n"],
                    f"{summary['mean']:.10g}",
                    f"{summary['std']:.10g}",
                    f"{summary['sem']:.10g}",
                    f"{summary.get('ci95', float('nan')):.10g}",
                    f"{summary['min']:.10g}",
                    f"{summary['max']:.10g}",
                ]
            )


# ---------------------------------------------------------------------------
# Bericht
# ---------------------------------------------------------------------------
def build_report(records: Sequence[Dict[str, Any]], params: Dict[str, Any], elapsed: float) -> str:
    lines: List[str] = []
    add = lines.append

    add("=" * 78)
    add("WIEDERHOLUNGSLAUF -- REFERENZCONFIG")
    add("=" * 78)
    add(f"Wiederholungen (gueltig): {len(records)}")
    add(f"Simulationsdauer:         {params['t_total']:.1f} s")
    add(f"Rechenpartikel (Start):   {int(params['initial_particles'])}")
    add(f"Gesamtlaufzeit:           {elapsed:.1f} s")
    wall = stats_mod.summarize_scalar(records, "wall_time_s")
    add(f"Rechenzeit je Lauf:       {wall['mean']:.1f} s  (min {wall['min']:.1f} / max {wall['max']:.1f})")
    add("")

    add("-" * 78)
    add("PRUEFUNGEN")
    add("-" * 78)

    distinct = stats_mod.seeds_are_distinct(records, "n_comp")
    if distinct["ok"] is None:
        add(f"  Seeds wirken:            uebersprungen ({distinct['reason']})")
    else:
        mark = "OK  " if distinct["ok"] else "FEHL"
        add(f"  [{mark}] Seeds wirken:      {distinct['n_unique']}/{distinct['n_total']} Laeufe unterscheidbar")
        if not distinct["ok"]:
            add("         -> Identische Laeufe trotz verschiedener Seeds: ein Untermodul")
            add("            haelt vermutlich einen eingefrorenen Zufallsgenerator.")

    solid = stats_mod.conservation_check(records, "solid_conservation_error", tolerance=1e-9)
    mark = "OK  " if solid["ok"] else "FEHL"
    add(f"  [{mark}] Feststoffbilanz:   groesster |Fehler| = {solid['worst_abs']:.3e} "
        f"(Toleranz {solid['tolerance']:.0e})")
    add(f"         Streuung ueber Seeds: std = {solid['std']:.3e}")

    liquid = stats_mod.conservation_check(records, "liquid_conservation_error", tolerance=1e-6)
    mark = "OK  " if liquid["ok"] else "FEHL"
    add(f"  [{mark}] Fluessigkeitsbilanz: groesster |Fehler| = {liquid['worst_abs']:.3e} "
        f"(Toleranz {liquid['tolerance']:.0e})")
    add("         (zugefuehrte Fluessigkeit gegen im System gefundene)")

    dose = stats_mod.summarize_scalar(records, "liquid_dose_error")
    if dose["n"] == 0:
        add("  Dosierung:               uebersprungen (Lauf endet vor dem Nukleationsfenster)")
    else:
        worst_dose = max(abs(dose["min"]), abs(dose["max"]))
        mark = "OK  " if worst_dose <= 1e-3 else "WARN"
        add(f"  [{mark}] Dosierung:         groesster |Fehler| = {worst_dose:.3e} "
            f"(geplante gegen zugefuehrte Menge)")

    conv = stats_mod.convergence_check(records, "n_phys")
    if conv["ok"] is None:
        add(f"  Konvergenz:              uebersprungen ({conv['reason']})")
    else:
        mark = "OK  " if conv["ok"] else "WARN"
        add(f"  [{mark}] Konvergenz n_phys: Steigung log(sem) ueber log(N) = {conv['slope']:+.2f} "
            f"(erwartet {conv['expected_slope']:+.2f})")
        if not conv["ok"]:
            add("         -> Der Standardfehler faellt nicht wie 1/sqrt(N). Entweder sind")
            add("            die Laeufe korreliert, oder N ist noch zu klein fuer die Aussage.")
    add("")

    add("-" * 78)
    add("ERGEBNISSE (Mittelwert +/- Standardfehler ueber die Wiederholungen)")
    add("-" * 78)
    display = [
        ("Rechenpartikel am Ende", "n_comp_final", "{:.1f}"),
        ("n_phys am Ende [1/m^3]", "n_phys_final", "{:.4e}"),
        ("mittl. Durchmesser [m]", "diameter_wmean_final_m", "{:.4e}"),
        ("mittl. Porositaet", "porosity_wmean_final", "{:.6f}"),
        ("mittl. Saettigung", "saturation_wmean_final", "{:.6f}"),
        ("Agglomerationen", "agg_events_total", "{:.0f}"),
        ("Brueche", "break_events_total", "{:.0f}"),
        ("Tropfen zugefuehrt", "droplets_actual", "{:.2f}"),
    ]
    for label, key, fmt in display:
        summary = stats_mod.summarize_scalar(records, key)
        if summary["n"] == 0:
            add(f"  {label:<26s} keine Daten")
            continue
        rel = abs(summary["sem"] / summary["mean"]) * 100.0 if summary["mean"] else float("nan")
        add(
            f"  {label:<26s} {fmt.format(summary['mean']):>14s} "
            f"+/- {fmt.format(summary['sem']):<14s} ({rel:5.2f} %)"
        )
    add("")
    add("  Die Prozentangabe ist der Standardfehler relativ zum Mittelwert:")
    add("  wie genau der Mittelwert mit dieser Zahl an Wiederholungen bestimmt ist.")
    add("=" * 78)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=10, help="Anzahl Seeds (Default: 10)")
    parser.add_argument("--workers", type=int, default=6, help="Parallele Prozesse (Default: 6)")
    parser.add_argument("--base-seed", type=int, default=205, help="Basiswert fuer die Seeds (Default: 205)")
    parser.add_argument("--t-total", type=float, default=None, help="Simulationsdauer [s] (Default: 100)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Ausgabeverzeichnis")
    parser.add_argument("--no-resume", action="store_true", help="Vorhandene Ergebnisse ignorieren")
    parser.add_argument("--check", action="store_true", help="Zusaetzlich seriell gegen parallel pruefen")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Kurzer Rauchtest: t_total=4 s, 4 Wiederholungen, mit Aequivalenzpruefung",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    pin_threads()

    params: Dict[str, Any] = {}
    if args.smoke:
        params["t_total"] = 4.0
        args.repeats = min(args.repeats, 4)
        args.check = True
        args.output = os.path.join(args.output, "smoke")
    if args.t_total is not None:
        params["t_total"] = float(args.t_total)
    params = resolve_params(params)

    os.makedirs(args.output, exist_ok=True)

    if args.check:
        print("Aequivalenzpruefung seriell gegen parallel ...", flush=True)
        check = compare_serial_parallel(params, n_repeats=2, base_seed=args.base_seed, workers=2)
        if check["identical"]:
            print("  [OK  ] Parallel liefert bitgenau dieselben Zahlen wie seriell.")
        else:
            print(f"  [FEHL] Abweichung {check.get('max_abs_diff')} in '{check.get('worst_key')}' "
                  f"({check.get('reason', '')})")
            print("         Es ist Zustand ueber die Prozessgrenze gewandert. Abbruch.")
            return 1
        if check["seeds_differ"] is False:
            print("  [FEHL] Zwei verschiedene Seeds liefern identische Laeufe. Abbruch.")
            return 1
        print("  [OK  ] Verschiedene Seeds liefern verschiedene Laeufe.")
        print("")

    print(f"Starte {args.repeats} Wiederholungen auf {args.workers} Prozess(en), "
          f"t_total={params['t_total']:.1f} s ...", flush=True)

    def progress(record: Dict[str, Any], done: int, total: int) -> None:
        print(
            f"  [{done:2d}/{total}] Seed {record['seed_label']:>4s}  "
            f"{record['scalars']['wall_time_s']:6.1f} s  "
            f"n_comp={record['scalars']['n_comp_final']:.0f}  "
            f"Feststofffehler={record['scalars']['solid_conservation_error']:+.2e}",
            flush=True,
        )

    t_start = time.time()
    records = run_repeats(
        params=params,
        n_repeats=args.repeats,
        base_seed=args.base_seed,
        workers=args.workers,
        output_dir=args.output,
        resume=not args.no_resume,
        on_progress=progress,
        # Eine einzige Konfiguration in einem eigenen Verzeichnis: hier ist ein
        # fremder Fingerabdruck im Pool ein echter Hinweis.
        warn_on_foreign_results=True,
    )
    elapsed = time.time() - t_start

    if not records:
        print("Keine gueltigen Ergebnisse.")
        return 1

    records = sorted(records, key=lambda r: r["index"])

    write_timeseries_per_seed(records, os.path.join(args.output, "timeseries_per_seed.csv"))
    write_timeseries_summary(records, os.path.join(args.output, "timeseries_summary.csv"))
    write_scalars(
        records,
        os.path.join(args.output, "scalars_per_seed.csv"),
        os.path.join(args.output, "scalars_summary.csv"),
    )
    with open(os.path.join(args.output, "params.json"), "w", encoding="utf-8") as handle:
        json.dump({"params": params, "base_seed": args.base_seed,
                   "n_repeats": args.repeats}, handle, indent=2, default=str)

    report = build_report(records, params, elapsed)
    print("")
    print(report)
    with open(os.path.join(args.output, "report.txt"), "w", encoding="utf-8") as handle:
        handle.write(report + "\n")

    print(f"\nDateien in: {args.output}")
    return 0


if __name__ == "__main__":
    # This guard is mandatory on Windows: worker processes re-import this
    # module. Without it, every repeat would start the whole campaign again.
    raise SystemExit(main())
