"""
Spyder: Datei in den Ordner studies/sensitivity legen, oeffnen, F5.
PRUEFUNG SPLITTING-FEHLER -- 12 Punkte aus gesamt_v1 noch einmal mit N_COMP 500 statt 125.

Warum: Grafik Splitting_break_deviation_vs_R (Bruchraten-Abweichung gegen Event-Rate R).
N_COMP 500 hebt R um etwa x4. Aendern sich die Ergebnisse dabei nur im Rahmen des
Seed-Rauschens, ist v1 auch dort belastbar, wo R/k_comp am kleinsten war.

  schlecht   10 Punkte mit den meisten Bruch-Events bei R/k < 5 (je Saltelli-Block einer)
  Kontrolle   2 Punkte mit viel Bruch, aber R/k >= 10 in jedem 5-s-Fenster

Gleiche 5 Seeds wie v1. Gewichtsbezogene Numerik relativ gleich wie in v1:
INITIAL_WEIGHT, AGG_DW_MAX, BREAK_DW_MAX x 125/500. Box, alles andere unveraendert.

Ergebnis: Ordner pruefung_R_ncomp500 (neben ergebnisse_gesamt_v1). Den ganzen Ordner
zurueck auf den Laptop kopieren, dort wird ausgewertet.
Fortsetzbar: nach einem Abbruch wieder F5, fertige Laeufe werden uebersprungen.
Dauer: unbekannt, sicher viele Stunden (Kontrollpunkte am laengsten, siehe Fortschritt).
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")       # vor numba, Kinder erben es
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import json                                            # noqa: E402

import numpy as np                                     # noqa: E402

WORKERS = 12
N_NEW, N_OLD = 500, 125
TIME_LIMIT_S = 24 * 3600.0                             # nur Notbremse gegen Haenger
OUT = HERE / "pruefung_R_ncomp500"
CONFIG = "sa_config_gesamt.py"

# Punkt -> (Auswahl, mittlere Laufzeit in v1 [s], Zeile im Zieh-Raum aus gesamt_v1/design.npz)
POINTS = {
    4536: ("schlecht", 101, [-4.8022155912593005, 1.0702646532841027, -6.164924683980644, -0.9063171086619093, 0.7773041103035212, 1.888521734625101, 1.1432057516649365, 1.1144297092966735, 2.21536538861692, 0.678484834358096, 0.9767781682312489, 0.3081603300757706, -0.660022858530283, 10.960289772041142, 0.2518059015274048, -4.13777573670931, -0.16344909497002258, 0.3945872129090552]),
    1333: ("schlecht", 74, [-4.85565957352519, 1.0022391298878937, -6.79542885068804, -2.4929285434792234, 0.46128815524280076, 1.4742074459791183, 1.210853861644864, 1.21009118007496, 2.7181104550138113, 1.1704613164067268, 0.043365039862692356, 0.39804745260626084, -0.5703420341014862, 10.864749506488442, 0.6893202640116215, -4.24638565569477, -1.0609859436417568, 0.8422611244884566]),
    1016: ("schlecht", 76, [-4.902451330050826, 0.7632101811375469, -6.871926870197058, -1.6273543866137459, 0.8115834126248955, 1.7028335463255644, 1.7453147768974304, 1.257319673402235, 2.7302717806771395, 1.385548998042941, 0.7170937703922391, 0.2739553579129279, -0.6108981668949127, 9.837342968396843, 0.4143637493252754, -3.884060020530296, -1.0455185858199942, 1.1709474513124494]),
    1646: ("schlecht", 36, [-4.876402016915381, 0.8436311857588589, -6.552332658320665, -0.7863903095731928, 0.6502012340351939, 1.0894816685467958, 1.8064646795392036, 1.0660762775875627, 2.5404420351609587, 1.2797888653352856, 0.09863607585430145, 0.3541214779950679, -0.709438256919384, 10.578691342379898, 0.5498387001454831, -4.234549487439052, -0.40450849511126175, 1.697808026587414]),
    1786: ("schlecht", 32, [-5.16415543500334, 1.2786174638010561, -6.704577883705497, -2.6953046353320076, 0.863463395088911, 1.4859265685081482, 1.041032461449504, 1.3073357996065171, 2.3054305497556924, 1.3938250213861465, 0.8623541900888085, 0.4261451750993729, -0.7946125324815512, 10.432113796845078, 0.3820975378155709, -4.193140576023672, -0.605601372536515, 1.5064256418356454]),
    272: ("schlecht", 28, [-4.911630336381495, 1.097869179872796, -6.730029683560133, -0.9903835555324747, 0.4258532840758562, 1.890742613002658, 1.3555478770285845, 1.2967876610625537, 2.234718463756144, 0.16537563502788544, 0.8292927695438266, 0.15843014968559146, -1.785541269928217, 10.878977913688868, 0.7878197990357876, -4.126103620185399, -1.0386746696794975, 1.300167054224095]),
    4376: ("schlecht", 28, [-4.930331427790224, 0.8286456661112607, -6.098630395717919, -1.566919873819728, 0.5160733142867684, 1.445221371948719, 1.0851961709558964, 1.2442543654330076, 2.367366161197424, 0.14142814856022595, 0.08609050512313843, 0.18210350750014187, -0.9649461917579174, 10.653469814080745, 0.050190684199333195, -4.294589638680018, -0.2840702294970502, 0.6397340461407995]),
    3743: ("schlecht", 9, [-5.427539497800171, 1.402463510017842, -6.232860718853772, -2.3635126923928453, 0.5203719252720476, 1.2841674834489822, 1.579514354467392, 1.2171481514256448, 2.64089522305876, 1.3832571607083082, 0.033760616555809975, 0.23638799311593175, -1.5853159744292498, 11.356366587989031, 0.7981411375105382, -4.172642052185888, -0.37440808430710926, 1.2263068915281021]),
    2982: ("schlecht", 20, [-5.002897813171148, 0.8248362433351576, -6.408170193433762, -2.07365336211123, 0.4765368761494756, 1.9241959359496832, 1.4812127240002155, 1.3219307920802386, 2.548555486649275, 1.1479243887588382, 0.7866130042821169, 0.17571350550279022, -0.5691402982920408, 10.608306909911335, 0.15157098323106766, -3.9899319724087663, -0.3234102151955117, 1.4693037340397852]),
    1223: ("schlecht", 24, [-4.846999947167933, 1.250269383052364, -6.979808586649597, -0.7509401416192247, 0.577416175417602, 1.7787550892680883, 1.048215557821095, 1.2375688111037015, 2.5058154510334134, 0.2567893398925662, 0.6872860817238688, 0.25856238305568696, -1.1102408207952976, 11.344031136669217, 0.02447004541754723, -4.090543259531261, -0.4855243008250941, 0.5065570780164668]),
    2932: ("Kontrolle", 303, [-4.880395209603011, 0.7922155260294675, -6.244859801605344, -1.777079242632766, 0.4498597906902433, 1.1640534345060587, 1.6860250299796462, 1.0794485200103372, 2.335341427102685, 0.47744019776582713, 0.27546214312314987, 0.42180734211578974, -1.6063261572271585, 9.964542275667192, 0.02693179175257683, -4.07574993145134, -0.1481335147882451, 1.6321250359545894]),
    3480: ("Kontrolle", 198, [-5.229149460047483, 0.8191611697338521, -6.541388970799744, -1.6416297195265486, 0.886151654087007, 1.9058497324585915, 1.9085592525079846, 1.158410475095734, 2.5882394591346385, 1.3450641827657819, 0.4663246227428317, 0.39598313234746463, -2.228098690509796, 9.920649359747769, 0.09679804667830468, -4.057725531432228, -0.562258403156196, 0.5305200919662494]),
}


def say(text: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "fortschritt.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def checks(cfg: dict) -> None:
    """Gleicher Code wie die Studie? Gleiche Design-Zeilen?"""
    import run_sensitivity as rs
    from sa_core import CODE_REFERENCE_FILE, code_fingerprint, compare_fingerprint
    ref = json.loads((HERE / CODE_REFERENCE_FILE).read_text(encoding="utf-8"))["dateien"]
    diff = compare_fingerprint(ref, code_fingerprint())
    if any(diff.values()):
        raise SystemExit(f"CODE WEICHT AB von {CODE_REFERENCE_FILE}: {diff}")
    X = rs.design_matrix(cfg)
    for pt, (_, _, row) in POINTS.items():
        if not np.array_equal(X[pt], np.asarray(row, dtype=float)):
            raise SystemExit(f"Design-Zeile von Punkt {pt} passt nicht zu {CONFIG} (andere Config/SALib?)")
    # Speichern vorab pruefen (ohne Rechnen): genau der Weg, den ein fertiger Lauf am Ende nimmt
    import io
    from sa_core import _collect
    t = build_tasks(cfg, skip_done=False)[0]
    np.savez_compressed(io.BytesIO(), **_collect(t, None, None, "failed", "Probe", 0.0,
                                                 np.asarray(t["t_save"]), t["t_full"]))
    say("Checks OK: Code = code_referenz.json, Design-Zeilen = gesamt_v1, Speichern geht")


def build_tasks(cfg: dict, skip_done: bool = True) -> list:
    import run_sensitivity as rs
    from sa_core import replicate_is_done, replicate_seeds
    n_rep = int(cfg["N_REPLICATES"])
    seeds = replicate_seeds(int(cfg["BASE_PARAMS"]["MC_SEED"]), n_rep)
    f = N_OLD / N_NEW
    tasks = []
    # laengste zuerst (sonst haengt am Ende ein langer Lauf allein)
    for pt, (kind, _, row) in sorted(POINTS.items(), key=lambda kv: -kv[1][1]):
        params, values = rs.to_params(cfg, np.asarray(row, dtype=float))
        if int(params["INITIAL_PARTICLES"]) != N_OLD:
            raise SystemExit(f"INITIAL_PARTICLES ist {params['INITIAL_PARTICLES']}, erwartet {N_OLD}")
        params = dict(params)
        params["INITIAL_PARTICLES"] = N_NEW
        for key in ("INITIAL_WEIGHT", "AGG_DW_MAX", "BREAK_DW_MAX"):
            params[key] = float(params[key]) * f
        for rep in range(n_rep):
            out_path = OUT / "runs" / f"p{pt:05d}_r{rep:02d}.npz"
            if skip_done and replicate_is_done(out_path):
                continue
            # sa_values nur Zahlen (sa_core speichert sie als float); die Auswahl steht in auswahl.json
            tasks.append({"params": params, "sa_values": values,
                          "point": int(pt), "rep": rep, "n_rep_point": n_rep, "seed": seeds[rep],
                          "t_save": np.asarray(cfg["T_SAVE"], dtype=float), "t_full": list(cfg["T_FULL"]),
                          "bin_dt": float(cfg["BIN_DT"]), "probe_dt": float(cfg["PROBE_DT"]),
                          "time_limit_s": TIME_LIMIT_S, "study": "pruefung_R_ncomp500",
                          "out_path": str(out_path)})
    return tasks


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    import run_sensitivity as rs
    from sa_core import run_replicate
    cfg = rs.load_config(HERE / CONFIG)
    checks(cfg)
    (OUT / "auswahl.json").write_text(json.dumps(
        {str(p): {"auswahl": k, "wall_v1_s": w} for p, (k, w, _) in POINTS.items()}, indent=1))
    tasks = build_tasks(cfg)
    total = len(POINTS) * int(cfg["N_REPLICATES"])
    say(f"Start: {len(tasks)} von {total} Laeufen offen, {WORKERS} parallel, N_COMP {N_NEW}")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(run_replicate, t) for t in tasks]
        for k, fu in enumerate(as_completed(futs), 1):
            try:
                r = fu.result()
            except Exception as exc:                   # ein kaputter Lauf haelt die anderen nicht auf
                say(f"FEHLER in einem Lauf: {type(exc).__name__}: {exc} | {k}/{len(tasks)}")
                continue
            say(f"Punkt {r['point']:5d} Seed {r['rep']}: {r['status']} ({r['wall_s'] / 60:.0f} min, "
                f"{r['events']} Events) | {k}/{len(tasks)} | {(time.perf_counter() - t0) / 3600:.1f} h"
                + (f" | {r['error']}" if r["error"] else ""))
    say("fertig. Ordner pruefung_R_ncomp500 auf den Laptop kopieren.")


if __name__ == "__main__":
    main()
