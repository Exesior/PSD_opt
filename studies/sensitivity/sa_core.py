"""
Kern der Sensitivitaetsstudie: EIN Replikat bauen, rechnen, beobachten, speichern.

Der Solver wird ueber ``WetGranulationAdapter._build_solver`` des echten Fit-Adapters
gebaut -> Kernel-Verdrahtung, Startpopulation und Nucleation sind identisch zum
KF3-Fit. Es gibt KEINEN Eingriff in mcpbe, den Adapter oder den Optimizer.

Beobachtet wird ueber den ``cancel_flag``-Haken des Solvers: ``MCPBESolver.solve``
fragt ``cancel_flag.get("cancel")`` genau einmal pro MC-Event ab. ``RunProbe`` nutzt
diesen Aufruf, um
    - Events zu zaehlen (gesamt, akzeptierte Agglomeration, Bruch) und zeitlich zu binnen,
    - in festen Sim-Zeitabstaenden die TATSAECHLICHEN Raten der kontinuierlichen
      Prozesse (Internalisierung, Kompression) aus dem Partikelzustand zu messen,
    - das Zeitlimit durchzusetzen (dann liefert get() True -> Solver bricht ab).
Der Haken liest nur; er veraendert keinen Solver-Zustand.

Gespeichert wird pro Replikat eine .npz-Datei (Rohdaten). Alle Kennzahlen
(Q3 an Maschen, d50, Verschiebung, Diskretisierungs-Markierung, ...) rechnet erst
die Auswertung daraus -> eine geaenderte Kennzahl braucht keinen neuen Lauf.
"""

from __future__ import annotations

import json
import os
import re
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sa_psd import GRID_UM, diameters_and_mass, q3_at

# Format der .npz-Dateien; bei inkompatiblen Aenderungen hochzaehlen.
NPZ_FORMAT = 2

# Fertige Lauf-Dateien heissen p<punkt 5-stellig>_r<replikat 2-stellig>.npz. Halb
# geschriebene Dateien heissen _part_..., passen also NIE auf dieses Muster.
RUN_FILE_RE = re.compile(r"^p(\d{5})_r(\d{2})\.npz$")

# Feste Nummer des Basispunkts (unabhaengig von N, damit das Design verlaengerbar bleibt).
BASE_ID = 99999

# Alle Schluessel, die WetGranulationAdapter._build_solver liest, plus die vier, die
# build_solver / run_replicate selbst brauchen. Fehlt einer, wuerde der Adapter STILL
# seinen Standardwert nehmen (z. B. EKE_CORR_BETA 5e-8 statt 1.3e-15) -> wir pruefen.
REQUIRED_KEYS = (
    "N_MIXER", "AGG_A", "EKE_CORR_BETA", "EKE_C_MIXER",
    "STOKES_U_COLL_REF", "STOKES_C_VEL", "STOKES_BINDER_VISCOSITY", "STOKES_RHO_SOLID",
    "STOKES_RHO_LIQUID", "STOKES_H_A",
    "BREAK_P1", "BREAK_P2", "BREAK_C_MIXER", "BREAKRVAL", "RUMPF_K", "RUMPF_ALPHA",
    "SURFACE_TENSION", "CONTACT_ANGLE", "BREAK_X_S", "BREAK_PORO_MAX",
    "CONE_K_AGG", "CONE_K_BREAK", "CONE_DEFAULT_POROSITY",
    "COMPRESSION_RATE", "MIN_POROSITY", "COMPRESSION_C_MIXER", "LIQUID_INTERNALIZATION_RATE",
    "USE_LIQ_INTERNALISATION_AGGLOMERATION", "USE_UNIFORM_WEIGHTED",
    "MC_SEED", "MAYBE_DOUBLE_VC", "SIZEEVAL", "AGG_DW_MIN", "AGG_DW_MAX", "BREAK_DW_MAX",
    "MERGER_LOOKUP", "MERGER_TOLERANCE",
    "INITIAL_POROSITY", "INITIAL_WEIGHT", "CONTROL_VOLUME", "INITIAL_PARTICLES", "INITIAL_DIAMETER",
    "VOLUMETRIC_FLOW_RATE", "DROPLET_DIAMETER", "LIQUID_ADDITION_START", "LIQUID_ADDITION_DURATION",
    "NUCLEATION_BATCH_SIZE", "MAXITER", "MAX_PARTICLES",
)


def check_params(params: dict) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in params]
    if missing:
        raise KeyError(f"BASE_PARAMS fehlen Schluessel (der Adapter naehme sonst Standardwerte): {missing}")


# ===========================================================================
#  Seeds
# ===========================================================================
def replicate_seeds(base_seed: int, n_rep: int) -> list[int]:
    """Die festen MC-Seeds: fuer JEDEN Parameterpunkt dieselben (Common Random Numbers).

    Abgeleitet wie im Fit-Adapter (``SeedSequence(MC_SEED).generate_state``). Es werden
    immer die ersten ``n_rep`` aus einer festen Liste von 64 genommen -> erhoeht man
    spaeter die Replikatzahl, bleiben die bereits gerechneten Replikate gueltig (und die
    ersten 10 sind genau die Seeds des Fits).
    """
    if not 1 <= n_rep <= 64:
        raise ValueError(f"n_rep must lie in 1..64, got {n_rep}")
    return [int(s) for s in np.random.SeedSequence(int(base_seed)).generate_state(64)[:n_rep]]


# ===========================================================================
#  Code-Stand (gleicher Simulationscode auf Laptop und Remote-Rechner?)
# ===========================================================================
CODE_REFERENCE_FILE = "code_referenz.json"


def code_fingerprint() -> dict:
    """Pruefsumme je Datei des Simulationscodes, der wirklich importiert wird:
    das ganze Paket wmcpbe (ohne Trials/) und der Adapter samt Basisklasse.
    Zeilenenden werden vereinheitlicht (Git wandelt LF/CRLF je nach Rechner um)."""
    import hashlib

    import wmcpbe
    from pbe_optimizer.adapter import wet_granulation_adapter as wga

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]

    files = {}
    root = Path(wmcpbe.__file__).parent
    for f in sorted(root.rglob("*.py")):
        rel = f.relative_to(root)
        if rel.parts[0] == "Trials" or "__pycache__" in rel.parts:
            continue
        files["wmcpbe/" + rel.as_posix()] = digest(f)
    adapter_dir = Path(wga.__file__).parent
    for name in ("wet_granulation_adapter.py", "adapters_api_basics.py"):
        files["pbe_optimizer/adapter/" + name] = digest(adapter_dir / name)
    return files


def compare_fingerprint(ref: dict, now: dict) -> dict:
    """Unterschiede zweier Pruefsummen-Listen: abweichend / fehlt hier / nur hier."""
    return {"abweichend": sorted(k for k in ref if k in now and ref[k] != now[k]),
            "fehlt hier": sorted(k for k in ref if k not in now),
            "nur hier": sorted(k for k in now if k not in ref)}


# ===========================================================================
#  Solver bauen
# ===========================================================================
def build_solver(params: dict, t_vec: np.ndarray, seed: int):
    """WMCPBE-Solver fuer einen vollstaendigen pop_params-Satz ``params`` bauen."""
    from pbe_optimizer.adapter.wet_granulation_adapter import WetGranulationAdapter

    check_params(params)
    t_vec = np.asarray(t_vec, dtype=float)
    # Gleiche Argumente wie OptCore (make_solver(..., t_vec=..., load_attr=False)).
    adapter = WetGranulationAdapter(opt=SimpleNamespace(t_vec=t_vec, max_iter_time=0.0),
                                    role="main", dim=1, t_vec=t_vec, load_attr=False)
    for key, value in params.items():
        setattr(adapter, key, value)
    adapter._initial_diameters = np.full(int(params["INITIAL_PARTICLES"]),
                                         float(params["INITIAL_DIAMETER"]), dtype=float)
    solver = adapter._build_solver(t_vec, seed=int(seed))
    adapter.impl._close(gc_clean=False)          # Platzhalter-Solver des Adapters freigeben
    return solver


# ===========================================================================
#  Beobachter
# ===========================================================================
class RunProbe:
    """Wird als ``solver.cancel_flag`` eingehaengt; ``get("cancel")`` kommt einmal pro Event.

    Ablauf im Solver (``mcpbe_base.solve``): oben in jeder Schleifen-Iteration
    ``cancel_flag.get``, dann EIN MC-Event (Agglomeration oder Bruch, ggf. abgelehnt),
    dann die kontinuierlichen Prozesse ueber das Zeitstueck seit dem letzten Event.
    Beim k-ten Aufruf (k >= 2) ist also das Event der Iteration k-2 fertig und lag bei
    ``solver._elapsed``.

    Gemessene Raten [1/s] (Definition siehe README, Abschnitt Diskretisierung):
        Internalisierung  ODE dl/dt = k_int (L - l)(V_pore - l)  (l = innen, L = gesamt)
                          lokale Relaxationsrate  lambda = k_int * ((V_pore - l) + (L - l))
                          nur Partikel, bei denen beides vorhanden ist (freie Pore UND
                          Aussenfluessigkeit) -> der Prozess arbeitet dort wirklich.
        Kompression       k = rate * n_mixer^c_mixer (konstant), wirkt auf Partikel mit
                          Porositaet > eps_min; gespeichert wird der Massenanteil, der
                          gerade komprimiert wird.

    Physikalische Ereigniszaehler: ``solver.real_agg_events`` / ``real_break_events``
    zaehlen reale Ereignisse in der AKTUELLEN Box. Verdoppelt der Solver die Box
    (MAYBE_DOUBLE_VC), steht jedes spaetere Ereignis fuer ein doppelt so grosses
    Volumen. Darum wird jedes Inkrement mit Vc0/Vc auf die Anfangsbox umgerechnet
    (``real_agg_norm`` / ``real_break_norm``).
    """

    def __init__(self, solver, *, t_end: float, bin_dt: float, probe_dt: float,
                 k_int: float, min_porosity: float, deadline: float | None):
        self.solver = solver
        self.t_end = float(t_end)
        self.bin_dt = float(bin_dt)
        self.probe_dt = float(probe_dt)
        self.k_int = float(k_int)
        self.min_porosity = float(min_porosity)
        self.deadline = deadline
        self.vc0 = float(solver.Vc)

        n_bins = int(np.ceil(self.t_end / self.bin_dt)) + 1
        self.events_per_bin = np.zeros(n_bins, dtype=np.int64)
        self.agg_acc_per_bin = np.zeros(n_bins, dtype=np.int64)
        self.break_per_bin = np.zeros(n_bins, dtype=np.int64)

        self.n_calls = 0
        self.timed_out = False
        self._last_agg = 0.0
        self._last_break = 0.0
        self.real_agg_norm = 0.0
        self.real_break_norm = 0.0
        self._next_probe = 0.0
        self.probe_rows: list[tuple] = []

    # -- vom Solver aufgerufen ------------------------------------------------
    def get(self, key, default=None):
        if key != "cancel":
            return default
        self.n_calls += 1
        if self.n_calls >= 2:
            self._account_event()
        s = self.solver
        if s._elapsed >= self._next_probe:
            self._probe()
            self._next_probe += self.probe_dt
            while self._next_probe <= s._elapsed:     # bei grossen Zeitspruengen
                self._next_probe += self.probe_dt
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            self.timed_out = True
        return self.timed_out

    def __setitem__(self, key, value):
        if key == "cancel" and value:
            self.timed_out = True

    # -- intern ---------------------------------------------------------------
    def _account_event(self):
        s = self.solver
        b = min(int(s._elapsed / self.bin_dt), self.events_per_bin.size - 1)
        self.events_per_bin[b] += 1
        agg = float(s.real_agg_events)
        brk = float(s.real_break_events)
        scale = self.vc0 / float(s.Vc)
        if agg > self._last_agg:
            self.agg_acc_per_bin[b] += 1
            self.real_agg_norm += (agg - self._last_agg) * scale
        if brk > self._last_break:
            self.break_per_bin[b] += 1
            self.real_break_norm += (brk - self._last_break) * scale
        self._last_agg, self._last_break = agg, brk

    def _probe(self):
        s = self.solver
        n = int(s.a_tot)
        t = float(s._elapsed)
        if n < 1:
            self.probe_rows.append((t, 0, float(s.Vc), np.nan, np.nan, 0.0, np.nan, 0.0,
                                    self.real_agg_norm, self.real_break_norm))
            return
        v_solid = s.V_flat[0, :n]
        v_dry = s.V_flat[-1, :n]
        w = s.W[:n]
        poro = s.porosity[:n]
        sat = s.saturation[:n]
        liq = s.liquid_volume[:n]
        mass = w * v_solid
        m_tot = float(np.sum(mass))

        with np.errstate(invalid="ignore"):
            pore_ok = np.isfinite(poro) & (poro > 0.0)
            v_pore = np.where(pore_ok, v_dry * poro, 0.0)
            l_in = np.clip(sat, 0.0, 1.0) * v_pore
            free = v_pore - l_in                      # freie Pore
            ext = liq - l_in                          # Aussenfluessigkeit
            tiny = 1e-9 * np.maximum(v_dry, 1e-300)
            acting = pore_ok & (free > tiny) & (ext > tiny) & np.isfinite(liq)
            comp_acting = pore_ok & (poro > self.min_porosity + 1e-12)

        if np.any(acting):
            lam = self.k_int * (free[acting] + ext[acting])
            m_act = mass[acting]
            lam_mean = float(np.sum(m_act * lam) / np.sum(m_act))
            lam_max = float(np.max(lam))
            lam_frac = float(np.sum(m_act) / m_tot) if m_tot > 0 else 0.0
        else:
            lam_mean, lam_max, lam_frac = np.nan, np.nan, 0.0
        comp_frac = float(np.sum(mass[comp_acting]) / m_tot) if m_tot > 0 else 0.0
        # Reihenfolge = PROBE_COLUMNS
        self.probe_rows.append((t, n, float(s.Vc), lam_mean, lam_max, lam_frac,
                                float(np.sum(w)), comp_frac, self.real_agg_norm, self.real_break_norm))

    def finish(self):
        """Nach solve(): das Event der letzten Iteration nachtragen (falls nicht abgebrochen)."""
        if self.n_calls >= 1 and not self.timed_out:
            self._account_event()
        self._probe()


PROBE_COLUMNS = ("t", "n_comp", "Vc", "lam_int_mean", "lam_int_max",
                 "lam_int_massfrac", "sum_W", "comp_massfrac", "real_agg_norm", "real_break_norm")


# ===========================================================================
#  Ein Replikat rechnen
# ===========================================================================
def run_replicate(task: dict) -> dict:
    """Einen Lauf (ein Parameterpunkt, ein Seed) rechnen und als .npz speichern.

    task-Schluessel:
        params        vollstaendiges pop_params-dict (Basis + Punkt-Werte)
        sa_values     dict nur der variierten Groessen (fuer die Datei/Info)
        point, rep    Indizes; seed
        t_save        Sim-Zeiten der Schnappschuesse [s] (inkl. 0)
        t_full        Zeiten, zu denen die KOMPLETTE Partikelliste gespeichert wird
        bin_dt        Breite der Event-Zaehl-Bins [s]
        probe_dt      Abstand der Raten-Messungen [s]
        time_limit_s  Wandzeit-Limit fuer das Rechnen dieses Replikats [s] (0 = keins);
                      zaehlt ab fertig gebautem Solver (Import/JIT nicht mitgerechnet)
        out_path      Ziel-.npz
        study         Name der Studie (Info)
    Rueckgabe: kleines dict fuer die Fortschrittsanzeige.
    """
    out_path = Path(task["out_path"])
    params = dict(task["params"])
    t_save = np.asarray(task["t_save"], dtype=float)
    t_full = [float(t) for t in task["t_full"]]
    t_end = float(t_save[-1])
    limit = float(task.get("time_limit_s", 0.0) or 0.0)

    wall0 = time.perf_counter()
    status, error = "ok", ""
    solver = probe = None
    try:
        solver = build_solver(params, t_save, int(task["seed"]))
        deadline = time.perf_counter() + limit if limit > 0 else None
        probe = RunProbe(solver, t_end=t_end, bin_dt=float(task["bin_dt"]),
                         probe_dt=float(task["probe_dt"]),
                         k_int=float(params["LIQUID_INTERNALIZATION_RATE"]),
                         min_porosity=float(params["MIN_POROSITY"]), deadline=deadline)
        solver.cancel_flag = probe
        solver.solve(maxiter=int(params["MAXITER"]), max_particles=int(params["MAX_PARTICLES"]))
        probe.finish()
        complete = len(solver.V_save) == t_save.size
        if solver._hit_particle_limit:
            status = "particle_limit"
        elif probe.timed_out and not complete:
            status = "timeout"
        elif not complete:
            status = "incomplete"
    except Exception:                                   # dieses Replikat ist kaputt, Rest laeuft
        status, error = "failed", traceback.format_exc(limit=6)
    finally:
        if solver is not None:
            solver.cancel_flag = None                   # Referenzzyklus Solver <-> Probe loesen
    wall = time.perf_counter() - wall0

    payload = _collect(task, solver, probe, status, error, wall, t_save, t_full)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f"_part_{out_path.stem}_{os.getpid()}.npz")
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, out_path)                          # atomar: halbe Dateien gibt es nicht

    n_events = int(probe.events_per_bin.sum()) if probe is not None else 0
    n_comp_end = int(solver.a_tot) if solver is not None else 0
    if solver is not None:
        solver._close(gc_clean=False)
    return {"point": int(task["point"]), "rep": int(task["rep"]), "status": status,
            "wall_s": wall, "events": n_events, "n_comp_end": n_comp_end,
            "error": error.strip().splitlines()[-1] if error else ""}


def _collect(task, solver, probe, status, error, wall, t_save, t_full) -> dict:
    """Alles, was gespeichert wird (nur numpy-Arrays + ein JSON-String)."""
    params = task["params"]
    k_comp = float(params["COMPRESSION_RATE"]) * float(params["N_MIXER"]) ** float(params["COMPRESSION_C_MIXER"])
    meta = {
        "npz_format": NPZ_FORMAT,
        "study": task.get("study", ""),
        "point": int(task["point"]), "rep": int(task["rep"]), "seed": int(task["seed"]),
        "status": status, "error": error, "wall_s": wall,
        "sa_values": {k: float(v) for k, v in task.get("sa_values", {}).items()},
        "params": {k: (v if isinstance(v, (bool, int, float, str)) or v is None else str(v))
                   for k, v in params.items()},
        "k_comp": k_comp,
        "t_end": float(t_save[-1]),
        "bin_dt": float(task["bin_dt"]), "probe_dt": float(task["probe_dt"]),
        "time_limit_s": float(task.get("time_limit_s", 0.0) or 0.0),
    }
    out = {"meta_json": np.array(json.dumps(meta))}
    if solver is None:
        return out

    n_snap = len(solver.V_save)
    out["t_snap"] = t_save[:n_snap]
    # Q3-Kurven auf dem feinen Raster fuer JEDEN Schnappschuss
    q3 = np.full((n_snap, GRID_UM.size), np.nan)
    n_comp = np.zeros(n_snap, dtype=np.int64)
    for i in range(n_snap):
        d, m = diameters_and_mass(solver.V_save[i], solver.W_save[i])
        q3[i] = q3_at(d, m, GRID_UM)
        n_comp[i] = solver.W_save[i].size
    out["q3_grid"] = q3
    out["n_comp_snap"] = n_comp
    out["Vc_snap"] = np.asarray(solver.Vc_save[:n_snap], dtype=float)
    out["real_agg_snap"] = np.asarray(solver.real_agg_events_save[:n_snap], dtype=float)
    out["real_break_snap"] = np.asarray(solver.real_break_events_save[:n_snap], dtype=float)

    # komplette Partikellisten zu den gewuenschten Zeiten (= "die Verteilung")
    for tf in t_full:
        idx = np.flatnonzero(np.isclose(t_save[:n_snap], tf))
        if idx.size == 0:
            continue
        i = int(idx[0])
        tag = f"t{int(round(tf))}"
        out[f"{tag}_V"] = np.asarray(solver.V_save[i], dtype=float)
        out[f"{tag}_W"] = np.asarray(solver.W_save[i], dtype=float)
        out[f"{tag}_poro"] = np.asarray(solver.porosity_save[i], dtype=float)
        out[f"{tag}_sat"] = np.asarray(solver.saturation_save[i], dtype=float)
        out[f"{tag}_liq"] = np.asarray(solver.liquid_volume_save[i], dtype=float)

    if probe is not None:
        out["events_per_bin"] = probe.events_per_bin
        out["agg_acc_per_bin"] = probe.agg_acc_per_bin
        out["break_per_bin"] = probe.break_per_bin
        out["probe"] = np.asarray(probe.probe_rows, dtype=float).reshape(-1, len(PROBE_COLUMNS))
        out["probe_columns"] = np.array(PROBE_COLUMNS)
    return out


# ===========================================================================
#  Lesen
# ===========================================================================
def read_replicate(path: Path) -> dict:
    """.npz eines Replikats lesen -> dict mit 'meta' (dict) und allen Arrays."""
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    data["meta"] = json.loads(str(data.pop("meta_json")))
    return data


def read_meta(path: Path) -> dict | None:
    """Nur die Metadaten eines Laufs (schnell); None, wenn die Datei fehlt/kaputt ist."""
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            return json.loads(str(z["meta_json"]))
    except Exception:
        return None


def replicate_is_done(path: Path) -> bool:
    """True, wenn die Datei existiert und lesbar ist (fuer das Fortsetzen nach Abbruch)."""
    return read_meta(path) is not None


def run_files(runs_dir: Path) -> list[Path]:
    """Alle FERTIGEN Lauf-Dateien (keine halb geschriebenen _part_-Dateien)."""
    return sorted(p for p in Path(runs_dir).glob("*.npz") if RUN_FILE_RE.match(p.name))
