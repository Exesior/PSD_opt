"""Operator-Splitting-Diagnose: wie stark aendert sich der Zustand zwischen zwei Events?

`solve()` fuehrt in jeder Iteration erst das MC-Event aus (mit Raten und
Porositaeten vom Stand t_{n-1}) und zieht danach die kontinuierlichen Prozesse
(Fluessigkeitsaufnahme + Kompression) ueber das ganze Intervall
dt_n = t_n - t_{n-1} nach (mcpbe_base.py, `continuous_processes.step`). Dieses
Skript misst, ohne den Lauf zu veraendern, wie gross der dabei gemachte Fehler
ist:

Kennzahl 1 -- Zustandsaenderung pro Intervall
    theta_n   = max_i k_eff,i * dt_n, aus der tatsaechlichen Porositaetsaenderung
                rueckgerechnet: ln((eps_alt - eps_min) / (eps_neu - eps_min)).
    |d eps|, |d S|, |dX/X|: massengewichteter Mittelwert und Maximum.

Kennzahl 2 -- Drift der Gesamtpropensity durch den kontinuierlichen Schritt
    delta_n = |A_nach - A_vor| / A_vor, getrennt fuer Agglomeration, Bruch und
    die Gesamtrate im mix-Modus. Dazu werden die Propensities unmittelbar vor
    und nach `continuous_processes.step` voll neu berechnet. Nukleation ist
    NICHT enthalten -- gemessen wird nur der kontinuierliche Operator.
    Zusaetzlich |d p_agg|: Aenderung der Wahrscheinlichkeit, dass im mix-Modus
    Agglomeration statt Bruch gewuerfelt wird.

Kennzahl 3 -- gekippte Akzeptanz-Entscheidungen (Stokes-Kriterium)
    Bei jedem Agglomerationsversuch wird das Partnerpaar vor dem Merge kopiert,
    ueber dt_n mit denselben Kernels vorgezogen (eigene Kopie, der Solver wird
    nicht beruehrt) und das Akzeptanzkriterium erneut ausgewertet. Ein "Flip"
    ist ein Versuch, dessen Entscheidung mit dem Zustand bei t_n anders
    ausgefallen waere als mit dem Zustand bei t_{n-1}. Vergleichsmassstab ist
    das Poisson-Rauschen sqrt(N_akzeptiert).

Die Diagnose haengt sich nur ueber Wrapper auf INSTANZ-Ebene ein (Handler-
`step`, `_do_one_agg`, `_do_one_break`, `accept_collision`). Die zusaetzlichen
Propensity-Rebuilds werden vom regulaeren Rebuild am Ende jeder Iteration
ueberschrieben. Mit `--check-identical` wird das bitweise nachgewiesen: derselbe
Lauf ohne Diagnose muss exakt denselben Endzustand liefern.

Aufruf (aus dem Ordner `mcpbe`, damit `tests.bench.scenarios` importierbar ist):

    python -m studies.operator_splitting.diagnose_splitting --check-identical
        (Default-Szenario: wet_granulation_production, siehe production_scenario.py)
    python -m studies.operator_splitting.diagnose_splitting \
        --scenario granulation_rumpf_dynamic_1d --maxiter 300 --check-identical

Ausgabe: `results/<scenario>_n<maxiter>_s<seed>/` mit
    per_iteration.npz  -- eine Zeile pro Iteration (alle Kennzahlen), immer
    per_iteration.csv  -- dasselbe als CSV (repr-genau), solange es in Excel passt
    summary.json       -- Verteilungen, Schwellen, Flip-Bilanz, Zeitfenster
Spalte `event`: 0 = kein Event, 1 = Agglomeration, 2 = Bruch.
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import json
import math
import sys
import time
from array import array
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from tests.bench.scenarios import SCENARIOS, build

from . import production_scenario as prod

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
TOLERANCES = (1e-3, 1e-2, 1e-1)
EVENT_CODE = {"": 0, "agg": 1, "break": 2}
EXCEL_MAX_ROWS = 1_048_575
# v2 (2026-10-14): Partikel mit eps - eps_min <= 1e-8 fliessen nicht mehr in theta ein.
THETA_ESTIMATOR = "v2"
# v3 (2026-10-19): zusaetzliche Spalten k_S (Fluessigkeitsaufnahme je Partikel, exakt
# ueber das Intervall), groesster Partikeldurchmesser, Bruchraten-Aenderung je Partikel.
DIAG_VERSION = "v3"
THETA_S_CAP = 50.0   # theta_S, wenn ein Partikel im Intervall das Gleichgewicht erreicht


class _ColumnStore:
    """Eine Zeile pro Iteration, spaltenweise als array('d') -- 8 Byte je Wert
    statt eines Python-Dicts je Iteration (Produktionslaeufe: viele 1e5 Events)."""

    def __init__(self):
        self._cols: Dict[str, array] = {}
        self.n = 0

    def append(self, rec: Dict[str, Any]) -> None:
        if not self._cols:
            self._cols = {k: array("d") for k in rec}
        for k, v in rec.items():
            self._cols[k].append(float(v))
        self.n += 1

    def arrays(self) -> Dict[str, np.ndarray]:
        return {k: np.frombuffer(v, dtype=float) for k, v in self._cols.items()}


# ---------------------------------------------------------------------------
# Zwei-Partikel-Kopie fuer die Flip-Auswertung
# ---------------------------------------------------------------------------
class _PairShim:
    """Zwei kopierte Partikel, die sich fuer Handler und Kernels wie ein Solver
    mit `a_tot = 2` verhalten.

    Alles, was die Kernels SCHREIBEN (porosity, V_flat, saturation, X), ist eine
    eigene Kopie; `sync_particle_diameters` ist hier neu definiert, damit nicht
    die gebundene Methode des echten Solvers dessen `X` ueberschreibt. Nur
    LESENDE Zugriffe (X0, process_type, _vol2diam, ...) gehen an den Solver.
    """

    def __init__(self, solver, state: Dict[str, np.ndarray]):
        self._solver = solver
        self.a_tot = 2
        self.porosity = state["porosity"].copy()
        self.saturation = state["saturation"].copy()
        self.liquid_volume = state["liquid_volume"].copy()
        self.V_flat = state["V_flat"].copy()
        self.X = state["X"].copy()
        self._particle_merger = None
        self.mcpbe_debug_mass = False

    def sync_particle_diameters(self, mask=None) -> None:
        sel = np.arange(2) if mask is None else np.flatnonzero(mask)
        if sel.size:
            self.X[sel] = self._solver._vol2diam(self.V_flat[-1, sel])

    def __getattr__(self, name):  # nur fuer Attribute, die oben fehlen
        return getattr(self._solver, name)


def _pair_state(solver, i: int, j: int) -> Dict[str, np.ndarray]:
    idx = np.array([i, j], dtype=np.int64)
    return {
        "porosity": solver.porosity[idx].copy(),
        "saturation": solver.saturation[idx].copy(),
        "liquid_volume": solver.liquid_volume[idx].copy(),
        "V_flat": solver.V_flat[:, idx].copy(),
        "X": solver.X[idx].copy(),
    }


# ---------------------------------------------------------------------------
# Diagnose
# ---------------------------------------------------------------------------
class SplittingDiagnostics:
    def __init__(self, solver, delta_stride: int = 1):
        self.solver = solver
        self.delta_stride = max(1, int(delta_stride))
        self.records = _ColumnStore()
        self._t_wall0 = time.perf_counter()
        self._t_wall_last = self._t_wall0
        self._iter = 0
        self._event_kind = ""
        self._pending_pair: Optional[Dict[str, Any]] = None

        handler = getattr(solver, "continuous_processes", None)
        if handler is None:
            raise RuntimeError("Szenario hat keinen ContinuousProcessesHandler.")
        self.handler = handler
        self._orig_step = handler.step
        self.eps_min = self._resolve_eps_min(handler)

        self._orig_do_one_agg = solver._do_one_agg
        self._orig_do_one_break = solver._do_one_break
        acc = solver.kernel_manager.agglomeration_acceptance_kernel
        self.acceptance = acc
        self._orig_accept = acc.accept_collision if acc is not None else None

    @staticmethod
    def _resolve_eps_min(handler) -> float:
        kernel = handler.compression_kernel
        for name in ("eps_min", "min_porosity"):
            val = getattr(kernel, name, None)
            if val is not None:
                return float(val)
        params = getattr(kernel, "params", None) or {}
        if "min_porosity" in params:
            return float(params["min_porosity"])
        return float(handler.config.min_porosity)

    # --- Wrapper installieren ---------------------------------------------
    def install(self) -> None:
        s = self.solver

        def do_one_agg():
            self._event_kind = "agg"
            return self._orig_do_one_agg()

        def do_one_break():
            self._event_kind = "break"
            return self._orig_do_one_break()

        s._do_one_agg = do_one_agg
        s._do_one_break = do_one_break
        self.handler.step = self._step

        if self.acceptance is not None:
            orig = self._orig_accept

            def accept_collision(r1, r2, v_dry1, v_dry2, particle1_idx=None,
                                 particle2_idx=None, solver=None):
                res = orig(r1=r1, r2=r2, v_dry1=v_dry1, v_dry2=v_dry2,
                           particle1_idx=particle1_idx, particle2_idx=particle2_idx,
                           solver=solver)
                if solver is s and particle1_idx is not None and particle2_idx is not None:
                    self._pending_pair = {
                        "decision": bool(res),
                        "state": _pair_state(s, int(particle1_idx), int(particle2_idx)),
                    }
                return res

            self.acceptance.accept_collision = accept_collision

    # --- Propensity-Summen ------------------------------------------------
    def _rates(self) -> Dict[str, float]:
        s = self.solver
        a = int(s.a_tot)
        pt = s.process_type
        a_agg = a_br = float("nan")
        if pt in ("agglomeration", "mix"):
            s._rebuild_all_propensities()
            a_agg = float(np.sum(s._r_agg[:a]))
        br = None
        if pt in ("breakage", "mix"):
            s._calc_break_rates_full()
            br = s._break_rate[:a].copy()     # Einzelraten, fuer die Auswertung je Partikel
            a_br = float(np.sum(br))
        out = {"A_agg": a_agg, "A_break": a_br, "_br": br}
        if pt == "mix":
            r_agg = s._agg_rate_from_sum_prop(a_agg)
            r_tot = s._mix_total_rate_from_sum_prop(a_agg, a_br)
            out["rate"] = r_tot
            out["p_agg"] = r_agg / r_tot if r_tot > 0 else float("nan")
        elif pt == "agglomeration":
            out["rate"] = s._agg_rate_from_sum_prop(a_agg)
            out["p_agg"] = 1.0
        else:
            out["rate"] = a_br
            out["p_agg"] = 0.0
        return out

    # --- der eigentliche Mess-Schritt ---------------------------------------
    def _step(self, current_time: float, dt_event: float) -> None:
        s = self.solver
        a = int(s.a_tot)
        n = self._iter
        self._iter += 1

        rec: Dict[str, Any] = {
            "iter": n,
            "t": float(current_time),
            "dt": float(dt_event),
            "event": EVENT_CODE[self._event_kind],
            "dW_agg": float(getattr(s, "_last_agg_dW", 0.0)) if self._event_kind == "agg" else 0.0,
            "dW_break": float(getattr(s, "_last_break_dW", 0.0)) if self._event_kind == "break" else 0.0,
            "a_tot": a,
        }
        self._event_kind = ""

        poro0 = s.porosity[:a].copy()
        sat0 = s.saturation[:a].copy()
        X0 = s.X[:a].copy()
        vdry0 = s.V_flat[-1, :a].copy()
        W0 = s.W[:a].copy()
        liq0 = s.liquid_volume[:a].copy()

        measure_delta = (n % self.delta_stride) == 0
        pre = self._rates() if measure_delta else None

        self._orig_step(current_time, dt_event)

        post = self._rates() if measure_delta else None
        rec.update(self._state_change(poro0, sat0, X0, vdry0, W0, a, float(dt_event)))

        for key in ("A_agg", "A_break", "rate", "p_agg"):
            rec[f"{key}_pre"] = pre[key] if pre else float("nan")
            rec[f"{key}_post"] = post[key] if post else float("nan")
        for key, name in (("A_agg", "delta_agg"), ("A_break", "delta_break"), ("rate", "delta_rate")):
            v0, v1 = rec[f"{key}_pre"], rec[f"{key}_post"]
            rec[name] = (v1 - v0) / v0 if (np.isfinite(v0) and v0 > 0 and np.isfinite(v1)) else float("nan")
        rec["dp_agg"] = rec["p_agg_post"] - rec["p_agg_pre"]

        rec.update(self._flip(float(dt_event)))
        br0 = pre["_br"] if pre else None
        br1 = post["_br"] if post else None
        rec.update(self._saturation_rate(poro0, sat0, vdry0, liq0, W0, X0, br0, float(dt_event)))
        rec.update(self._break_per_particle(br0, br1))
        self.records.append(rec)

        now = time.perf_counter()
        if now - self._t_wall_last >= 60.0:
            self._t_wall_last = now
            print(f"  [diag] t = {current_time:9.3f} s, Iteration {n + 1}, a_tot {a}, "
                  f"{now - self._t_wall0:7.0f} s Wandzeit", flush=True)

    def _state_change(self, poro0, sat0, X0, vdry0, W0, a, dt) -> Dict[str, float]:
        s = self.solver
        poro1 = s.porosity[:a]
        sat1 = s.saturation[:a]
        X1 = s.X[:a]

        porous = ~np.isnan(poro0)
        v_solid = np.where(porous, vdry0 * (1.0 - np.nan_to_num(poro0)), vdry0)
        mass = W0 * v_solid
        m_tot = float(np.sum(mass))

        def wmean(x):
            return float(np.sum(mass * x) / m_tot) if m_tot > 0 else float("nan")

        d_eps = np.where(porous, np.abs(np.nan_to_num(poro1 - poro0)), 0.0)
        d_sat = np.abs(np.nan_to_num(sat1 - sat0))
        with np.errstate(divide="ignore", invalid="ignore"):
            d_x = np.where(X0 > 0, np.abs(X1 - X0) / X0, 0.0)

        # k_eff * dt aus der tatsaechlichen Porositaetsaenderung
        num = poro0 - self.eps_min
        den = poro1 - self.eps_min
        # Abstand zu eps_min > 1e-8: direkt an eps_min ist der log-Quotient
        # Rundungsrauschen (gemessen: theta = 0.29 bei |d eps| ~ 1e-10, wahr 0.16).
        ok = porous & (num > 1e-8) & (den > 0) & (poro1 < poro0)
        kdt = np.zeros(a)
        kdt[ok] = np.log(num[ok] / den[ok])

        return {
            "theta_max": float(np.max(kdt)) if a else 0.0,
            "theta_mean_m": wmean(kdt),
            "k_eff_max": float(np.max(kdt)) / dt if (a and dt > 0) else float("nan"),
            "deps_mean_m": wmean(d_eps),
            "deps_max": float(np.max(d_eps)) if a else 0.0,
            "dS_mean_m": wmean(d_sat),
            "dS_max": float(np.max(d_sat)) if a else 0.0,
            "dX_rel_mean_m": wmean(d_x),
            "dX_rel_max": float(np.max(d_x)) if a else 0.0,
        }

    def _saturation_rate(self, poro0, sat0, vdry0, liq0, W0, X0, br0, dt) -> Dict[str, float]:
        """Wie schnell laeuft die Fluessigkeitsaufnahme je Partikel? (exakt ueber das Intervall)

        Derselbe Kernel-Aufruf wie in ContinuousProcessesHandler._apply_internalization,
        aber auf KOPIEN des Zustands bei t_(n-1) -- nur der Internalisierungs-Anteil,
        ohne den Einfluss der anschliessenden Kompression. Je Partikel:
            d      = min(L, V_p) - l_int      Abstand zum Gleichgewicht
            theta_S = ln(d_alt / d_neu),  k_S = theta_S / dt
        analog zu theta = ln((eps_alt - eps_min) / (eps_neu - eps_min)) der Kompression.
        Erreicht ein Partikel im Intervall das Gleichgewicht (d_neu = 0), wird
        theta_S auf THETA_S_CAP gesetzt und in n_S_capped gezaehlt.
        """
        nan = float("nan")
        out = {"thetaS_max": nan, "kS_max": nan, "kS_mean_m": nan, "kS_mean_br": nan,
               "n_S_active": 0, "n_S_capped": 0,
               "x_max": float(np.max(X0)) if X0.size else nan}
        h = self.handler
        if dt <= 0.0 or not h.config.enabled or X0.size == 0:
            return out
        v_pore = vdry0 * poro0
        mask = ~np.isnan(poro0) & (poro0 > 0) & (v_pore > 0) & np.isfinite(v_pore)
        if not np.any(mask):
            return out
        vp, L, s_old = v_pore[mask], liq0[mask], sat0[mask]
        if h.internalization_kernel is not None:
            s_new = h.internalization_kernel.compute_array(
                saturation=s_old.copy(), v_pore=vp, l_total=L, dt=dt)
        else:
            s_new = h._internalization_fallback_array(s_old.copy(), vp, L, dt)
        l_eq = np.minimum(L, vp)
        d_old = l_eq - np.clip(s_old, 0.0, 1.0) * vp
        d_new = l_eq - np.clip(s_new, 0.0, 1.0) * vp
        act = (L > 0) & (d_old > 1e-8 * vp)          # nimmt noch Fluessigkeit auf
        n_act = int(np.count_nonzero(act))
        out["n_S_active"] = n_act
        if n_act == 0:
            return out
        do, dn = d_old[act], d_new[act]
        theta = np.full(n_act, THETA_S_CAP)
        pos = dn > 0
        theta[pos] = np.log(do[pos] / dn[pos])
        theta = np.clip(theta, 0.0, THETA_S_CAP)
        out["n_S_capped"] = int(np.count_nonzero(theta >= THETA_S_CAP))
        k_s = theta / dt

        mass = (W0 * vdry0 * (1.0 - poro0))[mask][act]
        out["thetaS_max"] = float(np.max(theta))
        out["kS_max"] = float(np.max(k_s))
        if np.sum(mass) > 0:
            out["kS_mean_m"] = float(np.sum(mass * k_s) / np.sum(mass))
        if br0 is not None:
            w = br0[mask][act]
            if np.sum(w) > 0:
                out["kS_mean_br"] = float(np.sum(w * k_s) / np.sum(w))
        return out

    @staticmethod
    def _break_per_particle(br0, br1) -> Dict[str, float]:
        """Groesste relative Bruchraten-Aenderung EINES Partikels im kontinuierlichen Schritt.

        r_i = (S_i,nach - S_i,vor) / S_i,vor fuer alle Partikel mit S_i,vor > 0, dazu der
        Anteil dieses Partikels an der Gesamtbruchrate vor dem Schritt (wie wahrscheinlich
        es als Bruchpartikel gewuerfelt worden waere). Partikel mit S_vor = 0 und S_nach > 0
        (relative Aenderung unendlich) werden in n_br_from_zero gezaehlt.
        """
        nan = float("nan")
        out = {"dbr_part_min": nan, "dbr_part_min_share": nan,
               "dbr_part_max": nan, "dbr_part_max_share": nan, "n_br_from_zero": 0}
        if br0 is None or br1 is None or br0.size == 0:
            return out
        pos = br0 > 0
        out["n_br_from_zero"] = int(np.count_nonzero(~pos & (br1 > 0)))
        tot = float(np.sum(br0))
        if not np.any(pos) or tot <= 0:
            return out
        r = (br1[pos] - br0[pos]) / br0[pos]
        share = br0[pos] / tot
        i, j = int(np.argmin(r)), int(np.argmax(r))
        out.update({"dbr_part_min": float(r[i]), "dbr_part_min_share": float(share[i]),
                    "dbr_part_max": float(r[j]), "dbr_part_max_share": float(share[j])})
        return out

    def _flip(self, dt: float) -> Dict[str, Any]:
        pending, self._pending_pair = self._pending_pair, None
        if pending is None:
            return {"acc_checked": 0, "acc_old": -1, "acc_new": -1, "flip": 0, "shim_mismatch": 0}
        # Gegenprobe: unveraenderte Kopie (dt = 0) muss exakt die Original-
        # Entscheidung reproduzieren, sonst rechnet die Kopie falsch.
        same = self._accept_on(_PairShim(self.solver, pending["state"]))
        shim = _PairShim(self.solver, pending["state"])
        if dt > 0.0 and self.handler.config.enabled:
            h = copy.copy(self.handler)
            h.solver = shim
            h._apply_internalization(dt)
            if h.config.compression_enabled:
                h._apply_compression(dt)
        new = self._accept_on(shim)
        old = pending["decision"]
        return {"acc_checked": 1, "acc_old": int(old), "acc_new": int(new),
                "flip": int(old != new), "shim_mismatch": int(same != old)}

    def _accept_on(self, shim: _PairShim) -> bool:
        return bool(self._orig_accept(
            r1=float(shim.X[0]) * 0.5, r2=float(shim.X[1]) * 0.5,
            v_dry1=float(shim.V_flat[-1, 0]), v_dry2=float(shim.V_flat[-1, 1]),
            particle1_idx=0, particle2_idx=1, solver=shim,
        ))


# ---------------------------------------------------------------------------
# Auswertung
# ---------------------------------------------------------------------------
def _quantiles(x: np.ndarray) -> Dict[str, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    q = np.quantile(x, [0.5, 0.9, 0.99])
    return {"n": int(x.size), "median": float(q[0]), "p90": float(q[1]),
            "p99": float(q[2]), "max": float(np.max(x))}


def summarize(col: Dict[str, np.ndarray], n_bins: int = 10) -> Dict[str, Any]:
    t, dt = col["t"].astype(float), col["dt"].astype(float)
    is_agg = col["event"] == EVENT_CODE["agg"]
    dW_agg = col["dW_agg"].astype(float)
    abs_rate = np.abs(col["delta_rate"].astype(float))

    out: Dict[str, Any] = {
        "n_iterations": int(t.size),
        "t_end_reached": float(t[-1]),
        "n_agg_events": int(np.sum(is_agg)),
        "n_break_events": int(np.sum(col["event"] == EVENT_CODE["break"])),
        "dt": _quantiles(dt),
        "theta_max": _quantiles(col["theta_max"].astype(float)),
        "deps_max": _quantiles(col["deps_max"].astype(float)),
        "dS_max": _quantiles(col["dS_max"].astype(float)),
        "dX_rel_max": _quantiles(col["dX_rel_max"].astype(float)),
        "abs_delta_agg": _quantiles(np.abs(col["delta_agg"].astype(float))),
        "abs_delta_break": _quantiles(np.abs(col["delta_break"].astype(float))),
        "abs_delta_rate": _quantiles(abs_rate),
        "abs_dp_agg": _quantiles(np.abs(col["dp_agg"].astype(float))),
    }
    # v3-Spalten (fehlen in aelteren Laeufen)
    for key, src in (("thetaS_max", "thetaS_max"), ("kS_max", "kS_max"),
                     ("abs_dbr_part_min", "dbr_part_min"), ("x_max", "x_max")):
        if src in col:
            out[key] = _quantiles(np.abs(col[src].astype(float)))

    # Anteil Iterationen bzw. Agglomerations-Gewicht oberhalb der Toleranzen
    finite = np.isfinite(abs_rate)
    thr = {}
    for tol in TOLERANCES:
        above = finite & (abs_rate > tol)
        w_tot = float(np.sum(dW_agg[finite]))
        thr[f"{tol:g}"] = {
            "frac_iterations": float(np.sum(above) / max(1, np.sum(finite))),
            "frac_agg_weight": float(np.sum(dW_agg[above]) / w_tot) if w_tot > 0 else float("nan"),
            "t_last_above": float(np.max(t[above])) if np.any(above) else None,
        }
    out["delta_rate_thresholds"] = thr

    # Geschaetzter Zeitversatz durch eingefrorene Rate: sum 0.5 * delta * dt
    d = np.nan_to_num(col["delta_rate"].astype(float))
    out["clock_shift_estimate"] = {
        "signed_s": float(np.sum(0.5 * d * dt)),
        "abs_s": float(np.sum(0.5 * np.abs(d) * dt)),
        "relative_to_t": float(np.sum(0.5 * np.abs(d) * dt) / t[-1]) if t[-1] > 0 else float("nan"),
        "note": "nur aus Iterationen mit Delta-Messung (delta_stride beachten)",
    }

    # Flip-Bilanz
    checked = col["acc_checked"].astype(int) == 1
    old = col["acc_old"].astype(int)
    new = col["acc_new"].astype(int)
    n_acc = int(np.sum(checked & (old == 1)))
    a2r = int(np.sum(checked & (old == 1) & (new == 0)))
    r2a = int(np.sum(checked & (old == 0) & (new == 1)))
    flips = a2r + r2a
    out["acceptance_flips"] = {
        "n_checked": int(np.sum(checked)),
        "n_accepted": n_acc,
        "flips_total": flips,
        "accepted_to_rejected": a2r,
        "rejected_to_accepted": r2a,
        "net_change_in_accepted": r2a - a2r,
        "sqrt_n_accepted": math.sqrt(n_acc),
        "flips_over_sqrt_n": flips / math.sqrt(n_acc) if n_acc > 0 else float("nan"),
        "dW_of_accepted_to_rejected": float(np.sum(dW_agg[checked & (old == 1) & (new == 0)])),
        "dW_accepted_total": float(np.sum(dW_agg[checked & (old == 1)])),
        "shim_mismatches_dt0": int(np.sum(col["shim_mismatch"].astype(int))),
    }
    # Erwartete gekippte Ereignistyp-Wahlen im mix-Modus
    dp = np.abs(np.nan_to_num(col["dp_agg"].astype(float)))
    out["event_type_flips_expected"] = float(np.sum(dp))

    # Zeitfenster
    edges = np.linspace(0.0, t[-1], n_bins + 1)
    bins = []
    for k in range(n_bins):
        m = (t >= edges[k]) & ((t < edges[k + 1]) if k < n_bins - 1 else (t <= edges[k + 1]))
        if not np.any(m):
            continue
        ar = abs_rate[m]
        ar = ar[np.isfinite(ar)]
        bins.append({
            "t_from": float(edges[k]), "t_to": float(edges[k + 1]),
            "n_iter": int(np.sum(m)),
            "dt_median": float(np.median(dt[m])),
            "theta_max_median": float(np.median(col["theta_max"][m].astype(float))),
            "abs_delta_rate_median": float(np.median(ar)) if ar.size else None,
            "abs_delta_rate_max": float(np.max(ar)) if ar.size else None,
            "flips": int(np.sum(col["flip"][m].astype(int))),
            "n_checked": int(np.sum(checked[m])),
        })
    out["time_bins"] = bins
    return out


def print_summary(sm: Dict[str, Any]) -> None:
    def q(name):
        d = sm[name]
        if d.get("n", 0) == 0:
            return f"{name:16s} -"
        return (f"{name:16s} median {d['median']:.3e}  p90 {d['p90']:.3e}  "
                f"p99 {d['p99']:.3e}  max {d['max']:.3e}")

    print(f"\nIterationen {sm['n_iterations']}, t erreicht {sm['t_end_reached']:.3f} s, "
          f"Agg {sm['n_agg_events']}, Bruch {sm['n_break_events']}")
    for name in ("dt", "theta_max", "deps_max", "dS_max", "dX_rel_max",
                 "abs_delta_agg", "abs_delta_break", "abs_delta_rate", "abs_dp_agg",
                 "thetaS_max", "kS_max", "abs_dbr_part_min", "x_max"):
        if name in sm:
            print("  " + q(name))
    print("\n|delta_rate| oberhalb Toleranz:")
    for tol, d in sm["delta_rate_thresholds"].items():
        print(f"  > {tol:6s}: {100 * d['frac_iterations']:6.2f} % der Iterationen, "
              f"{100 * d['frac_agg_weight']:6.2f} % des Agg-Gewichts, "
              f"zuletzt bei t = {d['t_last_above']}")
    c = sm["clock_shift_estimate"]
    print(f"\nZeitversatz-Schaetzung: {c['abs_s']:.3e} s (|.|), {c['signed_s']:+.3e} s (signiert), "
          f"{100 * c['relative_to_t']:.4f} % von t")
    f = sm["acceptance_flips"]
    print(f"\nAkzeptanz-Flips: {f['flips_total']} von {f['n_checked']} Versuchen "
          f"(akz->abg {f['accepted_to_rejected']}, abg->akz {f['rejected_to_accepted']}); "
          f"N_akz = {f['n_accepted']}, sqrt(N_akz) = {f['sqrt_n_accepted']:.2f}, "
          f"F/sqrt(N) = {f['flips_over_sqrt_n']:.3f}")
    print(f"  Gegenprobe dt=0 (muss 0 sein): {f['shim_mismatches_dt0']} Abweichungen")
    print(f"Erwartete gekippte Ereignistyp-Wahlen (mix): {sm['event_type_flips_expected']:.3e}")
    print("\nZeitfenster:  t_von   t_bis  n_iter   dt_med     theta_med  |d|_med    |d|_max    flips/checked")
    for b in sm["time_bins"]:
        dm = b["abs_delta_rate_median"]
        dx = b["abs_delta_rate_max"]
        print(f"  {b['t_from']:8.2f} {b['t_to']:7.2f} {b['n_iter']:6d}  {b['dt_median']:.3e}  "
              f"{b['theta_max_median']:.3e}  "
              f"{(dm if dm is not None else float('nan')):.3e}  "
              f"{(dx if dx is not None else float('nan')):.3e}  {b['flips']}/{b['n_checked']}")


# ---------------------------------------------------------------------------
# Lauf
# ---------------------------------------------------------------------------
def _final_state(solver) -> Dict[str, np.ndarray]:
    a = int(solver.a_tot)
    return {
        "V_flat": solver.V_flat[:, :a].copy(),
        "W": solver.W[:a].copy(),
        "porosity": solver.porosity[:a].copy(),
        "saturation": solver.saturation[:a].copy(),
        "liquid_volume": solver.liquid_volume[:a].copy(),
        "t_right": np.asarray(solver.t_right, dtype=float).copy(),
    }


def _identical(s1, s2) -> Dict[str, bool]:
    out = {}
    for k in s1:
        a, b = s1[k], s2[k]
        out[k] = bool(a.shape == b.shape and np.array_equal(a, b, equal_nan=True))
    return out


def _run_spec(args):
    """(Name, Builder ohne Argumente, maxiter, max_particles, seed) je Szenario."""
    prod_pops = {prod.NAME: prod.POP, prod.NAME_BEST_1070: prod.POP_BEST_1070}
    if args.scenario in prod_pops:
        pop = prod_pops[args.scenario]
        seed = prod.replicate_seed(0) if args.seed is None else int(args.seed)
        maxiter = int(pop["MAXITER"]) if args.maxiter is None else int(args.maxiter)
        return (args.scenario, lambda: prod.build_production(seed=seed, pop=pop),
                maxiter, int(pop["MAX_PARTICLES"]), seed)
    sc = SCENARIOS[args.scenario]
    repl = {}
    if args.maxiter is not None:
        repl["maxiter"] = args.maxiter
    if args.seed is not None:
        repl["seed"] = args.seed
    sc = dataclasses.replace(sc, **repl)
    return sc.name, (lambda: build(sc)), sc.maxiter, None, sc.seed


def run_one(name, make, maxiter, max_particles, seed, delta_stride=1,
            check_identical=False, pop=None, quiet=False) -> Path:
    """Einen Lauf mit Diagnose rechnen und unter results/ ablegen; gibt den Ordner zurueck.

    `pop` (optional): die Parameter des Laufs, werden als pop.json mitgeschrieben.
    """
    out_dir = RESULTS_DIR / f"{name}_n{maxiter}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    solver = make()
    diag = SplittingDiagnostics(solver, delta_stride=delta_stride)
    diag.install()
    t0 = time.perf_counter()
    solver.solve(maxiter=maxiter, max_particles=max_particles)
    wall = time.perf_counter() - t0
    hit_limit = bool(getattr(solver, "_hit_particle_limit", False))
    print(f"Lauf mit Diagnose: {name}, maxiter {maxiter}, seed {seed}, "
          f"{wall:.1f} s Wandzeit, eps_min = {diag.eps_min}"
          + ("  [MAX_PARTICLES erreicht]" if hit_limit else ""), flush=True)

    identical = None
    wall_ref = None
    if check_identical:
        ref = make()
        t0 = time.perf_counter()
        ref.solve(maxiter=maxiter, max_particles=max_particles)
        wall_ref = time.perf_counter() - t0
        identical = _identical(_final_state(solver), _final_state(ref))
        ok = all(identical.values())
        print(f"Referenzlauf ohne Diagnose: {wall_ref:.1f} s; Endzustand bitgleich: {ok}", flush=True)
        if not ok:
            print("  ABWEICHUNG in: " + ", ".join(k for k, v in identical.items() if not v))

    if diag.records.n == 0:
        raise RuntimeError(f"{name}: keine Iterationen aufgezeichnet.")
    cols = diag.records.arrays()
    np.savez_compressed(out_dir / "per_iteration.npz", **cols)
    if diag.records.n <= EXCEL_MAX_ROWS:
        keys = list(cols)
        with open(out_dir / "per_iteration.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(keys)
            for row in zip(*(cols[k] for k in keys)):
                w.writerow([repr(float(v)) for v in row])

    sm = summarize(cols)
    sm["run"] = {"scenario": name, "maxiter": maxiter, "max_particles": max_particles,
                 "seed": seed, "delta_stride": delta_stride, "eps_min": diag.eps_min,
                 "wall_s": wall, "wall_ref_s": wall_ref, "hit_particle_limit": hit_limit,
                 "identical_to_undiagnosed": identical,
                 "event_code": EVENT_CODE,
                 "theta_estimator": THETA_ESTIMATOR, "diag_version": DIAG_VERSION}
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(sm, fh, indent=2)
    if pop is not None:
        with open(out_dir / "pop.json", "w", encoding="utf-8") as fh:
            json.dump(pop, fh, indent=2)

    if not quiet:
        print_summary(sm)
    print(f"Geschrieben: {out_dir}", flush=True)
    return out_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scenario", default=prod.NAME,
                    choices=[prod.NAME, prod.NAME_BEST_1070] + sorted(k for k, v in SCENARIOS.items() if v.granulation))
    ap.add_argument("--maxiter", type=int, default=None,
                    help="Event-Budget (Default: das des Szenarios)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--delta-stride", type=int, default=1,
                    help="Kennzahl 2 nur in jeder k-ten Iteration messen (kostet 2 Rebuilds)")
    ap.add_argument("--check-identical", action="store_true",
                    help="Zweiter Lauf ohne Diagnose; Endzustand muss bitgleich sein")
    args = ap.parse_args(argv)

    name, make, maxiter, max_particles, seed = _run_spec(args)
    pop = {prod.NAME: prod.POP, prod.NAME_BEST_1070: prod.POP_BEST_1070}.get(args.scenario)
    run_one(name, make, maxiter, max_particles, seed, delta_stride=args.delta_stride,
            check_identical=args.check_identical, pop=pop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
