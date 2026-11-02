"""Adapter between the optimizer and the 1-D WMCPBE wet-granulation model."""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
import warnings

import numpy as np
import pandas as pd

from wmcpbe import MCPBESolver

from .adapters_api_basics import WriteThroughAdapter


class WetGranulationAdapter(WriteThroughAdapter):
    """Run a fresh wet-granulation simulation for every optimizer candidate.

    Two cost functions, selected by the pop_params key ``COST_MODE``
    (missing key -> ``"xq_mse"``):

        "xq_mse"        x(Q, t) at fixed Q3 quantiles, sheet ``x_Q3``, MSE in m^2
                        via ``opt.cost_fun`` (Betreuer-Konvention).
        "q3_sieve_mae"  Q3 at the sieve meshes, sheet ``Q3_sieve`` (one row per
                        measurement, several test series), weighted mean
                        absolute error, dimensionless (Eric, 2026-10-02).
    """

    # Score of a candidate whose time limit ran out before ANY replicate finished.
    # Deliberately different from the generic failure score 10 (hard-coded in
    # OptCore.calc_delta), so both cases stay distinguishable in the SQLite.
    TIMEOUT_SCORE = 10.1
    COST_MODES = ("xq_mse", "q3_sieve_mae")

    def __init__(self, *, opt, role: Literal["main"] = "main", **kw: Any):
        dim = int(kw.pop("dim", 1))
        kw.pop("init", None)
        if dim != 1 or role != "main":
            raise ValueError("WetGranulationAdapter supports only role='main', dim=1.")

        # WriteThroughAdapter needs an object before the experimental Q3 is
        # available.  MCPBESolver requires an aggregation kernel even when
        # ``init=False``; this zero-rate placeholder is replaced in solve().
        super().__init__(MCPBESolver(
            dim=1,
            init=False,
            agg_kernel_name="constant",
            agg_kernel_params={"corr_beta": 0.0},
            **kw,
        ))
        self._skip.update({
            "role", "opt", "data_path", "data_mod", "failure_reason",
            "_q_grid", "_data_exp", "_initial_diameters",
            "_mc_seeds", "_n_rep_ok", "_n_rep_total", "_last_eval_seconds",
            "eval_info", "_score_override",
            "_cost_mode", "_mesh_m", "_obs_weight", "_obs_tidx", "_mesh_weight",
        })
        self.role = role
        self.opt = opt
        self.data_path = None
        self.data_mod = None
        self.failure_reason = None
        self._q_grid = None            # quantile grid Q of the x(Q, t) target
        self._data_exp = None
        # COST_MODE as read in get_all_data, plus the "q3_sieve_mae" target layout:
        self._cost_mode = None
        self._mesh_m = None            # sieve meshes [m]
        self._obs_weight = None        # weight of each measurement (sum 1)
        self._obs_tidx = None          # index into t_vec of each measurement
        self._mesh_weight = None       # optional weight per mesh (sheet "Maschengewicht"); None = plain mean
        self._initial_diameters = None
        self.calc_status = True
        # MC-replicate bookkeeping (filled per evaluation in solve()):
        self._mc_seeds = None          # the fixed list of distinct seeds
        self._n_rep_ok = 0
        self._n_rep_total = 0
        self._last_eval_seconds = 0.0
        # Per-candidate diagnostics, read by OptCoreRay.step and stored as extra
        # SQLite columns: status, n_rep_ok, n_rep_total, eval_seconds, failure_reason.
        self.eval_info = {}
        self._score_override = None    # TIMEOUT_SCORE when no replicate finished in time

    def set_comp_para(self, data_path: str) -> None:
        """Record the data root; Excel parsing is deferred until data are requested."""
        self.data_path = Path(data_path)
        self.opt.set_comp_para_flag = True

    def reset_params(self) -> None:
        """No derived parameters exist before a candidate is evaluated."""
        return None

    def calc_init_from_data(self, exp_data_paths, init_flag) -> None:
        """The adapter builds a monodisperse initial population in :meth:`get_all_data`."""
        return None

    def calc_matrix(self, init_N) -> None:
        """WMCPBE is initialized afresh in :meth:`solve`; no matrix is needed."""
        return None

    def get_all_data(self, exp_data_path) -> tuple[np.ndarray, np.ndarray]:
        """Read the experimental target for the selected ``COST_MODE``.

            "xq_mse"        sheet ``x_Q3``    -> :meth:`_read_x_q3`
            "q3_sieve_mae"  sheet ``Q3_sieve`` -> :meth:`_read_q3_sieve`

        Both modes start from a monodisperse population at ``INITIAL_DIAMETER``
        -- it is NOT derived from the experimental curve.
        """
        if isinstance(exp_data_path, (list, tuple)):
            raise ValueError("WetGranulationAdapter supports one experimental workbook only.")
        mode = str(self._value("COST_MODE", "xq_mse"))
        if mode not in self.COST_MODES:
            raise ValueError(f"COST_MODE must be one of {self.COST_MODES}, got {mode!r}.")

        workbook = Path(exp_data_path)
        t_vec = np.asarray(self.opt.t_vec, dtype=float)
        if mode == "q3_sieve_mae":
            x_uni, data_exp = self._read_q3_sieve(workbook, t_vec)
        else:
            x_uni, data_exp = self._read_x_q3(workbook, t_vec)

        n_particles = int(self._value("INITIAL_PARTICLES", 80))
        if n_particles < 2:
            raise ValueError("INITIAL_PARTICLES must be at least 2.")
        d0 = float(self._value("INITIAL_DIAMETER", 34e-6))
        if not (d0 > 0.0) or not np.isfinite(d0):
            raise ValueError("INITIAL_DIAMETER must be a positive, finite length [m].")

        self._initial_diameters = np.full(n_particles, d0, dtype=float)
        self._cost_mode = mode
        self._data_exp = data_exp
        return x_uni, data_exp

    def _read_x_q3(self, workbook: Path, t_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``COST_MODE="xq_mse"``: the target in **x(Q, t)** form from sheet ``x_Q3``.

        Convention (Betreuer): the target is not the CDF value on a diameter
        grid, but the DIAMETER x at a fixed grid of Q3 quantiles -- the
        inverse CDF. Sheet layout
        (see ``studies/optimization/dataprep/prepare_aj30_xq.py``):

            column 0    ``Q``  -- strictly increasing quantile grid on [0, 1]
                        (endpoints 0 and 1 allowed: x(Q=0) = d0,
                        x(Q=1) = coarsest sieve)
            columns 1.. one per snapshot time; header = time in SECONDS,
                        values = diameter x in MILLIMETRES at that quantile,
                        non-decreasing down the Q column.

        Every value in ``opt.t_vec`` must match exactly one column header,
        including ``t_vec[0]`` (initial state; the prep script writes a
        monodisperse t=0 column, x = d0 for every Q).

        ``data_exp`` has shape (n_Q, n_t) and holds x in METRES.
        """
        frame = pd.read_excel(workbook, sheet_name="x_Q3")
        if frame.shape[1] < 2:
            raise ValueError(f"{workbook} has no time columns in sheet 'x_Q3'.")

        q_grid = frame.iloc[:, 0].to_numpy(dtype=float)
        x_all_mm = frame.iloc[:, 1:].to_numpy(dtype=float)
        time_all = np.asarray([float(c) for c in frame.columns[1:]], dtype=float)

        if not np.all(np.isfinite(q_grid)) or not np.all(np.diff(q_grid) > 0.0):
            raise ValueError("x_Q3 quantile grid must be finite and strictly increasing.")
        if not (q_grid[0] >= 0.0 and q_grid[-1] <= 1.0):
            raise ValueError("x_Q3 quantiles must lie within [0, 1].")
        if not np.all(np.isfinite(x_all_mm)) or np.any(x_all_mm <= 0.0):
            raise ValueError("x_Q3 sheet contains non-finite or non-positive diameters.")

        cols = []
        for t in t_vec:
            match = np.flatnonzero(np.isclose(time_all, t, rtol=0.0, atol=1e-6))
            if match.size != 1:
                raise ValueError(
                    f"Expected one x_Q3 column at t={t:g} s; available times are {time_all.tolist()}."
                )
            cols.append(int(match[0]))
        data_exp = x_all_mm[:, cols] * 1e-3  # mm -> m
        if np.any(np.diff(data_exp, axis=0) < -1e-12):
            raise ValueError("x_Q3 columns must be non-decreasing along the quantile grid.")

        self._q_grid = q_grid
        return q_grid, data_exp

    def _read_q3_sieve(self, workbook: Path, t_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``COST_MODE="q3_sieve_mae"``: Q3 at the sieve meshes from sheet ``Q3_sieve``.

        One row per measurement (see ``studies/optimization/dataprep/prepare_aj_q3_sieve.py``):

            t_sim_s     sim time of the measurement [s]; must be in ``opt.t_vec``
                        (several rows may share a time, e.g. replicate experiments)
            Gewicht     weight of the measurement in the cost; all rows sum to 1
            <number>    one column per sieve mesh, header = mesh in MICROMETRES,
                        value = Q3 = mass fraction below that mesh (0..1)
            other text columns (Reihe, Gruppe, Probe, Sieb_g) are information only

        Optional sheet ``Maschengewicht`` (Eric, 2026-10-06): one row with the same
        mesh headers, value = weight of that mesh in the cost (>= 0, sum 1).
        Without it every mesh counts equally and the cost is computed exactly as
        before (plain mean over the meshes, bit for bit).

        ``data_exp`` has shape (n_obs, n_mesh). Returns the meshes in METRES.
        """
        frame = pd.read_excel(workbook, sheet_name="Q3_sieve")
        for col in ("t_sim_s", "Gewicht"):
            if col not in frame.columns:
                raise ValueError(f"{workbook}: sheet 'Q3_sieve' has no column {col!r}.")

        def _mesh(col):
            try:
                return float(col)
            except (TypeError, ValueError):
                return None
        mesh_cols = [c for c in frame.columns if _mesh(c) is not None]
        mesh_um = np.asarray([_mesh(c) for c in mesh_cols], dtype=float)
        if mesh_um.size == 0 or np.any(mesh_um <= 0.0) or np.any(np.diff(mesh_um) <= 0.0):
            raise ValueError("Q3_sieve mesh headers must be positive and strictly increasing [um].")

        q3 = frame[mesh_cols].to_numpy(dtype=float)
        t_obs = frame["t_sim_s"].to_numpy(dtype=float)
        weight = frame["Gewicht"].to_numpy(dtype=float)
        if not (np.all(np.isfinite(q3)) and np.all(np.isfinite(t_obs)) and np.all(np.isfinite(weight))):
            raise ValueError("Q3_sieve contains non-finite values.")
        if np.any(q3 < 0.0) or np.any(q3 > 1.0):
            raise ValueError("Q3_sieve values must lie within [0, 1].")
        if np.any(np.diff(q3, axis=1) < -1e-12):
            raise ValueError("Q3_sieve rows must be non-decreasing along the meshes.")
        if np.any(weight < 0.0) or abs(weight.sum() - 1.0) > 1e-9:
            raise ValueError(f"Q3_sieve weights must be >= 0 and sum to 1 (sum = {weight.sum():.12g}).")

        tidx = []
        for t in t_obs:
            match = np.flatnonzero(np.isclose(t_vec, t, rtol=0.0, atol=1e-6))
            if match.size != 1:
                raise ValueError(f"Measurement at t={t:g} s has no matching t_vec entry {t_vec.tolist()}.")
            tidx.append(int(match[0]))

        self._mesh_weight = None
        if "Maschengewicht" in pd.ExcelFile(workbook).sheet_names:
            mw_frame = pd.read_excel(workbook, sheet_name="Maschengewicht")
            mw_cols = [c for c in mw_frame.columns if _mesh(c) is not None]
            if [_mesh(c) for c in mw_cols] != mesh_um.tolist() or len(mw_frame) != 1:
                raise ValueError("Sheet 'Maschengewicht' needs exactly one row with the mesh "
                                 f"headers of 'Q3_sieve' {mesh_um.tolist()}.")
            mesh_weight = mw_frame[mw_cols].to_numpy(dtype=float)[0]
            if (not np.all(np.isfinite(mesh_weight)) or np.any(mesh_weight < 0.0)
                    or abs(mesh_weight.sum() - 1.0) > 1e-9):
                raise ValueError("Maschengewicht values must be finite, >= 0 and sum to 1 "
                                 f"(sum = {mesh_weight.sum():.12g}).")
            self._mesh_weight = mesh_weight

        self._mesh_m = mesh_um * 1e-6
        self._obs_weight = weight
        self._obs_tidx = np.asarray(tidx, dtype=int)
        return self._mesh_m, q3

    def _value(self, name: str, default: Any) -> Any:
        return getattr(self, name, default)

    def _build_solver(self, t_vec: np.ndarray, seed: int | None = None) -> MCPBESolver:
        """Build a fresh WMCPBE solver for the current candidate.

        ``seed`` overrides ``MC_SEED`` for this replicate; ``None`` -> use MC_SEED.

        Kernel set (mixer-speed-driven wet granulation), mirroring the
        reference scenario ``granulation_rumpf_dynamic_1d`` in
        ``mcpbe/tests/bench/scenarios.py``:

            aggregation ............... kinetic_blend  (blend = AGG_A;
                                        AGG_A = 1 is eke_darelius2005 bit for bit)
            agglomeration acceptance .. stokes_dynamik
            breakage .................. powerlaw_rumpf_dynamic
            porosity growth ........... cone_model
            porosity compression ...... porosity_compression_dynamik
            liquid internalization .... liquid_internalization
            liquid internalization
              during agglomeration .... liq_internalisation_agglomeration
            liquid distribution ....... uniform_weighted

        Every mixer-speed-driven kernel (kinetic_blend, stokes_dynamik,
        powerlaw_rumpf_dynamic, porosity_compression_dynamik) gets the SAME
        ``n_mixer`` -- the solver raises otherwise
        (``assert_consistent_mixer_speed``).

        ``AGG_A`` is passed to kinetic_blend unchanged as ``blend``
        (Eric, 2026-10-06; before: blend = 2 * AGG_A), v ~ m^(-AGG_A/2):
        0 = all particles equally fast, 1 = EKE, 2 = ETM.
        ``EKE_CORR_BETA`` stays its corr_beta.
        """
        v = self._value

        n_mixer = float(v("N_MIXER", 20.0))
        agg_a = float(v("AGG_A", 1.0))
        if not (0.0 <= agg_a <= 2.0):
            raise ValueError(f"AGG_A must lie within [0, 2], got {agg_a}.")
        k_int = float(v("LIQUID_INTERNALIZATION_RATE", 1e11))
        compression_rate = float(v("COMPRESSION_RATE", 0.0134))
        min_porosity = float(v("MIN_POROSITY", 0.2))

        use_liq_agg = bool(v("USE_LIQ_INTERNALISATION_AGGLOMERATION", True))
        use_uniform = bool(v("USE_UNIFORM_WEIGHTED", True))

        seed_val = int(seed) if seed is not None else int(v("MC_SEED", 7))
        solver = MCPBESolver(
            dim=1,
            t_vec=t_vec,
            verbose=False,
            load_attr=False,
            init=False,
            seed=seed_val,
            maybe_double_control_volume=bool(v("MAYBE_DOUBLE_VC", False)),
            recon_enable=False,
            agg_kernel_name="kinetic_blend",
            agg_kernel_params={
                "corr_beta": float(v("EKE_CORR_BETA", 5e-8)),
                "blend": agg_a,
                "n_mixer": n_mixer,
                "c_mixer": float(v("EKE_C_MIXER", 0.0995)),
            },
            agg_acceptance_kernel_name="stokes_dynamik",
            agg_acceptance_kernel_params={
                "U_coll_ref": float(v("STOKES_U_COLL_REF", 0.0794)),
                "n_mixer": n_mixer,
                "c_vel": float(v("STOKES_C_VEL", 0.2852)),
                "binder_viscosity": float(v("STOKES_BINDER_VISCOSITY", 0.1)),
                "rho_solid": float(v("STOKES_RHO_SOLID", 600.0)),
                "rho_liquid": float(v("STOKES_RHO_LIQUID", 1000.0)),
                "h_a": float(v("STOKES_H_A", 500e-9)),
            },
            break_kernel_name="powerlaw_rumpf_dynamic",
            break_kernel_params={
                "p1": float(v("BREAK_P1", 1e13)),
                "p2": float(v("BREAK_P2", 1.0)),
                "n_mixer": n_mixer,
                "c_mixer": float(v("BREAK_C_MIXER", 0.6699)),
                "breakrval": int(v("BREAKRVAL", 4)),
                "k": float(v("RUMPF_K", 2.5)),
                "alpha": float(v("RUMPF_ALPHA", 1.0)),
                "gamma": float(v("SURFACE_TENSION", 0.072)),
                "delta": float(v("CONTACT_ANGLE", 0.0)),
                "x_s": v("BREAK_X_S", None),
                "poro_max": float(v("BREAK_PORO_MAX", 0.9999)),
            },
            porosity_growth_kernel_name="cone_model",
            porosity_growth_kernel_params={
                "k_agg": float(v("CONE_K_AGG", 1.0)),
                "k_break": float(v("CONE_K_BREAK", 1.0)),
                "default_porosity": float(v("CONE_DEFAULT_POROSITY", 0.0)),
            },
            porosity_compression_kernel_name="porosity_compression_dynamik",
            porosity_compression_kernel_params={
                "rate": compression_rate,
                "min_porosity": min_porosity,
                "n_mixer": n_mixer,
                "c_mixer": float(v("COMPRESSION_C_MIXER", 0.6699)),
            },
            liquid_internalization_kernel_name="liquid_internalization",
            liquid_internalization_kernel_params={"k_int": k_int},
            liq_internalisation_agglomeration_kernel_name=(
                "liq_internalisation_agglomeration" if use_liq_agg else None
            ),
            liquid_dist_kernel_name="uniform_weighted" if use_uniform else None,
        )

        solver.process_type = "mix"
        solver.agg_propensity_mode = "moment"   # kinetic_blend has no moment form -> falls back to pairwise
        solver.SIZEEVAL = int(v("SIZEEVAL", 0))
        solver.agg_dW_min = float(v("AGG_DW_MIN", 1.0))
        solver.agg_dW_max = float(v("AGG_DW_MAX", 20.0))
        solver.break_dW_max = float(v("BREAK_DW_MAX", 50.0))
        solver._enable_particle_merging = True
        # "scan" = linearer Vektor-Scan, KEIN Hash-Index. Der Hash-Key haengt an
        # V_dry/porosity/saturation, die die kontinuierlichen Prozesse jeden
        # Event fuer fast die ganze Population neu schreiben -> Hash muesste
        # jeden Event neu aufgebaut werden. Gemessen auf powerlaw_rumpf_dynamic_full:
        # 431 s -> 55 s. (siehe mcpbe_base._MERGER_LOOKUP_MODES)
        solver.merger_lookup = str(v("MERGER_LOOKUP", "scan"))
        solver._merger_use_hash_index = (solver.merger_lookup != "scan")
        solver._fragment_merge_tol = float(v("MERGER_TOLERANCE", 1e-6))

        # Explicit monodisperse start (V_flat / W_init), as the Trials scripts
        # and _initialize_monodisperse() in scenarios.py do.
        porosity = float(v("INITIAL_POROSITY", 0.0))
        diameters = np.asarray(self._initial_diameters, dtype=float)
        v_dry = np.pi / 6.0 * diameters**3
        v_flat = np.vstack((v_dry * (1.0 - porosity), v_dry))
        weights = np.full(diameters.size, float(v("INITIAL_WEIGHT", 600.0)))
        solver.Vc = float(v("CONTROL_VOLUME", 1.0))
        solver._initialize_particles(init_Vc=False, V_flat=v_flat, W_init=weights)
        solver.porosity[:solver.a_tot] = porosity
        solver.liquid_volume[:solver.a_tot] = 0.0
        solver.saturation[:solver.a_tot] = 0.0
        solver._init_lmc()
        solver._initialize_samplers()

        solver.create_nucleation_handler(
            enabled=True,
            volumetric_flow_rate=float(v("VOLUMETRIC_FLOW_RATE", 1.767e-9)),
            droplet_diameter=float(v("DROPLET_DIAMETER", 20e-6)),
            liquid_addition_start=float(v("LIQUID_ADDITION_START", 0.0)),
            liquid_addition_duration=float(v("LIQUID_ADDITION_DURATION", 60.0)),
            batch_size=float(v("NUCLEATION_BATCH_SIZE", 20.0)),
        )
        solver.create_continuous_processes_handler(
            enabled=True,
            k_int=k_int,
            compression_enabled=True,
            compression_rate=compression_rate,
            min_porosity=min_porosity,
        )
        return solver

    @staticmethod
    def _x_at_quantiles_from_snapshot(v_snapshot: np.ndarray, weights: np.ndarray,
                                      q_grid: np.ndarray) -> np.ndarray:
        """Diameter x at each Q3 quantile in ``q_grid`` (the inverse mass CDF).

        Sieving classifies a granule by its outer diameter (from ``V_dry``)
        and weighs its mass (proportional to ``V_solid``). The mass CDF Q3(x)
        is built from the weighted particles and inverted by linear
        interpolation -- ``np.interp(Q, Q3_cumulative, x_sorted)`` -- exactly
        as ``dpbe_post.re_calc_distribution`` derives x_10/x_50/x_90.
        """
        v_snapshot = np.asarray(v_snapshot, dtype=float)
        v_dry = np.asarray(v_snapshot[-1], dtype=float)
        v_solid = np.asarray(v_snapshot[0], dtype=float)
        diameters = np.cbrt(6.0 * v_dry / np.pi)
        weights = np.asarray(weights, dtype=float)
        amount = weights * v_solid  # mass proxy per computational particle
        valid = (np.isfinite(diameters) & np.isfinite(amount)
                 & (diameters > 0.0) & (amount > 0.0))
        if not np.any(valid):
            raise ValueError("Snapshot has no positive, finite weighted particles.")

        diameters = diameters[valid]
        amount = amount[valid]
        order = np.argsort(diameters)
        diameters = diameters[order]
        q3_cum = np.cumsum(amount[order])
        q3_cum /= q3_cum[-1]
        # np.interp clamps to diameters[0] / diameters[-1] outside [q3_cum[0], 1].
        return np.interp(q_grid, q3_cum, diameters)

    @staticmethod
    def _q3_at_meshes_from_snapshot(v_snapshot: np.ndarray, weights: np.ndarray,
                                    mesh_m: np.ndarray) -> np.ndarray:
        """Q3 at each sieve mesh: mass fraction of particles with outer diameter < mesh.

        Same particle convention as :meth:`_x_at_quantiles_from_snapshot`
        (diameter from ``V_dry``, mass proportional to ``W * V_solid``), but no
        interpolation: a sieve passes exactly the particles smaller than its mesh.
        """
        v_snapshot = np.asarray(v_snapshot, dtype=float)
        v_dry = np.asarray(v_snapshot[-1], dtype=float)
        v_solid = np.asarray(v_snapshot[0], dtype=float)
        diameters = np.cbrt(6.0 * v_dry / np.pi)
        amount = np.asarray(weights, dtype=float) * v_solid
        valid = (np.isfinite(diameters) & np.isfinite(amount)
                 & (diameters > 0.0) & (amount > 0.0))
        if not np.any(valid):
            raise ValueError("Snapshot has no positive, finite weighted particles.")

        diameters = diameters[valid]
        amount = amount[valid]
        order = np.argsort(diameters)
        diameters = diameters[order]
        q3_cum = np.cumsum(amount[order])
        q3_cum /= q3_cum[-1]
        n_below = np.searchsorted(diameters, np.asarray(mesh_m, dtype=float), side="left")
        return np.where(n_below > 0, q3_cum[np.maximum(n_below - 1, 0)], 0.0)

    def _resolve_mc_seeds(self) -> list[int]:
        """The FIXED list of ``N_MC_REPLICATES`` distinct seeds for every candidate.

        Deterministically derived from the base ``MC_SEED`` via
        ``numpy.random.SeedSequence`` -- same 10 seeds for every parameter set, so
        candidates are compared on the same set of MC realisations, but the 10 are
        genuinely different (otherwise the repetition is pointless).
        """
        n_rep = max(1, int(self._value("N_MC_REPLICATES", 1)))
        base = int(self._value("MC_SEED", 7))
        if self._mc_seeds is None or len(self._mc_seeds) != n_rep:
            self._mc_seeds = [int(s) for s in
                              np.random.SeedSequence(base).generate_state(n_rep)]
        return self._mc_seeds

    def _log_replicate_failures(self, status: str, n_ok: int, n_rep: int, detail: str | None) -> None:
        """Append one line per affected candidate to a persistent log next to the data.

        Uses a FileLock so concurrent Ray workers cannot interleave lines.
        """
        try:
            root = Path(self.data_path) if self.data_path is not None else Path.cwd()
            log = (root if root.is_dir() else root.parent) / "wet_granulation_replicate_failures.log"
            stamp = datetime.now().isoformat(timespec="seconds")
            line = (f"{stamp}  status={status}  {n_ok}/{n_rep} replicate(s) evaluated"
                    f"  |  {detail or '-'}\n")
            try:
                from filelock import FileLock
                with FileLock(str(log) + ".lock", timeout=10):
                    with open(log, "a", encoding="utf-8") as fh:
                        fh.write(line)
            except Exception:
                with open(log, "a", encoding="utf-8") as fh:   # lock unavailable -> best effort
                    fh.write(line)
        except Exception:  # logging must never break the optimisation
            pass

    def _solve_until(self, deadline: float | None, maxiter: int, max_particles: int) -> bool:
        """Run ``self.impl.solve()`` and stop it at ``deadline`` (``time.perf_counter`` scale).

        Uses the solver's own ``cancel_flag`` hook: ``MCPBESolver.solve`` checks
        ``cancel_flag["cancel"]`` once per event and leaves its loop when it is set.
        A timer thread sets the flag when the deadline is reached.
        Returns True if the timer fired.
        """
        if deadline is None:
            self.impl.solve(maxiter=maxiter, max_particles=max_particles)
            return False
        flag = {"cancel": False}
        self.impl.cancel_flag = flag
        timer = threading.Timer(max(0.0, deadline - time.perf_counter()),
                                flag.__setitem__, args=("cancel", True))
        timer.daemon = True
        timer.start()
        try:
            self.impl.solve(maxiter=maxiter, max_particles=max_particles)
        finally:
            timer.cancel()
        return flag["cancel"]

    def solve(self, t_vec) -> None:
        """Evaluate one candidate: ``N_MC_REPLICATES`` runs, their mean in ``data_mod``.

        ``data_mod`` has the shape of ``data_exp``: x(Q, t) for ``COST_MODE="xq_mse"``,
        Q3 at the sieve meshes for each measurement for ``"q3_sieve_mae"``.

        Time limit: ``max_iter_time`` [s] from the config's ``algo_params``
        (0 = none) applies to the whole CANDIDATE. When it runs out, the running
        replicate is stopped and the remaining ones are not started.

        Outcome in ``eval_info["status"]`` (-> SQLite column ``status``):

            ok               all replicates finished                  score = cost (MSE or MAE, see COST_MODE)
            partial          some replicates failed, mean over rest   score = cost
            timeout_partial  time limit hit, >= 1 replicate finished  score = cost (mean over finished)
            timeout          time limit hit, no replicate finished    score = TIMEOUT_SCORE (10.1)
            failed           every replicate failed (e.g. MAX_PARTICLES)  score = 10
            setup_error      adapter not ready (no simulation run)    score = 10
        """
        t_vec = np.asarray(t_vec, dtype=float)
        self.calc_status = True
        self.data_mod = None
        self.failure_reason = None
        self._score_override = None
        self._n_rep_ok = 0
        self.eval_info = {}
        t0 = time.perf_counter()

        try:
            if self._data_exp is None or self._initial_diameters is None:
                raise RuntimeError("get_all_data() must run before solve().")
            if not np.allclose(t_vec, np.asarray(self.opt.t_vec, dtype=float)):
                raise ValueError("Adapter.solve received a time grid different from opt.t_vec.")
        except Exception as exc:
            n_rep = max(1, int(self._value("N_MC_REPLICATES", 1)))
            self.calc_status = False
            self.failure_reason = str(exc)
            self._finish_eval("setup_error", 0, n_rep, t0)
            warnings.warn(f"Wet-granulation setup failed: {exc}", RuntimeWarning, stacklevel=2)
            return

        seeds = self._resolve_mc_seeds()
        n_rep = len(seeds)
        self._n_rep_total = n_rep
        maxiter = int(self._value("MAXITER", 1_000))
        max_particles = int(self._value("MAX_PARTICLES", 1_000))
        time_limit = float(getattr(self.opt, "max_iter_time", 0.0) or 0.0)
        deadline = t0 + time_limit if time_limit > 0.0 else None

        replicate_mods: list[np.ndarray] = []
        fails: list[tuple[int, int, str]] = []
        timed_out = False

        for i, sd in enumerate(seeds):
            if deadline is not None and time.perf_counter() >= deadline:
                timed_out = True               # remaining replicates are not started
                break
            try:
                self.impl._close(gc_clean=False)
                self.impl = self._build_solver(t_vec, seed=sd)
                cancelled = self._solve_until(deadline, maxiter, max_particles)
                complete = (len(self.impl.V_save) == t_vec.size
                            and len(self.impl.W_save) == t_vec.size)
                # A replicate that reached t_end counts as finished even if the
                # timer fired right after it.
                if cancelled and not complete:
                    timed_out = True
                    fails.append((i, int(sd), f"time limit {time_limit:g} s reached"))
                    break
                if self.impl._hit_particle_limit:
                    raise RuntimeError("reached MAX_PARTICLES")
                if not complete:
                    raise RuntimeError("snapshot count != len(t_vec)")
                if self._cost_mode == "q3_sieve_mae":
                    q3_t = np.vstack([
                        self._q3_at_meshes_from_snapshot(v, w, self._mesh_m)
                        for v, w in zip(self.impl.V_save, self.impl.W_save)
                    ])
                    replicate_mods.append(q3_t[self._obs_tidx])   # one row per measurement
                else:
                    replicate_mods.append(np.column_stack([
                        self._x_at_quantiles_from_snapshot(v, w, self._q_grid)
                        for v, w in zip(self.impl.V_save, self.impl.W_save)
                    ]))
            except Exception as exc:  # this replicate only
                fails.append((i, int(sd), str(exc)))

        n_ok = len(replicate_mods)
        self._n_rep_ok = n_ok
        detail = "; ".join(f"seed={s}: {msg}" for _, s, msg in fails) or None

        if n_ok == 0:
            if timed_out:
                status = "timeout"
                # calc_status stays True, so OptCore.calc_delta asks calc_delta_pop,
                # which returns TIMEOUT_SCORE instead of the hard-coded 10.
                self._score_override = self.TIMEOUT_SCORE
            else:
                status = "failed"
                self.calc_status = False
        else:
            # Average x(Q, t) resp. Q3 over the FINISHED replicates (Eric: variant b,
            # also after a time-out: "vermerken, aber auswerten").
            self.data_mod = np.stack(replicate_mods, axis=0).mean(axis=0)
            status = "timeout_partial" if timed_out else ("partial" if fails else "ok")

        self.failure_reason = detail
        self._finish_eval(status, n_ok, n_rep, t0)
        if status != "ok":
            warnings.warn(f"Wet-granulation: status={status}, {n_ok}/{n_rep} replicates evaluated"
                          f" | {detail}", RuntimeWarning, stacklevel=2)

    def _finish_eval(self, status: str, n_ok: int, n_rep: int, t0: float) -> None:
        """Fill ``eval_info`` and write the failure-log line for a non-``ok`` candidate."""
        self._last_eval_seconds = time.perf_counter() - t0
        self.eval_info = {
            "status": status,
            "n_rep_ok": int(n_ok),
            "n_rep_total": int(n_rep),
            "eval_seconds": float(self._last_eval_seconds),
            "failure_reason": self.failure_reason,
        }
        if status != "ok":
            self._log_replicate_failures(status, n_ok, n_rep, self.failure_reason)

    def calc_delta_pop(self, x_uni_exp, data_exp) -> float:
        # x_uni_exp is the quantile grid resp. the sieve meshes (unused here).
        #   xq_mse        data_exp / data_mod = x(Q, t) in metres -> opt.cost_fun (MSE)
        #   q3_sieve_mae  data_exp / data_mod = Q3 per (measurement, mesh)
        #                 -> J = sum_i w_i * mean_k |Q3_sim - Q3_exp|  (0 = perfect, max 1)
        #                    with sheet "Maschengewicht": sum_k m_k |...| instead of mean_k
        if self._score_override is not None:
            return float(self._score_override)
        if self.data_mod is None or np.shape(data_exp) != np.shape(self.data_mod):
            raise ValueError("Experimental and simulated target arrays must have the same shape.")
        if self._cost_mode == "q3_sieve_mae":
            abs_err = np.abs(np.asarray(self.data_mod) - np.asarray(data_exp))
            if self._mesh_weight is None:
                err = abs_err.mean(axis=1)
            else:
                err = abs_err @ self._mesh_weight
            return float(np.dot(self._obs_weight, err))
        return float(self.opt.cost_fun(data_exp, self.data_mod, self.opt.cost_flag, self.opt.data_flag))

    def close(self) -> None:
        if self.impl is not None:
            self.impl._close()
