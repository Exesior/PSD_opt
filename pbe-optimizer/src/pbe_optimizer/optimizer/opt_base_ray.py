# -*- coding: utf-8 -*-
"""
Optimization algorithm based on ray-Tune
"""
# import uuid
import os
import sqlite3
from datetime import datetime
from filelock import FileLock
import json
import numpy as np
# import math
# import ray
from ray import tune
# from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
# from ray.util.placement_group import placement_group, placement_group_table
import ray.util.multiprocessing as mp
from ray.tune.search.optuna import OptunaSearch
from optuna.samplers import GPSampler,CmaEsSampler,TPESampler,NSGAIIISampler,QMCSampler
from ray.tune.search import ConcurrencyLimiter
from .opt_core_ray import OptCoreRay
from .opt_core_multi_ray import OptCoreMultiRay

from ray._private import state


# Extra per-candidate columns of the warm-start SQLite (besides id, data_name,
# param_json, score). Filled from the trial result where the adapter provides the
# values (wet_granulation adapter via OptCoreRay.step), NULL otherwise.
_EXTRA_COLUMNS = (
    ("status", "TEXT"),            # ok | partial | timeout_partial | timeout | failed | setup_error
    ("n_rep_ok", "INTEGER"),       # evaluated MC replicates
    ("n_rep_total", "INTEGER"),    # planned MC replicates
    ("eval_seconds", "REAL"),      # wall time of the candidate
    ("failure_reason", "TEXT"),    # per-replicate reasons, e.g. "seed=..: reached MAX_PARTICLES"
    ("worker_pid", "INTEGER"),     # Ray worker process that evaluated the candidate
    ("reuse_num", "INTEGER"),      # actor reuse counter at that candidate
    ("finished_at", "TEXT"),       # local time the candidate finished
)


def _extras_from_result(result):
    """Pick the values of the extra SQLite columns out of a Ray Tune result dict."""
    result = result or {}
    finished = result.get("finished_at", result.get("timestamp"))
    if finished is not None:
        finished = datetime.fromtimestamp(float(finished)).isoformat(sep=" ", timespec="milliseconds")
    return {
        "status": result.get("status"),
        "n_rep_ok": result.get("n_rep_ok"),
        "n_rep_total": result.get("n_rep_total"),
        "eval_seconds": result.get("eval_seconds"),
        "failure_reason": result.get("failure_reason"),
        "worker_pid": result.get("pid"),
        "reuse_num": result.get("reuse_num"),
        "finished_at": finished,
    }


def _sql_value(value):
    """numpy scalars -> plain Python values (sqlite3 cannot bind numpy types)."""
    return value.item() if hasattr(value, "item") else value


class _ProgressReporter(tune.Callback):
    """Print progress and optionally persist completed trials in batches."""

    def __init__(self, total, checkpoint_interval=0, checkpoint=None):
        self._total = int(total)
        self._done = 0
        self._best = float("inf")
        self._t_sum = 0.0
        self._checkpoint_interval = int(checkpoint_interval)
        if self._checkpoint_interval < 0:
            raise ValueError("checkpoint_interval must be non-negative.")
        self._checkpoint = checkpoint
        self._params = []
        self._scores = []
        self._extras = []

    def on_trial_result(self, iteration, trials, trial, result, **info):
        self._done += 1
        loss = result.get("loss", float("nan"))
        dt = float(result.get("time_this_iter_s", float("nan")))
        if loss == loss and loss < self._best:      # NaN-safe
            self._best = loss
        if loss == loss:
            self._params.append(dict(trial.config))
            self._scores.append(float(loss))
            self._extras.append(_extras_from_result(result))
        if dt == dt:
            self._t_sum += dt
        avg = self._t_sum / max(self._done, 1)
        status = result.get("status")
        tag = (f"   [{status} {result.get('n_rep_ok')}/{result.get('n_rep_total')}]"
               if status is not None else "")
        print(f"[fit {self._done}/{self._total}]  MSE = {loss:.4e}   "
              f"best = {self._best:.4e}   ({dt:.1f} s, Ø {avg:.1f} s){tag}", flush=True)
        if (self._checkpoint is not None and self._checkpoint_interval
                and self._done % self._checkpoint_interval == 0):
            self._checkpoint(self._done, self._params, self._scores, self._extras)

    def on_trial_error(self, iteration, trials, trial, **info):
        self._done += 1
        print(f"[fit {self._done}/{self._total}]  TRIAL ERROR  ({getattr(trial, 'trial_id', '?')})",
              flush=True)


class OptBaseRay():
    def __init__(self, base):
        self.base = base
        
    def _filter_points_to_evaluate(self, points: list[dict], rt_space: dict) -> list[dict]:
        """Keep only keys that correspond to search distributions (tune.*),
        drop fixed constants and any extra metadata keys not in Optuna space.
        """
        from ray.tune.search.sample import Domain
    
        # Keys that OptunaSearch will consider "searchable"
        searchable = {k for k, v in rt_space.items() if isinstance(v, Domain)}
    
        filtered = []
        for p in points:
            q = {k: p[k] for k in searchable if k in p}
            filtered.append(q)
        return filtered
    
    def _select_top_k_params(
        self,
        params_full: list[dict],
        scores_full: list[float],
        k: int,
    ) -> list[dict]:
        """Select top-k configs (lowest score) from history, keep original full configs."""
        if k <= 0 or not params_full or not scores_full:
            return []
        k = min(int(k), len(scores_full))
        idx = np.argsort(np.asarray(scores_full, dtype=float))[:k]
        return [params_full[int(i)] for i in idx]
    
    def print_current_actors(self):
        """
        Print the current number of active Ray actors.
    
        Returns
        -------
        None
        """
        actors = state.actors()
        print(f"Number of actors: {len(actors)}")
        for actor_id, actor_info in actors.items():
            print(f"Actor ID: {actor_id}, State: {actor_info['State']}")
    
    
    def multi_optimierer_ray(self, opt_params_space, exp_data_paths=None, known_params=None):
        """
        Optimize PBE parameters using multiple Ray Tuners, managed via Ray multiprocessing.
    
        This method enables multiple instances of Ray Tune to run concurrently, each tuning 
        with different experimental data sets or parameters. It splits the job queue based on 
        the number of available jobs and distributes the tasks across multiple processes 
        managed by Ray's multiprocessing pool.
    
        Parameters
        ----------
        opt_params_space : dict
            A dictionary of optimization parameters, where each key corresponds to a parameter name 
            and its value contains information about the bounds and scaling (logarithmic or linear).
        exp_data_paths : list of str, optional
            Paths to the experimental data for each tuning instance.
        known_params : list of dict, optional
            Known parameters to be passed to the optimization process.
        
        Returns
        -------
        list
            A list of dictionaries, where each dictionary contains the optimized parameters and 
            the corresponding objective score (delta).
        """
        base = self.base
        # --- build the Ray Tune search space once ---
        base.RT_space = {}
        for name, info in opt_params_space.items():
            # 1) fixed
            if "fixed" in info:
                base.RT_space[name] = info["fixed"]
                continue
        
            # 2) discrete choices: e.g. integers in [2,3,4,5,6,7,8]
            if "choices" in info:
                choices = info["choices"]
                if not isinstance(choices, (list, tuple)) or len(choices) == 0:
                    raise ValueError(f"{name}: 'choices' must be a non-empty list/tuple.")
                base.RT_space[name] = tune.choice(list(choices))
                continue
        
            # 3) integer range: all integers in [lo, hi]
            if "int_bounds" in info:
                lo, hi = info["int_bounds"]
                lo = int(lo); hi = int(hi)
                if hi < lo:
                    raise ValueError(f"{name}: int_bounds hi < lo.")
                # tune.randint upper is exclusive, so +1
                base.RT_space[name] = tune.randint(lo, hi + 1)
                continue
        
            # 4) existing float bounds
            lo, hi = info["bounds"]
            if info.get("log_scale", False):
                base.RT_space[name] = tune.loguniform(10**lo, 10**hi)
            else:
                base.RT_space[name] = tune.uniform(lo, hi)
    
        # Build the job queue
        job_queue = list(zip(exp_data_paths or [], known_params or []))
        queue_length = len(job_queue)
    
        # Ensure num_jobs does not exceed queue length
        self.check_num_jobs(queue_length)
        num_workers = base.core.num_jobs
    
        # The worker will pull its slice of the queue, look up warm-start params,
        # and pass them into optimierer_ray
        def worker(job_queue_slice):
            job_results = []
            for paths, params in job_queue_slice:
                # call optimierer_ray with warm start
                result = self.optimierer_ray(
                    exp_data_paths=paths,
                    known_params=params,
                )
                job_results.append(result)
            return job_results
    
        # split into roughly equal slices
        job_slices = [job_queue[i::num_workers] for i in range(num_workers)]
    
        # dispatch with multiprocessing
        results = []
        with mp.Pool(num_workers) as pool:
            batches = pool.map(worker, job_slices)
        for batch in batches:
            results.extend(batch)
    
        return results
    
    def check_num_jobs(self, queue_length):
        """
        Adjust the number of concurrent Tuners to the available job queue length.
    
        Parameters
        ----------
        queue_length : int
            The length of the job queue, representing the number of data sets available for tuning.
    
        Returns
        -------
        None
        """
        base = self.base
        if queue_length < base.core.num_jobs:
            base.core.num_jobs = queue_length
    
    def _save_warm_params(self, db_path, data_name, all_params, all_score, all_extras=None):
        assert len(all_params) == len(all_score)
        if all_extras is None:
            all_extras = [{}] * len(all_params)
        assert len(all_extras) == len(all_params)
        extra_names = [name for name, _ in _EXTRA_COLUMNS]
        lock_path = db_path + ".lock"

        with FileLock(lock_path):  # Ensure concurrent safety
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Create a table（if not exist）
            extra_ddl = "".join(f",\n                {name} {sqltype}" for name, sqltype in _EXTRA_COLUMNS)
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS warm_params (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_name TEXT,
                param_json TEXT,
                score REAL{extra_ddl}
            )
            """)
            # Files written before the extra columns existed: add them.
            have = {row[1] for row in cursor.execute("PRAGMA table_info(warm_params)")}
            for name, sqltype in _EXTRA_COLUMNS:
                if name not in have:
                    cursor.execute(f"ALTER TABLE warm_params ADD COLUMN {name} {sqltype}")

            # delete the old data_name (if exist)
            cursor.execute("DELETE FROM warm_params WHERE data_name = ?", (data_name,))
            # Rewriting the same <N>.sqlite (checkpoint + final save) would otherwise
            # continue the AUTOINCREMENT ids (N+1 .. 2N). Empty table -> ids restart at 1.
            if cursor.execute("SELECT COUNT(*) FROM warm_params").fetchone()[0] == 0:
                cursor.execute("DELETE FROM sqlite_sequence WHERE name = 'warm_params'")

            # insert new values
            cols = ["data_name", "param_json", "score"] + extra_names
            records = [(data_name, json.dumps(p), s,
                        *[_sql_value((e or {}).get(name)) for name in extra_names])
                       for p, s, e in zip(all_params, all_score, all_extras)]
            cursor.executemany(f"INSERT INTO warm_params ({', '.join(cols)}) "
                               f"VALUES ({', '.join('?' * len(cols))})", records)

            conn.commit()
            conn.close()


    def _load_warm_params(self, db_path, data_name):
        if not os.path.exists(db_path):
            return [], []

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT param_json, score FROM warm_params WHERE data_name = ? ORDER BY id", (data_name,))
        rows = cursor.fetchall()
        conn.close()

        all_params = [json.loads(p) for p, _ in rows]
        all_score = [s for _, s in rows]
        return all_params, all_score

    def _load_warm_extras(self, db_path, data_name):
        """Extra columns of a warm-start file, row-aligned with `_load_warm_params`.

        Older files without these columns give dicts of None.
        """
        if not os.path.exists(db_path):
            return []
        names = [name for name, _ in _EXTRA_COLUMNS]
        conn = sqlite3.connect(db_path)
        try:
            have = {row[1] for row in conn.execute("PRAGMA table_info(warm_params)")}
            select = ", ".join(name if name in have else "NULL" for name in names)
            rows = conn.execute(f"SELECT {select} FROM warm_params WHERE data_name = ? ORDER BY id",
                                (data_name,)).fetchall()
        finally:
            conn.close()
        return [dict(zip(names, row)) for row in rows]
    
    
    def optimierer_ray(self, opt_params_space=None, exp_data_paths=None,known_params=None, evaluated_params=None):
        """
        Optimize PBE parameters using Ray's Tune module, based on the delta calculated by the method `calc_delta_agg`.
    
        This method utilizes Ray Tune for hyperparameter optimization. It sets up the search 
        space and runs the optimization process, either for single or multi-dimensional 
        PBEs. The results are saved and returned in a dictionary 
        that contains the optimized parameters and the corresponding objective score.
    
        Parameters
        ----------
        opt_params_space : dict, optional
            A dictionary of optimization parameters, where each key corresponds to a parameter name 
            and its value contains information about the bounds and scaling (logarithmic or linear).
        exp_data_paths : list of str, optional
            Paths to the experimental or synthetic data to be used for optimization. For multi-case, 
            this should be a list containing the paths for 1D and 2D data.
        known_params : dict, optional
            Known parameters to be passed to the optimization process.
    
        Returns
        -------
        dict
            A dictionary containing:
                - "opt_score": The optimized objective score (delta value).
                - "opt_params_space": The optimized parameters from the search space.
                - "file_path": The path(s) to the experimental data used for optimization.
        """
        base = self.base
        # evaluated_params = getattr(self.core, 'evaluated_params', None)
        # evaluated_rewards = getattr(self.core, 'evaluated_rewards', None)
        evaluated_params_for_algo = None
        evaluated_rewards_for_algo = None
        
        # Prepare experimental data (either for 1D or 2D)
        if isinstance(exp_data_paths, list):
            # When set to multi, the exp_data_paths entered here is a list containing one 2d data name and two 1d data names.
            # x_uni_exp = []
            # data_exp = []
            # for exp_data_paths_tem in exp_data_paths:
            #     x_uni_exp_tem, data_exp_tem = base.core.p.get_all_data(exp_data_paths_tem)
            #     x_uni_exp.append(x_uni_exp_tem)
            #     data_exp.append(data_exp_tem)
            data_name = getattr(base.core, 'data_name_tune', os.path.basename(exp_data_paths[0]))
        else:
            # When not set to multi or optimization of 1d-data, the exp_data_paths contain the name of that data.
            # x_uni_exp, data_exp = base.core.p.get_all_data(exp_data_paths)
            data_name = os.path.basename(exp_data_paths)

        # Keep the SQLite lookup key identical to the key used when saving.
        if data_name.startswith("Sim_"):
            data_name = data_name[len("Sim_"):]
        if data_name.endswith(".xlsx"):
            data_name = data_name[:-len(".xlsx")]
            
        # Reuse the previous parameters as warm-up for new optimization
        resume_unfinished = getattr(base.core, 'resume_unfinished', False)
        result_dir = getattr(base.core, 'result_dir', base.core.tune_storage_path)
        evaluated_params_full = []
        evaluated_rewards_full = []
        evaluated_extras_full = []

        n_prev = int(getattr(base.core, 'n_iter_prev', 0))
        if resume_unfinished:
            warm_params_path = os.path.join(result_dir, f"{n_prev}.sqlite")
            evaluated_params_full, evaluated_rewards_full = self._load_warm_params(warm_params_path, data_name)
            evaluated_extras_full = self._load_warm_extras(warm_params_path, data_name)
            if len(evaluated_extras_full) != len(evaluated_params_full):
                evaluated_extras_full = [{}] * len(evaluated_params_full)
            # evaluated_rewards = None
                
        # Set up the Ray Tune search space    
        if opt_params_space is not None:   
            base.RT_space = {}
            for name, info in opt_params_space.items():
                # 1) fixed
                if "fixed" in info:
                    base.RT_space[name] = info["fixed"]
                    continue
            
                # 2) discrete choices: e.g. integers in [2,3,4,5,6,7,8]
                if "choices" in info:
                    choices = info["choices"]
                    if not isinstance(choices, (list, tuple)) or len(choices) == 0:
                        raise ValueError(f"{name}: 'choices' must be a non-empty list/tuple.")
                    base.RT_space[name] = tune.choice(list(choices))
                    continue
            
                # 3) integer range: all integers in [lo, hi]
                if "int_bounds" in info:
                    lo, hi = info["int_bounds"]
                    lo = int(lo); hi = int(hi)
                    if hi < lo:
                        raise ValueError(f"{name}: int_bounds hi < lo.")
                    # tune.randint upper is exclusive, so +1
                    base.RT_space[name] = tune.randint(lo, hi + 1)
                    continue
            
                # 4) existing float bounds
                lo, hi = info["bounds"]
                if info.get("log_scale", False):
                    base.RT_space[name] = tune.loguniform(10**lo, 10**hi)
                else:
                    base.RT_space[name] = tune.uniform(lo, hi)
        base.RT_space["__exp_paths"] = exp_data_paths
        base.RT_space["__known_params"] = known_params
                
        opt_top_K = int(getattr(base.core, "opt_top_K", 0))
        if resume_unfinished and evaluated_params_full:
            # Default behavior: warm start with history (points + rewards)
            evaluated_params_for_algo = self._filter_points_to_evaluate(evaluated_params_full, base.RT_space)
            evaluated_rewards_for_algo = evaluated_rewards_full
        
            # If opt_top_K > 0: re-evaluate the historical best K points
            # IMPORTANT: pass only evaluated_params (points_to_evaluate), keep evaluated_rewards as None
            if opt_top_K > 0:
                topk_full = self._select_top_k_params(evaluated_params_full, evaluated_rewards_full, opt_top_K)
                topk_filtered = self._filter_points_to_evaluate(topk_full, base.RT_space)
        
                # Override: only force these points to be evaluated; do NOT pass evaluated_rewards
                evaluated_params_for_algo = topk_filtered
                evaluated_rewards_for_algo = None
                
        algo = self.create_algo(
            evaluated_params=evaluated_params_for_algo,
            evaluated_rewards=evaluated_rewards_for_algo,
        )
        # Define the directory name creator for each trial
        def trial_dirname_creator(trial):
            return f"trial_{trial.trial_id}"

        checkpoint_interval = int(getattr(base.core, "checkpoint_interval", 0))

        def save_progress(done, current_params, current_scores, current_extras):
            n_checkpoint = n_prev + int(done)
            checkpoint_path = os.path.join(result_dir, f"{n_checkpoint}.sqlite")
            self._save_warm_params(
                checkpoint_path,
                data_name,
                list(evaluated_params_full) + list(current_params),
                list(evaluated_rewards_full) + list(current_scores),
                list(evaluated_extras_full) + list(current_extras),
            )
            print(f"[checkpoint] {n_checkpoint} completed trials -> {checkpoint_path}", flush=True)
        # Set up the trainable function based on the multi_flag
        # if not base.multi_flag:
        #     trainable = tune.with_parameters(OptCoreRay, core_params=base.core_params, pop_params=base.pop_params,
        #                                      data_path=base.data_path, exp_data_paths=exp_data_paths,
        #                                      x_uni_exp=x_uni_exp, data_exp=data_exp, known_params=known_params, 
        #                                      exp_case=base.core.exp_data)
        # else:
        #     trainable = tune.with_parameters(OptCoreMultiRay, core_params=base.core_params, pop_params=base.pop_params,
        #                                      data_path=base.data_path, exp_data_paths=exp_data_paths,
        #                                      x_uni_exp=x_uni_exp, data_exp=data_exp, known_params=known_params,
        #                                      exp_case=base.core.exp_data)    
        trainable = tune.with_parameters(
            OptCoreRay if not base.multi_flag else OptCoreMultiRay,
            core_params=base.core_params,
            pop_params=base.pop_params,
            data_path=base.data_path,
        ) 
        # Define the resources used for each trial using PlacementGroupFactory
        # trainable_with_resources  = tune.with_resources(trainable, 
        #                                               resources=tune.PlacementGroupFactory([{"CPU": base.core.cpus_per_trail}]),
        # )
        trainable_with_resources  = tune.with_resources(trainable,                             
            {"cpu": base.core.cpus_per_trail}, 
        )
        
        # checkpoint_path_save = os.path.join(base.core.tune_storage_path, f"{data_name}_checkpoint_{n_save}.pkl")
        # if resume_unfinished:
        #     # checkpoint_path_re = os.path.join(base.core.tune_storage_path, f"{data_name}_checkpoint_{n_prev}.pkl")
        #     # Resume tuning
        #     # algo.restore_from_dir(
        #     #     os.path.join(base.core.tune_storage_path, data_name))
        #     # algo.restore(checkpoint_path_re)
        #     # Set up a new Ray Tune Tuner
        #     tuner = tune.Tuner(
        #         trainable_with_resources,
        #         param_space=base.RT_space,
        #         tune_config=tune.TuneConfig(
        #             num_samples=base.core.n_iter,
        #             search_alg=algo,
        #             reuse_actors=True,
        #             trial_dirname_creator=trial_dirname_creator,
        #         ),
        #         run_config=tune.RunConfig(
        #         storage_path =base.core.tune_storage_path,
        #         name = data_name,
        #         verbose = base.core.verbose, # verbose=0: no trial info, 1: basic info, 2: detailed info
        #         stop={"training_iteration": 1},
        #         )
        #     )
        # else:
        # Set up a new Ray Tune Tuner
        tuner = tune.Tuner(
            trainable_with_resources,
            param_space=base.RT_space,
            tune_config=tune.TuneConfig(
                num_samples=base.core.n_iter,
                search_alg=algo,
                reuse_actors=True,
                trial_dirname_creator=trial_dirname_creator,
            ),
            run_config=tune.RunConfig(
            storage_path =base.core.tune_storage_path,
            name = data_name,
            verbose = base.core.verbose, # verbose=0: no trial info, 1: basic info, 2: detailed info
            stop={"training_iteration": 1},
            callbacks=[_ProgressReporter(
                int(base.core.n_iter) + n_prev,
                checkpoint_interval=checkpoint_interval,
                checkpoint=save_progress if checkpoint_interval else None,
            )],
            )
        )
        # Run the optimization process
        results = tuner.fit()
        # algo.save(checkpoint_path_save)
        
        # df = results.get_dataframe()
        # df_path = os.path.join(r"C:\Users\px2030\Code\PSD_opt\optframework\utils\general_scripts\Parameter_study", data_name+".csv")
        # df.to_csv(df_path, index=False)
        
        # Collect *current* trials
        curr_params = []
        curr_scores = []
        curr_extras = []
        for trial in results:
            config = trial.config
            score = trial.metrics.get("loss", None)
            if score is not None:
                curr_params.append(config)
                curr_scores.append(float(score))
                curr_extras.append(_extras_from_result(trial.metrics))
        # Completion order, as in the checkpoint files -> `id` means the same in every file.
        order = sorted(range(len(curr_params)),
                       key=lambda i: (curr_extras[i]["finished_at"] is None, curr_extras[i]["finished_at"] or ""))
        curr_params = [curr_params[i] for i in order]
        curr_scores = [curr_scores[i] for i in order]
        curr_extras = [curr_extras[i] for i in order]

        # Merge: history (full) + current
        # NOTE: this guarantees every sqlite contains the full history up to that iteration.
        merged_params = list(evaluated_params_full) + list(curr_params)
        merged_scores = list(evaluated_rewards_full) + list(curr_scores)
        merged_extras = list(evaluated_extras_full) + list(curr_extras)

        n_save = int(base.core.n_iter) + int(getattr(base.core, "n_iter_prev", 0))
        warm_params_path = os.path.join(result_dir, f"{n_save}.sqlite")
        self._save_warm_params(warm_params_path, data_name, merged_params, merged_scores, merged_extras)
        # Get the best result from the optimization
        opt_result = results.get_best_result(metric="loss", mode="min")
        opt_exp_data_paths = data_name
        new_params = opt_result.config
        new_score = opt_result.metrics["loss"]
        # Historical best (use FULL history configs)
        if evaluated_rewards_full:
            hist_best_idx = int(np.argmin(np.asarray(evaluated_rewards_full, dtype=float)))
            hist_score = float(evaluated_rewards_full[hist_best_idx])
            hist_params = evaluated_params_full[hist_best_idx]
        else:
            hist_score = None
            hist_params = None
        
        # Final decision
        if hist_score is not None and hist_score < new_score and opt_top_K == 0:
            opt_score = hist_score
            opt_params = hist_params
        else:
            opt_score = new_score
            opt_params = new_params
        
        result_dict = {
            "opt_score": opt_score,
            "opt_params": opt_params,
            "file_path": opt_exp_data_paths,
        }
        return result_dict
        
    def create_algo(self, batch=False, evaluated_params=None, evaluated_rewards=None):
        """
        Create and return the search algorithm to be used for hyperparameter optimization.

        This method creates a search algorithm based on the `method` attribute of the core object. 
        It supports a variety of search algorithms from the Optuna library, including Bayesian 
        optimization (`GP`), tree-structured Parzen estimators (`TPE`), covariance matrix adaptation 
        evolution strategy (`Cmaes`), NSGA-II (`NSGA`), and quasi-Monte Carlo sampling (`QMC`). 
        Optionally, it can also limit the number of concurrent trials.
        
        The number of concurrent trials controls the parallelism of the optimization process. In theory, 
        increasing the number of concurrent trials speeds up the calculation, but it may reduce the 
        convergence rate due to less frequent information sharing between trials. Empirically, a range of 
        2-12 concurrent trials tends to work well. 
    
        The `batch` parameter controls whether a new batch of trials is submitted only after the current 
        batch finishes all trials. Note that for some algorithms, the batch setting is fixed.
    
        Parameters
        ----------
        batch : bool, optional
            Whether to use batch mode for the concurrency limiter. Default is False.
    
        Returns
        -------
        search_alg : object
            The search algorithm instance to be used for optimization.
    
        """
        base = self.base
        if base.core.method == 'GP': 
            try:
                import torch  # just to check if PyTorch is installed
            except ImportError:
                raise ImportError(
                "PyTorch is required for GPSampler but not installed. "
            )
            search_alg = OptunaSearch(metric="loss", mode="min", sampler=GPSampler(seed=base.core.random_seed), 
                                      points_to_evaluate=evaluated_params, evaluated_rewards=evaluated_rewards)
        elif base.core.method == 'TPE': 
            search_alg = OptunaSearch(metric="loss", mode="min", sampler=TPESampler(seed=base.core.random_seed), 
                                      points_to_evaluate=evaluated_params, evaluated_rewards=evaluated_rewards)
        elif base.core.method == 'Cmaes':    
            search_alg = OptunaSearch(metric="loss", mode="min", sampler=CmaEsSampler(seed=base.core.random_seed), 
                                      points_to_evaluate=evaluated_params, evaluated_rewards=evaluated_rewards)
        elif base.core.method == 'NSGA':    
            search_alg = OptunaSearch(metric="loss", mode="min", sampler=NSGAIIISampler(seed=base.core.random_seed), 
                                      points_to_evaluate=evaluated_params, evaluated_rewards=evaluated_rewards)
        elif base.core.method == 'QMC':    
            search_alg = OptunaSearch(metric="loss", mode="min", sampler=QMCSampler(scramble=True, seed=base.core.random_seed), 
                                      points_to_evaluate=evaluated_params, evaluated_rewards=evaluated_rewards)
        else:
            raise ValueError(f"Unsupported sampler detected: {base.core.method}")
        # If no concurrency limit is set, return the search algorithm directly    
        if base.core.max_concurrent is None:
            return search_alg
        else:
            # Limit the number of concurrent trials using ConcurrencyLimiter
            return ConcurrencyLimiter(search_alg, max_concurrent=base.core.max_concurrent, batch=batch)
