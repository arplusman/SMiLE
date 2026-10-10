"""
benchmarks/fairness/process_new.py
====================================
Trains OLD SMiLE and NEW SMiLE on the fairness benchmark and saves
both sets of results to results_new/.

Both models are trained for the same (benchmark, epsilon) configurations
and use the same PRESMLE warm-start strategy. The only difference is the
auxiliary architecture:

  Old SMiLE : two independent h_lower / h_upper networks (core.model_baseline)
  New SMiLE : base + delta  h_base / h_delta  networks   (core.model_new)

Each pkl stores an 'approach' key ('old' or 'new') so postprocess and
analysis scripts can distinguish them from a single results_new/ directory.

Run via:
    python run_benchmark.py --benchmark fairness
"""

import os
import time
import pickle
import shutil
import random
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

# Old SMiLE imports
from core.model_baseline import (
    Net as NetOld,
    BaseNet as BaseNetOld,
    keras as keras_old,
    tf as tf_old,
)

# New SMiLE imports
from core.model_new import (
    Net as NetNew,
    BaseNet as BaseNetNew,
    keras as keras_new,
    np, tf,
)

from core.data import FairnessData

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ---------------------------------------------------------------------------
# Logit helpers  (used for counterfactual variation evaluation)
# ---------------------------------------------------------------------------

class LogitOld(keras_old.Model):
    """Forward pass for old SMiLE using h_lower / h_upper clipping."""
    def __init__(self, f, **kwargs):
        super().__init__(**kwargs)
        self.f = f

    def call(self, inputs):
        return self.f.g(
            tf_old.maximum(
                tf_old.minimum(self.f.h(inputs), self.f.h_upper(inputs)),
                self.f.h_lower(inputs),
            )
        )


class LogitNew(keras_new.Model):
    """Forward pass for new SMiLE using h_base / h_delta clipping."""
    def __init__(self, f, **kwargs):
        super().__init__(**kwargs)
        self.f = f

    def call(self, inputs):
        h_base_out  = self.f.h_base(self.f.flatten(inputs))
        h_upper_out = h_base_out + self.f.h_delta(self.f.flatten(inputs))
        return self.f.g(
            tf.maximum(tf.minimum(self.f.h(inputs), h_upper_out), h_base_out)
        )


# ---------------------------------------------------------------------------
# Hyperparameters  (identical for both approaches — fair comparison)
# ---------------------------------------------------------------------------
name    = "benchmarks/fairness"
res_dir = f"{name}/results_new"

epochs     = 50
batch_size = {"compas": 256, "law": 1024, "crime": 128}
optimizer  = "adam"
benchmarks = ["compas", "law", "crime"]

data_obj = FairnessData()

hidden_dim           = 8
h_architecture       = [(4*hidden_dim, "relu"), (4*hidden_dim, "relu"),
                        (hidden_dim, "linear")]
h_aux_architecture   = [(int(1/2*hidden_dim), "relu"), (hidden_dim, "linear")]
h_base_architecture  = [(int(1/2*hidden_dim), "relu"), (hidden_dim, "linear")]
h_delta_architecture = [(int(1/2*hidden_dim), "relu"), (hidden_dim, "relu")]
g_architecture       = [(1, "linear")]

epsilons = {
    "compas": [0.2, 0.4, 0.6, 0.8, 1.0],
    "crime":  [0.02, 0.04, 0.06, 0.08, 0.10],
    "law":    [1.0, 2.0, 3.0, 4.0, 5.0],
}

generation_frequency  = 1
generation_timelimit  = 2
generation_solvelimit = 8

Path(res_dir).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Shared result builder (avoids repetition between the two approaches)
# ---------------------------------------------------------------------------

def build_result(approach, benchmark, epsilon, f, logit_fn,
                 x_test, x_test_counter, y_test,
                 task, presmle_time, train_time):
    """Pack all metrics and solver logs into a result dict."""
    y_pred = f(x_test).numpy().flatten()
    if task == "classification":
        y_pred = np.round(y_pred)
    return {
        "approach":      approach,
        "benchmark":     benchmark,
        "epsilon":       epsilon,
        "r2":            r2_score(y_pred=y_pred, y_true=y_test)
                         if task == "regression" else None,
        "mse":           mean_squared_error(y_pred=y_pred, y_true=y_test)
                         if task == "regression" else None,
        "accuracy":      accuracy_score(y_pred=y_pred, y_true=y_test)
                         if task == "classification" else None,
        "counter_var":   float(np.max(np.abs(
                             logit_fn(x_test) - logit_fn(x_test_counter)))),
        "training_time": np.round(presmle_time + train_time, 2),
        "weights":       f.get_weights(),
        # Solver logs (used by analyze_degeneracy.py)
        "phase":         list(f.generator.phases),
        "model":         list(f.generator.modelcounts),
        "timelimit":     list(f.generator.timelimits),
        "time":          list(f.generator.times),
        "measured_time": list(f.generator.measured_times),
        "primal":        list(f.generator.primals),
        "dual":          list(f.generator.duals),
        "termination":   list(f.generator.terminations),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

for benchmark in benchmarks:
    (input_dim, binary_split, l, u,
     x_train, y_train, x_test, y_test) = data_obj.load(benchmark=benchmark)

    x_test_counter = x_test.copy()
    x_test_counter.iloc[:, -1] = 1 - x_test_counter.iloc[:, -1]

    task = "regression" if benchmark == "crime" else "classification"
    loss = "mse" if benchmark == "crime" else "binary_crossentropy"

    print("=" * 90)
    print(f"benchmark={benchmark}  task={task}")
    print("=" * 90)

    # ======================================================================
    # OLD SMiLE
    # ======================================================================
    print(f"\n--- Old SMiLE: PRESMLE ({benchmark}) ---")
    old_pre_file = f"{res_dir}/{benchmark}_presmle_old.pkl"

    if os.path.isfile(old_pre_file):
        old_pre       = pickle.load(open(old_pre_file, "rb"))
        preweights_old = old_pre["weights"]
        presmle_time_old = old_pre["training_time"]
        print(f"  [SKIP] loaded {old_pre_file}")
    else:
        h_old       = BaseNetOld(h_architecture)
        h_lower_old = BaseNetOld(h_aux_architecture)
        h_upper_old = BaseNetOld(h_aux_architecture)
        g_old       = BaseNetOld(g_architecture)

        f_pre = NetOld(
            model_path=None, mode=task,
            h=h_old, h_lower=h_lower_old, h_upper=h_upper_old, g=g_old,
            binary_split=None, l=None, u=None,
            d_lower=None, d_upper=None, e_lower=None, e_upper=None,
            pretrain=True,
            generation_timelimit=None, generation_solvelimit=None,
            generation_frequency=None,
        )
        f_pre.compile(optimizer=optimizer, loss=loss)
        t0 = time.time()
        f_pre.fit(x_train, y_train, epochs=epochs,
                  batch_size=batch_size[benchmark])
        presmle_time_old = time.time() - t0
        preweights_old   = f_pre.get_weights()
        pickle.dump({"weights": preweights_old,
                     "training_time": np.round(presmle_time_old, 2)},
                    open(old_pre_file, "wb"))
        print(f"  pretrain done ({presmle_time_old:.1f}s)")

    for epsilon in epsilons[benchmark]:
        delta   = 0.00
        d_lower = [-delta] * input_dim;  d_lower[-1] = -1
        d_upper = [ delta] * input_dim;  d_upper[-1] =  1
        e_lower, e_upper = -epsilon, epsilon
        eps_str = str(epsilon).replace(".", "")
        old_file = f"{res_dir}/{benchmark}_smle_{eps_str}_old.pkl"

        print(f"  Old SMiLE  epsilon={epsilon}", end="  ")
        if os.path.isfile(old_file):
            print("[SKIP]")
            continue

        solver_dir = f"{res_dir}/{benchmark}_smle_{eps_str}_old_solver"
        Path(solver_dir).mkdir(parents=True, exist_ok=True)

        h_old       = BaseNetOld(h_architecture)
        h_lower_old = BaseNetOld(h_aux_architecture)
        h_upper_old = BaseNetOld(h_aux_architecture)
        g_old       = BaseNetOld(g_architecture)

        f_old = NetOld(
            model_path=solver_dir, mode=task,
            h=h_old, h_lower=h_lower_old, h_upper=h_upper_old, g=g_old,
            binary_split=binary_split, l=l, u=u,
            d_lower=d_lower, d_upper=d_upper,
            e_lower=e_lower, e_upper=e_upper,
            pretrain=False,
            generation_timelimit=generation_timelimit,
            generation_solvelimit=generation_solvelimit,
            generation_frequency=generation_frequency,
        )
        f_old.compile(optimizer=optimizer, loss=loss)
        f_old(x_test)
        f_old.set_weights(preweights_old)

        t0 = time.time()
        f_old.fit(x_train, y_train, epochs=epochs,
                  batch_size=batch_size[benchmark])
        f_old.enforce(l=l, u=u, d_lower=d_lower, d_upper=d_upper,
                      e_lower=e_lower, e_upper=e_upper)
        train_time = time.time() - t0

        result = build_result(
            "old", benchmark, epsilon,
            f_old, LogitOld(f_old),
            x_test, x_test_counter, y_test,
            task, presmle_time_old, train_time,
        )
        pickle.dump(result, open(old_file, "wb"))
        shutil.rmtree(solver_dir)
        print(f"r2={result['r2']}  acc={result['accuracy']}  "
              f"cv={result['counter_var']:.4f}")

    # ======================================================================
    # NEW SMiLE
    # ======================================================================
    print(f"\n--- New SMiLE: PRESMLE ({benchmark}) ---")
    new_pre_file = f"{res_dir}/{benchmark}_presmle_new.pkl"

    if os.path.isfile(new_pre_file):
        new_pre        = pickle.load(open(new_pre_file, "rb"))
        preweights_new  = new_pre["weights"]
        presmle_time_new = new_pre["training_time"]
        print(f"  [SKIP] loaded {new_pre_file}")
    else:
        h_new       = BaseNetNew(h_architecture)
        h_base_new  = BaseNetNew(h_base_architecture)
        h_delta_new = BaseNetNew(h_delta_architecture, output_bias=0.001)
        g_new       = BaseNetNew(g_architecture)

        f_pre = NetNew(
            model_path=None, mode=task,
            h=h_new, h_base=h_base_new, h_delta=h_delta_new, g=g_new,
            binary_split=None, l=None, u=None,
            d_lower=None, d_upper=None, e_lower=None, e_upper=None,
            pretrain=True,
            generation_timelimit=None, generation_solvelimit=None,
            generation_frequency=None,
        )
        f_pre.compile(optimizer=optimizer, loss=loss)
        t0 = time.time()
        f_pre.fit(x_train, y_train, epochs=epochs,
                  batch_size=batch_size[benchmark])
        presmle_time_new = time.time() - t0
        preweights_new   = f_pre.get_weights()
        pickle.dump({"weights": preweights_new,
                     "training_time": np.round(presmle_time_new, 2)},
                    open(new_pre_file, "wb"))
        print(f"  pretrain done ({presmle_time_new:.1f}s)")

    for epsilon in epsilons[benchmark]:
        delta   = 0.00
        d_lower = [-delta] * input_dim;  d_lower[-1] = -1
        d_upper = [ delta] * input_dim;  d_upper[-1] =  1
        e_lower, e_upper = -epsilon, epsilon
        eps_str  = str(epsilon).replace(".", "")
        new_file = f"{res_dir}/{benchmark}_smle_{eps_str}_new.pkl"

        print(f"  New SMiLE  epsilon={epsilon}", end="  ")
        if os.path.isfile(new_file):
            print("[SKIP]")
            continue

        solver_dir = f"{res_dir}/{benchmark}_smle_{eps_str}_new_solver"
        Path(solver_dir).mkdir(parents=True, exist_ok=True)

        h_new       = BaseNetNew(h_architecture)
        h_base_new  = BaseNetNew(h_base_architecture)
        h_delta_new = BaseNetNew(h_delta_architecture, output_bias=0.001)
        g_new       = BaseNetNew(g_architecture)

        f_new = NetNew(
            model_path=solver_dir, mode=task,
            h=h_new, h_base=h_base_new, h_delta=h_delta_new, g=g_new,
            binary_split=binary_split, l=l, u=u,
            d_lower=d_lower, d_upper=d_upper,
            e_lower=e_lower, e_upper=e_upper,
            pretrain=False,
            generation_timelimit=generation_timelimit,
            generation_solvelimit=generation_solvelimit,
            generation_frequency=generation_frequency,
        )
        f_new.compile(optimizer=optimizer, loss=loss)
        f_new(x_test)
        f_new.set_weights(preweights_new)

        t0 = time.time()
        f_new.fit(x_train, y_train, epochs=epochs,
                  batch_size=batch_size[benchmark])
        f_new.enforce(l=l, u=u, d_lower=d_lower, d_upper=d_upper,
                      e_lower=e_lower, e_upper=e_upper)
        train_time = time.time() - t0

        result = build_result(
            "new", benchmark, epsilon,
            f_new, LogitNew(f_new),
            x_test, x_test_counter, y_test,
            task, presmle_time_new, train_time,
        )
        pickle.dump(result, open(new_file, "wb"))
        shutil.rmtree(solver_dir)
        print(f"r2={result['r2']}  acc={result['accuracy']}  "
              f"cv={result['counter_var']:.4f}")