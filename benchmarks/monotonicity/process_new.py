"""
benchmarks/monotonicity/process_new.py
=======================================
Trains OLD SMiLE and NEW SMiLE on the monotonicity benchmark and saves
both sets of results to results_new/.

Each result pkl stores an 'approach' key ('old' or 'new') so that
postprocess_new.py and analyze_degeneracy.py can distinguish them
without reading from two separate directories.

Approach summary
----------------
Old SMiLE  : two independent auxiliary networks h_lower / h_upper (core.model_baseline).
             h_lower and h_upper have unconstrained linear outputs — no structural
             ordering guarantee, source of box degeneracy.

New SMiLE  : base + delta parameterisation (core.model_new).
             h_lower(x) = h_base(x)
             h_upper(x) = h_base(x) + h_delta(x),  h_delta >= 0 (ReLU output)
             Ordering h_lower <= h_upper is structurally guaranteed.

Run via:
    python run_benchmark.py --benchmark monotonicity
"""

import os
import time
import pickle
import shutil
import random
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

# Old SMiLE imports
from core.model_baseline import (
    Net as NetOld,
    BaseNet as BaseNetOld,
)

# New SMiLE imports
from core.model_new import (
    Net as NetNew,
    BaseNet as BaseNetNew,
    np, tf,       # numpy / tensorflow re-exported by model_new via  from core.model_new import *
)

from core.data import Synthetic

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ---------------------------------------------------------------------------
# Hyperparameters  (identical for both approaches — fair comparison)
# ---------------------------------------------------------------------------
name    = "benchmarks/monotonicity"
res_dir = f"{name}/results_new"

preepochs  = 1000          # epochs for the pretrain phase
epochs     = 100           # epochs for the enforcement phase
batch_size = 64
optimizer  = "adam"
loss       = "mse"
early_stopping = EarlyStopping(monitor="loss", patience=10,
                                restore_best_weights=True)

n_samples = 1000
input_dim = 1

alphas  = [2.0, 3.0, 4.0]
omegas  = [0.4, 0.6, 0.8]
delta   = 0.0    # input perturbation bound  (d_lower = data.l, d_upper = delta)
epsilon = 0.15   # output deviation bound

hidden_dim     = 8
aux_complexity = 32

# Shared encoder — same for both approaches
h_architecture = [
    (16, "relu"), (32, "relu"), (64, "relu"),
    (32, "relu"), (16, "relu"), (hidden_dim, "linear"),
]
g_architecture = [(1, "linear")]

# Old approach: two independent auxiliary networks (linear output)
h_aux_architecture = [(aux_complexity, "relu"), (hidden_dim, "linear")]

# New approach: base + non-negative delta (ReLU output on delta)
h_base_architecture  = [(aux_complexity, "relu"), (hidden_dim, "linear")]
h_delta_architecture = [(aux_complexity, "relu"), (hidden_dim, "relu")]

lambda_box_pretrain  = 1.5
lambda_box_train     = 1.5
generation_frequency  = 1
generation_timelimit  = 2
generation_solvelimit = 8


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
Path(res_dir).mkdir(parents=True, exist_ok=True)

for alpha in alphas:
    for omega in omegas:

        # Generate data for this (alpha, omega) task
        task  = lambda x, a=alpha, w=omega: x + a * np.sin(w * x)
        l, u  = [-10], [10]
        data  = Synthetic(n_samples=n_samples, input_dim=input_dim,
                          l=l, u=u, func=task)
        x, y, x_train, y_train, x_test, y_test = data.generate(0, standardize=True)

        d_lower = [data.l]   # lower bound on input difference
        d_upper = [delta]    # upper bound on input difference (0 → exact monotonicity)
        e_lower = y.min()    # lower bound on output difference
        e_upper = epsilon    # upper bound on output difference

        tag = f"{str(alpha).replace('.', '')}_{str(omega).replace('.', '')}"
        print("=" * 80)
        print(f"alpha={alpha}  omega={omega}  tag={tag}")
        print("=" * 80)

        # ==================================================================
        # OLD SMiLE
        # ==================================================================
        old_file = f"{res_dir}/{tag}_old.pkl"
        if os.path.isfile(old_file):
            print(f"  [SKIP] old SMiLE already exists: {old_file}")
        else:
            print("  --- Old SMiLE: pretrain ---")
            solver_dir = f"{res_dir}/{tag}_old_solver"
            Path(solver_dir).mkdir(parents=True, exist_ok=True)

            h_old       = BaseNetOld(h_architecture)
            h_lower_old = BaseNetOld(h_aux_architecture)
            h_upper_old = BaseNetOld(h_aux_architecture)
            g_old       = BaseNetOld(g_architecture)

            t0 = time.time()

            # Phase 1 — pretrain (align bounds, warm-start encoder)
            f_old = NetOld(
                model_path=None, mode="regression",
                h=h_old, h_lower=h_lower_old, h_upper=h_upper_old, g=g_old,
                binary_split=None, l=None, u=None,
                d_lower=None, d_upper=None, e_lower=None, e_upper=None,
                pretrain=True,
                generation_timelimit=None, generation_solvelimit=None,
                generation_frequency=None,
                lambda_box_pretrain=lambda_box_pretrain,
                lambda_box_train=lambda_box_train,
            )
            f_old.compile(optimizer=optimizer, loss=loss)
            f_old.fit(x_train, y_train, epochs=preepochs, batch_size=batch_size,
                      callbacks=[early_stopping])

            print("  --- Old SMiLE: enforcement ---")
            # Phase 2 — train with MILP enforcement
            f_old = NetOld(
                model_path=solver_dir, mode="regression",
                h=f_old.h, h_lower=f_old.h_lower, h_upper=f_old.h_upper,
                g=f_old.g,
                binary_split=input_dim, l=data.l, u=data.u,
                d_lower=d_lower, d_upper=d_upper, e_lower=e_lower,
                e_upper=e_upper, pretrain=False,
                generation_timelimit=generation_timelimit,
                generation_solvelimit=generation_solvelimit,
                generation_frequency=generation_frequency,
                lambda_box_pretrain=lambda_box_pretrain,
                lambda_box_train=lambda_box_train,
            )
            f_old.compile(optimizer=optimizer, loss=loss)
            f_old.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
            f_old.enforce(l=data.l, u=data.u, d_lower=d_lower, d_upper=d_upper,
                          e_lower=e_lower, e_upper=e_upper)
            training_time = np.round(time.time() - t0, 2)

            r2 = r2_score(y_true=y_test,
                          y_pred=f_old(x_test).numpy().flatten())
            pickle.dump({
                "approach":       "old",
                "alpha":          alpha,
                "omega":          omega,
                "r2":             r2,
                "training_time":  training_time,
                "weights":        f_old.get_weights(),
                # Solver logs (used by analyze_degeneracy.py)
                "phase":          list(f_old.generator.phases),
                "model":          list(f_old.generator.modelcounts),
                "timelimit":      list(f_old.generator.timelimits),
                "time":           list(f_old.generator.times),
                "measured_time":  list(f_old.generator.measured_times),
                "primal":         list(f_old.generator.primals),
                "dual":           list(f_old.generator.duals),
                "termination":    list(f_old.generator.terminations),
            }, open(old_file, "wb"))
            shutil.rmtree(solver_dir)
            print(f"  old SMiLE  r2={r2:.4f}  time={training_time}s")

        # ==================================================================
        # NEW SMiLE
        # ==================================================================
        new_file = f"{res_dir}/{tag}_new.pkl"
        if os.path.isfile(new_file):
            print(f"  [SKIP] new SMiLE already exists: {new_file}")
        else:
            print("  --- New SMiLE: pretrain ---")
            solver_dir = f"{res_dir}/{tag}_new_solver"
            Path(solver_dir).mkdir(parents=True, exist_ok=True)

            h_new       = BaseNetNew(h_architecture)
            h_base_new  = BaseNetNew(h_base_architecture)
            # output_bias=0.1  shifts final-layer biases positive at init
            # to prevent dead ReLU neurons in h_delta at the start of training.
            h_delta_new = BaseNetNew(h_delta_architecture, output_bias=0.1)
            g_new       = BaseNetNew(g_architecture)

            t0 = time.time()

            # Phase 1 — pretrain
            f_new = NetNew(
                model_path=None, mode="regression",
                h=h_new, h_base=h_base_new, h_delta=h_delta_new, g=g_new,
                binary_split=None, l=None, u=None,
                d_lower=None, d_upper=None, e_lower=None, e_upper=None,
                pretrain=True,
                generation_timelimit=None, generation_solvelimit=None,
                generation_frequency=None,
                lambda_box_pretrain=lambda_box_pretrain,
                lambda_box_train=lambda_box_train,
            )
            f_new.compile(optimizer=optimizer, loss=loss)
            f_new.fit(x_train, y_train, epochs=preepochs, batch_size=batch_size,
                      callbacks=[early_stopping])

            print("  --- New SMiLE: enforcement ---")
            # Phase 2 — train with MILP enforcement
            f_new = NetNew(
                model_path=solver_dir, mode="regression",
                h=f_new.h, h_base=f_new.h_base, h_delta=f_new.h_delta,
                g=f_new.g,
                binary_split=input_dim, l=data.l, u=data.u,
                d_lower=d_lower, d_upper=d_upper, e_lower=e_lower,
                e_upper=e_upper, pretrain=False,
                generation_timelimit=generation_timelimit,
                generation_solvelimit=generation_solvelimit,
                generation_frequency=generation_frequency,
                lambda_box_pretrain=lambda_box_pretrain,
                lambda_box_train=lambda_box_train,
            )
            f_new.compile(optimizer=optimizer, loss=loss)
            f_new.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
            f_new.enforce(l=data.l, u=data.u, d_lower=d_lower, d_upper=d_upper,
                          e_lower=e_lower, e_upper=e_upper)
            training_time = np.round(time.time() - t0, 2)

            r2 = r2_score(y_true=y_test,
                          y_pred=f_new(x_test).numpy().flatten())
            pickle.dump({
                "approach":       "new",
                "alpha":          alpha,
                "omega":          omega,
                "r2":             r2,
                "training_time":  training_time,
                "weights":        f_new.get_weights(),
                "phase":          list(f_new.generator.phases),
                "model":          list(f_new.generator.modelcounts),
                "timelimit":      list(f_new.generator.timelimits),
                "time":           list(f_new.generator.times),
                "measured_time":  list(f_new.generator.measured_times),
                "primal":         list(f_new.generator.primals),
                "dual":           list(f_new.generator.duals),
                "termination":    list(f_new.generator.terminations),
            }, open(new_file, "wb"))
            shutil.rmtree(solver_dir)
            print(f"  new SMiLE  r2={r2:.4f}  time={training_time}s")