"""
benchmarks/monotonicity/process_ablation.py

Pretrain loss ablation for the MONOTONICITY benchmark.
Tests three pretrain_mode variants — all else identical to process_new.py:

  'full'     — all accuracy pathways + box penalty   (paper formulation)
  'accuracy' — main accuracy term only, no box       (valid with base+delta)
  'none'     — no pretraining at all

Results saved to results_ablation/.

Run via:
    python run_benchmark.py --benchmark monotonicity --analysis ablation
"""

import os
import time
import pickle
import shutil
import random
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from core.model_new import *
from core.data import *

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


####################################################################################################
# Config — must match process_new.py exactly
####################################################################################################
name    = 'benchmarks/monotonicity'
res_dir = f'{name}/results_ablation'

preepochs  = 1000
epochs     = 100
batch_size = 64
optimizer  = 'adam'
loss       = 'mse'
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

n_samples = 1000
input_dim = 1

alphas  = [2.0, 3.0, 4.0]
omegas  = [0.4, 0.6, 0.8]
delta   = 0.0
epsilon = 0.15

hidden_dim           = 8
aux_complexity       = 32
h_architecture       = [(16, 'relu'), (32, 'relu'), (64, 'relu'), (32, 'relu'), (16, 'relu'), (hidden_dim, 'linear')]
h_base_architecture  = [(aux_complexity, 'relu'), (hidden_dim, 'linear')]
h_delta_architecture = [(aux_complexity, 'relu'), (hidden_dim, 'relu')]
g_architecture       = [(1, 'linear')]

lambda_box_pretrain  = 1.5
lambda_box_train     = 1.5
generation_frequency  = 1
generation_timelimit  = 2
generation_solvelimit = 8

PRETRAIN_MODES = ['full', 'accuracy', 'none']


####################################################################################################
# Process
####################################################################################################
for alpha in alphas:
    for omega in omegas:
        task = lambda x, a=alpha, w=omega: x + a * np.sin(w * x)
        l, u = [-10], [10]
        data = Synthetic(n_samples=n_samples, input_dim=input_dim, l=l, u=u, func=task)
        x, y, x_train, y_train, x_test, y_test = data.generate(0, standardize=True)

        d_lower, d_upper = [data.l], [delta]
        e_lower, e_upper = y.min(), epsilon

        print('='*90)
        print(f'MONOTONICITY ABLATION  alpha={alpha}  omega={omega}')
        print('='*90)

        for pretrain_mode in PRETRAIN_MODES:
            tag        = f'{str(alpha).replace(".","")}_{str(omega).replace(".","")}_{pretrain_mode}'
            model_file = f'{res_dir}/{tag}.pkl'
            model_dir  = f'{res_dir}/{tag}_solver'

            print(f'\n  pretrain_mode={pretrain_mode!r}')

            if os.path.isfile(model_file):
                print(f'  [SKIP] already exists.')
                continue

            os.makedirs(model_dir, exist_ok=True)

            h_base  = BaseNet(h_base_architecture)
            h_delta = BaseNet(h_delta_architecture, output_bias=0.1)
            h       = BaseNet(h_architecture)
            g       = BaseNet(g_architecture)

            t0 = time.time()

            # ------------------------------------------------------------------
            # Pretrain phase
            # 'full'     : all accuracy pathways + box penalty (paper formulation)
            # 'accuracy' : main pathway only, no box (safe with base+delta)
            # 'none'     : skip pretrain entirely, start from random init
            # ------------------------------------------------------------------
            if pretrain_mode != 'none':
                f_pre = Net(
                    model_path=None, mode='regression',
                    h=h, h_base=h_base, h_delta=h_delta, g=g,
                    binary_split=None, l=None, u=None,
                    d_lower=None, d_upper=None, e_lower=None, e_upper=None,
                    pretrain=True, pretrain_mode=pretrain_mode,
                    generation_timelimit=None, generation_solvelimit=None,
                    generation_frequency=None,
                    lambda_box_pretrain=lambda_box_pretrain,
                    lambda_box_train=lambda_box_train,
                )
                f_pre.compile(optimizer=optimizer, loss=loss)
                f_pre.fit(x_train, y_train, epochs=preepochs, batch_size=batch_size,
                          callbacks=[early_stopping])
                # Pass trained sub-networks to enforcement Net
                h, h_base, h_delta, g = f_pre.h, f_pre.h_base, f_pre.h_delta, f_pre.g
            else:
                # Build graph to initialise weights, then use random init directly.
                f_dummy = Net(
                    model_path=None, mode='regression',
                    h=h, h_base=h_base, h_delta=h_delta, g=g,
                    binary_split=None, l=None, u=None,
                    d_lower=None, d_upper=None, e_lower=None, e_upper=None,
                    pretrain=False, pretrain_mode='full',
                    generation_timelimit=None, generation_solvelimit=None,
                    generation_frequency=None,
                )
                f_dummy.compile(optimizer=optimizer, loss=loss)
                f_dummy(x_test)   # build graph — weights already random-initialised

            # ------------------------------------------------------------------
            # Train + Enforce
            # ------------------------------------------------------------------
            f = Net(
                model_path=model_dir, mode='regression',
                h=h, h_base=h_base, h_delta=h_delta, g=g,
                binary_split=input_dim, l=data.l, u=data.u,
                d_lower=d_lower, d_upper=d_upper, e_lower=e_lower, e_upper=e_upper,
                pretrain=False, pretrain_mode='full',
                generation_timelimit=generation_timelimit,
                generation_solvelimit=generation_solvelimit,
                generation_frequency=generation_frequency,
                lambda_box_pretrain=lambda_box_pretrain,
                lambda_box_train=lambda_box_train,
            )
            f.compile(optimizer=optimizer, loss=loss)
            f.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
            f.enforce(l=data.l, u=data.u, d_lower=d_lower, d_upper=d_upper,
                      e_lower=e_lower, e_upper=e_upper)
            training_time = np.round(time.time() - t0, 2)

            r2 = r2_score(y_true=y_test, y_pred=f(x_test).numpy().flatten())

            results = {
                'alpha':          alpha,
                'omega':          omega,
                'pretrain_mode':  pretrain_mode,
                'weights':        f.get_weights(),
                'r2':             r2,
                'training_time':  training_time,
                'phase':          list(f.generator.phases),
                'model':          list(f.generator.modelcounts),
                'timelimit':      list(f.generator.timelimits),
                'time':           list(f.generator.times),
                'measured_time':  list(f.generator.measured_times),
                'primal':         list(f.generator.primals),
                'dual':           list(f.generator.duals),
                'termination':    list(f.generator.terminations),
            }
            pickle.dump(results, open(model_file, 'wb'))
            shutil.rmtree(model_dir)
            print(f'  r2={r2:.4f}  training_time={training_time}s')