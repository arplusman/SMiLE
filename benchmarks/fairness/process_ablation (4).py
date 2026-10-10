"""
benchmarks/fairness/process_ablation.py

Pretrain loss ablation for the FAIRNESS benchmark.
Tests three pretrain_mode variants — all else identical to process_new.py:

  'full'     — all accuracy pathways + box penalty   (paper formulation)
  'accuracy' — main accuracy term only, no box       (valid with base+delta)
  'none'     — no pretraining at all

Results saved to results_ablation/ with suffix _ablation_{mode}.

Run via:
    python run_benchmark.py --benchmark fairness --analysis ablation
"""

import os
import time
import pickle
import shutil
import random
from core.model_new import *
from core.data import *

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score


class Logit(keras.Model):
    def __init__(self, f, **kwargs):
        super().__init__(**kwargs)
        self.f = f

    def call(self, inputs):
        if self.f.h_base:
            h_base_out  = self.f.h_base(self.f.flatten(inputs))
            h_upper_out = h_base_out + self.f.h_delta(self.f.flatten(inputs))
            return self.f.g(
                tf.maximum(tf.minimum(self.f.h(inputs), h_upper_out), h_base_out)
            )
        return self.f.g(self.f.h(inputs))


####################################################################################################
# Config
####################################################################################################
name    = 'benchmarks/fairness'
res_dir = f'{name}/results_ablation'

epochs     = 50
batch_size = {'compas': 256, 'law': 1024, 'crime': 128}
benchmarks = ['compas', 'law', 'crime']
data       = FairnessData()

hidden_dim           = 8
h_architecture       = [(4*hidden_dim, 'relu'), (4*hidden_dim, 'relu'), (hidden_dim, 'linear')]
h_base_architecture  = [(int(1/2*hidden_dim), 'relu'), (hidden_dim, 'linear')]
h_delta_architecture = [(int(1/2*hidden_dim), 'relu'), (hidden_dim, 'relu')]
g_architecture       = [(1, 'linear')]

epsilons = {
    'compas': [0.2, 0.4, 0.6, 0.8, 1.0],
    'crime':  [0.02, 0.04, 0.06, 0.08, 0.10],
    'law':    [1.0, 2.0, 3.0, 4.0, 5.0],
}

generation_frequency  = 1
generation_timelimit  = 2
generation_solvelimit = 8

PRETRAIN_MODES = ['full', 'accuracy', 'none']


####################################################################################################
# Process
####################################################################################################
for benchmark in benchmarks:
    input_dim, binary_split, l, u, x_train, y_train, x_test, y_test = data.load(benchmark=benchmark)
    x_test_counter = x_test.copy()
    x_test_counter.iloc[:, -1] = 1 - x_test_counter.iloc[:, -1]
    task = 'regression' if benchmark == 'crime' else 'classification'
    loss = 'mse' if benchmark == 'crime' else 'binary_crossentropy'

    print('='*90)
    print(f'FAIRNESS ABLATION  benchmark: {benchmark}  task: {task}')
    print('='*90)

    for pretrain_mode in PRETRAIN_MODES:
        for epsilon in epsilons[benchmark]:
            delta   = 0.00
            d_lower, d_upper = [-delta]*input_dim, [delta]*input_dim
            d_lower[-1], d_upper[-1] = -1, 1   # protected attribute always last, always binary
            e_lower, e_upper = -epsilon, epsilon

            tag        = f'{str(delta).replace(".","")}_{str(epsilon).replace(".","")}'
            model_path = f'{res_dir}/{benchmark}_smle_{tag}_ablation_{pretrain_mode}'
            model_file = f'{model_path}.pkl'

            print(f'\n  pretrain_mode={pretrain_mode!r}  benchmark={benchmark}  epsilon={epsilon}')

            if os.path.isfile(model_file):
                print(f'  [SKIP] already exists.')
                continue

            os.makedirs(f'{model_path}_solver', exist_ok=True)

            h_base  = BaseNet(h_base_architecture)
            h_delta = BaseNet(h_delta_architecture, output_bias=0.001)
            h       = BaseNet(h_architecture)
            g       = BaseNet(g_architecture)

            # ------------------------------------------------------------------
            # Pretrain phase
            # ------------------------------------------------------------------
            if pretrain_mode != 'none':
                f_pre = Net(
                    model_path=None, mode=task,
                    h=h, h_base=h_base, h_delta=h_delta, g=g,
                    binary_split=None, l=None, u=None,
                    d_lower=None, d_upper=None, e_lower=None, e_upper=None,
                    pretrain=True, pretrain_mode=pretrain_mode,
                    generation_timelimit=None, generation_solvelimit=None,
                    generation_frequency=None,
                )
                f_pre.compile(optimizer='adam', loss=loss)
                pretrain_time = time.time()
                f_pre.fit(x_train, y_train,
                          epochs=epochs, batch_size=batch_size[benchmark])
                pretrain_time = time.time() - pretrain_time
                preweights    = f_pre.get_weights()
            else:
                # No pretrain: build graph with a dummy forward pass, keep random init.
                f_dummy = Net(
                    model_path=None, mode=task,
                    h=h, h_base=h_base, h_delta=h_delta, g=g,
                    binary_split=None, l=None, u=None,
                    d_lower=None, d_upper=None, e_lower=None, e_upper=None,
                    pretrain=False, pretrain_mode='full',
                    generation_timelimit=None, generation_solvelimit=None,
                    generation_frequency=None,
                )
                f_dummy.compile(optimizer='adam', loss=loss)
                f_dummy(x_test)
                preweights    = f_dummy.get_weights()
                pretrain_time = 0.0

            # ------------------------------------------------------------------
            # Train + Enforce
            # ------------------------------------------------------------------
            f = Net(
                model_path=f'{model_path}_solver', mode=task,
                h=h, h_base=h_base, h_delta=h_delta, g=g,
                binary_split=binary_split, l=l, u=u,
                d_lower=d_lower, d_upper=d_upper, e_lower=e_lower, e_upper=e_upper,
                pretrain=False, pretrain_mode='full',
                generation_timelimit=generation_timelimit,
                generation_solvelimit=generation_solvelimit,
                generation_frequency=generation_frequency,
            )
            f.compile(optimizer='adam', loss=loss)
            f(x_test)
            f.set_weights(preweights)

            training_time = time.time()
            f.fit(x_train, y_train, epochs=epochs, batch_size=batch_size[benchmark])
            f_logit = Logit(f)
            f.enforce(l=l, u=u, d_lower=d_lower, d_upper=d_upper,
                      e_lower=e_lower, e_upper=e_upper)
            training_time = time.time() - training_time

            y_test_pred = f(x_test).numpy().flatten()
            y_test_pred = np.round(y_test_pred) if task == 'classification' else y_test_pred

            results = {
                'benchmark':      benchmark,
                'pretrain_mode':  pretrain_mode,
                'delta':          delta,
                'epsilon':        epsilon,
                'lambda':         None,
                'phase':          list(f.generator.phases),
                'model':          list(f.generator.modelcounts),
                'timelimit':      list(f.generator.timelimits),
                'time':           list(f.generator.times),
                'measured_time':  list(f.generator.measured_times),
                'primal':         list(f.generator.primals),
                'dual':           list(f.generator.duals),
                'termination':    list(f.generator.terminations),
                'weights':        f.get_weights(),
                'r2':             r2_score(y_pred=y_test_pred, y_true=y_test) if task == 'regression' else None,
                'mse':            mean_squared_error(y_pred=y_test_pred, y_true=y_test) if task == 'regression' else None,
                'accuracy':       accuracy_score(y_pred=y_test_pred, y_true=y_test) if task == 'classification' else None,
                'counter_var':    float(np.max(np.abs(f_logit(x_test) - f_logit(x_test_counter)))),
                'training_time':  np.round(pretrain_time + training_time, 2),
            }
            pickle.dump(results, open(model_file, 'wb'))
            shutil.rmtree(f'{model_path}_solver')

            print(f'  r2={results["r2"]}  acc={results["accuracy"]}  '
                  f'counter_var={results["counter_var"]:.4f}')