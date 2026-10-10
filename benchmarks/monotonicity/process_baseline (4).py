import pickle
import shutil
import random
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from core.model import *
from core.data import *
import matplotlib.pyplot as plt

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


####################################################################################################
# Hyperparameters
####################################################################################################
# Meta
name = 'benchmarks/monotonicity'
res_dir = f'{name}/results_baseline'          # ← baseline-specific output dir

# Training
preepochs = 1000
epochs = 100
batch_size = 64
optimizer = 'adam'
loss = 'mse'
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

# Tasks
n_samples = 1000
input_dim = 1

# Properties
alphas = [2.0, 3.0, 4.0]
omegas = [0.4, 0.6, 0.8]
delta = 0.0
epsilon = 0.15

# Architecture
hidden_dim = 8
aux_complexity = 32
h_architecture = [(16, 'relu'), (32, 'relu'), (64, 'relu'), (32, 'relu'), (16, 'relu'), (hidden_dim, 'linear')]
h_aux_architecture = [(aux_complexity, 'relu'), (hidden_dim, 'linear')]
g_architecture = [(1, 'linear')]

# Enforcement
lambda_box_pretrain = 1.5
lambda_box_train = 1.5
generation_frequency = 1
generation_timelimit = 2
generation_solvelimit = 8


####################################################################################################
# Monotonicity — Baseline (independent h_lower / h_upper networks)
####################################################################################################
for alpha in alphas:
    for omega in omegas:
        task = lambda x, a=alpha, w=omega: x + a*np.sin(w*x)
        l, u = [-10], [10]
        data = Synthetic(n_samples=n_samples, input_dim=input_dim, l=l, u=u, func=task)
        x, y, x_train, y_train, x_test, y_test = data.generate(0, standardize=True)

        d_lower, d_upper = [data.l], [delta]
        e_lower, e_upper = y.min(), epsilon

        print('====================================================================================================')
        print(f'BASELINE -- alpha: {alpha} -- omega: {omega}')
        print('====================================================================================================')
        model_file = f'{res_dir}/{str(alpha).replace(".","")}_{str(omega).replace(".","")}.pkl'
        if not os.path.isfile(model_file):
            os.makedirs(f'{res_dir}/model', exist_ok=True)

            # Baseline auxiliary architecture: two independent networks.
            # h_lower and h_upper are trained separately with no structural
            # ordering guarantee — this is the source of degeneracy being
            # addressed by the new approach.
            h = BaseNet(h_architecture)
            h_lower = BaseNet(h_aux_architecture)
            h_upper = BaseNet(h_aux_architecture)
            g = BaseNet(g_architecture)

            # Pretrain
            t0 = time.time()
            f = Net(model_path=None, mode='regression',
                    h=h, h_lower=h_lower, h_upper=h_upper, g=g,
                    binary_split=None, l=None, u=None,
                    d_lower=None, d_upper=None, e_lower=None, e_upper=None,
                    pretrain=True,
                    generation_timelimit=None, generation_solvelimit=None, generation_frequency=None,
                    lambda_box_pretrain=lambda_box_pretrain, lambda_box_train=lambda_box_train)
            f.compile(optimizer=optimizer, loss=loss)
            f.fit(x_train, y_train, epochs=preepochs, batch_size=batch_size, callbacks=[early_stopping])

            # Train with property enforcement
            f = Net(model_path=f'{res_dir}/model', mode='regression',
                    h=f.h, h_lower=f.h_lower, h_upper=f.h_upper, g=f.g,
                    binary_split=input_dim, l=data.l, u=data.u,
                    d_lower=d_lower, d_upper=d_upper, e_lower=e_lower, e_upper=e_upper,
                    pretrain=False,
                    generation_timelimit=generation_timelimit,
                    generation_solvelimit=generation_solvelimit,
                    generation_frequency=generation_frequency,
                    lambda_box_pretrain=lambda_box_pretrain, lambda_box_train=lambda_box_train)
            f.compile(optimizer=optimizer, loss=loss)
            f.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

            f.enforce(l=data.l, u=data.u, d_lower=d_lower, d_upper=d_upper, e_lower=e_lower, e_upper=e_upper)
            training_time = np.round(time.time() - t0, 2)

            results = {
                'weights':        f.get_weights(),
                'alpha':          alpha,
                'omega':          omega,
                'r2':             r2_score(y_true=y_test, y_pred=f(x_test).numpy().flatten()),
                'phase':          list(f.generator.phases),
                'model':          list(f.generator.modelcounts),
                'timelimit':      list(f.generator.timelimits),
                'time':           list(f.generator.times),
                'measured_time':  list(f.generator.measured_times),
                'primal':         list(f.generator.primals),
                'dual':           list(f.generator.duals),
                'termination':    list(f.generator.terminations),
                'training_time':  training_time,
            }
            pickle.dump(results, open(model_file, 'wb'))
            shutil.rmtree(f'{res_dir}/model')