import os
import time
import pickle
import shutil
from core.model_baseline import *
from core.data import *
# from competitors.fairness.competitor import CertiFair
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

class Logit(keras.Model):
    def __init__(self, f, **kwargs):
        super().__init__(**kwargs)
        self.f = f

    def call(self, inputs):
           if self.f.h_lower:
               outputs = self.f.g(tf.maximum(tf.minimum(self.f.h(inputs), self.f.h_upper(inputs)), self.f.h_lower(inputs)))
           else:
               outputs = self.f.g(self.f.h(inputs))

           return outputs


####################################################################################################
# Hyperparameters
####################################################################################################
# Meta
name = 'benchmarks/fairness'
data_dir = f'{name}/data'
res_dir = f'{name}/results_baseline'

# Training
epochs = 50
batch_size = {'compas' : 256, 'law' : 1024, 'crime' : 128}
optimizer = 'adam'
benchmarks = ['compas', 'law', 'crime']

# Data
data = FairnessData()

# Architecture
hidden_dim = 8
h_architecture = [(4*hidden_dim, 'relu'), (4*hidden_dim, 'relu'), (hidden_dim, 'linear')]
h_aux_architecture = [(int(1/2*hidden_dim), 'relu'), (hidden_dim, 'linear')]
g_architecture = [(1, 'linear')]

# Property
epsilons = {
        'compas' : [0.2, 0.4, 0.6, 0.8, 1.0],
        'crime' : [0.02, 0.04, 0.06, 0.08, 0.10],
        'law' : [1.0, 2.0, 3.0, 4.0, 5.0],
        }
lambdas = {
        'compas' : [0.025, 0.050, 0.075, 0.100, 0.125],
        'crime' : [0.08, 0.09, 0.10, 0.11, 0.12],
        'law' : [0.025, 0.050, 0.075, 0.100, 0.125],
        }


# Enforcement 
generation_frequency = 1
generation_timelimit = 2
generation_solvelimit = 8


####################################################################################################
# Process
####################################################################################################
for benchmark in benchmarks:
    input_dim, binary_split, l, u, x_train, y_train, x_test, y_test = data.load(benchmark=benchmark)
    x_test_counter = x_test.copy()
    x_test_counter.iloc[:,-1] = 1 - x_test_counter.iloc[:,-1]
    task = 'regression' if benchmark=='crime' else 'classification'
    loss = 'mse' if benchmark=='crime' else 'binary_crossentropy'

    print('====================================================================================================')
    print(f'benchmark: {benchmark}')
    print('====================================================================================================')
    print(f'train_size --> {x_train.shape[0]}')
    print(f'input_dim --> {input_dim}')
    print(f'binary_split --> {binary_split}')
    print(f'task --> {task}')
    print(f'loss --> {loss}')
    print(f'features --> {x_train.columns}')
    print(f'target --> {y_train.name}')

    ####################################################################################################
    # Agnostic
    ####################################################################################################
    print('====================================================================================================')
    print(f'AGNOSTIC')
    print('====================================================================================================')
    model_file = f'{res_dir}/{benchmark}_agnostic.pkl'
    if not os.path.isfile(model_file):
        delta, epsilon, lambda_ = None, None, None
        h = BaseNet(h_architecture)
        g = BaseNet(g_architecture)

        f = Net(model_path=None, mode=task,
                h=h, h_lower=None, h_upper=None, g=g, 
                binary_split=None, l=None, u=None, d_lower=None, d_upper=None, e_lower=None, e_upper=None, 
                pretrain=False, generation_timelimit=None, generation_solvelimit=None, generation_frequency=None)
        f.compile(optimizer='adam', loss=loss)
        training_time = time.time()
        f.fit(x_train, y_train, epochs=epochs, batch_size=batch_size[benchmark])
        training_time = time.time() - training_time

        y_test_pred = f(x_test).numpy().flatten()
        y_test_pred = np.round(y_test_pred) if task=='classification' else y_test_pred
        f_logit = Logit(f)

        results = {
            'benchmark' : benchmark,
            'delta' : delta,
            'epsilon' : epsilon,
            'lambda' : lambda_,
            'phase' : None,
            'model' : None,
            'timelimit' : None,
            'time' : None,
            'measured_time' : None,
            'primal' : None,
            'dual' : None,
            'termination' : None,
            'weights' : f.get_weights(),
            'r2' : r2_score(y_pred=y_test_pred, y_true=y_test) if task=='regression' else None,
            'mse' : mean_squared_error(y_pred=y_test_pred, y_true=y_test) if task=='regression' else None,
            'accuracy' : accuracy_score(y_pred=y_test_pred, y_true=y_test) if task=='classification' else None,
            'counter_var' : np.max(np.abs(f_logit(x_test) - f_logit(x_test_counter))),
            'training_time' : np.round(training_time, 2),
        }
        pickle.dump(results, open(model_file, 'wb'))
        print(f'r2 --> {results["r2"]}')
        print(f'accuracy --> {results["accuracy"]}')
        print(f'counter_var --> {results["counter_var"]}')


    ####################################################################################################
    # PRESMLE
    ####################################################################################################
    print('====================================================================================================')
    print(f'PRESMLE')
    print('====================================================================================================')
    model_file = f'{res_dir}/{benchmark}_presmle.pkl'
    if not os.path.isfile(model_file):
        delta, epsilon, lambda_ = None, None, None
        h_lower = BaseNet(h_aux_architecture)
        h_upper = BaseNet(h_aux_architecture)
        h = BaseNet(h_architecture)
        g = BaseNet(g_architecture)

        f = Net(model_path=None, mode=task,
                h=h, h_lower=h_lower, h_upper=h_upper, g=g, 
                binary_split=None, l=None, u=None, d_lower=None, d_upper=None, e_lower=None, e_upper=None, 
                pretrain=True, generation_timelimit=None, generation_solvelimit=None, generation_frequency=None)
        f.compile(optimizer='adam', loss=loss)
        training_time = time.time()
        f.fit(x_train, y_train, epochs=epochs, batch_size=batch_size[benchmark])
        training_time = time.time() - training_time

        y_test_pred = f(x_test).numpy().flatten()
        y_test_pred = np.round(y_test_pred) if task=='classification' else y_test_pred
        f_logit = Logit(f)

        results = {
            'benchmark' : benchmark,
            'delta' : delta,
            'epsilon' : epsilon,
            'lambda' : lambda_,
            'phase' : None,
            'model' : None,
            'timelimit' : None,
            'time' : None,
            'measured_time' : None,
            'primal' : None,
            'dual' : None,
            'termination' : None,
            'weights' : f.get_weights(),
            'r2' : r2_score(y_pred=y_test_pred, y_true=y_test) if task=='regression' else None,
            'mse' : mean_squared_error(y_pred=y_test_pred, y_true=y_test) if task=='regression' else None,
            'accuracy' : accuracy_score(y_pred=y_test_pred, y_true=y_test) if task=='classification' else None,
            'counter_var' : np.max(np.abs(f_logit(x_test) - f_logit(x_test_counter))),
            'training_time' : np.round(training_time, 2),
        }
        pickle.dump(results, open(model_file, 'wb'))
        preweights = f.get_weights()
        presmle_training_time = training_time
        print(f'r2 --> {results["r2"]}')
        print(f'accuracy --> {results["accuracy"]}')
        print(f'counter_var --> {results["counter_var"]}')
    else:
        results = pickle.load(open(model_file, 'rb'))
        preweights = results['weights']
        presmle_training_time = results['training_time']


    # for lambda_ in lambdas[benchmark]:
    #     ####################################################################################################
    #     # CertiRair
    #     ####################################################################################################
    #     model_name = f'{benchmark}_certifair_{str(lambda_).replace(".", "")}'
    #     model_file = f'{res_dir}/{model_name}.pkl'
    #     print('====================================================================================================')
    #     print(f'CertiFair -- lambda: {lambda_}')
    #     print('====================================================================================================')
    #     if not os.path.isfile(model_file):
    #         delta, epsilon = None, None
    #         h = BaseNet(h_architecture)
    #         g = BaseNet(g_architecture)
    #         weights, training_time = CertiFair(model_name=model_name, benchmark=benchmark, 
    #                                            lambda_fair=lambda_, architecture=h_architecture + g_architecture, 
    #                                            batch_size=batch_size[benchmark], epochs=epochs, task=task)
    #         f = Net(model_path=None, mode=task,
    #                 h=h, h_lower=None, h_upper=None, g=g, 
    #                 binary_split=None, l=None, u=None, d_lower=None, d_upper=None, e_lower=None, e_upper=None, 
    #                 pretrain=False, generation_timelimit=None, generation_solvelimit=None, generation_frequency=None)
    #         f.compile(optimizer='adam', loss=loss)
    #         f(x_test)
    #         f.set_weights(weights)

    #         y_test_pred = f(x_test).numpy().flatten()
    #         y_test_pred = np.round(y_test_pred) if task=='classification' else y_test_pred
    #         f_logit = Logit(f)

    #         results = {
    #             'benchmark' : benchmark,
    #             'delta' : delta,
    #             'epsilon' : epsilon,
    #             'lambda' : lambda_,
    #             'phase' : None,
    #             'model' : None,
    #             'timelimit' : None,
    #             'time' : None,
    #             'measured_time' : None,
    #             'primal' : None,
    #             'dual' : None,
    #             'termination' : None,
    #             'weights' : f.get_weights(),
    #             'r2' : r2_score(y_pred=y_test_pred, y_true=y_test) if task=='regression' else None,
    #             'mse' : mean_squared_error(y_pred=y_test_pred, y_true=y_test) if task=='regression' else None,
    #             'accuracy' : accuracy_score(y_pred=y_test_pred, y_true=y_test) if task=='classification' else None,
    #             'counter_var' : np.max(np.abs(f_logit(x_test) - f_logit(x_test_counter))),
    #             'training_time' : np.round(training_time, 2),
    #         }
    #         pickle.dump(results, open(model_file, 'wb'))
    #         print(f'r2 --> {results["r2"]}')
    #         print(f'accuracy --> {results["accuracy"]}')
    #         print(f'counter_var --> {results["counter_var"]}')


    ####################################################################################################
    # SMLE
    ####################################################################################################
    for epsilon in epsilons[benchmark]:
        delta, lambda_ = 0.00, None
        print('====================================================================================================')
        print(f'SMLE -- delta: {delta} -- epsilon: {epsilon}')
        print('====================================================================================================')
        d_lower, d_upper = [-delta]*input_dim, [delta]*input_dim
        d_lower[-1], d_upper[-1] = -1, 1 # The protected attribute is always the last feature and always binary 
        e_lower, e_upper = -epsilon, epsilon
        model_path = f'{res_dir}/{benchmark}_smle_{str(delta).replace(".", "")}_{str(epsilon).replace(".", "")}'
        model_file = f'{model_path}.pkl'

        if not os.path.isfile(model_file):
            # os.system(f'mkdir {model_path}')
            os.mkdir(model_path)
            h_lower = BaseNet(h_aux_architecture)
            h_upper = BaseNet(h_aux_architecture)
            h = BaseNet(h_architecture)
            g = BaseNet(g_architecture)

            print(f'========== Train ==========') 
            f = Net(model_path=model_path, mode=task,
                    h=h, h_lower=h_lower, h_upper=h_upper, g=g, 
                    binary_split=binary_split, l=l, u=u, d_lower=d_lower, d_upper=d_upper, e_lower=e_lower, e_upper=e_upper, 
                    pretrain=False, generation_timelimit=generation_timelimit, generation_solvelimit=generation_solvelimit, generation_frequency=generation_frequency)
            f.compile(optimizer='adam', loss=loss)
            f(x_test)
            f.set_weights(preweights)
            training_time = time.time()
            f.fit(x_train, y_train, epochs=epochs, batch_size=batch_size[benchmark])
            f_logit = Logit(f)

            print(f'========== Enforce ==========') 
            f.enforce(l=l, u=u, d_lower=d_lower, d_upper=d_upper, e_lower=e_lower, e_upper=e_upper)
            training_time = time.time() - training_time

            y_test_pred = f(x_test).numpy().flatten()
            y_test_pred = np.round(y_test_pred) if task=='classification' else y_test_pred

            results = {
                    'benchmark' : benchmark,
                    'delta' : delta,
                    'epsilon' : epsilon,
                    'lambda' : lambda_,
                    'phase' : list(f.generator.phases),
                    'model' : list(f.generator.modelcounts),
                    'timelimit' : list(f.generator.timelimits),
                    'time' : list(f.generator.times),
                    'measured_time' : list(f.generator.measured_times),
                    'primal' : list(f.generator.primals),
                    'dual' : list(f.generator.duals),
                    'termination' : list(f.generator.terminations),
                    'weights' : f.get_weights(),
                    'r2' : r2_score(y_pred=y_test_pred, y_true=y_test) if task=='regression' else None,
                    'mse' : mean_squared_error(y_pred=y_test_pred, y_true=y_test) if task=='regression' else None,
                    'accuracy' : accuracy_score(y_pred=y_test_pred, y_true=y_test) if task=='classification' else None,
                    'counter_var' : np.max(np.abs(f_logit(x_test) - f_logit(x_test_counter))),
                    'training_time' : np.round(presmle_training_time + training_time, 2),
                }
            pickle.dump(results, open(model_file, 'wb'))
            # os.system(f'rm -rf {model_path}')
            shutil.rmtree(model_path)
            print(f'r2 --> {results["r2"]}')
            print(f'accuracy --> {results["accuracy"]}')
            print(f'counter_var --> {results["counter_var"]}')
