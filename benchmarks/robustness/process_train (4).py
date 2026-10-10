import os
import json
import pickle
import shutil
import torch
from core.model_baseline import *
from core.data import *
from competitors.robustness.competitor import *
import time


####################################################################################################
# Hyperparameters
####################################################################################################
# Meta
name = 'benchmarks/robustness'
res_dir = f'{name}/results/train'
auxiliary = f'{name}/results/auxiliary'

# Training
epochs = 20
batch_size = 256
optimizer = 'adam'
loss = 'binary_crossentropy'

# Tasks
data = MNIST()
l, u, x_train, y_train, x_test, y_test = data.load(binarize='zero-nonzero')

# Dimensions
input_dim = 28*28
hidden_dim = 8

# Architecture
smle_h_architecture = [
            (1*hidden_dim, 'conv'), 
            (1*hidden_dim, 'conv'), 
            (2*hidden_dim, 'conv'), 
            (2*hidden_dim, 'conv'), 
            (4*hidden_dim, 'conv'), 
            (4*hidden_dim, 'conv'), 
            (None, 'flatten'), 
            (4*hidden_dim, 'relu'), 
            (hidden_dim, 'linear')
            ]
smle_h_aux_architecture = [(hidden_dim, 'linear')]
smle_g_architecture = [(1, 'linear')]
crownibp_architecture = smle_h_architecture + [(2, 'linear')] 

# Property
deltas = [0.010, 0.025, 0.050, 0.075, 0.100]
epsilons = [1.25, 1.0, 0.75]
generation_frequency = 1
generation_timelimit = 2
generation_solvelimit = 8

print(f'deltas: {deltas}')
print(f'epsilons: {epsilons}')


####################################################################################################
# Agnostic
####################################################################################################
print('====================================================================================================')
print(f'AGNOSTIC')
print('====================================================================================================')
model_file = f'{res_dir}/agnostic.pkl'
if not os.path.isfile(model_file):
    delta, epsilon = 0.0, np.inf
    h = BaseNet(smle_h_architecture)
    g = BaseNet(smle_g_architecture)

    f = Net(model_path=None, mode='classification',
            h=h, h_lower=None, h_upper=None, g=g, 
            binary_split=None, l=l, u=u, d_lower=None, d_upper=None, e_lower=None, e_upper=None, 
            pretrain=False, generation_timelimit=None, generation_solvelimit=None, generation_frequency=None)
    f.compile(optimizer='adam', loss=loss)
    training_time = time.time()
    f.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    training_time = time.time() - training_time

    f_logit = lambda x: f.g(f.h(x))
    evaluation_time = time.time()
    y_test_pred = np.round(f(x_test)).flatten()
    verified = (f_logit(x_test).numpy().flatten() < -epsilon) | (f_logit(x_test).numpy().flatten() > epsilon)
    evaluation_time = time.time() - evaluation_time

    results = {
        'delta' : delta,
        'epsilon' : epsilon,
        'phase' : None,
        'model' : None,
        'timelimit' : None,
        'time' : None,
        'measured_time' : None,
        'primal' : None,
        'dual' : None,
        'termination' : None,
        'weights' : f.get_weights(),
        'clean_accuracy' : np.sum(y_test_pred == y_test)/len(y_test),
        'verified_accuracy' : np.sum((y_test_pred == y_test) & verified)/len(y_test),
        'rejection_rate' : np.sum(~verified)/len(y_test),
        'training_time' : np.round(training_time, 2),
        'evaluation_time' : np.round(evaluation_time, 2),
    }
    pickle.dump(results, open(model_file, 'wb'))


####################################################################################################
# CROWN-IBP
####################################################################################################
for delta in deltas:
    print('====================================================================================================')
    print(f'CROWN-IBP -- delta: {delta}')
    print('====================================================================================================')
    epsilon = -1
    aux_model_file = f'{auxiliary}/crownibp_{str(delta).replace(".", "")}'
    model_file = f'{res_dir}/crownibp_{str(delta).replace(".", "")}.pkl'
    if not os.path.isfile(model_file):
    
        weights, clean_accuracy, verified_accuracy, training_time, evaluation_time = CROWN_IBP(model_file=aux_model_file, delta=delta, architecture=crownibp_architecture)

        results = {
            'delta' : delta,
            'epsilon' : epsilon,
            'phase' : None,
            'model' : None,
            'timelimit' : None,
            'time' : None,
            'measured_time' : None,
            'primal' : None,
            'dual' : None,
            'termination' : None,
            'weights' : weights,
            'clean_accuracy' : clean_accuracy,
            'verified_accuracy' : verified_accuracy,
            'rejection_rate' : None,
            'training_time' : np.round(training_time, 2),
            'evaluation_time' : np.round(evaluation_time, 2),
        }
        pickle.dump(results, open(model_file, 'wb'))


####################################################################################################
# SMLE
####################################################################################################
print('====================================================================================================')
print(f'PRESMLE')
print('====================================================================================================')
model_path = f'{res_dir}/presmle'
model_file = f'{model_path}.pkl'

if not os.path.isfile(model_file):
    delta, epsilon = 0.0, np.inf

    # os.system(f'mkdir {model_path}')
    os.mkdir(model_path)
    h_lower = BaseNet(smle_h_aux_architecture)
    h_upper = BaseNet(smle_h_aux_architecture)
    h = BaseNet(smle_h_architecture)
    g = BaseNet(smle_g_architecture)

    f = Net(model_path=model_path, mode='classification',
            h=h, h_lower=h_lower, h_upper=h_upper, g=g, 
            binary_split=None, l=l, u=u, d_lower=None, d_upper=None, e_lower=None, e_upper=None, 
            pretrain=True, generation_timelimit=None, generation_solvelimit=None, generation_frequency=None)
    f.compile(optimizer='adam', loss=loss)
    training_time = time.time()
    f.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    training_time = time.time() - training_time

    f_logit = lambda x: f.g(tf.maximum(tf.minimum(f.h(x), f.h_upper(f.flatten(x))), f.h_lower(f.flatten(x))))
    evaluation_time = time.time()
    y_test_pred = np.round(f(x_test)).flatten()
    verified = (f_logit(x_test).numpy().flatten() < -epsilon) | (f_logit(x_test).numpy().flatten() > epsilon)
    evaluation_time = time.time() - evaluation_time

    results = {
        'delta' : delta,
        'epsilon' : epsilon,
        'phase' : None,
        'model' : None,
        'timelimit' : None,
        'time' : None,
        'measured_time' : None,
        'primal' : None,
        'dual' : None,
        'termination' : None,
        'weights' : f.get_weights(),
        'clean_accuracy' : np.sum(y_test_pred == y_test)/len(y_test),
        'verified_accuracy' : np.sum((y_test_pred == y_test) & verified)/len(y_test),
        'rejection_rate' : np.sum(~verified)/len(y_test),
        'training_time' : np.round(training_time, 2),
        'evaluation_time' : np.round(evaluation_time, 2),
    }
    pickle.dump(results, open(model_file, 'wb'))
    preweights = f.get_weights()
    # os.system(f'rm -rf {model_path}')
    shutil.rmtree(model_path)

else:
    results = pickle.load(open(model_file, 'rb'))
    preweights = results['weights']

for delta in deltas:
    for epsilon in epsilons:
        print('====================================================================================================')
        print(f'SMLE -- delta: {delta} -- epsilon: {epsilon}')
        print('====================================================================================================')
        d_lower, d_upper = [-delta]*input_dim, [delta]*input_dim
        e_lower, e_upper = -epsilon, epsilon
        model_path = f'{res_dir}/smle_{str(delta).replace(".", "")}_{str(epsilon).replace(".", "")}'
        model_file = f'{model_path}.pkl'

        if not os.path.isfile(model_file):
            # os.system(f'mkdir {model_path}')
            os.mkdir(model_path)
            h_lower = BaseNet(smle_h_aux_architecture)
            h_upper = BaseNet(smle_h_aux_architecture)
            h = BaseNet(smle_h_architecture)
            g = BaseNet(smle_g_architecture)

            print(f'========== Train ==========') 
            f = Net(model_path=model_path, mode='classification',
                    h=h, h_lower=h_lower, h_upper=h_upper, g=g, 
                    binary_split=input_dim, l=l, u=u, d_lower=d_lower, d_upper=d_upper, e_lower=e_lower, e_upper=e_upper, 
                    pretrain=False, generation_timelimit=generation_timelimit, generation_solvelimit=generation_solvelimit, generation_frequency=generation_frequency)
            f.compile(optimizer='adam', loss=loss)
            f(x_test)
            f.set_weights(preweights)
            training_time = time.time()
            f.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

            print(f'========== Enforce ==========') 
            f.enforce(l=l, u=u, d_lower=d_lower, d_upper=d_upper, e_lower=e_lower, e_upper=e_upper)
            training_time = time.time() - training_time

            f_logit = lambda x: f.g(tf.maximum(tf.minimum(f.h(x), f.h_upper(f.flatten(x))), f.h_lower(f.flatten(x))))
            evaluation_time = time.time()
            y_test_pred = np.round(f(x_test)).flatten()
            verified = (f_logit(x_test).numpy().flatten() < -epsilon) | (f_logit(x_test).numpy().flatten() > epsilon)
            evaluation_time = time.time() - evaluation_time

            results = {
                    'delta' : delta,
                    'epsilon' : epsilon,
                    'phase' : list(f.generator.phases),
                    'model' : list(f.generator.modelcounts),
                    'timelimit' : list(f.generator.timelimits),
                    'time' : list(f.generator.times),
                    'measured_time' : list(f.generator.measured_times),
                    'primal' : list(f.generator.primals),
                    'dual' : list(f.generator.duals),
                    'termination' : list(f.generator.terminations),
                    'weights' : f.get_weights(),
                    'clean_accuracy' : np.sum(y_test_pred == y_test)/len(y_test),
                    'verified_accuracy' : np.sum((y_test_pred == y_test) & verified)/len(y_test),
                    'rejection_rate' : np.sum(~verified)/len(y_test),
                    'training_time' : np.round(training_time, 2),
                    'evaluation_time' : np.round(evaluation_time, 2),
                }
            pickle.dump(results, open(model_file, 'wb'))
            # os.system(f'rm -rf {model_path}')
            shutil.rmtree(model_path)
