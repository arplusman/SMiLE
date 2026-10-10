import os
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from core.data import *
from core.model_baseline import *
sns.set_palette('deep')
sns.set_theme()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)
pd.set_option('display.expand_frame_repr', False)

class FinalModel(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def call(self, inputs):
        if self.model.g.get_weights()[0].shape[1] == 2:
            outputs = np.argmax(self.model(inputs), axis=1)
        else:
            outputs = int(self.model(inputs) >= 0)

        return outputs


####################################################################################################
# Hyperparameters
####################################################################################################
# Meta
name = 'benchmarks/robustness'
train_res_dir = f'{name}/results/train'
attack_res_dir = f'{name}/results/attack'
agg_res_dir = f'{name}/aggregated_results'

# Testing
termination = {
        2 : 'optimal',
        3 : 'infeasible',
        4 : 'inforunb',
        5 : 'unbounded',
        9 : 'timelimit',
        }

# Tasks
data = MNIST()
l, u, x_train, y_train, x_test, y_test = data.load('zero-nonzero')

# Dimensions
input_dim = 28*28
l, u = [0.]*input_dim, [1.]*input_dim
hidden_dim = 8

# Property
deltas = [0.01, 0.025, 0.05, 0.075, 0.10]
epsilons = [1.25, 1.00, 0.75]

# Architecture
h_architecture = [
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
h_aux_architecture = [(hidden_dim, 'linear')]


####################################################################################################
# Load Model
####################################################################################################
results = {}
attacks = []
for model in os.listdir(train_res_dir):
    if 'pkl' in model:
        result_train = pickle.load(open(f'{train_res_dir}/{model}', 'rb'))
        result_attack = pickle.load(open(f'{attack_res_dir}/{model}', 'rb'))
        model = model.split('.')[0]
        results[model] = {}
        delta = result_train['delta']
        epsilon = result_train['epsilon']

        results[model]['delta'] = delta
        results[model]['epsilon'] = epsilon

        # Learning Results
        results[model]['clean_accuracy'] = result_train['clean_accuracy']
        if model in ['agnostic', 'presmle']:
            for delta in deltas:
                results[model][f'pgd_accuracy'] = {delta: result_attack[delta]['pgd_accuracy'] for delta in deltas}
        else:
            results[model]['pgd_accuracy'] = {delta: result_attack['pgd_accuracy']}
        results[model]['verified_accuracy'] = result_train['verified_accuracy']
        results[model]['rejection_rate'] = result_train['rejection_rate']
        results[model]['training_time'] = result_train['training_time']
        results[model]['evaluation_time'] = result_train['evaluation_time']

        # Generation Results
        if model not in ['agnostic', 'presmle'] and 'crownibp' not in model:
            del result_train['weights']
            del result_train['delta']
            del result_train['epsilon']
            del result_train['clean_accuracy']
            del result_train['verified_accuracy']
            del result_train['rejection_rate']
            del result_train['training_time']
            del result_train['evaluation_time']
            result_train = pd.DataFrame(result_train)
            result_train['exceeded_time'] = np.maximum(0., result_train['time'] - 3*result_train['timelimit']/2)
            result_train['termination'] = result_train['termination'].replace(termination)
            results[model]['generation'] = result_train
            results[model]['guarantee'] = result_train['termination'].iloc[-1] == 'inforunb'
            results[model]['training_time'] = results[model]['training_time'] - result_train['exceeded_time'].sum()
        else:
            results[model]['generation'] = None
            results[model]['guarantee'] = None


####################################################################################################
# Accuracy 
####################################################################################################
accuracy = pd.DataFrame({
    'model' : [model.split('_')[0] for model in results],
    'delta' : [results[model]['delta'] for model in results],
    'epsilon' : [results[model]['epsilon'] for model in results],
    'clean_accuracy' : [results[model]['clean_accuracy'] for model in results],
    'verified_accuracy' : [results[model]['verified_accuracy'] for model in results],
    'rejection_rate' : [results[model]['rejection_rate'] for model in results],
    'training_time' : [results[model]['training_time'] for model in results],
    'evaluation_time' : [results[model]['evaluation_time'] for model in results],
    })
pgd_accuracy = []
for model in results:
    frame = pd.DataFrame({
        'model' : [model.split('_')[0] for delta in results[model]['pgd_accuracy']],
        'delta' : [results[model]['delta'] for delta in results[model]['pgd_accuracy']],
        'epsilon' : [results[model]['epsilon'] for delta in results[model]['pgd_accuracy']],
        'delta_attack' : results[model]['pgd_accuracy'].keys(),
        'pgd_accuracy' : results[model]['pgd_accuracy'].values(),
        })
    pgd_accuracy.append(frame)
pgd_accuracy = pd.concat(pgd_accuracy, ignore_index=True)
accuracy = pd.merge(accuracy, pgd_accuracy)

accuracy.loc[accuracy['model'] == 'smle', 'training_time'] += accuracy[accuracy['model'] == 'presmle']['training_time'].iloc[0]
accuracy['delta'] = accuracy['delta'].replace(0., np.nan)
accuracy['epsilon'] = accuracy['epsilon'].replace({np.inf: np.nan, -1: np.nan})
accuracy['rejection_rate'] = accuracy['rejection_rate'].replace(1., np.nan)
accuracy = accuracy[accuracy['model'] != 'presmle']

accuracy = accuracy[['model', 'delta', 'epsilon', 'delta_attack', 
                     'clean_accuracy', 'pgd_accuracy', 'verified_accuracy', 'rejection_rate',
                     'training_time', 'evaluation_time']] 
accuracy = accuracy.sort_values(['delta', 'model', 'epsilon'], ascending = [True, True, False])
accuracy = accuracy.reset_index().drop('index', axis=1)

print('\n==============================================================================================================')
print('Accuracy')
print('==============================================================================================================')
time = accuracy[['model', 'training_time', 'evaluation_time']]

accuracy = accuracy[['model', 'delta_attack', 'epsilon', 'clean_accuracy', 'pgd_accuracy', 'verified_accuracy']]
accuracy = accuracy.groupby(['delta_attack', 'model'])[['clean_accuracy', 'pgd_accuracy', 'verified_accuracy']].mean().reset_index()
cols = ['clean_accuracy', 'pgd_accuracy', 'verified_accuracy']
accuracy[cols] = accuracy[cols].apply(lambda x: (100 * x).map('{:.2f}'.format))
accuracy.to_csv(f'{agg_res_dir}/robustness_accuracy.csv', index=False)
print(accuracy)


print('\n==============================================================================================================')
print('Time')
print('==============================================================================================================')
time = time.groupby('model')[['training_time', 'evaluation_time']].mean().reset_index()
time = time.sort_values(['model'], ascending = [True])
time[['training_time', 'evaluation_time']] = time[['training_time', 'evaluation_time']].round(2)
time.to_csv(f'{agg_res_dir}/robustness_time.csv', index=False)
print(time)
