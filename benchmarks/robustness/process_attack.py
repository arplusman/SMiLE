import os
import tensorflow as tf
import numpy as np
import pickle
from core.data import *
from core.model_baseline import *
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import ProjectedGradientDescent


####################################################################################################
# Utilities
####################################################################################################SS
class WrappedModel(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def call(self, inputs):
        if self.model.g.get_weights()[0].shape[1] == 2:
            outputs = self.model(inputs)
        else:
            logits = self.model(inputs)
            outputs = tf.concat([0*logits, logits], axis=1) 

        return outputs 


####################################################################################################
# Hyperparameters
####################################################################################################
# Meta
name = 'benchmarks/robustness'
train_res_dir = f'{name}/results/train'
attack_res_dir = f'{name}/results/attack'

# Tasks
data = MNIST()
l, u, x_train, y_train, x_test, y_test = data.load('zero-nonzero')

# Dimensions
input_dim = 28*28
l, u = [0.]*input_dim, [1.]*input_dim
hidden_dim = 8

# Property
deltas = [0.01, 0.025, 0.050, 0.075, 0.10]
epsilons = [1.5, 1.0, 0.75]
attack_range = (0, 1000)


####################################################################################################
# Load
####################################################################################################
models = {}
for model in os.listdir(train_res_dir):
    if 'pkl' in model:
        result = pickle.load(open(f'{train_res_dir}/{model}', 'rb'))
        model = model.split('.')[0]
        models[model] = {}

        # Load Model
        if 'crownibp' in model:
            h = BaseNet([
                (1*hidden_dim, 'conv'),
                (1*hidden_dim, 'conv'),
                (2*hidden_dim, 'conv'),
                (2*hidden_dim, 'conv'),
                (4*hidden_dim, 'conv'),
                (4*hidden_dim, 'conv'),
                (None, 'flatten'),
                (4*hidden_dim, 'relu'),
                (hidden_dim, 'linear')
                ], model='crownibp')
            h_lower = None
            h_upper = None
            g = BaseNet([(2, 'linear')])
        elif 'agnostic' in model:
            h = BaseNet([
                (1*hidden_dim, 'conv'),
                (1*hidden_dim, 'conv'),
                (2*hidden_dim, 'conv'),
                (2*hidden_dim, 'conv'),
                (4*hidden_dim, 'conv'),
                (4*hidden_dim, 'conv'),
                (None, 'flatten'),
                (4*hidden_dim, 'relu'),
                (hidden_dim, 'linear')
                ])
            h_lower = None
            h_upper = None
            g = BaseNet([(1, 'linear')])
        else:
            h = BaseNet([
                (1*hidden_dim, 'conv'),
                (1*hidden_dim, 'conv'),
                (2*hidden_dim, 'conv'),
                (2*hidden_dim, 'conv'),
                (4*hidden_dim, 'conv'),
                (4*hidden_dim, 'conv'),
                (None, 'flatten'),
                (4*hidden_dim, 'relu'),
                (hidden_dim, 'linear')
                ])
            h_lower = BaseNet([(hidden_dim, 'linear')])
            h_upper = BaseNet([(hidden_dim, 'linear')])
            g = BaseNet([(1, 'linear')])

        f = Net(model_path=None, mode='regression',
                h=h, h_lower=h_lower, h_upper=h_upper, g=g,
                binary_split=input_dim, l=l, u=u, d_lower=None, d_upper=None, e_lower=None, e_upper=None,
                pretrain=False, generation_timelimit=None, generation_solvelimit=None, generation_frequency=None)
        f(x_test)
        f.set_weights(result['weights'])
        models[model]['f'] = f
        models[model]['delta'] = result['delta']
        models[model]['epsilon'] = result['epsilon']


####################################################################################################
# Attacks
####################################################################################################SS
for model in models:
    if model not in ['agnostic', 'presmle']:
        f = WrappedModel(model=models[model]['f'])
        classifier = TensorFlowV2Classifier(model=f, nb_classes=2, input_shape=(28, 28, 1), loss_object=tf.keras.losses.BinaryCrossentropy(from_logits=True), clip_values=(0.0, 1.0))
        delta = models[model]['delta']

        if not os.path.isfile(f'{attack_res_dir}/{model}.pkl'):
            attacks = {'attacks': []}

            print('====================================================================================================')
            print(f'model: {model} -- delta: {delta}')
            attack = ProjectedGradientDescent(estimator=classifier, norm=np.inf, eps=delta, max_iter=100, targeted=False)

            for i in range(attack_range[0], attack_range[1]):
                x = x_test[[i]]
                y = y_test[[i]]
                x_adv = attack.generate(x=x, y=y)
                attacks['attacks'].append((i, x, x_adv))

            x_attacks = np.concatenate([attacks['attacks'][i][2] for i in range(len(attacks['attacks']))], axis=0)
            y_test_attacks = y_test[[attacks['attacks'][i][0] for i in range(len(attacks['attacks']))]]
            attacks['pgd_accuracy'] = np.sum(np.argmax(f(x_attacks), axis=1) == y_test_attacks)/len(x_attacks)

            pickle.dump(attacks, open(f'{attack_res_dir}/{model}.pkl', 'wb'))
            print('====================================================================================================')

for model in ['agnostic', 'presmle']:
    f = WrappedModel(model=models[model]['f'])
    classifier = TensorFlowV2Classifier(model=f, nb_classes=2, input_shape=(28, 28, 1), loss_object=tf.keras.losses.BinaryCrossentropy(from_logits=True), clip_values=(0.0, 1.0))

    if not os.path.isfile(f'{attack_res_dir}/{model}.pkl'):
        attacks = {delta: {'attacks': []} for delta in deltas}

        for delta in deltas:
            print('====================================================================================================')
            print(f'model: {model} -- delta: {delta}')
            attack = ProjectedGradientDescent(estimator=classifier, norm=np.inf, eps=delta, max_iter=100, targeted=False)

            for i in range(attack_range[0], attack_range[1]):
                x = x_test[[i]]
                y = y_test[[i]]
                x_adv = attack.generate(x=x, y=y)
                attacks[delta]['attacks'].append((i, x, x_adv))
            x_attacks = np.concatenate([attacks[delta]['attacks'][i][2] for i in range(len(attacks[delta]['attacks']))], axis=0)
            y_test_attacks = y_test[[attacks[delta]['attacks'][i][0] for i in range(len(attacks[delta]['attacks']))]]
            attacks[delta]['pgd_accuracy'] = np.sum(np.argmax(f(x_attacks), axis=1) == y_test_attacks)/len(x_attacks)

            pickle.dump(attacks, open(f'{attack_res_dir}/{model}.pkl', 'wb'))
            print('====================================================================================================')
