import pickle
import shutil
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda
from sklearn.metrics import r2_score
from core.model_baseline import *
from core.data import *
import matplotlib.pyplot as plt


####################################################################################################
# Utilities
####################################################################################################
def max_epsilon(x, y, delta):
    l, u = min(x), max(x)
    func = lambda z: np.interp(z, x.flatten(), y.flatten())

    deltas = np.linspace(0, delta, num=10, endpoint=True)
    epsilon = 0
    for delta in deltas:
        x1 = np.linspace(l-delta, u+delta, num=1000, endpoint=True)
        x2 = x1 + delta
        x1, x2 = np.clip(x1, l, u), np.clip(x2, l, u)
        y1 = func(x1)
        y2 = func(x2)
        epsilon_new = np.max(np.abs(y1 - y2))
        epsilon = epsilon_new if epsilon < epsilon_new else epsilon

    return epsilon


####################################################################################################
# Hyperparameters
####################################################################################################
# Meta
name = 'synthetic_example'
res_dir = f'{name}/results'

# Training
preepochs = 1000
epochs = 100
batch_size = 64
optimizer = 'adam'
loss = 'mse'
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

# Tasks - Properties
n_samples = 1000
input_dim = 1
task_robustness = lambda x: x**3
delta = 0.1

# Architecture
aux_complexity = 24
h_architecture = [(16, 'relu'), (32, 'relu'), (64, 'relu'), (32, 'relu'), (16, 'relu'), (1, 'linear')]
h_aux_architecture = [(aux_complexity, 'relu'), (1, 'linear')]
input_layer = Input(shape=(1,))  
identity_output = Lambda(lambda x: x)(input_layer)
g = Model(inputs=input_layer, outputs=identity_output)

# Enforcement
lambda_box_pretrain = 1.5
lambda_box_train = 1.5
generation_frequency = 1
generation_timelimit = 2
generation_solvelimit = 8


####################################################################################################
# Robustness
####################################################################################################
# Data
l, u = [-3], [3]
data = Synthetic(n_samples=n_samples, input_dim=input_dim, l=l, u=u, func=task_robustness)
x, y, x_train, y_train, x_test, y_test = data.generate(0, standardize=True)

factor = None
if not os.path.isfile(f'{res_dir}/robustness_0_None.pkl'):
    h = BaseNet(h_architecture)
    h_lower = BaseNet(h_aux_architecture)
    h_upper = BaseNet(h_aux_architecture)

    f = Net(model_path=None, mode='regression',
            h=h, h_lower=h_lower, h_upper=h_upper, g=g,
            binary_split=None, l=None, u=None, d_lower=None, d_upper=None, e_lower=None, e_upper=None,
            pretrain=True, generation_timelimit=None, generation_solvelimit=None, generation_frequency=None,
            lambda_box_pretrain=lambda_box_pretrain, lambda_box_train=lambda_box_train)
    f.compile(optimizer=optimizer, loss=loss)

    #######################################################
    # Initialization
    #######################################################
    f(x_test)
    model_file = f'{res_dir}/robustness_0_None.pkl'
    results = {
            'weights' : f.get_weights(),
            'property': 'robustness',
            'factor': factor,
            'r2' : r2_score(y_true=y_test, y_pred=f(x_test).numpy().flatten())
            }
    pickle.dump(results, open(model_file, 'wb'))

    #######################################################
    # Pretraining
    #######################################################
    f.fit(x_train, y_train, epochs=preepochs, batch_size=batch_size, callbacks=[early_stopping])
    model_file = f'{res_dir}/robustness_1_None.pkl'
    results = {
            'weights' : f.get_weights(),
            'property': 'robustness',
            'factor': factor,
            'r2' : r2_score(y_true=y_test, y_pred=f(x_test).numpy().flatten())
            }
    pickle.dump(results, open(model_file, 'wb'))
    preweights = f.get_weights()
else:
    model_file = f'{res_dir}/robustness_1_None.pkl'
    results = pickle.load(open(model_file, 'rb'))
    preweights = results['weights']


factor = 0.5
epsilon = factor*max_epsilon(x, y, delta=delta)
d_lower, d_upper = [-delta], [delta]
e_lower, e_upper = -epsilon , epsilon
print('====================================================================================================')
print(f'property: robustness -- factor: {factor} -- d_lower: {d_lower} -- d_upper: {d_upper} -- e_lower: {e_lower} -- e_upper: {e_upper}')
print('====================================================================================================')
if not os.path.isfile(f'{res_dir}/robustness_2_{str(factor).replace(".", "")}.pkl'):
    model_path = f'{res_dir}/model'
#     os.system(f'mkdir {model_path}')
    os.mkdir(model_path)
    h = BaseNet(h_architecture)
    h_lower = BaseNet(h_aux_architecture)
    h_upper = BaseNet(h_aux_architecture)

    f = Net(model_path=model_path, mode='regression',
            h=h, h_lower=h_lower, h_upper=h_upper, g=g,
            binary_split=input_dim, l=data.l, u=data.u, d_lower=d_lower, d_upper=d_upper, e_lower=e_lower, e_upper=e_upper,
            pretrain=False, generation_timelimit=generation_timelimit, generation_solvelimit=generation_solvelimit, generation_frequency=generation_frequency,
            lambda_box_pretrain=lambda_box_pretrain, lambda_box_train=lambda_box_train)
    f.compile(optimizer=optimizer, loss=loss)
    f(x_test)
    f.set_weights(preweights)

    #######################################################
    # Training
    #######################################################
    f.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    model_file = f'{res_dir}/robustness_2_{str(factor).replace(".", "")}.pkl'
    results = {
            'weights' : f.get_weights(),
            'property': 'robustness',
            'factor': factor,
            'r2' : r2_score(y_true=y_test, y_pred=f(x_test).numpy().flatten())
            }
    pickle.dump(results, open(model_file, 'wb'))

    #######################################################
    # Posttraining
    #######################################################
    f.enforce(l=data.l, u=data.u, d_lower=d_lower, d_upper=d_upper, e_lower=e_lower, e_upper=e_upper)
    model_file = f'{res_dir}/robustness_3_{str(factor).replace(".", "")}.pkl'
    results = {
            'weights' : f.get_weights(),
            'property': 'robustness',
            'factor': factor,
            'r2' : r2_score(y_true=y_test, y_pred=f(x_test).numpy().flatten())
            }
    pickle.dump(results, open(model_file, 'wb'))
#     os.system(f'rmdir {model_path}')
    shutil.rmtree(model_path)
