import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.datasets import mnist


class FairnessData:
    def _law(self):
        data = pd.read_stata('benchmarks/fairness/data/law.dta')
        data = data.replace('', np.nan)
        data = data.drop('race',axis = 1)
        data = data.rename(columns={'white': 'race'})
        data = data.drop(['enroll', 'asian', 'black', 'hispanic', 'missingrace', 'urm'], axis=1)
        data = pd.get_dummies(data, columns=['college', 'year'], dtype=int, drop_first=True)

        numerics = ['lsat', 'gpa']
        binaries = [col for col in data.columns if 'college' in col or 'year' in col] + ['resident', 'gender', 'race']
        binary_split = len(numerics)
        target = 'admit'

        return data, numerics, binaries, binary_split, target

    def _crime(self):
        data = pd.read_csv('benchmarks/fairness/data/crime.csv')
        data = data.drop(columns=['state', 'county', 'community', 'communityname', 'fold'])
        data = data.replace('?', np.nan)
        data = data.dropna(axis=1)
        data['black'] = (data['racepctblack'] >= 0.06).astype(int)

        numerics = [col for col in data.columns if col not in ['black', 'ViolentCrimesPerPop']]
        binaries = ['black']
        binary_split = len(numerics)
        target = 'ViolentCrimesPerPop'

        return data, numerics, binaries, binary_split, target
    
    def _compas(self):
        data = pd.read_csv('benchmarks/fairness/data/compas.csv')
        data = data[
            (data['days_b_screening_arrest'] <= 30) &
            (data['days_b_screening_arrest'] >= -30) &
            (data['is_recid'] != -1) &
            (data['c_charge_degree'] != 'O') &
            (data['score_text'] != 'N/A')
            ]
        data['c_charge_degree'] = (data['c_charge_degree'] == 'M').astype(int)
        data['in_custody'] = pd.to_datetime(data['in_custody'])
        data['out_custody'] = pd.to_datetime(data['out_custody'])
        data['custody_time'] = (data['out_custody'] - data['in_custody']).dt.total_seconds()
        data['c_jail_in'] = pd.to_datetime(data['c_jail_in'])
        data['c_jail_out'] = pd.to_datetime(data['c_jail_out'])
        data['jail_time'] = (data['c_jail_out'] - data['c_jail_in']).dt.total_seconds()
        data = data[data['race'].isin(['African-American', 'Caucasian'])]
        data['race'] = (data['race'] == 'Caucasian').astype(int)
        data['sex'] = (data['sex'] == 'Female').astype(int)
        data = pd.get_dummies(data, columns=['v_score_text'], dtype=int, drop_first=True)
        data['two_year_recid'] = 1 - data['two_year_recid'] # 1 == not recidiv 

        numerics = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'custody_time', 'jail_time']
        binaries = [col for col in data.columns if 'v_score_text' in col] + ['c_charge_degree', 'sex', 'race']
        binary_split = len(numerics)
        target = 'two_year_recid'

        return data, numerics, binaries, binary_split, target

    def load(self, benchmark):
        if benchmark=='law':
            data, numerics, binaries, binary_split, target = self._law()
        elif benchmark=='crime':
            data, numerics, binaries, binary_split, target = self._crime()
        elif benchmark=='compas':
            data, numerics, binaries, binary_split, target = self._compas()

        features = numerics + binaries
        input_dim = len(features)
        data = data[features + [target]]
        data = data.dropna()

        data = (data - data.min())/(data.max() - data.min())
        l, u = list(data[features].min()), list(data[features].max())

        train_set, test_set = train_test_split(data, test_size=0.3, random_state=3)
        x_train, y_train = train_set[features], train_set[target]
        x_test, y_test = test_set[features], test_set[target]

        return input_dim, binary_split, l, u, x_train, y_train, x_test, y_test


class MNIST:
    def load(self, binarize=None):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype("float32")/255.0
        x_test = x_test.astype("float32")/255.0
        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]

        input_dim = 28*28
        l, u = [0.]*input_dim, [1.]*input_dim
        
        if binarize == 'zero-nonzero':
            y_train, y_test = y_train == 0, y_test == 0
            y_train, y_test = y_train.astype(int), y_test.astype(int)
        elif binarize == 'even-odd':
            y_train, y_test = y_train % 2, y_test % 2
        elif binarize == 'small-large':
            y_train, y_test = y_train >= 5, y_test >= 5
        y_train, y_test = y_train.astype(int), y_test.astype(int)
        
        # For computational reasons, we want to reduce the the evaluation to a subset of the test set
        x_test, y_test = x_test[:1000], y_test[:1000] 

        return l, u, x_train, y_train, x_test, y_test


class Synthetic:
    def __init__(self, n_samples, input_dim, l, u, func):
        self.n_samples = n_samples
        self.input_dim = input_dim
        self.l = l
        self.u = u
        self.func = func

    def generate(self, seed, standardize, split_perc=0.8, noise_type=None, noise=None):
        np.random.seed(seed)
        x = np.random.uniform(low=self.l, high=self.u, size=(self.n_samples, self.input_dim))
        y = np.apply_along_axis(arr=x, func1d=self.func, axis=1).reshape(self.n_samples, 1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=split_perc, random_state=seed)

        if noise_type == 'gaussian':
            noise = np.random.normal(0, np.abs(np.max(y_train) - np.min(y_train))*noise, y_train.shape)
        elif noise_type == 'cauchy':
            noise = np.abs(np.max(y_train) - np.min(y_train)) * noise
            noise = np.random.standard_cauchy(size=y_train.shape) * noise
        else:
            noise = 0.
        y_train = y_train + noise

        x = np.linspace(self.l, self.u, 1000)
        y = np.apply_along_axis(arr=x, func1d=self.func, axis=1).reshape(self.n_samples, 1)

        if standardize:
            x_mean = np.mean(x_train, axis=0)
            x_std = np.std(x_train, axis=0)
            y_mean = np.mean(y_train, axis=0)
            y_std = np.std(y_train, axis=0)

            x = (x - x_mean)/x_std
            y= (y - y_mean)/y_std
            x_train = (x_train - x_mean)/x_std
            x_test = (x_test - x_mean)/x_std
            y_train = (y_train - y_mean)/y_std
            y_test = (y_test - y_mean)/y_std

            self.l = (self.l - x_mean)/x_std
            self.u = (self.u - x_mean)/x_std

        return x, y, x_train, y_train, x_test, y_test
