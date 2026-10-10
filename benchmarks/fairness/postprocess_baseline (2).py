import os
import json
import pickle
from core.model_baseline import *
from core.data import *
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_palette('deep')
sns.set_theme()
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
pd.set_option('display.max_rows', 500)
import matplotlib.cm as cm
import matplotlib.colors as colors

sns.set_palette('dark')
palette = sns.color_palette('dark')
sns.set_theme()
sns.set_context("notebook", rc={
    'font.size': 12,          # Base font size
    'axes.titlesize': 32,     # Title of each subplot
    'axes.labelsize': 32,     # X and Y axis labels
    'xtick.labelsize': 26,
    'ytick.labelsize': 26,
    'legend.fontsize': 24,
    'figure.titlesize': 20    # Title of the whole figure
})


####################################################################################################
# Hyperparameters
####################################################################################################
# Meta
name = 'benchmarks/fairness'
data_dir = f'{name}/data'
res_dir = f'{name}/results_baseline'
agg_res_dir = f'{name}/aggregated_results_baseline'

# Testing
termination = {
        2 : 'optimal',
        3 : 'infeasible',
        4 : 'inforunb',
        5 : 'unbounded',
        9 : 'timelimit',
        }

# Tasks
data = FairnessData()
benchmarks = ['compas', 'law', 'crime'] 

# Architecture
hidden_dim = 8
h_architecture = {
        'compas' : [(4*hidden_dim, 'relu'), (4*hidden_dim, 'relu'), (hidden_dim, 'linear')],
        'crime' : [(4*hidden_dim, 'relu'), (4*hidden_dim, 'relu'), (hidden_dim, 'linear')],
        'law' : [(4*hidden_dim, 'relu'), (4*hidden_dim, 'relu'), (hidden_dim, 'linear')],
        }
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
        'crime' : [0.08, 0.09, 0.10, 0.11, 0.11],
        'law' : [0.025, 0.050, 0.075, 0.100, 0.125],
        }


####################################################################################################
# Load
####################################################################################################
results = {}
for model in os.listdir(f'{res_dir}'):
    if 'pkl' in model:
        result = pickle.load(open(f'{res_dir}/{model}', 'rb'))
        model = model.split('.')[0]
        results[model] = {}

        if 'smle' in model:
            h_lower = BaseNet(h_aux_architecture)
            h_upper = BaseNet(h_aux_architecture)
            h = BaseNet(h_architecture[result['benchmark']])
            g = BaseNet(g_architecture)
        else:
            h_lower = None
            h_upper = None
            h = BaseNet(h_architecture[result['benchmark']])
            g = BaseNet(g_architecture)
        f = Net(model_path=None, mode='regression',
                h=h, h_lower=h_lower, h_upper=h_upper, g=g, 
                binary_split=None, l=None, u=None, d_lower=None, d_upper=None, e_lower=None, e_upper=None, 
                pretrain=False, generation_timelimit=None, generation_solvelimit=None, generation_frequency=None)
        input_dim, binary_split, l, u, _, _, x_test, y_test = data.load(benchmark=result['benchmark'])
        f(x_test)
        f.set_weights(result['weights'])

        results[model]['f'] = f
        results[model]['benchmark'] = result['benchmark']
        results[model]['delta'] = result['delta']
        results[model]['epsilon'] = result['epsilon']
        results[model]['lambda'] = result['lambda']
        results[model]['r2'] = result['r2']
        results[model]['mse'] = result['mse']
        results[model]['accuracy'] = result['accuracy']
        results[model]['counter_var'] = result['counter_var']
        results[model]['training_time'] = result['training_time']

        if '_smle_' in model:
            del result['weights']
            del result['benchmark']
            del result['delta']
            del result['epsilon']
            del result['lambda']
            del result['r2']
            del result['mse']
            del result['accuracy']
            del result['counter_var']
            del result['training_time']

            result = pd.DataFrame(result)
            result['exceeded_time'] = np.maximum(0., result['time'] - 3*result['timelimit']/2)
            result['termination'] = result['termination'].replace(termination)
            results[model]['generation'] = result
            results[model]['guarantee'] = result['termination'].iloc[-1] == 'inforunb'
            results[model]['training_time'] = results[model]['training_time'] - result['exceeded_time'].sum()
        else:
            results[model]['generation'] = None
            results[model]['guarantee'] = False
        

####################################################################################################
# Aggregate 
####################################################################################################
#####################
# Table
#####################
table = pd.DataFrame({
    'benchmark' : [results[model]['benchmark'] for model in results],
    'model' : [model.split('_')[1] for model in results],
    'delta' : [results[model]['delta'] for model in results],
    'epsilon' : [results[model]['epsilon'] for model in results],
    'lambda' : [results[model]['lambda'] for model in results],
    'guarantee' : [results[model]['guarantee'] for model in results],
    'r2' : [results[model]['r2'] for model in results],
#   'mse' : [results[model]['mse'] for model in results],
    'accuracy' : [results[model]['accuracy'] for model in results],
    'counter_var' : [results[model]['counter_var'] for model in results],
    'training_time' : [results[model]['training_time'] for model in results],
    })
table = table[table['model'] != 'presmle']
table = table.sort_values(['benchmark', 'model', 'delta', 'epsilon', 'lambda'], ascending = [True, True, True, False, True])


#####################
# Accuracy - Fairness Plots
#####################
fig, axs = plt.subplots(ncols=3, figsize=(18, 6), sharex=False, sharey=False)
handles, labels = None, None
for i, benchmark in enumerate(benchmarks):
    metric = 'r2' if benchmark == 'crime' else 'accuracy'
    subset = table[table['benchmark'] == benchmark]

    legend_option = 'brief' if i == 0 else False

    sns.scatterplot(
        data=subset,
        x='counter_var',
        y=metric,
        hue='model',
        style='model',
        # markers={'agnostic': 'o', 'smle': 'D', 'certifair': 's'},
        markers={'agnostic': 'o', 'smle': 'D'},
        s=200,
        ax=axs[i],
        legend=legend_option
    )

    if i == 0:
        handles, labels = axs[i].get_legend_handles_labels()
        label_map = {
            'agnostic': 'Agnostic',
            # 'certifair': 'CertiFair',
            'smle': 'SMiLE'
        }
        labels = [label_map.get(label, label) for label in labels]
        axs[i].legend_.remove()

    axs[i].set_title(benchmark.capitalize())

axs[0].set_xlabel('')
axs[1].set_xlabel('Counterfactual Variation')
axs[2].set_xlabel('')
axs[0].set_ylabel('Predictive Quality')
axs[1].set_ylabel('')
axs[2].set_ylabel('')


fig.legend(
    handles,
    labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.),
    ncol=len(labels),
    frameon=True
)

fig.subplots_adjust(wspace=0.22, top=0.80)

plt.savefig(f'{agg_res_dir}/fairness.pdf', bbox_inches='tight')
plt.close()


#####################
# Sample-Wise Fairness
#####################
features = {
    'compas': ['age', 'priors_count'], 
    'law': ['lsat', 'gpa'],
    'crime': ['population', 'householdsize'], 
    }
counter_variations = {}
bin_counter_variations = {}
threshold = {'compas': 0., 'law': 0., 'crime': 0.5} 


selected_epsilons = {
        'compas' : ['02', '06', '10'],
        'crime' : ['002', '006', '01'],
        'law' : ['10', '30', '50'],
        }

epsilons = {
        'compas' : [0.2, 0.6, 1.0],
        'crime' : [0.02, 0.06, 0.1],
        'law' : [1.0, 3.0, 5.0],
        }

for benchmark in benchmarks:
    _, _, _, _, _, _, x_test, y_test = data.load(benchmark=benchmark)
    x_test_counter = x_test.copy()
    x_test_counter.iloc[:,-1] = 1-x_test_counter.iloc[:,-1]
    counter_variations[benchmark] = x_test[features[benchmark]]
    bin_counter_variations[benchmark] = x_test[features[benchmark]]

    for model in results:
        if results[model]['benchmark'] == benchmark:
            f = results[model]['f']
            model = model.split('_')[1:]
            model = '_'.join(model)

            y_test_pred = f(x_test)
            y_test_pred_counter = f(x_test_counter)
            counter_variations[benchmark][model] = np.abs(y_test_pred - y_test_pred_counter).flatten()

            y_test_pred = y_test_pred <= threshold[benchmark]
            y_test_pred_counter = f(x_test_counter) <= threshold[benchmark]
            bin_counter_variations[benchmark][model] = y_test_pred == y_test_pred_counter


benchmark = 'crime'
counter_variations[benchmark] = counter_variations[benchmark][features[benchmark] + [f'smle_00_{epsilon}' for epsilon in selected_epsilons[benchmark]]]
min_var = min(counter_variations[benchmark][[col for col in counter_variations[benchmark].columns if col not in features[benchmark]]].min())
max_var = max(counter_variations[benchmark][[col for col in counter_variations[benchmark].columns if col not in features[benchmark]]].max())
norm = plt.Normalize(min_var, max_var)
cmap = sns.color_palette("crest", as_cmap=True)

#SMLE
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  
for i, epsilon in enumerate(epsilons[benchmark][::-1]):
    colors = cmap(norm(counter_variations[benchmark][f'smle_00_{str(epsilon).replace(".","")}']))
    axes[i].scatter(
            counter_variations[benchmark][features[benchmark][0]],
            counter_variations[benchmark][features[benchmark][1]],
            c=colors,
            s=40,
            edgecolor='k',
            linewidth=0.5,
            alpha=0.8
        )
    axes[i].set_title(f'eps = {epsilon}')

axes[1].set_xlabel(features[benchmark][0])
axes[0].set_ylabel(features[benchmark][1])

cbar_ax = fig.add_axes([0.25, 0.95, 0.5, 0.05])  # [left, bottom, width, height] in figure coords
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.ax.xaxis.set_ticks_position('top')


# Adjust layout to make space for the colorbar
fig.subplots_adjust(bottom=0.18, top=0.85, wspace=0.03)

fig.savefig(f'{agg_res_dir}/{benchmark}_smle_counter_var.pdf', bbox_inches='tight')
plt.close()
#plt.show()
