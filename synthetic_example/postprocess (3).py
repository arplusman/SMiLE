import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda
from core.model_baseline import *
from core.data import *
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette('dark')
palette = sns.color_palette('dark')
sns.set_theme()
sns.set_context("notebook", rc={
    'font.size': 12,          # Base font size
    'axes.titlesize': 24,     # Title of each subplot
    'axes.labelsize': 12,     # X and Y axis labels
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 22,
    'figure.titlesize': 20    # Title of the whole figure
})


####################################################################################################
# Hyperparameters
####################################################################################################
# Meta
name = 'synthetic_example'
res_dir = f'{name}/results'
agg_res_dir = f'{name}/aggregated_results'

# Tasks - Properties
n_samples = 1000
input_dim = 1
task_robustness = lambda x: x**3
task_monotonicity = lambda x: x + 1.5*np.tanh(x)*np.sin(x)

# Architecture
aux_complexity = 24
h_architecture = [(16, 'relu'), (32, 'relu'), (64, 'relu'), (32, 'relu'), (16, 'relu'), (1, 'linear')]
h_aux_architecture = [(aux_complexity, 'relu'), (1, 'linear')]
input_layer = Input(shape=(1,))
identity_output = Lambda(lambda x: x)(input_layer)
g = Model(inputs=input_layer, outputs=identity_output)


####################################################################################################
# Task -- Data
####################################################################################################
results = {}
for model in os.listdir(res_dir):
    if 'pkl' in model:
        result = pickle.load(open(f'{res_dir}/{model}', 'rb'))
        model = model.split('.pkl')[0]
        results[model] = {}

        h = BaseNet(h_architecture)
        h_lower = BaseNet(h_aux_architecture)
        h_upper = BaseNet(h_aux_architecture)
        f = Net(model_path=None, mode='regression',
                h=h, h_lower=h_lower, h_upper=h_upper, g=g,
                binary_split=None, l=None, u=None, d_lower=None, d_upper=None, e_lower=None, e_upper=None,
                pretrain=False, generation_timelimit=None, generation_solvelimit=None, generation_frequency=None)
        l, u = [-3], [3]
        if result['property'] == 'robustness':
            data = Synthetic(n_samples=n_samples, input_dim=input_dim, l=l, u=u, func=task_robustness)
        else:
            data = Synthetic(n_samples=n_samples, input_dim=input_dim, l=l, u=u, func=task_monotonicity)
        _, _, _, _, x_test, y_test = data.generate(0, standardize=True)
        f(x_test)
        f.set_weights(result['weights'])

        results[model]['f'] = f
        results[model]['property'] = result['property']
        results[model]['factor'] = result['factor']
        results[model]['r2'] = result['r2']


####################################################################################################
# Aggregate
####################################################################################################
table = pd.DataFrame({
        'model' : [model.split('_')[0] for model in results],
        'phase' : [model.split('_')[1] for model in results],
        'property' : [results[model]['property'] for model in results],
        'factor' : [results[model]['factor'] for model in results],
        'r2' : [results[model]['r2'] for model in results]
        })
#table = table.replace(np.nan, None)
table = table.sort_values(['property', 'model', 'phase', 'factor'])
print(table)
table = table.dropna()

prop = 'robustness'
factors = list(table.loc[table['property']==prop,'factor'])
l, u = [-3], [3]
if prop == 'robustness':
    data = Synthetic(n_samples=n_samples, input_dim=input_dim, l=l, u=u, func=task_robustness)
    xlim, ylim = 2.0, 3.0
else:
    data = Synthetic(n_samples=n_samples, input_dim=input_dim, l=l, u=u, func=task_monotonicity)
    xlim, ylim = 2.0, 2.5
x, y, _, _, _, _  = data.generate(0, standardize=True)

f0 = results[f'{prop}_0_None']['f']
f1 = results[f'{prop}_1_None']['f']
for factor in factors:
    f2 = results[f'{prop}_2_{str(factor).replace(".","")}']['f']
    f3 = results[f'{prop}_3_{str(factor).replace(".","")}']['f']

    linewidth = 1
    ###########
    # 2 rows
    ###########
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

    axes[0,0].plot(x, y, label='task', linewidth=linewidth)
    axes[0,0].plot(x, f1.h(x), label='h', linewidth=linewidth)
    axes[0,0].plot(x, f1.h_lower(x), label='h_lower', linewidth=linewidth, linestyle='--')
    axes[0,0].plot(x, f1.h_upper(x), label='h_upper', linewidth=linewidth, linestyle='--')
    axes[0,0].set_xlim(-xlim, xlim)
    axes[0,0].set_ylim(-ylim, ylim)
    axes[0,0].text(
        0.05, 0.95, 'Pretraining',                # x and y in axes fraction (0-1)
        transform=axes[0,0].transAxes,
        fontsize=24,                      # adjust size as needed
        verticalalignment='top',
#           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', edgecolor='gray')
    )

    axes[0,1].plot(x, y, label='task', linewidth=linewidth)
    axes[0,1].plot(x, f2.h(x), label='h', linewidth=linewidth)
    axes[0,1].plot(x, f2.h_lower(x), label='h_lower', linewidth=linewidth, linestyle='--')
    axes[0,1].plot(x, f2.h_upper(x), label='h_upper', linewidth=linewidth, linestyle='--')
    axes[0,1].set_xlim(-xlim, xlim)
    axes[0,1].set_ylim(-ylim, ylim)
    axes[0,1].text(
        0.05, 0.95, 'Training',                # x and y in axes fraction (0-1)
        transform=axes[0,1].transAxes,
        fontsize=24,                      # adjust size as needed
        verticalalignment='top',
#           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', edgecolor='gray')
    )

    axes[1,0].plot(x, y, label='task', linewidth=linewidth)
    axes[1,0].plot(x, f3.h(x), label='h', linewidth=linewidth)
    axes[1,0].plot(x, f3.h_lower(x), label='h_lower', linewidth=linewidth, linestyle='--')
    axes[1,0].plot(x, f3.h_upper(x), label='h_upper', linewidth=linewidth, linestyle='--')
    axes[1,0].set_xlim(-xlim, xlim)
    axes[1,0].set_ylim(-ylim, ylim)
    axes[1,0].text(
        0.05, 0.95, 'Posttraining',                # x and y in axes fraction (0-1)
        transform=axes[1,0].transAxes,
        fontsize=24,                      # adjust size as needed
        verticalalignment='top',
#           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', edgecolor='gray')
    )

    axes[1,1].plot(x, y, label='task', linewidth=linewidth)
    axes[1,1].plot(x, f3(x), label='f', linewidth=linewidth, color=palette[6])
    axes[1,1].set_xlim(-xlim, xlim)
    axes[1,1].set_ylim(-ylim, ylim)
    axes[1,1].text(
        0.05, 0.95, 'Final',                # x and y in axes fraction (0-1)
        transform=axes[1,1].transAxes,
        fontsize=24,                      # adjust size as needed
        verticalalignment='top',
#           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', edgecolor='gray')
    )

    handles1, labels1 = axes[0,0].get_legend_handles_labels()
    handles2, labels2 = axes[1,1].get_legend_handles_labels()
    filtered = [(h, l) for h, l in zip(handles2, labels2) if l != 'task']
    handles2, labels2 = zip(*filtered) if filtered else ([], [])
    all_handles = list(handles1) + list(handles2)
    all_labels = list(labels1) + list(labels2)

    for ax in axes.flatten():
        ax.tick_params(axis='both', which='both', length=0)  # hide tick marks
        ax.set_xticklabels([])  # hide tick labels
        ax.set_yticklabels([])
        ax.grid(True)

    fig.legend(all_handles, all_labels, loc='upper center',
               bbox_to_anchor=(0.512, 0.97),  # adjust vertical position if needed
               ncol=len(all_labels),
                handlelength=1.0,  # default is ~2.0; lower values = shorter lines
                handleheight=1.0,
#                  frameon=False
               )
    fig.subplots_adjust(wspace=0.03, hspace=0.03)

    fig.savefig(f'{agg_res_dir}/{prop}_{str(factor).replace(".", "")}.pdf', bbox_inches='tight')
    plt.close(fig)
