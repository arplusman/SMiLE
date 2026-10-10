import pickle
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
    'axes.labelsize': 30,     # X and Y axis labels
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 26,
    'figure.titlesize': 20    # Title of the whole figure
})



####################################################################################################
# Hyperparameters
####################################################################################################
# Meta
name = 'benchmarks/monotonicity'
res_dir = f'{name}/results_baseline'
agg_res_dir = f'{name}/aggregated_results_baseline'

# Tasks
n_samples = 1000
input_dim = 1

# # Properties
# alphas = [1.0, 3.0, 5.0]
# omegas = [0.2, 0.6, 1.]
# delta = 0.0
# epsilon = 0.0

# Architecture
hidden_dim = 8
aux_complexity = 32
h_architecture = [(16, 'relu'), (32, 'relu'), (64, 'relu'), (32, 'relu'), (16, 'relu'), (hidden_dim, 'linear')]
h_aux_architecture = [(aux_complexity, 'relu'), (hidden_dim, 'linear')]
g_architecture = [(1, 'linear')]


####################################################################################################
# Task -- Data
####################################################################################################
results = {}
for model in os.listdir(res_dir):
    if 'pkl' in model:
        result = pickle.load(open(f'{res_dir}/{model}', 'rb'))
        model = model.split('.pkl')[0]
        results[model] = {}

        if 'agnostic' in model:
            h_lower = None
            h_upper = None
        else:
            h_lower = BaseNet(h_aux_architecture)
            h_upper = BaseNet(h_aux_architecture)
        h = BaseNet(h_architecture)
        g = BaseNet(g_architecture)
        f = Net(model_path=None, mode='regression',
                h=h, h_lower=h_lower, h_upper=h_upper, g=g,
                binary_split=None, l=None, u=None, d_lower=None, d_upper=None, e_lower=None, e_upper=None,
                pretrain=False, generation_timelimit=None, generation_solvelimit=None, generation_frequency=None)

        task = lambda x: x + result['alpha']*np.sin(result['omega']*x)
        l, u = [-10], [10]
        data = Synthetic(n_samples=n_samples, input_dim=input_dim, l=l, u=u, func=task)
        x, y, x_train, y_train, x_test, y_test = data.generate(0, standardize=True)

        f(x_test)
        f.set_weights(result['weights'])

        results[model]['f'] = f
        results[model]['alpha'] = result['alpha']
        results[model]['omega'] = result['omega']
        results[model]['r2'] = result['r2']


####################################################################################################
# Aggregate
####################################################################################################
table = pd.DataFrame({
        'alpha' : [results[model]['alpha'] for model in results],
        'omega' : [results[model]['omega'] for model in results],
        'r2' : [results[model]['r2'] for model in results]
        })
table = table.sort_values(['alpha', 'omega'])
table = table.groupby(['alpha', 'omega'])['r2'].mean().reset_index()
print(table)

model = '20_06'
f = results[model]['f']
alpha = results[model]['alpha']
omega = results[model]['omega']
model = model.split('_')[0]

task = lambda x: x + alpha*np.sin(omega*x)
l, u = [-10], [10]
data = Synthetic(n_samples=n_samples, input_dim=input_dim, l=l, u=u, func=task)
x, y, x_train, y_train, x_test, y_test = data.generate(0, standardize=True)

plt.figure(figsize=(7,5))
plt.plot(x, y, label=f'task')
# plt.plot(x, f(x), label='SMiLE' if model=='smile' else 'Agnostic', color=palette[4])
plt.plot(x, f(x), label='SMiLE', color=palette[4])
plt.legend(loc='upper left')
plt.gcf().subplots_adjust(left=0.1, right=1, top=1, bottom=0.17)
plt.savefig(f'{agg_res_dir}/monotonicity_example.pdf')
plt.close()


# Heatmap 
alphas = sorted(table['alpha'].unique())
omegas = sorted(table['omega'].unique(), reverse=True)

heatmap_data = table.pivot_table(
    index='omega',
    columns='alpha',
    values='r2',
    fill_value=0  
).reindex(index=omegas, columns=alphas)
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="crest", cbar=False, annot_kws={"size": 28})
#plt.title("R² Heatmap")
plt.xlabel("alpha")
plt.ylabel("omega")
plt.tight_layout()
plt.savefig(f'{agg_res_dir}/monotonicity.pdf', bbox_inches='tight')