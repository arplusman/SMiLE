"""
benchmarks/fairness/postprocess_ablation.py

Aggregates pretrain loss ablation results for the FAIRNESS benchmark.
Compares three pretrain_mode variants across compas / crime / law on:
  - Predictive quality (R² / accuracy) vs epsilon
  - Counterfactual variation (enforcement effectiveness) vs epsilon
  - Solver: timeout rate, duality gap, training time

Run via:
    python run_benchmark.py --benchmark fairness --analysis ablation
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme()
sns.set_palette('dark')
palette = sns.color_palette('dark')
sns.set_context("notebook", rc={
    'axes.titlesize': 20, 'axes.labelsize': 18,
    'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 12,
})

####################################################################################################
# Config
####################################################################################################
name    = 'benchmarks/fairness'
res_dir = f'{name}/results_ablation'
out_dir = f'{name}/aggregated_results_ablation'
Path(out_dir).mkdir(parents=True, exist_ok=True)

PRETRAIN_MODES = ['full', 'accuracy', 'none']
MODE_LABELS    = {
    'full':     'Full pretrain\n(all pathways + box)',
    'accuracy': 'Accuracy only\n(main pathway)',
    'none':     'No pretrain',
}
MODE_COLOURS   = {
    'full':     palette[0],
    'accuracy': palette[1],
    'none':     palette[2],
}
TERMINATION = {2:'optimal', 3:'infeasible', 4:'inforunb', 5:'unbounded', 9:'timelimit'}
EPS         = 1e-8
benchmarks  = ['compas', 'law', 'crime']
metric_col  = {'compas': 'accuracy', 'law': 'r2', 'crime': 'r2'}


####################################################################################################
# Load
####################################################################################################
rows_perf, rows_solver = [], []

for fname in os.listdir(res_dir):
    if not fname.endswith('.pkl') or '_smle_' not in fname:
        continue
    result = pickle.load(open(f'{res_dir}/{fname}', 'rb'))

    rows_perf.append({
        'benchmark':     result['benchmark'],
        'pretrain_mode': result['pretrain_mode'],
        'epsilon':       result['epsilon'],
        'r2':            result['r2'],
        'accuracy':      result['accuracy'],
        'counter_var':   result['counter_var'],
        'training_time': result['training_time'],
    })

    for i, (t, tl, p, d, term) in enumerate(zip(
            result['time'], result['timelimit'], result['primal'],
            result['dual'], result['termination'])):
        gap = abs(d - p) / max(abs(p), EPS) if p is not None else np.nan
        rows_solver.append({
            'benchmark':     result['benchmark'],
            'pretrain_mode': result['pretrain_mode'],
            'epsilon':       result['epsilon'],
            'call_idx':      i,
            'termination':   TERMINATION.get(term, str(term)),
            'time':          t,
            'timelimit':     tl,
            'duality_gap':   gap,
            'timed_out':     TERMINATION.get(term, '') == 'timelimit',
        })

if not rows_perf:
    raise RuntimeError(f'No ablation results in {res_dir}. '
                       'Run process_ablation.py first.')

perf   = pd.DataFrame(rows_perf)
solver = pd.DataFrame(rows_solver)
print(f'[INFO] Loaded {len(perf)} experiments, {len(solver)} solver calls.')
ncols = len(benchmarks)
x     = np.arange(len(PRETRAIN_MODES))


####################################################################################################
# Tables
####################################################################################################
print('\n' + '='*70)
print('ABLATION — PERFORMANCE TABLE')
print('='*70)
for benchmark in benchmarks:
    m   = metric_col[benchmark]
    sub = perf[perf['benchmark'] == benchmark]
    tbl = (sub.groupby('pretrain_mode')
              .agg(**{f'mean_{m}': (m, 'mean'),
                      'mean_counter_var': ('counter_var', 'mean'),
                      'mean_training_time': ('training_time', 'mean')})
              .round(4))
    print(f'\n  {benchmark.upper()}')
    print(tbl.to_string())
tbl_all = perf.groupby(['benchmark','pretrain_mode']).agg(
    r2=('r2','mean'), accuracy=('accuracy','mean'),
    counter_var=('counter_var','mean'), training_time=('training_time','mean')
).round(4).reset_index()
tbl_all.to_csv(f'{out_dir}/ablation_performance.csv', index=False)

print('\n' + '='*70)
print('ABLATION — SOLVER METRICS TABLE')
print('='*70)
solver_tbl = (
    solver.groupby(['benchmark','pretrain_mode'])
    .agg(timeout_rate=('timed_out','mean'), mean_time=('time','mean'),
         median_gap=('duality_gap','median'))
    .round(4).reset_index()
)
solver_tbl['timeout_pct'] = (solver_tbl['timeout_rate']*100).round(1)
print(solver_tbl.to_string(index=False))
solver_tbl.to_csv(f'{out_dir}/ablation_solver.csv', index=False)


####################################################################################################
# Plot 1 — Predictive quality vs epsilon
####################################################################################################
fig, axs = plt.subplots(1, ncols, figsize=(6*ncols, 5))
for ax, benchmark in zip(axs, benchmarks):
    m   = metric_col[benchmark]
    sub = perf[perf['benchmark'] == benchmark]
    for mode in PRETRAIN_MODES:
        ms = sub[sub['pretrain_mode']==mode].sort_values('epsilon')
        ax.plot(ms['epsilon'], ms[m], marker='o',
                label=MODE_LABELS[mode], color=MODE_COLOURS[mode])
    ax.set_title(benchmark.capitalize())
    ax.set_xlabel('epsilon')
    ax.set_ylabel(m.upper() if benchmark == benchmarks[0] else '')
    if benchmark == benchmarks[0]:
        ax.legend(fontsize=11)

fig.suptitle('Predictive Quality vs. Epsilon — Fairness Ablation', y=1.02)
fig.tight_layout()
fig.savefig(f'{out_dir}/quality_vs_epsilon.pdf', bbox_inches='tight')
plt.close()
print('[INFO] Saved quality_vs_epsilon.pdf')


####################################################################################################
# Plot 2 — Counterfactual variation vs epsilon
####################################################################################################
fig, axs = plt.subplots(1, ncols, figsize=(6*ncols, 5))
for ax, benchmark in zip(axs, benchmarks):
    sub = perf[perf['benchmark'] == benchmark]
    for mode in PRETRAIN_MODES:
        ms = sub[sub['pretrain_mode']==mode].sort_values('epsilon')
        ax.plot(ms['epsilon'], ms['counter_var'], marker='o',
                label=MODE_LABELS[mode], color=MODE_COLOURS[mode])
    ax.set_title(benchmark.capitalize())
    ax.set_xlabel('epsilon')
    ax.set_ylabel('Max counterfactual variation'
                  if benchmark == benchmarks[0] else '')
    if benchmark == benchmarks[0]:
        ax.legend(fontsize=11)

fig.suptitle('Enforcement Effectiveness vs. Epsilon — Fairness Ablation', y=1.02)
fig.tight_layout()
fig.savefig(f'{out_dir}/counter_var_vs_epsilon.pdf', bbox_inches='tight')
plt.close()
print('[INFO] Saved counter_var_vs_epsilon.pdf')


####################################################################################################
# Plot 3 — Timeout rate per mode
####################################################################################################
fig, axs = plt.subplots(1, ncols, figsize=(6*ncols, 5), sharey=True)
for ax, benchmark in zip(axs, benchmarks):
    sub  = solver_tbl[solver_tbl['benchmark'] == benchmark]
    vals = [sub.loc[sub['pretrain_mode']==m, 'timeout_pct'].values[0]
            if m in sub['pretrain_mode'].values else 0. for m in PRETRAIN_MODES]
    bars = ax.bar(x, vals,
                  color=[MODE_COLOURS[m] for m in PRETRAIN_MODES],
                  edgecolor='k', linewidth=0.7, width=0.5)
    ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m] for m in PRETRAIN_MODES],
                       rotation=10, ha='right')
    ax.set_title(benchmark.capitalize())
    ax.set_ylabel('Timeout rate (%)' if benchmark == benchmarks[0] else '')
    ax.set_ylim(0, max(max(vals)*1.3, 5))

fig.suptitle('Timeout Rate by Pretrain Mode — Fairness Ablation', y=1.02)
fig.tight_layout()
fig.savefig(f'{out_dir}/timeout_rate.pdf', bbox_inches='tight')
plt.close()
print('[INFO] Saved timeout_rate.pdf')


####################################################################################################
# Plot 4 — Duality gap distributions
####################################################################################################
fig, axs = plt.subplots(1, ncols, figsize=(6*ncols, 5))
for ax, benchmark in zip(axs, benchmarks):
    sub = solver[(solver['benchmark']==benchmark) & solver['duality_gap'].notna()]
    sns.boxplot(data=sub, x='pretrain_mode', y='duality_gap',
                order=PRETRAIN_MODES,
                palette=[MODE_COLOURS[m] for m in PRETRAIN_MODES],
                width=0.5, fliersize=3, ax=ax)
    ax.set_yscale('log')
    ax.set_xticklabels([MODE_LABELS[m] for m in PRETRAIN_MODES],
                       rotation=10, ha='right')
    ax.set_title(benchmark.capitalize())
    ax.set_xlabel('')
    ax.set_ylabel('Duality gap (log)' if benchmark == benchmarks[0] else '')

fig.suptitle('LP Relaxation Duality Gap by Pretrain Mode — Fairness Ablation\n'
             '(lower = tighter relaxation)', y=1.02)
fig.tight_layout()
fig.savefig(f'{out_dir}/duality_gap.pdf', bbox_inches='tight')
plt.close()
print('[INFO] Saved duality_gap.pdf')


####################################################################################################
# Plot 5 — Training time
####################################################################################################
fig, axs = plt.subplots(1, ncols, figsize=(6*ncols, 5))
for ax, benchmark in zip(axs, benchmarks):
    sub   = perf[perf['benchmark'] == benchmark]
    means = [sub.loc[sub['pretrain_mode']==m, 'training_time'].mean()
             for m in PRETRAIN_MODES]
    bars  = ax.bar(x, means,
                   color=[MODE_COLOURS[m] for m in PRETRAIN_MODES],
                   edgecolor='k', linewidth=0.7, width=0.5)
    ax.bar_label(bars, fmt='%.0fs', padding=3, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m] for m in PRETRAIN_MODES],
                       rotation=10, ha='right')
    ax.set_title(benchmark.capitalize())
    ax.set_ylabel('Mean training time (s)' if benchmark == benchmarks[0] else '')

fig.suptitle('Mean Training Time by Pretrain Mode — Fairness Ablation', y=1.02)
fig.tight_layout()
fig.savefig(f'{out_dir}/training_time.pdf', bbox_inches='tight')
plt.close()
print('[INFO] Saved training_time.pdf')

print(f'\n[INFO] All outputs saved to {out_dir}/')