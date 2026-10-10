"""
benchmarks/monotonicity/postprocess_ablation.py

Aggregates pretrain loss ablation results for the MONOTONICITY benchmark.
Compares three pretrain_mode variants across all (alpha, omega) configurations on:
  - R² quality vs omega (for each alpha)
  - Solver: timeout rate, duality gap, training time

Run via:
    python run_benchmark.py --benchmark monotonicity --analysis ablation
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
name    = 'benchmarks/monotonicity'
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


####################################################################################################
# Load
####################################################################################################
rows_perf, rows_solver = [], []

for fname in os.listdir(res_dir):
    # Ablation files are named {alpha}_{omega}_{pretrain_mode}.pkl
    if not fname.endswith('.pkl'):
        continue
    result = pickle.load(open(f'{res_dir}/{fname}', 'rb'))

    rows_perf.append({
        'alpha':         result['alpha'],
        'omega':         result['omega'],
        'pretrain_mode': result['pretrain_mode'],
        'r2':            result['r2'],
        'training_time': result['training_time'],
    })

    for i, (t, tl, p, d, term) in enumerate(zip(
            result['time'], result['timelimit'], result['primal'],
            result['dual'], result['termination'])):
        gap = abs(d - p) / max(abs(p), EPS) if p is not None else np.nan
        rows_solver.append({
            'alpha':         result['alpha'],
            'omega':         result['omega'],
            'pretrain_mode': result['pretrain_mode'],
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
alphas = sorted(perf['alpha'].unique())
omegas = sorted(perf['omega'].unique())
x      = np.arange(len(PRETRAIN_MODES))
print(f'[INFO] Loaded {len(perf)} experiments, {len(solver)} solver calls.')


####################################################################################################
# Tables
####################################################################################################
print('\n' + '='*70)
print('ABLATION — R² TABLE')
print('='*70)
r2_tbl = (
    perf.groupby(['alpha', 'omega', 'pretrain_mode'])['r2']
    .mean().unstack('pretrain_mode').round(4)
)
print(r2_tbl.to_string())
r2_tbl.to_csv(f'{out_dir}/ablation_r2.csv')

solver_tbl = (
    solver.groupby(['alpha', 'omega', 'pretrain_mode'])
    .agg(timeout_rate=('timed_out','mean'), median_gap=('duality_gap','median'))
    .round(4).reset_index()
)
solver_tbl['timeout_pct'] = (solver_tbl['timeout_rate']*100).round(1)
solver_tbl.to_csv(f'{out_dir}/ablation_solver.csv', index=False)


####################################################################################################
# Plot 1 — R² vs omega, one panel per alpha, lines per mode
####################################################################################################
fig, axs = plt.subplots(1, len(alphas), figsize=(6*len(alphas), 5), sharey=False)
for ax, alpha in zip(axs, alphas):
    sub = perf[perf['alpha'] == alpha]
    for mode in PRETRAIN_MODES:
        ms = sub[sub['pretrain_mode']==mode].sort_values('omega')
        ax.plot(ms['omega'], ms['r2'], marker='o',
                label=MODE_LABELS[mode], color=MODE_COLOURS[mode])
    ax.set_title(f'α={alpha}')
    ax.set_xlabel('omega')
    ax.set_ylabel('R²' if alpha == alphas[0] else '')
    if alpha == alphas[0]:
        ax.legend(fontsize=11)

fig.suptitle('R² vs. Omega by Pretrain Mode — Monotonicity Ablation', y=1.02)
fig.tight_layout()
fig.savefig(f'{out_dir}/r2_vs_omega.pdf', bbox_inches='tight')
plt.close()
print('[INFO] Saved r2_vs_omega.pdf')


####################################################################################################
# Plot 2 — Timeout rate, grouped bars per mode, one panel per alpha
####################################################################################################
fig, axs = plt.subplots(1, len(alphas), figsize=(6*len(alphas), 5), sharey=True)
for ax, alpha in zip(axs, alphas):
    sub   = solver_tbl[solver_tbl['alpha'] == alpha]
    # Average timeout across omegas for each mode
    means = [sub.loc[sub['pretrain_mode']==m, 'timeout_pct'].mean()
             if m in sub['pretrain_mode'].values else 0. for m in PRETRAIN_MODES]
    bars  = ax.bar(x, means,
                   color=[MODE_COLOURS[m] for m in PRETRAIN_MODES],
                   edgecolor='k', linewidth=0.7, width=0.5)
    ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m] for m in PRETRAIN_MODES],
                       rotation=10, ha='right')
    ax.set_title(f'α={alpha}')
    ax.set_ylabel('Mean timeout rate (%)' if alpha == alphas[0] else '')
    ax.set_ylim(0, max(max(means)*1.35, 5))

fig.suptitle('Timeout Rate by Pretrain Mode — Monotonicity Ablation\n'
             '(mean across omegas)', y=1.02)
fig.tight_layout()
fig.savefig(f'{out_dir}/timeout_rate.pdf', bbox_inches='tight')
plt.close()
print('[INFO] Saved timeout_rate.pdf')


####################################################################################################
# Plot 3 — Duality gap distributions per mode (pooled across all tasks)
####################################################################################################
fig, ax = plt.subplots(figsize=(7, 5))
sub = solver[solver['duality_gap'].notna()]
sns.boxplot(data=sub, x='pretrain_mode', y='duality_gap', order=PRETRAIN_MODES,
            palette=[MODE_COLOURS[m] for m in PRETRAIN_MODES],
            width=0.5, fliersize=3, ax=ax)
ax.set_yscale('log')
ax.set_xticklabels([MODE_LABELS[m] for m in PRETRAIN_MODES],
                   rotation=10, ha='right')
ax.set_xlabel('')
ax.set_ylabel('Duality gap (log)')
ax.set_title('LP Relaxation Duality Gap by Pretrain Mode\n'
             'Monotonicity Ablation (all tasks pooled)')
fig.tight_layout()
fig.savefig(f'{out_dir}/duality_gap.pdf', bbox_inches='tight')
plt.close()
print('[INFO] Saved duality_gap.pdf')


####################################################################################################
# Plot 4 — Training time
####################################################################################################
fig, axs = plt.subplots(1, len(alphas), figsize=(6*len(alphas), 5))
for ax, alpha in zip(axs, alphas):
    sub   = perf[perf['alpha'] == alpha]
    means = [sub.loc[sub['pretrain_mode']==m, 'training_time'].mean()
             for m in PRETRAIN_MODES]
    bars  = ax.bar(x, means,
                   color=[MODE_COLOURS[m] for m in PRETRAIN_MODES],
                   edgecolor='k', linewidth=0.7, width=0.5)
    ax.bar_label(bars, fmt='%.0fs', padding=3, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m] for m in PRETRAIN_MODES],
                       rotation=10, ha='right')
    ax.set_title(f'α={alpha}')
    ax.set_ylabel('Mean training time (s)' if alpha == alphas[0] else '')

fig.suptitle('Mean Training Time by Pretrain Mode — Monotonicity Ablation', y=1.02)
fig.tight_layout()
fig.savefig(f'{out_dir}/training_time.pdf', bbox_inches='tight')
plt.close()
print('[INFO] Saved training_time.pdf')

print(f'\n[INFO] All outputs saved to {out_dir}/')