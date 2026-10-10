"""
benchmarks/fairness/postprocess_new.py
========================================
Loads OLD and NEW SMiLE results from results_new/ and produces
comparison plots and tables.

Distinguishes the two approaches via the 'approach' key stored in each
pkl ('old' or 'new') — no second directory needed.

Produces
--------
  fairness_comparison.pdf   — quality vs counterfactual variation scatter
  quality_vs_epsilon.pdf    — predictive quality trend across epsilons
  counter_var_vs_epsilon.pdf— enforcement effectiveness across epsilons
  quality_delta.pdf         — per-epsilon quality gain (new − old)
  crime_counter_var.pdf     — sample-wise counterfactual variation heatmap
  r2_table.csv
  guarantee_table.csv

Run via:
    python run_benchmark.py --benchmark fairness
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_palette("dark")
palette = sns.color_palette("dark")
sns.set_theme()
sns.set_context("notebook", rc={
    "font.size": 12,
    "axes.titlesize": 32,
    "axes.labelsize": 32,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
    "legend.fontsize": 24,
})

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
name        = "benchmarks/fairness"
res_dir     = f"{name}/results_new"
agg_res_dir = f"{name}/aggregated_results_new"
Path(agg_res_dir).mkdir(parents=True, exist_ok=True)

TERMINATION = {2: "optimal", 3: "infeasible", 4: "inforunb",
               5: "unbounded", 9: "timelimit"}

benchmarks = ["compas", "law", "crime"]
metric_col = {"compas": "accuracy", "law": "accuracy", "crime": "r2"}

APPROACH_LABELS  = {"old": "Old SMiLE\n(indep. bounds)",
                    "new": "New SMiLE\n(base + delta)"}
APPROACH_COLOURS = {"old": palette[4], "new": palette[1]}
APPROACH_MARKERS = {"old": "D", "new": "o"}


# ---------------------------------------------------------------------------
# Load — metrics only, no Net reconstruction needed
# ---------------------------------------------------------------------------
rows = []
for fname in sorted(os.listdir(res_dir)):
    if not fname.endswith(".pkl") or "_smle_" not in fname:
        continue
    result = pickle.load(open(f"{res_dir}/{fname}", "rb"))
    if "approach" not in result:
        continue

    # derive guarantee from last termination code
    term_last = (result.get("termination") or [None])[-1]
    guarantee = TERMINATION.get(term_last, "") == "inforunb"

    rows.append({
        "approach":      result["approach"],
        "benchmark":     result["benchmark"],
        "epsilon":       result["epsilon"],
        "r2":            result["r2"],
        "accuracy":      result["accuracy"],
        "counter_var":   result["counter_var"],
        "guarantee":     guarantee,
        "training_time": result["training_time"],
    })

if not rows:
    raise RuntimeError(f"No _smle_ results found in {res_dir}. "
                       "Run process_new.py first.")

table = pd.DataFrame(rows).sort_values(["approach", "benchmark", "epsilon"])

# ---- tables ----------------------------------------------------------------
table.drop(columns=["r2", "accuracy"]).to_csv(
    f"{agg_res_dir}/r2_table.csv", index=False)

guarantee_tbl = (
    table.groupby(["approach", "benchmark"])["guarantee"]
    .mean().mul(100).round(1)
    .reset_index()
    .rename(columns={"guarantee": "guarantee_rate_%"})
)
guarantee_tbl.to_csv(f"{agg_res_dir}/guarantee_table.csv", index=False)

print("\n=== Old vs New SMiLE — Fairness Benchmark ===")
print(guarantee_tbl.to_string(index=False))


# ---------------------------------------------------------------------------
# Plot 1 — Quality vs Counterfactual Variation scatter
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, len(benchmarks),
                        figsize=(7 * len(benchmarks), 6),
                        sharex=False, sharey=False)

for i, benchmark in enumerate(benchmarks):
    metric = metric_col[benchmark]
    sub    = table[table["benchmark"] == benchmark]
    ax     = axs[i]

    for approach in ("old", "new"):
        ap = sub[sub["approach"] == approach]
        ax.scatter(ap["counter_var"], ap[metric],
                   label=APPROACH_LABELS[approach].replace("\n", " "),
                   color=APPROACH_COLOURS[approach],
                   marker=APPROACH_MARKERS[approach],
                   s=200, zorder=3)

    ax.set_title(benchmark.capitalize())
    ax.set_xlabel("Counterfactual Variation" if i == 1 else "")
    ax.set_ylabel("Predictive Quality" if i == 0 else "")

handles = [plt.scatter([], [], color=APPROACH_COLOURS[a],
                       marker=APPROACH_MARKERS[a], s=150,
                       label=APPROACH_LABELS[a].replace("\n", " "))
           for a in ("old", "new")]
fig.legend(handles=handles, loc="upper center",
           bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=True)
fig.subplots_adjust(wspace=0.22, top=0.82)
fig.savefig(f"{agg_res_dir}/fairness_comparison.pdf", bbox_inches="tight")
plt.close()
print("\n[INFO] Saved fairness_comparison.pdf")


# ---------------------------------------------------------------------------
# Plot 2 — Quality vs epsilon (line per approach)
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, len(benchmarks),
                        figsize=(6 * len(benchmarks), 5))
for i, benchmark in enumerate(benchmarks):
    metric = metric_col[benchmark]
    sub    = table[table["benchmark"] == benchmark]
    ax     = axs[i]
    for approach in ("old", "new"):
        ap = sub[sub["approach"] == approach].sort_values("epsilon")
        ax.plot(ap["epsilon"], ap[metric],
                marker="o",
                label=APPROACH_LABELS[approach].replace("\n", " "),
                color=APPROACH_COLOURS[approach])
    ax.set_title(benchmark.capitalize())
    ax.set_xlabel("epsilon")
    ax.set_ylabel(metric.upper() if i == 0 else "")
    if i == 0:
        ax.legend(fontsize=16)

fig.suptitle("Predictive Quality vs. Epsilon — Old vs New SMiLE", y=1.02)
fig.tight_layout()
fig.savefig(f"{agg_res_dir}/quality_vs_epsilon.pdf", bbox_inches="tight")
plt.close()
print("[INFO] Saved quality_vs_epsilon.pdf")


# ---------------------------------------------------------------------------
# Plot 3 — Counter_var vs epsilon
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, len(benchmarks),
                        figsize=(6 * len(benchmarks), 5))
for i, benchmark in enumerate(benchmarks):
    sub = table[table["benchmark"] == benchmark]
    ax  = axs[i]
    for approach in ("old", "new"):
        ap = sub[sub["approach"] == approach].sort_values("epsilon")
        ax.plot(ap["epsilon"], ap["counter_var"],
                marker="o",
                label=APPROACH_LABELS[approach].replace("\n", " "),
                color=APPROACH_COLOURS[approach])
    ax.set_title(benchmark.capitalize())
    ax.set_xlabel("epsilon")
    ax.set_ylabel("Max counterfactual variation" if i == 0 else "")
    if i == 0:
        ax.legend(fontsize=16)

fig.suptitle("Enforcement Effectiveness vs. Epsilon — Old vs New SMiLE",
             y=1.02)
fig.tight_layout()
fig.savefig(f"{agg_res_dir}/counter_var_vs_epsilon.pdf", bbox_inches="tight")
plt.close()
print("[INFO] Saved counter_var_vs_epsilon.pdf")


# ---------------------------------------------------------------------------
# Plot 4 — Delta heatmap: new quality − old quality per benchmark/epsilon
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, len(benchmarks),
                        figsize=(5 * len(benchmarks), 4))
for i, benchmark in enumerate(benchmarks):
    metric = metric_col[benchmark]
    sub    = table[table["benchmark"] == benchmark]
    old_q  = sub[sub["approach"] == "old"].set_index("epsilon")[metric]
    new_q  = sub[sub["approach"] == "new"].set_index("epsilon")[metric]
    delta  = (new_q - old_q).reset_index()
    delta.columns = ["epsilon", "delta"]

    ax     = axs[i]
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in delta["delta"]]
    bars   = ax.bar(delta["epsilon"].astype(str), delta["delta"],
                    color=colors, edgecolor="k", linewidth=0.7)
    ax.bar_label(bars, fmt="%+.3f", padding=2, fontsize=11)
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_title(benchmark.capitalize())
    ax.set_xlabel("epsilon")
    ax.set_ylabel(f"Δ {metric.upper()} (new − old)" if i == 0 else "")
    ax.tick_params(axis="x", labelrotation=30)

fig.suptitle("Quality Gain: New SMiLE − Old SMiLE\n"
             "(green = new better, red = old better)", y=1.02)
fig.tight_layout()
fig.savefig(f"{agg_res_dir}/quality_delta.pdf", bbox_inches="tight")
plt.close()
print("[INFO] Saved quality_delta.pdf")


# ---------------------------------------------------------------------------
# Plot 5 — Sample-wise counterfactual variation for crime (new SMiLE)
#          Scatter coloured by |y(x) - y(x_counter)| for three epsilons
# ---------------------------------------------------------------------------
features = {
    "compas": ["age", "priors_count"],
    "law":    ["lsat", "gpa"],
    "crime":  ["population", "householdsize"],
}
selected_epsilons = {"crime": [0.02, 0.06, 0.10]}

from core.data import FairnessData
data_obj = FairnessData()

# We need Net objects to compute per-sample counterfactual variation.
# Import new model for reconstruction.
from core.model_new import Net as NetNew, BaseNet as BaseNetNew

hidden_dim           = 8
h_architecture_r     = [(4*hidden_dim, "relu"), (4*hidden_dim, "relu"),
                        (hidden_dim, "linear")]
h_base_architecture_r  = [(int(1/2*hidden_dim), "relu"), (hidden_dim, "linear")]
h_delta_architecture_r = [(int(1/2*hidden_dim), "relu"), (hidden_dim, "relu")]
g_architecture_r     = [(1, "linear")]

benchmark = "crime"
_, _, _, _, _, _, x_test, y_test = data_obj.load(benchmark=benchmark)
x_test_counter = x_test.copy()
x_test_counter.iloc[:, -1] = 1 - x_test_counter.iloc[:, -1]

epsilons_plot = selected_epsilons[benchmark]
cmap = sns.color_palette("crest", as_cmap=True)

# Collect per-sample variation for each epsilon
cv_per_eps = {}
for epsilon in epsilons_plot:
    eps_str  = str(epsilon).replace(".", "")
    new_file = f"{res_dir}/{benchmark}_smle_{eps_str}_new.pkl"
    if not os.path.isfile(new_file):
        print(f"[WARN] {new_file} not found, skipping crime scatter plot.")
        break
    result  = pickle.load(open(new_file, "rb"))
    h_r     = BaseNetNew(h_architecture_r)
    h_base_r  = BaseNetNew(h_base_architecture_r)
    h_delta_r = BaseNetNew(h_delta_architecture_r)
    g_r     = BaseNetNew(g_architecture_r)
    f_r = NetNew(
        model_path=None, mode="regression",
        h=h_r, h_base=h_base_r, h_delta=h_delta_r, g=g_r,
        binary_split=None, l=None, u=None,
        d_lower=None, d_upper=None, e_lower=None, e_upper=None,
        pretrain=False,
        generation_timelimit=None, generation_solvelimit=None,
        generation_frequency=None,
    )
    f_r(x_test)
    f_r.set_weights(result["weights"])
    cv_per_eps[epsilon] = np.abs(
        f_r(x_test).numpy().flatten() - f_r(x_test_counter).numpy().flatten()
    )

if len(cv_per_eps) == len(epsilons_plot):
    all_vals = np.concatenate(list(cv_per_eps.values()))
    norm = plt.Normalize(all_vals.min(), all_vals.max())
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig, axes = plt.subplots(1, len(epsilons_plot),
                             figsize=(20, 6), sharex=True, sharey=True)
    for ax, epsilon in zip(axes, epsilons_plot[::-1]):
        col_colors = cmap(norm(cv_per_eps[epsilon]))
        ax.scatter(
            x_test[features[benchmark][0]],
            x_test[features[benchmark][1]],
            c=col_colors, s=40, edgecolor="k", linewidth=0.5, alpha=0.8,
        )
        ax.set_title(f"eps = {epsilon}")

    axes[1].set_xlabel(features[benchmark][0])
    axes[0].set_ylabel(features[benchmark][1])

    cbar_ax = fig.add_axes([0.25, 0.95, 0.5, 0.05])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.set_label("|ŷ(x) − ŷ(x_counter)|", labelpad=8)

    fig.subplots_adjust(bottom=0.18, top=0.85, wspace=0.03)
    fig.savefig(f"{agg_res_dir}/crime_counter_var.pdf", bbox_inches="tight")
    plt.close()
    print("[INFO] Saved crime_counter_var.pdf")

print(f"\n[INFO] All outputs saved to {agg_res_dir}/")