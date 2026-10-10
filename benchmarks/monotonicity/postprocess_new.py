"""
benchmarks/monotonicity/postprocess_new.py
===========================================
Loads OLD and NEW SMiLE results from results_new/ and produces
comparison plots and tables.

Distinguishes the two approaches via the 'approach' field stored
in each pkl ('old' or 'new') — no second directory needed.

Produces
--------
  monotonicity_comparison.pdf  — side-by-side R² heatmaps (shared colour scale)
  monotonicity_delta.pdf       — heatmap of (new R² − old R²)
  monotonicity_example.pdf     — example curve overlay for one (alpha, omega)
  r2_table.csv

Run via:
    python run_benchmark.py --benchmark monotonicity
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Old approach — needed only for reconstructing the Net for the curve plot
from core.model_baseline import (
    Net as NetOld,
    BaseNet as BaseNetOld,
)
# New approach
from core.model_new import (
    Net as NetNew,
    BaseNet as BaseNetNew,
    tf,
)
from core.data import Synthetic

sns.set_palette("dark")
palette = sns.color_palette("dark")
sns.set_theme()
sns.set_context("notebook", rc={
    "font.size": 12,
    "axes.titlesize": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 22,
})

# ---------------------------------------------------------------------------
# Config  (must match process_new.py exactly so weights reload correctly)
# ---------------------------------------------------------------------------
name        = "benchmarks/monotonicity"
res_dir     = f"{name}/results_new"
agg_res_dir = f"{name}/aggregated_results_new"
Path(agg_res_dir).mkdir(parents=True, exist_ok=True)

n_samples = 1000
input_dim = 1

hidden_dim     = 8
aux_complexity = 32
h_architecture       = [(16, "relu"), (32, "relu"), (64, "relu"),
                        (32, "relu"), (16, "relu"), (hidden_dim, "linear")]
g_architecture       = [(1, "linear")]
h_aux_architecture   = [(aux_complexity, "relu"), (hidden_dim, "linear")]
h_base_architecture  = [(aux_complexity, "relu"), (hidden_dim, "linear")]
h_delta_architecture = [(aux_complexity, "relu"), (hidden_dim, "relu")]

APPROACH_LABELS  = {"old": "Old SMiLE\n(indep. bounds)",
                    "new": "New SMiLE\n(base + delta)"}
APPROACH_COLOURS = {"old": palette[4], "new": palette[1]}
APPROACH_LINES   = {"old": "--", "new": "-"}


# ---------------------------------------------------------------------------
# Load metrics from pkl files (no Net reconstruction needed for tables/heatmaps)
# ---------------------------------------------------------------------------
rows = []
for fname in sorted(os.listdir(res_dir)):
    if not fname.endswith(".pkl"):
        continue
    result = pickle.load(open(f"{res_dir}/{fname}", "rb"))
    if "approach" not in result:
        continue   # skip presmle / ablation files if present
    rows.append({
        "approach": result["approach"],
        "alpha":    result["alpha"],
        "omega":    result["omega"],
        "r2":       result["r2"],
    })

if not rows:
    raise RuntimeError(f"No results found in {res_dir}. "
                       "Run process_new.py first.")

table = pd.DataFrame(rows).sort_values(["approach", "alpha", "omega"])
table.to_csv(f"{agg_res_dir}/r2_table.csv", index=False)

print("\n=== R² Table: Old vs New SMiLE ===")
pivot = (table
         .pivot_table(index=["alpha", "omega"], columns="approach", values="r2")
         .round(4))
print(pivot.to_string())


# ---------------------------------------------------------------------------
# Plot 1 — Side-by-side R² heatmaps (shared colour scale)
# ---------------------------------------------------------------------------
alphas = sorted(table["alpha"].unique())
omegas = sorted(table["omega"].unique(), reverse=True)   # high omega on top


def make_heatmap(df, approach):
    return (df[df["approach"] == approach]
            .pivot_table(index="omega", columns="alpha", values="r2",
                         fill_value=0)
            .reindex(index=omegas, columns=alphas))


hm_old = make_heatmap(table, "old")
hm_new = make_heatmap(table, "new")
vmin   = min(hm_old.values.min(), hm_new.values.min())
vmax   = 1.0

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
for ax, hm, approach in zip(axs, [hm_old, hm_new], ["old", "new"]):
    sns.heatmap(hm, annot=True, fmt=".2f", cmap="crest",
                vmin=vmin, vmax=vmax,
                cbar=(ax is axs[1]),
                ax=ax, annot_kws={"size": 20})
    ax.set_title(APPROACH_LABELS[approach].replace("\n", " "))
    ax.set_xlabel("alpha")
    ax.set_ylabel("omega")

fig.suptitle("R² — Monotonicity Benchmark", y=1.02)
fig.tight_layout()
fig.savefig(f"{agg_res_dir}/monotonicity_comparison.pdf", bbox_inches="tight")
plt.close()
print("\n[INFO] Saved monotonicity_comparison.pdf")


# ---------------------------------------------------------------------------
# Plot 2 — Delta heatmap: new R² − old R²
#          Green = new is better; Red = old is better
# ---------------------------------------------------------------------------
delta_hm = hm_new - hm_old
vabs     = max(abs(delta_hm.values.min()), abs(delta_hm.values.max()))

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(delta_hm, annot=True, fmt="+.3f", cmap="RdYlGn",
            vmin=-vabs, vmax=vabs, center=0, cbar=True,
            ax=ax, annot_kws={"size": 20})
ax.set_title("R² gain: New − Old SMiLE\n(green = new better, red = old better)")
ax.set_xlabel("alpha")
ax.set_ylabel("omega")
fig.tight_layout()
fig.savefig(f"{agg_res_dir}/monotonicity_delta.pdf", bbox_inches="tight")
plt.close()
print("[INFO] Saved monotonicity_delta.pdf")


# ---------------------------------------------------------------------------
# Plot 3 — Example curve overlay: task + old SMiLE + new SMiLE
#          Uses the representative mid-difficulty task alpha=2.0, omega=0.6
# ---------------------------------------------------------------------------
EXAMPLE_ALPHA, EXAMPLE_OMEGA = 2.0, 0.6
example_tag = (f"{str(EXAMPLE_ALPHA).replace('.', '')}"
               f"_{str(EXAMPLE_OMEGA).replace('.', '')}")

# Recreate the exact data split used during training (seed=0, standardize=True)
task_fn   = lambda x: x + EXAMPLE_ALPHA * np.sin(EXAMPLE_OMEGA * x)
data_ex   = Synthetic(n_samples=n_samples, input_dim=input_dim,
                      l=[-10], u=[10], func=task_fn)
x_ex, y_ex, _, _, x_test_ex, _ = data_ex.generate(0, standardize=True)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x_ex, y_ex, label="task", color="steelblue", linewidth=2)

for approach in ("old", "new"):
    fpath = f"{res_dir}/{example_tag}_{approach}.pkl"
    if not os.path.isfile(fpath):
        print(f"[WARN] Example file not found, skipping: {fpath}")
        continue

    result = pickle.load(open(fpath, "rb"))

    if approach == "old":
        # Reconstruct old SMiLE (h_lower / h_upper API)
        h_net = BaseNetOld(h_architecture)
        h_lo  = BaseNetOld(h_aux_architecture)
        h_up  = BaseNetOld(h_aux_architecture)
        g_net = BaseNetOld(g_architecture)
        f = NetOld(
            model_path=None, mode="regression",
            h=h_net, h_lower=h_lo, h_upper=h_up, g=g_net,
            binary_split=None, l=None, u=None,
            d_lower=None, d_upper=None, e_lower=None, e_upper=None,
            pretrain=False,
            generation_timelimit=None, generation_solvelimit=None,
            generation_frequency=None,
        )
    else:
        # Reconstruct new SMiLE (h_base / h_delta API)
        h_net     = BaseNetNew(h_architecture)
        h_base_n  = BaseNetNew(h_base_architecture)
        h_delta_n = BaseNetNew(h_delta_architecture)
        g_net     = BaseNetNew(g_architecture)
        f = NetNew(
            model_path=None, mode="regression",
            h=h_net, h_base=h_base_n, h_delta=h_delta_n, g=g_net,
            binary_split=None, l=None, u=None,
            d_lower=None, d_upper=None, e_lower=None, e_upper=None,
            pretrain=False,
            generation_timelimit=None, generation_solvelimit=None,
            generation_frequency=None,
        )

    f(x_test_ex)                    # build graph before loading weights
    f.set_weights(result["weights"])
    ax.plot(x_ex, f(x_ex),
            label=APPROACH_LABELS[approach].replace("\n", " "),
            color=APPROACH_COLOURS[approach],
            linestyle=APPROACH_LINES[approach],
            linewidth=2)

ax.legend(loc="upper left")
ax.set_title(f"α={EXAMPLE_ALPHA}  ω={EXAMPLE_OMEGA}")
fig.subplots_adjust(left=0.1, right=1.0, top=1.0, bottom=0.17)
fig.savefig(f"{agg_res_dir}/monotonicity_example.pdf", bbox_inches="tight")
plt.close()
print("[INFO] Saved monotonicity_example.pdf")

print(f"\n[INFO] All outputs saved to {agg_res_dir}/")