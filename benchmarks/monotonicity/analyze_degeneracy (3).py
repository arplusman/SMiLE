"""
benchmarks/monotonicity/analyze_degeneracy.py
==============================================
Structured analysis of box degeneracy elimination for the MONOTONICITY
benchmark.

Reads solver logs from results_new/ (which contains pkl files for BOTH
old and new SMiLE, distinguished by result['approach']).

Metrics compared
----------------
  Timeout rate   : fraction of MILP calls hitting the time limit
  Solve time     : wall-clock distribution per call
  Duality gap    : |dual - primal| / max(|primal|, eps)
                   measures LP relaxation quality directly
  Exceeded time  : max(0, time - 1.5 * timelimit)
  Termination    : status code breakdown

Box degeneracy chain:
    flat LP faces → weak relaxation → large duality gap
                 → poor branching  → more nodes → timeout
                 → escalating timelimits → exceeded time

Run via:
    python run_benchmark.py --benchmark monotonicity --analysis degeneracy
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme()
sns.set_palette("dark")
palette = sns.color_palette("dark")
sns.set_context("notebook", rc={
    "axes.titlesize": 20, "axes.labelsize": 18,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
})

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
name    = "benchmarks/monotonicity"
res_dir = f"{name}/results_new"      # single directory — both approaches live here
out_dir = f"{name}/degeneracy_analysis"
Path(out_dir).mkdir(parents=True, exist_ok=True)

TERMINATION = {
    2: "optimal", 3: "infeasible", 4: "inforunb",
    5: "unbounded", 9: "timelimit",
}
EPS = 1e-8

approaches       = ["old", "new"]
approach_colours = {"old": palette[4], "new": palette[1]}
approach_labels  = {
    "old": "Old SMiLE\n(indep. bounds)",
    "new": "New SMiLE\n(base + delta)",
}


# ---------------------------------------------------------------------------
# Load solver logs from results_new/
# ---------------------------------------------------------------------------
def load_solver_logs(res_dir: str) -> pd.DataFrame:
    """
    Scan results_new/ for all _old.pkl and _new.pkl files.
    Returns one row per MILP solve call with the 'approach' column
    set to the value stored in the pkl.
    """
    rows = []
    for fname in os.listdir(res_dir):
        if not fname.endswith(".pkl"):
            continue
        result = pickle.load(open(f"{res_dir}/{fname}", "rb"))
        # Only process SMiLE training results (skip presmle / ablation files)
        if "approach" not in result or "time" not in result:
            continue
        approach = result["approach"]
        alpha    = result["alpha"]
        omega    = result["omega"]

        for i, (t, tl, p, d, term) in enumerate(zip(
                result["time"], result["timelimit"], result["primal"],
                result["dual"], result["termination"])):
            gap = abs(d - p) / max(abs(p), EPS) if p is not None else np.nan
            rows.append({
                "approach":      approach,
                "alpha":         alpha,
                "omega":         omega,
                "task":          f"α={alpha} ω={omega}",
                "call_idx":      i,
                "termination":   TERMINATION.get(term, str(term)),
                "time":          t,
                "timelimit":     tl,
                "duality_gap":   gap,
                "exceeded_time": max(0., t - 1.5 * tl),
                "timed_out":     TERMINATION.get(term, "") == "timelimit",
            })
    return pd.DataFrame(rows)


data = load_solver_logs(res_dir)
if data.empty:
    raise RuntimeError(
        f"No solver logs found in {res_dir}. Run process_new.py first."
    )

print(f"[INFO] Loaded {len(data)} solver calls "
      f"({data[data['approach']=='old'].shape[0]} old, "
      f"{data[data['approach']=='new'].shape[0]} new)")

alphas = sorted(data["alpha"].unique())
omegas = sorted(data["omega"].unique())


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
summary = (
    data.groupby(["approach", "alpha", "omega"])
    .agg(
        n_calls       =("time",         "count"),
        timeout_rate  =("timed_out",    "mean"),
        mean_time     =("time",         "mean"),
        median_time   =("time",         "median"),
        p95_time      =("time",         lambda x: np.percentile(x, 95)),
        mean_gap      =("duality_gap",  "mean"),
        median_gap    =("duality_gap",  "median"),
        mean_exceeded =("exceeded_time","mean"),
    )
    .round(4).reset_index()
)
summary["timeout_pct"] = (summary["timeout_rate"] * 100).round(1)

print("\n" + "="*70)
print("DEGENERACY ANALYSIS — MONOTONICITY — SUMMARY TABLE")
print("="*70)
print(summary.to_string(index=False))
summary.to_csv(f"{out_dir}/summary.csv", index=False)


# ---------------------------------------------------------------------------
# Plot 1 — Timeout rate grid (rows = alpha, cols = omega)
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(len(alphas), len(omegas),
                        figsize=(4*len(omegas), 3.5*len(alphas)),
                        sharey=True, sharex=False)
for i, alpha in enumerate(alphas):
    for j, omega in enumerate(omegas):
        ax  = axs[i][j]
        sub = summary[(summary["alpha"]==alpha) & (summary["omega"]==omega)]
        vals = [sub.loc[sub["approach"]==a, "timeout_pct"].values[0]
                if a in sub["approach"].values else 0. for a in approaches]
        bars = ax.bar([approach_labels[a].replace("\n"," ") for a in approaches],
                      vals,
                      color=[approach_colours[a] for a in approaches],
                      width=0.5, edgecolor="k", linewidth=0.7)
        ax.bar_label(bars, fmt="%.0f%%", padding=2, fontsize=10)
        ax.set_title(f"α={alpha}  ω={omega}", fontsize=12)
        ax.set_ylim(0, max(max(vals)*1.35, 5))
        ax.set_ylabel("Timeout (%)" if j == 0 else "")
        ax.tick_params(axis="x", labelrotation=10, labelsize=9)

fig.suptitle("Timeout Rate — Monotonicity Benchmark", y=1.01)
fig.tight_layout()
fig.savefig(f"{out_dir}/timeout_rate.pdf", bbox_inches="tight")
plt.close()
print("[INFO] Saved timeout_rate.pdf")


# ---------------------------------------------------------------------------
# Plot 2 — Solve time distributions (one panel per omega)
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, len(omegas), figsize=(5*len(omegas), 5), sharey=False)
for ax, omega in zip(axs, omegas):
    sub = data[data["omega"] == omega]
    sns.boxplot(data=sub, x="alpha", y="time", hue="approach",
                hue_order=approaches,
                palette=[approach_colours[a] for a in approaches],
                width=0.6, fliersize=3, ax=ax)
    ax.set_title(f"ω={omega}")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Solve time (s)" if omega == omegas[0] else "")
    handles, labels_leg = ax.get_legend_handles_labels()
    ax.get_legend().remove()

fig.legend(handles,
           [approach_labels[a].replace("\n", " ") for a in approaches],
           loc="upper center", bbox_to_anchor=(0.5, 1.04),
           ncol=2, frameon=True)
fig.suptitle("Solve Time Distribution — Monotonicity Benchmark", y=1.08)
fig.tight_layout()
fig.savefig(f"{out_dir}/solve_time.pdf", bbox_inches="tight")
plt.close()
print("[INFO] Saved solve_time.pdf")


# ---------------------------------------------------------------------------
# Plot 3 — Duality gap distributions (log scale, same layout)
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, len(omegas), figsize=(5*len(omegas), 5), sharey=False)
for ax, omega in zip(axs, omegas):
    sub = data[(data["omega"]==omega) & data["duality_gap"].notna()]
    sns.boxplot(data=sub, x="alpha", y="duality_gap", hue="approach",
                hue_order=approaches,
                palette=[approach_colours[a] for a in approaches],
                width=0.6, fliersize=3, ax=ax)
    ax.set_yscale("log")
    ax.set_title(f"ω={omega}")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Duality gap (log)" if omega == omegas[0] else "")
    ax.get_legend().remove()

fig.legend(handles,
           [approach_labels[a].replace("\n", " ") for a in approaches],
           loc="upper center", bbox_to_anchor=(0.5, 1.04),
           ncol=2, frameon=True)
fig.suptitle("LP Relaxation Duality Gap — Monotonicity Benchmark\n"
             "(lower = tighter relaxation = less degeneracy)", y=1.08)
fig.tight_layout()
fig.savefig(f"{out_dir}/duality_gap.pdf", bbox_inches="tight")
plt.close()
print("[INFO] Saved duality_gap.pdf")


# ---------------------------------------------------------------------------
# Plot 4 — Termination breakdown (aggregated over all tasks)
# ---------------------------------------------------------------------------
term_order   = ["optimal", "infeasible", "inforunb", "timelimit", "unbounded"]
term_colours = {"optimal": "#2ecc71", "infeasible": "#e74c3c",
                "inforunb": "#3498db", "timelimit": "#e67e22",
                "unbounded": "#9b59b6"}

fig, ax = plt.subplots(figsize=(6, 5))
counts = (data.groupby(["approach", "termination"]).size()
              .unstack(fill_value=0)
              .reindex(index=approaches))
pcts   = counts.div(counts.sum(axis=1), axis=0) * 100
bottom = np.zeros(len(approaches))
x      = np.arange(len(approaches))
for term in term_order:
    if term not in pcts.columns:
        continue
    vals = pcts[term].values
    ax.bar(x, vals, bottom=bottom,
           color=term_colours.get(term, "grey"),
           label=term, edgecolor="k", linewidth=0.5)
    bottom += vals

ax.set_xticks(x)
ax.set_xticklabels([approach_labels[a].replace("\n", " ")
                    for a in approaches])
ax.set_ylabel("% of MILP calls")
ax.set_ylim(0, 110)
ax.legend(loc="upper right", fontsize=11)
ax.set_title("Termination Breakdown — Monotonicity (all tasks)")
fig.tight_layout()
fig.savefig(f"{out_dir}/termination.pdf", bbox_inches="tight")
plt.close()
print("[INFO] Saved termination.pdf")


# ---------------------------------------------------------------------------
# Plot 5 — Median duality gap vs omega (one panel per alpha)
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, len(alphas), figsize=(5*len(alphas), 5))
for ax, alpha in zip(axs, alphas):
    sub = data[(data["alpha"]==alpha) & data["duality_gap"].notna()]
    gap_by_om = (sub.groupby(["approach", "omega"])["duality_gap"]
                    .median().reset_index())
    for approach in approaches:
        ap = gap_by_om[gap_by_om["approach"]==approach].sort_values("omega")
        ax.plot(ap["omega"], ap["duality_gap"],
                marker="o",
                label=approach_labels[approach].replace("\n", " "),
                color=approach_colours[approach])
    ax.set_yscale("log")
    ax.set_title(f"α={alpha}")
    ax.set_xlabel("omega")
    ax.set_ylabel("Median duality gap" if alpha == alphas[0] else "")
    if alpha == alphas[0]:
        ax.legend()

fig.suptitle("Median Duality Gap vs. Omega — Monotonicity Benchmark",
             y=1.02)
fig.tight_layout()
fig.savefig(f"{out_dir}/gap_vs_omega.pdf", bbox_inches="tight")
plt.close()
print("[INFO] Saved gap_vs_omega.pdf")


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("KEY METRICS (aggregated across all alpha/omega tasks)")
print("="*70)
agg = (
    data.groupby("approach")
    .agg(
        timeout_rate  =("timed_out",    lambda x: f"{100*x.mean():.1f}%"),
        mean_time     =("time",         lambda x: f"{x.mean():.3f}s"),
        median_gap    =("duality_gap",  lambda x: f"{x.median():.4f}"),
        mean_exceeded =("exceeded_time",lambda x: f"{x.mean():.3f}s"),
    )
    .rename(index={k: v.replace("\n", " ")
                   for k, v in approach_labels.items()})
)
print(agg.to_string())
print(f"\n[INFO] All outputs saved to {out_dir}/")