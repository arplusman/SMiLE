"""
benchmarks/fairness/analyze_degeneracy.py
==========================================
Structured analysis of box degeneracy elimination for the FAIRNESS benchmark.

Reads solver logs from results_new/ (which contains pkl files for BOTH
old and new SMiLE, distinguished by result['approach']).

Metrics compared
----------------
  Timeout rate   : fraction of MILP calls hitting the time limit
  Solve time     : wall-clock distribution per call
  Duality gap    : |dual - primal| / max(|primal|, eps)
  Exceeded time  : max(0, time - 1.5 * timelimit)
  Termination    : status code breakdown

Run via:
    python run_benchmark.py --benchmark fairness --analysis degeneracy
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
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
})

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
name    = "benchmarks/fairness"
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
    Scan results_new/ for all _old.pkl and _new.pkl SMiLE files.
    Returns one row per MILP solve call with the 'approach' column
    set to the value stored in the pkl.
    """
    rows = []
    for fname in os.listdir(res_dir):
        if not fname.endswith(".pkl") or "_smle_" not in fname:
            continue
        result = pickle.load(open(f"{res_dir}/{fname}", "rb"))
        if "approach" not in result or "time" not in result:
            continue

        approach  = result["approach"]
        benchmark = result["benchmark"]
        epsilon   = result["epsilon"]

        for i, (t, tl, p, d, term) in enumerate(zip(
                result["time"], result["timelimit"], result["primal"],
                result["dual"], result["termination"])):
            gap = abs(d - p) / max(abs(p), EPS) if p is not None else np.nan
            rows.append({
                "approach":      approach,
                "benchmark":     benchmark,
                "epsilon":       epsilon,
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

benchmarks = sorted(data["benchmark"].unique())
ncols = len(benchmarks)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
summary = (
    data.groupby(["approach", "benchmark"])
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
print("DEGENERACY ANALYSIS — FAIRNESS — SUMMARY TABLE")
print("="*70)
print(summary.to_string(index=False))
summary.to_csv(f"{out_dir}/summary.csv", index=False)


# ---------------------------------------------------------------------------
# Plot 1 — Timeout rate
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, ncols, figsize=(5*ncols, 5), sharey=True)
for ax, benchmark in zip(axs, benchmarks):
    sub  = summary[summary["benchmark"] == benchmark]
    vals = [sub.loc[sub["approach"]==a, "timeout_pct"].values[0]
            if a in sub["approach"].values else 0. for a in approaches]
    bars = ax.bar([approach_labels[a].replace("\n", " ") for a in approaches],
                  vals,
                  color=[approach_colours[a] for a in approaches],
                  width=0.5, edgecolor="k", linewidth=0.8)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=13)
    ax.set_title(benchmark.capitalize())
    ax.set_ylabel("Timeout rate (%)" if benchmark == benchmarks[0] else "")
    ax.set_ylim(0, max(max(vals)*1.3, 5))
    ax.tick_params(axis="x", labelrotation=10)

fig.suptitle("Timeout Rate — Fairness Benchmark", y=1.02)
fig.tight_layout()
fig.savefig(f"{out_dir}/timeout_rate.pdf", bbox_inches="tight")
plt.close()
print("[INFO] Saved timeout_rate.pdf")


# ---------------------------------------------------------------------------
# Plot 2 — Solve time distributions
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, ncols, figsize=(5*ncols, 5))
for ax, benchmark in zip(axs, benchmarks):
    sub = data[data["benchmark"] == benchmark]
    sns.boxplot(data=sub, x="approach", y="time", order=approaches,
                palette=[approach_colours[a] for a in approaches],
                width=0.5, fliersize=3, ax=ax)
    ax.set_xticklabels([approach_labels[a].replace("\n", " ")
                        for a in approaches], rotation=10, ha="right")
    ax.set_title(benchmark.capitalize())
    ax.set_xlabel("")
    ax.set_ylabel("Solve time (s)" if benchmark == benchmarks[0] else "")

fig.suptitle("Solve Time Distribution — Fairness Benchmark", y=1.02)
fig.tight_layout()
fig.savefig(f"{out_dir}/solve_time.pdf", bbox_inches="tight")
plt.close()
print("[INFO] Saved solve_time.pdf")


# ---------------------------------------------------------------------------
# Plot 3 — Duality gap (log scale)
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, ncols, figsize=(5*ncols, 5))
for ax, benchmark in zip(axs, benchmarks):
    sub = data[(data["benchmark"]==benchmark) & data["duality_gap"].notna()]
    sns.boxplot(data=sub, x="approach", y="duality_gap", order=approaches,
                palette=[approach_colours[a] for a in approaches],
                width=0.5, fliersize=3, ax=ax)
    ax.set_yscale("log")
    ax.set_xticklabels([approach_labels[a].replace("\n", " ")
                        for a in approaches], rotation=10, ha="right")
    ax.set_title(benchmark.capitalize())
    ax.set_xlabel("")
    ax.set_ylabel("Duality gap (log)" if benchmark == benchmarks[0] else "")

fig.suptitle("LP Relaxation Duality Gap — Fairness Benchmark\n"
             "(lower = tighter relaxation = less degeneracy)", y=1.02)
fig.tight_layout()
fig.savefig(f"{out_dir}/duality_gap.pdf", bbox_inches="tight")
plt.close()
print("[INFO] Saved duality_gap.pdf")


# ---------------------------------------------------------------------------
# Plot 4 — Termination breakdown (stacked bar)
# ---------------------------------------------------------------------------
term_order   = ["optimal", "infeasible", "inforunb", "timelimit", "unbounded"]
term_colours = {"optimal": "#2ecc71", "infeasible": "#e74c3c",
                "inforunb": "#3498db", "timelimit": "#e67e22",
                "unbounded": "#9b59b6"}

fig, axs = plt.subplots(1, ncols, figsize=(5*ncols, 5), sharey=True)
for ax, benchmark in zip(axs, benchmarks):
    sub    = data[data["benchmark"] == benchmark]
    counts = (sub.groupby(["approach", "termination"]).size()
                 .unstack(fill_value=0).reindex(index=approaches))
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
                        for a in approaches], rotation=10, ha="right")
    ax.set_title(benchmark.capitalize())
    ax.set_ylabel("% of MILP calls" if benchmark == benchmarks[0] else "")
    ax.set_ylim(0, 110)

handles = [plt.Rectangle((0,0),1,1, color=term_colours.get(t, "grey"))
           for t in term_order if t in data["termination"].unique()]
labels  = [t for t in term_order if t in data["termination"].unique()]
fig.legend(handles, labels, loc="upper center",
           bbox_to_anchor=(0.5, 1.08), ncol=len(labels), frameon=True)
fig.suptitle("Termination Breakdown — Fairness Benchmark", y=1.12)
fig.tight_layout()
fig.savefig(f"{out_dir}/termination.pdf", bbox_inches="tight")
plt.close()
print("[INFO] Saved termination.pdf")


# ---------------------------------------------------------------------------
# Plot 5 — Median duality gap vs epsilon
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, ncols, figsize=(5*ncols, 5))
for ax, benchmark in zip(axs, benchmarks):
    sub = data[(data["benchmark"]==benchmark) & data["duality_gap"].notna()]
    gap_by_eps = (sub.groupby(["approach", "epsilon"])["duality_gap"]
                     .median().reset_index())
    for approach in approaches:
        ap = gap_by_eps[gap_by_eps["approach"]==approach].sort_values("epsilon")
        ax.plot(ap["epsilon"], ap["duality_gap"],
                marker="o",
                label=approach_labels[approach].replace("\n", " "),
                color=approach_colours[approach])
    ax.set_yscale("log")
    ax.set_title(benchmark.capitalize())
    ax.set_xlabel("epsilon")
    ax.set_ylabel("Median duality gap" if benchmark == benchmarks[0] else "")
    if benchmark == benchmarks[0]:
        ax.legend()

fig.suptitle("Median Duality Gap vs. Epsilon — Fairness Benchmark", y=1.02)
fig.tight_layout()
fig.savefig(f"{out_dir}/gap_vs_epsilon.pdf", bbox_inches="tight")
plt.close()
print("[INFO] Saved gap_vs_epsilon.pdf")


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("KEY METRICS (aggregated across compas / crime / law)")
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