"""
run_benchmark.py
================
Thin orchestrator for SMiLE benchmarks.

Every benchmark is self-contained in its *_new.py scripts:
  - process_new.py      trains BOTH old SMiLE and new SMiLE and saves to results_new/
  - postprocess_new.py  loads both sets of results and produces comparison plots

Optional analyses (run AFTER process_new.py):
  - analyze_degeneracy.py  reads solver logs from results_new/, compares MILP quality
  - process_ablation.py    ablation over pretrain loss variants (new approach only)
  - postprocess_ablation.py

USAGE (from project root):
    python run_bench.py --benchmark monotonicity
    python run_bench.py --benchmark fairness
    python run_bench.py --benchmark fairness --analysis degeneracy
    python run_bench.py --benchmark fairness --analysis ablation
    python run_bench.py --benchmark fairness --analysis all
    python run_bench.py --benchmark fairness --skip-process   # postprocess only
    python run_bench.py --benchmark fairness --skip-postprocess

PREREQUISITES:
    core/model_baseline.py        original SMiLE  (h_lower / h_upper API)
    core/optimization_baseline.py
    core/model_new.py             new SMiLE        (h_base / h_delta API)
    core/optimization_new.py

CONVENTION — each benchmark folder must contain:
    benchmarks/{benchmark}/process_new.py
    benchmarks/{benchmark}/postprocess_new.py
  And optionally:
    benchmarks/{benchmark}/analyze_degeneracy.py
    benchmarks/{benchmark}/process_ablation.py
    benchmarks/{benchmark}/postprocess_ablation.py
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def info(msg):    print(f"[INFO]  {msg}", flush=True)
def warn(msg):    print(f"[WARN]  {msg}", flush=True)
def error(msg):   print(f"[ERROR] {msg}", file=sys.stderr, flush=True)
def section(msg): print(f"\n{'='*54}\n  {msg}\n{'='*54}", flush=True)
def timestamp():  return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_step(label: str, script: Path, log_path: Path) -> None:
    """
    Execute a benchmark script as  python -m benchmarks.{bm}.{script}
    so that  'from core.x import *'  resolves from the project root.
    Stdout is tee-d to the terminal and to log_path simultaneously.
    """
    module = ".".join(script.with_suffix("").parts)
    info(f"Running : python -m {module}")
    info(f"Log     : {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as log_fh:
        proc = subprocess.Popen(
            [sys.executable, "-m", module],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(Path(__file__).parent),
        )
        if proc.stdout is None:
            error(f"Failed to capture stdout for {label}.")
            sys.exit(1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log_fh.write(line)
        proc.wait()

    if proc.returncode != 0:
        error(f"{label} FAILED (exit {proc.returncode}). See {log_path}")
        sys.exit(proc.returncode)
    info(f"{label} completed successfully.")


# ---------------------------------------------------------------------------
# Validation & directory setup
# ---------------------------------------------------------------------------

def validate_environment(benchmark: str, skip_process: bool,
                         skip_postprocess: bool, analysis: str) -> None:
    root = Path(__file__).parent
    required_core = [
        "core/model_baseline.py",
        "core/optimization_baseline.py",
        "core/model_new.py",
        "core/optimization_new.py",
    ]
    required_scripts = []
    if not skip_process:
        required_scripts.append(f"benchmarks/{benchmark}/process_new.py")
    if not skip_postprocess:
        required_scripts.append(f"benchmarks/{benchmark}/postprocess_new.py")
    if analysis in ("degeneracy", "all"):
        required_scripts.append(f"benchmarks/{benchmark}/analyze_degeneracy.py")
    if analysis in ("ablation", "all"):
        required_scripts += [
            f"benchmarks/{benchmark}/process_ablation.py",
            f"benchmarks/{benchmark}/postprocess_ablation.py",
        ]

    missing = [f for f in required_core + required_scripts
               if not (root / f).exists()]
    if missing:
        error("Missing required files:")
        for f in missing:
            error(f"  {f}")
        sys.exit(1)


def make_dirs(benchmark: str, analysis: str) -> None:
    dirs = [
        f"benchmarks/{benchmark}/results_new",
        f"benchmarks/{benchmark}/aggregated_results_new",
        f"benchmarks/{benchmark}/logs",
    ]
    if analysis in ("ablation", "all"):
        dirs += [
            f"benchmarks/{benchmark}/results_ablation",
            f"benchmarks/{benchmark}/aggregated_results_ablation",
        ]
    if analysis in ("degeneracy", "all"):
        dirs.append(f"benchmarks/{benchmark}/degeneracy_analysis")
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SMiLE benchmark experiments."
    )
    parser.add_argument("--benchmark", required=True,
                        help="Benchmark name, e.g. monotonicity or fairness.")
    parser.add_argument("--skip-process", action="store_true",
                        help="Skip process_new.py (use existing results).")
    parser.add_argument("--skip-postprocess", action="store_true",
                        help="Skip postprocess_new.py.")
    parser.add_argument("--analysis", default="none",
                        choices=["none", "degeneracy", "ablation", "all"],
                        help="Optional analysis to run after the benchmark.")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)
    validate_environment(args.benchmark, args.skip_process,
                         args.skip_postprocess, args.analysis)
    make_dirs(args.benchmark, args.analysis)

    bm   = Path(f"benchmarks/{args.benchmark}")
    logs = bm / "logs"

    # ------------------------------------------------------------------
    # Main experiment: train old SMiLE + new SMiLE, then compare
    # ------------------------------------------------------------------
    if not args.skip_process:
        section(f"PROCESS  (old SMiLE + new SMiLE)  {timestamp()}")
        run_step("Process", bm / "process_new.py", logs / "process.log")
        info(f"Results -> benchmarks/{args.benchmark}/results_new/")
    else:
        warn("Skipping process_new.py (--skip-process).")

    if not args.skip_postprocess:
        section(f"POSTPROCESS  {timestamp()}")
        run_step("Postprocess", bm / "postprocess_new.py",
                 logs / "postprocess.log")
        info(f"Plots   -> benchmarks/{args.benchmark}/aggregated_results_new/")
    else:
        warn("Skipping postprocess_new.py (--skip-postprocess).")

    # ------------------------------------------------------------------
    # Optional analyses
    # ------------------------------------------------------------------
    if args.analysis in ("degeneracy", "all"):
        section(f"DEGENERACY ANALYSIS  {timestamp()}")
        run_step("Degeneracy analysis",
                 bm / "analyze_degeneracy.py",
                 logs / "degeneracy_analysis.log")
        info(f"Plots -> benchmarks/{args.benchmark}/degeneracy_analysis/")

    if args.analysis in ("ablation", "all"):
        section(f"PRETRAIN ABLATION — process  {timestamp()}")
        run_step("Ablation process",
                 bm / "process_ablation.py",
                 logs / "ablation_process.log")

        section(f"PRETRAIN ABLATION — postprocess  {timestamp()}")
        run_step("Ablation postprocess",
                 bm / "postprocess_ablation.py",
                 logs / "ablation_postprocess.log")
        info(f"Plots -> benchmarks/{args.benchmark}/aggregated_results_ablation/")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    section(f"BENCHMARK COMPLETE  {timestamp()}")
    print("\n  Logs:")
    for log in sorted(logs.glob("*.log")):
        print(f"    {log}")
    print("\n  Aggregated results:")
    for d in sorted(bm.iterdir()):
        if d.is_dir() and d.name.startswith("aggregated_results"):
            for pdf in sorted(d.glob("*.pdf")):
                print(f"    {pdf}")
    print()


if __name__ == "__main__":
    main()