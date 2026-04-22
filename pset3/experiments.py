#!/usr/bin/env python3
"""
run_experiments.py
Generates 50 random instances, runs all 7 algorithms, and produces
summary tables + a comparison plot.
"""
import time
import random
import statistics
import csv

random.seed(0)

from partition import (
    karmarkar_karp,
    repeated_random,
    hill_climbing,
    simulated_annealing,
)

NUM_INSTANCES = 50
N = 100
MAX_VAL = 10**12

ALGORITHMS = {
    "KK":                    lambda A: karmarkar_karp(A),
    "Repeated Random":       lambda A: repeated_random(A,       prepartition=False),
    "Hill Climbing":         lambda A: hill_climbing(A,         prepartition=False),
    "Simulated Annealing":   lambda A: simulated_annealing(A,   prepartition=False),
    "PP Repeated Random":    lambda A: repeated_random(A,       prepartition=True),
    "PP Hill Climbing":      lambda A: hill_climbing(A,         prepartition=True),
    "PP Simulated Annealing":lambda A: simulated_annealing(A,   prepartition=True),
}

def generate_instance():
    return [random.randint(1, MAX_VAL) for _ in range(N)]

def run_experiments():
    # results[alg_name] = list of residues (one per instance)
    results = {name: [] for name in ALGORITHMS}

    # runtimes[alg_name]= list of runtimes (one per instance)
    runtimes = {name: [] for name in ALGORITHMS} 


    for inst in range(1, NUM_INSTANCES + 1):
        A = generate_instance()
        print(f"Instance {inst}/{NUM_INSTANCES}...", flush=True)
        for name, fn in ALGORITHMS.items():
            start = time.perf_counter()
            residue = fn(A)
            end = time.perf_counter()

            results[name].append(residue)
            runtimes[name].append(end - start)

    return results, runtimes

def print_residue_table(results):
    names = list(results.keys())
    col_w = 26

    header = f"{'Algorithm':<{col_w}} {'Mean':>18} {'Median':>18} {'Std Dev':>18}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for name in names:
        vals = results[name]
        mean   = statistics.mean(vals)
        median = statistics.median(vals)
        std    = statistics.stdev(vals)
        print(f"{name:<{col_w}} {mean:>18.1f} {median:>18.1f} {std:>18.1f}")

    print("=" * len(header))

def print_runtime_table(runtimes):
    names = list(runtimes.keys())
    col_w = 26

    header = f"{'Algorithm':<{col_w}} {'Avg Runtime (s)':>18}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for name in names:
        avg_time = statistics.mean(runtimes[name])
        print(f"{name:<{col_w}} {avg_time:>18.6f}")

    print("=" * len(header))

def save_residue_csv(results, path="results.csv"):
    names = list(results.keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Instance"] + names)
        for i in range(NUM_INSTANCES):
            writer.writerow([i + 1] + [results[n][i] for n in names])

    # also write summary
    summary_path = path.replace(".csv", "_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Mean", "Median", "Std Dev", "Min", "Max"])
        for name in names:
            vals = results[name]
            writer.writerow([
                name,
                f"{statistics.mean(vals):.1f}",
                f"{statistics.median(vals):.1f}",
                f"{statistics.stdev(vals):.1f}",
                min(vals),
                max(vals),
            ])
    print(f"\nSaved per-instance results to {path}")
    print(f"Saved summary to {summary_path}")

def save_runtime_csv(runtimes, path="runtime_summary.csv"):
    names = list(runtimes.keys())

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Average Runtime (seconds)"])

        for name in names:
            avg_time = statistics.mean(runtimes[name])
            writer.writerow([name, avg_time])

    print(f"Saved runtime summary to {path}")

def plot_results(results):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import statistics
    except ImportError:
        print("\n(matplotlib not available — skipping plot)")
        return

    names  = list(results.keys())
    means  = [statistics.mean(results[n]) for n in names]
    stds   = [statistics.stdev(results[n]) for n in names]

    # ── Bar chart of means ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(len(names))
    bars = axes[0].bar(x, means, yerr=stds, capsize=5, alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    axes[0].set_ylabel("Residue")
    axes[0].set_title("Mean Residue (±1 std dev) across 50 Instances")
    axes[0].set_yscale("log")

    # ── Box plot (no colors) ────────────────────────────────────────
    data = [results[n] for n in names]
    axes[1].boxplot(data, notch=False)
    axes[1].set_xticks(range(1, len(names) + 1))
    axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    axes[1].set_ylabel("Residue")
    axes[1].set_title("Distribution of Residues across 50 Instances")
    axes[1].set_yscale("log")
    axes[1].yaxis.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("results_plot.png", dpi=150)
    print("Saved plot to results_plot.png")
    plt.show()
if __name__ == "__main__":
    print("Running experiments — this may take a few minutes...")
    results, runtimes = run_experiments()
    print_residue_table(results)
    print_runtime_table(runtimes)
    save_residue_csv(results)
    save_runtime_csv(runtimes)
    plot_results(results)