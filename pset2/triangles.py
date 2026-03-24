import math
import random
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


from strassen import zeros, strassen_multiply_optimized

Matrix = List[List[int]]

def random_undirected_graph_adjacency(n: int, p: float) -> Matrix:
    A = zeros(n, n)

    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                A[i][j] = 1
                A[j][i] = 1

    return A


def matrix_trace(A: Matrix) -> int:
    s = 0
    for i in range(len(A)):
        s += A[i][i]
    return s


def triangle_count_via_a3(A: Matrix, n_0: int) -> int:
    A2 = strassen_multiply_optimized(A, A, n_0=n_0)
    A3 = strassen_multiply_optimized(A2, A, n_0=n_0)
    return matrix_trace(A3) // 6

def expected_triangle_count(n: int, p: float) -> float:
    return math.comb(n, 3) * (p ** 3)


def run_triangle_experiment(n: int = 1024, p_values=None, n_0: int = 32, trials: int = 5):
    if p_values is None:
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]

    results = []

    for p in p_values:
        print(f"Running p = {p:.2f} with {trials} trials...")

        observed_list = []
        build_times = []
        mult_times = []

        for t in range(trials):
            print(f"  Trial {t+1}...")

            t0 = time.perf_counter()
            A = random_undirected_graph_adjacency(n, p)
            build_time = time.perf_counter() - t0

            t1 = time.perf_counter()
            observed = triangle_count_via_a3(A, n_0=n_0)
            mult_time = time.perf_counter() - t1

            observed_list.append(observed)
            build_times.append(build_time)
            mult_times.append(mult_time)

            print(f"    observed = {observed}")

        # Aggregate
        avg_observed = sum(observed_list) / trials
        avg_build_time = sum(build_times) / trials
        avg_mult_time = sum(mult_times) / trials

        expected = expected_triangle_count(n, p)

        results.append({
            "p": p,
            "observed": avg_observed,
            "expected": expected,
            "observed_all": observed_list,
            "graph_build_time_sec": avg_build_time,
            "triangle_count_time_sec": avg_mult_time,
        })

        print(f"  avg observed = {avg_observed:.2f}")
        print(f"  expected     = {expected:.2f}")
        print()

    return results


def print_results_table(results):
    print(f"{'p':>6}  {'observed':>15}  {'expected':>15}  {'build_time(s)':>15}  {'count_time(s)':>15}")
    print("-" * 78)
    for r in results:
        print(
            f"{r['p']:>6.2f}  "
            f"{r['observed']:>15}  "
            f"{r['expected']:>15.2f}  "
            f"{r['graph_build_time_sec']:>15.3f}  "
            f"{r['triangle_count_time_sec']:>15.3f}"
        )


def plot_triangle_results(results):
    p_vals = [r["p"] for r in results]
    observed = [r["observed"] for r in results]
    expected = [r["expected"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(p_vals, observed, marker="o", label="Observed triangles")
    plt.plot(p_vals, expected, marker="s", label="Expected triangles")
    plt.xlabel("p")
    plt.ylabel("Number of triangles")
    plt.title("Observed vs Expected Number of Triangles in G(1024, p)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results = run_triangle_experiment(
        n=1024,
        p_values=[0.01, 0.02, 0.03, 0.04, 0.05],
        n_0=32
    )

    print_results_table(results)
    plot_triangle_results(results)