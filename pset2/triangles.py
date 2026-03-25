import math
import random
import time
import matplotlib.pyplot as plt
from typing import List
from strassen import zeros, strassen_multiply_optimized

Matrix = List[List[int]]

def random_graph(n, p):
    A = zeros(n, n)

    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                A[i][j] = 1
                A[j][i] = 1

    return A

def triangles(A):
    A2 = strassen_multiply_optimized(A, A, 37)
    A3 = strassen_multiply_optimized(A2, A, 37)

    tr = 0
    for i in range(len(A3)):
        tr += A3[i][i]
    return tr // 6

def expected_triangle_count(n, p):
    return math.comb(n, 3) * (p ** 3)

def run_triangle_experiment(n, p_values):
    results = []

    for p in p_values:
        observed_list = []

        for t in range(5):
            print("im on trial:", t)
            A = random_graph(n, p)
            observed = triangles(A)
            observed_list.append(observed)

        avg_observed = sum(observed_list) / 5

        expected = expected_triangle_count(n, p)

        results.append({
            "p": p,
            "observed": avg_observed,
            "expected": expected,
            "observed_all": observed_list
        })

        print(results)

    return results



if __name__ == "__main__":
    results = run_triangle_experiment(1024, [0.01, 0.02, 0.03, 0.04, 0.05])

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