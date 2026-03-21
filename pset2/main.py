import random
import time
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

Matrix = List[List[int]]


# ----------------------------
# Operation counter
# ----------------------------

@dataclass
class OpCounter:
    adds: int = 0
    mults: int = 0
    subs: int = 0

    @property
    def total(self) -> int:
        return self.adds + self.subs + self.mults


# ----------------------------
# Basic matrix utilities
# ----------------------------

def zeros(n: int, m: int) -> Matrix:
    return [[0 for _ in range(m)] for _ in range(n)]


def copy_matrix(A: Matrix) -> Matrix:
    return [row[:] for row in A]


def random_matrix(n: int, low: int = -10, high: int = 10) -> Matrix:
    return [[random.randint(low, high) for _ in range(n)] for _ in range(n)]


def matrix_equal(A: Matrix, B: Matrix) -> bool:
    n = len(A)
    m = len(A[0])
    if len(B) != n or len(B[0]) != m:
        return False
    for i in range(n):
        for j in range(m):
            if A[i][j] != B[i][j]:
                return False
    return True


def next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def pad_matrix(A: Matrix, size: int) -> Matrix:
    n = len(A)
    m = len(A[0])
    P = zeros(size, size)
    for i in range(n):
        for j in range(m):
            P[i][j] = A[i][j]
    return P


def unpad_matrix(A: Matrix, rows: int, cols: int) -> Matrix:
    return [A[i][:cols] for i in range(rows)]


# ----------------------------
# Matrix add/subtract
# ----------------------------

def add_matrix(A: Matrix, B: Matrix, counter: Optional[OpCounter] = None) -> Matrix:
    n = len(A)
    m = len(A[0])
    C = zeros(n, m)
    for i in range(n):
        for j in range(m):
            C[i][j] = A[i][j] + B[i][j]
            if counter is not None:
                counter.adds += 1
    return C

def add_into(C, A, B, size, a_row, a_col, b_row, b_col): # in-place matrix addition
    for i in range(size):
        Ci = C[i]
        Ai = A[a_row + i]
        Bi = B[b_row + i]
        for j in range(size):
            Ci[j] = Ai[a_col + j] + Bi[b_col + j]


def sub_matrix(A: Matrix, B: Matrix, counter: Optional[OpCounter] = None) -> Matrix:
    n = len(A)
    m = len(A[0])
    C = zeros(n, m)
    for i in range(n):
        for j in range(m):
            C[i][j] = A[i][j] - B[i][j]
            if counter is not None:
                counter.subs += 1
    return C


def sub_into(C, A, B, size, a_row, a_col, b_row, b_col): # in-place matrix subtraction
    for i in range(size):
        Ci = C[i]
        Ai = A[a_row + i]
        Bi = B[b_row + i]
        for j in range(size):
            Ci[j] = Ai[a_col + j] - Bi[b_col + j]


# ----------------------------
# Conventional multiplication
# ----------------------------

def conventional_multiply(A: Matrix, B: Matrix, counter: Optional[OpCounter] = None) -> Matrix:
    """
    Standard O(n^3) matrix multiplication for square matrices,
    implemented without numpy.
    """
    n = len(A)
    C = zeros(n, n)

    for i in range(n):
        Ci = C[i] # cache row reference
        for k in range(n):
            aik = A[i][k] # reuse A[i][k] in the inner loop
            Bk = B[k] # cache row reference
            for j in range(n):
                Ci[j] += aik * Bk[j] # B[k][j] accessed row-wise which is contiguous in memory --> practical speediup
                if counter is not None:
                    counter.mults += 1
                    if k > 0:
                        counter.adds += 1
    return C

    '''
    for i in range(n):
        for j in range(n):
            s = 0
            for k in range(n):
                prod = A[i][k] * B[k][j]
                if counter is not None:
                    counter.mults += 1

                if k == 0:
                    s = prod
                else:
                    s = s + prod
                    if counter is not None:
                        counter.adds += 1

            C[i][j] = s

    return C
    '''



# ----------------------------
# Split / combine helpers
# ----------------------------

def split_matrix(A: Matrix) -> Tuple[Matrix, Matrix, Matrix, Matrix]:
    n = len(A)
    mid = n // 2

    A11 = [row[:mid] for row in A[:mid]]
    A12 = [row[mid:] for row in A[:mid]]
    A21 = [row[:mid] for row in A[mid:]]
    A22 = [row[mid:] for row in A[mid:]]

    return A11, A12, A21, A22


def combine_quadrants(C11: Matrix, C12: Matrix, C21: Matrix, C22: Matrix) -> Matrix:
    top = [r1 + r2 for r1, r2 in zip(C11, C12)]
    bot = [r1 + r2 for r1, r2 in zip(C21, C22)]
    return top + bot


# ----------------------------
# Strassen with threshold n_0
# ----------------------------

def strassen_multiply(A: Matrix, B: Matrix, n_0: int, counter: Optional[OpCounter] = None) -> Matrix:
    """
    Strassen multiplication with threshold n_0.
    If current matrix size n <= n_0, switch to conventional multiplication.

    Works for arbitrary square matrices by padding to the next power of 2.
    """
    n = len(A)
    assert n == len(A[0]) == len(B) == len(B[0]), "Matrices must be square and same size."

    size = next_power_of_2(n)
    if size != n:
        A_pad = pad_matrix(A, size)
        B_pad = pad_matrix(B, size)
        C_pad = _strassen_power_of_2(A_pad, B_pad, n_0, counter)
        return unpad_matrix(C_pad, n, n)
    else:
        return _strassen_power_of_2(A, B, n_0, counter)
    
def strassen_multiply_optimized(A: Matrix, B: Matrix, n_0: int):
    n = len(A)
    size = next_power_of_2(n)

    A_pad = pad_matrix(A, size)
    B_pad = pad_matrix(B, size)

    C_pad = zeros(size, size)

    _strassen_power_of_2_optimized(A_pad, B_pad, C_pad, size, n_0)

    return unpad_matrix(C_pad, n, n)


def _strassen_power_of_2(A: Matrix, B: Matrix, n_0: int, counter: Optional[OpCounter]) -> Matrix:
    n = len(A)

    # Base case: use conventional multiplication
    if n <= n_0:
        return conventional_multiply(A, B, counter)

    # Smallest nontrivial size safeguard
    if n == 1:
        c = [[A[0][0] * B[0][0]]]
        if counter is not None:
            counter.mults += 1
        return c

    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)

    # Strassen's 7 products
    M1 = _strassen_power_of_2(add_matrix(A11, A22, counter), add_matrix(B11, B22, counter), n_0, counter)
    M2 = _strassen_power_of_2(add_matrix(A21, A22, counter), B11, n_0, counter)
    M3 = _strassen_power_of_2(A11, sub_matrix(B12, B22, counter), n_0, counter)
    M4 = _strassen_power_of_2(A22, sub_matrix(B21, B11, counter), n_0, counter)
    M5 = _strassen_power_of_2(add_matrix(A11, A12, counter), B22, n_0, counter)
    M6 = _strassen_power_of_2(sub_matrix(A21, A11, counter), add_matrix(B11, B12, counter), n_0, counter)
    M7 = _strassen_power_of_2(sub_matrix(A12, A22, counter), add_matrix(B21, B22, counter), n_0, counter)

    # Recombine
    C11 = add_matrix(sub_matrix(add_matrix(M1, M4, counter), M5, counter), M7, counter)
    C12 = add_matrix(M3, M5, counter)
    C21 = add_matrix(M2, M4, counter)
    C22 = add_matrix(sub_matrix(add_matrix(M1, M3, counter), M2, counter), M6, counter)

    return combine_quadrants(C11, C12, C21, C22)

def _strassen_power_of_2_optimized(A, B, C, size, n_0, a_row=0, a_col=0, b_row=0, b_col=0):
    
    if size <= n_0:
        # base case: write directly into C
        for i in range(size):
            Ci = C[i]
            Ai = A[a_row + i]

            for k in range(size):
                aik = Ai[a_col + k]
                Bk = B[b_row + k]

                for j in range(size):
                    Ci[j] += aik * Bk[b_col + j]
        return

    mid = size // 2

    # allocate temporaries ONCE per level
    T1 = zeros(mid, mid)
    T2 = zeros(mid, mid)

    M1 = zeros(mid, mid)
    M2 = zeros(mid, mid)
    M3 = zeros(mid, mid)
    M4 = zeros(mid, mid)
    M5 = zeros(mid, mid)
    M6 = zeros(mid, mid)
    M7 = zeros(mid, mid)

    # ---- M1 = (A11 + A22)(B11 + B22)
    add_into(T1, A, A, mid, a_row, a_col, a_row + mid, a_col + mid)
    add_into(T2, B, B, mid, b_row, b_col, b_row + mid, b_col + mid)
    _strassen_power_of_2_optimized(T1, T2, M1, mid, n_0)

    # ---- M2 = (A21 + A22) B11
    add_into(T1, A, A, mid, a_row + mid, a_col, a_row + mid, a_col + mid)
    _strassen_power_of_2_optimized(T1, B, M2, mid, n_0, 0, 0, b_row, b_col)

    # ---- M3 = A11 (B12 - B22)
    sub_into(T2, B, B, mid, b_row, b_col + mid, b_row + mid, b_col + mid)
    _strassen_power_of_2_optimized(A, T2, M3, mid, n_0, a_row, a_col, 0, 0)

    # ---- M4 = A22 (B21 - B11)
    sub_into(T2, B, B, mid, b_row + mid, b_col, b_row, b_col)
    _strassen_power_of_2_optimized(A, T2, M4, mid, n_0, a_row + mid, a_col + mid, 0, 0)

    # ---- M5 = (A11 + A12) B22
    add_into(T1, A, A, mid, a_row, a_col, a_row, a_col + mid)
    _strassen_power_of_2_optimized(T1, B, M5, mid, n_0, 0, 0, b_row + mid, b_col + mid)

    # ---- M6 = (A21 - A11)(B11 + B12)
    sub_into(T1, A, A, mid, a_row + mid, a_col, a_row, a_col)
    add_into(T2, B, B, mid, b_row, b_col, b_row, b_col + mid)
    _strassen_power_of_2_optimized(T1, T2, M6, mid, n_0)

    # ---- M7 = (A12 - A22)(B21 + B22)
    sub_into(T1, A, A, mid, a_row, a_col + mid, a_row + mid, a_col + mid)
    add_into(T2, B, B, mid, b_row + mid, b_col, b_row + mid, b_col + mid)
    _strassen_power_of_2_optimized(T1, T2, M7, mid, n_0)

    # ---- Combine directly into C
    for i in range(mid):
        Ci = C[i]
        Ci_mid = C[i + mid]

        for j in range(mid):
            C11 = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j]
            C12 = M3[i][j] + M5[i][j]
            C21 = M2[i][j] + M4[i][j]
            C22 = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j]

            Ci[j] = C11
            Ci[j + mid] = C12
            Ci_mid[j] = C21
            Ci_mid[j + mid] = C22


# ----------------------------
# Timing / benchmarking
# ----------------------------

def time_function(func, *args, repeats: int = 3, **kwargs) -> float:
    """
    Return average runtime in seconds over `repeats` runs.
    """
    func(*args, **kwargs)  # warm-up

    total = 0.0
    for _ in range(repeats):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        total += (end - start)
    return total / repeats


def count_operations_for_strassen(A: Matrix, B: Matrix, n_0: int) -> OpCounter:
    counter = OpCounter()
    _ = strassen_multiply(A, B, n_0=n_0, counter=counter)
    return counter


def count_operations_for_conventional(A: Matrix, B: Matrix) -> OpCounter:
    counter = OpCounter()
    _ = conventional_multiply(A, B, counter=counter)
    return counter


'''
def benchmark_n0_values(
    n: int,
    n0_values: List[int],
    repeats: int = 3,
    seed: int = 0,
    low: int = -10,
    high: int = 10,
    verify: bool = True
):
    """
    For a fixed matrix size n, test multiple n_0 values.
    Reports:
      - average time in seconds
      - total operation count
      - detailed adds/subs/mults
    """
    random.seed(seed)
    A = random_matrix(n, low, high)
    B = random_matrix(n, low, high)

    if verify:
        C_ref = conventional_multiply(A, B)
    else:
        C_ref = None

    results = []

    for n_0 in n0_values:
        # correctness check
        if verify:
            C_test = strassen_multiply(A, B, n_0=n_0)
            if not matrix_equal(C_ref, C_test):
                raise ValueError(f"Incorrect result for n_0 = {n_0}")

        avg_time = time_function(strassen_multiply, A, B, n_0, repeats=repeats)
        ops = count_operations_for_strassen(A, B, n_0)

        results.append({
            "n_0": n_0,
            "time_seconds": avg_time,
            "adds": ops.adds,
            "subs": ops.subs,
            "mults": ops.mults,
            "total_ops": ops.total,
        })

    return results
'''

def benchmark_n0_values(
    n: int,
    n0_values: List[int],
    repeats: int = 3,
    seed: int = 0,
    verify: bool = True
):
    random.seed(seed)
    A = random_matrix(n)
    B = random_matrix(n)

    if verify:
        C_ref = conventional_multiply(A, B)

    results = []

    for n_0 in n0_values:

        # correctness
        if verify:
            C_test = strassen_multiply_optimized(A, B, n_0)
            if not matrix_equal(C_ref, C_test):
                raise ValueError(f"Incorrect result for n_0 = {n_0}")

        # timing
        avg_time = time_function(
            strassen_multiply_optimized,
            A, B, n_0,
            repeats=repeats
        )

        results.append({
            "n_0": n_0,
            "time_seconds": avg_time,
        })

    return results


def find_best_n0_by_time(results) -> dict:
    return min(results, key=lambda x: x["time_seconds"])


def find_best_n0_by_operations(results) -> dict:
    return min(results, key=lambda x: x["total_ops"])

def print_benchmark_table(results):
    print(f"{'n_0':>6}  {'time(s)':>12}")
    print("-" * 22)
    for r in results:
        print(
            f"{r['n_0']:>6}  "
            f"{r['time_seconds']:>12.6f}"
        )

'''
def print_benchmark_table(results):
    print(
        f"{'n_0':>6}  {'time(s)':>12}  {'adds':>12}  {'subs':>12}  {'mults':>12}  {'total_ops':>12}"
    )
    print("-" * 74)
    for r in results:
        print(
            f"{r['n_0']:>6}  "
            f"{r['time_seconds']:>12.6f}  "
            f"{r['adds']:>12}  "
            f"{r['subs']:>12}  "
            f"{r['mults']:>12}  "
            f"{r['total_ops']:>12}"
        )
'''


# =========================================================
# Random graph + triangle counting
# =========================================================

def random_undirected_graph_adjacency(n: int, p: float, seed: Optional[int] = None) -> Matrix:
    """
    Build adjacency matrix of G(n,p), undirected, no self-loops.
    """
    if seed is not None:
        random.seed(seed)

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
    """
    Number of triangles = trace(A^3) / 6 for a simple undirected graph.
    """
    A2 = strassen_multiply(A, A, n_0=n_0)
    A3 = strassen_multiply(A2, A, n_0=n_0)
    return matrix_trace(A3) // 6


def expected_triangle_count(n: int, p: float) -> float:
    return math.comb(n, 3) * (p ** 3)


# =========================================================
# Experiment runner
# =========================================================

def run_triangle_experiment(n: int = 1024, p_values=None, n_0: int = 32, seed: int = 0):
    if p_values is None:
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]

    results = []

    for idx, p in enumerate(p_values):
        print(f"Running p = {p:.2f} ...")
        graph_seed = seed + idx

        t0 = time.perf_counter()
        A = random_undirected_graph_adjacency(n, p, seed=graph_seed)
        build_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        observed = triangle_count_via_a3(A, n_0=n_0)
        mult_time = time.perf_counter() - t1

        expected = expected_triangle_count(n, p)

        results.append({
            "p": p,
            "observed": observed,
            "expected": expected,
            "graph_build_time_sec": build_time,
            "triangle_count_time_sec": mult_time,
        })

        print(f"  observed = {observed}")
        print(f"  expected = {expected:.2f}")
        print(f"  build time = {build_time:.3f}s")
        print(f"  triangle count time = {mult_time:.3f}s")
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


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    # You can change n_0 if you found a better threshold experimentally.
    '''
    results = run_triangle_experiment(
        n=1024,
        p_values=[0.01, 0.02, 0.03, 0.04, 0.05],
        n_0=32,
        seed=42
    )

    print_results_table(results)
    plot_triangle_results(results)
    '''


    
    # CODE FOR FINDING OPTIMAL N_0 VALUE
    n = 128
    n0_values = [2, 4, 8, 16, 32, 64]

    results = benchmark_n0_values(
        n=n,
        n0_values=n0_values,
        repeats=3,
        seed=42,
        verify=True
    )

    print_benchmark_table(results)

    best_time = find_best_n0_by_time(results)
    # best_ops = find_best_n0_by_operations(results)

    print("\nBest n_0 by actual runtime:")
    print(best_time)

    # print("\nBest n_0 by operation count:")
    # print(best_ops)

    # Conventional baseline
    random.seed(42)
    A = random_matrix(n)
    B = random_matrix(n)

    conv_time = time_function(conventional_multiply, A, B, repeats=3)
    conv_ops = count_operations_for_conventional(A, B)

    print("\nConventional multiplication baseline:")
    print({
        "time_seconds": conv_time,
        "adds": conv_ops.adds,
        "subs": conv_ops.subs,
        "mults": conv_ops.mults,
        "total_ops": conv_ops.total,
        
    })
    