import random
import time
from typing import List
import matplotlib.pyplot as plt  # <-- added

from strassen import (
    random_matrix,
    strassen_multiply_optimized,
    random_binary_matrix
)

def n0_experiment(n=256, repeats=1):
    random.seed(42)
    A = random_binary_matrix(n)
    B = random_binary_matrix(n)

    best_n0 = None
    best_time = float("inf")

    n0_options = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    n0_values = []       # <-- added
    runtime_values = []  # <-- added

    for n0 in range(1, 100):
        
        if n0 <= n:
            print("i am running +", n0)
            total = 0.0

            for _ in range(repeats):
                start = time.perf_counter()
                strassen_multiply_optimized(A, B, n0)
                end = time.perf_counter()
                total += (end - start)

            avg_time = total / repeats

            n0_values.append(n0)            # <-- added
            runtime_values.append(avg_time) # <-- added

            if avg_time < best_time:
                best_time = avg_time
                best_n0 = n0

    print("for n = ", n, " \n the best n0:", best_n0, "\n runtime:", total)

    
    # ---- plotting ----
    plt.figure()
    plt.plot(n0_values, runtime_values)
    plt.xlabel("n0")
    plt.ylabel("Average Runtime (s)")
    plt.title(f"Runtime vs n0 (n = {n})")
    plt.show()
    # ------------------
    
    return best_n0


if __name__ == "__main__":
    n_values = [1024, 2048]
    
    for n in n_values:
        n0_experiment(n)
    