import random
import time
from typing import List
import matplotlib.pyplot as plt

from strassen import (
    random_matrix,
    strassen_multiply_optimized,
    random_binary_matrix
)

def n0_experiment(n):
    A = random_binary_matrix(n)
    B = random_binary_matrix(n)

    best_n0 = None
    best_time = float("inf")

    n0_values = []     
    runtime_values = [] 

    for n0 in range(1, n):
        if n0 <= n:
            print("i am running +", n0)
            total = 0.0

            for _ in range(3):
                start = time.perf_counter()
                strassen_multiply_optimized(A, B, n0)
                end = time.perf_counter()
                total += (end - start)

            avg_time = total / 3

            n0_values.append(n0)           
            runtime_values.append(avg_time) 

            if avg_time < best_time:
                best_time = avg_time
                best_n0 = n0

    print("for n = ", n, " \n the best n0:", best_n0, "\n runtime:", total)

    plt.figure()
    plt.plot(n0_values, runtime_values)
    plt.xlabel("n0")
    plt.ylabel("Average Runtime (s)")
    plt.title(f"Runtime vs n0 (n = {n})")
    plt.show()
    
    return best_n0


if __name__ == "__main__":
    n_values = [1024, 2048]
    
    for n in n_values:
        n0_experiment(n)
    