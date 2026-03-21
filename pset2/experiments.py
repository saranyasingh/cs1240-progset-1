import random
import time
from typing import List

from strassen import (
    random_matrix,
    strassen_multiply_optimized
)

def n0_experiment(n=256, repeats=3):
    random.seed(42)
    A = random_matrix(n)
    B = random_matrix(n)

    best_n0 = None
    best_time = float("inf")

    for n0 in range(1, n + 1):
        total = 0.0

        for _ in range(repeats):
            start = time.perf_counter()
            strassen_multiply_optimized(A, B, n0)
            end = time.perf_counter()
            total += (end - start)

        avg_time = total / repeats

        # print(n0, avg_time)

        if avg_time < best_time:
            best_time = avg_time
            best_n0 = n0

    print("for n = ", n, " \n the best n0:", best_n0)
    return best_n0


if __name__ == "__main__":
    n_values = [63, 64, 127, 128, 256, 512, 1024]
    for n in n_values:
        n0_experiment(n)