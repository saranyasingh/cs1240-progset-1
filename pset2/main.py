import math
import random


def zero_matrix(n):
    return [[0 for _ in range(n)] for _ in range(n)]


def matrix_add(A, B):
    n = len(A)
    C = zero_matrix(n)
    ops = 0
    for i in range(n):
        for j in range(n):
            C[i][j] = A[i][j] + B[i][j]
            ops += 1  # one scalar addition
    return C, ops


def matrix_sub(A, B):
    n = len(A)
    C = zero_matrix(n)
    ops = 0
    for i in range(n):
        for j in range(n):
            C[i][j] = A[i][j] - B[i][j]
            ops += 1  # one scalar subtraction
    return C, ops


def conventional_matmul_manual(A, B):
    """
    Fully manual conventional matrix multiplication.
    No NumPy, no vectorization, every multiply/add is explicit.

    Returns:
        C, mult_ops, add_ops
    where
        mult_ops = number of scalar multiplications
        add_ops  = number of scalar additions
    """
    n = len(A)
    C = zero_matrix(n)

    mult_ops = 0
    add_ops = 0

    for i in range(n):
        for j in range(n):
            total = 0
            for k in range(n):
                prod = A[i][k] * B[k][j]
                mult_ops += 1

                if k == 0:
                    total = prod
                else:
                    total = total + prod
                    add_ops += 1

            C[i][j] = total

    return C, mult_ops, add_ops


def split_matrix(A):
    n = len(A)
    mid = n // 2

    A11 = [row[:mid] for row in A[:mid]]
    A12 = [row[mid:] for row in A[:mid]]
    A21 = [row[:mid] for row in A[mid:]]
    A22 = [row[mid:] for row in A[mid:]]

    return A11, A12, A21, A22


def combine_quadrants(C11, C12, C21, C22):
    n2 = len(C11)
    n = 2 * n2
    C = zero_matrix(n)

    for i in range(n2):
        for j in range(n2):
            C[i][j] = C11[i][j]
            C[i][j + n2] = C12[i][j]
            C[i + n2][j] = C21[i][j]
            C[i + n2][j + n2] = C22[i][j]

    return C


def next_power_of_2(n):
    if n == 0:
        return 1
    return 2 ** math.ceil(math.log2(n))


def pad_matrix(A, size):
    n = len(A)
    P = zero_matrix(size)
    for i in range(n):
        for j in range(n):
            P[i][j] = A[i][j]
    return P


def unpad_matrix(A, size):
    return [row[:size] for row in A[:size]]


def strassen(A, B, n0=32):
    """
    Strassen with threshold n0.
    Uses fully manual conventional multiplication at/below threshold.

    Returns:
        C, mult_ops, add_ops
    """
    n = len(A)

    if n <= n0:
        return conventional_matmul_manual(A, B)

    # Handle non-power-of-2 sizes by padding
    m = next_power_of_2(n)
    if m != n:
        A_pad = pad_matrix(A, m)
        B_pad = pad_matrix(B, m)
        C_pad, mult_ops, add_ops = strassen(A_pad, B_pad, n0)
        return unpad_matrix(C_pad, n), mult_ops, add_ops

    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)

    total_mult_ops = 0
    total_add_ops = 0

    S1, ops = matrix_add(A11, A22)
    total_add_ops += ops
    S2, ops = matrix_add(B11, B22)
    total_add_ops += ops
    M1, m_ops, a_ops = strassen(S1, S2, n0)
    total_mult_ops += m_ops
    total_add_ops += a_ops

    S3, ops = matrix_add(A21, A22)
    total_add_ops += ops
    M2, m_ops, a_ops = strassen(S3, B11, n0)
    total_mult_ops += m_ops
    total_add_ops += a_ops

    S4, ops = matrix_sub(B12, B22)
    total_add_ops += ops
    M3, m_ops, a_ops = strassen(A11, S4, n0)
    total_mult_ops += m_ops
    total_add_ops += a_ops

    S5, ops = matrix_sub(B21, B11)
    total_add_ops += ops
    M4, m_ops, a_ops = strassen(A22, S5, n0)
    total_mult_ops += m_ops
    total_add_ops += a_ops

    S6, ops = matrix_add(A11, A12)
    total_add_ops += ops
    M5, m_ops, a_ops = strassen(S6, B22, n0)
    total_mult_ops += m_ops
    total_add_ops += a_ops

    S7, ops = matrix_sub(A21, A11)
    total_add_ops += ops
    S8, ops2 = matrix_add(B11, B12)
    total_add_ops += ops2
    M6, m_ops, a_ops = strassen(S7, S8, n0)
    total_mult_ops += m_ops
    total_add_ops += a_ops

    S9, ops = matrix_sub(A12, A22)
    total_add_ops += ops
    S10, ops2 = matrix_add(B21, B22)
    total_add_ops += ops2
    M7, m_ops, a_ops = strassen(S9, S10, n0)
    total_mult_ops += m_ops
    total_add_ops += a_ops

    T1, ops = matrix_add(M1, M4)
    total_add_ops += ops
    T2, ops = matrix_sub(T1, M5)
    total_add_ops += ops
    C11, ops = matrix_add(T2, M7)
    total_add_ops += ops

    C12, ops = matrix_add(M3, M5)
    total_add_ops += ops

    C21, ops = matrix_add(M2, M4)
    total_add_ops += ops

    T3, ops = matrix_sub(M1, M2)
    total_add_ops += ops
    T4, ops = matrix_add(T3, M3)
    total_add_ops += ops
    C22, ops = matrix_add(T4, M6)
    total_add_ops += ops

    C = combine_quadrants(C11, C12, C21, C22)
    return C, total_mult_ops, total_add_ops


def random_matrix(n, low=-5, high=5):
    return [[random.randint(low, high) for _ in range(n)] for _ in range(n)]


def matrices_equal(A, B):
    n = len(A)
    for i in range(n):
        for j in range(n):
            if A[i][j] != B[i][j]:
                return False
    return True


def find_best_n0_by_ops(n, thresholds, trials=3):
    """
    Experimentally determine best n0 by operation count.
    """
    results = {n0: [] for n0 in thresholds}

    for _ in range(trials):
        A = random_matrix(n)
        B = random_matrix(n)

        # ground truth from manual conventional multiplication
        C_true, _, _ = conventional_matmul_manual(A, B)

        for n0 in thresholds:
            C, mult_ops, add_ops = strassen(A, B, n0=n0)

            if not matrices_equal(C, C_true):
                raise ValueError(f"Incorrect result for n0={n0}")

            total_ops = mult_ops + add_ops
            results[n0].append(total_ops)

    avg_results = {
        n0: sum(results[n0]) / len(results[n0])
        for n0 in thresholds
    }

    best_n0 = min(avg_results, key=avg_results.get)
    return best_n0, avg_results


if __name__ == "__main__":
    n = 64
    thresholds = [2, 4, 8, 16, 32]

    best_n0, avg_results = find_best_n0_by_ops(n, thresholds, trials=5)

    print("Average operation counts:")
    for n0 in thresholds:
        print(f"n0 = {n0:2d}: {avg_results[n0]}")

    print(f"\nBest n0 = {best_n0}")