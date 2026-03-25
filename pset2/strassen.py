import random
from typing import List
import sys

Matrix = List[List[int]]

# HELPER FUNCTIONS
def zeros(n, m):
    return [[0 for _ in range(m)] for _ in range(n)]

def random_matrix(n):
    return [[random.randint(-10, 10) for _ in range(n)] for _ in range(n)]

def random_binary_matrix(n):
    return [[random.randint(0, 1) for _ in range(n)] for _ in range(n)]

def add_into(C, A, B, size, a_row, a_col, b_row, b_col):
    for i in range(size):
        Ci = C[i]
        Ai = A[a_row + i]
        Bi = B[b_row + i]
        for j in range(size):
            Ci[j] = Ai[a_col + j] + Bi[b_col + j]

def sub_into(C, A, B, size, a_row, a_col, b_row, b_col): 
    for i in range(size):
        Ci = C[i]
        Ai = A[a_row + i]
        Bi = B[b_row + i]
        for j in range(size):
            Ci[j] = Ai[a_col + j] - Bi[b_col + j]


# conventional mutliplication optimized
def conventional_multiply(A, B):
    n = len(A)
    C = zeros(n, n)

    for i in range(n):
        Ci = C[i] # cache row reference
        for k in range(n):
            aik = A[i][k]
            Bk = B[k]
            for j in range(n):
                Ci[j] += aik * Bk[j] 
    return C

# STRASSEN
def pad_matrix(A, A_row, A_col, size, padded_size):
    res = zeros(padded_size, padded_size)
    for i in range(size):
        res_i = res[i]
        A_i = A[A_row+ i]
        for j in range(size):
            res_i[j] = A_i[A_col + j]
    return res


def strassen_multiply_optimized(A, B, n_0):
    def helper(A, B, C, size, n_0, a_row, a_col, b_row, b_col):
        # threshold / base case
        if size <= n_0:
            # can't just call other conventional function due to memory optimizations
            for i in range(size):
                Ci = C[i]
                Ai = A[a_row + i]
                for k in range(size):
                    aik = Ai[a_col + k]
                    Bk = B[b_row + k]
                    for j in range(size):
                        Ci[j] += aik * Bk[b_col + j]
            return

        # pad if odd
        if size % 2 != 0:
            padded = size + 1

            A_pad = pad_matrix(A, a_row, a_col, size, padded)
            B_pad = pad_matrix(B, b_row, b_col, size, padded)
            C_pad = zeros(padded, padded)

            helper(A_pad, B_pad, C_pad, padded, n_0, 0, 0, 0, 0)

            # unpadding
            for i in range(size):
                Ci = C[i]
                Cpi = C_pad[i]
                for j in range(size):
                    Ci[j] = Cpi[j]
            return

        mid = size // 2

        T1 = zeros(mid, mid)
        T2 = zeros(mid, mid)

        P1 = zeros(mid, mid)
        P2 = zeros(mid, mid)
        P3 = zeros(mid, mid)
        P4 = zeros(mid, mid)
        P5 = zeros(mid, mid)
        P6 = zeros(mid, mid)
        P7 = zeros(mid, mid)

        # P1
        add_into(T1, A, A, mid, a_row, a_col, a_row + mid, a_col + mid)
        add_into(T2, B, B, mid, b_row, b_col, b_row + mid, b_col + mid)
        helper(T1, T2, P1, mid, n_0, 0, 0, 0, 0)

        # P2
        add_into(T1, A, A, mid, a_row + mid, a_col, a_row + mid, a_col + mid)
        helper(T1, B, P2, mid, n_0, 0, 0, b_row, b_col)

        # P3
        sub_into(T2, B, B, mid, b_row, b_col + mid, b_row + mid, b_col + mid)
        helper(A, T2, P3, mid, n_0, a_row, a_col, 0, 0)

        # P4
        sub_into(T2, B, B, mid, b_row + mid, b_col, b_row, b_col)
        helper(A, T2, P4, mid, n_0, a_row + mid, a_col + mid, 0, 0)

        # P5
        add_into(T1, A, A, mid, a_row, a_col, a_row, a_col + mid)
        helper(T1, B, P5, mid, n_0, 0, 0, b_row + mid, b_col + mid)

        # P6
        sub_into(T1, A, A, mid, a_row + mid, a_col, a_row, a_col)
        add_into(T2, B, B, mid, b_row, b_col, b_row, b_col + mid)
        helper(T1, T2, P6, mid, n_0, 0, 0, 0, 0)

        # P7
        sub_into(T1, A, A, mid, a_row, a_col + mid, a_row + mid, a_col + mid)
        add_into(T2, B, B, mid, b_row + mid, b_col, b_row + mid, b_col + mid)
        helper(T1, T2, P7, mid, n_0, 0, 0, 0, 0)

        # assemble
        for i in range(mid):
            Ci = C[i]
            Ci_mid = C[i + mid]

            P1i = P1[i]
            P2i = P2[i]
            P3i = P3[i]
            P4i = P4[i]
            P5i = P5[i]
            P6i = P6[i]
            P7i = P7[i]

            for j in range(mid):
                Ci[j] = P1i[j] + P4i[j] - P5i[j] + P7i[j]          # C11
                Ci[j + mid] = P3i[j] + P5i[j]                      # C12
                Ci_mid[j] = P2i[j] + P4i[j]                        # C21
                Ci_mid[j + mid] = P1i[j] - P2i[j] + P3i[j] + P6i[j]  # C22
    
    n = len(A)
    C = zeros(n, n)
    helper(A, B, C, n, n_0, 0, 0, 0, 0)
    return C


if __name__ == "__main__":
    d = int(sys.argv[2])
    filepath = sys.argv[3]

    with open(filepath, "r") as f:
        nums = [int(line.strip()) for line in f if line.strip()]

    A = []
    idx = 0
    for i in range(d):
        row = nums[idx:idx + d]
        A.append(row)
        idx += d

    B = []
    for i in range(d):
        row = nums[idx:idx + d]
        B.append(row)
        idx += d

    # Using theoretical n_0 threshold
    C = strassen_multiply_optimized(A, B, 37)
    for i in range(len(C)):
        print(C[i][i])


    