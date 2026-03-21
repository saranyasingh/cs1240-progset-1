import random
import time
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
import sys

Matrix = List[List[int]]

# HELPER FUNCTIONS
def zeros(n: int, m: int) -> Matrix:
    return [[0 for _ in range(m)] for _ in range(n)]

def random_matrix(n: int, low: int = -10, high: int = 10) -> Matrix:
    return [[random.randint(low, high) for _ in range(n)] for _ in range(n)]

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

def add_matrix(A: Matrix, B: Matrix) -> Matrix:
    n = len(A)
    m = len(A[0])
    C = zeros(n, m)
    for i in range(n):
        for j in range(m):
            C[i][j] = A[i][j] + B[i][j]
    return C

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
def conventional_multiply(A: Matrix, B: Matrix) -> Matrix:
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

# STRASSEN

def strassen_multiply_optimized(A: Matrix, B: Matrix, n_0: int):
    n = len(A)
    size = next_power_of_2(n)

    A_pad = pad_matrix(A, size)
    B_pad = pad_matrix(B, size)

    C_pad = zeros(size, size)

    _strassen_power_of_2_optimized(A_pad, B_pad, C_pad, size, n_0)

    return unpad_matrix(C_pad, n, n)

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


def read_input_file(filepath, d):
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

    return A, B


def print_diagonal(C):
    d = len(C)
    for i in range(d):
        print(C[i][i])

if __name__ == "__main__":
   
    if len(sys.argv) != 4:
        print("Usage: ./strassen <flag> <dimension> <inputfile>")
        sys.exit(1)

    flag = int(sys.argv[1])
    d = int(sys.argv[2])
    inputfile = sys.argv[3]

    A, B = read_input_file(inputfile, d)

    n0 = 32 
    
    print_diagonal(strassen_multiply_optimized(A, B, n_0=n0))


    