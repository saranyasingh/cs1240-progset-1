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

def random_binary_matrix(n: int) -> Matrix:
    return [[random.randint(0, 2) for _ in range(n)] for _ in range(n)]

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
def copy_block_to_padded(
    src: Matrix,
    src_row: int,
    src_col: int,
    size: int,
    padded_size: int,
) -> Matrix:
    """
    Copy the size x size block from src[src_row:src_row+size][src_col:src_col+size]
    into the top-left corner of a new padded_size x padded_size zero matrix.
    """
    out = zeros(padded_size, padded_size)
    for i in range(size):
        out_i = out[i]
        src_i = src[src_row + i]
        for j in range(size):
            out_i[j] = src_i[src_col + j]
    return out


def _strassen_recursive(
    A: Matrix,
    B: Matrix,
    C: Matrix,
    size: int,
    n_0: int,
    a_row: int = 0,
    a_col: int = 0,
    b_row: int = 0,
    b_col: int = 0,
) -> None:
    """
    Computes:
        C[0:size][0:size] = A[a_row:a_row+size][a_col:a_col+size] *
                            B[b_row:b_row+size][b_col:b_col+size]

    C is assumed to be a local output matrix of shape at least size x size.
    """

    # Base case: conventional multiplication directly into C
    if size <= n_0:
        for i in range(size):
            Ci = C[i]
            Ai = A[a_row + i]
            for k in range(size):
                aik = Ai[a_col + k]
                Bk = B[b_row + k]
                for j in range(size):
                    Ci[j] += aik * Bk[b_col + j]
        return

    # If the current recursive subproblem is odd-sized, pad THIS subproblem only.
    if size % 2 != 0:
        padded = size + 1

        A_pad = copy_block_to_padded(A, a_row, a_col, size, padded)
        B_pad = copy_block_to_padded(B, b_row, b_col, size, padded)
        C_pad = zeros(padded, padded)

        _strassen_recursive(A_pad, B_pad, C_pad, padded, n_0, 0, 0, 0, 0)

        # Copy only the true size x size result back into C
        for i in range(size):
            Ci = C[i]
            Cpi = C_pad[i]
            for j in range(size):
                Ci[j] = Cpi[j]
        return

    # Even size: do Strassen with offsets and local temporaries
    mid = size // 2

    T1 = zeros(mid, mid)
    T2 = zeros(mid, mid)

    M1 = zeros(mid, mid)
    M2 = zeros(mid, mid)
    M3 = zeros(mid, mid)
    M4 = zeros(mid, mid)
    M5 = zeros(mid, mid)
    M6 = zeros(mid, mid)
    M7 = zeros(mid, mid)

    # M1 = (A11 + A22)(B11 + B22)
    add_into(T1, A, A, mid, a_row, a_col, a_row + mid, a_col + mid)
    add_into(T2, B, B, mid, b_row, b_col, b_row + mid, b_col + mid)
    _strassen_recursive(T1, T2, M1, mid, n_0)

    # M2 = (A21 + A22)B11
    add_into(T1, A, A, mid, a_row + mid, a_col, a_row + mid, a_col + mid)
    _strassen_recursive(T1, B, M2, mid, n_0, 0, 0, b_row, b_col)

    # M3 = A11(B12 - B22)
    sub_into(T2, B, B, mid, b_row, b_col + mid, b_row + mid, b_col + mid)
    _strassen_recursive(A, T2, M3, mid, n_0, a_row, a_col, 0, 0)

    # M4 = A22(B21 - B11)
    sub_into(T2, B, B, mid, b_row + mid, b_col, b_row, b_col)
    _strassen_recursive(A, T2, M4, mid, n_0, a_row + mid, a_col + mid, 0, 0)

    # M5 = (A11 + A12)B22
    add_into(T1, A, A, mid, a_row, a_col, a_row, a_col + mid)
    _strassen_recursive(T1, B, M5, mid, n_0, 0, 0, b_row + mid, b_col + mid)

    # M6 = (A21 - A11)(B11 + B12)
    sub_into(T1, A, A, mid, a_row + mid, a_col, a_row, a_col)
    add_into(T2, B, B, mid, b_row, b_col, b_row, b_col + mid)
    _strassen_recursive(T1, T2, M6, mid, n_0)

    # M7 = (A12 - A22)(B21 + B22)
    sub_into(T1, A, A, mid, a_row, a_col + mid, a_row + mid, a_col + mid)
    add_into(T2, B, B, mid, b_row + mid, b_col, b_row + mid, b_col + mid)
    _strassen_recursive(T1, T2, M7, mid, n_0)

    # Assemble result into C
    for i in range(mid):
        Ci = C[i]
        Ci_mid = C[i + mid]

        M1i = M1[i]
        M2i = M2[i]
        M3i = M3[i]
        M4i = M4[i]
        M5i = M5[i]
        M6i = M6[i]
        M7i = M7[i]

        for j in range(mid):
            Ci[j] = M1i[j] + M4i[j] - M5i[j] + M7i[j]          # C11
            Ci[j + mid] = M3i[j] + M5i[j]                      # C12
            Ci_mid[j] = M2i[j] + M4i[j]                        # C21
            Ci_mid[j + mid] = M1i[j] - M2i[j] + M3i[j] + M6i[j]  # C22


def strassen_multiply_optimized(A: Matrix, B: Matrix, n_0: int) -> Matrix:
    n = len(A)

    C = zeros(n, n)
    _strassen_recursive(A, B, C, n, n_0)
    return C

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


    