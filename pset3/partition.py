import sys
import random
import heapq
import math
 
MAX_ITER = 25000

def karmarkar_karp(A):
    heap = [-a for a in A]         
    heapq.heapify(heap)
    while len(heap) > 1:
        a1 = -heapq.heappop(heap)
        a2 = -heapq.heappop(heap)
        heapq.heappush(heap, -(abs(a1 - a2)))
    return abs(heap[0])

# residue helper

def residue_standard(A, S):
    return abs(sum(s * a for s, a in zip(S, A)))

def residue_prepartition(A, P):
    n = len(A)
    Ap = [0] * n
    for j in range(n):
        Ap[P[j] - 1] += A[j]          # p_j is 1-indexed
    return karmarkar_karp(Ap)

# random solution generators 

def random_standard(n):
    return [random.choice([-1, 1]) for _ in range(n)]

def random_prepartition(n):
    return [random.randint(1, n) for _ in range(n)]

def random_neighbor_standard(S):
    n = len(S)
    S2 = S[:]
    i, j = random.sample(range(n), 2)
    S2[i] = -S2[i]
    if random.random() < 0.5:
        S2[j] = -S2[j]
    return S2

def random_neighbor_prepartition(P):
    n = len(P)
    P2 = P[:]
    i, j = random.sample(range(n), 2)
    while P2[i] == j:          # ensure p_i actually changes
        j = random.randint(1, n)
    P2[i] = j
    return P2

# algs 

def repeated_random(A, prepartition):
    n = len(A)
    if prepartition:
        S = random_prepartition(n)
        res = residue_prepartition(A, S)
        for _ in range(MAX_ITER):
            Sp = random_prepartition(n)
            rp = residue_prepartition(A, Sp)
            if rp < res:
                S, res = Sp, rp
    else:
        S = random_standard(n)
        res = residue_standard(A, S)
        for _ in range(MAX_ITER):
            Sp = random_standard(n)
            rp = residue_standard(A, Sp)
            if rp < res:
                S, res = Sp, rp
    return res


def hill_climbing(A, prepartition=False):
    n = len(A)
    if prepartition:
        S = random_prepartition(n)
        res = residue_prepartition(A, S)
        for _ in range(MAX_ITER):
            Sp = random_neighbor_prepartition(S)
            rp = residue_prepartition(A, Sp)
            if rp < res:
                S, res = Sp, rp
    else:
        S = random_standard(n)
        res = residue_standard(A, S)
        for _ in range(MAX_ITER):
            Sp = random_neighbor_standard(S)
            rp = residue_standard(A, Sp)
            if rp < res:
                S, res = Sp, rp
    return res

# cooling schedule 
def T(iteration):
    return 10**10 * (0.8 ** (iteration // 300))

def simulated_annealing(A, prepartition=False):
    n = len(A)
    if prepartition:
        S   = random_prepartition(n)
        res = residue_prepartition(A, S)
        best_S, best_res = S[:], res
        for it in range(1, MAX_ITER + 1):
            Sp = random_neighbor_prepartition(S)
            rp = residue_prepartition(A, Sp)
            if rp < res:
                S, res = Sp, rp
            else:
                delta = rp - res
                prob  = math.exp(-delta / T(it))
                if random.random() < prob:
                    S, res = Sp, rp
            if res < best_res:
                best_S, best_res = S[:], res
    else:
        S   = random_standard(n)
        res = residue_standard(A, S)
        best_S, best_res = S[:], res
        for it in range(1, MAX_ITER + 1):
            Sp = random_neighbor_standard(S)
            rp = residue_standard(A, Sp)
            if rp < res:
                S, res = Sp, rp
            else:
                delta = rp - res
                prob  = math.exp(-delta / T(it))
                if random.random() < prob:
                    S, res = Sp, rp
            if res < best_res:
                best_S, best_res = S[:], res
    return best_res


ALGORITHMS = {
    0:  lambda A: karmarkar_karp(A),
    1:  lambda A: repeated_random(A,  prepartition=False),
    2:  lambda A: hill_climbing(A,    prepartition=False),
    3:  lambda A: simulated_annealing(A, prepartition=False),
    11: lambda A: repeated_random(A,  prepartition=True),
    12: lambda A: hill_climbing(A,    prepartition=True),
    13: lambda A: simulated_annealing(A, prepartition=True),
}

def main():
    flag     = int(sys.argv[1])
    alg_code  = int(sys.argv[2])
    inputfile = sys.argv[3]

    with open(inputfile) as f:
        A = [int(line.strip()) for line in f if line.strip()]

    result = ALGORITHMS[alg_code](A)
    print(result)


if __name__ == "__main__":
    main()