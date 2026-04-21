import sys
import random
import math
 
MAX_ITER = 25000

# HEAP IMPLEMENTATION

class MaxHeap:
    def __init__(self):
        self.data = []

    def push(self, value):
        self.data.append(value)
        self._bubble_up(len(self.data) - 1)

    def pop(self):
        if not self.data:
            raise IndexError("pop from empty heap")

        root = self.data[0]
        last = self.data.pop()

        if self.data:
            self.data[0] = last
            self._bubble_down(0)

        return root

    def build_heap(self, values):
        self.data = list(values)
        for i in range((len(self.data) // 2) - 1, -1, -1):
            self._bubble_down(i)

    def _bubble_up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if self.data[i] > self.data[parent]:
                self.data[i], self.data[parent] = self.data[parent], self.data[i]
                i = parent
            else:
                break

    def _bubble_down(self, i):
        n = len(self.data)
        while True:
            left = 2 * i + 1
            right = 2 * i + 2
            largest = i

            if left < n and self.data[left] > self.data[largest]:
                largest = left
            if right < n and self.data[right] > self.data[largest]:
                largest = right

            if largest == i:
                break

            self.data[i], self.data[largest] = self.data[largest], self.data[i]
            i = largest

    def __len__(self):
        return len(self.data)

    def is_empty(self):
        return len(self.data) == 0


def karmarkar_karp(A):
    heap = MaxHeap()
    heap.build_heap(A)

    while len(heap) > 1:
        a1 = heap.pop()
        a2 = heap.pop()
        heap.push(abs(a1 - a2))

    return 0 if heap.is_empty() else heap.pop()

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