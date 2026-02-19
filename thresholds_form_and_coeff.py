import random
import math
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# MST (Prim) with decrease-key heap (works for adjacency lists)
# ----------------------------
class MinHeapDK:
    def __init__(self):
        self.data = []
        self.pos = {}

    def push(self, vertex, weight):
        if vertex in self.pos:
            self.decrease_key(vertex, weight)
        else:
            i = len(self.data)
            self.data.append((weight, vertex))
            self.pos[vertex] = i
            self._bubble_up(i)

    def decrease_key(self, vertex, new_weight):
        i = self.pos[vertex]
        old_weight, _ = self.data[i]
        if new_weight >= old_weight:
            return
        self.data[i] = (new_weight, vertex)
        self._bubble_up(i)

    def pop(self):
        root = self.data[0]
        last = self.data.pop()
        del self.pos[root[1]]
        if self.data:
            self.data[0] = last
            self.pos[last[1]] = 0
            self._bubble_down(0)
        return root

    def _bubble_up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if self.data[i][0] < self.data[parent][0]:
                self._swap(i, parent)
                i = parent
            else:
                break

    def _bubble_down(self, i):
        n = len(self.data)
        while True:
            left = 2 * i + 1
            right = 2 * i + 2
            smallest = i
            if left < n and self.data[left][0] < self.data[smallest][0]:
                smallest = left
            if right < n and self.data[right][0] < self.data[smallest][0]:
                smallest = right
            if smallest == i:
                break
            self._swap(i, smallest)
            i = smallest

    def _swap(self, i, j):
        self.pos[self.data[i][1]], self.pos[self.data[j][1]] = j, i
        self.data[i], self.data[j] = self.data[j], self.data[i]

    def is_empty(self):
        return len(self.data) == 0


def prim_mst_decrease_key(adj):
    start = next(iter(adj))
    visited = {start}
    heap = MinHeapDK()
    total_weight = 0.0
    edges_used = 0

    for v, w in adj[start]:
        heap.push(v, w)

    while not heap.is_empty():
        w, v = heap.pop()
        if v in visited:
            continue
        visited.add(v)
        total_weight += w
        edges_used += 1
        for to, wt in adj[v]:
            if to not in visited:
                heap.push(to, wt)

    return edges_used, total_weight


# ----------------------------
# Connectivity (to ensure pruning didn't disconnect)
# ----------------------------
def is_connected(adj):
    start = next(iter(adj))
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v, _w in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == len(adj)


# ----------------------------
# Graph generators: build FULL graph once, then prune by threshold k
# Key idea: SAME underlying random graph is used for all k in a trial.
# ----------------------------

# FULL complete weighted graph (n small only)
def complete_full(n):
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            w = random.uniform(0, 1)
            adj[i].append((j, w))
            adj[j].append((i, w))
    return adj

def prune_by_weight(adj_full, k):
    k = max(0.0, min(1.0, k))
    adj = {u: [] for u in adj_full}
    for u in adj_full:
        for v, w in adj_full[u]:
            if w <= k:
                adj[u].append((v, w))
    return adj


# FULL hypercube-style graph (sparse) with Unif(0,1) weights
def hypercube_full(n):
    adj = {i: [] for i in range(n)}
    for i in range(n):
        power = 1
        while power < n:
            j = i + power
            if j < n:
                w = random.uniform(0, 1)
                adj[i].append((j, w))
                adj[j].append((i, w))
            power *= 2
    return adj


# FULL geometric graph = complete graph on points with Euclidean weights (n small only)
def geometric_full(n, dim):
    pts = [tuple(random.random() for _ in range(dim)) for _ in range(n)]
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            dist2 = 0.0
            for t in range(dim):
                d = pts[i][t] - pts[j][t]
                dist2 += d * d
            w = math.sqrt(dist2)
            adj[i].append((j, w))
            adj[j].append((i, w))
    return adj


# ----------------------------
# Core experiment:
# For each n, find smallest k such that
#   MST(pruned) == MST(full)   (weight match) AND edges_used == n-1
# with success rate >= target.
# ----------------------------
def find_min_k_for_success(full_graph_fn, n, k_grid, trials, success_target, dim=None, weight_tol=1e-9):
    successes = {k: 0 for k in k_grid}

    for _ in range(trials):
        # build ONE underlying full graph per trial
        if dim is None:
            full = full_graph_fn(n)
        else:
            full = full_graph_fn(n, dim)

        # ground truth MST on full graph
        full_edges, full_w = prim_mst_decrease_key(full)
        if full_edges != n - 1:
            # should not happen on connected full graphs, but be safe
            continue

        # test all k on this same underlying graph
        for k in k_grid:
            pruned = prune_by_weight(full, k)

            # must be connected enough to produce spanning MST
            pr_edges, pr_w = prim_mst_decrease_key(pruned)

            ok = (pr_edges == n - 1) and (abs(pr_w - full_w) <= weight_tol)
            if ok:
                successes[k] += 1

    # convert to rates and pick smallest k meeting target
    rates = {k: successes[k] / trials for k in k_grid}
    chosen = None
    for k in k_grid:
        if rates[k] >= success_target:
            chosen = k
            break
    return chosen, rates


# ----------------------------
# Plotting helpers
# ----------------------------
def plot_rates_vs_k(rates_by_n, n_values, k_grid, title, target=0.9):
    plt.figure()
    for n in n_values:
        ys = [rates_by_n[n][k] for k in k_grid]
        plt.plot(k_grid, ys, marker="o", label=f"n={n}")
    plt.axhline(target, linestyle="--")
    plt.xlabel("k (raw threshold)")
    plt.ylabel("Success rate (pruned MST matches full MST)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_C_vs_n(C_by_n, title):
    xs = np.array(sorted(C_by_n.keys()), dtype=float)
    ys = np.array([C_by_n[int(x)] for x in xs], dtype=float)
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xscale("log")
    plt.xlabel("n")
    plt.ylabel("Implied C (normalized)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ----------------------------
# Run the form tests you want
# ----------------------------
if __name__ == "__main__":
    # Use small n where FULL graphs are feasible.
    # (Complete + geometric are O(n^2) edges.)
    n_values = [64, 96, 128, 192, 256]
    trials = 30
    target = 0.9

    # Threshold sweep grids
    # For Unif(0,1) weights (complete/hypercube), k in [0,1], but small is enough.
    k_grid_weight = list(np.linspace(0.0, 0.25, 26))  # 0.00, 0.01, ..., 0.25
    # For geometric distances, k is a distance; small values matter.
    k_grid_geom = list(np.linspace(0.0, 0.5, 26))

    # ----------------------------
    # COMPLETE: expect k(n) ~ C log n / n
    # ----------------------------
    complete_min_k = {}
    complete_rates = {}

    for n in n_values:
        k_star, rates = find_min_k_for_success(
            full_graph_fn=complete_full,
            n=n,
            k_grid=k_grid_weight,
            trials=trials,
            success_target=target,
        )
        complete_min_k[n] = k_star
        complete_rates[n] = rates

    plot_rates_vs_k(complete_rates, n_values, k_grid_weight,
                    title=f"Complete: success rate vs raw k (trials={trials})",
                    target=target)

    # Normalize to implied C under the hypothesized scaling k = C log n / n
    complete_C = {n: (complete_min_k[n] * n / math.log(n)) for n in n_values}
    plot_C_vs_n(complete_C, title="Complete: implied C_n = k(n)·n/log n  (should be ~constant)")


    # ----------------------------
    # HYPERCUBE: expect k(n) ~ constant
    # ----------------------------
    hyper_min_k = {}
    hyper_rates = {}

    for n in n_values:
        k_star, rates = find_min_k_for_success(
            full_graph_fn=hypercube_full,
            n=n,
            k_grid=k_grid_weight,
            trials=trials,
            success_target=target,
        )
        hyper_min_k[n] = k_star
        hyper_rates[n] = rates

    plot_rates_vs_k(hyper_rates, n_values, k_grid_weight,
                    title=f"Hypercube: success rate vs raw k (trials={trials})",
                    target=target)

    # Normalize to implied C under constant scaling k = C
    hyper_C = {n: hyper_min_k[n] for n in n_values}
    plot_C_vs_n(hyper_C, title="Hypercube: implied C_n = k(n)  (should be ~constant)")


    # ----------------------------
    # GEOMETRIC: expect k(n) ~ C (log n / n)^(1/d)
    # ----------------------------
    for d in (2, 3, 4):
        geom_min_k = {}
        geom_rates = {}

        for n in n_values:
            k_star, rates = find_min_k_for_success(
                full_graph_fn=geometric_full,
                n=n,
                k_grid=k_grid_geom,
                trials=trials,
                success_target=target,
                dim=d,
            )
            geom_min_k[n] = k_star
            geom_rates[n] = rates

        plot_rates_vs_k(geom_rates, n_values, k_grid_geom,
                        title=f"Geometric d={d}: success rate vs raw k (trials={trials})",
                        target=target)

        # Normalize to implied C under k = C (log n / n)^(1/d)
        geom_C = {n: (geom_min_k[n] / ((math.log(n) / n) ** (1.0 / d))) for n in n_values}
        plot_C_vs_n(geom_C, title=f"Geometric d={d}: implied C_n = k(n) / (log n / n)^(1/d)  (should be ~constant)")
