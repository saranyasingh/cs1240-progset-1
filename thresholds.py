import random 
import math
import time
import sys

class MinHeapDK:
    def __init__(self):
        self.data = []       
        # for mapping vertex to its position in the heap    
        self.pos = {}           

    def push(self, vertex, weight):
        if vertex in self.pos:
            # just decrease the key (if new weight is smaller) instead of creating a new key
            self.decrease_key(vertex, weight)
        else:
            # push new vertex with weight
            i = len(self.data)
            self.data.append((weight, vertex))
            self.pos[vertex] = i
            self._bubble_up(i)

    # decrease key for existing vertices than creating them again
    def decrease_key(self, vertex, new_weight):
        i = self.pos[vertex]
        old_weight, _ = self.data[i]
        if new_weight >= old_weight:
            return  # no need to decrease
        self.data[i] = (new_weight, vertex)
        self._bubble_up(i)

    def pop(self):
        if not self.data:
            raise IndexError("pop from empty heap")
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
            left = 2*i + 1
            right = 2*i + 2
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

# ------ GRAPH IMPELEMENTATION -------

# keeps only the light edges, which is the edges we are more likely to use
def complete_weighted_graph_pruned(n, C):
    adj = {p: [] for p in range(n)}
    k = C * math.log(n) / n  # threshold for edge weights

    for i in range(n):
        for j in range(i + 1, n):
            weight = random.uniform(0, 1)
            if weight <= k:
                adj[i].append((j, weight))
                adj[j].append((i, weight))

    return adj


def hypercube_pruned(n, C):
    import random
    import math

    adj = {i: [] for i in range(n)}

    k = C  # threshold for pruning (approx 0.7 works for MST)
    
    for i in range(n):
        power = 1
        while power < n:
            j = i + power
            if j < n:
                weight = random.uniform(0, 1)
                if weight <= k:
                    adj[i].append((j, weight))
                    adj[j].append((i, weight))
            power *= 2

    return adj

def geometric_graph_pruned(n, C, dim=2):
    rand = random.random
    sqrt = math.sqrt
    log = math.log

    points = [tuple(rand() for _ in range(dim)) for _ in range(n)]
    adj = {p: [] for p in points}

    k = C * ((log(n) / n) ** (1.0 / dim))
    k2 = k * k  # compare squared distances for efficiency

    if dim == 2:
        for i in range(n):
            ux, uy = points[i]
            u = points[i]
            for j in range(i + 1, n):
                vx, vy = points[j]
                dx = ux - vx
                dy = uy - vy
                dist2 = dx*dx + dy*dy
                if dist2 <= k2:
                    w = sqrt(dist2)  # sqrt only for kept edges
                    v = points[j]
                    adj[u].append((v, w))
                    adj[v].append((u, w))

    elif dim == 3:
        for i in range(n):
            ux, uy, uz = points[i]
            u = points[i]
            for j in range(i + 1, n):
                vx, vy, vz = points[j]
                dx = ux - vx
                dy = uy - vy
                dz = uz - vz
                dist2 = dx*dx + dy*dy + dz*dz
                if dist2 <= k2:
                    w = sqrt(dist2)
                    v = points[j]
                    adj[u].append((v, w))
                    adj[v].append((u, w))

    elif dim == 4:
        for i in range(n):
            u0, u1, u2, u3 = points[i]
            u = points[i]
            for j in range(i + 1, n):
                v0, v1, v2, v3 = points[j]
                d0 = u0 - v0
                d1 = u1 - v1
                d2 = u2 - v2
                d3 = u3 - v3
                dist2 = d0*d0 + d1*d1 + d2*d2 + d3*d3
                if dist2 <= k2:
                    w = sqrt(dist2)
                    v = points[j]
                    adj[u].append((v, w))
                    adj[v].append((u, w))

    return points, adj


def prim_mst_decrease_key(G):
    n = len(G)
    start = next(iter(G))  # pick any starting vertex
    visited = set()
    visited.add(start)

    heap = MinHeapDK()
    mst_edges = []
    total_weight = 0

    # Initialize distances for neighbors of start
    for neighbor, weight in G[start]:
        heap.push(neighbor, weight)

    while not heap.is_empty():
        weight, v = heap.pop()
        if v in visited:
            continue
        visited.add(v)
        total_weight += weight
        mst_edges.append((v, weight)) 

        for neighbor, w in G[v]:
            if neighbor not in visited:
                heap.push(neighbor, w)  # decrease-key automatically handled

    return mst_edges, total_weight

def prim_array_dense_geometric(points):
    n = len(points)
    visited = [False] * n
    min_edge = [float('inf')] * n
    parent = [None] * n

    # Start from vertex 0
    min_edge[0] = 0
    total_weight = 0
    mst_edges = []

    for _ in range(n):
        # Pick unvisited vertex with smallest min_edge
        u_idx = -1
        min_val = float('inf')
        for i in range(n):
            if not visited[i] and min_edge[i] < min_val:
                min_val = min_edge[i]
                u_idx = i

        if u_idx == -1:
            break

        visited[u_idx] = True
        # total_weight += min_edge[u_idx]
        total_weight += math.sqrt(min_edge[u_idx])
        if parent[u_idx] is not None:
            mst_edges.append((parent[u_idx], u_idx, min_edge[u_idx]))

        u = points[u_idx]

        # Update min_edge for all unvisited vertices on the fly
        for v_idx in range(n):
            if not visited[v_idx]:
                v = points[v_idx]
                # Euclidean distance in d-dimensions
                # weight = math.sqrt(sum((u[k]-v[k])**2 for k in range(len(u))))
                weight = sum((u[k]-v[k])**2 for k in range(len(u)))
                if weight < min_edge[v_idx]:
                    min_edge[v_idx] = weight
                    parent[v_idx] = u_idx

    return mst_edges, total_weight

# for testing pruning connectivity 
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

import matplotlib.pyplot as plt

def find_min_C_for_connectivity(
    n_values,
    C_values,
    trials=20,
    conn_target=0.9,          # "most of the time"
    geometric_dims=(2, 3, 4),
):
    """
    For each graph + n (and each dim for geometric), find the *smallest* C such that
    connectivity_rate >= conn_target.

    Returns:
      bestC: dict with bestC["complete"][n], bestC["hypercube"][n], bestC["geometric"][dim][n]
      stats: dict with connectivity rates for plotting/inspection
    """
    def conn_rate_for(graph_name, n, C, dim=None):
        ok = 0
        for _ in range(trials):
            if graph_name == "complete":
                G = complete_weighted_graph_pruned(n, C)
            elif graph_name == "hypercube":
                G = hypercube_pruned(n, C)
            elif graph_name == "geometric":
                _, G = geometric_graph_pruned(n, dim=dim, C=C)
            else:
                raise ValueError("unknown graph")

            if is_connected(G):
                ok += 1
        return ok / trials

    bestC = {"complete": {}, "hypercube": {}, "geometric": {d: {} for d in geometric_dims}}
    stats = {"complete": {}, "hypercube": {}, "geometric": {d: {} for d in geometric_dims}}

    # ---- Complete + Hypercube ----
    for graph_name in ["complete", "hypercube"]:
        for n in n_values:
            stats[graph_name][n] = {}
            chosen = None
            for C in C_values:   # assumes C_values sorted ascending
                r = conn_rate_for(graph_name, n, C)
                stats[graph_name][n][C] = r
                if chosen is None and r >= conn_target:
                    chosen = C
            bestC[graph_name][n] = chosen

    # ---- Geometric for each dim ----
    for d in geometric_dims:
        for n in n_values:
            stats["geometric"][d][n] = {}
            chosen = None
            for C in C_values:
                r = conn_rate_for("geometric", n, C, dim=d)
                stats["geometric"][d][n][C] = r
                if chosen is None and r >= conn_target:
                    chosen = C
            bestC["geometric"][d][n] = chosen

    return bestC, stats


def plot_connectivity_curves(stats, n_values, C_values, title):
    """Connectivity rate vs C (one line per n)."""
    plt.figure()
    for n in n_values:
        ys = [stats[n][C] for C in C_values]
        plt.plot(C_values, ys, marker="o", label=f"n={n}")
    plt.axhline(0.9, linestyle="--")  # change if you used a different conn_target
    plt.xlabel("C")
    plt.ylabel("Connectivity rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_bestC_vs_n(bestC_for_graph, n_values, title):
    """Best (smallest) C that hits the target vs n."""
    plt.figure()
    xs = list(n_values)
    ys = [bestC_for_graph[n] if bestC_for_graph[n] is not None else float("nan") for n in xs]
    plt.plot(xs, ys, marker="o")
    plt.xlabel("n")
    plt.ylabel("Smallest C meeting target")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ----------------------------
# Example usage:
# ----------------------------
if __name__ == "__main__":
    n_values = [128, 256, 512, 1024, 2048]
    C_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]  # ascending
    trials = 5
    conn_target = 0.9

    bestC, stats = find_min_C_for_connectivity(
        n_values=n_values,
        C_values=C_values,
        trials=trials,
        conn_target=conn_target,
        geometric_dims=(2, 3, 4),
    )

    print("Smallest C hitting target connectivity:")
    print("complete:", bestC["complete"])
    print("hypercube:", bestC["hypercube"])
    for d in bestC["geometric"]:
        print(f"geometric dim={d}:", bestC["geometric"][d])

    # plots
    plot_connectivity_curves(stats["complete"], n_values, C_values,
                             f"Complete: connectivity vs C (trials={trials}, target={conn_target})")
    plot_bestC_vs_n(bestC["complete"], n_values,
                    f"Complete: smallest C vs n (target={conn_target})")

    plot_connectivity_curves(stats["hypercube"], n_values, C_values,
                             f"Hypercube: connectivity vs C (trials={trials}, target={conn_target})")
    plot_bestC_vs_n(bestC["hypercube"], n_values,
                    f"Hypercube: smallest C vs n (target={conn_target})")

    for d in (2, 3, 4):
        plot_connectivity_curves(stats["geometric"][d], n_values, C_values,
                                 f"Geometric dim={d}: connectivity vs C (trials={trials}, target={conn_target})")
        plot_bestC_vs_n(bestC["geometric"][d], n_values,
                        f"Geometric dim={d}: smallest C vs n (target={conn_target})")
