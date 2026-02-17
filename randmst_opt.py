import random 
import math
import time

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
def complete_weighted_graph_pruned(n, C=2.0):
    adj = {p: [] for p in range(n)}
    k = C * math.log(n) / n  # threshold for edge weights

    for i in range(n):
        for j in range(i + 1, n):
            weight = random.uniform(0, 1)
            if weight <= k:
                adj[i].append((j, weight))
                adj[j].append((i, weight))

    return adj


def hypercube_pruned(n, C=0.7):
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


def geometric_graph_pruned(n, dim=2, C=2.0):
    points = [tuple(random.random() for _ in range(dim)) for _ in range(n)]
    adj = {p: [] for p in points}

    k = C * ((math.log(n) / n) ** (1.0 / dim))  # max edge length to include

    for i in range(n):
        u = points[i]
        for j in range(i + 1, n):
            v = points[j]
            weight = math.sqrt(sum((u[d]-v[d])**2 for d in range(dim)))
            if weight <= k:  # only keep "short" edges
                adj[u].append((v, weight))
                adj[v].append((u, weight))

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

def main():
    sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    trials = 5
    

    # Complete weighted
    for n in sizes:
        total_weight = 0
        for _ in range(trials):
            graph = complete_weighted_graph_pruned(n)

            # only run the test if it is connected which accounts for any randomness in pruning resulting in non-connectivity
            while not is_connected(graph):
                graph = complete_weighted_graph_pruned(n)

            mst, weight = prim_mst_decrease_key(graph)
            total_weight += weight

        avg_weight = total_weight / trials
        print(f"Complete weighted for {n}: {avg_weight:.4f}")
    
    sizes_hyper = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
    # Hypercube
    for n in sizes_hyper:
        total_weight = 0
        for _ in range(trials):
            graph = hypercube_pruned(n)
            # only run the test if it is connected which accounts for any randomness in pruning resulting in non-connectivity
            while not is_connected(graph):
                graph = hypercube_pruned(n)
            mst, weight = prim_mst_decrease_key(graph)
            total_weight += weight

        avg_weight = total_weight / trials
        print(f"Hypercube for {n}: {avg_weight:.4f}")
    
    # Unit square
    for n in sizes:
        total_weight = 0
        for _ in range(trials):
            points, graph = geometric_graph_pruned(n, dim=2, C=2.0)
            # only run the test if it is connected which accounts for any randomness in pruning resulting in non-connectivity
            while not is_connected(graph):
                points, graph = geometric_graph_pruned(n, dim=2, C=2.0)
            mst, weight = prim_mst_decrease_key(graph)
            total_weight += weight

        avg_weight = total_weight / trials
        print(f"Unit square for {n}: {avg_weight:.4f}")
    
    # Unit cube 
    for n in sizes:
        total_weight = 0
        for _ in range(trials):
            points, graph = geometric_graph_pruned(n, dim=3, C=2.0)
            # only run the test if it is connected which accounts for any randomness in pruning resulting in non-connectivity
            while not is_connected(graph):
                points, graph = geometric_graph_pruned(n, dim=3, C=2.0)
            mst, weight = prim_mst_decrease_key(graph)
            total_weight += weight

        avg_weight = total_weight / trials
        print(f"Unit cube for {n}: {avg_weight:.4f}")

    # Unit hypercube
    for n in sizes:
        total_weight = 0
        for _ in range(trials):
            points, graph = geometric_graph_pruned(n, dim=4, C=2.0)
            # only run the test if it is connected which accounts for any randomness in pruning resulting in non-connectivity
            while not is_connected(graph):
                points, graph = geometric_graph_pruned(n, dim=4, C=2.0)
            "calling prim now"
            mst, weight = prim_mst_decrease_key(graph)
            print(f"Unit hypercube for {n}: {weight:.4f}")
            total_weight += weight

        avg_weight = total_weight / trials
        print(f"Average unit hypercube for {n}: {avg_weight:.4f}")

main()

