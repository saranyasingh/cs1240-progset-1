import random 
import math
import time

class MinHeapDK:
    def __init__(self):
        self.data = []           # list of (weight, vertex)
        self.pos = {}            # maps vertex -> index in self.data

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


def complete_weighted_graph(n):
    adj = {p: [] for p in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
                weight = random.uniform(0, 1)
                adj[i].append((j, weight))
                adj[j].append((i, weight))

    return adj

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

def hypercube(n):
    adj = {p: [] for p in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
                d = j - i
                # bit manipulation to check if its a power of 2
                if (d & (d - 1) == 0):
                    weight = random.uniform(0, 1)
                    adj[i].append((j, weight))
                    adj[j].append((i, weight))

    return adj

'''
def hypercube_pruned(n, C=2.0):
    adj = {p: [] for p in range(n)}
    k = C * math.log(n) / n

    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            if (d & (d - 1) == 0):  # hypercube adjacency
                weight = random.uniform(0, 1)
                if weight <= k:
                    adj[i].append((j, weight))
                    adj[j].append((i, weight))
    return adj
'''

def hypercube_graph_pruned(n, C=0.7):
    """
    Generate a hypercube graph with n vertices and prune edges longer than threshold k.
    Each vertex 0..n-1 is connected to vertices at distance 2^i if weight <= k.
    Returns: adjacency dict {vertex: [(neighbor, weight), ...]}
    """
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

def unit_hypercube_graph_general(n, dim=2):
    """
    Generate n points uniformly at random in a d-dimensional unit hypercube.
    Returns: list of coordinate tuples [(x1,x2,...,xd), ...]
    """
    points = [tuple(random.random() for _ in range(dim)) for _ in range(n)]
    return points

def geometric_graph_pruned(n, dim=2, C=2.0):
    """
    Generate n points in d dimensions and prune edges longer than k(n) = C / n^(1/d)
    Returns: points, adjacency dict
    """
    points = [tuple(random.random() for _ in range(dim)) for _ in range(n)]
    adj = {p: [] for p in points}

    k = C / (n ** (1.0 / dim))  # max edge length to include

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
        mst_edges.append((v, weight))  # you can also store parent if needed

        for neighbor, w in G[v]:
            if neighbor not in visited:
                heap.push(neighbor, w)  # decrease-key automatically handled

    return mst_edges, total_weight

def prim_array_dense(G):
    """
    Array-based Prim's algorithm for dense graphs.
    G: adjacency dict {v: [(neighbor, weight), ...]}
    Returns: list of MST edges [(u,v,w), ...], total_weight
    """
    n = len(G)
    vertices = list(G.keys())
    idx_map = {v: i for i, v in enumerate(vertices)}  # map vertex -> index

    visited = [False] * n
    min_edge = [float('inf')] * n
    parent = [None] * n

    # Start from the first vertex
    min_edge[0] = 0
    total_weight = 0
    mst_edges = []

    for _ in range(n):
        # 1. Pick unvisited vertex with smallest min_edge
        u_idx = -1
        min_val = float('inf')
        for i in range(n):
            if not visited[i] and min_edge[i] < min_val:
                min_val = min_edge[i]
                u_idx = i

        if u_idx == -1:
            break  # all vertices visited

        visited[u_idx] = True
        total_weight += min_edge[u_idx]
        u = vertices[u_idx]

        if parent[u_idx] is not None:
            mst_edges.append((parent[u_idx], u, min_edge[u_idx]))

        # 2. Update neighbors
        for v, w in G[u]:
            v_idx = idx_map[v]
            if not visited[v_idx] and w < min_edge[v_idx]:
                min_edge[v_idx] = w
                parent[v_idx] = u

    return mst_edges, total_weight

def prim_array_dense_geometric(points):
    """
    Array-based Prim for dense geometric graphs (unit square, cube, hypercube).
    points: list of coordinates tuples [(x1,y1,...), ...]
    Returns: list of MST edges [(u,v,weight), ...], total_weight
    """
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


def test_mst():
    # Test case 1: Small graph, n=3
    graph = {
        (0, 0): [((1, 0), 1), ((2, 0), 2)],
        (1, 0): [((0, 0), 1), ((2, 0), 1)],
        (2, 0): [((0, 0), 2), ((1, 0), 1)]
    }
    mst, total_weight = prim_array_dense(graph)

    # Check the number of edges in the MST
    assert len(mst) == 2, f"Test failed! Expected 2 edges in MST, got {len(mst)}"

    # Check if the total weight of MST is correct (expected weight is 2)
    assert total_weight == 2, f"Test failed! Expected MST weight 2, got {total_weight}"

    # Test case 2: Single edge (trivial graph, n=2)
    graph = {
        (0, 0): [((1, 1), 3)],
        (1, 1): [((0, 0), 3)]
    }
    mst, total_weight = prim_array_dense(graph)

    # Only one edge in the MST
    assert len(mst) == 1, f"Test failed! Expected 1 edge in MST, got {len(mst)}"

    # Check if the MST weight is correct
    assert total_weight == 3, f"Test failed! Expected MST weight 3, got {total_weight}"

    # Test case 3: Larger graph, n=4 (manually checked)
    graph = {
        (0, 0): [((1, 1), 1), ((2, 2), 2), ((3, 3), 3)],
        (1, 1): [((0, 0), 1), ((2, 2), 1), ((3, 3), 2)],
        (2, 2): [((0, 0), 2), ((1, 1), 1), ((3, 3), 1)],
        (3, 3): [((0, 0), 3), ((1, 1), 2), ((2, 2), 1)],
    }
    mst, total_weight = prim_array_dense(graph)

    # MST has to have exactly 3 edges for a graph with 4 vertices
    assert len(mst) == 3, f"Test failed! Expected 3 edges in MST, got {len(mst)}"

    # Expected total weight based on the manually calculated MST (edges: (0,0)-(1,1), (1,1)-(2,2), (2,2)-(3,3))
    assert total_weight == 3, f"Test failed! Expected MST weight 3, got {total_weight}"

    print("All tests passed successfully!")

def test_pruned_mst():

    n = 16  # small test size for sanity check
    C = 2.0

    # Test geometric pruning
    points, adj_geom = geometric_graph_pruned(n, dim=2, C=C)
    mst_edges, total_weight = prim_array_dense(adj_geom)
    assert len(mst_edges) == n-1, f"Geometric MST incomplete, edges: {len(mst_edges)}"

    # Test hypercube pruning
    adj_hyper = hypercube_graph_pruned(n, C=C)
    mst_edges, total_weight = prim_mst_decrease_key(adj_hyper)
    assert len(mst_edges) == n-1, f"Hypercube MST incomplete, edges: {len(mst_edges)}"

    print("All pruning MST tests passed!")

def main():
    sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    trials = 5

    for n in sizes:
        total_time = 0

        for _ in range(trials):
            start = time.time()

            # points = unit_hypercube_graph_general(n, 2)
            # mst, total_weight = prim_array_dense_geometric(points)

            points, adj = geometric_graph_pruned(n, dim=4, C=2.0)
            mst_edges, total_weight = prim_mst_decrease_key(adj)


            # graph = hypercube_graph_pruned(n)
            # mst, total_weight = prim_mst_decrease_key(graph)

            # graph = complete_weighted_graph_pruned(n)
            # mst, total_weight = prim_mst_decrease_key(graph)

            end = time.time()
            total_time += (end - start)

        avg_time = total_time / trials
        print(f"{avg_time:.4f}")


# test_pruned_mst()
main()


