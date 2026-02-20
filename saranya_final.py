import random 
import math
import time
import sys

# HEAP IMPLEMENTATION

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

# GRAPH IMPELMETNATION

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

def geometric_graph_pruned(n, dim=2, C=1.5):
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


# PRIM IMPLEMENTATION

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

# TESTING

def rand_mst(points, trials, dimensions):
    
    total_weight = 0
    for _ in range(trials):
        if dimensions == 0:
            graph = complete_weighted_graph_pruned(points)
           
        elif dimensions == 1:
            graph = hypercube_pruned(points)
           
        elif dimensions == 2:
            _, graph = geometric_graph_pruned(points, dim=2, C=2.0)
            
        elif dimensions == 3:
            _, graph = geometric_graph_pruned(points, dim=3, C=2.0)
           
        else:
            _, graph = geometric_graph_pruned(points, dim=4, C=2.0)

        
        mst, weight = prim_mst_decrease_key(graph)
        # ensure connectedness
        while len(mst) != points - 1:
            if dimensions == 0:
                graph = complete_weighted_graph_pruned(points)
            elif dimensions == 1:
                graph = hypercube_pruned(points)
            
            elif dimensions == 2:
                _, graph = geometric_graph_pruned(points, dim=2, C=2.0)
                
            elif dimensions == 3:
                _, graph = geometric_graph_pruned(points, dim=3, C=2.0)
            
            else:
                _, graph = geometric_graph_pruned(points, dim=4, C=2.0)

            mst, weight = prim_mst_decrease_key(graph)
         
        total_weight += weight

    avg_weight = total_weight / trials
    
    return avg_weight, points, trials, dimensions
    

if __name__ == "__main__":
    points = int(sys.argv[2])
    trials = int(sys.argv[3])
    dimensions = int(sys.argv[4])

    avg, points, trials, dimensions = rand_mst(points, trials, dimensions)

    print(f"{avg} {points} {trials} {dimensions}")