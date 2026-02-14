import random 
import math

class MinHeap:
    def __init__(self):
        self.data = []

    def push(self, item):
        self.data.append(item)
        i = len(self.data) - 1

        # bubble up 
        while i > 0:
            parent = (i - 1) // 2
            if self.data[i] < self.data[parent]:
                self.data[i], self.data[parent] = self.data[parent], self.data[i]
                i = parent
            else:
                break

    def pop(self):
        self.data[0], self.data[len(self.data) - 1] = self.data[len(self.data) - 1], self.data[0]
        item = self.data.pop()
        i = 0
        n = len(self.data)

        # bubble down 
        while True:
            left = 2*i + 1
            right = 2*i + 2
            smallest = i

            if left < n and self.data[left] < self.data[smallest]:
                smallest = left
            if right < n and self.data[right] < self.data[smallest]:
                smallest = right

            if smallest == i:
                break

            self.data[i], self.data[smallest] = self.data[smallest], self.data[i]
            i = smallest
        return item


def complete_weighted_graph(n):
    adj = {p: [] for p in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
                weight = random.uniform(0, 1)
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

def unit_square_graph(n):
    points = [(random.random(), random.random()) for _ in range(n)]
    adj = {p: [] for p in points}

    for i in range(n):
        p1 = points[i]
        x1, y1 = p1
        for j in range(i + 1, n):
            p2 = points[j]
            x2, y2 = p2

            weight = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

            adj[p1].append((p2, weight))
            adj[p2].append((p1, weight))

    return adj

def unit_cube_graph(n):
    points = [tuple(random.random() for _ in range(3)) for _ in range(n)]
    adj = {p: [] for p in points}

    for i in range(n):
        p1 = points[i]
        x1, y1 = p1
        for j in range(i + 1, n):
            p2 = points[j]
            x2, y2 = p2

            weight = math.sqrt(sum((p1[k] - p2[k])**2 for k in range(3)))


            adj[p1].append((p2, weight))
            adj[p2].append((p1, weight))

    return adj

def unit_hypercube_graph(n):
    points = [tuple(random.random() for _ in range(4)) for _ in range(n)]
    adj = {p: [] for p in points}

    for i in range(n):
        p1 = points[i]
        x1, y1 = p1
        for j in range(i + 1, n):
            p2 = points[j]
            x2, y2 = p2

            weight = math.sqrt(sum((p1[k] - p2[k])**2 for k in range(4)))


            adj[p1].append((p2, weight))
            adj[p2].append((p1, weight))

    return adj

def prim_mst(G):
    start = G[0]
    visited = {start}
    heap = MinHeap()

    mst_edges = []
    total_weight = 0

    # Add initial edges for the start vertex
    for neighbor, weight in G[start]:
        heap.push((weight, start, neighbor))

    while not heap.is_empty() and len(visited) < len(G):
        weight, u, v = heap.pop()

        if v in visited:
            continue

        visited.add(v)
        mst_edges.append((u, v, weight))
        total_weight += weight

        for neighbor, w in G[v]:
            if neighbor not in visited:
                heap.push((w, v, neighbor))

    return mst_edges, total_weight