import random 
import math

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

def prim_alg(G):
     