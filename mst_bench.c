/*
Compile:
  gcc -O2 -std=c11 mst_bench.c -lm -o mst_bench

Run:
  ./mst_bench

This translates your Python code structure into C:
- MinHeap for (weight, u, v)
- Graph generators:
    complete_weighted_graph
    hypercube
    unit_square_graph
    unit_cube_graph
    unit_hypercube_graph
- Prim's MST
- Benchmark loop (sizes, 5 trials, average time)

WARNING:
Complete graphs use O(n^2) edges. For n=32768 this is enormous and will almost surely run out of memory.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

/* ---------------- timing ---------------- */
static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* ---------------- RNG helpers ---------------- */
static inline double rand01(void) {
    return (double)rand() / (double)RAND_MAX;
}

/* ---------------- MinHeap for (w,u,v) ---------------- */
typedef struct {
    double w;
    int u;
    int v;
} HeapItem;

typedef struct {
    HeapItem *data;
    int size;
    int cap;
} MinHeap;

static bool heap_item_less(HeapItem a, HeapItem b) {
    /* Python tuple comparison: (w,u,v) lexicographic */
    if (a.w < b.w) return true;
    if (a.w > b.w) return false;
    if (a.u < b.u) return true;
    if (a.u > b.u) return false;
    return a.v < b.v;
}

static void heap_init(MinHeap *h) {
    h->data = NULL;
    h->size = 0;
    h->cap = 0;
}

static void heap_free(MinHeap *h) {
    free(h->data);
    h->data = NULL;
    h->size = 0;
    h->cap = 0;
}

static void heap_push(MinHeap *h, HeapItem item) {
    if (h->size == h->cap) {
        int new_cap = (h->cap == 0) ? 16 : h->cap * 2;
        HeapItem *new_data = (HeapItem*)realloc(h->data, (size_t)new_cap * sizeof(HeapItem));
        if (!new_data) {
            fprintf(stderr, "Out of memory in heap_push\n");
            exit(1);
        }
        h->data = new_data;
        h->cap = new_cap;
    }

    int i = h->size++;
    h->data[i] = item;

    /* bubble up */
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (heap_item_less(h->data[i], h->data[parent])) {
            HeapItem tmp = h->data[i];
            h->data[i] = h->data[parent];
            h->data[parent] = tmp;
            i = parent;
        } else {
            break;
        }
    }
}

static HeapItem heap_pop(MinHeap *h) {
    if (h->size <= 0) {
        fprintf(stderr, "heap_pop on empty heap\n");
        exit(1);
    }

    /* swap root with last, pop last */
    HeapItem root = h->data[0];
    h->size--;
    if (h->size > 0) {
        h->data[0] = h->data[h->size];

        /* bubble down */
        int i = 0;
        while (1) {
            int left = 2*i + 1;
            int right = 2*i + 2;
            int smallest = i;

            if (left < h->size && heap_item_less(h->data[left], h->data[smallest])) {
                smallest = left;
            }
            if (right < h->size && heap_item_less(h->data[right], h->data[smallest])) {
                smallest = right;
            }
            if (smallest == i) break;

            HeapItem tmp = h->data[i];
            h->data[i] = h->data[smallest];
            h->data[smallest] = tmp;
            i = smallest;
        }
    }
    return root;
}

static bool heap_is_empty(const MinHeap *h) {
    return h->size == 0;
}

/* ---------------- Graph (adjacency lists) ---------------- */
typedef struct {
    int to;
    double w;
} Edge;

typedef struct {
    Edge *edges;
    int size;
    int cap;
} AdjList;

typedef struct {
    int n;
    AdjList *adj; /* length n */
} Graph;

static void adj_init(AdjList *a) {
    a->edges = NULL;
    a->size = 0;
    a->cap = 0;
}

static void adj_free(AdjList *a) {
    free(a->edges);
    a->edges = NULL;
    a->size = 0;
    a->cap = 0;
}

static void adj_push(AdjList *a, int to, double w) {
    if (a->size == a->cap) {
        int new_cap = (a->cap == 0) ? 8 : a->cap * 2;
        Edge *new_edges = (Edge*)realloc(a->edges, (size_t)new_cap * sizeof(Edge));
        if (!new_edges) {
            fprintf(stderr, "Out of memory in adj_push\n");
            exit(1);
        }
        a->edges = new_edges;
        a->cap = new_cap;
    }
    a->edges[a->size].to = to;
    a->edges[a->size].w = w;
    a->size++;
}

static Graph *graph_create(int n) {
    Graph *g = (Graph*)malloc(sizeof(Graph));
    if (!g) { fprintf(stderr, "Out of memory creating graph\n"); exit(1); }
    g->n = n;
    g->adj = (AdjList*)malloc((size_t)n * sizeof(AdjList));
    if (!g->adj) { fprintf(stderr, "Out of memory creating adj lists\n"); exit(1); }
    for (int i = 0; i < n; i++) adj_init(&g->adj[i]);
    return g;
}

static void graph_free(Graph *g) {
    if (!g) return;
    for (int i = 0; i < g->n; i++) adj_free(&g->adj[i]);
    free(g->adj);
    free(g);
}

/* ---------------- Graph generators ---------------- */

/* complete_weighted_graph(n): vertices 0..n-1, edge weights uniform in [0,1] */
static Graph *complete_weighted_graph(int n) {
    Graph *g = graph_create(n);
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double w = rand01();
            adj_push(&g->adj[i], j, w);
            adj_push(&g->adj[j], i, w);
        }
    }
    return g;
}

/* hypercube(n) as in your Python: connect i<j if (j-i) is a power of 2 */
static Graph *hypercube_graph(int n) {
    Graph *g = graph_create(n);
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            int d = j - i;
            if ((d & (d - 1)) == 0) { /* power of 2 */
                double w = rand01();
                adj_push(&g->adj[i], j, w);
                adj_push(&g->adj[j], i, w);
            }
        }
    }
    return g;
}

/* Points for geometric graphs */
typedef struct {
    double x, y, z, t;
} Point4;

/* unit_square_graph(n): complete graph on random points in [0,1]^2 weighted by Euclidean distance */
static Graph *unit_square_graph(int n) {
    Point4 *pts = (Point4*)malloc((size_t)n * sizeof(Point4));
    if (!pts) { fprintf(stderr, "Out of memory for points\n"); exit(1); }

    for (int i = 0; i < n; i++) {
        pts[i].x = rand01();
        pts[i].y = rand01();
    }

    Graph *g = graph_create(n);
    for (int i = 0; i < n; i++) {
        double x1 = pts[i].x, y1 = pts[i].y;
        for (int j = i + 1; j < n; j++) {
            double x2 = pts[j].x, y2 = pts[j].y;
            double dx = x1 - x2, dy = y1 - y2;
            double w = sqrt(dx*dx + dy*dy);
            adj_push(&g->adj[i], j, w);
            adj_push(&g->adj[j], i, w);
        }
    }

    free(pts);
    return g;
}

/* unit_cube_graph(n): random points in [0,1]^3, weighted by Euclidean distance */
static Graph *unit_cube_graph(int n) {
    Point4 *pts = (Point4*)malloc((size_t)n * sizeof(Point4));
    if (!pts) { fprintf(stderr, "Out of memory for points\n"); exit(1); }

    for (int i = 0; i < n; i++) {
        pts[i].x = rand01();
        pts[i].y = rand01();
        pts[i].z = rand01();
    }

    Graph *g = graph_create(n);
    for (int i = 0; i < n; i++) {
        double x1 = pts[i].x, y1 = pts[i].y, z1 = pts[i].z;
        for (int j = i + 1; j < n; j++) {
            double x2 = pts[j].x, y2 = pts[j].y, z2 = pts[j].z;
            double dx = x1 - x2, dy = y1 - y2, dz = z1 - z2;
            double w = sqrt(dx*dx + dy*dy + dz*dz);
            adj_push(&g->adj[i], j, w);
            adj_push(&g->adj[j], i, w);
        }
    }

    free(pts);
    return g;
}

/* unit_hypercube_graph(n): random points in [0,1]^4, weighted by Euclidean distance */
static Graph *unit_hypercube_graph(int n) {
    Point4 *pts = (Point4*)malloc((size_t)n * sizeof(Point4));
    if (!pts) { fprintf(stderr, "Out of memory for points\n"); exit(1); }

    for (int i = 0; i < n; i++) {
        pts[i].x = rand01();
        pts[i].y = rand01();
        pts[i].z = rand01();
        pts[i].t = rand01();
    }

    Graph *g = graph_create(n);
    for (int i = 0; i < n; i++) {
        double x1 = pts[i].x, y1 = pts[i].y, z1 = pts[i].z, t1 = pts[i].t;
        for (int j = i + 1; j < n; j++) {
            double x2 = pts[j].x, y2 = pts[j].y, z2 = pts[j].z, t2 = pts[j].t;
            double dx = x1 - x2, dy = y1 - y2, dz = z1 - z2, dt = t1 - t2;
            double w = sqrt(dx*dx + dy*dy + dz*dz + dt*dt);
            adj_push(&g->adj[i], j, w);
            adj_push(&g->adj[j], i, w);
        }
    }

    free(pts);
    return g;
}

/* ---------------- Prim's MST ---------------- */
typedef struct {
    int u, v;
    double w;
} MstEdge;

typedef struct {
    MstEdge *edges;
    int size;
    int cap;
    double total_weight;
} MstResult;

static void mst_init(MstResult *r) {
    r->edges = NULL;
    r->size = 0;
    r->cap = 0;
    r->total_weight = 0.0;
}

static void mst_free(MstResult *r) {
    free(r->edges);
    r->edges = NULL;
    r->size = 0;
    r->cap = 0;
    r->total_weight = 0.0;
}

static void mst_push_edge(MstResult *r, int u, int v, double w) {
    if (r->size == r->cap) {
        int new_cap = (r->cap == 0) ? 64 : r->cap * 2;
        MstEdge *new_edges = (MstEdge*)realloc(r->edges, (size_t)new_cap * sizeof(MstEdge));
        if (!new_edges) {
            fprintf(stderr, "Out of memory in mst_push_edge\n");
            exit(1);
        }
        r->edges = new_edges;
        r->cap = new_cap;
    }
    r->edges[r->size].u = u;
    r->edges[r->size].v = v;
    r->edges[r->size].w = w;
    r->size++;
    r->total_weight += w;
}

static MstResult prim_mst(const Graph *g) {
    MstResult res;
    mst_init(&res);

    int n = g->n;
    if (n == 0) return res;

    bool *visited = (bool*)calloc((size_t)n, sizeof(bool));
    if (!visited) { fprintf(stderr, "Out of memory for visited\n"); exit(1); }

    MinHeap heap;
    heap_init(&heap);

    int start = 0; /* Python: next(iter(G)) -> 0 for our representation */
    visited[start] = true;
    int visited_count = 1;

    /* add start edges */
    for (int i = 0; i < g->adj[start].size; i++) {
        Edge e = g->adj[start].edges[i];
        HeapItem it = { .w = e.w, .u = start, .v = e.to };
        heap_push(&heap, it);
    }

    while (!heap_is_empty(&heap) && visited_count < n) {
        HeapItem it = heap_pop(&heap);
        double w = it.w;
        int u = it.u;
        int v = it.v;

        if (visited[v]) continue;

        visited[v] = true;
        visited_count++;
        mst_push_edge(&res, u, v, w);

        /* push outgoing edges */
        for (int i = 0; i < g->adj[v].size; i++) {
            Edge e = g->adj[v].edges[i];
            if (!visited[e.to]) {
                HeapItem nxt = { .w = e.w, .u = v, .v = e.to };
                heap_push(&heap, nxt);
            }
        }
    }

    heap_free(&heap);
    free(visited);
    return res;
}

/* ---------------- main benchmark ---------------- */
int main(void) {
    /* seed RNG (change to fixed seed if you want reproducibility) */
    srand((unsigned)time(NULL));

    int sizes[] = {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
    int num_sizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
    int trials = 5;

    for (int si = 0; si < num_sizes; si++) {
        int n = sizes[si];
        double total_time = 0.0;

        for (int t = 0; t < trials; t++) {
            double start = now_seconds();

            /* Choose the graph generator you want: */
            /*Graph *g = complete_weighted_graph(n); */
            /*Graph *g = hypercube_graph(n); */
            /*Graph *g = unit_square_graph(n); */
            /*Graph *g = unit_cube_graph(n); */
            Graph *g = unit_hypercube_graph(n); 

            MstResult mst = prim_mst(g);

            graph_free(g);
            mst_free(&mst);

            double end = now_seconds();
            total_time += (end - start);
        }

        double avg = total_time / (double)trials;
        printf("n = %-6d | Average runtime over %d runs: %.4f seconds\n", n, trials, avg);
        fflush(stdout);
    }

    return 0;
}
