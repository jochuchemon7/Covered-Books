def edge_converter(graph):
    change = dict()
    val = 0
    for node in graph.keys():
        change[node] = val
        val += 1
    sample = []
    for start, rest in graph.items():
        for end in rest.keys():
            sample.append([change[start], change[end], graph[start][end]])
    return sample

def adjacent_matrix(graph):
    matrix = [[0 for _ in range(len(graph.keys()))] for _ in range(len(graph.keys()))]
    sample = edge_converter(graph)
    for edge in sample:
        matrix[edge[0]][edge[1]] = edge[2]
    return matrix

def BFS(matrix, source, sink, parent):
    visited = [False] * len(matrix)
    queue = []
    visited[source] = True
    queue.append(source)

    while queue:
        neighbor = queue.pop(0)
        for to in range(len(matrix[neighbor])):
            if visited[to] is False and matrix[neighbor][to] > 0:
                visited[to] = True
                queue.append(to)
                parent[to] = neighbor
    return True if visited[sink] else False


def ford_fulkerson(matrix, source, sink):
    max_flow = 0
    parent = [-1] * len(matrix)

    while BFS(matrix, source, sink, parent):
        path_flow = float('Inf')
        s = sink
        while s != source:
            path_flow = min(path_flow, matrix[parent[s]][s])
            s = parent[s]
        v = sink
        max_flow += path_flow
        while v != source:
            u = parent[v]
            matrix[u][v] -= path_flow
            matrix[v][u] += path_flow
            v = parent[v]
    return max_flow



graph = {'S': {'A': 7, 'B': 8},
         'A': {'B': 2, 'C': 5},
         'B': {'D': 10},
         'C': {'D': 2, 'T': 3},
         'D': {'T': 12},
         'T': {}
         }

matrix = adjacent_matrix(graph)
source, sink = 0, len(matrix)-1
max_flow = ford_fulkerson(matrix, source, sink)
print(max_flow)