
def BFS(matrix, source, sink, parent):
    visited = [False] * len(matrix)
    queue = []

    queue.append(source)
    visited[source] = True

    while queue:
        neighbor = queue.pop(0)
        for to in range(len(matrix[neighbor])):
            if visited[to] is False and matrix[neighbor][to] > 0:
                queue.append(to)
                visited[to] = True
                parent[to] = neighbor

    return True if visited[sink] else False


def Ford_Fulkerson(matrix, source, sink):
    parent = [-1] * (len(matrix))
    max_flow = 0

    while BFS(matrix, source, sink, parent):
        path_flow = float('Inf')
        s = sink

        while s != source:
            path_flow = min(path_flow, matrix[parent[s]][s])
            s = parent[s]

        max_flow += path_flow
        v = sink

        while v != source:
            u = parent[v]
            matrix[u][v] -= path_flow
            matrix[v][u] += path_flow
            v = parent[v]

    return max_flow


graph = [
    [0, 16, 13, 0, 0, 0],
    [0, 0, 10, 12, 0, 0],
    [0, 4, 0, 0, 14, 0],
    [0, 0, 9, 0, 0, 20],
    [0, 0, 0, 7, 0, 4],
    [0, 0, 0, 0, 0, 0],
]

source, sink = 0, 5
print(Ford_Fulkerson(graph, source, sink))