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


def prims_mst(matrix):
    INF = 9999999
    N = len(matrix)
    visited = [False for _ in range(N)]
    edge = 0
    visited[0] = True

    print('Edge : Weight\n')
    while (edge < N -1):
        min = INF
        a = 0
        b = 0
        for m in range(N):
            if visited[m]:
                for n in range(N):
                    if ((visited[n] == False) and matrix[m][n]):
                        if min > matrix[m][n]:
                            min = matrix[m][n]
                            a = m
                            b = n
        print(str(a) + '-' + str(b) + ':' + str(matrix[a][b]))
        visited[b] = True
        edge += 1


graph = {'A': {'B': 6, 'D': 3},
         'B': {'C': 4, 'E': 3, 'A': 6},
         'C': {'B': 4, 'F': 12},
         'D': {'A': 3, 'E': 1, 'G': 8},
         'E': {'B': 2, 'D': 1, 'F': 7, 'H': 9},
         'F': {'C': 12, 'E': 7, 'I': 10},
         'G': {'D': 8, 'H': 11},
         'H': {'E': 9, 'G': 11, 'I': 5},
         'I': {'F': 10, 'H': 5}}

matrix = adjacent_matrix(graph)
prims_mst(matrix)