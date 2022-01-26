def dfs(u, graph, visited_edge, path=[]):
    path = path + [u]
    for v in graph[u]:
        if visited_edge[u][v] == False:
            visited_edge[u][v], visited_edge[v][u] = True, True
            path = dfs(v, graph, visited_edge, path)
    return path


# for checking in graph has euler path or circuit
def check_circuit_or_path(graph, max_node):
    odd_degree_nodes = 0
    odd_node = -1
    for i in range(max_node):
        if i not in graph.keys():
            continue
        if len(graph[i]) % 2 == 1:
            odd_degree_nodes += 1
            odd_node = i
    if odd_degree_nodes == 0:
        return 1, odd_node
    if odd_degree_nodes == 2:
        return 2, odd_node
    return 3, odd_node


def check_euler(graph, max_node):
    visited_edge = [[False for _ in range(max_node + 1)] for _ in range(max_node + 1)]
    check, odd_node = check_circuit_or_path(graph, max_node)
    if check == 3:
        print("graph is not Eulerian")
        print("no path")
        return
    start_node = 1
    if check == 2:
        start_node = odd_node
        print("graph has a Euler path")
    if check == 1:
        print("graph has a Euler cycle")
    path = dfs(start_node, graph, visited_edge)
    print(path)
    return visited_edge


graph = {'A': ['E', 'B'],
         'B': ['F'],
         'C': ['B', 'G', 'D'],
         'D': ['G'],
         'E': ['A', 'F'],
         'F': ['C', 'G'],
         'G': ['H'],
         'H': ['D']}

G1 = {
    1: [2, 3, 4],
    2: [1, 3],
    3: [1, 2],
    4: [1, 5],
    5: [4]
}

max_node = 10
visited_edge = check_euler(G1, max_node)