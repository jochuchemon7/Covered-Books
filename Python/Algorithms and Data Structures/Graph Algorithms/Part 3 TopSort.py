# Fist Implementation
graph = {'A': ['B', 'C'],
         'B': ['C', 'D'],
         'C': ['D'],
         'D': ['C'],
         'E': ['F'],
         'F': ['C']}


def toposort(graph):
    visited = dict()
    for i in graph.keys():
        visited[i] = False
    result = []

    def DFS(node):
        if visited[node]:
            return
        visited[node] = True
        for adj in graph[node]:
            DFS(adj)
        result.append(node)

    for i in graph.keys():
        DFS(i)

    return result

sort = toposort(graph)
print(sort)


# Second Implementation (Not So Great)

#  edges = {"a": ["c", "b"], "b": ["d", "e"], "c": [], "d": [], "e": []}
#  vertices = ["a", "b", "c", "d", "e"]

edges = {'A': ['B', 'C'],
         'B': ['C', 'D'],
         'C': ['D'],
         'D': ['C'],
         'E': ['F'],
         'F': ['C']}
vertices = list(edges.keys())




def topological_sort(start, visited, sort):
    """Perform topolical sort on a directed acyclic graph."""
    current = start
    # add current to visited
    visited.append(current)
    neighbors = edges[current]
    for neighbor in neighbors:
        # if neighbor not in visited, visit
        if neighbor not in visited:
            sort = topological_sort(neighbor, visited, sort)
    # if all neighbors visited add current to sort
    sort.append(current)
    # if all vertices haven't been visited select a new one to visit
    if len(visited) != len(vertices):
        for vertice in vertices:
            if vertice not in visited:
                sort = topological_sort(vertice, visited, sort)
    # return sort
    return sort


newSort = topological_sort("A", [], [])
print(newSort)

#############################################

# Final Implementation (Prefer)

graph = {'A': ['B', 'C'],
         'B': ['C', 'D'],
         'C': ['D'],
         'D': ['C'],
         'E': ['F'],
         'F': ['C']}

def toposort(graph):
    visited = dict()
    for i in graph.keys():
        visited[i] = False
    result = []

    def DFS(node):
        if visited[node]:
            return
        visited[node] = True
        for adj in graph[node]:
            DFS(adj)
        result.append(node)

    for i in graph.keys():
        DFS(i)
    return result

sort = toposort(graph)
print(sort)