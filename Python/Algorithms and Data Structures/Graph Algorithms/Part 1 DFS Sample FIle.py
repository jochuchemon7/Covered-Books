# DSF First Implementation
graph = {'A': ['B', 'C'],
         'B': ['C', 'D'],
         'C': ['D'],
         'D': ['C'],
         'E': ['F'],
         'F': ['C']}
visited = dict()
for i in graph.keys():
    visited[i] = False


# DSF Function
def dfs(at):
    if visited[at]:
        return
    else:
        visited[at] = True
    neighbors = graph[at]
    for next in neighbors:
        dfs(next)


# Test DFS
dfs(list(graph.keys()[0]))  # Start from first Node
for key in visited:
    print("Node: %s  Visited: %s" % (key, visited[key]))


# Third Web Implementation

graph = {'A': ['B','C'],
         'B': ['D', 'E'],
         'C': ['F'],
         'D': [],
         'E': ['F'],
         'F': []}
visited = set()

def dfs(visited, graph, node):
    if node not in visited:
        print(node)
        visited.add(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)


# Test
dfs(visited, graph, 'A')

#############################################

# Final Implementation (Prefer)

graph = {'A': ['B', 'C'],
         'B': ['C', 'D'],
         'C': ['D'],
         'D': ['C'],
         'E': ['F'],
         'F': ['C']}

visited = dict()
for i in graph.keys():
    visited[i] = False
order = []
def DFS(node):
    if visited[node] is True:
        return
    else:
        order.append(node)
        visited[node] = True

    neighboors = graph[node]
    for i in neighboors:
        DFS(i)

DFS(list(graph.keys())[0])
for key in graph.keys():
    print("Node: %s  Visited: %s" % (key, visited[key]))
print("Visit Order: ", order)
