# BFS First Implementation
graph = {
  'A' : ['B','C'],
  'B' : ['D', 'E'],
  'C' : ['F'],
  'D' : [],
  'E' : ['F'],
  'F' : []
}

visited = [] # List to keep track of visited nodes.
queue = []  #  Initialize a queue

def bfs(visited, graph, node):
  visited.append(node)
  queue.append(node)

  while queue:
    s = queue.pop(0)

    for neighbour in graph[s]:
      if neighbour not in visited:
        visited.append(neighbour)
        queue.append(neighbour)

# Driver Code
bfs(visited, graph, 'A')


# BSF Second Implementation

graph = {'A': ['B', 'C'],
         'B': ['C', 'D'],
         'C': ['D'],
         'D': ['C'],
         'E': ['F'],
         'F': ['C']}

visited = []
queue = []
def bfs(visited, graph, node):
  visited.append(node)
  queue.append(node)
  while queue:
    s = queue.pop(0)
    for neighbour in graph[s]:
      if neighbour not in visited:
        visited.append(neighbour)
        queue.append(neighbour)
bfs(visited, graph, 'A')
print(visited)

#############################################

# Final Implementation (Prefer)
graph = {
  'A' : ['B','C'],
  'B' : ['D', 'E'],
  'C' : ['F'],
  'D' : [],
  'E' : ['F'],
  'F' : []
}

visited = dict()
for key in graph.keys():
  visited[key] = False
order = []
queue = []

def newBFS(visited, graph, node):
  visited[node] = True
  queue.append(node)
  order.append(node)

  while queue:
    neighbour = queue.pop(0)
    for key in graph[neighbour]:
      if visited[key] is False:
        visited[key] = True
        order.append(key)
        queue.append(key)

newBFS(visited, graph, list(graph.keys())[0])
for key in graph.keys():
  print("Node: %s  Visited: %s" % (key, visited[key]))
print("Order: ", order)