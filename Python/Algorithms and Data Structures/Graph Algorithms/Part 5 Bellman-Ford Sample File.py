import math
import sys

# Recursive function to print the path of a given vertex from source vertex
def printPath(parent, vertex):
    if vertex < 0:
        return
    printPath(parent, parent[vertex])
    print(vertex, end=' ')

# Function to run the Bellman–Ford algorithm from a given source
def bellmanFord(edges, source, N):
    # `distance[]` and `parent[]` stores the shortest path (least cost/path) info
    distance = [sys.maxsize] * N
    parent = [-1] * N
    # Initially, all vertices except source vertex weight INFINITY and no parent
    distance[source] = 0
    # relaxation step (run `V-1` times)
    for k in range(N - 1):
        # edge from `u` to `v` having weight `w`
        for (u, v, w) in edges:
            # if the distance to destination `v` can be
            # shortened by taking edge `u —> v`
            if distance[u] + w < distance[v]:
                # update distance to the new lower value
                distance[v] = distance[u] + w
                # set v's parent as `u`
                parent[v] = u

    # run relaxation step once more for N'th time to
    # check for negative-weight cycles
    for (u, v, w) in edges:  # edge from `u` to `v` having weight `w`
        # if the distance to destination `u` can be
        # shortened by taking edge `u —> v`
        if distance[u] + w < distance[v]:
            print("Negative-weight cycle is found!!")
            return

    for i in range(N):
        print("The distance of vertex", i, "from the source is", distance[i], end='.')
        print(" Its path is [ ", end='')
        printPath(parent, i)
        print("]")



edges = [
    # `(x, y, w)` —> edge from `x` to `y` having weight `w`
    (0, 1, -1), (0, 2, 4), (1, 2, 3), (1, 3, 2),
    (1, 4, 2), (3, 2, 5), (3, 1, 1), (4, 3, -3)
]
N = 5
source = 0
bellmanFord(edges, source, N)

#############################################

# Bellman-Ford Book/Final Implementation  (Prefer)
import math


def newPrintPath(parent, end, graph):
    if end not in graph.keys():
        return

    newPrintPath(parent, parent[end], graph)
    print(end, end=' ')

def newBellmanFord(graph, initial, N):
    distance = dict()
    for i in graph.keys():
        distance[i] = math.inf
    parent = dict()
    for i in graph.keys():
        parent[i] = -1
    distance[initial] = 0

    for k in range(N-1):
        for start, rest in graph.items():
            for end in rest.keys():
                # print("From: %s - To: %s - Cost: %d " % (start, end, rest[end]))
                if distance[start] + rest[end] < distance[end]:
                    distance[end] = distance[start] + rest[end]
                    parent[end] = start

    for start, rest in graph.items():
        for end in rest.keys():
            if distance[start] + rest[end] < distance[end]:
                distance[end] = math.inf * -1

    for i in range(N):
        end = list(graph.keys())[i]
        print("From: %s  -  To: %s  -  Cost: %d" % (initial, end, distance[end]), end = '.')
        #print("The distance of vertex", index, "from the source is", distance[index], end='.')
        print(" Its path is [ ", end='')
        if end != initial:
            newPrintPath(parent, end, graph)
        print("]")


graph = {"A": {"B": 2, "C": 3},
         "B": {"G": 5, "D": 3},
         "C": {"D": 3, "E": 3, "F": 3},
         "D": {"H": 1},
         "E": {"H": 6},
         "F": {"I": 6},
         "G": {"I": 2},
         "H": {"I": 1},
         "I": {}}

newBellmanFord(graph, "A", len(graph.keys()))
