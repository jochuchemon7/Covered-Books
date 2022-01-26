graph = {'A': ['B', 'C'],
         'B': ['C', 'D'],
         'C': ['D'],
         'D': ['C'],
         'E': ['F'],
         'F': ['C']}

"""
###########     DFS    ##################
"""

# dfs

def dfs(graph, visited, order, node):
    if visited[node] is True:
        return
    else:
        visited[node] = True
        order.append(node)
    neighbour = graph[node]
    for nextnode in neighbour:
        dfs(graph, visited, order, nextnode)
    return visited, order

# DFS TEST
visited = dict()
for i in graph.keys():
    visited[i] = False
order = []
visited, order = dfs(graph, visited, order, list(graph.keys())[0])
for i in graph.keys():
    print("Node: %s - Visited: %s" % (i, visited[i]))
print('Order: ', order)



"""
###########     BFS    ##################
"""

# bfs

def bfs(graph, visited, order, queue, node):
    visited[node] = True
    order.append(node)
    queue.append(node)

    while queue:
        neigbour = queue.pop(0)
        for to in graph[neigbour]:
            if visited[to] is False:
                visited[to] = True
                order.append(to)
                queue.append(to)

    return visited, order

# BFS TEST

visited = dict()
for i in graph.keys():
    visited[i] = False
queue = []
order = []
visited, order = bfs(graph, visited, order, queue, list(graph.keys())[0])
for node in graph.keys():
    print("Node: %s - Visited: %s" % (node, visited[node]))
print('Order: ', order)


"""
###########     TOP SORT    ##################
"""

def topsort(graph):
    visited = dict()
    for i in graph.keys():
        visited[i] = False
    order = []

    def DFS(node):
        if visited[node] is True:
            return
        else:
            visited[node] = True

        for nextnode in graph[node]:
            DFS(nextnode)
        order.append(node)

    for i in graph.keys():
        DFS(i)
    return order

# TOPSORT TEST
sort = topsort(graph)
print(sort)


"""
###########     DIJKSTRA    ##################
"""

# Dijkstra's
import math
from collections import deque

# Primary Queue
class QueueNode(object):
    def __init__(self, value, priority):
        self.value = value
        self.priority = priority

class PriorityQueue:
    def __init__(self):
        self.qlist = list()
    def __len__(self):
        return len(self.qlist)
    def isEmpty(self):
        return len(self.qlist) == 0
    def enqueue(self, value, priority):
        self.qlist.append(QueueNode(value, priority))
    def dequeue(self):
        assert not self.isEmpty(), "Cannot Remove From an Empty Queue"
        highest = self.qlist[0].priority
        index = 0
        for i in range(len(self.qlist)):
            if self.qlist[i].priority < highest:
                highest = self.qlist[i].priority
                index = i
        entry = self.qlist.pop(index)
        return entry.value, entry.priority

# Dijkstra
def dijkstra(graph, source):
    visited = dict()
    prev = dict()
    distance = dict()
    for node in graph.keys():
        visited[node] = False
        distance[node] = math.inf
    distance[source] = 0

    pq = PriorityQueue()
    pq.enqueue(source, 0)

    while pq.__len__() != 0:
        node, mindistance = pq.dequeue()
        visited[node] = True
        for endnode in graph[node].keys():
            if len(graph[node].keys()) == 0:
                continue
            newdistance = distance[node] + graph[node][endnode]
            if newdistance < distance[endnode]:
                distance[endnode] = newdistance
                prev[endnode] = node
                pq.enqueue(endnode, newdistance)
    return prev, distance

def findShortestPath(graph, source, sink):
    previous, distance = dijkstra(graph, source)
    path = deque()
    dest = previous[sink]

    while dest != source:
        path.appendleft(dest)
        dest = previous[dest]

    path.appendleft(sink)
    path.append(source)

    return distance[sink], list(path)

# Dijkstra Test

graph = {"A": {"B": 2, "C": 3},
         "B": {"G": 5, "D": 3},
         "C": {"D": 3, "E": 3, "F": 3},
         "D": {"H": 1},
         "E": {"H": 6},
         "F": {"I": 6},
         "G": {"I": 2},
         "H": {"I": 1},
         "I": {}}

dist, prev = dijkstra(graph, list(graph.keys())[0])
print(findShortestPath(graph, 'A', 'E'))



"""
###########     BELLMAN FORD    ##################
"""

# Bellman Ford
import math

def printPath(end, prev, graph):
    if end not in graph.keys():
        return
    printPath(prev[end], prev, graph)
    print(end, end = ' ')

def bellman(graph, source, N):
    distance = dict()
    prev = dict()
    for node in graph.keys():
        distance[node] = math.inf
        prev[node] = -1
    distance[source] = 0

    for iteration in range(N-1):
        for start, rest in graph.items():
            for end in rest.keys():
                newdistance = distance[start] + graph[start][end]
                if newdistance < distance[end]:
                    distance[end] = newdistance
                    prev[end] = start

    for start, rest in graph.items():
        for end in rest.keys():
            newdistance = distance[start] + graph[start][end]
            if newdistance < distance[end]:
                distance[end] = math.inf * -1

    for i in range(N):
        end = list(graph.keys())[i]
        print('From: %s - To: %s - Cost: %d' % (source, end, distance[end]), end = '.')
        print(' Path: [', end=' ')
        if end != source:
            printPath(end, prev, graph)
        print(' ]')

graph = {"A": {"B": 2, "C": 3},
         "B": {"G": 5, "D": 3},
         "C": {"D": 3, "E": 3, "F": 3},
         "D": {"H": 1},
         "E": {"H": 6},
         "F": {"I": 6},
         "G": {"I": 2},
         "H": {"I": 1},
         "I": {}}

bellman(graph, "A", len(graph.keys()))


"""
###########     FLOYD WARSHALL    ##################
"""

# Floyd Warshall

import math
from itertools import product

def adjacentMatrix(graph):
    change = dict()
    val = 0
    matrix = []
    for node in graph.keys():
        change[node] = val
        val += 1
    for start, rest in graph.items():
        for end in rest.keys():
            if len(rest.keys()) == 0:
                continue
            matrix.append([change[start], change[end], graph[start][end]])
    return matrix

def printPath(distance, next, N):
    print('Pair     Cost     Path')
    for start, end in product(range(N), repeat = 2):
        if start != end:
            path = [start]
            if path[-1] != end:
                path.append(next[path[-1]][end])
                print('%s -> %s  %4d     %s' % (start, end, distance[start][end],
                                                ' -> '.join(str(node+1) for node in path)))

def floydWarshall(graph):
    N = len(graph.keys())
    matrix = adjacentMatrix(graph)
    distance = [[math.inf] * N for _ in range(N)]
    next = [[0] * N for _ in range(N)]
    for i in range(N):
        distance[i][i] = 0

    for start, end, cost in matrix:
        distance[start][end] = cost
        next[start][end] = end

    for middle, start, end in product(range(N), repeat = 3):
        newdistance = distance[start][middle] + distance[middle][end]
        if newdistance < distance[start][end]:
            distance[start][end] = newdistance
            next[start][end] = next[start][middle]
    printPath(distance, next, N)


graph = {'A': {'C': -2},
         'B': {'C': 3, 'A': 4},
         'C': {'D': 2},
         'D': {'B': -1}}

floydWarshall(graph)


"""
###########     BRIDGES    ##################
"""

# Bridges

def findBridges(graph):
    visited = dict()
    lows = dict()
    ids = dict()
    id = 0

    for node in graph.keys():
        visited[node] = False
        lows[node] = 0
        ids[node] = 0

    def dfs(curNode, parent, bridges, id):
        id += 1
        visited[curNode] = True
        ids[curNode] = id
        lows[curNode] = id

        for node in graph[curNode]:
            if node == parent:
                continue
            if not visited[node]:
                dfs(node, curNode, bridges, id)
                lows[curNode] = min(lows[curNode], lows[node])
                if ids[curNode] < lows[node]:
                    bridges.append([curNode, node])
            else:
                lows[curNode] = min(lows[curNode], ids[node])

    bridges = []
    for node in graph.keys():
        if not visited[node]:
            dfs(node, -1, bridges, id)
    print(bridges)


graph = {
    'A': ['B', 'C'],
    'B': ['A', 'C'],
    'C': ['A', 'B', 'D', 'F'],
    'D': ['C', 'E'],
    'E': ['D'],
    'F': ['C', 'G', 'I'],
    'G': ['F', 'H'],
    'H': ['G', 'I'],
    'I': ['F', 'H'],
}


newGraph = {'A': ['B'],
            'B': ['C', 'D'],
            'C': ['A'],
            'D': []}


findBridges(graph)
findBridges(newGraph)


"""
###########     TARJAN    ##################
"""


# Tarjan

def printSCC(lows):
    ids = list(set(list(lows.values())))
    scc = [[] for _ in range(len(ids))]
    for node in lows:
        scc[ids.index(lows[node])].append(node)
    for id in range(len(ids)):
        print('SCC: #%d  -  Nodes: %s' % (id, scc[id]))

def tarjan(graph):
    UNVISITED = -1
    ids = dict()
    lows = dict()
    onStack = dict()
    stack = []
    id = 0
    sccCount = 0

    for node in graph.keys():
        ids[node] = UNVISITED
        lows[node] = 0
        onStack[node] = False

    def dfs(curNode, id, sccCount):
        id += 1
        ids[curNode] = id
        lows[curNode] = id
        onStack[curNode] = True
        stack.append(curNode)

        for nextnode in graph[curNode]:
            if ids[nextnode] == UNVISITED:
                dfs(nextnode, id, sccCount)
            if onStack[nextnode]:
                lows[curNode] = min(lows[curNode], lows[nextnode])

        if ids[curNode] == lows[curNode]:
            while stack.__len__() != 0:
                node = stack.pop()
                onStack[node] = False
                lows[node] = ids[curNode]
                if node == curNode:
                    break
            sccCount += 1


    for node in graph.keys():
        if ids[node] == UNVISITED:
            dfs(node, id, sccCount)
    printSCC(lows)


graph = {'A': ['E', 'B'],
         'B': ['F'],
         'C': ['B', 'G', 'D'],
         'D': ['G'],
         'E': ['A', 'F'],
         'F': ['C', 'G'],
         'G': ['H'],
         'H': ['D']}
tarjan(graph)


"""
###########     TSP_DP    ##################
"""


# TSP_DP
import copy
import time
import numpy as np
import matplotlib.pyplot as plt

def generate_matrix(num_points):
    coordinates = np.random.randint(0, 150, size=(num_points, 2))
    matrix = np.zeros(shape=(num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            matrix[i, j] = np.linalg.norm(coordinates[i] - coordinates[j])
    return coordinates, matrix

def path_plot(coordinates, min_path):
    counter = 0
    for p1, p2 in zip(min_path[:-1], min_path[1:]):
        p1 -= 1
        p2 -= 1
        if counter == 0:
            plt.plot(coordinates[p1, 0], coordinates[p1, 1], marker='D', color='red')
            plt.plot([coordinates[p1, 0], coordinates[p2, 0]], [coordinates[p1, 1], coordinates[p2, 1]])
            counter += 1
        else:
            plt.plot([coordinates[p1, 0], coordinates[p2, 0]], [coordinates[p1, 1], coordinates[p2, 1]])

def get_minimum(first, rest):
    if (first, rest) in memo:
        return memo[first, rest]

    values = []
    all_min = []

    for node in rest:
        to_visit = copy.deepcopy(list(rest))
        to_visit.remove(node)
        all_min.append([node, tuple(to_visit)])
        result = get_minimum(node, tuple(to_visit))
        values.append(matrix[first-1, node-1] + result)

    memo[first, rest] = min(values)
    path.append(((first, rest), all_min[values.index(memo[first, rest])]))
    return memo[first, rest]


def TSP_DP(matrix):
    data = list(range(1, len(matrix)+1))
    n = len(data)

    for i in range(1, n):
        memo[i+1, ()] = matrix[i, 0]

    min_cost = get_minimum(1, tuple(data[1:]))
    min_path = [1]
    solution = path.pop()
    min_path.append(solution[1][0])

    for _ in range(n-2):
        for new_solution in path:
            if tuple(solution[1]) == new_solution[0]:
                solution = new_solution
                min_path.append(solution[1][0])
                break
    min_path.append(1)
    return min_cost, min_path



num_points = 12
coordinates, matrix = generate_matrix(num_points)
memo = dict()
path = []

t = time.time()
min_cost, min_path = TSP_DP(matrix)
runtime = round(time.time() - t, 3)

print(f'Optimal Solution Found in {runtime} seconds')
print(f'Optimal Solution Cost: {round(min_cost, 3)}, Optimal Solution Path: {min_path}')

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(coordinates[:, 0], coordinates[:, 1])

plt.subplot(1, 2, 2)
path_plot(coordinates, min_path)

plt.tight_layout()


"""
###########     EULERIAN PATH AND CIRCUIT    ##################
"""


def dfs(curNode, graph, visited_edge, change, path = []):
    path = path + [curNode]
    for node in graph[curNode]:
        if visited_edge[change[curNode]][change[node]] == False:
            visited_edge[change[curNode]][change[node]], visited_edge[change[node]][change[curNode]] = True, True
            path = dfs(node, graph, visited_edge, change, path)
    return path

def check_circuit_or_path(graph):
    odd_degree_nodes = 0
    odd_node = ''
    for node in graph.keys():
        if node not in graph.keys():
            continue
        if len(graph[node]) % 2 == 1:
            odd_degree_nodes += 1
            odd_node = node
    if odd_degree_nodes == 0:
        return 1, odd_node
    elif odd_degree_nodes == 2:
        return 2, odd_node
    return 3, odd_node


def check_euler(graph, max_node):
    visited_edge = [[False for _ in range(max_node + 1)] for _ in range(max_node + 1)]
    check, odd_node = check_circuit_or_path(graph)
    if check == 3:
        print('Graph is not Eulerian')
        print('No Path')
        return
    start_node = list(graph.keys())[0]
    if check == 2:
        start_node = odd_node
        print('Graph has a Euler Path')
    if check == 1:
        print('Graph has a Euler Circuit')
    change = dict()
    val = 0
    for node in graph.keys():
        change[node] = val
        val += 1
    path = dfs(start_node, graph, visited_edge, change)
    print(path)


graph = {'A': ['E', 'B'],
         'B': ['F'],
         'C': ['B', 'G', 'D'],
         'D': ['G'],
         'E': ['A', 'F'],
         'F': ['C', 'G'],
         'G': ['H'],
         'H': ['D']}

G1 = {
    'A': ['B', 'C', 'D'],
    'B': ['A', 'C'],
    'C': ['A', 'B'],
    'D': ['A', 'E'],
    'E': ['D']
}

max_node = len(graph.keys())
check_euler(graph, max_node)

max_node = len(G1.keys())
check_euler(G1, max_node)


"""
###########     PRIM'S MST    ##################
"""


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

    print('Edge - Cost\n')
    while edge < N -1:
        a = 0
        b = 0
        min = INF
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



"""
###########     FORD-FUKLERSON MAX FLOW    ##################
"""


def edge_converter(graph):
    change = dict()
    val = 0
    sample = []
    for node in graph.keys():
        change[node] = val
        val += 1
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


graph = {'S': {'A': 10, 'B': 5, 'C': 10},
         'A': {'D': 10},
         'B': {'C': 10},
         'C': {'F': 15},
         'D': {'B': 20, 'G': 15},
         'E': {'B': 15, 'D': 3},
         'F': {'E': 4, 'I': 10},
         'G': {'H': 10, 'T': 15},
         'H': {'E': 10, 'F': 7},
         'I': {'T': 10},
         'T': {}
         }
matrix = adjacent_matrix(graph)
source, sink = 0, len(matrix) - 1
max_flow = Ford_Fulkerson(matrix, source, sink)
print(max_flow)


graph = {'S': {'A': 7, 'B': 8},
         'A': {'B': 2, 'C': 5},
         'B': {'D': 10},
         'C': {'D': 2, 'T': 3},
         'D': {'T': 12},
         'T': {}
         }
matrix = adjacent_matrix(graph)
source, sink = 0, len(matrix) - 1
max_flow = Ford_Fulkerson(matrix, source, sink)
print(max_flow)