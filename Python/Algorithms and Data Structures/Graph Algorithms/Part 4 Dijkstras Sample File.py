# Dijkstra's Algorithm (Website Implementation)
from collections import defaultdict, deque

class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(list)
        self.distances = {}
    def add_node(self, value):
        self.nodes.add(value)
    def add_edge(self, from_node, to_node, distance):
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.distances[(from_node, to_node)] = distance


def dijsktra(graph, initial):
    visited = {initial: 0}
    path = {}
    nodes = set(graph.nodes)

    while nodes:
        min_node = None
        for node in nodes:
            if node in visited:
                if min_node is None:
                    min_node = node
                elif visited[node] < visited[min_node]:
                    min_node = node
        if min_node is None:
            break

        nodes.remove(min_node)
        current_weight = visited[min_node]

        for edge in graph.edges[min_node]:
            try:
                weight = current_weight + graph.distances[(min_node, edge)]
            except:
                continue
            if edge not in visited or weight < visited[edge]:
                visited[edge] = weight
                path[edge] = min_node
    return visited, path


def shortest_path(graph, origin, destination):
    visited, paths = dijsktra(graph, origin)
    full_path = deque()
    _destination = paths[destination]

    while _destination != origin:
        full_path.appendleft(_destination)
        _destination = paths[_destination]

    full_path.appendleft(origin)
    full_path.append(destination)

    return visited[destination], list(full_path)


graph = {"A": {"B": 2, "C": 3},
         "B": {"G": 5, "D": 3},
         "C": {"D": 3, "E": 3, "F": 3},
         "D": {"H": 1},
         "E": {"H": 6},
         "F": {"I": 6},
         "G": {"I": 2},
         "H": {"I": 1},
         "I": {}}

sampleGraph = Graph()
for node in graph.keys():
    for to in graph[node].keys():
        print("From: %s - To: %s - Cost: %d" % (node, to, graph[node][to]))

for node in graph.keys():
    sampleGraph.add_node(node)
    for to in graph[node].keys():
        sampleGraph.add_edge(node, to, graph[node][to])

visited, path = dijsktra(sampleGraph, list(graph.keys())[0])
print(shortest_path(sampleGraph, 'A', 'D'))


#############################################

# Dijkstra's Book/Final Implementation  (Prefer)

# Priority Queue
import math
from collections import deque

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
        assert not self.isEmpty(), "Cannot Remove from empty Queue"
        highest = self.qlist[0].priority
        index = 0
        for i in range(len(self.qlist)):
            if self.qlist[i].priority < highest:
                highest = self.qlist[i].priority
                index = i
        entry = self.qlist.pop(index)
        return entry.value, entry.priority


# Dijkstra's

def newDijkstra(graph, initial):
    visit = dict()
    dist = dict()
    prev = dict()  # Prev
    for i in graph.keys():
        visit[i] = False
        dist[i] = math.inf
    dist[initial] = 0

    pq = PriorityQueue()
    pq.enqueue(initial, 0)
    while pq.__len__() != 0:
        index, minValue = pq.dequeue()
        visit[index] = True
        for edge in graph[index].keys():
            if len(graph[index].keys()) == 0:
                continue
            newDist = dist[index] + graph[index][edge]
            if newDist < dist[edge]:
                prev[edge] = index  # Prev
                dist[edge] = newDist
                pq.enqueue(edge, newDist)
    return dist, prev


def findShortestPath(graph, source, sink):
    distance, previous = newDijkstra(graph, source)
    path = deque()
    dest = previous[sink]

    while dest != source:
        path.appendleft(dest)
        dest = previous[dest]

    path.appendleft(source)
    path.append(sink)

    return distance[sink], list(path)



graph = {"A": {"B": 2, "C": 3},
         "B": {"G": 5, "D": 3},
         "C": {"D": 3, "E": 3, "F": 3},
         "D": {"H": 1},
         "E": {"H": 6},
         "F": {"I": 6},
         "G": {"I": 2},
         "H": {"I": 1},
         "I": {}}

dist, prev = newDijkstra(graph, list(graph.keys())[0])
print(findShortestPath(graph, 'A', 'E'))