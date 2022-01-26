# Finding Bridges Book/Final Implementation (Prefer)

def findBridges(graph):
    id = 0
    n = len(graph.keys())
    lows = [0] * n
    visited = [False] * n

    def dfs(at, parent, bridges, id):
        visited[at] = True
        lows[at] = id
        id += 1

        for to in graph[at]:
            if to == parent:
                pass
            elif not visited[to]:
                dfs(to, at, bridges, id)
                lows[at] = min(lows[at], lows[to])
                if at < lows[to]:
                    bridges.append([at, to])
            else:
                lows[at] = min(lows[at], to)

    bridges = []
    for i in range(n):
        if not visited[i]:
            dfs(i, -1, bridges, id)
    print(bridges)

# Testing

graph = {
    0: [1, 2],
    1: [0, 2],
    2: [0, 1, 3, 5],
    3: [2, 4],
    4: [3],
    5: [2, 6, 8],
    6: [5, 7],
    7: [6, 8],
    8: [5, 7],
}

new = [(0, 1), (1, 2), (2, 0), (1, 3)]
edges = {0: [1],
         1: [2, 3],
         2: [0],
         3: [], }

findBridges(edges)
findBridges(graph)


