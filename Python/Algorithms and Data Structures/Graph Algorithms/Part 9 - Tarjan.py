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