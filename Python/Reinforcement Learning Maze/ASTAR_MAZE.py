import matplotlib.pyplot as plt
import matplotlib.animation as animation
from MazeEnv import CreateMaze
import numpy as np
import heapq

# - MAZE CREATION -
env = CreateMaze()
nrow, ncol = np.random.randint(20, 35), np.random.randint(20, 35)
env.new_maze(50, 50)
env.make()
env.render()

# - VISITED DICTIONARY CREATION -
wall = 'w'
visited_astr = dict()
for i in range(env.get_maze().shape[0]):
    for j in range(env.get_maze().shape[1]):
        visited_astr[(i, j)] = False

# - ASTAR ALGORITHM -
astar_queue = []


# def dfs(maze, visited, node, target_node, order):
def astar(maze, visited_astr, node, target_node, order, path_cost):
    if node[0] == target_node[0] and node[1] == target_node[1]:
        return True
    else:
        visited_astr[(node[0], node[1])] = True
        order.append([node[0], node[1]])

        # Manhattan distance heuristic - distance in "blocks" to target
        h = (target_node[0] - node[0]) + (target_node[1] - node[1])
        node_cost = path_cost + h

        # Push current node location and costs to priority queue data structure - lowest cost node first
        heapq.heappush(astar_queue, (node_cost, [node[0], node[1]], path_cost))

        # Get lowest cost (test) node from queue - move back to the test node if cost is lower than current
        test_node = astar_queue[0]
        test_node_cost = test_node[0]
        test_node_coord = test_node[1]

        if test_node_cost < node_cost:
            node = test_node_coord
            # Reset path cost to test node cost
            path_cost = test_node[2]
            # Pop test node from queue
            astar_queue.pop()

    for adjacent_x, adjacent_y in [(-1, 0), (0, 1), (1, 0), (0, -1)]:  # Top, right, bottom and left neighbors
        new_node = node[0] + adjacent_x, node[1] + adjacent_y

        # Increment path cost to follow-on nodes (+1 for each node opening - do only once)
        if adjacent_x == -1 and adjacent_y == 0:
            path_cost += 1

        if 0 <= new_node[0] < len(maze) and 0 <= new_node[1] < len(maze[0]) and not visited_astr[
            (new_node[0], new_node[1])] and maze[new_node[0], new_node[1]] != wall:
            if astar(maze, visited_astr, new_node, target_node, order, path_cost):
                return True, visited_astr, order
    return False


# - ASTAR CALL -
start_node = env.get_start_node()
target_node = env.get_target_node()
order = []
demo_found, demo_result, new_order = astar(env.maze, visited_astr, start_node, target_node, order, 0)

# - VIEW ANIMATION -
if demo_found:
    demo_solution = np.zeros((env.get_maze().shape[0], env.get_maze().shape[1]))
    fig = plt.figure('A* Solution')
    img = []
    for cell in new_order:
        if demo_result[(cell[0], cell[1])]:
            demo_solution[cell[0], cell[1]] = 1
            img.append([plt.imshow(demo_solution)])

    ani = animation.ArtistAnimation(fig, img, interval=20, blit=True, repeat_delay=0)
    plt.show()


# - ALTERNATIVE ANIMATION VIEW -
if demo_found:
    new_solution = np.array(env.get_maze(), dtype='int')
    new_solution[env.get_target_node()[0], env.get_target_node()[1]] = 2
    fig = plt.figure('DFS')
    img = []
    for cell in new_order:
        if demo_result[(cell[0], cell[1])]:
            new_solution[cell[0], cell[1]] = 3
            img.append([plt.imshow(new_solution)])

    ani = animation.ArtistAnimation(fig, img, interval=20, blit=True, repeat_delay=0)
    plt.show()


plt.imshow(demo_solution)