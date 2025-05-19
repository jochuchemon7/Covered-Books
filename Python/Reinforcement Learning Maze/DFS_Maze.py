import matplotlib.pyplot as plt
import matplotlib.animation as animation
from MazeEnv import CreateMaze
import numpy as np

# - MAZE CREATION -
env = CreateMaze()
nrow, ncol = np.random.randint(20, 35),  np.random.randint(20, 35)
env.new_maze(50, 50)
env.make()
env.render()

# - VISITED DICTIONARY CREATION -
wall = 'w'
visited = dict()
for i in range(env.get_maze().shape[0]):
    for j in range(env.get_maze().shape[1]):
        visited[(i, j)] = False


# - DFS ALGORITHM -
def dfs(maze, visited, node, target_node, order):
    if node[0] == target_node[0] and node[1] == target_node[1]:
        return True
    else:
        visited[(node[0], node[1])] = True
        order.append([node[0], node[1]])

    for adjacent_x, adjacent_y in [(-1, 0), (0, 1), (1, 0), (0, -1)]:  # Top, right, bottom and left neighbors
        new_node = node[0] + adjacent_x, node[1] + adjacent_y
        if 0 <= new_node[0] < len(maze) and 0 <= new_node[1] < len(maze[0]) and not visited[(new_node[0], new_node[1])] and maze[new_node[0], new_node[1]] != wall:
            if dfs(maze, visited, new_node, target_node, order):
                return True, visited, order
    return False


# - DFS CALL -
start_node = env.get_start_node()
target_node = env.get_target_node()
order = []
demo_found, demo_result, new_order = dfs(env.maze, visited, start_node, target_node, order)


# - VIEW ANIMATION -
if demo_found:
    demo_solution = np.zeros((env.get_maze().shape[0], env.get_maze().shape[1]))
    fig = plt.figure('DFS')
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


# - FINAL DISPLAY -
plt.imshow(demo_solution)