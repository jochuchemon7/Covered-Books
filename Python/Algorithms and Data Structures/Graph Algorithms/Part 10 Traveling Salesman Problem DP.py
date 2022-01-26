import copy
import time
import numpy as np
import matplotlib.pyplot as plt

def generate_matrix(num_points):
    coordinates = np.random.randint(0, 100, size = (num_points, 2))
    matrix = np.zeros(shape=(num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            matrix[i][j] = np.linalg.norm(coordinates[i] - coordinates[j])
    return coordinates, matrix

def plot_path(coordinates, min_path):
    counter = 0
    for p1, p2 in zip(min_path[:-1], min_path[1:]):
        p1 -= 1
        p2 -= 1
        if counter == 0:
            plt.plot(coordinates[p1,0], coordinates[p1, 1], marker = 'D', color = 'red')
            plt.plot([coordinates[p1, 0], coordinates[p2, 0]], [coordinates[p1, 1], coordinates[p2,1]])
            counter += 1
        else:
            plt.plot([coordinates[p1, 0], coordinates[p2, 0]], [coordinates[p1, 1], coordinates[p2, 1]])

def get_minimum(last, visited):
    if (last, visited) in memo:
        return memo[last, visited]

    values = []
    all_min = []

    for node in visited:
        set_visited = copy.deepcopy(list(visited))
        set_visited.remove(node)
        all_min.append([node, tuple(set_visited)])
        result = get_minimum(node, tuple(set_visited))
        values.append(matrix[last-1][node-1] + result)

    memo[last, visited] = min(values)
    path.append(((last, visited), all_min[values.index(memo[last, visited])]))
    return memo[last, visited]


def TSP_DP(matrix):
    data = list(range(1, len(matrix) + 1))
    n = len(data)

    for i in range(1, n):
        memo[i+1, ()] = matrix[i,0]

    min_cost = get_minimum(1, tuple(data[1:]))
    min_path = [1]
    solution = path.pop()
    min_path.append(solution[1][0])
    for x in range(n-2):
        for new_solution in path:
            if tuple(solution[1]) == new_solution[0]:
                solution = new_solution
                min_path.append(solution[1][0])
                break
    min_path.append(1)

    return min_path, min_cost

# Testing

num_points = 13
coordinates, matrix = generate_matrix(num_points)
memo = dict()
path = []

t = time.time()
min_path, min_cost = TSP_DP(matrix)
runtime = round(time.time() - t, 3)
print(f"Found optimal path in {runtime} seconds.")
print(f"Optimal cost: {round(min_cost, 3)}, optimal path: {min_path}")


# Plotting
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.scatter(coordinates[:, 0], coordinates[:, 1])

plt.subplot(1,2,2)
plot_path(coordinates, min_path)

plt.tight_layout()