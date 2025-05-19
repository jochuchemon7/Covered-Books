import matplotlib.animation as animation
from matplotlib import pylab as plt
from MazeEnv import CreateMazeGym
import numpy as np
import random
import torch


""" Q-Learning : Main Training Loop """

# Since makeMove() expects a character, our Q-Learning algorithm only generates numbers we make dict

# - MAZE CREATION -
env = CreateMazeGym()
nrow, ncol = np.random.randint(20, 35),  np.random.randint(20, 35)
env.new_maze(nrow=7, ncol=7)   # BEST 11 x 11
env.make()
env.render()
trainer_maze = env.get_env()

""" ------------------------------------------------------------------------- Q-Network -----------------------------------------------------------------------------------"""

l1 = env.get_maze_().shape[0]*env.get_maze_().shape[1] * 3  # PREVIOUS [3] FOR (AGENT, GOAL, WALLS)
l2 = 480  #480 #round(l1 * 2.5)
l3 = 288 #288 #round(l2 * 0.6)
l4 = 4

model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.ReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.ReLU(),
    torch.nn.Linear(l3, l4)
)
loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
gamma = .9  # Discount Factor
epsilon = 1  # Epsilon for our selection model

action_set = {0: 'top', 1: 'right', 2: 'bottom', 3: 'left'}

# RATE: 1.3888888888888888  (i.e. 6*6*4*1.3888888888888888 = max_steps)

epochs = 500 #500 #355 # 35
max_steps = 250 #250 #355  # 250
losses = []
isAnimated = True
order = list()
rewards = list()
completeRewards = dict()
results = list()
states = list()
detailedState = list()
actions = list()
total_moves = dict()

""" - NOTE: STATE MUST CONTAIN THE PRESENCE OF THE AGENT AND GOAL AND WALLS (IGNORE)- """

""" - TRAINING LOOP - """
for i in range(epochs):

    order = []
    env.set_env(trainer_maze, max_steps)
    flatten_dim = env.get_maze_().shape[0] * env.get_maze_().shape[1] * 3  # PREVIOUS [3] FOR (AGENT, GOAL, WALLS)
    maze_ = env.get_complete_maze().reshape(1, flatten_dim) + np.random.rand(1, flatten_dim) / 10.0
    state1 = torch.from_numpy(maze_).float()
    status, counter = 1, 1
    completeRewards[i] = list()
    while status == 1 and counter < max_steps+1:
        qval = model(state1)
        detailedState.append(state1)
        qval_ = qval.data.numpy()
        if random.random() < epsilon:
            action_ = np.random.randint(0, 4)
        else:
            action_ = np.argmax(qval_)
        action = action_set[action_]
        actions.append(action)
        order.append([env.components['Agent'].position[0], env.components['Agent'].position[1]])
        env.step(action)
        state2_ = env.get_complete_maze().reshape(1, flatten_dim) + np.random.rand(1, flatten_dim) / 10.0
        state2 = torch.from_numpy(state2_).float()
        """ - APPENDING STATE - """
        curr_state = np.floor(state2[0].numpy())
        states.append(curr_state)
        #print(curr_state)
        """ - APPENDING STATE - """
        reward = env.reward()
        rewards.append(reward)
        completeRewards[i].append(reward)
        #print(f'Iter: {counter}, Action: {action}, Reward: {reward}')
        with torch.no_grad():
            newQ = model(state2)
        maxQ = torch.max(newQ)
        if reward != 1:  # If game is not over
            Y = reward + (gamma*maxQ)
        else:
            Y = reward
        Y = torch.Tensor([Y]).detach()
        X = qval.squeeze()[action_]
        loss = loss_fn(X, Y)
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        state1 = state2
        counter += 1
        if reward == 1:
            status = 0
            results.append(status)


    if status == 1:
        results.append(status)

    print(f'Epoch: {i}, Status: {status} (0: Pass!, 1: Failed)')
    total_moves[i] = len(order)
    if epsilon > 0.1:
        epsilon -= (1/epochs)

""" - TRAINING LOOP - """


""" - VIEW LATEST EPISODE - """
if isAnimated:
    demo_solution = env.get_maze()
    demo_solution = demo_solution.tolist()
    demo_solution[env.start_node[0]][env.start_node[1]] = 0.5
    demo_solution[env.target_node[0]][env.target_node[1]] = 0.5
    fig = plt.figure('DFS')
    img = []

    for cell in order:
        demo_solution[cell[0]][cell[1]] = 0.7
        img.append([plt.imshow(demo_solution)])
        demo_solution[cell[0]][cell[1]] = 1
    ani = animation.ArtistAnimation(fig, img, interval=20, blit=True, repeat_delay=0)
    plt.show()


""" Plotting the Loss result from every Epoch """  # When Using random mode layout loss does not find convergence
plt.plot(list(range(len(losses))), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')

""" Plotting the Rewards result from every Epoch """  # When Using random mode layout loss does not find convergence
plt.plot(list(range(len(rewards))), rewards[:len(rewards)])
plt.xlabel('Epochs')
plt.ylabel('Reward')

""" Plotting the Episode length result from every Epoch """  # When Using random mode layout loss does not find convergence
plt.plot(list(range(len(total_moves))), list(total_moves.values()))
plt.xlabel('Epochs')
plt.ylabel('Moves')

print(f'Unique rewards: {np.unique(np.array(rewards))}')


""" - CHECK IF THE STATES ARE THE SAME EVERY TIME THEY ARE PASSED TO THE MODEL (IGNORE) - """
def checkSimilarity(value):
    total = 0
    dictionary = dict()
    compare_to = value
    for i in range(len(states)):
        if sum(detailedState[compare_to][0] == detailedState[i][0]) == torch.tensor(len(detailedState[0][0])):
            total += 1
            dictionary[i] = total
    states_similar_to_first = len(dictionary)
    print(f'Out of the {len(states)} states passed only {states_similar_to_first} were identically to state {compare_to}. Or about {round(states_similar_to_first/len(states),3)*100} percent')
checkSimilarity(value=0)
list(map(checkSimilarity, range(10)))
""" - CHECK IF THE STATES ARE THE SAME EVERY TIME THEY ARE PASSED TO THE MODEL (IGNORE) - """



""" - TESTING MODEL - """
def test_model(model, animated=True):
    final_order = list()
    env.set_env(trainer_maze, max_steps)
    flatten_dim = env.get_maze_().shape[0] * env.get_maze_().shape[1] * 3  # PREVIOUS [3] FOR (AGENT, GOAL, WALLS)
    maze_ = env.get_complete_maze().reshape(1, flatten_dim) + np.random.rand(1, flatten_dim) / 10.0
    state = torch.from_numpy(maze_).float()
    status, counter = 1, 1
    i = 0

    while (status == 1):
        qval = model(state)
        qval_ = qval.data.numpy()
        action_ = np.argmax(qval_)  # NO EPSILON ACTION SELECTION ONLY THE BEST SCORE
        action = action_set[action_]
        if animated:
            print(f'Move: #{i}; Taking Action: {action}')
        env.step(action)
        final_order.append([env.components['Agent'].position[0], env.components['Agent'].position[1]])
        state_ = env.get_complete_maze().reshape(1, flatten_dim) + np.random.rand(1, flatten_dim) / 10.0
        state = torch.from_numpy(state_).float()  # NO State2 bc no optimizer and loss function used
        reward = env.reward()
        if reward == 1:
            print(f'Game Won! Reward: {reward}')
            status = 0
        i += 1
        if (i > 30):
            if animated:
                print(f'Game Lost; Too many moves')
            break
    win = True if status == 1 else False
    return win, final_order

result, final_order = test_model(model)  # TESTING A SINGLE EPOCH RUN WITH TRAINED MODEL (No loss calculation or backpropagation)
if isAnimated:
    demo_solution = env.get_maze()
    demo_solution = demo_solution.tolist()
    demo_solution[env.start_node[0]][env.start_node[1]] = 0.5
    demo_solution[env.target_node[0]][env.target_node[1]] = 0.5
    fig = plt.figure('DFS')
    img = []

    for cell in final_order:
        demo_solution[cell[0]][cell[1]] = 0.7
        img.append([plt.imshow(demo_solution)])
        demo_solution[cell[0]][cell[1]] = 1
    ani = animation.ArtistAnimation(fig, img, interval=20, blit=True, repeat_delay=0)
    plt.show()
""" - TESTING MODEL - """

