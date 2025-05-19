import matplotlib.animation as animation
from matplotlib import pylab as plt
from MazeEnv import CreateMazeGym
from collections import deque
import numpy as np
import random
import torch

env = CreateMazeGym()
nrow, ncol = np.random.randint(20, 35),  np.random.randint(20, 35)
env.new_maze(nrow=15, ncol=15)
env.make()
env.render()
trainer_maze = env.get_env()

l1 = env.get_maze().shape[0] * env.get_maze().shape[1] * 3
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
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()
gamma = .9
epsilon = 1.0

action_set = {0: 'top', 1: 'right', 2: 'bottom', 3: 'left'}

epochs = 1000  # 5000
losses = list()
rewards = list()
results = list()
mem_size = 100  # 1000  # Experience Replay memory size
batch_size = 30 # 100  # Mini-Batch size
replay = deque(maxlen=mem_size)  # Creates memory replay as a deque list
max_steps = 225 # 330  # Max number of moves
h = 0
isAnimated = True
order = list()
completeRewards = dict()
states = list()
detailedState = list()
actions = list()
total_moves = dict()

for i in range(epochs):

    order = []
    env.set_env(trainer_maze)
    flatten_dim = env.get_maze().shape[0] * env.get_maze().shape[1] * 3
    state1_ = env.get_complete_maze().reshape(1, flatten_dim) + np.random.rand(1, flatten_dim) / 100.0
    state1 = torch.from_numpy(state1_).float()
    status, counter = 1, 1
    while status == 1 and counter < max_steps+1:
        counter += 1
        qval = model(state1)  # Computes Q values from the input state in order to select an action
        qval_ = qval.data.numpy()
        if random.random() < epsilon:
            action_ = np.random.randint(0, 4)
        else:
            action_ = np.argmax(qval_)
        action = action_set[action_]
        actions.append(action)
        order.append([env.components['Agent'].position[0], env.components['Agent'].position[1]])
        env.step(action)
        state2_ = env.get_complete_maze().reshape(1, flatten_dim) + np.random.rand(1, flatten_dim) / 100.0
        state2 = torch.from_numpy(state2_).float()
        """ - APPENDING STATE - """
        curr_state = np.floor(state2[0].numpy())
        states.append(curr_state)
        """ - APPENDING STATE - """
        reward = env.reward()
        rewards.append(reward)
        done = True if reward > 0 else False
        exp = (state1, action_, state2, reward, done)  # Creates the (s, a, st+1, rt+1) tuple
        replay.append(exp)  # Appends the tuple to the list
        state1 = state2

        if len(replay) > batch_size:  # If replay list is at least as long as the mini-batch size (begins mini-batch training)

            minibatch = random.sample(replay, batch_size)  # Random Sub sample
            state1_batch = torch.cat([s1 for (s1, a, s2, r, d) in minibatch])  # Getting s1, a, s2, r, d batch
            action_batch = torch.Tensor([a for (s1, a, s2, r, d) in minibatch])
            state2_batch = torch.cat([s2 for (s1, a, s2, r, d) in minibatch])  # Batch states are (batch_size x 64)
            reward_batch = torch.Tensor([r for (s1, a, s2, r, d) in minibatch])
            done_batch = torch.Tensor([d for (s1, a, s2, r, d) in minibatch])

            Q1 = model(state1_batch)  # Recomputes Q values for mini-batch of states to get gradients
            with torch.no_grad():
                Q2 = model(state2_batch)
            Y = reward_batch + gamma * (
                        (1 - done_batch) * torch.max(Q2, dim=1)[0])  # Compute target Q values we want to learn
            X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(
                dim=1)).squeeze()  # Get Q-values associated with the action index
            loss = loss_fn(X, Y.detach())
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

#        counter += 1
        if reward == 1:  # If game is over resets status and mov number
            status = 0
            results.append(status)

    if status == 1:
        results.append(status)

    print(f'Epoch: {i}, Status: {status} (0: Pass!, 1: Failed)')
    total_moves[i] = len(order)
    if epsilon > 0.1:
        epsilon -= (1/epochs)


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



""" - TESTING MODEL - """
def test_model(model, animated=True):
    final_order = list()
    env.set_env(trainer_maze, max_steps)
    flatten_dim = env.get_maze_().shape[0] * env.get_maze_().shape[1] * 3  # PREVIOUS [3] FOR (AGENT, GOAL, WALLS)
    maze_ = env.get_complete_maze().reshape(1, flatten_dim) + np.random.rand(1, flatten_dim) / 100.0
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
        state_ = env.get_complete_maze().reshape(1, flatten_dim) + np.random.rand(1, flatten_dim) / 100.0
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

