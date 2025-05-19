import matplotlib.animation as animation
from matplotlib import pylab as plt
from MazeEnv import CreateMazeGym
from collections import deque
import numpy as np
import random
import torch
import copy


# - MAZE CREATION -
env = CreateMazeGym()
nrow, ncol = np.random.randint(20, 35),  np.random.randint(20, 35)
env.new_maze(nrow=7, ncol=7)
env.make()
env.render()
trainer_maze = env.get_env()

l1 = env.get_maze().shape[0]*env.get_maze().shape[1]*3
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

model1 = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.ReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.ReLU(),
    torch.nn.Linear(l3, l4)
)

model2 = copy.deepcopy(model1)  # Second model by making identical copy of Q-network
model2.load_state_dict(model1.state_dict())  # Copies the parameters of the original model (weights)

loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model1.parameters(), lr=learning_rate)

epochs = 1000  # 5000
mem_size = 120 # 1000
batch_size = 30  # 200
max_steps = 335  # 500
replay = deque(maxlen=mem_size)
h = 0
sync_freq = 50  # Sync the frequency parameter, every 50 steps will copy the parameters into model2
j = 0

gamma = .9  # Discount Factor
epsilon = 1  # Epsilon for our selection model
action_set = {0: 'top', 1: 'right', 2: 'bottom', 3: 'left'}
isAnimated = True

losses = list()
order = list()
completeRewards = dict()
rewards = list()
results = list()
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
    completeRewards[i] = list()
    while status == 1 and counter < max_steps+1:
        j += 1
        counter += 1
        qval = model1(state1)
        qval_ = qval.data.numpy()
        if random.random() < epsilon:
            action_ = np.random.randint(0, 4)
        else:
            action_ = np.argmax(qval_)
        action = action_set[action_]
        order.append([env.components['Agent'].position[0], env.components['Agent'].position[1]])
        env.step(action)
        state2_ = env.get_complete_maze().reshape(1, flatten_dim) + np.random.rand(1, flatten_dim) / 100.0
        state2 = torch.from_numpy(state2_).float()
        """ - MODIFIED - """
        curr_state = np.floor(state2[0].numpy())
        states.append(curr_state)
        """ - MODIFIED - """
        reward = env.reward()
        rewards.append(reward)
        completeRewards[i].append(reward)
        done = True if reward > 0 else False
        exp = (state1, action_, state2, reward, done)
        replay.append(exp)
        state1 = state2

        if len(replay) > batch_size:
            minibatch = random.sample(replay, batch_size)
            state1_batch = torch.cat([s1 for (s1, a, s2, r, d) in minibatch])
            action_batch = torch.Tensor([a for (s1, a, s2, r, d) in minibatch])
            state2_batch = torch.cat([s2 for (s1, a, s2, r, d) in minibatch])
            reward_batch = torch.Tensor([r for (s1, a, s2, r, d) in minibatch])
            done_batch = torch.Tensor([d for (s1, a, s2, r, d) in minibatch])

            Q1 = model1(state1_batch)
            with torch.no_grad():
                Q2 = model2(state2_batch)  # Uses target network to get maximum Q-value for next state

            Y = reward_batch + gamma * ((1-done_batch) * torch.max(Q2, dim=1)[0])
            X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
            loss = loss_fn(X, Y.detach())
#            print(i, loss.item())
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

            if j % sync_freq == 0:  # Copies the main model parameters to the target network after sync_freq iterations
                model2.load_state_dict(model.state_dict())

        if reward == 1:
            status = 0
            results.append(status)


    if status == 1:
        results.append(status)

    print(f'Epoch: {i}, Status: {status} (0: Pass!, 1: Failed)')
    total_moves[i] = len(order)
    if epsilon > 0.1:
        epsilon -= (1/epochs)



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

losses = np.array(losses)

""" Plotting the Loss result from every Epoch """  # When Using random mode layout loss does not find convergence
plt.plot(list(range(len(losses))), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')

""" Plotting the Rewards result from every Epoch """  # When Using random mode layout loss does not find convergence
plt.plot(list(range(len(rewards))), rewards)
plt.xlabel('Epochs')
plt.ylabel('Reward')



