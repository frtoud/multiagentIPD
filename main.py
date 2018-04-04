# based on http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import torch
from torch import nn
import torch.nn.functional as F
import gym
from torch.autograd import Variable
import random
from collections import namedtuple

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 12;
        self.lstm = nn.LSTM(4,self.hidden_dim) #bigger hidden layer?
        self.cell = self.init_hidden()
        self.fc0 = nn.Linear(self.hidden_dim,2)

    def forward(self, x, cell = None):
        if cell == None:
            cell = self.cell
        x, self.cell = self.lstm(x, cell) #no activation function needed at this point as they're included in the LSTM
        q = F.relu(self.fc0(x))
        return q

    def init_hidden(self):
        return (Variable(torch.zeros(1, 1, self.hidden_dim)),
                Variable(torch.zeros(1, 1, self.hidden_dim)))

    def get_hidden(self):
        return (copy.deepcopy(self.cell[0].data), copy.deepcopy(self.cell[1].data));


class DuelingDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fadvantage = nn.Linear(512, 2)

        self.fvalue = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        advantage = self.fadvantage(x)

        value = self.fvalue(x)

        q = value + advantage
        return q

# from https://github.com/ghliu/pytorch-ddpg/blob/master/util.py

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class Agent(object):
    def __init__(self, gamma=0.99, batch_size=128):
        self.target_Q = DQN()
        self.Q = DQN()
        self.gamma = gamma
        self.batch_size = 128
        hard_update(self.target_Q, self.Q)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=0.001)
        self.optimizer_target = torch.optim.Adam(self.target_Q.parameters(), lr=0.001)

        self.transitions = ReplayMemory(10000)
        self.current_reward = 0
        self.rewards = []

    def resetMemory(self):
        self.Q.cell = self.Q.init_hidden()
        self.target_Q.cell = self.target_Q.init_hidden()

    def getMemory(self):
        return self.Q.get_hidden();

    def appendReward(self):
        self.rewards.append(self.current_reward)
        self.current_reward = 0

    def act(self, x, epsilon = 0.9):
        # TODO
        rand = np.random.uniform(0.0,1.0)
        if(epsilon > rand):
            return Variable(torch.from_numpy(np.array([np.random.randint(0,2)])).type(torch.LongTensor)).view(1,-1)
        else:
            _, best_a = torch.max(self.Q.forward(x), 2, keepdim=False)

            return best_a
        # fonction utiles: torch.max()

        pass

    def backward(self, transitions):
        batch = Transition(*zip(*transitions))
        input = Variable(torch.cat(batch.prev_output)).view(1, batch_size, -1)
        cell0 = Variable(torch.cat(batch.cell0)).view(1, batch_size, -1)
        cell1 = Variable(torch.cat(batch.cell1)).view(1, batch_size, -1)
        output = Variable(torch.cat(batch.output)).view(1, batch_size, -1)

        action = torch.index_select(output, 2, Variable(torch.LongTensor([0]))) #.data

        real_value = torch.gather(self.Q.forward(input, (cell0, cell1)), 2, action.long())
        #_, best_a = torch.max(self.Q.forward(next_state),1, keepdim=False)

        cell0_2, cell1_2 = self.getMemory()

        _, best_a = torch.max(self.Q.forward(output), 2, keepdim=True)
        expected = self.target_Q.forward(output, (Variable(cell0_2), Variable(cell1_2)))
        value = torch.gather(expected, 2, best_a)

        y = self.gamma * value + torch.index_select(output, 2, Variable(torch.LongTensor([2]))) + 0.8 * torch.index_select(output, 2, Variable(torch.LongTensor([3])))

        ##_, best_a_target = torch.max(self.target_Q.forward(next_state),1, keepdim=False)
        #expected_value_target= self.target_Q.forward(next_state)
        ##expected_value = self.Q.forward(next_state)
        #value= torch.gather(expected_value_target, 1, best_a.view(128,1))
        ##value_target= torch.gather(expected_value, 1, best_a_target.view(128,1))
        #y= done.view(128,1)*self.gamma*(value) + reward.view(128,1)
        ##y_target= done.view(128,1)*self.gamma*(value_target) + reward.view(128,1)

        #real_value = torch.gather(self.Q.forward(state), 1, action.view(128, 1))
        #real_value_target = torch.gather(self.target_Q.forward(state), 1, action.view(128, 1))

        #Qs = self.Q.forward(state)
        #_, best_a_current = torch.max(Qs,1, keepdim=False)
        #state_value = self.target_Q.forward(state)
        #V_state= torch.gather(state_value, 1, best_a_current.view(128,1))

        #Q_action= torch.gather(Qs, 1, action.view(128,1))

        #A = Q_action - V_state
        #A_expected = y - V_state

        #Y = Variable(A_expected.data)
        #Y.volatile=False

        loss = torch.nn.functional.smooth_l1_loss(real_value, y.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        soft_update(self.target_Q,self.Q,0.005)

        #loss_target = torch.nn.functional.smooth_l1_loss(real_value_target, y_target.detach())
        #self.optimizer_target.zero_grad()
        #loss_target.backward()
        #self.optimizer_target.step()

        # LOSS = y - d
        # GRAD DESCENT WITH LOSS + GRAD
        # TODO
        # fonctions utiles: torch.gather(), torch.detach()
        # torch.nn.functional.smooth_l1_loss()
        self.resetMemory()
        pass

Transition = namedtuple('Transition',
                        #('state', 'action', 'next_state', 'reward', 'done'))
                        ('prev_output', 'cell0', 'cell1', 'output'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

env = gym.make('CartPole-v0')
agent = Agent()
memory = ReplayMemory(100000)

epsilon = 1.0
rewards = []

Agents = []

nbAgents = 2
epochs = 50
iterations = 100
batch_size = 50

for i in range(nbAgents):
    Agents.append(Agent())
    pass

def Game(actionA, actionB):
    R = 3
    S = 0
    T = 5
    P = 1
    #1 is cooperate 0 is defect
    rewardA = 0
    rewardB = 0
    if actionA == 1:
        if actionB == 1:
            rewardA = R
            rewardB = R
            pass
        else:
            rewardA = S
            rewardB = T
            pass
        pass
    else:
        if actionB == 1:
            rewardA = T
            rewardB = S
            pass
        else:
            rewardA = P
            rewardB = P
            pass
        pass
    return (torch.FloatTensor([actionA, actionB, rewardA, rewardB]), torch.FloatTensor([actionB, actionA, rewardB, rewardA]))

for e in range(epochs):
    epsilon *= 0.95
    epsilon = max(epsilon, 0.01)
    for A in Agents:
        for B in Agents:
            stateA = torch.FloatTensor([0, 0, 0, 0])
            stateB = torch.FloatTensor([0, 0, 0, 0])
            A.resetMemory()
            B.resetMemory()
            for i in range(iterations):
                print(e)
                Amem = A.getMemory()
                Bmem = B.getMemory()
                ActionA = A.act(Variable(stateA).view(1,1,-1), epsilon)
                ActionB = B.act(Variable(stateB).view(1,1,-1), epsilon)
                resultsA, resultsB = Game(ActionA.data[0][0], ActionB.data[0][0])
                A.transitions.push(copy.deepcopy(stateA), Amem[0], Amem[1], copy.deepcopy(resultsA))
                B.transitions.push(copy.deepcopy(stateB), Bmem[0], Bmem[1], copy.deepcopy(resultsB))
                stateA = copy.deepcopy(resultsA)
                stateB = copy.deepcopy(resultsB)
                #keep track of cumulative score
                A.current_reward += resultsA[2]
                B.current_reward += resultsB[2]
                print(resultsA)

    for C in Agents:
        C.appendReward()
        batch = C.transitions.sample(batch_size)
        C.backward(batch)

#for i in range(5000):
#    obs = env.reset()
#    done = False
#    total_reward = 0
#    epsilon *= 0.99
#    while not done:
#        epsilon = max(epsilon, 0.01)
#        obs_input = Variable(torch.from_numpy(obs).type(torch.FloatTensor))
#        action = agent.act(obs_input, epsilon)
#        next_obs, reward, done, _ = env.step(action.data.numpy()[0])
#        memory.push(obs_input.data.view(1,-1), action.data,
#                    torch.from_numpy(next_obs).type(torch.FloatTensor).view(1,-1), torch.Tensor([reward]),
#                    torch.Tensor([done]))
#        obs = next_obs
#        total_reward += reward
#    rewards.append(total_reward)
#    if memory.__len__() > 10000:
#        batch = memory.sample(batch_size)
#        agent.backward(batch)

#pd.DataFrame(rewards).rolling(50, center=False).mean().plot()
#plt.show()
for C in Agents:
    print(C.rewards)
