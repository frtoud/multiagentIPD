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
import uuid

class lstmDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 16
        self.lstm = nn.LSTM(4,self.hidden_dim) #bigger hidden layer?
        self.cell = self.init_hidden()
        self.fc0 = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.fc1 = nn.Linear(self.hidden_dim,2)

    def forward(self, x, cell = None):
        if cell == None:
            cell = self.cell
        x, self.cell = self.lstm(x, cell) #no activation function needed at this point as they're included in the LSTM
        x = F.relu(self.fc0(x))
        q = F.relu(self.fc1(x))
        return q

    def init_hidden(self):
        return (Variable(torch.zeros(1, 1, self.hidden_dim)),
                Variable(torch.zeros(1, 1, self.hidden_dim)))

    def get_hidden(self):
        return (copy.deepcopy(self.cell[0].data), copy.deepcopy(self.cell[1].data));


class rDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 32
        self.fc1 = nn.Linear(12, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = F.relu(self.fc3(x))

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


learnrate = 0.001

class Agent(object):
    def __init__(self, gamma=0.99, batch_size=128):
        self.target_Q = lstmDQN()
        self.Q = lstmDQN()
        self.gamma = gamma
        self.batch_size = 128
        hard_update(self.target_Q, self.Q)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=learnrate)
        #self.optimizer_target = torch.optim.Adam(self.target_Q.parameters(), lr=learnrate)

        self.transitions = ReplayMemory(10000)
        self.current_reward = 0
        self.rewards = []

        self.ID = uuid.uuid4()
        self.resetTrust()

    def resetTrust(self):
        self.Trust = {}

    def setTrust(self, increment, uuid):
        if self.Trust.get(uuid) is None:
            self.Trust[uuid] = 0
        self.Trust[uuid] += increment

    def getMistrusted(self):
        worstTrust = list(self.Trust.keys())[0]
        for thisUuid, trust in self.Trust.items():
            if self.Trust[worstTrust] > trust:
                worstTrust = thisUuid

        return worstTrust
    def getTrusted(self):
        bestTrust = list(self.Trust.keys())[0]
        for thisUuid, trust in self.Trust.items():
            if self.Trust[bestTrust] < trust:
                bestTrust = thisUuid

        return bestTrust

    def resetMemory(self):
        self.Q.cell = self.Q.init_hidden()
        self.target_Q.cell = self.target_Q.init_hidden()

    def getMemory(self):
        return self.Q.get_hidden()

    def offspring(self, inherit=0.99):
        newAgent = Agent(self.gamma, self.batch_size)
        soft_update(newAgent.Q, self.Q, inherit)
        soft_update(newAgent.target_Q, self.target_Q, inherit)
        newAgent.current_reward = self.current_reward
        newAgent.rewards = copy.deepcopy(self.rewards)
        newAgent.transitions = copy.deepcopy(self.transitions)
        return newAgent

    def appendReward(self):
        self.rewards.append(self.current_reward)
        self.current_reward = 0

    def act(self, x, epsilon = 0.9):
        rand = np.random.uniform(0.0,1.0)
        if(epsilon > rand):
            return Variable(torch.from_numpy(np.array([np.random.randint(0,2)])).type(torch.LongTensor)).view(1,-1)
        else:
            _, best_a = torch.max(self.Q.forward(x), 2, keepdim=False)

            return best_a

        pass

    def backward(self, transitions, batch_size):
        batch = Transition(*zip(*transitions))
        input = Variable(torch.cat(batch.prev_output)).view(1, batch_size, -1)
        cell0 = Variable(torch.cat(batch.cell0)).view(1, batch_size, -1)
        cell1 = Variable(torch.cat(batch.cell1)).view(1, batch_size, -1)
        output = Variable(torch.cat(batch.output)).view(1, batch_size, -1)

        action = torch.index_select(output, 2, Variable(torch.LongTensor([0]))) #.data

        real_value = torch.gather(self.Q.forward(input, (cell0, cell1)), 2, action.long())

        cell0_2, cell1_2 = self.getMemory()

        _, best_a = torch.max(self.Q.forward(output), 2, keepdim=True)
        expected = self.target_Q.forward(output, (Variable(cell0_2), Variable(cell1_2)))
        value = torch.gather(expected, 2, best_a)

        y = self.gamma * value + torch.index_select(output, 2, Variable(torch.LongTensor([2])))

        loss = torch.nn.functional.smooth_l1_loss(real_value, y.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        soft_update(self.target_Q,self.Q,0.005)

        self.resetMemory()
        pass

class RnnAgent(object):
    def __init__(self, gamma=0.99, batch_size=128):
        self.target_Q = rDQN()
        self.Q = rDQN()
        self.gamma = gamma
        self.batch_size = 128
        hard_update(self.target_Q, self.Q)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=learnrate)
        #self.optimizer_target = torch.optim.Adam(self.target_Q.parameters(), lr=learnrate)

        self.resetMemory()
        self.transitions = ReplayMemory(10000)
        self.current_reward = 0
        self.rewards = []

        self.ID = uuid.uuid4()
        self.resetTrust()

    def resetTrust(self):
        self.Trust = {}

    def setTrust(self, increment, uuid):
        if self.Trust.get(uuid) is None:
            self.Trust[uuid] = 0
        self.Trust[uuid] += increment

    def getMistrusted(self):
        worstTrust = list(self.Trust.keys())[0]
        for thisUuid, trust in self.Trust.items():
            if self.Trust[worstTrust] > trust:
                worstTrust = thisUuid

        return worstTrust
    def getTrusted(self):
        bestTrust = list(self.Trust.keys())[0]
        for thisUuid, trust in self.Trust.items():
            if self.Trust[bestTrust] < trust:
                bestTrust = thisUuid

        return bestTrust

    def resetMemory(self):
        self.prev2 = Variable(torch.zeros(1, 1, 4))
        self.prev3 = Variable(torch.zeros(1, 1, 4))

    def getMemory(self):
        return copy.deepcopy((self.prev2.data, self.prev3.data))

    def offspring(self, inherit=0.99):
        newAgent = RnnAgent(self.gamma, self.batch_size)
        soft_update(newAgent.Q, self.Q, inherit)
        soft_update(newAgent.target_Q, self.target_Q, inherit)
        newAgent.current_reward = self.current_reward
        newAgent.rewards = copy.deepcopy(self.rewards)
        newAgent.transitions = copy.deepcopy(self.transitions)
        return newAgent

    def appendReward(self):
        self.rewards.append(self.current_reward)
        self.current_reward = 0

    def act(self, x, epsilon = 0.9):
        #TODO: Concat prev2 and prev3 to X as Input
        input = torch.cat((self.prev3, self.prev2, x), dim=2)
        rand = np.random.uniform(0.0,1.0)
        self.prev3 = self.prev2
        self.prev2 = x
        if(epsilon > rand):
            return Variable(torch.from_numpy(np.array([np.random.randint(0,2)])).type(torch.LongTensor)).view(1,-1)
        else:
            _, best_a = torch.max(self.Q.forward(input), 2, keepdim=False)
            return best_a

    def backward(self, transitions, batch_size):
        batch = Transition(*zip(*transitions))
        prev1 = Variable(torch.cat(batch.prev_output)).view(1, batch_size, -1)
        prev2 = Variable(torch.cat(batch.cell0)).view(1, batch_size, -1)
        prev3 = Variable(torch.cat(batch.cell1)).view(1, batch_size, -1)
        output = Variable(torch.cat(batch.output)).view(1, batch_size, -1)

        input = torch.cat((prev3, prev2, prev1), dim=2)
        input_next = torch.cat((prev2, prev1, output), dim=2)

        action = torch.index_select(output, 2, Variable(torch.LongTensor([0]))) #.data

        real_value = torch.gather(self.Q.forward(input), 2, action.long())

        _, best_a = torch.max(self.Q.forward(input_next), 2, keepdim=True)
        expected = self.target_Q.forward(input_next)
        value = torch.gather(expected, 2, best_a)

        y = self.gamma * value + torch.index_select(output, 2, Variable(torch.LongTensor([2])))

        loss = torch.nn.functional.smooth_l1_loss(real_value, y.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        soft_update(self.target_Q,self.Q,0.005)

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

env = gym.make('CartPole-v0') #DELET

epsilon = 1.0
anneal = 0.90
rewards = []

Agents = []
DeadAgents = []

nbAgents = 10
epochs = 100
iterations = 20

batch_size = min(50, 2 * nbAgents * iterations)
learnrate = 0.002
inheritance = 0.95
replacement = 0.1
selectionEpoch = epochs * 0.25


def Game(actionA, actionB):
    O = -2
    R = 3 + O
    S = 0 + O
    T = 5 + O
    P = 1 + O
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

#Darwins
def BestWorst(qt = 1):

    for i in range(qt):
        WorstAgent = Agents[0]
        WorstReward = WorstAgent.current_reward
        BestAgent = Agents[0]
        BestReward = BestAgent.current_reward

        for C in Agents:
            if (WorstReward > C.current_reward):
                WorstAgent = C
                WorstReward = WorstAgent.current_reward

            if (BestReward < C.current_reward):
                BestAgent = C
                BestReward = BestAgent.current_reward

        Agents.remove(WorstAgent)
        WorstAgent.appendReward()
        DeadAgents.append(WorstAgent.rewards)
        Agents.append(BestAgent.offspring(inheritance))
def Starve(min = 0):
    free = nbAgents - len(Agents)
    KillAgents = []
    NewAgents = []

    for C in Agents:
        if (min > C.current_reward):
            KillAgents.append(C)
            C.appendReward()
            free += 1
        else:
            if (free > 0):
                free -= 1
                NewAgents.append(C.offspring(inheritance))
    for K in KillAgents:
        Agents.remove(K)
        DeadAgents.append(K.rewards)
    for N in NewAgents:
        Agents.append(N)
def Social(qt = 1):

    TrustVotes = {}
    AgentReferences = {}

    for C in Agents:
        TrustVotes[C.ID] = 0
        AgentReferences[C.ID] = C

    for C in Agents:
        TrustVotes[C.getTrusted()] += 1
        TrustVotes[C.getMistrusted()] -= 1

    for i in range(qt):
        BestUuid = list(TrustVotes.keys())[0]
        WorstUuid = list(TrustVotes.keys())[0]
        for uuid, trust in TrustVotes.items():
            if TrustVotes[BestUuid] < trust:
                BestUuid = uuid
            if TrustVotes[WorstUuid] > trust:
                WorstUuid = uuid

        if WorstUuid != BestUuid:
            Agents.remove(AgentReferences[WorstUuid])
            AgentReferences[WorstUuid].appendReward()
            DeadAgents.append(AgentReferences[WorstUuid].rewards)
            Agents.append(AgentReferences[BestUuid].offspring(inheritance))

def Comp():
    epsilon = 1.0

    for i in range(nbAgents):
        Agents.append(Agent())
        pass

    for e in range(epochs):
        epsilon *= anneal
        epsilon = max(epsilon, 0.01)
        for A in Agents:
            A.resetTrust()
        #for A in Agents:
        #    for B in Agents:
        print(e)
        for a in range(len(Agents)):
           for b in range(a, len(Agents)):
                A = Agents[a]
                B = Agents[b]
                stateA = torch.FloatTensor([0, 0, 0, 0])
                stateB = torch.FloatTensor([0, 0, 0, 0])
                A.resetMemory()
                B.resetMemory()
                for i in range(iterations):
                #for i in range(int((random.random() + 0.5)* iterations)):
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

                    A.setTrust((ActionB.data[0][0] - 0.5)*2, B.ID)
                    B.setTrust((ActionA.data[0][0] - 0.5)*2, A.ID)


                    #print(e, resultsA[0], resultsA[1])

        if (e >= selectionEpoch):
        #    BestWorst(max(1, int(replacement * nbAgents)))
            Social(max(1, int(replacement * nbAgents)))
        #    Starve(nbAgents * 2 * iterations * 0)
            pass

        for C in Agents:
            C.appendReward()
            batch = C.transitions.sample(batch_size)
            C.backward(batch, batch_size)

def TFT(b_size = iterations, nb = 1, agents=None):

    epsilon = 0
    learn=True
    if (agents == None):
        learn=False
        agents = []
        for i in range(nb):
            agents.append(Agent())
            pass
        epsilon = 1

    for e in range(epochs):
        for agent in agents:
            epsilon *= anneal
            epsilon = max(epsilon, 0.01)
            state = torch.FloatTensor([1, 1, 0, 0])
            agent.resetMemory()
            ActionA = Variable(torch.LongTensor([[1]])).view(1,-1)
            for i in range(iterations):
                mem = agent.getMemory()
                ActionB = copy.deepcopy(ActionA)
                ActionA = agent.act(Variable(state).view(1,1,-1), epsilon)
                results, _ = Game(ActionA.data[0][0], ActionB.data[0][0])
                agent.transitions.push(copy.deepcopy(state), mem[0], mem[1], copy.deepcopy(results))
                state = copy.deepcopy(results)
                #keep track of cumulative score
                agent.current_reward += results[2]

                print(e, results[0], results[1])

            if (learn):
                agent.appendReward()
                batch = agent.transitions.sample(b_size)
                agent.backward(batch, b_size)

    for agent in agents:
        plt.plot(agent.rewards)
    plt.show()

def Validation(agents=None):
    if (agents == None):
        return
    epsilon=0
    e = 0
    for agent in agents:
        e += 1
        state = torch.FloatTensor([1, 1, 0, 0])
        agent.resetMemory()
        for i in range(10):
            mem = agent.getMemory()
            ActionA = agent.act(Variable(state).view(1,1,-1), epsilon)
            results, _ = Game(ActionA.data[0][0], 1)
            agent.transitions.push(copy.deepcopy(state), mem[0], mem[1], copy.deepcopy(results))
            state = copy.deepcopy(results)
            print("C", e, results[0], results[1])
        state = torch.FloatTensor([1, 1, 0, 0])
        agent.resetMemory()
        for i in range(10):
            mem = agent.getMemory()
            ActionA = agent.act(Variable(state).view(1,1,-1), epsilon)
            results, _ = Game(ActionA.data[0][0], 0)
            agent.transitions.push(copy.deepcopy(state), mem[0], mem[1], copy.deepcopy(results))
            state = copy.deepcopy(results)
            print("D", e, results[0], results[1])


Comp()

#TFT(iterations, nbAgents)

data = []
for C in Agents:
    print(C.rewards)
#    data.append(C.rewards)
    plt.plot(C.rewards)

for C in DeadAgents:
    print("DEAD:", C)
#    data.append(C.rewards)
    plt.plot(C)
#plot = np.array(data)
#plot = plot.reshape(-1, plot.shape[0])
#print("plot dimension: ", plot.shape[0], plot.shape[1])
#pd.DataFrame(data).plot()
#plt.plot()
plt.show()

Validation(agents=Agents)
