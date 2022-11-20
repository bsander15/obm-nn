import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os  # For saving model
import numpy as np

class LinearFFNet(nn.Module):
    def __init__(self, input_vector_size, num_tasks, hidden_size=100):
        """
        This is 4 Layer NN that is parameterized with different numbers of inputs (# workers + # tasks in gMission), the
        number of outputs (# tasks + 1, where the extra output indicates that we should skip a worker) and the # of
        nodes we should have in our hidden layers.

        Outputs indicate the likelihood that we should match a worker to the nth node, so to make a decision we just
        take the argmax of the output and attempt to connect the input worker to the nth task. If that's not possible,
        I guess we'll skip the node - not entirely clear what to do in that case.
        """
        super(LinearFFNet, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(input_vector_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_tasks + 1)  # n+1st output is a "skip node" indicating we should skip this worker
        )

    def forward(self, x):
        """
        Takes as input an input vector describing a worker presented in an OLBM problem of form:

        [w_0, ... , w_U, m_0, ..., m_U]

        where w_n is the weight of the edge from worker_t to node_n and m_n is a binary mask representing whether or not
        a worker has already been matched to the nth task.

        Outputs a vector of probabilities of size U + 1 where the additional action represents skipping this worker.
        """
        return self.ff(x)


class QTrainer:
    """
    Class used for training neural network models using Q-Learning.

    QTrainer class adapted from David's old Cribbage project. Code for that project can be found here:
    https://github.com/yanivam/cribbage-agent-cs5100/blob/main/cribbage/QModel.py
    """
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # Predicted Q-values given input state:
        pred = self.model(state)

        # Target Q-Vals:
        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]  # The observed reward
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))  # 1-step Bellman Equation

            # Update reward:
            target[idx][action[idx].item()] = Q_new

        # Adjust the model so as to minimize difference between the Q-estimates computed above and the actual rewards:
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
