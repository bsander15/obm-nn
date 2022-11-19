import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os  # For saving model
import numpy as np

class LinearFFNet(nn.Module):
    def __init__(self, input_vector_size, num_tasks, hidden_size):
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
