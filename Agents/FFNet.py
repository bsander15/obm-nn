import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os  # For saving model
import numpy as np

from Data.load_dataset import GMission, OLBMInstance

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Automatically set the device for computation


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
            nn.Linear(hidden_size, num_tasks + 1),
            # n+1st output is a "skip node" indicating we should skip this worker
            nn.Softmax()
        )
        self.action_space = np.arange(num_tasks + 1, dtype=np.int16)

    def forward(self, x):
        """
        Takes as input an input vector describing a worker presented in an OLBM problem of form:

        [w_0, ... , w_U, m_0, ..., m_U]

        where w_n is the weight of the edge from worker_t to node_n and m_n is a binary mask representing whether or not
        a worker has already been matched to the nth task.

        Outputs a chosen action, e.g. a task to match the input worker to, plus the log-probability of taking that
        action.

        This code was partially adapted from Noufal Samsudin's REINFORCE implementation, found at:
        https://github.com/kvsnoufal/reinforce
        """
        actions = self.ff(x.float())
        action = np.random.choice(self.action_space, p=actions.squeeze(0).detach().cpu().numpy())  # TODO: necessary?
        log_prob_action = torch.log(actions.squeeze(0))[action]
        return action, log_prob_action


class OLBMReinforceTrainer:
    def __init__(self, model, lr=0.0001, gamma=0.9, num_tasks=10, num_workers=30):
        self.model = model.to(DEVICE)
        self.lr = lr
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.gmission_dataset = GMission()
        self.all_rewards = []  # TODO: What is this for?
        self.best_rolling = -99999  # TODO: What is this for?
        self.num_tasks = num_tasks  # Should refactor this to get direct from self.model?
        self.num_workers = num_workers  # Should refactor this to get direct from self.model?

    def train_iteration(self):
        # Generate an OLBM problem:
        problem = self.gmission_dataset.generate_olbm_instance(num_tasks=self.num_tasks, num_workers=self.num_workers)

        log_probs = []  # vector of log-probabilities
        rewards = []  # vector of rewards

        # Use policy (neural network) to complete OLBM problem, recording action probabilities from policy, reward
        # from environment, and action at each step
        while problem.has_unseen_workers():
            worker, state = problem.get_next_nn_input()
            state = torch.from_numpy(state).to(DEVICE)
            action, log_prob = self.model(state)  # Choose an action based on the model
            reward = problem.match(action, worker)

            rewards.append(reward)
            log_probs.append(log_prob)

        self.all_rewards.append(np.sum(rewards))

        # Calculate discounted rewards for each action
        discounted_rewards = []
        for t in range(len(rewards)):
            Gt = 0
            exponent = 0

            for reward in rewards[t:]:
                Gt = Gt + self.gamma ** exponent * reward  # REINFORCE/Bellman Eqn
                exponent += 1
            discounted_rewards.append(Gt)
        discounted_rewards = np.array(discounted_rewards)  # Cast to np.array

        # Adjust weights of Policy (NN) by backpropagating error to increase rewards:
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32, device=DEVICE)
        discounted_rewards = (discounted_rewards - torch.mean(discounted_rewards)) / (torch.std(discounted_rewards))
        log_prob = torch.stack(log_probs)

        policy_gradient = -log_prob.to(DEVICE) * discounted_rewards

        self.model.zero_grad()
        policy_gradient.sum().backward()
        self.optimizer.step()
        return np.sum(rewards)

    def train_N_iterations(self, N=100):
        for episode in range(N):
            reward = self.train_iteration()
            if episode % 100 == 0:
                print(f"EPISODE {episode} - SCORE: {reward}")
