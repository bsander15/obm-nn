import torch
import time  # For saving log files
import torch.nn as nn
from collections import deque
import numpy as np
from tqdm import tqdm

from Data.load_dataset import GMission, OLBMInstance

DEVICE = "cpu"

class LinearFFNet(nn.Module):
    def __init__(self, input_vector_size, num_tasks, hidden_size=100):
        """
        This is 4 Layer NN that is parameterized with different numbers of inputs (# workers + # tasks in gMission), the
        number of outputs (# tasks + 1, where the extra output indicates that we should skip a worker) and the # of
        nodes we should have in our hidden layers.
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
        action. The action is chosen by sampling from all possible actions according the the probability distribution
        computed over all the actions by the NN - in other words, the network will *typically* take actions that are
        predicted to be high-value, but in a somewhat stochastic manner to promote exploration of policy space.

        This code was partially adapted from Noufal Samsudin's REINFORCE implementation, found at:
        https://github.com/kvsnoufal/reinforce
        """
        actions = self.ff(x.float())
        if np.isnan(actions.squeeze(0).detach().cpu().numpy()).any():
            # TODO: there is a bug where the weights go to NaN, which results in NaN ouput.
            # Trying to decrease the learning rate...
            print("STOP!")
        action = np.random.choice(self.action_space, p=actions.squeeze(0).detach().cpu().numpy())  # TODO: necessary?
        log_prob_action = torch.log(actions.squeeze(0))[action]
        return action, log_prob_action

    def name(self):
        return "LINEAR_FF_NET"


class OLBMReinforceTrainer:
    def __init__(self, model, lr=0.0001, gamma=0.9, num_tasks=10, num_workers=30, reward_mode="SARSA_REWARD"):
        self.model = model.to(DEVICE)
        self.lr = lr
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.gmission_dataset = GMission()
        self.all_rewards = []
        self.num_tasks = num_tasks  # Should refactor this to get direct from self.model?
        self.num_workers = num_workers  # Should refactor this to get direct from self.model?
        self.reward_mode = reward_mode
        self._time = str(time.time())
        self.log_file = self._time + "_OLBM_REINFORCE_TRAINING_LOG.csv"
        self.model_name = self._time + "_" + self.model.name() + "_" + self.reward_mode

    def train_iteration(self, problem_generator_seed=1234):
        # Generate an OLBM problem:
        problem = self.gmission_dataset.generate_olbm_instance(num_tasks=self.num_tasks,
                                                               num_workers=self.num_workers,
                                                               random_seed=problem_generator_seed)

        log_probs = []  # vector of log-probabilities of model outputs for each step of the problem
        rewards = []  # vector of rewards

        # Use policy (neural network) to complete OLBM problem, recording action probabilities from policy, the action
        # that was taken, and the reward for taking that action (in this context, reward is the weight of the edge that
        # we choose to add to the network):
        while problem.has_unseen_workers():
            worker, state = problem.get_next_nn_input()  # Pick the next worker to match and get the input as a vector
            state = torch.from_numpy(state).to(DEVICE)  # Need to convert the datatype to a tensor for pytorch
            action, log_prob = self.model(state)  # Choose an action based on the model
            reward = problem.match(action, worker)  # Perform matching, calculate reward

            if self.reward_mode == "SARSA_REWARD":
                rewards.append(reward)  # Keep track of the reward we got for taking the action with highest log-prob
            elif self.reward_mode == "TOTAL_REWARD":
                rewards.append(problem.get_matching_score())  # Reward is sum of all weights included in matching so far
            elif self.reward_mode == "FINAL_REWARD":
                if problem.has_unseen_workers():
                    rewards.append(0)  # Just give a point for continuing to play the games
                else:
                    rewards.append(problem.get_matching_score())  # All discounted rewards will be based on final score
            else:
                print("OLBMTRAINER ERROR: Unrecognized reward mode!")
                exit(-1)
            log_probs.append(log_prob)  # Keep track of associated model output that generated the above reward

        # Keep track of "total reward" generated throughout the iteration for plotting later. We'll want to see these
        # numbers going up over time.
        self.all_rewards.append(np.sum(rewards))

        # Calculate "discounted rewards" for each action. At a high level, the reason for doing this is because each
        # action should take into account not just the reward generated for immediately taking the action, but also
        # all the future rewards that are gained *as a result* of taking this action - in other words, the rewards of
        # *future* actions. However, it's assumed (in the Bellman Eqn.) that *future* rewards matter *less* than
        # immediate rewards, so we caclulate the "total discounted reward" at each time step as the sum of the rewards
        # generated in the future, discounted at a rate of gamma**t * reward_t, where t is the number of time steps
        # into the future.
        discounted_rewards = []
        for t in range(len(rewards)):
            discounted_reward_at_time_t = 0
            exponent = 0

            # calculate reward for t'th action as reward for t'th action plus sum of future discounted actions:
            for reward in rewards[t:]:
                discounted_reward_at_time_t = discounted_reward_at_time_t + self.gamma ** exponent * reward
                exponent += 1
            discounted_rewards.append(discounted_reward_at_time_t)
        discounted_rewards = np.array(discounted_rewards)  # Cast to np.array

        # Adjust weights of Policy (NN) by backpropagating error to increase rewards:
        _discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32, device=DEVICE)

        # Center _discounted_rewards to keep gradients from exploding:
        discounted_rewards = (_discounted_rewards - torch.mean(_discounted_rewards)) / (torch.std(_discounted_rewards))
        log_prob = torch.stack(log_probs)

        # This conditional is necessary to check for the case where we've made a problem instance that has zero matching
        # workers and tasks. In this case, reward will be
        if torch.isnan(discounted_rewards).all():
            # This conditional is necessary to check for the case where we've made a problem instance that has zero
            # matching workers and tasks. In this case, _discounted_rewards will be entirely 0s, so the centering
            # above will cause a divide-by-zero error and a bunch of NaNs. In reality, we want the gradient to be 0
            # since we don't really learn anything from these cases
            policy_gradient = -log_prob.to(DEVICE) * torch.zeros_like(discounted_rewards).to(DEVICE)
        else:
            # In the REINFORCE algorithm, we are trying to maximize the Expected Reward. This can be calculated as the
            # product of the probability of each action at each time step (stored in log_prob) * the discounted-reward
            # for each action taken at each time-step, which we calculated as discounted_rewards above.
            policy_gradient = -log_prob.to(DEVICE) * discounted_rewards

        # Perform backpropagation:
        self.model.zero_grad()
        policy_gradient.sum().backward()
        self.optimizer.step()
        return problem.get_matching_score()

    def train_N_iterations(self, N=100):
        """
        Perform N training iterations. A single iteration consists of training the network on a single OLBMInstance
        sampled from GMission.
        """
        rolling_avg = deque(maxlen=100)
        for episode in tqdm(range(N)):
            reward = self.train_iteration(problem_generator_seed=episode)
            rolling_avg.append(reward)
            if episode % 1000 == 0:
                print(f"EPISODE {episode} - SCORE: {reward}")
                print(f"Rolling average over last {len(rolling_avg)} episodes = {np.mean(rolling_avg)}")
                with open(self.log_file, 'a+') as log:
                    log.write(f"{episode}, {rolling_avg}\n")
            if episode % 10000 == 0:
                # Save a snapshot of the model weights:
                torch.save(self.model.state_dict(), f"./{self.model_name}.pth")


