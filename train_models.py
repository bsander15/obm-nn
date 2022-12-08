"""
Training file for NN models for OLBM Problem.

Currently, this file just trains a FFNet using reinforcement
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from Data.load_dataset import GMission
from Agents.FFNet import LinearFFNet, OLBMReinforceTrainer,InvFFNet
import torch

NUM_TRAINING_ITERATIONS = 100
NUM_TESTS_TO_RUN = 5000
NUM_TASKS = [10, 10, 100]
NUM_WORKERS = [30, 60, 100]
REWARD_MODES = ['SARSA_REWARD', 'TOTAL_REWARD', 'FINAL_REWARD']
# REWARD_MODES = ['FINAL_REWARD']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Automatically set the device for computation

# REWARD_MODES = ["SARSA_REWARD", "TOTAL_REWARD", "FINAL_REWARD"]
REWARD_MODES = ["FINAL_REWARD"]

def main(policy="ff"):

    if policy == "ff":
    # TRAIN FFNET:
        for num_tasks, num_workers in zip(NUM_TASKS, NUM_WORKERS):
            input_vector_size = num_tasks * 2  # One entry for each edge, one entry for each value in bitmap
            model = LinearFFNet(input_vector_size, num_tasks)
            for mode in REWARD_MODES:
                print(f'NOW TRAINING: LinearFFNet_{mode}_{num_tasks}x{num_workers}:')
                trainer = OLBMReinforceTrainer(model=model, num_tasks=num_tasks, num_workers=num_workers, reward_mode=mode)
                trainer.train_N_iterations(NUM_TRAINING_ITERATIONS)  # This will take a while to run
    elif policy == "ff-inv":
           # TRAIN FF-INV:
        for num_tasks, num_workers in zip(NUM_TASKS, NUM_WORKERS):
            model = InvFFNet(3, num_tasks)
            for mode in REWARD_MODES:
                print(f'NOW TRAINING: InvFFNet_{mode}_{num_tasks}x{num_workers}:')
                trainer = OLBMReinforceTrainer(model=model,policy = policy, num_tasks=num_tasks, num_workers=num_workers, reward_mode=mode)
                trainer.train_N_iterations(NUM_TRAINING_ITERATIONS)  # This will take a while to run


if __name__ == '__main__':
    main("ff-inv")
