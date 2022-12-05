"""
Training file for NN models for OLBM Problem.

Currently, this file just trains a FFNet using reinforcement
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from Data.load_dataset import GMission
from Agents.FFNet import LinearFFNet, OLBMReinforceTrainer,InvFFNet
import torch
from tqdm import tqdm
from analysis import Analysis

NUM_TRAINING_ITERATIONS = 500000
NUM_TESTS_TO_RUN = 500
NUM_TASKS = 10
NUM_WORKERS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Automatically set the device for computation

# REWARD_MODES = ["SARSA_REWARD", "TOTAL_REWARD", "FINAL_REWARD"]
REWARD_MODES = ["SARSA_REWARD"]

def main(policy="ff"):
    # Load up the GMission Dataset:
    data = GMission()
    scores = []
    optimal_scores = []
    analysis = Analysis(NUM_WORKERS, NUM_TESTS_TO_RUN)

    # TRAIN FFNET with each of the reward modes
    for reward_mode in REWARD_MODES:
        if policy == "ff":
            print("NOW TRAINING: LinearFFNet:")
            input_vector_size = NUM_TASKS * 2  # One entry for each edge, one entry for each value in bitmap
            model = InvFFNet(input_vector_size, NUM_TASKS)
            trainer = OLBMReinforceTrainer(model=model,
                                            policy=policy,
                                        num_tasks=NUM_TASKS,
                                        num_workers=NUM_WORKERS,
                                        reward_mode=reward_mode)
            trainer.train_N_iterations(NUM_TRAINING_ITERATIONS)  # This will take a while to run
        elif policy == "ff-inv":
            print("NOW TRAINING: LinearFFNet:")
            model = InvFFNet(3, NUM_TASKS)
            trainer = OLBMReinforceTrainer(model=model,
                                            policy=policy,
                                        num_tasks=NUM_TASKS,
                                        num_workers=NUM_WORKERS,
                                        reward_mode=reward_mode)
            trainer.train_N_iterations(NUM_TRAINING_ITERATIONS)  # This will take a while to run
        

if __name__ == '__main__':
    main("ff-inv")
