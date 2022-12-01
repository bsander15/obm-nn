"""
Training file for NN models for OLBM Problem.

Currently, this file just trains a FFNet using reinforcement
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from Data.load_dataset import GMission
from Agents.FFNet import LinearFFNet, OLBMReinforceTrainer
import torch
from tqdm import tqdm
from analysis import Analysis

NUM_TRAINING_ITERATIONS = 500000
NUM_TESTS_TO_RUN = 5000
NUM_TASKS = 10
NUM_WORKERS = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Automatically set the device for computation

def main():
    # Load up the GMission Dataset:
    data = GMission()
    scores = []
    optimal_scores = []
    analysis = Analysis(NUM_WORKERS, NUM_TESTS_TO_RUN)

    # TRAIN FFNET:
    print("NOW TRAINING: LinearFFNet:")
    input_vector_size = NUM_TASKS * 2  # One entry for each edge, one entry for each value in bitmap
    model = LinearFFNet(input_vector_size, NUM_TASKS)
    trainer = OLBMReinforceTrainer(model=model, num_tasks=NUM_TASKS, num_workers=NUM_WORKERS)
    trainer.train_N_iterations(NUM_TRAINING_ITERATIONS)  # This will take a while to run

    # BASIC TEST FOR FFNET:
    print("NOW TESTING LinearFFNet:")
    analysis = Analysis(NUM_WORKERS, NUM_TESTS_TO_RUN)
    scores = []
    optimal_scores = []
    model.eval()
    with torch.no_grad():
        for test in tqdm(range(NUM_TESTS_TO_RUN)):
            # Solve the problem using the greedy algorithm:
            problem_to_solve = data.generate_olbm_instance(num_tasks=NUM_TASKS,
                                                           num_workers=NUM_WORKERS,
                                                           random_seed=test)
            while problem_to_solve.has_unseen_workers():
                worker, state = problem_to_solve.get_next_nn_input()
                state = torch.from_numpy(state).to(DEVICE)
                action, log_prob = model(state)
                analysis.store_agreement_data(worker, action, problem_to_solve)
                problem_to_solve.match(action, worker)
            scores.append(problem_to_solve.get_matching_score())

            # Find optimal solution to the problem:
            row_ind, col_ind = linear_sum_assignment(problem_to_solve.get_all_edges(), maximize=True)
            optimal_score = 0
            for i, j in zip(row_ind, col_ind):
                optimal_score += problem_to_solve.get_all_edges()[i, j]
            optimal_scores.append(optimal_score)
        analysis.agreement_by_t()

        print(f"FFNet agent mean score: {np.mean(scores)} over {NUM_TESTS_TO_RUN} trials")
        print(f"Mean optimal score: {np.mean(optimal_scores)} over {NUM_TESTS_TO_RUN} trials")
        print(f"Optimality Ratio of FFNet: {np.mean(scores) / np.mean(optimal_scores)} over {NUM_TESTS_TO_RUN} trials")

if __name__ == '__main__':
    main()