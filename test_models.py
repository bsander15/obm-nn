import glob
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

import train_models
from Agents.FFNet import LinearFFNet,InvFFNet
from Agents.Greedy import Greedy
from Data.load_dataset import GMission
from analysis import Analysis
from train_models import NUM_WORKERS, NUM_TASKS, NUM_TESTS_TO_RUN


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def test_greedy_algorithm(data):
    scores = []
    optimal_scores = []
    analysis = Analysis(NUM_WORKERS, NUM_TESTS_TO_RUN)
    for test in range(NUM_TESTS_TO_RUN):
        # Solve the problem using the greedy algorithm:
        problem_to_solve = data.generate_olbm_instance(num_tasks=NUM_TASKS, num_workers=NUM_WORKERS, random_seed=test)
        greedy_agent = Greedy(problem_to_solve)
        while problem_to_solve.has_unseen_workers():
            task, worker = greedy_agent.match()
            analysis.store_agreement_data(worker, task, problem_to_solve)
        scores.append(problem_to_solve.get_matching_score())

        # Find optimal solution to the problem:
        optimal_scores.append(calc_optimal_score(problem_to_solve))
    analysis.agreement_by_t()

    print(f"Greedy agent mean score: {np.mean(scores)} over {NUM_TESTS_TO_RUN} trials")
    print(f"Mean optimal score: {np.mean(optimal_scores)} over {NUM_TESTS_TO_RUN} trials")
    print(f"Optimality Ratio of Greedy: {np.mean(scores) / np.mean(optimal_scores)} over {NUM_TESTS_TO_RUN} trials")


def test_FFNet(model, data):
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
                state = torch.from_numpy(state).to(train_models.DEVICE)
                action, log_prob = model(state)
                analysis.store_agreement_data(worker, action, problem_to_solve)
                problem_to_solve.match(action, worker)
            scores.append(problem_to_solve.get_matching_score())

            optimal_scores.append(calc_optimal_score(problem_to_solve))
        analysis.agreement_by_t()

        print(f"FFNet agent mean score: {np.mean(scores)} over {NUM_TESTS_TO_RUN} trials")
        print(f"Mean optimal score: {np.mean(optimal_scores)} over {NUM_TESTS_TO_RUN} trials")
        print(f"Optimality Ratio of FFNet: {np.mean(scores) / np.mean(optimal_scores)} over {NUM_TESTS_TO_RUN} trials")


def test_InvFFNet(model, data):
    # BASIC TEST FOR FFNET:
    print("NOW TESTING InvFFNet:")
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
                worker, states = problem_to_solve.get_next_ff_inv_input()  # Pick the next worker to match and get the input as a vector
                states_log_prob = []
                states_log_prob_tf = []
                actions = []
                for state in states:
                    state = torch.from_numpy(np.array(state)).to(train_models.DEVICE)  # Need to convert the datatype to a tensor for pytorch
                    action, log_prob = model(state)  # Choose an action based on the model
                    states_log_prob.append(log_prob.detach().numpy())
                    states_log_prob_tf.append(log_prob)
                    actions.append(action)
                reward = problem_to_solve.match(actions[np.argmax(states_log_prob)], worker)  # Perform matching, calculate reward
                problem_to_solve.match(action, worker)
            scores.append(problem_to_solve.get_matching_score())

            optimal_scores.append(calc_optimal_score(problem_to_solve))
        analysis.agreement_by_t()

        print(f"InvFFNet agent mean score: {np.mean(scores)} over {NUM_TESTS_TO_RUN} trials")
        print(f"Mean optimal score: {np.mean(optimal_scores)} over {NUM_TESTS_TO_RUN} trials")
        print(f"Optimality Ratio of InvFFNet: {np.mean(scores) / np.mean(optimal_scores)} over {NUM_TESTS_TO_RUN} trials")


def calc_optimal_score(problem_to_solve):
    # Find optimal solution to the problem:
    row_ind, col_ind = linear_sum_assignment(problem_to_solve.get_all_edges(), maximize=True)
    optimal_score = 0
    for i, j in zip(row_ind, col_ind):
        optimal_score += problem_to_solve.get_all_edges()[i, j]
    return optimal_score

def main():
    # Load up the GMission Dataset:
    data = GMission()

    ############################
    # TEST FOR GREEDY ALGORITHM:
    ############################
    test_greedy_algorithm(data)

    # If no models have been trained locally, we want to perform training before we perform testing obviously:
    model_files = glob.glob("*.pth")
    if len(model_files) == 0:
        train_models.main()

    # Now we've run the training loop, we should have model files:
    model_files = glob.glob("*.pth")
    if len(model_files) == 0:
        raise FileNotFoundError("Training failed or Model files have not been generated")

    # # For each .pth file, load the model and perform the testing loop:
    # for model_file in model_files:
    #     model = LinearFFNet(NUM_TASKS * 2, NUM_TASKS)
    #     model.load_state_dict(torch.load(model_file))
    #     model.eval()
    #     test_FFNet(model, data)

    # For each .pth file, load the model and perform the testing loop:
    for model_file in model_files:
        model = InvFFNet(3, NUM_TASKS)
        model.load_state_dict(torch.load(model_file))
        model.eval()
        test_InvFFNet(model, data)

if __name__ == '__main__':
    main()