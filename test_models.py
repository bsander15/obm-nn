import glob
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

import train_models
from Agents.FFNet import LinearFFNet
from Agents.Greedy import Greedy
from Data.load_dataset import GMission
from analysis import Analysis
from train_models import NUM_WORKERS, NUM_TASKS, NUM_TESTS_TO_RUN, REWARD_MODES


def test_greedy_algorithm(data, num_tasks, num_workers, analysis):
    scores = []
    optimal_scores = []
    for test in range(NUM_TESTS_TO_RUN):
        # Solve the problem using the greedy algorithm:
        problem_to_solve = data.generate_olbm_instance(num_tasks=num_tasks, num_workers=num_workers, random_seed=test)
        greedy_agent = Greedy(problem_to_solve)
        while problem_to_solve.has_unseen_workers():
            task, worker = greedy_agent.match()
            analysis.store_agreement_data(worker, task, problem_to_solve, 'GREEDY')
        scores.append(problem_to_solve.get_matching_score())

        # Find optimal solution to the problem:
        optimal_scores.append(calc_optimal_score(problem_to_solve))
    analysis.store_opt_ratio_data('GREEDY', scores, optimal_scores)
    # analysis.agreement_by_t()

    print(f"Greedy agent mean score: {np.mean(scores)} over {NUM_TESTS_TO_RUN} trials")
    print(f"Mean optimal score: {np.mean(optimal_scores)} over {NUM_TESTS_TO_RUN} trials")
    print(f"Optimality Ratio of Greedy: {np.mean(scores) / np.mean(optimal_scores)} over {NUM_TESTS_TO_RUN} trials")


def test_FFNet(model, data, model_name, num_tasks, num_workers, analysis):
    # BASIC TEST FOR FFNET:
    print("NOW TESTING LinearFFNet:")
    scores = []
    optimal_scores = []
    model.eval()
    with torch.no_grad():
        for test in tqdm(range(NUM_TESTS_TO_RUN)):
            # Solve the problem using the greedy algorithm:
            problem_to_solve = data.generate_olbm_instance(num_tasks=num_tasks,
                                                           num_workers=num_workers,
                                                           random_seed=test)
            while problem_to_solve.has_unseen_workers():
                worker, state = problem_to_solve.get_next_nn_input()
                state = torch.from_numpy(state).to(train_models.DEVICE)
                action, log_prob = model(state)
                analysis.store_agreement_data(worker, action, problem_to_solve, model_name)
                problem_to_solve.match(action, worker)
            scores.append(problem_to_solve.get_matching_score())

            optimal_scores.append(calc_optimal_score(problem_to_solve))
        analysis.store_opt_ratio_data(model_name, scores, optimal_scores)

        print(f"FFNet agent mean score: {np.mean(scores)} over {NUM_TESTS_TO_RUN} trials")
        print(f"Mean optimal score: {np.mean(optimal_scores)} over {NUM_TESTS_TO_RUN} trials")
        print(f"Optimality Ratio of FFNet: {np.mean(scores) / np.mean(optimal_scores)} over {NUM_TESTS_TO_RUN} trials")


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

    # If no models have been trained locally, we want to perform training before we perform testing obviously:
    model_files = glob.glob("*.pth")
    if len(model_files) == 0:
        train_models.main()

    # Now we've run the training loop, we should have model files:
    model_files = glob.glob("*.pth")
    if len(model_files) == 0:
        raise FileNotFoundError("Training failed or Model files have not been generated")

    # For each .pth file, load the model and perform the testing loop:
    for num_tasks, num_workers in zip(NUM_TASKS, NUM_WORKERS):
        analysis = Analysis(num_tasks, num_workers, NUM_TESTS_TO_RUN)
        test_greedy_algorithm(data, num_tasks, num_workers, analysis)
        print(analysis.agreement_data)
        for reward in REWARD_MODES:
            model = LinearFFNet(num_tasks * 2, num_tasks)
            model_name = f'{reward}_{num_tasks}X{num_workers}'
            model.load_state_dict(torch.load(f'{model_name}.pth'))
            model.eval()
            test_FFNet(model, data, reward, num_tasks, num_workers, analysis)
            analysis.plot_logs(f'{model_name}_TRAINING_LOG.csv')
        analysis.optimality_boxplots()
        analysis.agreement_by_t()

if __name__ == '__main__':
    main()