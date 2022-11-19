import numpy as np
from scipy.optimize import linear_sum_assignment
from Data.load_dataset import GMission
from Agents.Greedy import Greedy

NUM_TESTS_TO_RUN = 5000


def main():
    # Load up the GMission Dataset:
    data = GMission()
    scores = []
    optimal_scores = []

    for test in range(NUM_TESTS_TO_RUN):
        # Solve the problem using the greedy algorithm:
        problem_to_solve = data.generate_olbm_instance(random_seed=test)
        greedy_agent = Greedy(problem_to_solve)
        greedy_agent.solve_olbm()
        scores.append(problem_to_solve.get_matching_score())

        # Find optimal solution to the problem:
        row_ind, col_ind = linear_sum_assignment(problem_to_solve.get_all_edges(), maximize=True)
        optimal_score = 0
        for i, j in zip(row_ind, col_ind):
            optimal_score += problem_to_solve.get_all_edges()[i, j]
        optimal_scores.append(optimal_score)


    print(f"Greedy agent mean score: {np.mean(scores)} over {NUM_TESTS_TO_RUN} trials")
    print(f"Mean optimal score: {np.mean(optimal_scores)} over {NUM_TESTS_TO_RUN} trials")
    print(f"Optimality Ratio of Greedy: {np.mean(scores) / np.mean(optimal_scores)} over {NUM_TESTS_TO_RUN} trials")


if __name__ == '__main__':
    main()
