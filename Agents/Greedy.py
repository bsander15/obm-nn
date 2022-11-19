import numpy as np
from Data.load_dataset import OLBMInstance


class Greedy:
    """
    Greedy algorithm for edge-weighted online bipartite matching
    """

    def __init__(self, olbm_problem: OLBMInstance):
        """
        Initialize class with set of fixed nodes
        Params:
            olbm_problem: an OLBMInstance to solve
        """
        self.olbm_problem = olbm_problem

    def match(self):
        """
        Get the next online worker from the OLBM problem instance and match it to the unmatched problem with the
        maximum connection to the next worker.
        """
        next_worker = self.olbm_problem.get_next_worker()  # Get the next online worker from the problem
        unmatched_tasks = self.olbm_problem.get_matched_bitmap()  # 1s indicate that the nth task is unmatched
        next_worker_edges = self.olbm_problem.get_worker_edges(next_worker)  # Get edges corresponding to next worker
        matchable_edges = next_worker_edges * unmatched_tasks

        # Check if there exists an unmatched task that the next worker can be matched to:
        if matchable_edges.any():
            best_task_to_match_next_worker_to = np.argmax(matchable_edges)
            self.olbm_problem.match(best_task_to_match_next_worker_to, next_worker)

    def solve_olbm(self):
        while self.olbm_problem.has_unseen_workers():
            self.match()

#Brief testing
# greedy = Greedy(['1','2','3','4'])
# nodes = {'1n': {'1': 5, '3': 9}, '2n': {'2': 4, '3': 5}, '3n': {'1': 10, '4': 1}, '4n' :{'4': 3}}
# for node in nodes.keys():
#     greedy.match(node, nodes[node])
# print(greedy.get_matches())


        