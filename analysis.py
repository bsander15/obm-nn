import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment



class Analysis():
    
    def __init__(self, num_workers, num_tests):
        self.test_data = {}
        self.agreement_data = np.zeros(num_workers)
        self.timestep = 0
        self.num_tests = num_tests
        self.matched_good_pass = 0
        self.matched_bad_pass = 0
        self.no_match_good_match = 0
        self.no_match_bad_match = 0
        self.pass_good_pass = 0
        self.pass_bad_pass = 0
        self.seen = set()

    def optimality_boxplots(self): 
        data = []
        for scores in self.test_data.values():
            data.append(scores['agent_scores']/scores['optimal_scores'])
        plt.title('Optimality Distributions')
        plt.boxplot(data, labels=self.test_data.keys())
        plt.show()
    
    def agreement_by_t(self):
        data = self.agreement_data/self.num_tests
        print(self.agreement_data)
        print(data)
        print(f'''Matched Good Pass: {self.matched_good_pass}
        Matched Bad Pass: {self.matched_bad_pass}
        No Match Good Match {self.no_match_good_match}
        No Match Bad Match {self.no_match_bad_match}
        Pass Good Pass {self.pass_good_pass}
        Pass Bad Pass {self.pass_bad_pass}
        ''')
        plt.title('Agreement per Timestep')
        x = np.linspace(1, len(data) + 1, len(data))
        plt.plot(x, data)
        plt.show()

    def store_test_data(self, agent_name, agent_scores, optimal_scores):
        self.test_data[agent_name] = {'agent_scores': np.array(agent_scores), 'optimal_scores': np.array(optimal_scores)}

    def store_agreement_data(self, worker, action, problem):
        row, col = linear_sum_assignment(problem.get_all_edges(), maximize=True)
        optimal_match = row[np.where(col == action)[0]]
        if len(optimal_match) > 0:
            if action in problem.get_matchings().keys():
                if worker not in row:
                    self.matched_good_pass +=1
                    self.agreement_data[self.timestep] += 1
                else:
                    self.matched_bad_pass += 1
            else:
                if problem.get_all_edges()[worker][action] <= 0:
                    if worker not in row:
                        self.agreement_data[self.timestep] += 1
                elif optimal_match[0] == worker:
                    self.no_match_good_match += 1
                    self.agreement_data[self.timestep] += 1
                else:
                    self.no_match_bad_match += 1

        else:
            if worker not in row:
                self.pass_good_pass += 1
                self.agreement_data[self.timestep] += 1
            self.pass_bad_pass
        if self.timestep == len(self.agreement_data) - 1:
            self.timestep = 0
            self.seen = set()
        else:
            self.timestep += 1
    

    def calculate_optimal(self, problem):
            row_ind, col_ind = linear_sum_assignment(problem.get_all_edges(), maximize=True)
            optimal_score = 0
            for i, j in zip(row_ind, col_ind):
                optimal_score += problem.get_all_edges()[i, j]
            # print(f'Optimal: {row_ind, col_ind}')
            return optimal_score

# a = Analysis()
# a.store_data('test', [1,2,4,2,2,4,1,3,4,6], [7,8,6,9,9,7,5,4,5,8])
# a.store_data('test1', [1,2,4,2,2,4,1,3,4,6], [7,8,6,9,9,7,5,4,5,8])
# a.optimality_boxplots()
t = {}
print('test' in t)
# t = np.array([1,2,3])/np.array([1,2,3])
# print(t)

