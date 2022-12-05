import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from train_models import NUM_TASKS, NUM_WORKERS, REWARD_MODES

class Analysis():
    
    def __init__(self, num_tasks, num_workers, num_tests):
        self.opt_ratio_data = {}
        self.agreement_data = {'GREEDY': np.zeros(num_workers), 'FINAL_REWARD': np.zeros(num_workers), 'SARSA_REWARD': np.zeros(num_workers), 'TOTAL_REWARD': np.zeros(num_workers)}
        self.timestep = 0
        self.num_tasks = num_tasks
        self.num_workers = num_workers
        self.num_tests = num_tests

    def optimality_boxplots(self):
        data = []
        for model in self.opt_ratio_data.values():
            data.append(model['agent_scores']/model['optimal_scores'])
        plt.title(f'Optimality Distributions {self.num_tasks}x{self.num_workers}')
        plt.boxplot(data, labels=self.opt_ratio_data.keys())
        plt.savefig(f'optimality_boxplots_{self.num_tasks}x{self.num_workers}.png')
        plt.close()
    
    def agreement_by_t(self):
        x = np.linspace(1, self.num_workers + 1, self.num_workers)
        for model, agreement in self.agreement_data.items():
            plt.plot(x, agreement/self.num_tests, label=model)
        plt.title(f'Agreement per Timestep {self.num_tasks}x{self.num_workers}')
        plt.legend()
        plt.savefig(f'agreement_by_t_{self.num_tasks}x{self.num_workers}.png')
        plt.close()

    def store_opt_ratio_data(self, agent_name, agent_scores, optimal_scores):
        self.opt_ratio_data[agent_name] = {'agent_scores': np.array(agent_scores), 'optimal_scores': np.array(optimal_scores)}

    def store_agreement_data(self, worker, action, problem, agent_name):
        row, col = linear_sum_assignment(problem.get_all_edges(), maximize=True)
        optimal_match = row[np.where(col == action)[0]]
        if len(optimal_match) > 0:
            if action in problem.get_matchings().keys():
                if worker not in row:
                    self.agreement_data[agent_name][self.timestep] += 1
            else:
                if problem.get_all_edges()[worker][action] <= 0:
                    if worker not in row:
                        self.agreement_data[agent_name][self.timestep] += 1
                elif optimal_match[0] == worker:
                    self.agreement_data[agent_name][self.timestep] += 1
        else:
            if worker not in row:
                self.agreement_data[agent_name][self.timestep] += 1

        if self.timestep == self.num_workers - 1:
            self.timestep = 0
        else:
            self.timestep += 1
    

    def calculate_optimal(self, problem):
            row_ind, col_ind = linear_sum_assignment(problem.get_all_edges(), maximize=True)
            optimal_score = 0
            for i, j in zip(row_ind, col_ind):
                optimal_score += problem.get_all_edges()[i, j]
            return optimal_score

    def plot_logs(self, log_file):
        data = np.loadtxt(log_file, delimiter=',', dtype=float)
        plt.xlabel('Episode')
        plt.ylabel('Matching Score')
        name = log_file.replace('_TRAINING_LOG.csv', '')
        name = name.replace('_', ' ')
        plt.title(name)
        plt.plot(data[:,0], data[:,1])
        plt.savefig(f"{log_file.replace('.csv', '')}_plot.png")
        plt.close()




        

