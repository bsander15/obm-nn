import numpy as np

gMission_edges = "gMissionDataset/edges.txt"

class gMission:
    """
    Return the workers,tasks and cost matrix of workers and tasks
    """
    def load(self):
        """
        :return (workers, tasks, cost) - workers is a set of all the workers, while tasks is a set
        of all tasks. The Costs is a worker by task cost array where the [worker][task]th item is
        the expected value of matching the worker to a task.
        """
        f_edges = open(gMission_edges, "r")
        edgeWeights = dict()
        for line in f_edges:
            vals = line.split(",")
            edgeWeights[vals[0]] = vals[1].strip()
        w = np.array(list(edgeWeights.values()), dtype="float")
        max_w = max(w)
        edges = {k: (float(v) / float(max_w)) for k, v in edgeWeights.items()}
        workers = set([int(float(key.split(";")[0])) for key in edges])
        tasks = set([int(float(key.split(";")[1])) for key in edges])
        cost = np.zeros((max(workers)+1, max(tasks)+1))
        for k,v in edges.items():
            i,j = k.split(";")
            i = int(float(i))
            j = int(float(j))
            cost[i][j] = v
        return workers,tasks,cost

if __name__ == '__main__':
    a = gMission()
    a.load()