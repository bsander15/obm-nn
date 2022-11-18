import numpy as np

gMission_edges = "gMissionDataset/edges.txt"

class gMission:
    """
    Return the workers,tasks and cost matrix of workers and tasks
    """
    def load(self):
        f_edges = open(gMission_edges, "r")
        edgeWeights = dict()
        for line in f_edges:
            vals = line.split(",")
            edgeWeights[vals[0]] = vals[1].split("\n")[0]
        w = np.array(list(edgeWeights.values()), dtype="float")
        max_w = max(w)
        edges = {k: (float(v) / float(max_w)) for k, v in edgeWeights.items()}
        workers = set([int(float(key.split(";")[0])) for key in edges])
        tasks = set([int(float(key.split(";")[1])) for key in edges])
        cost = np.ones((max(workers)+1, max(tasks)+1))
        for k,v in edges.items():
            i,j = k.split(";")
            i = int(float(i))
            j = int(float(j))
            cost[i][j] = v
        return workers,tasks,cost
