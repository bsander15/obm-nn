"""
Greedy algorithm for edge-weighted online bipartite matching
"""
class Greedy:

    """
    Initialize class with set of fixed nodes
    Params: 
        fixed_nodes (List)
    """
    def __init__(self, fixed_nodes):
        self.matched_nodes = {}
        for node in fixed_nodes:
            self.matched_nodes[node] = {'match': None, 'weight': None}
    
    """
    Match online node to set of fixed nodes, if possible
    Params:
        online_node: Node to be matched (Any)
        edges: Dictionary of fixed nodes and corresponding edge weight (Dict)
    """
    def match(self, online_node, edges):
        max_fixed_node = max(edges, key=edges.get)
        if self.matched_nodes[max_fixed_node]['match'] is None:
            self.matched_nodes[max_fixed_node].update({'match': online_node, 'weight': edges[max_fixed_node]})
        else:
            del edges[max_fixed_node]
            if edges:
                self.match(online_node, edges)
    
    """
    Return the set of matched nodes
    """
    def get_matches(self):
        return self.matched_nodes

#Brief testing
greedy = Greedy(['1','2','3','4'])
nodes = {'1n': {'1': 5, '3': 9}, '2n': {'2': 4, '3': 5}, '3n': {'1': 10, '4': 1}, '4n' :{'4': 3}}
for node in nodes.keys():
    greedy.match(node, nodes[node])
print(greedy.get_matches())


        