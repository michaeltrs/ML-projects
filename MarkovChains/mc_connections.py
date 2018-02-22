import numpy as np


class MarkovChain:

    def __init__(self, P, S=None):
        """
        """
        self.P = P
        if S is None:
            self.S = np.arange(np.shape(P)[0])
        else:
            self.S = np.array(S)

    def apply_ntimes(self, x, n, return_hist=False):
        hist = [x]
        for i in range(n):
            x = x.dot(self.P)
            hist.append(x)
        if return_hist:
            return hist
        else:
            return hist[-1]

    def transition_matrix2graph(self, P=None):
        if P is None:
            P = self.P
        graph = {}
        for i, s in enumerate(self.S):
            graph[s] = self.S[P[i, :]>0]
        return graph

    def strongly_connected_components(self, graph):
        """
        Tarjan's Algorithm (named for its discoverer, Robert Tarjan) is a graph theory algorithm
        for finding the strongly connected components of a graph.
        Based on: http://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
        """

        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        result = []

        def strongconnect(node):
            # set the depth index for this node to the smallest unused index
            index[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)

            # Consider successors of `node`
            try:
                successors = graph[node]
            except:
                successors = []
            for successor in successors:
                if successor not in lowlinks:
                    # Successor has not yet been visited; recurse on it
                    strongconnect(successor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[successor])
                elif successor in stack:
                    # the successor is in the stack and hence in the current strongly connected component (SCC)
                    lowlinks[node] = min(lowlinks[node], index[successor])

            # If `node` is a root node, pop the stack and generate an SCC
            if lowlinks[node] == index[node]:
                connected_component = []

                while True:
                    successor = stack.pop()
                    connected_component.append(successor)
                    if successor == node: break
                component = tuple(connected_component)
                # storing the result
                result.append(component)

        for node in graph:
            if node not in lowlinks:
                strongconnect(node)

return result
