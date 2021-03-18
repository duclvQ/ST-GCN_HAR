import numpy as np


class Graph():
    def __init__(self, max_hop=1, dilation=1, strategy='distance', isFPHAB=True):
        self.max_hop = max_hop
        self.dilation = dilation
        self.strategy = strategy
        self.isFPHAB   = isFPHAB
        # get edges
        self.num_node, self.edge, self.center = self._get_edge()

        # get adjacency matrix 
        self.hop_dis = self._get_hop_distance()

        # normalization
        self.A = self._get_adjacency()

    def __str__(self):
        return self.A

    def _get_edge(self):
        if self.isFPHAB == True:
            num_node = 21
            neighbor_1base = [(1, 2), (2,7), (7, 8), (8,9), 
                            (1, 3), (3, 10), (10,11), (11,12),
                            (1,4), (4,13), (13,14), (14, 15),
                            (1,5), (5,16), (16, 17), (17, 18),
                            (1,6), (6,19), (19, 20), (20, 21)]
            center = 0 
        
        self_link = [(i, i) for i in range(num_node)]
        neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
        edge = self_link + neighbor_link
        
        return (num_node, edge, center)

    def _get_hop_distance(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_adjacency(self):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        # for i, hop in enumerate(valid_hop):
        #     A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
        # return A
        if self.strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            return A
        elif self.strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            return A

    def _normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD

# graph = Graph(max_hop=2, strategy='spatial')
# graph.A = graph.A[:4]
# print(graph.A.shape)
# print("==================")
# graph = Graph(max_hop=1, strategy='spatial')
# print(graph.A[0])
