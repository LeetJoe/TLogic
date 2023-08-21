import numpy as np


class Temporal_Walk(object):
    def __init__(self, learn_data, inv_relation_id, transition_distr):
        """
        Initialize temporal random walk object.

        Parameters:
            learn_data (np.ndarray): data on which the rules should be learned
            inv_relation_id (dict): mapping of relation to inverse relation
            transition_distr (str): transition distribution
                                    "unif" - uniform distribution
                                    "exp"  - exponential distribution

        Returns:
            None
        """

        self.learn_data = learn_data
        self.inv_relation_id = inv_relation_id
        self.transition_distr = transition_distr

        # neighbors 以 entity id 为下标，长度与 entity2id 的大小相同，item 是 head 与 id(也是下标) 相同的所有四元组(id 形式)
        self.neighbors = store_neighbors(learn_data)
        # edges 以 relation id 为下标，item 是与 relation id(也是下标) 相同的所有四元组(id形式), 但是 len(edges) 与
        # relation 的数量不相同，可能有 reverse rel 的原因。
        self.edges = store_edges(learn_data)

    def sample_start_edge(self, rel_idx):
        """
        Define start edge distribution.

        Parameters:
            rel_idx (int): relation index

        Returns:
            start_edge (np.ndarray): start edge
        """

        rel_edges = self.edges[rel_idx]
        start_edge = rel_edges[np.random.choice(len(rel_edges))]

        return start_edge

    def sample_next_edge(self, filtered_edges, cur_ts):
        """
        Define next edge distribution. 如果使用均匀分布，则候选 edges 有相等的概率被选中；如果使用指数分布，则使用类似 softmax 的算法
        设定选中概率，ts 离 cur_ts 越近则选中的概率越大。

        Parameters:
            filtered_edges (np.ndarray): filtered (according to time) edges
            cur_ts (int): current timestamp

        Returns:
            next_edge (np.ndarray): next edge
        """

        if self.transition_distr == "unif":   # uniform distribution, 均匀分布，从候选中随机选一个
            next_edge = filtered_edges[np.random.choice(len(filtered_edges))]
        elif self.transition_distr == "exp":   # exponential distribution, 指数分布
            tss = filtered_edges[:, 3]   # 这里得到的是一个向量，下面应用 np.exp() 后得到的也是一个向量
            prob = np.exp(tss - cur_ts)   # 由于之前的筛选，tss 必定不大于 cur_ts；np.exp(x) 即 e^x。
            try:
                prob = prob / np.sum(prob)    # todo 这个不就是 transformer 里的 softmax 操作吗？
                next_edge = filtered_edges[
                    np.random.choice(range(len(filtered_edges)), p=prob)   # 第一个参数是候选列表或候选 range，p=prop 表示第一个参数中对应位置的候选被选中的概率。
                ]
            except ValueError:  # All timestamps are far away
                next_edge = filtered_edges[np.random.choice(len(filtered_edges))]

        return next_edge

    def transition_step(self, cur_node, cur_ts, prev_edge, start_node, step, L):
        """
        Sample a neighboring edge given the current node and timestamp.
        In the second step (step == 1), the next timestamp should be smaller than the current timestamp.
        In the other steps, the next timestamp should be smaller than or equal to the current timestamp.
        In the last step (step == L-1), the edge should connect to the source of the walk (cyclic walk).
        It is not allowed to go back using the inverse edge.

        Parameters:
            cur_node (int): current node 实际上就是当前4元组的 tail
            cur_ts (int): current timestamp 当前4元组的 ts
            prev_edge (np.ndarray): previous edge 当前4元组的 rel
            start_node (int): start node 当前4元组的 head
            step (int): number of current step
            L (int): length of random walk

        Returns:
            next_edge (np.ndarray): next edge
        """

        next_edges = self.neighbors[cur_node]

        # 论文里都是按时间从后往前找，这里也是如此实现的
        if step == 1:  # The next timestamp should be smaller than the current timestamp
            filtered_edges = next_edges[next_edges[:, 3] < cur_ts]
        else:  # The next timestamp should be smaller than or equal to the current timestamp
            filtered_edges = next_edges[next_edges[:, 3] <= cur_ts]
            # Delete inverse edge
            inv_edge = [
                cur_node,
                self.inv_relation_id[prev_edge[1]],
                prev_edge[0],
                cur_ts,
            ]
            # 这里 filtered_edges == inv_edge 返回的应该是一个数组，就是 filtered_edges 里所有4元组里对应位置的元素与 inv_edge 是否相等，
            # 得到一个由 true, false 构成的矩阵，然后用 where 把全为 true 的那些行的 id 返回。
            # todo 这里 axis=1 的含义表示：对这个 n*4 的矩阵，按纵向进行 reduction，最后得到的是一个 n*1 的向量，即在纵向上"收缩"，
            #  也就是按行进行逻辑与，只有整行都是 true 的行结果才是 true，这样就把目标找出来了。
            row_idx = np.where(np.all(filtered_edges == inv_edge, axis=1))
            # 这里的 axis=0 就是按行删除的意思
            filtered_edges = np.delete(filtered_edges, row_idx, axis=0)

        if step == L - 1:  # Find an edge that connects to the source of the walk
            filtered_edges = filtered_edges[filtered_edges[:, 2] == start_node]

        if len(filtered_edges):
            # 在 exp 模式下，ts 与 cur_ts 离得近的 edge 被选中的概率更高
            next_edge = self.sample_next_edge(filtered_edges, cur_ts)
        else:
            next_edge = []

        return next_edge

    def sample_walk(self, L, rel_idx):
        """
        Try to sample a cyclic temporal random walk of length L (for a rule of length L-1).

        Parameters:
            L (int): length of random walk, 示例里给的是 1,2,3 此值并不长(这里会先加1，所以传过来是2,3,4)。
            rel_idx (int): relation index, 基于 relation 来 walk

        Returns:
            walk_successful (bool): if a cyclic temporal random walk has been successfully sampled
            walk (dict): information about the walk (entities, relations, timestamps)
            todo walk 里 len(entities) = len(relations) + 1 = len(timestamps) + 1, 即一个 path: e0, r1, t1, e1, r2, t2, e2 ...
                或 walk 不为空，则有 L = len(relations)
        """

        walk_successful = True
        walk = dict()
        prev_edge = self.sample_start_edge(rel_idx)  # 从 self.edges[rel_idx] 里随机取一个四元组
        # 如果考虑到反向 relation 的存在，那么这里的 start node 有可能是原数据集里四元组的 tail 而非 head.
        start_node = prev_edge[0]
        cur_node = prev_edge[2]
        cur_ts = prev_edge[3]
        walk["entities"] = [start_node, cur_node]
        walk["relations"] = [prev_edge[1]]
        walk["timestamps"] = [cur_ts]

        for step in range(1, L):
            next_edge = self.transition_step(
                cur_node, cur_ts, prev_edge, start_node, step, L
            )
            if len(next_edge):
                cur_node = next_edge[2]
                cur_ts = next_edge[3]
                walk["relations"].append(next_edge[1])
                walk["entities"].append(cur_node)
                walk["timestamps"].append(cur_ts)
                prev_edge = next_edge
            else:  # No valid neighbors (due to temporal or cyclic constraints)
                walk_successful = False
                break

        return walk_successful, walk


def store_neighbors(quads):
    """
    Store all neighbors (outgoing edges) for each node.

    Parameters:
        quads (np.ndarray): indices of quadruples

    Returns:
        neighbors (dict): neighbors for each node
    """

    neighbors = dict()
    nodes = list(set(quads[:, 0]))
    for node in nodes:
        neighbors[node] = quads[quads[:, 0] == node]

    return neighbors


def store_edges(quads):
    """
    Store all edges for each relation.

    Parameters:
        quads (np.ndarray): indices of quadruples

    Returns:
        edges (dict): edges for each relation
    """

    edges = dict()
    relations = list(set(quads[:, 1]))
    for rel in relations:
        edges[rel] = quads[quads[:, 1] == rel]

    return edges
