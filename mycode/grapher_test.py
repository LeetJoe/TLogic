import json
import numpy as np


class Grapher(object):
    def __init__(self, dataset_dir):
        """
        Store information about the graph (train/valid/test set).
        Add corresponding inverse quadruples to the data.

        Parameters:
            dataset_dir (str): path to the graph dataset directory

        Returns:
            None
        """

        self.dataset_dir = dataset_dir
        self.entity2id = json.load(open(dataset_dir + "entity2id.json"))
        self.relation2id_old = json.load(open(dataset_dir + "relation2id.json"))
        self.relation2id = self.relation2id_old.copy()
        counter = len(self.relation2id_old)

        # 把 relation2id_old 复制一份加入到 relation2id 里，relation2id 里有两份 relation2id_old，第二份 key 加了 _ 前缀
        for relation in self.relation2id_old:
            self.relation2id["_" + relation] = counter  # Inverse relation
            counter += 1
        self.ts2id = json.load(open(dataset_dir + "ts2id.json"))
        self.id2entity = dict([(v, k) for k, v in self.entity2id.items()])
        self.id2relation = dict([(v, k) for k, v in self.relation2id.items()])
        self.id2ts = dict([(v, k) for k, v in self.ts2id.items()])

        # 这里是一串id，按 len(self.relation2id_old) 分成两段，一段分别是从 0 : len(self.relation2id_old) 到
        # len(self.relation2id_old)-1 : 2*len(self.relation2id_old)-1, 后面是 len(self.relation2id_old) : 0 到
        # 2*len(self.relation2id_old)-1 : len(self.relation2id_old)-1. 前半部分的 id:value 与后半部分的 id:value 刚好是反的。
        self.inv_relation_id = dict()
        num_relations = len(self.relation2id_old)
        for i in range(num_relations):
            self.inv_relation_id[i] = i + num_relations
        for i in range(num_relations, num_relations * 2):
            self.inv_relation_id[i] = i % num_relations

        # 使用 create_store 后，txt 里的数据被 id 化，返回数据在第 0 维上分两半，前面一半是 [head id, relation id, tail id, ts id],
        # 后面一半是 [tail id, reverse relation id, head id, ts id], 前后两部分长度完全相等。
        self.train_idx = self.create_store("train.txt")
        self.valid_idx = self.create_store("valid.txt")
        self.test_idx = self.create_store("test.txt")
        # 在第 0 维在拼在一起
        self.all_idx = np.vstack((self.train_idx, self.valid_idx, self.test_idx))

        print("Grapher initialized.")

    def create_store(self, file):
        """
        Store the quadruples from the file as indices.
        The quadruples in the file should be in the format "subject\trelation\tobject\ttimestamp\n".

        Parameters:
            file (str): file name

        Returns:
            store_idx (np.ndarray): indices of quadruples
        """

        with open(self.dataset_dir + file, "r", encoding="utf-8") as f:
            quads = f.readlines()

        # 把 txt 文件里的行数据组织成一个四元组列表
        store = self.split_quads(quads)

        # 把 store 里的四元组，按照 entity2id, relation2id,ts2id 等关系，将文本内容替换成对应的id。
        # 在 store 里一项是一个自然语言的四元组，使用 split_quads 后，就变成了一个整数构成的四元组。
        store_idx = self.map_to_idx(store)
        store_idx = self.add_inverses(store_idx)

        return store_idx

    def split_quads(self, quads):
        """
        Split quadruples into a list of strings.

        Parameters:
            quads (list): list of quadruples
                          Each quadruple has the form "subject\trelation\tobject\ttimestamp\n".

        Returns:
            split_q (list): list of quadruples
                            Each quadruple has the form [subject, relation, object, timestamp].
        """

        split_q = []
        for quad in quads:
            split_q.append(quad[:-1].split("\t"))

        return split_q

    def map_to_idx(self, quads):
        """
        Map quadruples to their indices.

        Parameters:
            quads (list): list of quadruples
                          Each quadruple has the form [subject, relation, object, timestamp].

        Returns:
            quads (np.ndarray): indices of quadruples
        """

        subs = [self.entity2id[x[0]] for x in quads]
        rels = [self.relation2id[x[1]] for x in quads]
        objs = [self.entity2id[x[2]] for x in quads]
        tss = [self.ts2id[x[3]] for x in quads]
        quads = np.column_stack((subs, rels, objs, tss))

        return quads

    def add_inverses(self, quads_idx):
        """
        Add the inverses of the quadruples as indices.

        Parameters:
            quads_idx (np.ndarray): indices of quadruples

        Returns:
            quads_idx (np.ndarray): indices of quadruples along with the indices of their inverses
        """

        subs = quads_idx[:, 2]  # id of tail entity
        rels = [self.inv_relation_id[x] for x in quads_idx[:, 1]]  # id of relation, 将 id 与下标互换
        objs = quads_idx[:, 0]  # id of head entity
        tss = quads_idx[:, 3]   # id of time
        inv_quads_idx = np.column_stack((subs, rels, objs, tss))
        quads_idx = np.vstack((quads_idx, inv_quads_idx))

        return quads_idx
