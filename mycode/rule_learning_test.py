import os
import json
import itertools
import numpy as np
from collections import Counter


class Rule_Learner(object):
    def __init__(self, edges, id2relation, inv_relation_id, dataset):
        """
        Initialize rule learner object.

        Parameters:
            edges (dict): edges for each relation
            id2relation (dict): mapping of index to relation
            inv_relation_id (dict): mapping of relation to inverse relation
            dataset (str): dataset name

        Returns:
            None
        """

        self.edges = edges
        self.id2relation = id2relation
        self.inv_relation_id = inv_relation_id

        self.found_rules = []
        self.rules_dict = dict()
        # self.rules_dict 的结构为：{head_rel:[{'head_rel': 0, 'body_rels': [222], 'var_constraints': [], 'conf': 0.12,
        # 'rule_supp': 54, 'body_supp': 430},...]}, 注意键 head_rel 对应的是一个列表, 列表中的每一项是一个 dict
        self.output_dir = "../output/" + dataset + "/"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def create_rule(self, walk):
        """
        Create a rule given a cyclic temporal random walk.
        The rule contains information about head relation, body relations,
        variable constraints, confidence, rule support, and body support.
        A rule is a dictionary with the content
        {"head_rel": int, "body_rels": list, "var_constraints": list,
         "conf": float, "rule_supp": int, "body_supp": int}

        Parameters:
            walk (dict): cyclic temporal random walk
                         {"entities": list, "relations": list, "timestamps": list}

        Returns:
            rule (dict): created rule
        """

        rule = dict()
        rule["head_rel"] = int(walk["relations"][0])
        # 除第一个 relation 外其它 relation 倒序并更改方向，放在 rule['body_rels'] 里
        rule["body_rels"] = [
            self.inv_relation_id[x] for x in walk["relations"][1:][::-1] # [::n]表示设置步长，n=-1 表示倒序输出
        ]
        # rule["var_constraints"] 里存放的是 entities 里那些在路径中出现不止一次的节点的"位置(下标)"
        rule["var_constraints"] = self.define_var_constraints(
            walk["entities"][1:][::-1]  # 表示除第一个 entity 外，其它 entities 的倒序
        )

        if rule not in self.found_rules:
            self.found_rules.append(rule.copy())   # 将 rule 记录下来
            (
                rule["conf"],
                rule["rule_supp"],
                rule["body_supp"],
            ) = self.estimate_confidence(rule)   # 对 rule 进行应用，评价其有效性，得到 confidence, rule support, body support

            if rule["conf"]:
                # 更新 self.rules_dict，其结构见定义处注释
                self.update_rules_dict(rule)

    def define_var_constraints(self, entities):
        """
        Define variable constraints, i.e., state the indices of reoccurring entities in a walk.
        返回 entities 里那些多次（大于1次）在其中出现的节点在 entities 里的位置（在一个 walk 里）

        Parameters:
            entities (list): entities in the temporal walk

        Returns:
            var_constraints (list): list of indices for reoccurring entities
        """

        var_constraints = []
        for ent in set(entities):   # set 一下相当于取 distinct
            # 找到 entities 里与当前 ent 相等的那些元素的位置
            all_idx = [idx for idx, x in enumerate(entities) if x == ent]
            var_constraints.append(all_idx)
        # 如果某元素不止一次出现在 entities 中，将这些元素在 entities 中的位置按元素分批依次放入 var_constraints(一维)
        var_constraints = [x for x in var_constraints if len(x) > 1]

        return sorted(var_constraints)  # 返回的时候排序了，也就是说顺序无关紧要

    def estimate_confidence(self, rule, num_samples=500):
        """
        Estimate the confidence of the rule by sampling bodies and checking the rule support.

        Parameters:
            rule (dict): rule
                         {"head_rel": int, "body_rels": list, "var_constraints": list}
            num_samples (int): number of samples

        Returns:
            confidence (float): confidence of the rule, rule_support/body_support
            rule_support (int): rule support
            body_support (int): body support
        """

        all_bodies = []
        for _ in range(num_samples):
            sample_successful, body_ents_tss = self.sample_body(
                rule["body_rels"], rule["var_constraints"]
            )
            if sample_successful:
                all_bodies.append(body_ents_tss)

        all_bodies.sort()  # 似乎是使用每一子 list 的第一个元素的值来升序排列的
        unique_bodies = list(x for x, _ in itertools.groupby(all_bodies))   # 除去完全一样的元素
        body_support = len(unique_bodies)

        confidence, rule_support = 0, 0
        if body_support:
            # todo 所以 rule_support 即是应用当前 rule 随机构500条路径，其中符合 rule 的要求的那些路径的数量。
            rule_support = self.calculate_rule_support(unique_bodies, rule["head_rel"])
            confidence = round(rule_support / body_support, 6)

        return confidence, rule_support, body_support

    def sample_body(self, body_rels, var_constraints):
        """
        Sample a walk according to the rule body.
        The sequence of timesteps should be non-decreasing.

        按 body_rels 提供关系序列，从body_rels[0] 开始随机选择一个边为开端，构建一个新的路径，而且要求这具新的路径里的重复节点的位置与数量
        要与 var_constraints 相同。

        Parameters:
            body_rels (list): relations in the rule body
            var_constraints (list): variable constraints for the entities

        Returns:
            sample_successful (bool): if a body has been successfully sampled
            body_ents_tss (list): entities and timestamps (alternately entity and timestamp)
                                  of the sampled body
        """

        sample_successful = True
        body_ents_tss = []
        cur_rel = body_rels[0]
        rel_edges = self.edges[cur_rel]
        next_edge = rel_edges[np.random.choice(len(rel_edges))]
        cur_ts = next_edge[3]
        cur_node = next_edge[2]
        body_ents_tss.append(next_edge[0])
        body_ents_tss.append(cur_ts)
        body_ents_tss.append(cur_node)

        for cur_rel in body_rels[1:]:
            next_edges = self.edges[cur_rel]
            mask = (next_edges[:, 0] == cur_node) * (next_edges[:, 3] >= cur_ts)
            filtered_edges = next_edges[mask]

            if len(filtered_edges):
                next_edge = filtered_edges[np.random.choice(len(filtered_edges))]
                cur_ts = next_edge[3]
                cur_node = next_edge[2]
                body_ents_tss.append(cur_ts)
                body_ents_tss.append(cur_node)
            else:
                sample_successful = False
                break

        # body_ents_tss 的结构为 [node0, ts1, node1, ts2, node2, ..., ts_n, node_n]
        if sample_successful and var_constraints:
            # Check variable constraints
            body_var_constraints = self.define_var_constraints(body_ents_tss[::2])   # body_ents_tss[::2] 即取所有的 nodes
            if body_var_constraints != var_constraints:   # todo 如果取样的实体里重复出现的实体的位置与数量不相等，则认为失败？
                sample_successful = False

        return sample_successful, body_ents_tss

    def calculate_rule_support(self, unique_bodies, head_rel):
        """
        Calculate the rule support. Check for each body if there is a timestamp
        (larger than the timestamps in the rule body) for which the rule head holds.

        Parameters:
            unique_bodies (list): bodies from self.sample_body
            head_rel (int): head relation

        Returns:
            rule_support (int): rule support
        """

        rule_support = 0
        head_rel_edges = self.edges[head_rel]
        # todo body in unique_bodies 的结构是 [node, ts, node]
        for body in unique_bodies:
            mask = (    # 这里面是三个向量相乘，最后只有三个条件都满足的位置才会是 True，得到的 mask 是一个由 true/false 组成的向量
                (head_rel_edges[:, 0] == body[0])
                * (head_rel_edges[:, 2] == body[-1])
                * (head_rel_edges[:, 3] > body[-2])
            )

            if True in mask:   # 这里似乎可以使用 rule_support += mask.sum()
                rule_support += 1

        return rule_support

    def update_rules_dict(self, rule):
        """
        Update the rules if a new rule has been found.

        Parameters:
            rule (dict): generated rule from self.create_rule

        Returns:
            None
        """

        try:
            self.rules_dict[rule["head_rel"]].append(rule)
        except KeyError:
            self.rules_dict[rule["head_rel"]] = [rule]

    def sort_rules_dict(self):
        """
        Sort the found rules for each head relation by decreasing confidence.

        Parameters:
            None

        Returns:
            None
        """

        for rel in self.rules_dict:
            self.rules_dict[rel] = sorted(
                self.rules_dict[rel], key=lambda x: x["conf"], reverse=True
            )

    def save_rules(self, dt, rule_lengths, num_walks, transition_distr, seed):
        """
        Save all rules.

        Parameters:
            dt (str): time now
            rule_lengths (list): rule lengths
            num_walks (int): number of walks
            transition_distr (str): transition distribution
            seed (int): random seed

        Returns:
            None
        """

        rules_dict = {int(k): v for k, v in self.rules_dict.items()}
        filename = "{0}_r{1}_n{2}_{3}_s{4}_rules.json".format(
            dt, rule_lengths, num_walks, transition_distr, seed
        )
        filename = filename.replace(" ", "")
        with open(self.output_dir + filename, "w", encoding="utf-8") as fout:
            json.dump(rules_dict, fout)

    def save_rules_verbalized(
        self, dt, rule_lengths, num_walks, transition_distr, seed
    ):
        """
        Save all rules in a human-readable format.

        Parameters:
            dt (str): time now
            rule_lengths (list): rule lengths
            num_walks (int): number of walks
            transition_distr (str): transition distribution
            seed (int): random seed

        Returns:
            None
        """

        rules_str = ""
        for rel in self.rules_dict:
            for rule in self.rules_dict[rel]:
                rules_str += verbalize_rule(rule, self.id2relation) + "\n"

        filename = "{0}_r{1}_n{2}_{3}_s{4}_rules.txt".format(
            dt, rule_lengths, num_walks, transition_distr, seed
        )
        filename = filename.replace(" ", "")
        with open(self.output_dir + filename, "w", encoding="utf-8") as fout:
            fout.write(rules_str)


def verbalize_rule(rule, id2relation):
    """
    Verbalize the rule to be in a human-readable format.

    Parameters:
        rule (dict): rule from Rule_Learner.create_rule
        id2relation (dict): mapping of index to relation

    Returns:
        rule_str (str): human-readable rule
    """

    if rule["var_constraints"]:
        var_constraints = rule["var_constraints"]
        constraints = [x for sublist in var_constraints for x in sublist]
        for i in range(len(rule["body_rels"]) + 1):
            if i not in constraints:
                var_constraints.append([i])
        var_constraints = sorted(var_constraints)
    else:
        var_constraints = [[x] for x in range(len(rule["body_rels"]) + 1)]

    rule_str = "{0:8.6f}  {1:4}  {2:4}  {3}(X0,X{4},T{5}) <- "
    obj_idx = [
        idx
        for idx in range(len(var_constraints))
        if len(rule["body_rels"]) in var_constraints[idx]
    ][0]
    rule_str = rule_str.format(
        rule["conf"],
        rule["rule_supp"],
        rule["body_supp"],
        id2relation[rule["head_rel"]],
        obj_idx,
        len(rule["body_rels"]),
    )

    for i in range(len(rule["body_rels"])):
        sub_idx = [
            idx for idx in range(len(var_constraints)) if i in var_constraints[idx]
        ][0]
        obj_idx = [
            idx for idx in range(len(var_constraints)) if i + 1 in var_constraints[idx]
        ][0]
        rule_str += "{0}(X{1},X{2},T{3}), ".format(
            id2relation[rule["body_rels"][i]], sub_idx, obj_idx, i
        )

    return rule_str[:-2]


def rules_statistics(rules_dict):
    """
    Show statistics of the rules.
    只是输出 rules_dict 的统计信息，没有什么实际的功能

    Parameters:
        rules_dict (dict): rules

    Returns:
        None
    """

    print(
        "Number of relations with rules: ", len(rules_dict)
    )  # Including inverse relations
    print("Total number of rules: ", sum([len(v) for k, v in rules_dict.items()]))

    lengths = []
    for rel in rules_dict:
        lengths += [len(x["body_rels"]) for x in rules_dict[rel]]   # + 就是把 list 直接拼接
    rule_lengths = [(k, v) for k, v in Counter(lengths).items()]  # Counter 的作用是对 lengths 里面的相同元素计数，即 lengths 里各个数字及其出现次数。
    # (key, key_count)，按 key 进行排序后输出
    print("Number of rules by length: ", sorted(rule_lengths))
