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

        if rule not in self.found_rules:  # todo 所以 in 可以实现对复杂对象按内容进行比较?
            self.found_rules.append(rule.copy())   # 将 rule 记录下来
            (
                rule["conf"],
                rule["rule_supp"],
                rule["body_supp"],
            ) = self.estimate_confidence(rule)   # 对 rule 进行初步验证，评价其有效性，得到 confidence, rule support, body support

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
                rule["body_rels"], rule["var_constraints"] # 传参是除去 head_rel 的其它 rels(倒序) 和除去第一个结点的其它 entities(倒序)
            )
            if sample_successful:
                # 如果成功, body_ents_tss 是一个[ent, ts,..., ent] 序列
                all_bodies.append(body_ents_tss)

        all_bodies.sort()  # 似乎是使用每一子 list 的第一个元素的值来升序排列的
        # todo 这里的 itertools.groupby() 似乎是用来排除内容完全相同的 body_ents_tss 的.
        #  in 不是也有这个功能吗, 在前面 append 的时候用 in 检查一下不就可以了?
        unique_bodies = list(x for x, _ in itertools.groupby(all_bodies))   # 除去完全一样的元素
        body_support = len(unique_bodies) # 所以 body support 是指在 num_samples 次的 sample_body 中, 得到符合条件的 body_ents_tss 的次数

        confidence, rule_support = 0, 0
        if body_support:
            # 所以 rule_support 是指前面找到的所有 body_ents_tss 中, 可以通过"完整的" rule 规则验证的, 而 body_ents_tss 的查找是不带 head_rel 的, 是"不完整"的
            rule_support = self.calculate_rule_support(unique_bodies, rule["head_rel"])
            confidence = round(rule_support / body_support, 6)

        return confidence, rule_support, body_support

    def sample_body(self, body_rels, var_constraints):
        """
        Sample a walk according to the rule body.
        The sequence of timesteps should be non-decreasing.

        按 body_rels 提供关系序列，从body_rels[0] 开始随机选择一个边为开端，构建一个新的路径，而且要求这条新的路径里的重复节点的位置与数量
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
        # body_ents_tss 的结构为 [node0, ts1, node1, ts2, node2, ..., ts_n, node_n]
        body_ents_tss = []
        cur_rel = body_rels[0]
        rel_edges = self.edges[cur_rel] # rel_edges 是所有 rel 为 cur_rel 的四元组
        next_edge = rel_edges[np.random.choice(len(rel_edges))] # 随机选一个四元组
        cur_ts = next_edge[3]  # 时间
        cur_node = next_edge[2]  # tail
        body_ents_tss.append(next_edge[0])  # body_ents_tss 以 next_edge 的 head 为开端, 形成一个 head->ts->tail->ts...序列
        body_ents_tss.append(cur_ts)
        body_ents_tss.append(cur_node)

        for cur_rel in body_rels[1:]:
            next_edges = self.edges[cur_rel]
            # 在 next_edges 里 head 与 cur_node 相等且 ts 不小于 cur_ts 的位置里为 true
            mask = (next_edges[:, 0] == cur_node) * (next_edges[:, 3] >= cur_ts)
            filtered_edges = next_edges[mask]

            if len(filtered_edges): # 按照 mask 找到了符合条件的 edges, 从中随机选一个加在后面继续遍历 body_rels
                next_edge = filtered_edges[np.random.choice(len(filtered_edges))]
                cur_ts = next_edge[3]
                cur_node = next_edge[2]
                body_ents_tss.append(cur_ts)
                body_ents_tss.append(cur_node)
            else:
                sample_successful = False
                break

        if sample_successful and var_constraints:
            # Check variable constraints
            # todo define_var_constraints 要对 body_ents_tss[::2] 进行去重的排序, 然后与 var_constraints 比较.
            #  然而 var_constraints 似乎没有做过排序, 而且 body_ents_tss 长度应该跟 var_constraints 以及 body_rels 长度都是相同的, 执行去重有什么意义?
            body_var_constraints = self.define_var_constraints(body_ents_tss[::2])   # body_ents_tss[::2] 即取所有的 nodes
            # todo 如果取样的实体里重复出现的实体的位置与数量不相等，则认为失败. 这样做的前提是 body_ents_tss 在不同的 ts 组合下有很多合适的的候选
            if body_var_constraints != var_constraints:
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

            if True in mask:
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
