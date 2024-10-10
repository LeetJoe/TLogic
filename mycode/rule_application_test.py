import json
import numpy as np
import pandas as pd
from collections import Counter

from temporal_walk import store_edges


def filter_rules(rules_dict, min_conf, min_body_supp, rule_lengths):
    """
    Filter for rules with a minimum confidence, minimum body support, and
    specified rule lengths.

    Parameters.
        rules_dict (dict): rules
        min_conf (float): minimum confidence value
        min_body_supp (int): minimum body support value
        rule_lengths (list): rule lengths

    Returns:
        new_rules_dict (dict): filtered rules
    """

    new_rules_dict = dict()
    for k in rules_dict:
        new_rules_dict[k] = []
        for rule in rules_dict[k]:
            cond = (
                (rule["conf"] >= min_conf)
                and (rule["body_supp"] >= min_body_supp)
                and (len(rule["body_rels"]) in rule_lengths)
            )
            if cond:
                new_rules_dict[k].append(rule)

    return new_rules_dict


def get_window_edges(all_data, test_query_ts, learn_edges, window=-1):
    """
    Get the edges in the data (for rule application) that occur in the specified time window.
    If window is 0, all edges before the test query timestamp are included.
    If window is -1, the edges on which the rules are learned are used.
    If window is an integer n > 0, all edges within n timestamps before the test query
    timestamp are included.

    Parameters:
        all_data (np.ndarray): complete dataset (train/valid/test)
        test_query_ts (np.ndarray): test query timestamp
        learn_edges (dict): edges on which the rules are learned
        window (int): time window used for rule application

    Returns:
        window_edges (dict): edges in the window for rule application
    """

    if window > 0:
        mask = (all_data[:, 3] < test_query_ts) * (
            all_data[:, 3] >= test_query_ts - window
        )
        window_edges = store_edges(all_data[mask])
    elif window == 0:
        mask = all_data[:, 3] < test_query_ts
        window_edges = store_edges(all_data[mask])
    elif window == -1:
        window_edges = learn_edges

    return window_edges


def match_body_relations(rule, edges, test_query_sub):
    """
    这个函数的功能是，从 edges 里找到以 test_query_sub 开头并与 rule 对应的路径；
    如 [[[1,2,10],[1,3,12]],[[3,5,16]] 表示的含义是，rule 的 body relations 有
    两条 rel，以 1 开头，按 rel[0] 能找到 [1,2,10],[1,3,12], 其含义为
    [head, tail ,ts]；再按 rel[1] 能找到 [3,5,16]；显然 [1,3,12] 和 [3,5,16]
    构成了一条路径，而 [1,2,10] 没能找到符合 rel[1] 的后继；若子表为空表，则表示没有找到
    对应的路径。

    Find edges that could constitute walks (starting from the test query subject)
    that match the rule.
    First, find edges whose subject match the query subject and the relation matches
    the first relation in the rule body. Then, find edges whose subjects match the
    current targets and the relation the next relation in the rule body.
    Memory-efficient implementation.

    Parameters:
        rule (dict): rule from rules_dict
        edges (dict): edges for rule application
        test_query_sub (int): test query subject, head node。

    Returns:
        walk_edges (list of np.ndarrays): edges that could constitute rule walks
    """

    rels = rule["body_rels"]
    # Match query subject and first body relation
    try:
        rel_edges = edges[rels[0]]  # 与目标 relation 匹配的那些 edges
        mask = rel_edges[:, 0] == test_query_sub   # 找到那些 head 与 subject 相同的 edges 的位置
        new_edges = rel_edges[mask]  # new_edges 现在是那些以 test_query_sub 开头且 r 为 rule["body_rels"] 里第一项的那些 edges
        walk_edges = [
            np.hstack((new_edges[:, 0:1], new_edges[:, 2:4]))
        ]  # list of [sub, obj, ts]
        cur_targets = np.array(list(set(walk_edges[0][:, 1])))  # cur_targets 是 walk_edges 里的 obj unique list(cands)

        for i in range(1, len(rels)):
            # Match current targets and next body relation
            try:
                rel_edges = edges[rels[i]]

                # rel_edges[:, 0] 是一个一维数组；cur_targets 也是一个一维数组；但是
                # cur_targets[:, None] 会给 cur_targets 增加一维，使其中每一个元组
                # 自成一个 list。这样再进行 rel_edges[:, 0] == cur_targets[:, None]
                # 的时候，得到的是一个二维数组；其中行长度与 rel_edges[:, 0] 相同，
                # i 行中每一个位置若有 rel_edges[:, 0] 与 cur_targets[:, None][i][0]
                # 相等，则为 True；否则为 false。假设 rel_edges[:, 0]=[1,2,3], cur_targets
                # = [1,2,3], 则 cur_targets[:, None] = [[1],[2],[3]]，rel_edges[:, 0]
                # == cur_targets[:, None] 是一个 3x3 的矩阵，只有对角元素为 True，其它为 False

                # np.any(np.array(), axis=0)) 表示第一个参数以第 0 维切分，切分得到的 list
                # 中只要有一个为 True，就返回 True，最终得到一个 list，长度与第一个参数在 0 维上的长度相同。
                mask = np.any(rel_edges[:, 0] == cur_targets[:, None], axis=0)
                new_edges = rel_edges[mask]  # 新的 edges，即经过上一步骤可以到达的 edges（连通到 head）
                walk_edges.append(
                    np.hstack((new_edges[:, 0:1], new_edges[:, 2:4]))
                )  # [sub, obj, ts]
                cur_targets = np.array(list(set(walk_edges[i][:, 1])))
            except KeyError:  # 这是个处理问题的好方法，只要 keyerror 就置空，免去了处理 key 异常的各种情况
                walk_edges.append([])
                break
    except KeyError:
        walk_edges = [[]]

    # walk_edges 可能全为空 [[]]，也可能部分为空 [[1,2],[]]; 存在空子列表意味着 walk 失败了：没有找到路径
    return walk_edges


# 在上面的方法的基础上，进一步约束每步 rel 之间必须能连接起来形成一条完整的路径。
# todo 但是这个方法实际上没用？？
def match_body_relations_complete(rule, edges, test_query_sub):
    """
    Find edges that could constitute walks (starting from the test query subject)
    that match the rule.
    First, find edges whose subject match the query subject and the relation matches
    the first relation in the rule body. Then, find edges whose subjects match the
    current targets and the relation the next relation in the rule body.

    Parameters:
        rule (dict): rule from rules_dict
        edges (dict): edges for rule application
        test_query_sub (int): test query subject

    Returns:
        walk_edges (list of np.ndarrays): edges that could constitute rule walks
    """

    rels = rule["body_rels"]
    # Match query subject and first body relation
    try:
        rel_edges = edges[rels[0]]
        mask = rel_edges[:, 0] == test_query_sub
        new_edges = rel_edges[mask]
        walk_edges = [new_edges]
        cur_targets = np.array(list(set(walk_edges[0][:, 2])))

        for i in range(1, len(rels)):
            # Match current targets and next body relation
            try:
                rel_edges = edges[rels[i]]
                # 下一步的 head 必须与当前的 target 相等，即能连接得起来。
                mask = np.any(rel_edges[:, 0] == cur_targets[:, None], axis=0)
                new_edges = rel_edges[mask]
                walk_edges.append(new_edges)
                cur_targets = np.array(list(set(walk_edges[i][:, 2])))
            except KeyError:
                walk_edges.append([])
                break
    except KeyError:
        walk_edges = [[]]

    return walk_edges


# 就是把上面的方法得到的 walk_edges 组成可以接通的直接路径，类似我实现的 get_ht_hop 方法。
def get_walks(rule, walk_edges):
    """
    Get walks for a given rule. Take the time constraints into account.
    Memory-efficient implementation.

    Parameters:
        rule (dict): rule from rules_dict
        walk_edges (list of np.ndarrays): edges from match_body_relations

    Returns:
        rule_walks (pd.DataFrame): all walks matching the rule
    """

    df_edges = []
    df = pd.DataFrame(
        walk_edges[0],
        columns=["entity_" + str(0), "entity_" + str(1), "timestamp_" + str(0)],
        dtype=np.uint16,
    )  # Change type if necessary for better memory efficiency
    if not rule["var_constraints"]:  # todo ??
        del df["entity_" + str(0)]
    df_edges.append(df)
    df = df[0:0]  # Memory efficiency

    for i in range(1, len(walk_edges)):
        df = pd.DataFrame(
            walk_edges[i],
            columns=["entity_" + str(i), "entity_" + str(i + 1), "timestamp_" + str(i)],
            dtype=np.uint16,
        )  # Change type if necessary
        df_edges.append(df)
        df = df[0:0]

    rule_walks = df_edges[0]
    df_edges[0] = df_edges[0][0:0]
    for i in range(1, len(df_edges)):
        rule_walks = pd.merge(rule_walks, df_edges[i], on=["entity_" + str(i)])
        rule_walks = rule_walks[
            rule_walks["timestamp_" + str(i - 1)] <= rule_walks["timestamp_" + str(i)]
        ]
        if not rule["var_constraints"]:
            del rule_walks["entity_" + str(i)]
        df_edges[i] = df_edges[i][0:0]

    for i in range(1, len(rule["body_rels"])):
        del rule_walks["timestamp_" + str(i)]

    return rule_walks


# 这个方法跟前面那个 match_body_relations_complete 方法一样，没有被使用过
def get_walks_complete(rule, walk_edges):
    """
    Get complete walks for a given rule. Take the time constraints into account.

    Parameters:
        rule (dict): rule from rules_dict
        walk_edges (list of np.ndarrays): edges from match_body_relations

    Returns:
        rule_walks (pd.DataFrame): all walks matching the rule
    """

    df_edges = []
    df = pd.DataFrame(
        walk_edges[0],
        columns=[
            "entity_" + str(0),
            "relation_" + str(0),
            "entity_" + str(1),
            "timestamp_" + str(0),
        ],
        dtype=np.uint16,
    )  # Change type if necessary for better memory efficiency
    df_edges.append(df)

    for i in range(1, len(walk_edges)):
        df = pd.DataFrame(
            walk_edges[i],
            columns=[
                "entity_" + str(i),
                "relation_" + str(i),
                "entity_" + str(i + 1),
                "timestamp_" + str(i),
            ],
            dtype=np.uint16,
        )  # Change type if necessary
        df_edges.append(df)

    rule_walks = df_edges[0]
    for i in range(1, len(df_edges)):
        rule_walks = pd.merge(rule_walks, df_edges[i], on=["entity_" + str(i)])
        rule_walks = rule_walks[
            rule_walks["timestamp_" + str(i - 1)] <= rule_walks["timestamp_" + str(i)]
        ]

    return rule_walks


# todo 这个 var_constraints 一直没明白作用是什么，之前看数据的时候，发现它应该是表示在规则路径中存在重复出现的实体的位置。
# todo 在查找规则路径的时候，除了上面的“首尾相接”式筛选，这里还有一个使用 var_constraints 的筛选。
# todo 是否意味着，对于某条规则，即便找到一条符合此规则的路径，也并不意味着这条路径有效，同时必须满足规则中 var_constraints 限定的约束才可以？
def check_var_constraints(var_constraints, rule_walks):
    """
    Check variable constraints of the rule.

    Parameters:
        var_constraints (list): variable constraints from the rule
        rule_walks (pd.DataFrame): all walks matching the rule

    Returns:
        rule_walks (pd.DataFrame): all walks matching the rule including the variable constraints
    """

    for const in var_constraints:
        for i in range(len(const) - 1):
            rule_walks = rule_walks[
                rule_walks["entity_" + str(const[i])]
                == rule_walks["entity_" + str(const[i + 1])]
            ]

    return rule_walks


def get_candidates(
    rule, rule_walks, test_query_ts, cands_dict, score_func, args, dicts_idx
):
    """
    Get from the walks that follow the rule the answer candidates.
    Add the confidence of the rule that leads to these candidates.

    Parameters:
        rule (dict): rule from rules_dict
        rule_walks (pd.DataFrame): rule walks (satisfying all constraints from the rule)
        test_query_ts (int): test query timestamp
        cands_dict (dict): candidates along with the confidences of the rules that generated these candidates
        score_func (function): function for calculating the candidate score
        args (list): arguments for the scoring function
        dicts_idx (list): indices for candidate dictionaries

    Returns:
        cands_dict (dict): updated candidates
    """

    # 取所有可能路径的最后一列实体作为候选，不必遍历所有
    max_entity = "entity_" + str(len(rule["body_rels"]))
    cands = set(rule_walks[max_entity])  # 去重

    for cand in cands:
        cands_walks = rule_walks[rule_walks[max_entity] == cand]
        for s in dicts_idx:
            score = score_func(rule, cands_walks, test_query_ts, *args[s]).astype(
                np.float32
            )
            try:
                cands_dict[s][cand].append(score)
            except KeyError:
                cands_dict[s][cand] = [score]

    return cands_dict


def save_candidates(
    rules_file, dir_path, all_candidates, rule_lengths, window, score_func_str
):
    """
    Save the candidates.

    Parameters:
        rules_file (str): name of rules file
        dir_path (str): path to output directory
        all_candidates (dict): candidates for all test queries
        rule_lengths (list): rule lengths
        window (int): time window used for rule application
        score_func_str (str): scoring function, 打分函数, 后面拼上了 lambda 和 coeff

    Returns:
        None
    """

    # 对 all_candidates 进行 reformat, 确保两层字典的 key 都是 int 类型.
    all_candidates = {int(k): v for k, v in all_candidates.items()}
    for k in all_candidates:
        all_candidates[k] = {int(cand): v for cand, v in all_candidates[k].items()}
    filename = "{0}_cands_r{1}_w{2}_{3}.json".format(
        rules_file[:-11], rule_lengths, window, score_func_str
    )
    filename = filename.replace(" ", "")
    with open(dir_path + filename, "w", encoding="utf-8") as fout:
        json.dump(all_candidates, fout)


def verbalize_walk(walk, data):
    """
    Verbalize walk from rule application.

    Parameters:
        walk (pandas.core.series.Series): walk that matches the rule body from get_walks
        data (grapher.Grapher): graph data

    Returns:
        walk_str (str): verbalized walk
    """

    l = len(walk) // 3
    walk = walk.values.tolist()

    walk_str = data.id2entity[walk[0]] + "\t"
    for j in range(l):
        walk_str += data.id2relation[walk[3 * j + 1]] + "\t"
        walk_str += data.id2entity[walk[3 * j + 2]] + "\t"
        walk_str += data.id2ts[walk[3 * j + 3]] + "\t"

    return walk_str[:-1]
