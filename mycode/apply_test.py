import json
import time
import argparse
import itertools
import numpy as np
from joblib import Parallel, delayed

import rule_application_test as ra
from grapher_test import Grapher
from temporal_walk_test import store_edges
from rule_learning_test import rules_statistics
from score_functions_test import score_12


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", default="", type=str)
parser.add_argument("--test_data", default="test", type=str)
parser.add_argument("--rules", "-r", default="", type=str)
parser.add_argument("--rule_lengths", "-l", default=1, type=int, nargs="+")
parser.add_argument("--window", "-w", default=-1, type=int)
parser.add_argument("--top_k", default=20, type=int)
parser.add_argument("--num_processes", "-p", default=1, type=int)
parsed = vars(parser.parse_args())

dataset = parsed["dataset"]
rules_file = parsed["rules"]
window = parsed["window"]
top_k = parsed["top_k"]
num_processes = parsed["num_processes"]
rule_lengths = parsed["rule_lengths"]
rule_lengths = [rule_lengths] if (type(rule_lengths) == int) else rule_lengths

dataset_dir = "../data/" + dataset + "/"
dir_path = "../output/" + dataset + "/"
data = Grapher(dataset_dir)
# 仓库里给的 run.txt 里的命令参数示例里没有修改 test_data 参数的情况, 都是默认值 test, 所以这里都是使用的 test_idx
# todo test_data 是 apply 过程的主要目标数据
test_data = data.test_idx if (parsed["test_data"] == "test") else data.valid_idx
rules_dict = json.load(open(dir_path + rules_file))
rules_dict = {int(k): v for k, v in rules_dict.items()} # 再次整理字典格式, 保证 key 都是 int 类型
print("Rules statistics:")
rules_statistics(rules_dict)

# 选出 confidence 不低于0.01，body_supply 不低于 2, len(rule["body_rels"]) in rule_lengths 的那些 rules
rules_dict = ra.filter_rules(
    rules_dict, min_conf=0.01, min_body_supp=2, rule_lengths=rule_lengths
)
print("Rules statistics after pruning:")
rules_statistics(rules_dict)
learn_edges = store_edges(data.train_idx)

score_func = score_12  # score function 有三个不同的实现
# It is possible to specify a list of list of arguments for tuning
args = [[0.1, 0.5]]  # args 是一组参数，表示 [lambda, coeff]，具体见 score_functions.score_12() 里的 lambda 和 a。


def apply_rules(i, num_queries):
    """
    Apply rules (multiprocessing possible).

    Parameters:
        i (int): process number
        num_queries (int): minimum number of queries for each process, 实际上相当于 batch size

    Returns:
        all_candidates (list): answer candidates with corresponding confidence scores
        no_cands_counter (int): number of queries with no answer candidates
    """

    print("Start process", i, "...")
    all_candidates = [dict() for _ in range(len(args))]  # 占位初始化
    no_cands_counter = 0

    num_rest_queries = len(test_data) - (i + 1) * num_queries
    if num_rest_queries >= num_queries:
        test_queries_idx = range(i * num_queries, (i + 1) * num_queries)
    else:
        test_queries_idx = range(i * num_queries, len(test_data))

    cur_ts = test_data[test_queries_idx[0]][3]  # 当前分组第一项条目的 ts
    edges = ra.get_window_edges(data.all_idx, cur_ts, learn_edges, window) # window 传 0 表示取 cur_ts 之前的所有数据

    it_start = time.time()
    for j in test_queries_idx:
        test_query = test_data[j]  # 取一条数据
        cands_dict = [dict() for _ in range(len(args))]

        if test_query[3] != cur_ts:  # 如果时间发生了变化，窗口数据也需要相应调整
            cur_ts = test_query[3]
            # data.all_idx, train+valid+test data. 若 window>0 返回 all_idx 里ts在 [cur_ts-window, cur_ts] 的数据；
            # 若 window=0，则返回 all_idx 里 ts 在[0, cur_ts] 里的数据；或 window<0，则直接返回 learn_edges.
            edges = ra.get_window_edges(data.all_idx, cur_ts, learn_edges, window)

        if test_query[1] in rules_dict:   # test_query[1] 是4元组的 relation，如果 relation 有 rule 存在(注意这里是 if 语句不是 for 语句)
            dicts_idx = list(range(len(args)))   # args 可以多给几对参数，然后评估不同的组合的评价效果
            for rule in rules_dict[test_query[1]]:  # test_query[1] 是 relation, 按 relation 从 rules_dict 中找到相关的 rule 进行遍历
                walk_edges = ra.match_body_relations(rule, edges, test_query[0]) # test_query[0] 是 head，尝试使用 rule 要找 walk

                if 0 not in [len(x) for x in walk_edges]:
                    rule_walks = ra.get_walks(rule, walk_edges)    # rule_walks 是一个 Numpy.DataFrame, 类似 excel 的二维数据组织结构。
                    if rule["var_constraints"]:
                        rule_walks = ra.check_var_constraints(
                            rule["var_constraints"], rule_walks
                        )

                    if not rule_walks.empty:
                        cands_dict = ra.get_candidates(   # 使用一个打分函数选择出 top_k 的候选
                            rule,
                            rule_walks,
                            cur_ts,
                            cands_dict,
                            score_func,
                            args,
                            dicts_idx,
                        )
                        for s in dicts_idx:
                            cands_dict[s] = {
                                x: sorted(cands_dict[s][x], reverse=True)
                                for x in cands_dict[s].keys()
                            }
                            cands_dict[s] = dict(
                                sorted(
                                    cands_dict[s].items(),
                                    key=lambda item: item[1],
                                    reverse=True,
                                )
                            )
                            top_k_scores = [v for _, v in cands_dict[s].items()][:top_k]
                            unique_scores = list(
                                scores for scores, _ in itertools.groupby(top_k_scores)
                            )
                            if len(unique_scores) >= top_k:
                                dicts_idx.remove(s)
                        if not dicts_idx:
                            break

            if cands_dict[0]:
                for s in range(len(args)):
                    # Calculate noisy-or scores
                    scores = list(
                        map(
                            lambda x: 1 - np.product(1 - np.array(x)),
                            cands_dict[s].values(),
                        )
                    )
                    cands_scores = dict(zip(cands_dict[s].keys(), scores))
                    noisy_or_cands = dict(
                        sorted(cands_scores.items(), key=lambda x: x[1], reverse=True)
                    )
                    all_candidates[s][j] = noisy_or_cands
            else:  # No candidates found by applying rules
                no_cands_counter += 1
                for s in range(len(args)):
                    all_candidates[s][j] = dict()

        else:  # No rules exist for this relation
            no_cands_counter += 1
            for s in range(len(args)):
                all_candidates[s][j] = dict()

        if not (j - test_queries_idx[0] + 1) % 100:    # j 每前进 100 轮打印一次
            it_end = time.time()
            it_time = round(it_end - it_start, 6)
            print(
                "Process {0}: test samples finished: {1}/{2}, {3} sec".format(
                    i, j - test_queries_idx[0] + 1, len(test_queries_idx), it_time
                )
            )
            it_start = time.time()

    return all_candidates, no_cands_counter


start = time.time()
num_queries = len(test_data) // num_processes
output = Parallel(n_jobs=num_processes)(
    delayed(apply_rules)(i, num_queries) for i in range(num_processes)
)
end = time.time()

# 合并结果
final_all_candidates = [dict() for _ in range(len(args))]
for s in range(len(args)):
    for i in range(num_processes):
        final_all_candidates[s].update(output[i][0][s])
        output[i][0][s].clear()

final_no_cands_counter = 0
for i in range(num_processes):
    final_no_cands_counter += output[i][1]

# 输出统计
total_time = round(end - start, 6)
print("Application finished in {} seconds.".format(total_time))
print("No candidates: ", final_no_cands_counter, " queries")

# 保存 candidates, args 是 [lambda, coeff] 参数对, 按不同的参数组合对分别保存 apply 结果
for s in range(len(args)):
    score_func_str = score_func.__name__ + str(args[s])
    score_func_str = score_func_str.replace(" ", "")
    ra.save_candidates(
        rules_file,
        dir_path,
        final_all_candidates[s],
        rule_lengths,
        window,
        score_func_str,
    )
