import time
import argparse
import numpy as np
from datetime import datetime
from joblib import Parallel, delayed

from grapher_test import Grapher
from temporal_walk_test import Temporal_Walk
from rule_learning_test import Rule_Learner, rules_statistics


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", default="", type=str)
parser.add_argument("--rule_lengths", "-l", default="3", type=int, nargs="+")
parser.add_argument("--num_walks", "-n", default="100", type=int)
parser.add_argument("--transition_distr", default="exp", type=str)
parser.add_argument("--num_processes", "-p", default=1, type=int)
parser.add_argument("--seed", "-s", default=None, type=int)
parsed = vars(parser.parse_args())

dataset = parsed["dataset"]
rule_lengths = parsed["rule_lengths"]
rule_lengths = [rule_lengths] if (type(rule_lengths) == int) else rule_lengths
num_walks = parsed["num_walks"]
transition_distr = parsed["transition_distr"]
num_processes = parsed["num_processes"]
seed = parsed["seed"]

dataset_dir = "../data/" + dataset + "/"
data = Grapher(dataset_dir)
temporal_walk = Temporal_Walk(data.train_idx, data.inv_relation_id, transition_distr)
rl = Rule_Learner(temporal_walk.edges, data.id2relation, data.inv_relation_id, dataset)
all_relations = sorted(temporal_walk.edges)  # Learn for all relations


def learn_rules(i, num_relations):
    """
    Learn rules (multiprocessing possible).

    Parameters:
        i (int): process number
        num_relations (int): minimum number of relations for each process

    Returns:
        rl.rules_dict (dict): rules dictionary
    """

    if seed:
        np.random.seed(seed)

    # 任务分界
    num_rest_relations = len(all_relations) - (i + 1) * num_relations
    if num_rest_relations >= num_relations:
        relations_idx = range(i * num_relations, (i + 1) * num_relations)
    else:
        relations_idx = range(i * num_relations, len(all_relations))

    num_rules = [0]
    for k in relations_idx:
        rel = all_relations[k]
        for length in rule_lengths:   # 参数
            it_start = time.time()
            for _ in range(num_walks):   # 参数
                walk_successful, walk = temporal_walk.sample_walk(length + 1, rel)
                # 当 length=1 的时候，通过两跳得到一个环，那么第二跳很可能是第一跳的逆。
                if walk_successful:
                    rl.create_rule(walk)   # walk 是一条路径，路径长度是 length+1，可能(一定？)是一个环。
            it_end = time.time()
            it_time = round(it_end - it_start, 6)
            num_rules.append(sum([len(v) for k, v in rl.rules_dict.items()]) // 2)
            num_new_rules = num_rules[-1] - num_rules[-2]
            print(
                "Process {0}: relation {1}/{2}, length {3}: {4} sec, {5} rules".format(
                    i,
                    k - relations_idx[0] + 1,
                    len(relations_idx),
                    length,
                    it_time,
                    num_new_rules,
                )
            )

    return rl.rules_dict


start = time.time()
# 向下取整，但余数代表的数据并未丢掉，在 learn_rules 方法里相应处理。
num_relations = len(all_relations) // num_processes
# 按 relation 进行任务划分，todo 有必要学习一下，可以并行执行任务的简单实现方法。
# 这里的 output 的第一层是一个 list，大小与 num_precesses 相同，其中的内容是各个子进程的返回，结构为一个 dict，keys 是对应进程处理的 relations
# 的 id; 对应的 values 是一个 list，里面是与 key 对应的 rules，每个 rule 有 rule.head_rel = key。
output = Parallel(n_jobs=num_processes)(
    delayed(learn_rules)(i, num_relations) for i in range(num_processes)
)
end = time.time()

all_rules = output[0]
for i in range(1, num_processes):
    all_rules.update(output[i])  # dict.update() 的作用是把原来存在的key内容更新，没有的key就添加。这里对 update 来说就是去掉第一层 list 进行合并。
# 所以 all_rules 就是把子任务返回的结果合并在一起了。

total_time = round(end - start, 6)
print("Learning finished in {} seconds.".format(total_time))

rl.rules_dict = all_rules
rl.sort_rules_dict()   # 对每个 head 按照概率(confidence)降序排列。
dt = datetime.now()
dt = dt.strftime("%d%m%y%H%M%S")
rl.save_rules(dt, rule_lengths, num_walks, transition_distr, seed)   # 将 rules_dict 保存到文件里，其它参数主要是用命名保存文件的。
rl.save_rules_verbalized(dt, rule_lengths, num_walks, transition_distr, seed)  # 与上面类似，不过这种是以一种 human readable 的方式存在 txt 里面。
rules_statistics(rl.rules_dict)
