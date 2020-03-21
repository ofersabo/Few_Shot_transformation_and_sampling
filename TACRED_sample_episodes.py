import json
import random
import os, sys, inspect
from numpy.random import choice
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grantparentdir = os.path.dirname(parentdir)
sys.path.insert(0, grantparentdir)

filename = sys.argv[1] if len(sys.argv) > 1 else "1_TACRED_split/test_data.json"
size = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
N = int(sys.argv[3]) if len(sys.argv) > 3 else 5
K = int(sys.argv[4]) if len(sys.argv) > 4 else 1
number_of_queries = int(sys.argv[5]) if len(sys.argv) > 5 else 3
seed = int(sys.argv[6]) if len(sys.argv) > 6 else 123
train_file = sys.argv[7] if len(sys.argv) > 7 else "tmp.json"

random.seed(seed)
np.random.seed(seed)


def remove_relations_with_too_few_instances(data):
    relations_to_remove = []
    for r in data:
        if len(data[r]) <= K:
            # we need to remove this relation type as we don't have enough instances.
            print("removed relation: ", r)
            relations_to_remove.append(r)
    data = {k: v for k, v in data.items() if k not in relations_to_remove}
    return data


def normalized_weights(other_list):
    weights = np.array(other_list)
    weights = weights / np.sum(weights)
    weights = weights.tolist()
    return weights


def TACRED_create_episode(all_data, weights_all_relation, uniform_dist_drop_no_relation, N, K,
                          queries_number):
    # uniform sampling of target relations without no_relation
    sampled_relation = choice(a=[*all_data], size=N, replace=False, p=uniform_dist_drop_no_relation).tolist()
    meta_train = [random.sample(all_data[i], K) for i in sampled_relation]

    meta_test_list = []
    target_list = []
    list_relations = []
    targets_relations = choice(a=[*all_data], size=queries_number, replace=True, p=weights_all_relation).tolist()
    for t in targets_relations:
        if t in sampled_relation:
            correct_target = sampled_relation.index(t)
            instance_in_supportset = meta_train[correct_target]
            temp = [x for x in all_data[t] if x not in instance_in_supportset]
            single_query = random.choice(temp)
        else:
            correct_target = N
            single_query = random.choice(all_data[t])

        assert type(single_query) is dict
        assert type(correct_target) is int
        meta_test_list.append(single_query)
        target_list.append(correct_target)
        list_relations.append(t)

    return {"meta_train": meta_train, "meta_test": meta_test_list}, target_list, [list_relations, sampled_relation]


def get_weights(all_data):
    weights_all_relation = [len(all_data[r]) for r in all_data]
    weights_all_relation = sum(weights_all_relation)
    return [len(all_data[r]) / weights_all_relation for r in all_data]


def get_query_weights(w, relations):
    z = w[:]
    no_relation_index = list(relations).index("no_relation")
    z[no_relation_index] = 0.0
    z = normalized_weights(z)
    return z


def main():
    whole_division = json.load(open(filename))
    assert "no_relation" in whole_division

    # Remove relations with less than K+1 instances
    whole_division = remove_relations_with_too_few_instances(whole_division)
    relations = whole_division.keys()

    weights_all_relation = get_weights(whole_division)
    query_weights = get_query_weights(weights_all_relation, relations)
    uniform_dist_drop_no_relation = [1 / (len(query_weights) - 1) if i > 0.0 else 0.0 for i in query_weights]
    create_episodes(data=whole_division, weights_all_relation=weights_all_relation,
                    uniform_dist_drop_no_relation=uniform_dist_drop_no_relation)


def create_episodes(data, weights_all_relation, uniform_dist_drop_no_relation):
    episodes = []
    targets_lists = []
    aux_data = []
    for i in range(size):
        episode, targets, [targets_relation_names, N_relations] = TACRED_create_episode(all_data=data,
                                                                                        weights_all_relation=weights_all_relation,
                                                                                        uniform_dist_drop_no_relation=uniform_dist_drop_no_relation,
                                                                                        N=N, K=K,
                                                                                        queries_number=number_of_queries)
        targets_lists.append(targets)
        episodes.append(episode)
        aux_data.append((N_relations, targets_relation_names))
    final_data = [episodes, targets_lists, aux_data]
    json.dump(final_data, open(train_file, "w"))


if __name__ == "__main__":
    main()
