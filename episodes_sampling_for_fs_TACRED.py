import json
import random
import os, sys, inspect, argparse
from numpy.random import choice
import numpy as np
from collections import OrderedDict


def remove_relations_with_too_few_instances(data,K):
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


def TACRED_create_episode(all_data, weights_all_relation, query_weights, uniform_dist_drop_no_relation, N, K,
                          number_of_queries, sample_uni):
    if sample_uni:
        # uniform sampling but remove no_relation
        sampled_relation = choice(a=[*all_data], size=N, replace=False, p=uniform_dist_drop_no_relation).tolist()
    else:
        sampled_relation = choice(a=[*all_data], size=N, replace=False, p=query_weights).tolist()
    meta_train = [random.sample(all_data[i], K) for i in sampled_relation]

    meta_test_list = []
    target_list = []
    list_relations = []
    if args.set_a_positive_instance_per_episode:
        targets_relations = ["no_relation"] * (number_of_queries - 1)
        positive_example = [random.choice(sampled_relation)]
        targets_relations = targets_relations + positive_example
    else:
        targets_relations = choice(a=[*all_data], size=number_of_queries, replace=True, p=weights_all_relation).tolist()
    for t in targets_relations:
        if t in sampled_relation:
            correct_target = sampled_relation.index(t)
            instance_in_ss = meta_train[correct_target]
            temp = [x for x in all_data[t] if x not in instance_in_ss]
            single_query = random.choice(temp)
        else:
            correct_target = N
            single_query = random.choice(all_data[t])

        assert type(single_query) is dict
        assert type(correct_target) is int
        meta_test_list.append(single_query)
        target_list.append(correct_target)
        list_relations.append(t)

    return {"meta_train": meta_train, "meta_test": meta_test_list}, target_list, [list_relations,sampled_relation]


def get_weights(all_data):
    weights_all_relation = [len(all_data[r]) for r in all_data]
    weights_all_relation = sum(weights_all_relation)
    return [len(all_data[r]) / weights_all_relation for r in all_data]


def get_query_weights(all_data,w):
    z = w[:]
    no_relation_index = list(all_data.keys()).index("no_relation")
    z[no_relation_index] = 0.0
    z = normalized_weights(z)
    return z


def main():
    random.seed(args.seed)
    np.random.seed(args.seed)

    whole_division = json.load(open(args.file_name))
    sample_uniform = args.sample_uniform
    assert "no_relation" in whole_division

    if args.set_a_positive_instance_per_episode:
        print("WARNING!!, CHANGING POSITIVE LABEL DISTRIBUTION")

    whole_division = remove_relations_with_too_few_instances(whole_division,args.K)

    weights_all_relation = get_weights(whole_division)
    query_weights = get_query_weights(whole_division,weights_all_relation)
    uniform_dist_drop_no_relation = [1 / (len(query_weights) - 1) if i > 0.0 else 0.0 for i in query_weights ]
    create_episodes(whole_division, weights_all_relation, query_weights, uniform_dist_drop_no_relation, sample_uniform)


def create_episodes(whole_division, weights_all_relation, query_weights, uniform_dist_drop_no_relation, do_sample_uniform):
    episodes = []
    targets_lists = []
    aux_data = []
    for i in range(args.episodes_size):
        episode, targets, [targets_relation_names, N_relations] = TACRED_create_episode(whole_division, weights_all_relation, query_weights,
                                                    uniform_dist_drop_no_relation, args.N, args.K,
                                                    args.number_of_queries, do_sample_uniform)
        targets_lists.append(targets)
        episodes.append(episode)
        aux_data.append((N_relations,targets_relation_names))
    final_data = [episodes, targets_lists,aux_data]
    json.dump(final_data, open(args.output_file_name, "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, required=True)
    parser.add_argument("--episodes_size", default=5000, type=int, required=False,
                        help="The number of episodes to create")
    parser.add_argument("--N", type=int, required=True,help="How many ways in each episode")
    parser.add_argument("--K", type=int, required=True,
                        help="How many instances represent each class")
    parser.add_argument("--number_of_queries", default=1, type=int, required=False)
    parser.add_argument("--seed", type=int, required=True,
                        help="seed number")
    parser.add_argument("--output_file_name", type=str, required=True,
                        help="The file name to be generated")

    parser.add_argument("--set_a_positive_instance_per_episode", action="store_true")

    parser.add_argument("--sample_uniform", action="store_false")

    global args
    args = parser.parse_args()

    main()
