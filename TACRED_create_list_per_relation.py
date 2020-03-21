import os
import json
import random

data_location = "raw/"

def get_set_of_relation_type(file):
    global type_relation
    type_relation = set()
    with open(file) as f:
        for line in f:
            type_relation.add(line.strip())
    return type_relation


def replace_to_rounded_parnthesis(list_of_tokens):
    for i, token in enumerate(list_of_tokens):
        if token == '-LRB-':
            list_of_tokens[i] = '('
        elif token == '-RRB-':
            list_of_tokens[i] = ')'
    return list_of_tokens


def read_json(data_location):
    directory = os.fsencode(data_location)
    all_data_dict_per_relation_list = {}
    data_sections = {"train":{},"dev":{},"test":{}}
    for file in os.listdir(directory):
        dataset = file.decode("utf-8")
        dataset = dataset.split('.')[0]
        if dataset not in data_sections: continue
        print(dataset)
        this_dataset_by_relation = data_sections[dataset]
        filename = os.path.join(directory, file)
        with open(filename, "r") as fp:
            data = json.load(fp)
        for x in data:
            tokens = replace_to_rounded_parnthesis(x['token'])
            head_start = x['subj_start']
            head_end = x['subj_end']
            tail_start = x['obj_start']
            tail_end = x['obj_end']
            head = " ".join(tokens[head_start:head_end + 1]).lower()
            tail = " ".join(tokens[tail_start:tail_end + 1]).lower()
            relation = x['relation']
            instance_dict = {"tokens": tokens, "h": [head, None, [[i for i in range(head_start, head_end + 1)]]]
                , "t": [tail, None, [[i for i in range(tail_start, tail_end + 1)]]]}
            insert_to_dict_data(all_data_dict_per_relation_list, relation, instance_dict)
            insert_to_dict_data(this_dataset_by_relation, relation, instance_dict)
        with open("raw/TACRED_" + dataset + ".json", "w") as f:
            json.dump(this_dataset_by_relation, f)

    # with open("TACRED_fewshot_merged.json", "w") as f:
    #     json.dump(all_data_dict_per_relation_list, f)
    return all_data_dict_per_relation_list


def insert_to_dict_data(dict_vat, relation, this_small_instance):
    dict_vat[relation] = dict_vat.get(relation, [])
    dict_vat[relation].append(this_small_instance)


all_data = read_json(data_location)

