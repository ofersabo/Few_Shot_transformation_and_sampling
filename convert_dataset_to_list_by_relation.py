import json
import os, argparse

def add_additional_data(x):
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
    instance_info = {**x, **instance_dict}
    return instance_info,relation


def replace_to_rounded_parnthesis(list_of_tokens):
    for i, token in enumerate(list_of_tokens):
        if token == '-LRB-':
            list_of_tokens[i] = '('
        elif token == '-RRB-':
            list_of_tokens[i] = ')'
    return list_of_tokens

def read_json(data_location,output_file):
    this_dataset_by_relation = {}
    with open(data_location, "r") as fp:
        data = json.load(fp)
    for x in data:
        instance_dict, relation = add_additional_data(x)
        insert_to_dict_data(this_dataset_by_relation, relation, instance_dict)
    with open(output_file, "w") as f:
        json.dump(this_dataset_by_relation, f)


def insert_to_dict_data(dict_vat, relation, this_small_instance):
    dict_vat[relation] = dict_vat.get(relation, [])
    dict_vat[relation].append(this_small_instance)

def main(args):
    all_data = read_json(args.dataset,args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, type=str, required=True,
                        help="Pointer to dataset, either train,dev or test dataset")

    parser.add_argument("--output_file", type=str, required=False,
                        help="The output file where the instances per relation are stored.")

    _args = parser.parse_args()
    main(_args)
