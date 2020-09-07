import json
import os, argparse

def read_json(data_location,output_file):
    this_dataset_by_relation = {}
    with open(data_location, "r") as fp:
        data = json.load(fp)
    for x in data:
        relation = x['relation']
        insert_to_dict_data(this_dataset_by_relation, relation, x)
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
