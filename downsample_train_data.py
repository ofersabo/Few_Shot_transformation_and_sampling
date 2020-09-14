import json,sys,pickle
import argparse

def do_downsampling_and_output_file(numbers,data_original,output):
    no_rel_original = data_original["no_relation"]
    new_no_rel = [no_rel_original[i] for i in numbers]
    final_data = data_original
    final_data["no_relation"] = new_no_rel
    with open(output,"w") as f:
        json.dump(final_data,f)


def main(args):
    original_data = args.dataset
    original_data = open(original_data)
    data_original = json.load(original_data)
    numbers = pickle.load(open("numbers_for_downsampling.pickle", "rb"))

    do_downsampling_and_output_file(numbers,data_original,args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, type=str, required=True,
                        help="Pointer to train dataset, to be downsampled")

    parser.add_argument("--output_file", type=str, required=False,
                        help="The output file where the downsampled dataset is stored.")

    _args = parser.parse_args()
    main(_args)
