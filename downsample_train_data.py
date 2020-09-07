import json,sys

def do_downsampling_and_output_file(numbers):
    no_rel_original = data_original["no_relation"]
    new_no_rel = [no_rel_original[i] for i in numbers]
    final_data = data_original
    final_data["no_relation"] = new_no_rel
    with open("new_downsampled_train_data.json","w") as f:
        json.dump(final_data,f)


original_data = sys.argv[1]
original_data = open(original_data)
data_original = json.load(original_data)

with open("numbers_for_downsampling.txt","r") as f:
    x = json.load(f)

numbers = x["the_required_numbers"]

do_downsampling_and_output_file(numbers)
