import random
import json
import sys
import matplotlib.pyplot as plt
import math, os

FIXED_CATEGORY_DIVISION = "categories_split.json"

NO_REL = "no_relation"
seed_for_random = int(sys.argv[1]) if len(sys.argv) > 1 else 5161
create_data_or_gather_stats = sys.argv[2] if len(sys.argv) > 2 else "create"
random_split_or_from_json_file = sys.argv[3] if len(sys.argv) > 3 else "false"

prefix = "raw/"
train_file = prefix + "TACRED_train.json"
dev_file = prefix + "TACRED_dev.json"
test_file = prefix + "TACRED_test.json"


def filter_out_no_relation_to_keep_dist(original_data, filtered_data):
    no_relation_pro = get_no_relation_propo(original_data)
    sizes, _ = extract_data(filtered_data, with_no_relation=False)
    save_this_amount = sum(sizes) / (no_relation_pro ** -1 - 1)
    save_this_amount = round(save_this_amount)
    filtered_data[NO_REL] = random.sample(filtered_data[NO_REL], save_this_amount)


def print_data(train_raw, dev_raw, test_raw, location=None):
    counts = "counts.png"
    proba_name = "probabilities.png"
    if location:
        counts = location + counts
        proba_name = location + proba_name
    if len(train_raw) == 0:
        train_raw = json.load(open(train_file))
        dev_raw = json.load(open(dev_file))
        test_raw = json.load(open(test_file))
        counts = prefix + counts
        proba_name = prefix + proba_name
    train_size, train_names = extract_data(train_raw, with_no_relation=True)
    dev_size, dev_names = extract_data(dev_raw, with_no_relation=True)
    test_size, test_names = extract_data(test_raw, with_no_relation=True)

    LABELS = list(train_names) + list(dev_names) + list(test_names)

    ax = plt.subplot(111)
    plot_figure(LABELS, ax, dev_size, test_size, train_names, train_size, counts)
    # ax = plt.figure(111)
    fig2, ax1 = plt.subplots(nrows=1, ncols=1)
    train_size = normalize(train_size)
    dev_size = normalize(dev_size)
    test_size = normalize(test_size)
    plot_figure(LABELS, ax1, dev_size, test_size, train_names, train_size, proba_name)
    plt.tight_layout()


def plot_figure(LABELS, ax, dev_size, test_size, train_names, train_size, name):
    ax.bar([i for i in range(len(train_size))], train_size, tick_label=train_names, color='g', align='center')
    ax.bar([i + len(train_size) for i in range(len(dev_size))], dev_size, color='y', align='center')
    ax.bar([i + len(train_size) + len(dev_size) for i in range(len(test_size))], test_size, color='r', align='center')
    # ax.bar(dev, color='g', align='center')
    # ax.bar(test,color='r', align='center')
    # ax.xaxis_date()
    plt.xticks([i for i in range(len(LABELS))], LABELS, rotation=90)
    ax.tick_params(axis='both', which='major', labelsize=5)
    ax.tick_params(axis='both', which='minor', labelsize=4)
    plt.tight_layout()
    plt.savefig(name, dpi=200)


def normalize(data_size):
    return [i / sum(data_size) for i in data_size]


def get_no_relation_propo(data_raw):
    sizes, _ = extract_data(data_raw, with_no_relation=True)
    no_rel_pro = sizes[0] / sum(sizes)
    return no_rel_pro


def extract_data(raw, with_no_relation=False):
    if with_no_relation:
        all_needed = [(k, v) for k, v in sorted(raw.items(), reverse=True, key=lambda item: len(item[1])) if k != ""]
    else:
        all_needed = [(k, v) for k, v in sorted(raw.items(), reverse=True, key=lambda item: len(item[1])) if
                      k != NO_REL]
    return list(map(lambda x: len(x[1]), all_needed)), list(map(lambda x: x[0], all_needed))
    # return list(map(lambda x: len(x), all_needed.values()))[:-1], list(all_needed.keys())[:-1]
    # return list(map(lambda x: len(x), raw.values())), list(raw.keys())


if create_data_or_gather_stats != "create":
    print_data([], [], [])
    exit()


def possible_class_names(rel_name, names, with_no_relation=False):
    if with_no_relation:
        return rel_name in names or rel_name == NO_REL
    else:
        return rel_name in names


def read_data(f):
    # for f in [train_file, dev_file, test_file]:
    fp = open(f, "r")
    data = fp.readlines()
    data = json.loads(data[0])
    return data


def setup_flipped_data(original_data, these_relations):
    data = {NO_REL: []}
    for k, v in original_data.items():
        if possible_class_names(k, these_relations) and k != NO_REL:
            data[k] = v
        else:
            data[NO_REL].extend(v)
    return data


def main():
    '''
    Given the three original supervised data sections
    We re-label all instances in the three sections such that each category
    label appears only in a single data section.
    We choose this sizes:
    Number of train categories is 25
    Number of dev categories is 6
    Number of test categories is 10

    You can either use the fixed split we created for TACRED or use your own
    random split.
    '''
    random.seed(seed_for_random)
    train_categories_size = 25
    dev_categories_size = 6
    test_categories_size = 10

    original_train_data = read_data(train_file)
    original_dev_data = read_data(dev_file)
    original_test_data = read_data(test_file)

    '''
    This script assumes the data is in the form of a dictionary where 
    the key is the class name and the value is a list of all class instances.
    Data form: {"class_name: [x_0,x_1,...,x_N]"}
    '''

    relations = original_train_data.keys()
    relations = set(relations).difference(set([NO_REL]))
    possible_test_relation = relations
    number_of_tests = math.ceil(len(relations) / test_categories_size)

    # predefined categories split
    predefined_split = json.load(open(FIXED_CATEGORY_DIVISION))

    for _test_split_index in range(1, number_of_tests):
        print(_test_split_index)
        if random_split_or_from_json_file != "false":
            # random spilt of relation
            if _test_split_index == number_of_tests - 1:
                test_relations = list(possible_test_relation)
            else:
                test_relations = random.sample(possible_test_relation, test_categories_size)

            all_other_relation = relations - set(test_relations)
            possible_test_relation = possible_test_relation.difference(set(test_relations))
            dev_relations = random.sample(all_other_relation, dev_categories_size)
            train_relations = [k for k in all_other_relation if k not in dev_relations]

        else:
            # get predefined classes
            spilt_number = "split" + str(_test_split_index)
            split = predefined_split[spilt_number]
            train_relations, dev_relations, test_relations = split["train"], split["dev"], split["test"]

        final_train_data, final_dev_data, final_test_data = relabel_data_sections(original_train_data,
                                                                                  original_dev_data, original_test_data,
                                                                                  train_relations, dev_relations,
                                                                                  test_relations)

        current_dir = str(_test_split_index) + "_TACRED_split/"
        create_dir_if_needed(current_dir)
        print_data(final_train_data,final_dev_data,final_test_data,current_dir)
        json.dump(final_train_data, open(current_dir + "train_data.json", "w"))
        json.dump(final_dev_data, open(current_dir + "dev_data.json", "w"))
        json.dump(final_test_data, open(current_dir + "test_data.json", "w"))


def relabel_data_sections(original_train_data, original_dev_data, original_test_data, train_relations, dev_relations,
                          test_relations):
    flipped_train_data = setup_flipped_data(original_train_data, train_relations)
    flipped_dev_data = setup_flipped_data(original_dev_data, dev_relations)
    flipped_test_data = setup_flipped_data(original_test_data, test_relations)
    return flipped_train_data, flipped_dev_data, flipped_test_data


def create_dir_if_needed(dir_name):
    if dir_name[-1] == '/':
        dir_name = dir_name[:-1]
    # Create target Directory if doesn't exist
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


if __name__ == "__main__":
    main()
