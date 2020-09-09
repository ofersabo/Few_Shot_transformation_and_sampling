# Few-Shot Transformation and Sampling 
#### In this repository we share the script to transform any supervised dataset into a Few-Shot dataset. 
#### Further, this repository contains a script to generate episodes as described in link to paper.

### To transform TACRED into our suggested Few-Shot TACRED please use this commad
##### Convert TACRED to a list of instances per relation, you can use this script, on each of these train dev test dataset.

1. python convert_dataset_to_list_by_relation.py --dataset raw/train.json --output_file raw/instances_per_relation/TACRED_train.json
2. python convert_dataset_to_list_by_relation.py --dataset raw/dev.json --output_file raw/instances_per_relation/TACRED_dev.json
3. python convert_dataset_to_list_by_relation.py --dataset raw/test.json --output_file raw/instances_per_relation/TACRED_test.json

##### Convert these datasets into Few-Shot, i.e. the classes are disjoint across.

1. transform TACRED into Few-Shot TACRED 
python data_transformation.py --train_data raw/instances_per_relation/TACRED_train.json --dev_data raw/instances_per_relation/TACRED_dev.json --test_data raw/instances_per_relation/TACRED_test.json --fixed_categories_split categories_split.json --test_size 10 --output_dir ./data_few_shot

voila, the new Few-Shot TACRED dataset, divided into train dev and test datasets.



### To generate episodes for Few-Shot TACRED with respect to data distribution

##### To generate the test episodes for Few-Shot TACRED 
python episodes_sampling_for_fs_TACRED.py --file_name [test_data] --episodes_size 10000 --N 5 --K 1 --number_of_queries 3 --seed 16029[0-4] --output_file_name [output_file_name] 
Remember to create four files of episodes with seed ranging from 160290 to 160294
Use the same script to generate train and dev episodes. 


### In our research we found that it is advantageous to downsample the train set. 
Here is the command that generates the same dataset as we used. If you choose to downsample the training data, then apply
this downsampling before generating episodes.

python downsample_train_data.py --dataset data_few_shot/_train_data.json --output_file data_few_shot/new_downsampled_train_data.json   

 