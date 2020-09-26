# Few-Shot Transformation and Sampling 
#### In this repository we share the script to transform any supervised dataset into a Few-Shot dataset. 
#### Further, this repository contains a script to generate episodes as described in the paper.

### To transform TACRED into our suggested Few-Shot TACRED please use this commad
##### Getting access to TACRED dataset
https://nlp.stanford.edu/projects/tacred/#access

##### Convert TACRED to a list of instances per relation, you can use this script, on each of these train dev test dataset.
These three commands convert each of TACRED train/dev/test into a list of instances per relation type.  

 ``` bash 
 python convert_dataset_to_list_by_relation.py --dataset raw/train.json --output_file raw/instances_per_relation/TACRED_train.json
 python convert_dataset_to_list_by_relation.py --dataset raw/dev.json --output_file raw/instances_per_relation/TACRED_dev.json
 python convert_dataset_to_list_by_relation.py --dataset raw/test.json --output_file raw/instances_per_relation/TACRED_test.json
 ```




##### Convert these datasets into Few-Shot, i.e. the classes are disjoint across the three data sections.

This command utilizes our method of transforming supervised dataset into Few-Shot Learning dataset on TACRED. 
``` bash 
python data_transformation.py --train_data raw/instances_per_relation/TACRED_train.json --dev_data raw/instances_per_relation/TACRED_dev.json --test_data raw/instances_per_relation/TACRED_test.json --fixed_categories_split categories_split.json --test_size 10 --output_dir ./data_few_shot
```

voila, the new Few-Shot TACRED dataset, divided into train dev and test datasets.

### To generate episodes for Few-Shot TACRED with respect to data distribution

``` bash 
python episodes_sampling_for_fs_TACRED.py --file_name [train/dev/test] --episodes_size [episodes_size] --N [N_way] --K [K_shot] --number_of_queries [number_of_test_instances] --seed [123] --output_file_name [output_file_name]
``` 

##### Generating Few-Shot TACRED test episodes 

To create the test episodes benchmark, use this shell script:
Creating five files of episodes with seed ranging from 160290 to 160294

Here is the shell command: 

``` bash
./create_test_episodes.sh
```


### Downsampling the NOTA category in the training set, as we found it to be advantageous. 
Here is the command that generates the same downsampled training dataset as we used. If you choose to downsample the training data, apply
this downsampling before generating episodes.

```bash 
python downsample_train_data.py --dataset data_few_shot/_train_data.json --output_file data_few_shot/new_downsampled_train_data.json   
```
 