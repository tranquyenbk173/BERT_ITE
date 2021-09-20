"""
Tien xu ly du lieu cua retail rocket data
"""

import csv
import logging
import subprocess

from src import settings
from src.data_preparation import data_preparation, sparse_vector

# config log
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

data_name = "retail_rocket/"
root_path = settings.DATA_ROOT_PATH + "site_data/" + data_name


def gen_cleaned_data(input_file, cleaned_implicit_data_path, cleaned_explicit_data_path):
    logging.info('Gen raw data')

    implicit_output = open(cleaned_implicit_data_path, 'w')
    explicit_output = open(cleaned_explicit_data_path, 'w')
    csv_writer_implicit = csv.writer(implicit_output, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
    csv_writer_explicit = csv.writer(explicit_output, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)

    with open(input_file, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
        line = next(csv_reader)  # skip header
        for line in csv_reader:
            timestamp = line[0]
            raw_uid = line[1]
            event = line[2]
            raw_item_id = line[3]
            csv_writer_implicit.writerow([raw_uid, raw_item_id, timestamp])
            if (event == 'transaction') or (event == 'addtocart'):
                csv_writer_explicit.writerow([raw_uid, raw_item_id, timestamp])
    # ghi tat ca interact vao file implicit de loai bo black list user cho chinh xac

    implicit_output.close()
    explicit_output.close()


def construct_pcat_repr():
    with open(root_path + 'i2index.txt') as f:
        active_item_list = [line.split(',')[0] for line in f]
    print(len(active_item_list))

    item_dict = {}
    with open(root_path + 'i2pcat.txt') as f:
        for line in f:
            item_id, pcat = line.split(',')
            if item_id not in item_dict:
                item_dict[item_id] = [int(pcat)]
            else:
                item_dict[item_id].append(int(pcat))
    print(len(item_dict))
    cat2parent = {}
    max_cat = 0
    with open(root_path + 'raw_data/category_tree.csv', newline='') as f:
        next(f)
        for line in f:
            cat, parent = line.strip().split(',')
            cat = int(cat)
            if cat > max_cat:
                max_cat = cat
            if parent != '':
                parent = int(parent)
                if parent > max_cat:
                    max_cat = parent
                cat2parent[cat] = parent
    if 231 in cat2parent:
        print('huhu')
    with open(root_path + 'item_repr.txt', 'w') as f:
        f.write(str(max_cat + 1) + '\n')
        csv_writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_ALL)
        for item_id in item_dict:
            sp_vec = banner_cats_to_vector(item_dict[item_id], cat2parent)
            sp_repr = sparse_vector.dict_sparse_vector_to_json_string(sp_vec)
            csv_writer.writerow((item_id, sp_repr))
    print('Done infer pcat repr of retail_rocket items !')


def banner_cats_to_vector(cats, cat_to_parent_cat):
    sv = {}
    for cat in cats:
        sv[cat] = 1.0
        while cat in cat_to_parent_cat:
            cat = cat_to_parent_cat[cat]
            sv[cat] = 1.0
    banner_spare_vector = sparse_vector.normalize(sv)
    return banner_spare_vector


def main():
    # data_path = settings.DATA_ROOT_PATH + "site_data/" + data_name + "/raw_data/events_purchase.csv"
    data_path = root_path + 'raw_data/new_events.csv'

    cleaned_implicit_data_path = root_path + '_implicit.clean.txt'
    cleaned_explicit_data_path = root_path + '_explicit.clean.txt'

    user_index_dict = root_path + 'u2index.txt'
    item_index_dict = root_path + 'i2index.txt'
    combined_data = root_path + 'ratings.txt'
    output_root_name = root_path + 'scene_1/'

    # # Tao ra implicit va explicit file
    # gen_cleaned_data(data_path, cleaned_implicit_data_path, cleaned_explicit_data_path)
    # logging.info('--> Done, gen_cleaned_data')

    # # Danh lai so thu tu cho user, item, cac user co tuong tac trong file implicit < 5 bi loai
    # # Tao du lieu co explicit chac chan se co implicit
    # data_preparation.gen_ratings_data_with_explicit(cleaned_implicit_data_path, cleaned_explicit_data_path,
    #                                                 user_index_dict, item_index_dict, combined_data)
    # logging.info('--> Done, gen_ratings_data_with_explicit')
    #
    # # div train test data with explicit
    # data_preparation.div_train_test_data_with_explicit(combined_data, output_root_name)
    # logging.info('--> Done, div_train_test_data_with_explicit')

    subprocess.call(['bash', 'bin/split.sh', output_root_name])
    logging.info("--> Done, split_train_data_into_partition")


if __name__ == '__main__':
    main()
    # construct_pcat_repr()
