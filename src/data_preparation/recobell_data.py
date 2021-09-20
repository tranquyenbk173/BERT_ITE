"""
Tien xu ly du lieu cua recobell data
"""
import csv
import datetime
import logging
import subprocess
import time
import pandas as pd
import progressbar

from src import settings

# config log
from src.data_preparation import data_preparation

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

data_name = "recobell/"
root_path = settings.DATA_ROOT_PATH + 'site_data/' + data_name


def convert_time(d):
    try:
        new_d = int(time.mktime(datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S.%f").timetuple()))
    except ValueError:
        new_d = int(time.mktime(datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S").timetuple()))
    return new_d


def gen_implicit_cleaned_data(input_file, output_file):
    logging.info("gen_implicit_cleaned_data")

    start_date = convert_time("2016-08-08 00:00:00.000")
    finish_date = convert_time("2016-08-15 00:00:00.000")

    file_out = open(output_file, "w")
    # f_2 = open(root_path + 'raw_data/tiny_site_view_log.csv000', "w")
    csv_writer = csv.writer(file_out, delimiter=",", quotechar="", quoting=csv.QUOTE_NONE)
    # csv_writer_2 = csv.writer(f_2, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)
    with open(input_file, "r") as f:

        csv_reader = csv.reader(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)

        for line in progressbar.ProgressBar()(csv_reader):
            raw_timestamp = line[0]
            raw_uid = line[3]
            raw_item_id = line[4]
            timestamp = convert_time(raw_timestamp)

            if start_date <= timestamp < finish_date:
                # csv_writer_2.writerow(line)
                csv_writer.writerow([raw_uid, raw_item_id, timestamp])
    file_out.close()
    # f_2.close()


def gen_explicit_cleaned_data(input_file, output_file):
    logging.info("gen_explicit_cleaned_data")

    start_date = convert_time("2016-08-08 00:00:00.000")
    finish_date = convert_time("2016-08-15 00:00:00.000")
    # convert uid, item_id

    file_out = open(output_file, "w")
    # f_2 = open(root_path + 'raw_data/tiny_site_order_log.csv000', "w")
    csv_writer = csv.writer(file_out, delimiter=",", quotechar="", quoting=csv.QUOTE_NONE)
    # csv_writer_2 = csv.writer(f_2, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)
    with open(input_file, "r") as f:

        csv_reader = csv.reader(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)

        for line in progressbar.ProgressBar()(csv_reader):
            raw_timestamp = line[0]
            raw_uid = line[3]
            raw_item_id = line[4]
            timestamp = convert_time(raw_timestamp)

            if start_date <= timestamp < finish_date:
                # csv_writer_2.writerow(line)
                csv_writer.writerow([raw_uid, raw_item_id, timestamp])
    file_out.close()
    # f_2.close()


def construct_item_vector():
    item_recobell_df = pd.read_csv(root_path + 'raw_data/site_product.csv000',
                                   header=None,
                                   names=['itemid', 'price', 'cat1', 'cat2', 'cat3', 'cat4', 'brandid'])

    cat_1_list = list(set(item_recobell_df.cat1))
    cat_1_dict = {cat_1_list[i]: i for i in range(len(cat_1_list))}
    a = len(cat_1_list)
    cat_2_list = list(set(item_recobell_df.cat2))
    cat_2_dict = {cat_2_list[i]: i + a for i in range(len(cat_2_list))}
    a += len(cat_2_list)
    cat_3_list = list(set(item_recobell_df.cat3))
    cat_3_dict = {cat_3_list[i]: i + a for i in range(len(cat_3_list))}
    a += len(cat_3_list)
    cat_4_list = list(set(item_recobell_df.cat4))
    cat_4_dict = {cat_4_list[i]: i + a for i in range(len(cat_4_list))}
    a += len(cat_4_list)
    with open('../../data/recobell_cat1_index.txt', 'w') as f:
        for k in cat_1_dict:
            f.write(k + ',' + str(cat_1_dict[k]) + '\n')
    with open('../../data/recobell_cat2_index.txt', 'w') as f:
        for k in cat_2_dict:
            f.write(k + ',' + str(cat_2_dict[k]) + '\n')
    with open('../../data/recobell_cat3_index.txt', 'w') as f:
        for k in cat_3_dict:
            f.write(k + ',' + str(cat_3_dict[k]) + '\n')
    with open('../../data/recobell_cat4_index.txt', 'w') as f:
        for k in cat_4_dict:
            f.write(k + ',' + str(cat_4_dict[k]) + '\n')
    # cat_1_dict = {}
    # cat_2_dict = {}
    # cat_3_dict = {}
    # cat_4_dict = {}
    # with open(root_path + 'recobell_cat1_index.txt', 'r') as f:
    #     for line in f:
    #         s_line = line.split(',')
    #         cat_1_dict[s_line[0]] = int(s_line[1])
    # print(cat_1_dict)
    # with open(root_path + 'recobell_cat2_index.txt', 'r') as f:
    #     for line in f:
    #         s_line = line.split(',')
    #         cat_2_dict[s_line[0]] = int(s_line[1])
    # with open(root_path + 'recobell_cat3_index.txt', 'r') as f:
    #     for line in f:
    #         s_line = line.split(',')
    #         cat_3_dict[s_line[0]] = int(s_line[1])
    # with open(root_path + 'recobell_cat4_index.txt', 'r') as f:
    #     for line in f:
    #         s_line = line.split(',')
    #         cat_4_dict[s_line[0]] = int(s_line[1])
    # with open(root_path + 'item_repr.txt', 'w') as f:
    #     csv_writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_ALL)
    #     for item in item_recobell_df.iterrows():
    #         sp_vec = {cat_1_dict[item[1].cat1]: 1.0, cat_2_dict[item[1].cat2]: 1.0, cat_3_dict[item[1].cat3]: 1.0,
    #                   cat_4_dict[item[1].cat4]: 1.0}
    #         sp_repr = dict_sparse_vector_to_json_string(sp_vec)
    #         csv_writer.writerow((item[1].itemid, sp_repr))
    # print('Done infer pcat repr of recobell items !')




def main():
    implicit_data_path = root_path + "/raw_data/new_tiny_site_view_log.csv000"
    explicit_data_path = root_path + "/raw_data/new_tiny_site_order_log.csv000"
    cleaned_implicit_data_path = root_path + "/_implicit.clean.txt"
    cleaned_explicit_data_path = root_path + "/_explicit.clean.txt"

    user_index_dict = root_path + '/u2index.txt'
    item_index_dict = root_path + '/i2index.txt'
    combined_data = root_path + "/ratings.txt"
    output_root_name = root_path + "/scene_1/"

    # # implicit
    # gen_implicit_cleaned_data(implicit_data_path, cleaned_implicit_data_path)
    # # explicit
    # gen_explicit_cleaned_data(explicit_data_path, cleaned_explicit_data_path)
    # logging.info("--> Done, gen_cleaned_data")
    #
    # # gen ratings with explicit
    # data_preparation.gen_ratings_data_with_explicit(cleaned_implicit_data_path, cleaned_explicit_data_path,
    #                                                 user_index_dict, item_index_dict, combined_data)
    # logging.info("--> Done, gen_ratings_data_with_explicit")

    # # div train test data with explicit
    # data_preparation.div_train_test_data_with_explicit(combined_data, output_root_name)
    # logging.info("--> Done, div_train_test_data_with_explicit")

    subprocess.call(['bash', 'bin/split.sh', output_root_name])
    logging.info("--> Done, split_train_data_into_partition")
    # data_for_VALS.preprocessing_for_VALS(data_name)
    # logging.info("--> Done, for_VALS")

    # output_root_name = root_path + "/scene_2/"
    # data_preparation.div_train_test_data_with_explicit_2(combined_data, output_root_name)
    # logging.info('--> Done, div_train_test_data_with_explicit_2')


if __name__ == "__main__":
    main()
