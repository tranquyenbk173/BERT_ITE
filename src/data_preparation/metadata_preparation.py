import csv

import pandas as pd
import progressbar
from pyspark.sql import SparkSession
from pyspark.sql.types import *

from src import settings
from src.data_preparation import sparse_vector
from src.data_preparation.sparse_vector import *

data_name = "retail_rocket/"
root_path = settings.DATA_ROOT_PATH + 'site_data/' + data_name
PCAT_DIMENSION = 1699
PCAT_THRESHOLD = 1e-5


def construct_item_vector():
    data_name = 'recobell/'
    root_path = settings.DATA_ROOT_PATH + 'site_data/' + data_name
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


def load_cat_parent_cat_to_dict(cat_filepath):
    cat_to_parent = {}
    with open(cat_filepath, 'r') as f:
        for line in f:
            cat, parent = line.strip().split('')
            if parent != '0':
                cat_to_parent[int(cat)] = int(parent)
    return cat_to_parent


def get_banner_cats(banner_cat_filepath):
    banner_to_cats = {}
    banner_no_cats = []
    with open(banner_cat_filepath, 'r') as f:
        for line in f:
            bannerid, banner_cats = line.split('\t')
            banner_cats = banner_cats.strip()
            if banner_cats != 'None' and len(banner_cats) > 1:
                banner_to_cats[int(bannerid)] = [int(cat) for cat in banner_cats.split(',')]
            else:
                banner_no_cats.append(bannerid)
    with open("../../../data/banner_no_cats.txt", 'wb') as f:
        for id in banner_no_cats:
            f.write(id + '\n')
    return banner_to_cats


def banner_cats_to_vector(cats, cat_to_parent_cat):
    sv = {}
    for cat in cats:
        sv[cat - 1] = 1.0
        while cat in cat_to_parent_cat.keys():
            cat = cat_to_parent_cat[cat]
            sv[cat - 1] = 1.0
    banner_spare_vector = normalize(sv)
    return banner_spare_vector


def normalize(y):
    vec = y[0]
    num = y[1]
    res = [val / num for val in vec]
    return res


def construct_user_vector():
    with open(root_path + 'item_repr.txt') as f:
        next(f)
        csv_reader = csv.reader(f, delimiter=",", quotechar='|', quoting=csv.QUOTE_ALL)
        item_repr_dict = {int(line[0]): line[1] for line in progressbar.ProgressBar()(csv_reader)}
    # initialise sparkContext
    spark = SparkSession.builder \
        .appName("construct_user_vector") \
        .getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    ex_schema = StructType([
        StructField('uid', IntegerType(), True),
        StructField('itemid', IntegerType(), True)
    ])

    ex_df = spark.read.csv(root_path + 'ui_ex.txt', schema=ex_schema)
    ex_df.show()
    user_repr_rdd = ex_df.rdd \
        .repartition(4) \
        .map(lambda x: (
        x[0], (sparse_vector.json_string_to_dense_vector(item_repr_dict[x[1]], dimension=PCAT_DIMENSION), 1))) \
        .reduceByKey(sparse_vector.add_avg) \
        .mapValues(normalize) \
        .mapValues(lambda x: sparse_vector.list_sparse_vector_to_json_string(
        sparse_vector.dense_vector_to_list_sparse_vector(x, threshold=PCAT_THRESHOLD)))

    ur_schema = StructType([
        StructField('uid', IntegerType(), True),
        StructField('repr', StringType(), True)
    ])
    user_repr_df = spark.createDataFrame(user_repr_rdd, schema=ur_schema)
    user_repr_df.show(truncate=False)
    user_repr_df.coalesce(1).write.mode('overwrite').csv(root_path + 'user_repr.txt', sep=',', quote='|', quoteAll=True)


def main():
    construct_user_vector()


if __name__ == '__main__':
    main()
