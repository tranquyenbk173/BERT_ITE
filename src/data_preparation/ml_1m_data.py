"""
Tien xu ly du lieu cua movielens data
"""
import csv
import logging
import subprocess

import progressbar
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

from src import settings
from src.data_preparation import data_preparation, sparse_vector

# config log
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

data_name = "ml-1m/"
root_path = settings.DATA_ROOT_PATH + 'site_data/' + data_name


def gen_cleaned_data(input_file, implicit_file, explicit_file):
    logging.info("gen_cleaned_data_with_explicit")

    implicit_output = open(implicit_file, "w")
    explicit_output = open(explicit_file, "w")
    csv_writer_implicit = csv.writer(implicit_output, delimiter=",", quotechar="", quoting=csv.QUOTE_NONE)
    csv_writer_explicit = csv.writer(explicit_output, delimiter=",", quotechar="", quoting=csv.QUOTE_NONE)

    with open(input_file, "r") as file:
        for line in progressbar.ProgressBar()(file):
            component = line.strip().split("::")
            user_id = component[0]
            item_id = component[1]
            rating = int(component[2])
            timestamp = component[3]
            csv_writer_implicit.writerow([user_id, item_id, timestamp, rating])
            if rating >= 4:
                csv_writer_explicit.writerow([user_id, item_id, timestamp, rating])

    implicit_output.close()
    explicit_output.close()


def construct_pcat_repr():
    cat_dict = {'Action': 0,
                'Adventure': 1,
                'Animation': 2,
                "Children's": 3,
                'Comedy': 4,
                'Crime': 5,
                'Documentary': 6,
                'Drama': 7,
                'Fantasy': 8,
                'Film-Noir': 9,
                'Horror': 10,
                'Musical': 11,
                'Mystery': 12,
                'Romance': 13,
                'Sci-Fi': 14,
                'Thriller': 15,
                'War': 16,
                'Western': 17}
    max_cat = len(cat_dict)
    item_dict = {}
    with open(root_path + 'i2pcat.txt') as f:
        for line in f:
            item_id, pcat = line.split(',')
            item_dict[item_id] = pcat

    print(len(item_dict))

    with open(root_path + 'item_repr.txt', 'w') as f:
        f.write(str(max_cat) + '\n')
        csv_writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_ALL)
        for item_id in item_dict:
            sp_vec = {}
            cat_list = item_dict[item_id].strip().split('|')
            for cat in cat_list:
                sp_vec[cat_dict[cat]] = 1.0
            sp_repr = sparse_vector.dict_sparse_vector_to_json_string(sp_vec)
            csv_writer.writerow((item_id, sp_repr))
    print('Done infer pcat repr of ml_1m items !')


def build_user_pcat_repr():
    # count = [0] * 5
    # with open(root_path + "_implicit.clean.txt", 'r') as f:
    #     csv_reader = csv.reader(f, delimiter=",", quotechar="", quoting=csv.QUOTE_NONE)
    #     for row in csv_reader:
    #         count[int(row[3]) - 1] += 1
    # print(count)
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
        x[0], (sparse_vector.json_string_to_dense_vector(item_repr_dict[x[1]], dimension=18), 1))) \
        .reduceByKey(sparse_vector.add_avg) \
        .mapValues(normalize) \
        .mapValues(lambda x: sparse_vector.list_sparse_vector_to_json_string(
        sparse_vector.dense_vector_to_list_sparse_vector(x, threshold=1e-5)))

    ur_schema = StructType([
        StructField('uid', IntegerType(), True),
        StructField('repr', StringType(), True)
    ])
    user_repr_df = spark.createDataFrame(user_repr_rdd, schema=ur_schema)
    user_repr_df.show(truncate=False)
    user_repr_df.coalesce(1).write.mode('overwrite').csv(root_path + 'user_repr.txt', sep=',', quote='|', quoteAll=True)


def normalize(y):
    vec = y[0]
    num = y[1]
    res = [val / num for val in vec]
    return res


def main():
    data_path = root_path + "raw_data/ratings.dat"

    cleaned_implicit_data_path = root_path + "_implicit.clean.txt"
    cleaned_explicit_data_path = root_path + "_explicit.clean.txt"

    user_index_dict = root_path + 'u2index.txt'
    item_index_dict = root_path + 'i2index.txt'
    combined_data = root_path + "ratings.txt"
    output_root_name = root_path + "scene_1/"

    # # gen cleaned data
    gen_cleaned_data(data_path, cleaned_implicit_data_path, cleaned_explicit_data_path)
    logging.info("--> Done, gen_cleaned_data_with_explicit")
    #
    # # # gen ratings with explicit
    # data_preparation.gen_ratings_data_with_explicit(cleaned_implicit_data_path, cleaned_explicit_data_path,
    #                                                 user_index_dict, item_index_dict, combined_data)
    # logging.info("--> Done, gen_ratings_data_with_explicit")

    # # div train test data with explicit
    # data_preparation.div_train_test_data_with_explicit(combined_data, output_root_name)
    # logging.info("--> Done, div_train_test_data_with_explicit")
    #
    # subprocess.call(['bash', 'bin/split.sh', output_root_name])
    # logging.info("--> Done, split_train_data_into_partition")


if __name__ == "__main__":
    # main()
    build_user_pcat_repr()
    # construct_pcat_repr()
