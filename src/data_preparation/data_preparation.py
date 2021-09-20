# lib
import csv, os
import logging
import random
import numpy as np
import progressbar
import scipy.sparse as sp


def div_train_test_data_with_explicit(input_file, output_file):
    if not os.path.isdir(output_file):
        os.makedirs(output_file)

    # Local function ---------------------------------------------------------------
    def process_and_write_record_with_explicit(rate_list, num_items):
        global explicit_remove
        num_negative = 999

        # # positive record
        explicit_positive_records = [int(i[1]) for i in rate_list if int(i[4]) > 0]

        # # negative record only for test
        set_total_items = range(num_items)
        population = list(set(set_total_items) - set(explicit_positive_records))
        explicit_negative_records = random.sample(population, num_negative)

        # test record
        explicit_max_time = 0
        for rate_record in rate_list:
            explicit_time = int(rate_record[4])
            if explicit_time >= explicit_max_time:
                explicit_max_time = explicit_time
                explicit_remove = rate_record

        # # write
        if explicit_max_time != 0:
            if explicit_remove in rate_list:
                rate_list.remove(explicit_remove)
            csv_writer_explicit_test.writerow(explicit_remove[0:2] + explicit_remove[4:])

            explicit_negative_record_line = ['(' + str(explicit_remove[0]) + ',' + str(explicit_remove[1]) + ')']
            explicit_negative_record_line += explicit_negative_records
            csv_writer_explicit_negative.writerow(explicit_negative_record_line)

        for rate in rate_list:
            csv_writer_train.writerow(rate)

    # End function ---------------------------------------------------------------

    # get num_users and num_items
    logging.info('Get num_users and num_items')
    num_users = 0
    num_items = 0
    with open(input_file, "r") as f:
        csv_reader = csv.reader(f, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
        for line in csv_reader:
            num_users = max(num_users, int(line[0]))
            num_items = max(num_items, int(line[1]))
    num_users += 1
    num_items += 1
    logging.info('num users: ' + str(num_users) + ';    num items: ' + str(num_items))

    # gen train, test, negative data
    logging.info('gen train, test, negative data')

    train_file = open(output_file + '_explicit.train.rating', 'w')
    csv_writer_train = csv.writer(train_file, delimiter='|', quotechar='', quoting=csv.QUOTE_NONE)

    explicit_test_file = open(output_file + '_explicit.test.rating', 'w')
    csv_writer_explicit_test = csv.writer(explicit_test_file, delimiter='|', quotechar='', quoting=csv.QUOTE_NONE)

    explicit_negative_file = open(output_file + '_explicit.test.negative', 'w')
    csv_writer_explicit_negative = csv.writer(explicit_negative_file, delimiter='|', quotechar='',
                                              quoting=csv.QUOTE_NONE)

    tmp_uid = 0
    item_list_of_tmp_user = []
    with open(input_file, "r") as f:
        csv_reader = csv.reader(f, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
        for line in progressbar.ProgressBar()(csv_reader):
            # type line = [uid item_id time_im num_im time_ex num_ex]
            uid = int(line[0])
            if uid != tmp_uid:
                process_and_write_record_with_explicit(item_list_of_tmp_user, num_items)
                tmp_uid = uid
                item_list_of_tmp_user = []
            item_list_of_tmp_user.append(line)
        # end for    
        process_and_write_record_with_explicit(item_list_of_tmp_user, num_items)

    train_file.close()
    explicit_test_file.close()
    explicit_negative_file.close()


def div_train_test_data_with_explicit_2(input_file, output_file):
    if not os.path.isdir(output_file):
        os.makedirs(output_file)

    # Local function ---------------------------------------------------------------
    def process_and_write_record_with_explicit(rate_list, num_items):
        global explicit_remove
        num_negative = 999

        # # positive record
        explicit_positive_records = [int(i[1]) for i in rate_list if int(i[4]) > 0]

        # # negative record only for test
        set_total_items = range(num_items)
        population = list(set(set_total_items) - set(explicit_positive_records))
        explicit_negative_records = random.sample(population, num_negative)

        # test record
        explicit_max_time = 0
        for rate_record in rate_list:
            explicit_time = int(rate_record[4])
            if explicit_time >= explicit_max_time:
                explicit_max_time = explicit_time
                explicit_remove = rate_record

        # # write
        if explicit_max_time != 0:
            rate_list.remove(explicit_remove)
            explicit_remove[4] = 0
            explicit_remove[5] = 0
            rate_list.append(explicit_remove)

            csv_writer_explicit_test.writerow(explicit_remove[0:2] + explicit_remove[4:])

            explicit_negative_record_line = ['(' + str(explicit_remove[0]) + ',' + str(explicit_remove[1]) + ')']
            explicit_negative_record_line += explicit_negative_records
            csv_writer_explicit_negative.writerow(explicit_negative_record_line)

        for rate in rate_list:
            csv_writer_train.writerow(rate)

    # End function ---------------------------------------------------------------

    # get num_users and num_items
    logging.info('Get num_users and num_items')
    num_users = 0
    num_items = 0
    with open(input_file, "r") as f:
        csv_reader = csv.reader(f, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
        for line in csv_reader:
            num_users = max(num_users, int(line[0]))
            num_items = max(num_items, int(line[1]))
    num_users += 1
    num_items += 1
    logging.info('num users: ' + str(num_users) + ';    num items: ' + str(num_items))

    # gen train, test, negative data
    logging.info('gen train, test, negative data')

    train_file = open(output_file + '_explicit.train.rating', 'w')
    csv_writer_train = csv.writer(train_file, delimiter='|', quotechar='', quoting=csv.QUOTE_NONE)

    explicit_test_file = open(output_file + '_explicit.test.rating', 'w')
    csv_writer_explicit_test = csv.writer(explicit_test_file, delimiter='|', quotechar='', quoting=csv.QUOTE_NONE)

    explicit_negative_file = open(output_file + '_explicit.test.negative', 'w')
    csv_writer_explicit_negative = csv.writer(explicit_negative_file, delimiter='|', quotechar='',
                                              quoting=csv.QUOTE_NONE)

    tmp_uid = 0
    tmp_item_list_single_user = []
    with open(input_file, "r") as f:
        csv_reader = csv.reader(f, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
        for line in progressbar.ProgressBar()(csv_reader):
            # type line = [uid item_id time_im num_im time_ex num_ex]
            uid = int(line[0])
            if uid != tmp_uid:
                process_and_write_record_with_explicit(tmp_item_list_single_user, num_items)
                tmp_uid = uid
                tmp_item_list_single_user = []
            tmp_item_list_single_user.append(line)
        # end for    
        process_and_write_record_with_explicit(tmp_item_list_single_user, num_items)

    train_file.close()
    explicit_test_file.close()
    explicit_negative_file.close()


def gen_ratings_data_with_explicit(implicit_input, explicit_input, user_index_output, item_index_output, output):
    """
        :param implicit_input: format raw_uid,raw_item_id,timestamp
        :param explicit_input: format raw_uid,raw_item_id,timestamp
        :param output: format raw_uid,raw_item_id,implicit_timestamp,num_implicicit,explicit_timestamp,num_explicit
        :return: none
        """
    # create uid blacklist
    logging.info('Create uid blacklist')
    interacted_dict = {}

    file_pointer = open(implicit_input, "r")
    csv_reader = csv.reader(file_pointer, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
    for line in csv_reader:
        raw_uid = line[0]  # string
        raw_item_id = line[1]
        if raw_uid in interacted_dict:
            interacted_dict[raw_uid].add(raw_item_id)
        else:
            # interacted_dict[raw_uid] = type set()
            interacted_dict[raw_uid] = {raw_item_id}

    uid_blacklist = set()
    for uid in interacted_dict:
        if len(interacted_dict[uid]) < 5:
            uid_blacklist.add(uid)
    file_pointer.close()
    print("len uid_blacklist:", len(uid_blacklist))

    # convert uid, item_id
    logging.info("convert uid, item_id")

    # uid_dict, item_id_dict la anh xa tu raw_id sang id dang so nguyen tu 0 den n
    user_id_dict = {}
    item_id_dict = {}

    implicit_rating_dict = {}
    uid_count = 0
    item_id_count = 0

    file_pointer = open(implicit_input, "r")
    csv_reader = csv.reader(file_pointer, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
    for line in csv_reader:
        raw_uid = line[0]  # string
        if raw_uid not in uid_blacklist:
            raw_item_id = line[1]
            raw_timestamp = line[2]
            if raw_uid not in user_id_dict:
                user_id_dict[raw_uid] = uid_count
                uid_count += 1
            if raw_item_id not in item_id_dict:
                item_id_dict[raw_item_id] = item_id_count
                item_id_count += 1

            # add user data to dict
            uid = user_id_dict[raw_uid]  # int
            if uid in implicit_rating_dict:
                implicit_rating_dict[uid].append((item_id_dict[raw_item_id], raw_timestamp))
            else:
                implicit_rating_dict[uid] = [(item_id_dict[raw_item_id], raw_timestamp)]
    file_pointer.close()
    print("len user list:", len(user_id_dict))
    print("len item list:", len(item_id_dict))
    print("len implicit_rating_dict:", len(implicit_rating_dict))

    with open(user_index_output, 'w') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
        for k in user_id_dict:
            csv_writer.writerow([k, user_id_dict[k]])
    with open(item_index_output, 'w') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
        for k in item_id_dict:
            csv_writer.writerow([k, item_id_dict[k]])
    # load explicit data put to explicit_rating_dict
    explicit_rating_dict = {}
    file_pointer = open(explicit_input, "r")

    csv_reader = csv.reader(file_pointer, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
    for line in csv_reader:
        raw_uid = line[0]
        raw_item_id = line[1]
        raw_timestamp = line[2]
        if (raw_uid in user_id_dict) and (raw_item_id in item_id_dict):
            uid = user_id_dict[raw_uid]
            if uid in explicit_rating_dict:
                explicit_rating_dict[uid].append((item_id_dict[raw_item_id], raw_timestamp))
            else:
                explicit_rating_dict[uid] = [(item_id_dict[raw_item_id], raw_timestamp)]
    file_pointer.close()

    # gen ratings data
    logging.info('gen ratings with explicit data')
    out_file = open(output, 'w')
    csv_writer = csv.writer(out_file, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)

    for uid in implicit_rating_dict:
        # check duplicate
        implicit_item_time_dict = {}
        explicit_item_time_dict = {}
        for his in implicit_rating_dict[uid]:
            item_id = his[0]
            timestamp = his[1]
            if item_id in implicit_item_time_dict:
                if timestamp > implicit_item_time_dict[item_id][0]:
                    implicit_item_time_dict[item_id][0] = timestamp
                implicit_item_time_dict[item_id][1] += 1
            else:
                implicit_item_time_dict[item_id] = [timestamp, 1]

        if uid in explicit_rating_dict:
            for his in explicit_rating_dict[uid]:
                item_id = his[0]
                timestamp = his[1]
                if item_id in explicit_item_time_dict:
                    if timestamp > explicit_item_time_dict[item_id][0]:
                        explicit_item_time_dict[item_id][0] = timestamp
                    explicit_item_time_dict[item_id][1] += 1
                else:
                    explicit_item_time_dict[item_id] = [timestamp, 1]

        # write to file
        for i in implicit_item_time_dict:
            if i in explicit_item_time_dict:
                if implicit_item_time_dict[i][0] > explicit_item_time_dict[i][0]:
                    implicit_item_time_dict[i][0] = explicit_item_time_dict[i][0]
                csv_writer.writerow([uid, i] + implicit_item_time_dict[i] + explicit_item_time_dict[i])
            else:
                csv_writer.writerow([uid, i] + implicit_item_time_dict[i] + [-1, 0])
        for j in explicit_item_time_dict:
            if j not in implicit_item_time_dict:
                csv_writer.writerow([uid, j] + explicit_item_time_dict[j] + explicit_item_time_dict[j])
    out_file.close()


