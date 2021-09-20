# lib
import heapq
import logging
import math
import os

import numpy as np
import progressbar
import tensorflow as tf
from terminaltables import AsciiTable

from src.data_preparation import data_utils

# config
tf.logging.set_verbosity(tf.logging.INFO)
np.random.seed(0)
tf.set_random_seed(0)


class MF:
    def __init__(self, root_path, params, log_path, file_model, save_log=True, save_model=True):
        self.root_path = root_path
        self.params = params
        self.log_path = log_path
        log_directory = '/'.join(log_path.split('/')[:-1])
        if not os.path.isdir(log_directory):
            os.makedirs(log_directory)
        self.log_dir = log_directory
        self.result_string = ''
        if not os.path.isdir(file_model):
            os.makedirs(file_model)
        self.file_model = file_model
        self.item_repr = None
        self.save_log = save_log
        self.save_model = save_model

    def run(self):
        data = self.load_data()
        model = self.create_model()
        self.train(model, data)

    @staticmethod
    def show_result_keyvalue(tuple_data):
        table_data = [['key', 'values']]
        for i in tuple_data:
            table_data.append([i[0], i[1]])
        table = AsciiTable(table_data)
        result = table.table
        print(result)
        return str(result)

    def load_data(self):
        logging.info('JOB INFO: ' + type(self).__name__)

        logging.info('Loading data ...')
        num_user, num_item = data_utils.load_representation_data(
            self.root_path + 'u2index.txt',
            self.root_path + 'i2index.txt')
        self.params['num_user'] = num_user
        self.params['num_item'] = num_item
        interact_mat = data_utils.load_interact_matrix(self.root_path + 'scene_1/_explicit.train.rating', num_user,
                                                       num_item)
        # training_dict = data_utils.load_train_data(self.train_path + 'scene_1/_explicit.train.rating')
        test_data = data_utils.load_test_data(self.root_path + "scene_1/_explicit.test.rating")
        negative_data = data_utils.load_negative_data(self.root_path + "scene_1/_explicit.test.negative")
        self.result_string += 'jobs: ' + type(self).__name__ + '\n\n' + MF.show_result_keyvalue(
            self.params.items()) + '\n\n'
        return {
            # 'training_dict': training_dict,
            'interact_mat': interact_mat,
            'test_data': test_data,
            'negative_data': negative_data
        }

    def create_model(self):
        return {}

    def restore_checkpoint(self, sess, saver):

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.file_model + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            print("--------------> Loading checkpoint <----------------")
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("--------------> Done! <----------------")
        else:
            print("-------------> Initializing weights <---------------")

    def train(self, model, data):
        # params and data
        epochs = self.params['epochs']
        num_negatives = self.params['num_negatives']
        batch_size = self.params['batch_size']
        verbose = self.params['verbose']
        eval_top_kS = self.params['eval_top_k'] #list of top k to eval
        num_user = self.params['num_user']
        num_item = self.params['num_item']
        # training_dict = data['training_dict']
        interact_mat = data['interact_mat']
        test_data = data['test_data']
        negative_data = data['negative_data']

        # jobs

        optimizer = model['optimizer']
        user_index = model['user_index']
        item_index = model['item_index']
        labels_ph = model['labels']
        # ex_indicators = model['ex_indicators']
        y1_indicators = model['y1_indicators']
        y2_indicators = model['y2_indicators']
        loss = model['loss']
        loss_implicit = model['loss_implicit']
        loss_explicit = model['loss_explicit']
        train_ex_prediction = model['train_ex_prediction']
        prediction_implicit = model['prediction_implicit']
        prediction_explicit = model['prediction_explicit']

        # train
        saver = tf.train.Saver()
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        # session_conf = tf.ConfigProto(log_device_placement=False)
        # session_conf.gpu_options.per_process_gpu_memory_fraction = 0.5
        session_conf.gpu_options.allow_growth = True
        session_conf.gpu_options.visible_device_list = '0'

        with tf.Session(config=session_conf) as sess:
            sess.run(tf.global_variables_initializer())
            model['sess'] = sess

            # for evaluate model_ite_onehot_log_loss
            explicit_best_hit, explicit_best_ndcg = ({
              5: [0, 0],
              10: [0, 0],
              20: [0, 0],
              30: [0, 0],
              40: [0, 0],
              50: [0, 0]},

              {
              5: [0, 0],
              10: [0, 0],
              20: [0, 0],
              30: [0, 0],
              40: [0, 0],
              50: [0, 0]})


            explicit_hit = {5:0, 10:0, 20:0, 30:0, 40:0, 50:0}
            explicit_ndcg = {5:0, 10:0, 20:0, 30:0, 40:0, 50:0}
            # for logging
            result = [["epoch", "total_loss", "explicit_hit_5", "explicit_ndcg_5", \
                                              "explicit_hit_10", "explicit_ndcg_10",\
                                              "explicit_hit_20", "explicit_ndcg_20",\
                                              "explicit_hit_30", "explicit_ndcg_30",\
                                              "explicit_hit_40", "explicit_ndcg_40",\
                                              "explicit_hit_50", "explicit_ndcg_50"]]
            print("testing at initialization ........")
            for k in eval_top_kS:
              explicit_hit[k], explicit_ndcg[k] = self.evaluate_model(model,
                                                              k,
                                                              test_data,
                                                              negative_data,
                                                              prediction_explicit)
              if explicit_hit[k] > explicit_best_hit[k][0]:
                  explicit_best_hit[k] = [explicit_hit[k], 'init']  # log result
           
              if explicit_ndcg[k] > explicit_best_ndcg[k][0]:  # log result
                  explicit_best_ndcg[k] = [explicit_ndcg[k], 'init']

            # log result
            table_data = {"total_loss": '_',
                          "eval_explicit_top5": (explicit_hit[5], explicit_ndcg[5]),
                          "eval_explicit_top10": (explicit_hit[10], explicit_ndcg[10]),
                          "eval_explicit_top20": (explicit_hit[20], explicit_ndcg[20]),
                          "eval_explicit_top30": (explicit_hit[30], explicit_ndcg[30]),
                          "eval_explicit_top40": (explicit_hit[40], explicit_ndcg[40]),
                          "eval_explicit_top50": (explicit_hit[50], explicit_ndcg[50])}
            
            temp = ['init', '_']
            for eval_top_k in eval_top_kS:
                temp.append(explicit_hit[eval_top_k])
                temp.append(explicit_ndcg[eval_top_k])

            result.append(temp)
            MF.show_result_keyvalue(table_data.items())

            for e in range(epochs):
                logging.warning("epochs: " + str(e))
                rloss = 0.0
                num_batch = 0
                partitioned_train_path = self.root_path + 'scene_1/partitioned_train_data/'

                for partition_name in sorted(os.listdir(partitioned_train_path)):
                    print(str(e + 1) + ':' + partition_name)
                    partitioned_path = partitioned_train_path + partition_name
                    # ----------------------- get train instances -------------------------------
                    user_ids, item_ids, labels, y1_indicator, y2_indicator = data_utils.get_train_instances_partition(
                        partitioned_path, interact_mat, num_negatives, num_user, num_item)

                    widgets = [" ", " ", progressbar.SimpleProgress(), " ", progressbar.Timer()]
                    for b in progressbar.ProgressBar(widgets=widgets)(range(0, len(user_ids), batch_size)):
                        uids = user_ids[b: b + batch_size]
                        iids = item_ids[b: b + batch_size]
                        las = labels[b: b + batch_size]
                        y1_indi = y1_indicator[b: b + batch_size]
                        y2_indi = y2_indicator[b: b + batch_size]

                        sess.run(optimizer,
                                 feed_dict={
                                     user_index: uids,
                                     item_index: iids,
                                     labels_ph: las,
                                     y1_indicators: y1_indi,
                                     y2_indicators: y2_indi
                                 })
                        if (e % verbose == 0):
                            rloss_tmp = sess.run(loss,
                                                 feed_dict={user_index: uids,
                                                            item_index: iids,
                                                            labels_ph: las,
                                                            y1_indicators: y1_indi,
                                                            y2_indicators: y2_indi})
                            rloss += rloss_tmp
                        num_batch += 1
                        # end for
                    # end for
                # Tinh loss phai cat ra batch, ko thi bi loi out of memory
                rloss /= num_batch

                if (e % verbose == 0):
                    # log for explicit
                    # raw_explicit_top = self.predict(model_ite_onehot_log_loss, user_ids, item_ids, prediction_explicit)
                    # dict_explicit_top = {i: raw_explicit_top[i] for i in range(len(raw_explicit_top))}
                    # explicit_top = {i: dict_explicit_top[i] for i in
                    #                 heapq.nlargest(6, dict_explicit_top, key=dict_explicit_top.get)}
                    # explicit_top = {}

                    print("testing........")
                    for k in eval_top_kS:
                      explicit_hit[k], explicit_ndcg[k] = self.evaluate_model(model,
                                                                      k,
                                                                      test_data,
                                                                      negative_data,
                                                                      prediction_explicit)
                      if explicit_hit[k] > explicit_best_hit[k][0]:
                          explicit_best_hit[k] = [explicit_hit[k], str(e)]  # log result
                  
                      if explicit_ndcg[k] > explicit_best_ndcg[k][0]:  # log result
                          explicit_best_ndcg[k] = [explicit_ndcg[k], str(e)]

                    # log result
                    table_data = {"total_loss": rloss,
                                  "eval_explicit_top5": (explicit_hit[5], explicit_ndcg[5]),
                                  "eval_explicit_top10": (explicit_hit[10], explicit_ndcg[10]),
                                  "eval_explicit_top20": (explicit_hit[20], explicit_ndcg[20]),
                                  "eval_explicit_top30": (explicit_hit[30], explicit_ndcg[30]),
                                  "eval_explicit_top40": (explicit_hit[40], explicit_ndcg[40]),
                                  "eval_explicit_top50": (explicit_hit[50], explicit_ndcg[50])}
                    
                    temp = [str(e), rloss]
                    for eval_top_k in eval_top_kS:
                        temp.append(explicit_hit[eval_top_k])
                        temp.append(explicit_ndcg[eval_top_k])

                    result.append(temp)

                    MF.show_result_keyvalue(table_data.items())

                sess.run(model['global_epoch'].assign(e))
                if self.save_model:
                    saver.save(sess, self.file_model + '/model')
                # end of each epoch

        logging.warning("-----------------------Train done ==> RESULT --------------------------------")
        best = {"explicit_best_hit_top5": explicit_best_hit[5], #5
                "explicit_best_ndcg_top5": explicit_best_ndcg[5],
                "explicit_best_hit_top10": explicit_best_hit[10], #10
                "explicit_best_ndcg_top10": explicit_best_ndcg[10],
                "explicit_best_hit_top20": explicit_best_hit[20],
                "explicit_best_ndcg_top20": explicit_best_ndcg[20],
                "explicit_best_hit_top30": explicit_best_hit[30],
                "explicit_best_ndcg_top30": explicit_best_ndcg[30],
                "explicit_best_hit_top40": explicit_best_hit[40],
                "explicit_best_ndcg_top40": explicit_best_ndcg[40],
                "explicit_best_hit_top50": explicit_best_hit[50],
                "explicit_best_ndcg_top50": explicit_best_ndcg[50]}
                
        MF.show_result_keyvalue(self.params.items())
        table_result = AsciiTable(result).table
        print(table_result)
        if self.save_log:
            with open(self.log_path, "w") as log:
                log.write(self.result_string + str(table_result) + "\n\n" + MF.show_result_keyvalue(best.items()))

    def predict(self, model, user, items, prediction):
        users = [user] * len(items)
        return model['sess'].run(prediction, feed_dict={model['user_index']: users, model['item_index']: items,
                                                        model['user_index']: users,
                                                        model['item_index']: items
                                                        })

    # for evaluate model_ite_onehot_log_loss
    def evaluate_model(self, model, top_k, test_data, negative_data, prediction):
        hits, ndcgs = [], []
        # Single thread
        widgets = [progressbar.Percentage(), ' ', progressbar.SimpleProgress(), ' ', progressbar.Timer()]
        for idx in progressbar.ProgressBar(widgets=widgets)(range(len(test_data))):
            hr, ndcg = self.eval_one_rating(model, idx, top_k, test_data, negative_data, prediction)
            hits.append(hr)
            ndcgs.append(ndcg)
        return np.array(hits).mean(), np.array(ndcgs).mean()

    def eval_one_rating(self, model, idx, top_k, test_data, negative_data, prediction):
        rating = test_data[idx]
        user = rating[0]
        gt_item = rating[1]
        items = negative_data[user]
        items.append(gt_item)
        # Get prediction scores
        map_item_score = {}
        # users = np.full(len(items), u, dtype='int64')
        predictions = self.predict(model, user, items, prediction)
        for i in range(len(items)):
            item = items[i]
            map_item_score[item] = predictions[i]
        items.pop()
        # Evaluate top rank list
        rank_list = heapq.nlargest(top_k, map_item_score, key=map_item_score.get)

        hr = self.get_hit_ratio(rank_list, gt_item)
        ndcg = self.get_ndcg(rank_list, gt_item)
        return hr, ndcg

    @staticmethod
    def get_hit_ratio(rank_list, gt_item):
        for item in rank_list:
            if item == gt_item:
                return 1.0
        return 0

    @staticmethod
    def get_ndcg(rank_list, gt_item):
        for i in range(len(rank_list)):
            item = rank_list[i]
            if item == gt_item:
                return math.log(2) / math.log(i + 2)
        return 0
