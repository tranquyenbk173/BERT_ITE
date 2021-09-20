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
        self.user_repr = None
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
        num_user, num_item, user_repr, item_repr, item_pcat_dimension = data_utils.load_representation_data_with_both_user_item_repr(
            self.root_path + 'u2index.txt',
            self.root_path + 'i2index.txt',
            self.root_path + 'user_repr.txt',
            self.root_path + 'item_repr.txt')
        self.params['num_user'] = num_user
        self.params['num_item'] = num_item
        self.params['item_pcat_dimension'] = item_pcat_dimension
        self.user_repr = user_repr
        self.item_repr = item_repr
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

    @staticmethod
    def get_input(user_pcat, item_pcat):
        ids_user_pcat = [data[0] for data in user_pcat]
        values_user_pcat = [data[1] for data in user_pcat]
        ids_item_pcat = [data[0] for data in item_pcat]
        values_item_pcat = [data[1] for data in item_pcat]

        # for batch input
        batch_user_pcat = {'ids': np.concatenate(ids_user_pcat), 'values': np.concatenate(values_user_pcat)}
        batch_item_pcat = {'ids': np.concatenate(ids_item_pcat), 'values': np.concatenate(values_item_pcat)}

        batch_indices_input = {'user_pcat': [], 'item_pcat': []}
        for i in range(len(ids_user_pcat)):
            batch_indices_input['user_pcat'] += [[i]] * len(ids_user_pcat[i])
            batch_indices_input['item_pcat'] += [[i]] * len(ids_item_pcat[i])

        return (
            (batch_indices_input['user_pcat'], batch_user_pcat['ids'], batch_user_pcat['values']),
            (batch_indices_input['item_pcat'], batch_item_pcat['ids'], batch_item_pcat['values'])
        )

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
        user_repr = self.user_repr
        item_repr = self.item_repr
        # training_dict = data['training_dict']
        interact_mat = data['interact_mat']
        test_data = data['test_data']
        negative_data = data['negative_data']

        # jobs

        optimizer = model['optimizer']
        user_index = model['user_index']
        item_index = model['item_index']
        user_pcat = model['user_pcat']
        item_pcat = model['item_pcat']
        labels_ph = model['labels']
        # ex_indicators = model['ex_indicators']
        y1_indicators = model['y1_indicators']
        y2_indicators = model['y2_indicators']
        # r_target_implicit = model_ite_onehot_log_loss['r_target_implicit']
        # r_target_explicit = model_ite_onehot_log_loss['r_target_explicit']
        loss = model['loss']
        loss_implicit = model['loss_implicit']
        loss_explicit = model['loss_explicit']
        train_ex_prediction = model['train_ex_prediction']
        prediction_implicit = model['prediction_implicit']
        prediction_explicit = model['prediction_explicit']

        loss_implicit = model["loss_implicit"]
        loss_explicit = model["loss_explicit"]
        # ex_parameter_layer_2 = model_ite_onehot_log_loss['ex_parameter_layer_2']
        # mlp_weights_w1 = model_ite_onehot_log_loss['mlp_weights_w1']
        # mlp_phi_2 = model_ite_onehot_log_loss['mlp_phi_2']
        # mlp_phi_3 = model_ite_onehot_log_loss['mlp_phi_3']
        # gmf_pu = model_ite_onehot_log_loss['gmf_pu']
        # gmf_qi = model_ite_onehot_log_loss['gmf_qi']
        # h_implicit = model_ite_onehot_log_loss['h_implicit']
        # mlp_pu = model_ite_onehot_log_loss['mlp_pu']
        # mlp_qi = model_ite_onehot_log_loss['mlp_qi']
        # gmf_embedding_weight_item_lda = model_ite_onehot_log_loss['gmf_embedding_weight_item_lda']
        # gmf_qi_lda = model_ite_onehot_log_loss['gmf_qi_lda']
        # global_epoch = model_ite_onehot_log_loss['global_epoch']

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
            '''
            num_passes = 0

            # tensorboard
            # summary = tf.summary.merge_all()

            # for e in range(epochs):
            self.restore_checkpoint(sess, saver)
            # e = 0
            # while True:
            #     e += 1

            continue_epoch = model_ite_onehot_log_loss['global_epoch'].eval()
            if continue_epoch >= epochs - 1:
                # continue_epoch = 0
                print('Train have done. Exiting program ...')
                return
            print("Start from epoch ", continue_epoch)
            '''
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

            zero_repr = np.array([0], dtype=np.int64), np.array([0], dtype=np.float32)
            for e in range(1, epochs + 1):
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
                        user_repr_pcat = [user_repr[i] if i in user_repr else zero_repr for i in uids]
                        item_repr_pcat = [item_repr[i] for i in iids]
                        input_tensor = self.get_input(user_repr_pcat, item_repr_pcat)

                        sess.run(optimizer,
                                 feed_dict={
                                     user_index: uids,
                                     item_index: iids,
                                     user_pcat[0]: input_tensor[0][0],
                                     user_pcat[1]: input_tensor[0][1],
                                     user_pcat[2]: input_tensor[0][2],
                                     item_pcat[0]: input_tensor[1][0],
                                     item_pcat[1]: input_tensor[1][1],
                                     item_pcat[2]: input_tensor[1][2],
                                     labels_ph: las,
                                     y1_indicators: y1_indi,
                                     y2_indicators: y2_indi
                                 })
                        if (e % verbose == 0):
                            rloss_tmp = sess.run(loss,
                                                 feed_dict={user_index: uids,
                                                            item_index: iids,
                                                            user_pcat[0]: input_tensor[0][0],
                                                            user_pcat[1]: input_tensor[0][1],
                                                            user_pcat[2]: input_tensor[0][2],
                                                            item_pcat[0]: input_tensor[1][0],
                                                            item_pcat[1]: input_tensor[1][1],
                                                            item_pcat[2]: input_tensor[1][2],
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
        user_repr_pcat = [self.user_repr[i] for i in users]
        item_repr_pcat = [self.item_repr[i] for i in items]
        input_tensor = self.get_input(user_repr_pcat, item_repr_pcat)
        return model['sess'].run(prediction, feed_dict={model['user_index']: users, model['item_index']: items,
                                                        model['user_index']: users,
                                                        model['item_index']: items,
                                                        model['user_pcat'][0]: input_tensor[0][0],
                                                        model['user_pcat'][1]: input_tensor[0][1],
                                                        model['user_pcat'][2]: input_tensor[0][2],
                                                        model['item_pcat'][0]: input_tensor[1][0],
                                                        model['item_pcat'][1]: input_tensor[1][1],
                                                        model['item_pcat'][2]: input_tensor[1][2]
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


#  or (e >= (epochs - 5))