import sys

import tensorflow as tf

from src import settings
from src.model_ite_onehot_log_loss import mf


class ImToEx(mf.MF):
    @staticmethod
    def get_place_holder():
        user_index = tf.placeholder(dtype=tf.int64, name='user_index')
        item_index = tf.placeholder(dtype=tf.int64, name='item_index')

        return user_index, item_index

    @staticmethod
    def get_embedding_weight(num_user, num_item, num_factors):
        """
        num_factors: number of factors in the last hidden layer of GMF and MLP part
        Refer to model_ite_onehot_log_loss architecture for better understand the values of num_factors_gmf and num_factors_mlp
        """

        num_factors_gmf = num_factors
        num_factors_mlp = 2 * num_factors

        gmf_embedding_weight_user_onehot = tf.Variable(
            tf.random_normal([num_user, num_factors_gmf]) * tf.sqrt(2 / num_factors_gmf),
            name='gmf_embedding_weight_user_onehot')
        gmf_embedding_weight_item_onehot = tf.Variable(
            tf.random_normal([num_item, num_factors_gmf]) * tf.sqrt(2 / num_factors_gmf),
            name='gmf_embedding_weight_item_onehot')
        mlp_embedding_weight_user_onehot = tf.Variable(
            tf.random_normal([num_user, num_factors_mlp]) * tf.sqrt(2 / num_factors_mlp),
            name='gmf_embedding_weight_user_onehot')
        mlp_embedding_weight_item_onehot = tf.Variable(
            tf.random_normal([num_item, num_factors_mlp]) * tf.sqrt(2 / num_factors_mlp),
            name='gmf_embedding_weight_item_onehot')

        return {'gmf_user_onehot': gmf_embedding_weight_user_onehot,
                'gmf_item_onehot': gmf_embedding_weight_item_onehot,
                'mlp_user_onehot': mlp_embedding_weight_user_onehot,
                'mlp_item_onehot': mlp_embedding_weight_item_onehot}

    def create_model(self):
        # custom params
        num_user = self.params['num_user']
        num_item = self.params['num_item']
        # user_pcat_dimension = self.params['user_pcat_dimension']
        # item_pcat_dimension = self.params['item_pcat_dimension']
        learning_rate = self.params['learning_rate']
        num_factors = self.params['num_factors']
        qlambda = self.params['lambda']
        eta_1 = self.params['eta_1']
        eta_2 = self.params['eta_2']
        batch_size = self.params['batch_size']
        # user_dense_shape = [300, 188]
        # item_dense_shape = [64, 188]
        global_epoch = tf.Variable(0, dtype=tf.int64, name='global_epoch')

        with tf.device('/gpu:0'):
            print('Chay voi GPU----------------------------------------------------->>>>>>')
            user_index, item_index = ImToEx.get_place_holder()
            embedding_weight = ImToEx.get_embedding_weight(num_user, num_item, num_factors)

            # -------------------------------- GMF part -------------------------------

            gmf_pu_onehot = tf.nn.embedding_lookup(embedding_weight['gmf_user_onehot'], user_index,
                                                   name="gmf_pu_onehot")
            gmf_qi_onehot = tf.nn.embedding_lookup(embedding_weight['gmf_item_onehot'], item_index,
                                                   name="gmf_qi_onehot")
            gmf_pu = tf.identity(gmf_pu_onehot, name='gmf_pu')
            gmf_qi = tf.identity(gmf_qi_onehot, name='gmf_qi')

            gmf_phi = tf.multiply(gmf_pu, gmf_qi, name='gmf_phi')
            gmf_h = tf.Variable(tf.random_uniform([num_factors, 1], minval=-1, maxval=1), name='gmf_h')

            # --------------------------------- MLP part --------------------------------
            mlp_pu_onehot = tf.nn.embedding_lookup(embedding_weight['mlp_user_onehot'], user_index,
                                                   name="mlp_pu_onehot")
            mlp_qi_onehot = tf.nn.embedding_lookup(embedding_weight['mlp_item_onehot'], item_index,
                                                   name="mlp_qi_onehot")

            mlp_pu = tf.identity(mlp_pu_onehot, name='mlp_pu')
            mlp_qi = tf.identity(mlp_qi_onehot, name='mlp_qi')

            mlp_weights = {
                'w1': tf.Variable(tf.random_normal([4 * num_factors, 2 * num_factors]) * tf.sqrt(1 / num_factors),
                                  name='mlp_weight1'),
                'w2': tf.Variable(tf.random_normal([2 * num_factors, num_factors]) * tf.sqrt(2 / num_factors),
                                  name='mlp_weight2'),
                'h': tf.Variable(tf.random_uniform([num_factors, 1], minval=-1, maxval=1), name='mlp_h')
            }
            mlp_biases = {
                'b1': tf.Variable(tf.random_normal([2 * num_factors]), name='mlp_bias1'),
                'b2': tf.Variable(tf.random_normal([num_factors]), name='mlp_bias2')
            }

            mlp_phi_1 = tf.concat([mlp_pu, mlp_qi], axis=-1, name='mlp_phi1')
            mlp_phi_2 = tf.nn.leaky_relu(tf.add(tf.matmul(mlp_phi_1, mlp_weights['w1']), mlp_biases['b1']),
                                         name='mlp_phi2')
            mlp_phi_3 = tf.nn.leaky_relu(tf.add(tf.matmul(mlp_phi_2, mlp_weights['w2']), mlp_biases['b2']),
                                         name='mlp_phi3')

            # --------------------------------- implicit part ------------------------------------
            # 1 x 2*num_factors
            im_phi = tf.concat([gmf_phi, mlp_phi_3], axis=1, name='im_phi')
            # 2*num_factors x 1
            h_implicit = tf.concat([gmf_h, mlp_weights['h']], axis=0, name='h_implicit')
            # tf.squeeze() 1 x 1
            im_prediction = tf.squeeze(tf.matmul(im_phi, h_implicit), name='prediction_implicit')

            # --------------------------------- explicit part ------------------------------------
            ex_weights = {
                'w1': tf.Variable(tf.random_normal([2 * num_factors, num_factors]) * tf.sqrt(2 / num_factors),
                                  name='ex_weight1'),
                'h': tf.Variable(tf.random_uniform([num_factors, 1], minval=-1, maxval=1), name='h_explicit')
            }
            ex_biases = {
                'b1': tf.Variable(tf.random_normal([num_factors]), name='ex_bias1'),
            }
            # 1 x num_factors
            ex_phi = tf.nn.leaky_relu(tf.add(tf.matmul(im_phi, ex_weights['w1']), ex_biases['b1']), name='ex_phi')
            train_ex_prediction = tf.squeeze(tf.matmul(ex_phi, ex_weights['h']), name='train_prediction_explicit')

            # ex_prediction = tf.squeeze(tf.multiply(im_prediction, train_ex_prediction), name='prediction_explicit')
            ex_prediction = tf.squeeze(tf.multiply(tf.nn.sigmoid(im_prediction), tf.nn.sigmoid(train_ex_prediction)),
                                       name='prediction_explicit')
            # ex_prediction = tf.squeeze(tf.nn.sigmoid(train_ex_prediction), name='prediction_explicit')
            '''
            # ---------------------------------- square loss ---------------------------------------------
            labels = tf.placeholder(tf.float32, shape=[None], name='labels')
            y1_indicators = tf.placeholder(tf.float32, shape=[None], name='y1_indicators')
            y2_indicators = tf.placeholder(tf.float32, shape=[None], name='y2_indicators')

            loss_implicit_list = tf.square(tf.subtract(labels, im_prediction), name='y1_loss_list')
            loss_implicit = tf.reduce_mean(tf.multiply(y1_indicators, loss_implicit_list), name='y1_loss')
            loss_explicit_list = tf.square(tf.subtract(labels, train_ex_prediction), name='y2_loss_list')
            loss_explicit = tf.reduce_mean(tf.multiply(y2_indicators, loss_explicit_list), name='y2_loss')

            regularizer = tf.add(tf.add(tf.reduce_mean(tf.square(gmf_pu)), tf.reduce_mean(tf.square(gmf_qi))),
                                 tf.add(tf.reduce_mean(tf.square(mlp_pu)), tf.reduce_mean(tf.square(mlp_qi))),
                                 name='regularizer')

            loss = tf.add(tf.add(tf.multiply(eta_1, loss_implicit), loss_explicit), tf.multiply(qlambda, regularizer),
                          name='loss')
            '''
            # ---------------------------------- log loss ---------------------------------------------
            labels = tf.placeholder(tf.float32, shape=[None], name='labels')
            y1_indicators = tf.placeholder(tf.float32, shape=[None], name='y1_indicators')
            y2_indicators = tf.placeholder(tf.float32, shape=[None], name='y2_indicators')

            loss_implicit_list = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                                         logits=im_prediction,
                                                                         name='y1_loss_list')
            loss_implicit = tf.reduce_mean(tf.multiply(y1_indicators, loss_implicit_list), name='y1_loss')
            loss_explicit_list = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                                         logits=train_ex_prediction,
                                                                         name='y2_loss_list')
            loss_explicit = tf.reduce_mean(tf.multiply(y2_indicators, loss_explicit_list), name='y2_loss')

            regularizer = tf.add(tf.add(tf.reduce_mean(tf.square(gmf_pu)), tf.reduce_mean(tf.square(gmf_qi))),
                                 tf.add(tf.reduce_mean(tf.square(mlp_pu)), tf.reduce_mean(tf.square(mlp_qi))),
                                 name='regularizer')

            loss = tf.add(tf.add(tf.multiply(eta_1, loss_implicit), loss_explicit), tf.multiply(qlambda, regularizer),
                          name='loss')

            # optimize
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, name='optimize')

            # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_implicit, name='optimize')
            # optimizer = tf.train.MomentumOptimizer(0.0001, 0.8).minimize(loss, name='optimize')

            print('--------------->>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<------------------------')
            return {
                'user_index': user_index,
                'item_index': item_index,
                'optimizer': optimizer,
                'labels': labels,
                # 'ex_indicators': ex_indicators,
                'y1_indicators': y1_indicators,
                'y2_indicators': y2_indicators,
                'loss': loss,
                'loss_implicit': loss_implicit,
                'loss_explicit': loss_explicit,
                'train_ex_prediction': train_ex_prediction,
                'prediction_explicit': ex_prediction,
                'prediction_implicit': im_prediction,
                'h_implicit': h_implicit,
                'global_epoch': global_epoch
            }

            


def training_batch_size(batch_size, data_name, save_path_name, epochs, num_negatives, save_log=True, save_model=True):
    num_factor = 8
    eta = 0.5
    root_path = settings.DATA_ROOT_PATH + "site_data/" + data_name
    log_path = root_path + 'log/{}/batch_size/{}_{}_{}_{}'.format(save_path_name, num_factor, batch_size, eta,
                                                                  num_negatives)
    file_model = root_path + 'saved_model/{}/batch_size/{}_{}_{}_{}'.format(save_path_name, num_factor, batch_size, eta,
                                                                            num_negatives)

    params = {'num_factors': num_factor,
              'learning_rate': 0.001,
              'epochs': epochs,
              'num_negatives': num_negatives,
              'batch_size': batch_size,
              'verbose': 10,
              'eval_top_k': [5, 10, 20, 30, 40, 50],
              'lambda': 0.005,
              'eta_1': eta,
              'eta_2': 1.0
              }

    co_neumf = ImToEx(root_path=root_path, params=params, log_path=log_path, file_model=file_model,
                      save_log=save_log,
                      save_model=save_model)
    co_neumf.run()


def training_num_factors(num_factor, data_name, save_path_name, epochs, num_negatives, save_log=True, save_model=True):
    batch_size = 2048
    eta = 0.1
    lr = 0.005
    root_path = settings.DATA_ROOT_PATH + "site_data/" + data_name
    log_path = root_path + 'log/{}/num_factor/{}_{}_{}_{}'.format(save_path_name, num_factor, batch_size, eta,
                                                                  lr)
    file_model = root_path + 'saved_model/{}/num_factor/{}_{}_{}_{}'.format(save_path_name, num_factor, batch_size, eta,
                                                                            lr)

    params = {'num_factors': num_factor,
              'learning_rate': lr,
              'epochs': epochs,
              'num_negatives': num_negatives,
              'batch_size': batch_size,
              'verbose': 10,
              'eval_top_k': [5, 10, 20, 30, 40, 50],
              'lambda': 0.005,
              'eta_1': eta,
              'eta_2': 1.0
              }
    co_neumf = ImToEx(root_path=root_path, params=params, log_path=log_path, file_model=file_model,
                      save_log=save_log,
                      save_model=save_model)
    co_neumf.run()


def training_eta(eta, data_name, save_path_name, epochs, num_negatives, save_log=True, save_model=True):
    num_factor = 8
    batch_size = 512
    root_path = settings.DATA_ROOT_PATH + "site_data/" + data_name
    log_path = root_path + 'log/{}/eta/{}_{}_{}_{}'.format(save_path_name, num_factor, batch_size, eta, num_negatives)
    file_model = root_path + 'saved_model/{}/eta/{}_{}_{}_{}'.format(save_path_name, num_factor, batch_size, eta,
                                                                     num_negatives)

    params = {'num_factors': num_factor,
              'learning_rate': 0.001,
              'epochs': epochs,
              'num_negatives': num_negatives,
              'batch_size': batch_size,
              'verbose': 10,
              'eval_top_k': [5, 10, 20, 30, 40, 50],
              'lambda': 0.005,
              'eta_1': eta,
              'eta_2': 1.0
              }

    co_neumf = ImToEx(root_path=root_path, params=params, log_path=log_path, file_model=file_model,
                      save_log=save_log,
                      save_model=save_model)
    co_neumf.run()
