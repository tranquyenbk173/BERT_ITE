import csv

import matplotlib
import matplotlib.pyplot as plt

from src import settings

root_path = settings.RESULT_ROOT_PATH


def compare_between_models_with_num_factor():
    num_factors = [8, 16, 32, 64]
    batch_sizes = [512] * 5
    etas = [0.5, 0.5, 0.5, 1.0, 0.5]
    model_names = ['one_hot_log_loss', 'item_pcat_log_loss', 'both_concat_onehot', 'NMTR', 'model_mtmf']
    label_names = ['ITE_one_hot', 'ITE_item_pcat', 'ITE_user_item_pcat', 'NMTR', 'MTMF']
    data_names = ['recobell', 'retail_rocket']
    marker_list = ['o', 's', 'v', '*', '.']
    # 'recobell',
    # data = 'lotte'
    # # data = 'movielens-1m'
    # # data = 'movielens-100k'
    # # data = 'recobell'
    # # data = 'retailrocket'
    # # data = 'yes24'

    for data in data_names:
        hit_results = []
        ndcg_results = []
        for z in range(len(model_names)):
            model = model_names[z]
            hit_res = []
            ndcg_res = []
            for factor in num_factors:

                path = root_path + data + '/' + model + '/num_factor/{}_{}_{}'.format(factor, batch_sizes[z], etas[z])
                print(path)
                try:
                    with open(path) as file:
                        for line in file:
                            if '| 50    |' in line:
                                hit = float(line.split('|')[-3].strip())
                                ndcg = float(line.split('|')[-2].strip())
                                hit_res.append(hit)
                                ndcg_res.append(ndcg)
                except FileNotFoundError:
                    print('?')
            hit_results.append(hit_res)
            ndcg_results.append(ndcg_res)
        for re in ndcg_results:
            print(re)

        # --------------------- HIT ----------------------
        plt.figure(num=None, figsize=(10, 10), dpi=120, facecolor='w', edgecolor='k')

        for z in range(len(model_names)):
            plt.plot(num_factors, hit_results[z], marker=marker_list[z], label=label_names[z], linewidth=2.0)

        plt.ylabel('HR@10')
        plt.xlabel('Number factors \n(Learning rate: 0.001; Batch size: 2048; Epochs: 51)')
        plt.title('Dataset: {}'.format(data))

        plt.axis('tight')
        plt.legend(prop={'size': 14})

        # plt.grid()
        # plt.xticks(np.arange(min(num_factors), max(num_factors) + 8, 8))
        plt.xticks(num_factors)
        # --------------------- NDCG -----------------------
        plt.figure(num=None, figsize=(10, 10), dpi=120, facecolor='w', edgecolor='k')
        for z in range(len(model_names)):
            plt.plot(num_factors, ndcg_results[z], marker=marker_list[z], label=label_names[z], linewidth=2.0)
        # plt.grid()

        plt.ylabel('NDCG@10')
        plt.xlabel('Number factors \n(Learning rate: 0.001; Batch size: 2048; Epochs: 51)')
        plt.title('Dataset: {}'.format(data))

        plt.axis('tight')
        plt.legend(prop={'size': 14})

        # plt.grid()
        plt.xticks(num_factors)

        # plt.subplots_adjust(top=0.94,
        #                     bottom=0.1,
        #                     left=0.285,
        #                     right=0.695,
        #                     hspace=0.365,
        #                     wspace=0.215)
    plt.show()


def factor_through_epochs():
    num_factors = [8, 16, 32, 64]
    batch_sizes = [2048] * 5
    etas = [0.5, 0.5, 0.5, 1.0, 0.5]
    model_names = ['one_hot_log_loss', 'item_pcat_log_loss', 'both_concat_onehot', 'NMTR', 'model_mtmf']
    label_names = ['ITE_one_hot', 'ITE_item_pcat', 'ITE_user_item_pcat', 'NMTR', 'MTMF']
    data_names = ['recobell', 'retail_rocket']
    marker_list = ['o', 's', 'v', '*', '.']
    # 'recobell',
    # data = 'lotte'
    # # data = 'movielens-1m'
    # # data = 'movielens-100k'
    # # data = 'recobell'
    # # data = 'retailrocket'
    # # data = 'yes24'

    for data in data_names:
        model_epochs = []
        hit_results = []
        ndcg_results = []
        for factor in num_factors:
            factor_model_e = []
            factor_hit = []
            factor_ndcg = []
            for z in range(len(model_names)):
                model_e = []
                hit_res = []
                ndcg_res = []
                model = model_names[z]
                path = root_path + data + '/' + model + '/num_factor/{}_{}_{}'.format(factor, batch_sizes[z], etas[z])
                # print(path)
                try:
                    with open(path) as file:
                        for line in file:
                            if 'init' not in line:
                                continue
                            else:
                                break
                        next(file)
                        for line in file:
                            print(line)
                            if not line.startswith('+-------'):
                                e = int(line.split('|')[1].strip())
                                print(e)
                                hit = float(line.split('|')[-3].strip())
                                ndcg = float(line.split('|')[-2].strip())
                                model_e.append(e)
                                hit_res.append(hit)
                                ndcg_res.append(ndcg)
                            else:
                                break
                except FileNotFoundError:
                    print('?')
                factor_model_e.append(model_e)
                factor_hit.append(hit_res)
                factor_ndcg.append(ndcg_res)
            hit_results.append(factor_hit)
            ndcg_results.append(factor_ndcg)
            model_epochs.append(factor_model_e)
        for re in ndcg_results:
            print(re)

        # --------------------- HIT ----------------------

        for i in range(len(num_factors)):
            plt.figure(num=None, figsize=(10, 10), dpi=120, facecolor='w', edgecolor='k')
            for z in range(len(model_names)):
                plt.plot(model_epochs[i][z], hit_results[i][z], marker=marker_list[z], label=label_names[z],
                         linewidth=2.0)

                plt.ylabel('HR@10')
                plt.xlabel('Epochs\n(Learning rate: 0.001; Batch size: 2048; Num factors: {})'.format(num_factors[i]))
                plt.title('Dataset: {}'.format(data))

                plt.axis('tight')
                plt.legend(prop={'size': 14})

        for i in range(len(num_factors)):
            plt.figure(num=None, figsize=(10, 10), dpi=120, facecolor='w', edgecolor='k')
            for z in range(len(model_names)):
                plt.plot(model_epochs[i][z], ndcg_results[i][z], marker=marker_list[z], label=label_names[z],
                         linewidth=2.0)

                plt.ylabel('NDCG@10')
                plt.xlabel('Epochs\n(Learning rate: 0.001; Batch size: 2048; Num factors: {})'.format(num_factors[i]))
                plt.title('Dataset: {}'.format(data))

                plt.axis('tight')
                plt.legend(prop={'size': 14})
        # plt.grid()
        # plt.xticks(np.arange(min(num_factors), max(num_factors) + 8, 8))
        # plt.xticks(num_factors)
        # --------------------- NDCG -----------------------
        # plt.subplot(2, 1, 2)
        # for z in range(len(model_names)):
        #     plt.plot(model_epochs[z], ndcg_results[z], marker=marker_list[z], label=label_names[z])
        # # plt.grid()
        #
        # plt.ylabel('NDCG@10')
        # plt.xlabel('Number factors \n(Learning rate: 0.001; Batch size: 2048; Epochs: 51)')
        # plt.title('Data: {}'.format(data))
        #
        # plt.legend()
        # plt.grid()
        # plt.xticks(num_factors)

        # plt.subplots_adjust(top=0.94,
        #                     bottom=0.1,
        #                     left=0.285,
        #                     right=0.695,
        #                     hspace=0.365,
        #                     wspace=0.215)
    plt.show()


def compare_ite_vcc():
    model_names = ['both_concat_embed', 'both_concat_embed_added_zone', 'both_concat_embed_added_zone_and_doc']
    label_names = ['ITE-2', 'ITE-3', 'ITE-4']
    num_factor = 32
    batch_size = 1024
    eta = 0.5
    data_names = 'vccorp'
    marker_list = ['o', 's', 'v']
    list_epoch_18_total = []
    hit_18_total = []
    recall_18_total = []
    list_epoch_19_total = []
    hit_19_total = []
    recall_19_total = []
    for z in range(len(model_names)):
        list_epoch_18 = []
        hit_18 = []
        recall_18 = []
        list_epoch_19 = []
        hit_19 = []
        recall_19 = []
        model = model_names[z]
        path = root_path + data_names + '/log/' + model + '/batch_size/1024.log'
        result = []
        with open(path) as f:
            csv_reader = csv.reader(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_ALL)
            for row in csv_reader:
                result.append(row)

        for row in result[10:25]:
            e = int(row[1])
            hit = float(row[3])
            recall = float(row[4])
            list_epoch_18.append(e)
            hit_18.append(hit)
            recall_18.append(recall)
        list_epoch_18_total.append(list_epoch_18)
        hit_18_total.append(hit_18)
        recall_18_total.append(recall_18)
        for row in result[26:]:
            e = int(row[1])
            hit = float(row[3])
            recall = float(row[4])
            list_epoch_19.append(e)
            hit_19.append(hit)
            recall_19.append(recall)
        list_epoch_19_total.append(list_epoch_19)
        hit_19_total.append(hit_19)
        recall_19_total.append(recall_19)

    plt.figure(num=None, figsize=(10, 10), dpi=120, facecolor='w', edgecolor='k')
    for z in range(len(model_names)):
        plt.plot(list_epoch_18_total[z], hit_18_total[z], marker=marker_list[z], label=label_names[z], linewidth=2.0)

    plt.ylabel('HR@10')
    plt.xlabel('Epochs \n(Learning rate: 0.0005; Batch size: 1024; Eta: 0.5; Num factors: 32)')
    plt.title('Date: 2019/04/18')

    plt.axis('tight')
    plt.legend(prop={'size': 14})

    plt.xticks(list_epoch_18_total[0][:-1] + [70])
    plt.figure(num=None, figsize=(10, 10), dpi=120, facecolor='w', edgecolor='k')

    for z in range(len(model_names)):
        plt.plot(list_epoch_18_total[z], recall_18_total[z], marker=marker_list[z], label=label_names[z], linewidth=2.0)

    plt.ylabel('RECALL@10')
    plt.xlabel('Epochs \n(Learning rate: 0.0005; Batch size: 1024; Eta: 0.5; Num factors: 32)')
    plt.title('Date: 2019/04/18')

    plt.axis('tight')
    plt.legend(prop={'size': 14})

    plt.xticks(list_epoch_18_total[0][:-1] + [70])
    plt.figure(num=None, figsize=(10, 10), dpi=120, facecolor='w', edgecolor='k')

    for z in range(len(model_names)):
        plt.plot(list_epoch_19_total[z], hit_19_total[z], marker=marker_list[z], label=label_names[z], linewidth=2.0)

    plt.ylabel('HR@10')
    plt.xlabel('Epochs \n(Learning rate: 0.0005; Batch size: 1024; Eta: 0.5; Num factors: 32)')
    plt.title('Date: 2019/04/19')

    plt.axis('tight')
    plt.legend(prop={'size': 14})

    plt.xticks(list_epoch_18_total[0][:-1] + [70])
    plt.figure(num=None, figsize=(10, 10), dpi=120, facecolor='w', edgecolor='k')
    for z in range(len(model_names)):
        plt.plot(list_epoch_19_total[z], recall_19_total[z], marker=marker_list[z], label=label_names[z], linewidth=2.0)

    plt.ylabel('RECALL@10')
    plt.xlabel('Epochs \n(Learning rate: 0.0005; Batch size: 1024; Eta: 0.5; Num factors: 32)')
    plt.title('Date: 2019/04/19')

    plt.axis('tight')
    plt.legend(prop={'size': 14})

    plt.xticks(list_epoch_18_total[0][:-1] + [70])
    plt.show()


def main():
    # font = {'weight': 'bold',
    #         'size': 16}
    # matplotlib.rc('font', **font)
    plt.rcParams.update({'font.size': 16})
    # factor_through_epochs()
    # compare_between_models_with_num_factor()
    # show_chart_batch_size()
    # show_chart_eta()

    compare_ite_vcc()


if __name__ == '__main__':
    main()
