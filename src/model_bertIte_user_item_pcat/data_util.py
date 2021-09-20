import csv
import datetime
import scipy.sparse as sp
import numpy as np
import progressbar
import random
from ast import literal_eval
import time

class Data_Utils:
    def __init__(self, train_path, root_path, config):
        self.train_path = train_path
        self.config = config
        
        self.num_user, self.num_item = self.load_representation_data(
            root_path + "u2index.txt",
            root_path + "i2index.txt")
        self.interact_mat = self.load_interact_matrix_s_ite(root_path + "without_implicit_in_train/ratings_train.txt", self.num_user,
                                                             self.num_item)
        self.max_len = config.max_len
        self.num_neg = config.num_neg
        self.item_representation, self.dimension = self.get_item_rep(config.path_rep_item)
        self.user_representation = self.get_user_rep_full(self.item_representation, self.get_dict_array_item_train(), self.dimension)

    def preprocess_test(self, test_data, negative_data):
        data = []
        dict_arritem_ids = self.get_dict_array_item_train()
        widgets = [progressbar.Percentage(), " ", progressbar.SimpleProgress(), " ", progressbar.Timer()]
        for idx in progressbar.ProgressBar(widgets=widgets)(range(len(test_data))):
            uid = test_data[idx][0]
            test_item = test_data[idx][1]
            neg_items = negative_data[uid]
            arr_items = dict_arritem_ids[uid]
            items = neg_items + [test_item]
            # user_ids, items_sequences, target_ids, user_reps, item_seq_reps, item_target_reps, attn_masks, labels = \
            #         self.raise_data_test_ver2(uid, arr_items, test_item, neg_items)
            # user_ids, items_sequences, target_ids, attn_masks, labels = \
            #         self.raise_data_test_ver2(uid, arr_items, test_item, neg_items)
            user_ids, items_sequences, target_ids, attn_masks, labels = \
                  self.raise_data_test_no_concat_test(uid, arr_items, test_item, neg_items)
            data.append({
                "user_ids": user_ids,
                "items_sequences": items_sequences,
                "target_ids": target_ids,
                # "user_reps": user_reps,
                # "item_seq_reps": item_seq_reps,
                # "item_target_reps": item_target_reps,
                "attn_masks": attn_masks,
                "labels": labels,
                "items": items
            })
        return data

    def get_dict_array_item_train(self):
        dict_arritem_ids = {}
        with open(self.train_path) as f:
            for line in progressbar.ProgressBar()(f.readlines()):
                elements = line.split('|')
                uid = int(elements[0].strip())

                itemids = elements[1].strip()[1:-1]
                itemids = itemids.split(',')
                arr_items = []
                for item in itemids:
                    arr_items.append(int(item.strip()))
                dict_arritem_ids[uid] = arr_items
        return dict_arritem_ids

    def load_representation_data(self, u2index_path, i2index_path):
        count = 0
        with open(u2index_path) as f:
            csv_reader = csv.reader(f, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
            for row in csv_reader:
                count += 1
        num_user = count
        print('Num user: ', num_user)

        count = 0
        with open(i2index_path) as f:
            csv_reader = csv.reader(f, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
            for row in csv_reader:
                count += 1
        num_item = count
        print('Num item: ', num_item)

        return num_user, num_item

    def load_interact_matrix_s_ite(self, file_path, num_user, num_item):
        start = datetime.datetime.now()
        # Construct matrix
        # ma tran thua voi num_user x num_item , init value la false.
        mat = sp.dok_matrix((num_user, num_item), dtype=np.bool_)

        with open(file_path, "r") as f:
            csv_reader = csv.reader(f, delimiter='|', quotechar='', quoting=csv.QUOTE_NONE)
            for line in csv_reader:
                uid = int(line[0])
                itemids = line[1].strip()[1:-1]
                itemids = itemids.split(",")
                for item in itemids:
                    itemid = int(item.strip())
                    mat[uid, itemid] = True
        print("time load_interact_matrix_s_ite: ", datetime.datetime.now() - start)
        return mat

    def load_negative_data(self, file_path):
        negative_dict = {}
        with open(file_path, "r") as f:
            csv_reader = csv.reader(f, delimiter='|', quotechar='', quoting=csv.QUOTE_NONE)
            # i = 0
            # raw data: (4,2504)|15383|6979|41741|79116|53192|12932|29099|
            for line in csv_reader:
                # line: ['(4,2504)', '15383', '6979', '41741', '79116', '53192', '12932',..]
                user = line[0].split(",")  # ['(158', '4966)']
                user = int(user[0][1:])  # 158
                # assert user == i
                # i += 1

                negative_dict[user] = []  # doi voi moi user co 999 item tuong ung.
                for x in line[1:]:  # danh sach cac item tuong ung voi user do khong co tuong tac explicit.
                    negative_dict[user].append(int(x))
        print('len negative data: ', len(negative_dict))
        return negative_dict  # user_id: list negative item.

    def load_test_data_s_ite(self, file_path):
        rating_list = []
        with open(file_path, "r") as f:
            csv_reader = csv.reader(f, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
            for line in csv_reader:
                user, item = int(line[0]), int(line[1])  # get uid, itemid
                rating_list.append([user, item])
        print('Length test data: ', len(rating_list))
        return rating_list

    def raise_data_test_ver2(self, uid, arr_items, test_item, negative_item):
        user_ids = []
        items_sequences = []
        attn_masks = []
        target_ids = negative_item + [test_item]
        labels = [0] * len(negative_item) + [1]
        k = len(arr_items)
        if k >= self.max_len:
            item_seq_truth = arr_items[1: self.max_len]
            attn_mask = [1]
            for i in range(self.max_len):
                attn_mask.append(1)
            for i in range(len(target_ids)):
                user_ids.append(uid)
                item_seq_pred = item_seq_truth + [target_ids[i]]
                attn_masks.append(attn_mask)
                items_sequences.append(item_seq_pred)
        else:
            item_seq_truth = []
            attn_mask = [1]
            for i in range(k):
                attn_mask.append(1)
                item_seq_truth.append(arr_items[i])
            attn_mask.append(1)
            for i in range(k + 1, self.max_len):
                attn_mask.append(0)
            for i in range(len(target_ids)):
                user_ids.append(uid)
                # user_reps.append(self.user_representation[uid])
                item_seq_pred = item_seq_truth + [target_ids[i]]
                # isr = [self.item_representation[x] for x in item_seq_pred]
                for j in range(k + 1, self.max_len):
                    item_seq_pred.append(self.num_item)
                    # isr.append(self.item_representation[-1])
                items_sequences.append(item_seq_pred)
                attn_masks.append(attn_mask)
        return user_ids, items_sequences, target_ids, attn_masks, labels

    def raise_data_ver2(self, uid, arr_items, arr_interacts):
        user_ids = []
        items_sequences = []
        target_ids = []
        # user_reps = []
        # item_seq_reps = []
        # item_target_reps = []
        im_labels = []
        ex_labels = []
        attn_masks = []
        k = min(len(arr_interacts), len(arr_items))
        if k >= self.max_len:
            for i in range(self.max_len):
                # tạo từng điểm dữ liệu 1
                attn_mask = [1]
                # mask như nhau trong mỗi chuỗi dữ liệu
                for j in range(i + 1):
                    attn_mask.append(1)
                for j in range(i+1, self.max_len):
                    attn_mask.append(0)
                item_truth = [arr_items[j] for j in range(i+1)]
                # item_truth_rep = [self.item_representation[arr_items[j]] for j in range(i+1)]
                for j in range(i+1, self.max_len):
                    item_truth.append(self.num_item)
                    # item_truth_rep.append(self.item_representation[-1])
                target_ids.append(arr_items[i])
                # item_target_reps.append(self.item_representation[arr_items[i]])
                items_sequences.append(item_truth)
                # item_seq_reps.append(item_truth_rep)
                im_labels.append(1)
                user_ids.append(uid)
                # user_reps.append(self.user_representation[uid])
                attn_masks.append(attn_mask)
                if arr_interacts[i]:
                    ex_labels.append(1)
                else:
                    ex_labels.append(0)
                for j in range(self.num_neg):
                    user_ids.append(uid)
                    # user_reps.append(self.user_representation[uid])
                    im_labels.append(0)
                    ex_labels.append(0)
                    item_seq = [arr_items[j] for j in range(i)]
                    # item_neg_seq_reps = [self.item_representation[arr_items[j]] for j in range(i)]
                    x = random.randrange(self.num_item)
                    while (uid, x) in self.interact_mat:
                        x = random.randrange(self.num_item)
                    item_seq.append(x)
                    # item_neg_seq_reps.append(self.item_representation[x])
                    for kk in range(i+1, self.max_len):
                        item_seq.append(self.num_item)
                        # item_neg_seq_reps.append(self.item_representation[-1])
                    items_sequences.append(item_seq)
                    # item_seq_reps.append(item_neg_seq_reps)
                    target_ids.append(x)
                    # item_target_reps.append(self.item_representation[x])
                    attn_masks.append(attn_mask)
        else:
            for i in range(k):
                # tạo từng điểm dữ liệu 1
                attn_mask = [1]
                # mask như nhau trong mỗi chuỗi dữ liệu
                for j in range(i + 1):
                    attn_mask.append(1)
                for j in range(i+1, self.max_len):
                    attn_mask.append(0)
                item_truth = [arr_items[j] for j in range(i+1)]
                # item_truth_rep = [self.item_representation[arr_items[j]] for j in range(i+1)]
                for j in range(i+1, self.max_len):
                    item_truth.append(self.num_item)
                    # item_truth_rep.append(self.item_representation[-1])
                target_ids.append(arr_items[i])
                # item_target_reps.append(self.item_representation[arr_items[i]])
                items_sequences.append(item_truth)
                # item_seq_reps.append(item_truth_rep)
                im_labels.append(1)
                user_ids.append(uid)
                # user_reps.append(self.user_representation[uid])
                attn_masks.append(attn_mask)
                if arr_interacts[i]:
                    ex_labels.append(1)
                else:
                    ex_labels.append(0)
                for j in range(self.num_neg):
                    user_ids.append(uid)
                    # user_reps.append(self.user_representation[uid])
                    im_labels.append(0)
                    ex_labels.append(0)
                    item_seq = [arr_items[j] for j in range(i)]
                    # item_neg_seq_reps = [self.item_representation[arr_items[j]] for j in range(i)]
                    x = random.randrange(self.num_item)
                    while (uid, x) in self.interact_mat:
                        x = random.randrange(self.num_item)
                    item_seq.append(x)
                    # item_neg_seq_reps.append(self.item_representation[x])
                    for kk in range(i+1, self.max_len):
                        item_seq.append(self.num_item)
                        # item_neg_seq_reps.append(self.item_representation[-1])
                    items_sequences.append(item_seq)
                    # item_seq_reps.append(item_neg_seq_reps)
                    target_ids.append(x)
                    # item_target_reps.append(self.item_representation[x])
                    attn_masks.append(attn_mask)

        return user_ids, items_sequences, target_ids, ex_labels, im_labels, attn_masks

    def get_item_rep(self, path):
        with open(path) as f:
            data = f.read()

        data = data.split("\n")
        dimension = int(data[0].strip())
        data = data[1:-1]
        item_repr = {}
        for line in data:
            parts = line.split("|,|")
            id = int(parts[0][1:].strip())
            arr = parts[1][:-1].strip()
            arr = literal_eval(arr)
            pos = [a[0] for a in arr]
            vl = [a[1] for a in arr]
            dense_arr = np.zeros(shape=(dimension,))
            for i in range(len(pos)):
                # print(i)
                dense_arr[pos[i]] = vl[i]
            item_repr[id] = dense_arr
        X = []
        for i in range(self.num_item):
            X.append(item_repr[i])
        return X, dimension

    def get_user_rep_full(self, item_repr, dict_arritem_ids, dimension):
        user_repr = []
        item_repr = np.array(item_repr)
        for i in progressbar.ProgressBar()(range(self.num_user)):
            u_rep = np.zeros(dimension)
            items = np.array(dict_arritem_ids[i])
            if len(items) != 0:
                i_reps_of_u_rep = item_repr[items]
                u_rep += np.sum(i_reps_of_u_rep, axis=0)
            user_repr.append(u_rep)
        return user_repr

    def get_user_rep(self, path, dimension):
        with open(path) as f:
            data = f.read()

        data = data.split("\n")
        # dimension = int(data[0].strip())
        data = data[:-1]
        user_repr = {}
        for line in data:
            parts = line.split("|,|")
            id = int(parts[0][1:].strip())
            arr = parts[1][:-1].strip()
            arr = literal_eval(arr)
            pos = [a[0] for a in arr]
            vl = [a[1] for a in arr]
            dense_arr = np.zeros(shape=(dimension,))
            for i in range(len(pos)):
                # print(i)
                dense_arr[pos[i]] = vl[i]
            user_repr[id] = dense_arr
        return user_repr

    def preprocess_data(self):
        users_ids = []
        items_seq = []
        targets_ids = []
        implicit_labels = []
        # lst_user_reps = []
        # lst_item_seq_reps = []
        # lst_item_target_reps = []
        explicit_labels = []
        attention_masks = []
        with open(self.train_path) as f:
            for line in progressbar.ProgressBar()(f.readlines()):
                elements = line.split('|')
                uid = int(elements[0].strip())

                itemids = elements[1].strip()[1:-1]
                itemids = itemids.split(',')
                arr_items = []
                for item in itemids:
                    arr_items.append(int(item.strip()))
                
                interacts = elements[2].strip()[1:-1]
                interacts = interacts.split(',')
                arr_interacts = []
                for interact in interacts:
                    arr_interacts.append(int(interact.strip()))
                # print(arr_interacts)
                # user_ids, item_sequences, target_ids, user_reps, item_seq_reps, item_target_reps, ex_labels, im_labels, attn_masks = \
                #             self.raise_data_ver2(uid, arr_items, arr_interacts)
                # user_ids, item_sequences, target_ids, ex_labels, im_labels, attn_masks = \
                #             self.raise_data_ver2(uid, arr_items, arr_interacts)
                user_ids, item_sequences, target_ids, ex_labels, im_labels, attn_masks = \
                          self.raise_data_no_concat_test(uid, arr_items, arr_interacts)
                users_ids.extend(user_ids)
                items_seq.extend(item_sequences)
                targets_ids.extend(target_ids)
                # lst_user_reps.extend(user_reps)
                # lst_item_seq_reps.extend(item_seq_reps)
                # lst_item_target_reps.extend(item_target_reps)
                implicit_labels.extend(im_labels)
                explicit_labels.extend(ex_labels)
                attention_masks.extend(attn_masks)
        # return users_ids, items_seq, targets_ids, lst_user_reps, lst_item_seq_reps,\
        #          lst_item_target_reps, implicit_labels, explicit_labels, attention_masks
        return users_ids, items_seq, targets_ids, implicit_labels, explicit_labels, attention_masks

    def raise_data_no_concat_test(self, uid, arr_items, arr_interacts):
        user_ids = []
        items_sequences = []
        target_ids = []
        im_labels = []
        ex_labels = []
        attn_masks = []
        k = min(len(arr_interacts), len(arr_items))
        if k >= self.max_len:
            for i in range(self.max_len):
                # tạo từng điểm dữ liệu
                attn_mask = [1]
                for j in range(i):
                    attn_mask.append(1)
                for j in range(i, self.max_len):
                    attn_mask.append(0)
                item_truth = [arr_items[j] for j in range(i)]
                for j in range(i, self.max_len):
                    item_truth.append(self.num_item)
                target_ids.append(arr_items[i])
                items_sequences.append(item_truth)
                im_labels.append(1)
                user_ids.append(uid)
                attn_masks.append(attn_mask)
                if arr_interacts[i]:
                    ex_labels.append(1)
                else:
                    ex_labels.append(0)
                for j in range(self.num_neg):
                    user_ids.append(uid)
                    im_labels.append(0)
                    ex_labels.append(0)
                    # item_seq = [arr_items[j] for j in range(i)]
                    x = random.randrange(self.num_item)
                    while (uid, x) in self.interact_mat:
                        x = random.randrange(self.num_item)
                    items_sequences.append(item_truth)
                    target_ids.append(x)
                    attn_masks.append(attn_mask)
        else:
            for i in range(k):
                attn_mask = [1]
                for j in range(i):
                    attn_mask.append(1)
                for j in range(i, self.max_len):
                    attn_mask.append(0)
                item_truth = [arr_items[j] for j in range(i)]
                for j in range(i, self.max_len):
                    item_truth.append(self.num_item)
                target_ids.append(arr_items[i])
                items_sequences.append(item_truth)
                im_labels.append(1)
                user_ids.append(uid)
                attn_masks.append(attn_mask)
                if arr_interacts[i]:
                    ex_labels.append(1)
                else:
                    ex_labels.append(0)
                for j in range(self.num_neg):
                    user_ids.append(uid)
                    im_labels.append(0)
                    ex_labels.append(0)
                    x = random.randrange(self.num_item)
                    while (uid, x) in self.interact_mat:
                        x = random.randrange(self.num_item)
                    items_sequences.append(item_truth)
                    target_ids.append(x)
                    attn_masks.append(attn_mask)
        return user_ids, items_sequences, target_ids, ex_labels, im_labels, attn_masks

    def raise_data_test_no_concat_test(self, uid, arr_items, test_item, negative_item):
        uid_ids = []
        items_sequences = []
        attn_masks = []
        target_ids = negative_item + [test_item]
        labels = [0] * len(negative_item) + [1]
        k = len(arr_items)
        if k >= self.max_len:
            item_seq_truth = arr_items[:self.max_len]
            attn_mask = [1]
            for i in range(self.max_len):
                attn_mask.append(1)
            for i in range(len(target_ids)):
                uid_ids.append(uid)
                items_sequences.append(item_seq_truth)
                attn_masks.append(attn_mask)
        else:
            item_seq_truth = arr_items
            for j in range(k, self.max_len):
                item_seq_truth.append(self.num_item)
            attn_mask = [1]
            for i in range(k):
                attn_mask.append(1)
            for i in range(k, self.max_len):
                attn_mask.append(0)
            for i in range(len(target_ids)):
                uid_ids.append(uid)
                items_sequences.append(item_seq_truth)
                attn_masks.append(attn_mask)

            # print("uid:", np.shape(np.array(uid_ids)))
            # print("item sequence: ", np.shape(np.array(items_sequences)))
            # print("target ids:", np.shape(np.array(target_ids)))
            # print("attn mask: ", np.shape(np.array(attn_masks)))
            # print("labels: ", np.shape(np.array(labels)))

        return uid_ids, items_sequences, target_ids, attn_masks, labels
