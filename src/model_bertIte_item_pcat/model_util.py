import progressbar
import math
import torch 
import heapq
import numpy as np


def get_hit_ratio(rank_list, gt_item):
    for item in rank_list:
        if item == gt_item:
            return 1.0
    return 0

def get_ndcg(rank_list, gt_item):
    for i in range(len(rank_list)):
        item = rank_list[i]
        if item == gt_item:
            return math.log(2) / math.log(i + 2)
    return 0

def evaluate_model(model, top_k, data, device):
    hits, ndcgs = [], []
    widgets = [progressbar.Percentage(), " ", progressbar.SimpleProgress(), " ", progressbar.Timer()]

    for idx in progressbar.ProgressBar(widgets=widgets)(range(len(data))):
        hr, ndcg = eval_one_rating(model, idx, top_k, data[idx], device)
        hits.append(hr)
        ndcgs.append(ndcg)
    return np.array(hits).mean(), np.array(ndcgs).mean()

def eval_one_rating(model, idx, top_k, data, device):
    # model = model.to("cpu")
    model.to(device)
    # user_ids, items_sequences, target_ids, user_reps, item_seq_reps, item_target_reps, labels, items, attn_masks = \
    #         data["user_ids"], data["item_sequences"], data["target_ids"], data["user_reps"], data["item_seq_reps"], data["item_target_reps"],\
    #              data["labels"], data["items"], data["attn_masks"]
    user_ids, items_sequences, target_ids, labels, items, attn_masks = data["user_ids"],\
                   data["items_sequences"], data["target_ids"], data["labels"], data["items"], data["attn_masks"]
    user_ids = torch.tensor(user_ids, dtype=torch.long).to(device)
    items_sequences = torch.tensor(items_sequences, dtype=torch.long).to(device)
    target_ids = torch.tensor(target_ids, dtype=torch.long).to(device)
    attn_masks = torch.tensor(attn_masks, dtype=torch.long).to(device)
    test_item = target_ids[-1]
    model.eval()
    # click, action = model.forward(user_ids, items_sequences, target_ids, user_reps, item_seq_reps, item_target_reps, attn_masks)
    click, action = model.forward(user_ids, items_sequences, target_ids, attn_masks)

    rating = torch.mul(click, action).detach().cpu().numpy()
    # print(rating)
    map_score_item = {}
    for i in range(len(user_ids)):
        item = items[i]
        # print(item)
        map_score_item[item] = rating[i]
    list_hr, list_ndcg = [], []
    for tk in top_k:
        rank_list = heapq.nlargest(tk, map_score_item, key=map_score_item.get)

        hr = get_hit_ratio(rank_list, test_item)
        ndcg = get_ndcg(rank_list, test_item)
        list_hr.append(hr)
        list_ndcg.append(ndcg)
    return list_hr, list_ndcg


def evaluate_model_ver3(model, top_k, data, device):
    hits, ndcgs = [], []
    reshit, resndcg = [], []
    for tk in top_k:
        hits.append([])
        ndcgs.append([])
    widgets = [progressbar.Percentage(), " ", progressbar.SimpleProgress(), " ", progressbar.Timer()]

    for idx in progressbar.ProgressBar(widgets=widgets)(range(len(data))):
        lst_hr, lst_ndcg = eval_one_rating_ver3(model, idx, top_k, data[idx], device)
        for i, (hr, ndcg) in enumerate(zip(lst_hr, lst_ndcg)):
            hits[i].append(hr)
            ndcgs[i].append(ndcg)
    for lh, ln in zip(hits, ndcgs):
        reshit.append(np.array(lh).mean())
        resndcg.append(np.array(ln).mean())
    return reshit, resndcg

def eval_one_rating_ver3(model, idx, top_k, data, device):
    # model = model.to("cpu")
    model.to(device)
    user_ids, items_sequences, target_ids, attn_masks, labels, items = data["user_ids"], data["items_sequences"],\
                                                                data["target_ids"], data["attn_masks"], data["labels"], data["items"]
    user_ids = torch.tensor(user_ids, dtype=torch.long).to(device)
    items_sequences = torch.tensor(items_sequences, dtype=torch.long).to(device)
    target_ids = torch.tensor(target_ids, dtype=torch.long).to(device)
    attn_masks = torch.tensor(attn_masks, dtype=torch.long).to(device)
    test_item = target_ids[-1]
    model.eval()
    click, action = model.forward(user_ids, items_sequences, target_ids, attn_masks)
    rating = torch.mul(click, action).detach().cpu().numpy()
    # print(rating)
    map_score_item = {}
    for i in range(len(user_ids)):
        item = items[i]
        # print(item)
        map_score_item[item] = rating[i]
    
    list_hr, list_ndcg = [], []
    for tk in top_k:
        rank_list = heapq.nlargest(tk, map_score_item, key=map_score_item.get)

        hr = get_hit_ratio(rank_list, test_item)
        ndcg = get_ndcg(rank_list, test_item)
        list_hr.append(hr)
        list_ndcg.append(ndcg)
    return list_hr, list_ndcg