import sys
import pandas as pd
import numpy as np
import os
import torch
import time

from sklearn.metrics import \
    auc, \
    roc_auc_score, \
    accuracy_score, \
    jaccard_score, \
    f1_score, \
    precision_score, \
    recall_score, \
    average_precision_score
from typing import List


sys.path.append('..')


def flat_indices_to_voc_size(indices: List[int], voc_size, exclude_indices=None) -> np.ndarray:
    if exclude_indices is not None:
        indices = [elm for elm in indices if elm not in exclude_indices]

    tmp = np.zeros(voc_size)
    tmp[indices] = 1

    return tmp


def flat_probs(probs: List[torch.tensor], preds: List[int]):
    tmp = torch.stack([prob[:-3] for prob in probs], dim=0)  # 去掉SOS, EOS, PAD
    tmp = torch.max(tmp, dim=0).values
    # tmp = np.max(tmp, axis=0)  # 对于没预测的药物，取每个位置上最大的概率，否则直接取对应的概率
    for i, pred in enumerate(preds):
        if pred < tmp.size(0):  # 去掉SOS, EOS, PAD
            tmp[pred] = probs[i][pred]

    return tmp


def cal_jaccard(y_true: torch.tensor, y_pred: torch.tensor):
    if y_true.numel() == 0 or y_pred.numel() == 0:
        return 0
    set1 = set(y_true.tolist())
    set2 = set(y_pred.tolist())
    a, b = len(set1 & set2), len(set1 | set2)
    return a / b


def rocauc(cur_day_preds, cur_day_probs, cur_day_labels, **kwargs):
    try:
        return roc_auc_score(cur_day_labels, cur_day_probs, average='macro')
    except ValueError:
        return 0


def accuracy(cur_day_preds, cur_day_probs, cur_day_labels, **kwargs):
    return accuracy_score(cur_day_labels, cur_day_preds)


def jaccard(cur_day_preds, cur_day_probs, cur_day_labels, is_01: bool = False, **kwargs):
    if is_01:
        return jaccard_score(cur_day_labels, cur_day_preds)

    target = np.where(cur_day_labels == 1)[0]
    preds  = np.where(cur_day_preds  == 1)[0]

    inter = set(preds) & set(target)
    union = set(preds) | set(target)
    score = 0 if len(union) == 0 else len(inter) / len(union)

    return score


def prauc(cur_day_preds, cur_day_probs, cur_day_labels, is_01: bool = False, **kwargs):
    return average_precision_score(cur_day_labels, cur_day_probs, average='macro')


def precision(cur_day_preds, cur_day_probs, cur_day_labels, is_01: bool = False, **kwargs):
    if is_01:
        return precision_score(cur_day_labels, cur_day_preds, zero_division=0)
    else:
        target = np.where(cur_day_labels == 1)[0]
        preds  = np.where(cur_day_preds  == 1)[0]
        inter = set(preds) & set(target)
        score = 0 if len(preds) == 0 else len(inter) / len(preds)

        return score


def recall(cur_day_preds, cur_day_probs, cur_day_labels, is_01: bool = False, **kwargs):
    if is_01:
        return recall_score(cur_day_labels, cur_day_preds, zero_division=0)
    else:
        target = np.where(cur_day_labels == 1)[0]
        preds  = np.where(cur_day_preds  == 1)[0]
        inter = set(preds) & set(target)
        score = 0 if len(target) == 0 else len(inter) / len(target)

        return score


def calculate_f1(row):
    prc = row['precision']
    rec = row['recall']
    if prc + rec == 0:
        return 0.0
    return 2 * (prc * rec) / (prc + rec)


def convert2df(logits: torch.tensor, labels: torch.tensor) -> pd.DataFrame:
    each_day_collector = []
    for d, (y_hat, y) in enumerate(zip(logits, labels)):
        y_hat = y_hat.sigmoid().cpu().tolist()
        y = y.cpu().tolist()

        day = d + 1  # 因为是从住院的第二天开始预测的，以此偏移量为1
        curr_day_df = pd.DataFrame({
            'score': y_hat,
            'label': y,
            'day': [day] * len(y)
        })
        each_day_collector.append(curr_day_df)
    return pd.concat(each_day_collector, axis=0)


def calc_metrics(results):
    return {
        "rocauc":       roc_auc_score(results['label'].values, results['score'].values, average='macro'),
        "ap": average_precision_score(results['label'].values, results['score'].values, average='macro'),
        "acc":         accuracy_score(results['label'].values, (results['score'] > 0.5).values),
        "precision":  precision_score(results['label'].values, (results['score'] > 0.5).values),
        "recall":        recall_score(results['label'].values, (results['score'] > 0.5).values, zero_division=0),
        "f1_score":          f1_score(results['label'].values, (results['score'] > 0.5).values, zero_division=0)
    }


def save_results(path_dir_results, results, ckpt_filename, notes):
    assert notes is not None  # 实验备注必须填写！
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    new_row = {
        "timestamp": timestamp,
        "ckpt_filename": ckpt_filename,
        "notes": notes,
        **calc_metrics(results)
    }
    result_file = os.path.join(path_dir_results, "results.csv")
    if os.path.exists(result_file):
        df_results = pd.read_csv(result_file, index_col=0)
        df_results = pd.concat([df_results, pd.DataFrame(new_row, index=[0])], ignore_index=True)
    else:
        df_results = pd.DataFrame(new_row, index=[0])
    df_results.to_csv(result_file, index=False)
