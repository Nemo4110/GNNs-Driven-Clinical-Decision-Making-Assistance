r""" Get the best threshold using train set data.
"""

import sys
import numpy as np
import os
import torch
import pickle

from sklearn.metrics import roc_curve
from collections import defaultdict
from deprecated import deprecated

sys.path.append('..')


@deprecated
class BestThresholdLogger:
    def __init__(self, max_timestep, save_dir_path: str) -> None:
        self.save_dir_path = save_dir_path

        self.list_preds_each_timestep = [[] for _ in range(max_timestep)]
        self.list_trues_each_timestep = [[] for _ in range(max_timestep)]

        self.dict_best_threholds = {}

    def log(self, list_y_pred, list_y_true):
        for timestep, (y_pred, y_true) in enumerate(zip(list_y_pred, list_y_true)):
            self.list_preds_each_timestep[timestep].append(y_pred.cpu().detach())
            self.list_trues_each_timestep[timestep].append(y_true.cpu().detach())

    def save(self, prefix: str):
        list_preds_each_timestep_concated = [torch.cat(list_preds, dim=0) for list_preds in self.list_preds_each_timestep]
        list_tures_each_timestep_concated = [torch.cat(list_trues, dim=0) for list_trues in self.list_trues_each_timestep]

        # total performance & best threshold
        preds_total_concated = torch.cat(list_preds_each_timestep_concated, dim=0)
        trues_total_concated = torch.cat(list_tures_each_timestep_concated, dim=0)
        fpr, tpr, thresholds = roc_curve(trues_total_concated, preds_total_concated)
        self.dict_best_threholds['total'] = thresholds[np.argmax(tpr - fpr)]  # youden index

        list_best_threshold_each_timestep = []
        for preds_current_timestep, trues_current_timestep in zip(list_preds_each_timestep_concated, list_tures_each_timestep_concated):
            fpr, tpr, thresholds = roc_curve(trues_current_timestep, preds_current_timestep)
            best_threshold_curr_timestep = thresholds[np.argmax(tpr - fpr)]  # youden index
            list_best_threshold_each_timestep.append(best_threshold_curr_timestep)
        self.dict_best_threholds['each_timestep'] = list_best_threshold_each_timestep
        
        with open(os.path.join(self.save_dir_path, f"{prefix}_best_thresholds.pickle"), 'wb') as f:
            pickle.dump(self.dict_best_threholds, f)


class BestThresholdLoggerV2:
    def __init__(self, path_to_save):
        self.best_thresholds_by_days = defaultdict(list)
        self.path_to_save = path_to_save

    def log_cur_batch(self, logits, labels, adm_lens):
        logits, labels = logits.cpu().detach(), labels.cpu().detach()
        B, max_adm_len = logits.size(0), logits.size(1)
        for i in range(B):  # 一个一个患者
            for j in range(adm_lens[i]):  # day
                cur_day = j + 1  # 因为从第二天（idx=1）开始预测
                cur_day_pred = logits[i, j]
                cur_day_true = labels[i, j]
                if cur_day_true.sum() == 0:  # 若这天没有正样本，则跳过
                    continue
                fpr, tpr, thresholds = roc_curve(cur_day_true, cur_day_pred)
                cur_day_best_th = thresholds[np.argmax(tpr - fpr)]
                self.best_thresholds_by_days[cur_day].append(cur_day_best_th)

    def save(self):
        best_thresholds_by_days = {day: sum(l) / len(l) for day, l in self.best_thresholds_by_days.items()}
        with open(os.path.join(self.path_to_save, "bth_by_day.pickle"), 'wb') as f:
            pickle.dump(best_thresholds_by_days, f)
