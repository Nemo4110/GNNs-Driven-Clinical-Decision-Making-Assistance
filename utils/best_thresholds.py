r""" Get the best threshold using train set data.
"""

import sys
import numpy as np
import os
import torch
import pickle

from sklearn.metrics import roc_curve

sys.path.append('..')


class BestThreshldLogger:
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