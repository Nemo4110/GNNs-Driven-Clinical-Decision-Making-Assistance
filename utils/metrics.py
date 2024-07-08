import sys
import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F
import pickle

from collections import defaultdict
from tqdm import tqdm
from torcheval.metrics import MulticlassAUPRC, MulticlassAUROC, MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from sklearn.metrics import \
    auc, \
    roc_auc_score, \
    accuracy_score, \
    jaccard_score, \
    f1_score, \
    precision_score, \
    recall_score, \
    precision_recall_curve
from utils.ddi import DDICalculator

sys.path.append('..')


def cal_jaccard(y_true: torch.tensor, y_pred: torch.tensor):
    if y_true.numel() == 0 or y_pred.numel() == 0:
        return 0
    set1 = set(y_true)
    set2 = set(y_pred)
    a, b = len(set1 & set2), len(set1 | set2)
    return a / b


class SeqLogger:
    def __init__(self, save_dir_path: str, is_calc_ddi: bool = False):
        self.save_dir_path = save_dir_path
        self.is_calc_ddi = is_calc_ddi

        self.result = defaultdict(list)

        if self.is_calc_ddi:
            self.ddi_calculator = DDICalculator()

    def log(self, scores, labels, *args):
        """
        Args:
            scores: [T, B, X, H]
            labels: [T, B, X]
        """
        self.calc_metrics_for_batch(scores, labels)

    def calc_metrics_for_batch(self, scores: torch.tensor, labels: torch.tensor):
        scores = scores.cpu().permute(1, 0, 2, 3)  # [B, T, X, H]
        labels = labels.cpu().permute(1, 0, 2)  # [B, T, X]

        for cur_adm_scr, cur_adm_lbl in zip(scores, labels):
            # flatten
            cur_adm_scr = cur_adm_scr.contiguous().view(-1, cur_adm_scr.size(-1))  # [T*X, H]
            cur_adm_lbl = cur_adm_lbl.contiguous().view(-1)  # [T*X]

            # filter out special tokens (PAD)
            no_pad_indices = (cur_adm_lbl != cur_adm_scr.size(-1)).int()
            cur_adm_scr = torch.index_select(cur_adm_scr, 0, no_pad_indices)
            cur_adm_lbl = torch.index_select(cur_adm_lbl, 0, no_pad_indices)

            # for solving: number of classes in y_true not equal to the number of columns in 'y_score' when calc rocauc
            cur_adm_scr = cur_adm_scr[:, :-1]  # drop last dimension of PAD

            cur_adm_scr = F.softmax(cur_adm_scr, dim=-1)
            cur_adm_prd = torch.argmax(cur_adm_scr, dim=-1)  # TODO: beam search

            auroc = MulticlassAUROC(num_classes=cur_adm_scr.size(-1))
            auroc.update(cur_adm_scr, cur_adm_lbl)
            self.result['rocauc'].append(auroc.compute().item())

            auprc = MulticlassAUPRC(num_classes=cur_adm_scr.size(-1))
            auprc.update(cur_adm_scr, cur_adm_lbl)
            self.result['prauc'].append(auprc.compute().item())

            accuracy = MulticlassAccuracy()
            accuracy.update(cur_adm_prd, cur_adm_lbl)
            self.result['accuracy'].append(accuracy.compute().item())

            self.result['jaccard'].append(cal_jaccard(y_true=cur_adm_lbl, y_pred=cur_adm_prd))

            precision = MulticlassPrecision(num_classes=cur_adm_scr.size(-1))
            precision.update(cur_adm_prd, cur_adm_lbl)
            self.result['precision'].append(precision.compute().item())

            recall = MulticlassRecall(num_classes=cur_adm_scr.size(-1))
            recall.update(cur_adm_prd, cur_adm_lbl)
            self.result['recall'].append(recall.compute().item())

            f1 = MulticlassF1Score(num_classes=cur_adm_scr.size(-1))
            f1.update(cur_adm_prd, cur_adm_lbl)
            self.result['f1'].append(f1.compute().item())

            if self.is_calc_ddi:
                self.result['ddi_pred'].append(self.ddi_calculator.calc_ddi_rate(cur_adm_prd.unique()))
                self.result['ddi_true'].append(self.ddi_calculator.calc_ddi_rate(cur_adm_lbl.unique()))

    def save(self, description):
        result = {}
        for metric, values in self.result.items():
            result[f'{metric}_mean'] = np.array(values).mean()
            result[f'{metric}_std'] = np.array(values).std()

        with open(os.path.join(self.save_dir_path, f"{description}.pickle"), 'wb') as f:
            pickle.dump(result, f)


class Logger:
    r"""For logging testing metrics.

    Note: <https://math.stackexchange.com/questions/4205313/total-average-of-averages-not-same-as-the-average-of-total-values>
    """
    def __init__(self, max_timestep, save_dir_path: str, best_thresholdspath: str, is_calc_ddi: bool = False):
        self.save_dir_path = save_dir_path
        self.is_calc_ddi = is_calc_ddi

        self.list_preds_each_timestep = [[] for _ in range(max_timestep)]
        self.list_trues_each_timestep = [[] for _ in range(max_timestep)]

        with open(best_thresholdspath, 'rb') as f:
            self.dict_best_threholds = pickle.load(f)

        if self.is_calc_ddi:
            self.ddi_calculator = DDICalculator()
            self.list_edge_indices_each_timestep = [[] for _ in range(max_timestep)]
            self.list_ddi_pred = [[] for _ in range(max_timestep)]
            self.list_ddi_true = [[] for _ in range(max_timestep)]

    def log(self, list_y_pred, list_y_true, list_edge_indices=None):
        if self.is_calc_ddi and list_edge_indices is not None:
            for timestep, edge_indices in enumerate(list_edge_indices):
                self.list_edge_indices_each_timestep[timestep].append(edge_indices.cpu())

        for timestep, (y_pred, y_true) in enumerate(zip(list_y_pred, list_y_true)):
            self.list_preds_each_timestep[timestep].append(y_pred.cpu())
            self.list_trues_each_timestep[timestep].append(y_true.cpu())

    def get_results(self):
        list_preds_each_timestep_concated = [torch.cat(list_preds, dim=0) for list_preds in self.list_preds_each_timestep]
        list_tures_each_timestep_concated = [torch.cat(list_trues, dim=0) for list_trues in self.list_trues_each_timestep]

        # total performance & best threshold
        preds_total_concated = torch.cat(list_preds_each_timestep_concated, dim=0)
        trues_total_concated = torch.cat(list_tures_each_timestep_concated, dim=0)

        dict_total_performance = {}
        dict_total_performance['rocauc'] = roc_auc_score(trues_total_concated, preds_total_concated)
        precision, recall, ____ = precision_recall_curve(trues_total_concated, preds_total_concated)
        dict_total_performance['prauc'] = auc(recall, precision)
        # fpr, tpr, thresholds = roc_curve(trues_total_concated, preds_total_concated)
        # best_threshold = thresholds[np.argmax(tpr - fpr)]  # youden index
        preds_total_concated_bool = preds_total_concated > self.dict_best_threholds['total']
        dict_total_performance['accuracy']   = accuracy_score(trues_total_concated, preds_total_concated_bool)
        dict_total_performance['jaccard']     = jaccard_score(trues_total_concated, preds_total_concated_bool)
        dict_total_performance['precision'] = precision_score(trues_total_concated, preds_total_concated_bool)
        dict_total_performance['recall']       = recall_score(trues_total_concated, preds_total_concated_bool)
        dict_total_performance['f1']               = f1_score(trues_total_concated, preds_total_concated_bool)

        # performance of each timestep
        df_performance_each_timestep = pd.DataFrame()

        list_rocauc    = []
        list_prauc     = []
        list_accuracy  = []
        list_jaccard   = []
        list_precision = []
        list_recall    = []
        list_f1        = []

        for current_timestep, (preds_current_timestep, trues_current_timestep) in enumerate(
                zip(list_preds_each_timestep_concated, list_tures_each_timestep_concated)):
            list_rocauc.append(roc_auc_score(trues_current_timestep, preds_current_timestep))
            precision, recall, _ = precision_recall_curve(trues_current_timestep, preds_current_timestep)
            list_prauc.append(auc(recall, precision))

            # fpr, tpr, thresholds = roc_curve(trues_current_timestep, preds_current_timestep)
            # best_threshold_curr_timestep = thresholds[np.argmax(tpr - fpr)]  # youden index
            preds_current_timestep_bool = preds_current_timestep > self.dict_best_threholds['each_timestep'][current_timestep]

            list_accuracy.append(  accuracy_score(trues_current_timestep, preds_current_timestep_bool))
            list_jaccard.append(    jaccard_score(trues_current_timestep, preds_current_timestep_bool))
            list_precision.append(precision_score(trues_current_timestep, preds_current_timestep_bool))
            list_recall.append(      recall_score(trues_current_timestep, preds_current_timestep_bool))
            list_f1.append(              f1_score(trues_current_timestep, preds_current_timestep_bool))

        df_performance_each_timestep['rocauc']    = list_rocauc
        df_performance_each_timestep['prauc']     = list_prauc
        df_performance_each_timestep['accuracy']  = list_accuracy
        df_performance_each_timestep['jaccard']   = list_jaccard
        df_performance_each_timestep['precision'] = list_precision
        df_performance_each_timestep['recall']    = list_recall
        df_performance_each_timestep['f1']        = list_f1

        # DDI
        if self.is_calc_ddi:
            # iterate every timestep
            for timestep, (preds_current_timestep, trues_current_timestep, edge_indices_current_timestep) in tqdm(
                    enumerate(zip(self.list_preds_each_timestep, self.list_trues_each_timestep, self.list_edge_indices_each_timestep)),
                    leave=False
            ):
                # iterate every hadm batches
                for y_pred, y_true, edge_indices in tqdm(zip(preds_current_timestep, trues_current_timestep,
                                                             edge_indices_current_timestep), leave=False):
                    y_pred_bool = y_pred > self.dict_best_threholds['each_timestep'][timestep]
                    self.list_ddi_pred[timestep].extend(self.ddi_calculator.calc_ddis_for_batch_admi(y_pred_bool, edge_indices))
                    self.list_ddi_true[timestep].extend(self.ddi_calculator.calc_ddis_for_batch_admi(y_true,      edge_indices))

            total_ddis_pred = [np.array(ddi_pred_current_timestep) for ddi_pred_current_timestep in self.list_ddi_pred]
            total_ddis_true = [np.array(ddi_true_current_timestep) for ddi_true_current_timestep in self.list_ddi_true]
            dict_total_performance['ddi_pred'] = np.concatenate(total_ddis_pred, axis=None).mean()
            dict_total_performance['ddi_true'] = np.concatenate(total_ddis_true, axis=None).mean()

            df_performance_each_timestep['ddi_pred'] = [np.mean(ddi_pred_current_timestep) for ddi_pred_current_timestep in self.list_ddi_pred]
            df_performance_each_timestep['ddi_true'] = [np.mean(ddi_true_current_timestep) for ddi_true_current_timestep in self.list_ddi_true]

        return dict_total_performance, df_performance_each_timestep

    def save(self, description: str):
        self.dict_total, self.df = self.get_results()

        path4dict = os.path.join(self.save_dir_path, "total_dicts")
        path4df   = os.path.join(self.save_dir_path, "dfs")

        os.mkdir(path4dict) if not os.path.exists(path4dict) else None
        os.mkdir(path4df)   if not os.path.exists(path4df)   else None

        with open(os.path.join(path4dict, f"{description}.pickle"), 'wb') as f:
            pickle.dump(self.dict_total, f)

        self.df.to_csv(os.path.join(path4df, f"{description}.csv"))
