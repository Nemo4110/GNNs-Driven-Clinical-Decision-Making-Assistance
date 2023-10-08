import sys; sys.path.append('..')
import pandas as pd
import numpy as np
import os

from sklearn.metrics import \
    auc, \
    roc_auc_score, \
    roc_curve, \
    accuracy_score, \
    jaccard_score, \
    f1_score, \
    precision_score, \
    recall_score, \
    precision_recall_curve
from utils.ddi import DDICalculator


class Logger:
    r"""For logging evaluating metrics."""
    def __init__(self, max_timestep, is_calc_ddi: bool=False):
        self.is_calc_ddi = is_calc_ddi
        self.df = pd.DataFrame()

        self.df["auc"]       = [0] * max_timestep  # note: labels start from timestep = 1!
        self.df["acc"]       = [0] * max_timestep
        self.df["jaccard"]   = [0] * max_timestep
        self.df["precision"] = [0] * max_timestep
        self.df["recall"]    = [0] * max_timestep
        self.df["F1"]        = [0] * max_timestep
        self.df["PRAUC"]     = [0] * max_timestep

        if self.is_calc_ddi:
            self.df["DDI_true"] = [0] * max_timestep
            self.df["DDI_pred"] = [0] * max_timestep
            self.ddi_calculator = DDICalculator()

        self.cnt = 0

    def log(self, list_y_pred, list_y_true, list_edge_indices=None):
        if self.is_calc_ddi and list_edge_indices is not None:
            self.log_ddi(list_y_pred, list_y_true, list_edge_indices)

        list_auc       = []
        list_acc       = []
        list_jaccard   = []
        list_precision = []
        list_recall    = []
        list_F1        = []
        list_PRAUC     = []

        for y_pred, y_true in zip(list_y_pred, list_y_true):
            y_pred = y_pred.cpu()
            y_true = y_true.cpu()

            list_auc.append(roc_auc_score(y_true, y_pred))
            fpr, tpr, thresholds = roc_curve(y_true, y_pred, drop_intermediate=False)
            best_threshold = thresholds[np.argmax(tpr - fpr)]  # youden index
            list_acc.append(accuracy_score(y_true, y_pred > best_threshold))
            list_jaccard.append(jaccard_score(y_true, y_pred > best_threshold))
            list_precision.append(precision_score(y_true, y_pred > best_threshold))
            list_recall.append(recall_score(y_true, y_pred > best_threshold))
            list_F1.append(f1_score(y_true, y_pred > best_threshold))
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            list_PRAUC.append(auc(recall, precision))

        self.cnt += 1

        self.df["auc"]       += list_auc
        self.df["acc"]       += list_acc
        self.df["jaccard"]   += list_jaccard
        self.df["precision"] += list_precision
        self.df["recall"]    += list_recall
        self.df["F1"]        += list_F1
        self.df["PRAUC"]     += list_PRAUC

    def log_ddi(self, list_y_pred, list_y_true, list_edge_indices):
        list_ddi_true = []
        list_ddi_pred = []
        for y_pred, y_true, edge_indices in zip(list_y_pred, list_y_true, list_edge_indices):
            y_pred = y_pred.cpu()
            y_true = y_true.cpu()
            edge_indices = edge_indices.cpu()

            fpr, tpr, thresholds = roc_curve(y_true, y_pred, drop_intermediate=False)
            best_threshold = thresholds[np.argmax(tpr - fpr)]  # youden index

            list_ddi_true.append(self.ddi_calculator.calc_mean_ddi_rate_for_batch_admi(y_true, edge_indices))
            list_ddi_pred.append(self.ddi_calculator.calc_mean_ddi_rate_for_batch_admi(y_pred > best_threshold, edge_indices))

        # self.cnt += 1  # already done in `log`
        self.df["DDI_true"] += list_ddi_true
        self.df["DDI_pred"] += list_ddi_pred

    def get_curr_auc(self):
        return (self.df["auc"] / self.cnt).mean()

    def save(self, path, description):
        self.df["auc"]       = self.df["auc"]       / self.cnt
        self.df["acc"]       = self.df["acc"]       / self.cnt
        self.df["jaccard"]   = self.df["jaccard"]   / self.cnt
        self.df["precision"] = self.df["precision"] / self.cnt
        self.df["recall"]    = self.df["recall"]    / self.cnt
        self.df["F1"]        = self.df["F1"]        / self.cnt
        self.df["PRAUC"]     = self.df["PRAUC"]     / self.cnt

        if self.is_calc_ddi:
            self.df["DDI_true"] = self.df["DDI_true"] / self.cnt
            self.df["DDI_pred"] = self.df["DDI_pred"] / self.cnt

        self.df.to_csv(os.path.join(path, f"{description}.csv"))
