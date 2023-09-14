import sys; sys.path.append('..')
import pandas as pd
import numpy as np
import os
import torch

from sklearn.metrics import \
    auc, \
    roc_auc_score, \
    roc_curve, \
    accuracy_score, \
    f1_score, \
    precision_score, \
    recall_score, \
    precision_recall_curve, \
    jaccard_score
from tqdm import tqdm

from utils.ddi import DDICalculator


class GridOfRessults:
    r"""For recording validation metrics."""
    def __init__(self,
                 max_timestep: int,
                 is_calc_ddi: bool=False) -> None:
        self.is_calc_ddi = is_calc_ddi
        self.df = pd.DataFrame()

        self.df["auc"]       = [0] * max_timestep  # note: labels start from timestep = 1!
        self.df["acc"]       = [0] * max_timestep
        self.df["F1"]        = [0] * max_timestep
        self.df["Precision"] = [0] * max_timestep
        self.df["recall"]    = [0] * max_timestep
        self.df["PRAUC"]     = [0] * max_timestep
        self.df["Jaccard"]   = [0] * max_timestep

        if self.is_calc_ddi:
            self.ddi_calculator = DDICalculator()
            self.df["DDI"]   = [0] * max_timestep

        self.cnt = 0

    def add_batch_result_per_timestep(self, scores: torch.tensor, labels: torch.tensor):
        list_auc       = []
        list_acc       = []
        list_F1        = []
        list_Precision = []
        list_Recall    = []
        list_PRAUC     = []
        list_Jaccard   = []
        if self.is_calc_ddi:
            list_ddi   = []

        for y_pred, y_true in zip(scores, labels):
            if self.is_calc_ddi:
                y_pred_cp = y_pred.clone()

            y_pred = torch.flatten(y_pred, start_dim=0).cpu()
            y_true = torch.flatten(y_true, start_dim=0).cpu()

            fpr, tpr, thresholds = roc_curve(y_true, y_pred, drop_intermediate=False)  # arg `drop_intermediate` must open
            # <https://blog.csdn.net/qq_39917365/article/details/108273866>
            # <https://blog.csdn.net/weixin_43543177/article/details/107565947>
            # <https://cspaperead.com/rocqu-xian-ji-aucping-gu-zhi-biao/>
            idx = (tpr - fpr).tolist().index(max(tpr - fpr))  # Youden Index
            # or:
            # <https://stats.stackexchange.com/questions/123124>
            # idx = np.argmax(tpr - fpr)
            best_threshold = thresholds[idx]

            list_auc.append(        roc_auc_score(y_true, y_pred))
            list_acc.append(       accuracy_score(y_true, y_pred > best_threshold))
            list_F1.append(              f1_score(y_true, y_pred > best_threshold, pos_label=1, average='binary'))
            list_Precision.append(precision_score(y_true, y_pred > best_threshold, pos_label=1, average='binary'))
            list_Recall.append(      recall_score(y_true, y_pred > best_threshold, pos_label=1, average='binary'))
            list_Jaccard.append(    jaccard_score(y_true, y_pred > best_threshold, pos_label=1, average='binary'))

            # PRAUC
            prc, rec, _ = precision_recall_curve(y_true, y_pred > best_threshold)
            list_PRAUC.append(auc(prc, rec))

            if self.is_calc_ddi:
                ddi_total_this_batch_hadm = 0
                cnt_total_this_batch_hadm = 0
                t_loop = tqdm(y_pred_cp, leave=False)
                for y_pred_curr_admi in t_loop:
                    curr_ddi = self.ddi_calculator.calc_ddi_rate(y_pred_curr_admi > best_threshold)
                    t_loop.set_postfix_str(f"DDI: {curr_ddi:.6f}")
                    ddi_total_this_batch_hadm += curr_ddi
                    cnt_total_this_batch_hadm += 1
                list_ddi.append(ddi_total_this_batch_hadm / cnt_total_this_batch_hadm)

        self.cnt += 1

        self.df["auc"]       += list_auc
        self.df["acc"]       += list_acc
        self.df["F1"]        += list_F1
        self.df["Precision"] += list_Precision
        self.df["recall"]    += list_Recall
        self.df["PRAUC"]     += list_PRAUC
        self.df["Jaccard"]   += list_Jaccard
        if self.is_calc_ddi:
            self.df["DDI"]   += list_ddi

    def get_curr_auc(self):
        return (self.df["auc"] / self.cnt).mean()

    def save_results(self,
                     pth: str,
                     description: str):
        # mean
        self.df["auc"]       = self.df["auc"]       / self.cnt
        self.df["acc"]       = self.df["acc"]       / self.cnt
        self.df["F1"]        = self.df["F1"]        / self.cnt
        self.df["Precision"] = self.df["Precision"] / self.cnt
        self.df["recall"]    = self.df["recall"]    / self.cnt
        self.df["PRAUC"]     = self.df["PRAUC"]     / self.cnt
        self.df["Jaccard"]   = self.df["Jaccard"]   / self.cnt
        if self.is_calc_ddi:
            self.df["DDI"]   = self.df["DDI"]       / self.cnt

        self.df.to_csv(os.path.join(pth, f"{description}.csv"))


if __name__ == "__main__":
    scores = torch.rand(20, 3, 7)
    labels = torch.rand(20, 3, 7) > 0.5

    grid = GridOfRessults(max_timestep=20)
    grid.add_batch_result_per_timestep(scores, labels)
    # UndefinedMetricWarning: Precision and F-score are ill-defined
    #                         and being set to 0.0 in labels with no predicted samples.
    # <https://stackoverflow.com/questions/54150147/classification-report-precision-and-f-score-are-ill-defined>

    print(grid.df)
