import pandas as pd
import numpy as np
import os
import torch

from sklearn.metrics import roc_auc_score, f1_score, roc_curve, classification_report

class GridOfRessults:
    r"""For recording validation metrics."""
    def __init__(self, max_timestep: int) -> None:
        self.df = pd.DataFrame()

        self.df["auc"]                    = [0] * max_timestep  # note: labels start from timestep = 1!
        self.df["acc"]                    = [0] * max_timestep
        self.df["f1_marco_avg"]           = [0] * max_timestep
        self.df["f1_weighted_avg"]        = [0] * max_timestep
        self.df["precision_marco_avg"]    = [0] * max_timestep
        self.df["precision_weighted_avg"] = [0] * max_timestep
        self.df["recall_marco_avg"]       = [0] * max_timestep
        self.df["recall_weighted_avg"]    = [0] * max_timestep

        self.cnt = 0

    def add_batch_result_per_timestep(self, scores: torch.tensor, labels: torch.tensor):
        list_auc                    = []
        list_f1_macro_avg           = []
        list_f1_weighted_avg        = []
        list_acc                    = []
        list_precision_marco_avg    = []
        list_precision_weighted_avg = []
        list_recall_marco_avg       = []
        list_recall_weighted_avg    = []

        for y_pred, y_true in zip(scores, labels):
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
            
            # TODO: should we add the f1-score on the pos_label class like `sklearn.metrics.f1_score` does defaultly?
            dict_report = classification_report(y_true, y_pred > best_threshold, output_dict=True)

            list_auc.append(roc_auc_score(y_true, y_pred))
            list_acc.append(dict_report["accuracy"])
            list_f1_macro_avg.append(dict_report["macro avg"]["f1-score"])
            list_f1_weighted_avg.append(dict_report["weighted avg"]["f1-score"])
            list_precision_marco_avg.append(dict_report["macro avg"]["precision"])  # averaging the unweighted mean per label
            list_precision_weighted_avg.append(dict_report["weighted avg"]["precision"])  # averaging the support-weighted mean per label
            list_recall_marco_avg.append(dict_report["macro avg"]["recall"])
            list_recall_weighted_avg.append(dict_report["weighted avg"]["recall"])

            # IMPORTANT: difference between macro-avg and weighted-avg, see this discussion : <https://datascience.stackexchange.com/questions/65839>

        self.cnt += 1

        self.df["auc"]                    += list_auc
        self.df["acc"]                    += list_acc
        self.df["f1_marco_avg"]           += list_f1_macro_avg
        self.df["f1_weighted_avg"]        += list_f1_weighted_avg
        self.df["precision_marco_avg"]    += list_precision_marco_avg
        self.df["precision_weighted_avg"] += list_precision_weighted_avg
        self.df["recall_marco_avg"]       += list_recall_marco_avg
        self.df["recall_weighted_avg"]    += list_recall_weighted_avg

    def save_results(self, pth: str, description: str):
        # mean
        self.df["auc"]                    = self.df["auc"]                    / self.cnt
        self.df["acc"]                    = self.df["acc"]                    / self.cnt
        self.df["f1_marco_avg"]           = self.df["f1_marco_avg"]           / self.cnt
        self.df["f1_weighted_avg"]        = self.df["f1_weighted_avg"]        / self.cnt
        self.df["precision_marco_avg"]    = self.df["precision_marco_avg"]    / self.cnt
        self.df["precision_weighted_avg"] = self.df["precision_weighted_avg"] / self.cnt
        self.df["recall_marco_avg"]       = self.df["recall_marco_avg"]       / self.cnt
        self.df["recall_weighted_avg"]    = self.df["recall_weighted_avg"]    / self.cnt

        self.df.to_csv(os.path.join(pth, f"{description}_auc={self.df.auc.mean():.4f}.csv"))


if __name__ == "__main__":
    scores = torch.rand(20, 3, 7)
    labels = torch.rand(20, 3, 7) > 0.5

    grid = GridOfRessults(max_timestep=20)
    grid.add_batch_result_per_timestep(scores, labels)

    # UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
    # <https://stackoverflow.com/questions/54150147/classification-report-precision-and-f-score-are-ill-defined>

    print(grid.df)
