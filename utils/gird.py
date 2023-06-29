import pandas as pd
import numpy as np
import os
import torch

from sklearn.metrics import roc_auc_score, f1_score, roc_curve

class GridOfRessults:
    r"""For recording validation metrics."""
    def __init__(self, max_timestep: int) -> None:
        self.df = pd.DataFrame()
        self.df["auc"] = [0] * max_timestep  # note: labels start from timestep = 1!
        self.df["f1"] = [0] * max_timestep
        self.cnt = 0

    def add_batch_result_per_timestep(self, scores: torch.tensor, labels: torch.tensor):
        list_auc = []
        list_f1 = []
        for y_pred, y_true in zip(scores, labels):
            y_pred = torch.flatten(y_pred, start_dim=0).cpu()
            y_true = torch.flatten(y_true, start_dim=0).cpu()

            fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)

            # <https://blog.csdn.net/qq_39917365/article/details/108273866>
            # <https://blog.csdn.net/weixin_43543177/article/details/107565947>
            # <https://cspaperead.com/rocqu-xian-ji-aucping-gu-zhi-biao/>
            idx = (tpr - fpr).tolist().index(max(tpr - fpr))  # Youden Index

            # or:
            # <https://stats.stackexchange.com/questions/123124>
            # idx = np.argmax(tpr - fpr)

            best_threshold = thresholds[idx]

            list_auc.append(roc_auc_score(y_true, y_pred))
            list_f1.append(f1_score(y_true, y_pred > best_threshold))

        self.cnt += 1
        self.df["auc"] += list_auc
        self.df["f1"] += list_f1

    def save_results(self, pth: str, description: str):
        # mean
        self.df["auc"] = self.df["auc"] / self.cnt
        self.df["f1"] = self.df["f1"] / self.cnt

        self.df.to_csv(os.path.join(pth, f"{description}_auc={self.df.auc.mean():.4f}_f1={self.df.f1.mean():.4f}.csv"))


if __name__ == "__main__":
    scores = torch.rand(50, 3, 3)
    labels = torch.rand(50, 3, 3) > 0.5

    grid = GridOfRessults(max_timestep=50)
    grid.add_batch_result_per_timestep(scores, labels)

    print(grid.df)
