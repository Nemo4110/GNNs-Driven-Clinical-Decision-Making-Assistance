import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from d2l import torch as d2l
from sklearn.metrics import roc_auc_score

from dataset.hgs import MyOwnDataset
from model.lers import LERS


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ********* Hyperparams ********* #

    # following arguments are model settings
    parser.add_argument("--max_timestep", type=int, default=20, help="The maximum `TIMESTEP`")
    parser.add_argument("--gnn_type", default="GINEConv",help="Specify the `conv` that being used as MessagePassing")
    parser.add_argument("--num_decoder_layers_admission", type=int, default=6, help="Number of decoder layers for admission")
    parser.add_argument("--num_decoder_layers_labitem", type=int, default=6, help="Number of decoder layers for labitem")
    parser.add_argument("--hidden_dim", type=int, default=64, help="hidden dimension")

    parser.add_argument("--lr", type=float, default=0.0003, help="learning rate")

    # Experiment settings
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--val", action="store_true", default=True, help="specify whether do validating")
    parser.add_argument("--root_path_dataset", default=r"../datasets/mimic-iii-hgs", help="path where dataset directory locates")  # in linux
    # parser.add_argument("--root_path_dataset", default=r"dataset/mimic-iii-hgs", help="path where dataset directory locates")  # in linux
    # parser.add_argument("--root_path_dataset", default=r"dataset\mimic-iii-hgs", help="path where dataset directory locates")  # in windows
    parser.add_argument("--batch_size_by_HADMID", type=int, default=512, help="specified the batch size that will be used for splitting the dataset by HADM_ID")
    parser.add_argument("--use_gpu", action="store_true", default=True, help="specify whether to use GPU")
    parser.add_argument("--num_gpu", type=int, default=0, help="specify which GPU to be used firstly")
    parser.add_argument("--record_metrics", action="store_true", default=True, help="specify whether to record metrics")

    args = parser.parse_args()
    # print(args)

    # ********* Experiment ********* #
    device = d2l.try_gpu(args.num_gpu) if args.use_gpu else torch.device('cpu')
    print(f"*** Using device: {device} ***")

    # dataset
    root_path = os.path.join(args.root_path_dataset, f"batch_size_{args.batch_size_by_HADMID}")
    print(f"*** Choosing dataset at: {root_path} ***")
    train_set = MyOwnDataset(root_path=root_path, usage="train")
    val_set = MyOwnDataset(root_path=root_path, usage="val") if args.val else None

    model = LERS(max_timestep=args.max_timestep,
                 gnn_type=args.gnn_type,
                 num_decoder_layers_admission=args.num_decoder_layers_admission,
                 num_decoder_layers_labitem=args.num_decoder_layers_labitem,
                 hidden_dim=args.hidden_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # with torch.autograd.detect_anomaly():
    t_loop = tqdm(range(args.epochs))
    for epoch in t_loop:
        model.train()
        metric_train = d2l.Accumulator(2)  # total_examples, total_loss
        t_loop_train_set = tqdm(train_set, leave=False)
        for hg in t_loop_train_set:
            optimizer.zero_grad()
            hg = hg.to(device)
            scores, labels = model(hg)  # Note: now scores and labels have the same shape
            scores = torch.flatten(scores, start_dim=1)
            labels = torch.flatten(labels, start_dim=1)
            loss = F.binary_cross_entropy_with_logits(scores, labels)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            metric_train.add(scores.numel(), loss * (scores.numel()))
            t_loop_train_set.set_postfix_str(f'\033[32m Current loss: {loss:.4f} \033[0m')

        if val_set is not None:
            model.eval()
            with torch.no_grad():
                metric_val = d2l.Accumulator(2)  # cnt, auc
                t_loop_val_set = tqdm(val_set, leave=False)
                for hg in t_loop_val_set:
                    hg = hg.to(device)
                    scores, labels = model(hg)
                    scores = torch.flatten(scores, start_dim=0).cpu().numpy()
                    labels = torch.flatten(labels, start_dim=0).cpu().numpy()
                    auc = roc_auc_score(labels, scores)
                    metric_val.add(1, auc)
                    t_loop_val_set.set_postfix_str(f'Current auc: {auc:.4f}')

        train_loss = metric_train[1] / metric_train[0]
        avg_auc = metric_val[1] / metric_val[0]

        t_loop.set_postfix_str(f'\033[32m Training Loss: {train_loss:.4f}, Validation AUC: {avg_auc:.4f} \033[0m')
