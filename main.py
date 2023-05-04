import argparse
import os
import torch
import torch.nn.functional as F

from tqdm import tqdm
from d2l import torch as d2l
from sklearn.metrics import roc_auc_score

from dataset.hgs import MyOwnDataset
from model.lers import LERS


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparams:

    # following arguments are model settings
    parser.add_argument("--max_timestep", help="The maximum number of `TIMESTEP`")
    parser.add_argument("--num_decoder_layers_admission", type=int)
    parser.add_argument("--num_decoder_layers_labitem", type=int)
    parser.add_argument("--hidden_dim", type=int, default=128, help="hidden dimension")

    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--gnn_drop", type=float, default=0.3, help="dropout rate of gnn layers")

    # Experiment settings
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--val", action="store_true", help="specify whether do validating")
    parser.add_argument("--root_path_dataset", default="../datasets/mimic/mimic-iii-hgs", help="path where dataset directory locates")  # in linux
    parser.add_argument("--batch_size_by_HADMID", type=int, help="specified the batch size that will be used for splitting the dataset by HADM_ID")
    parser.add_argument("--use_gpu", action="store_true", default=False, help="specify whether to use GPU")
    parser.add_argument("--record_metrics", action="store_true", default=True, help="specify whether to record metrics")

    args = parser.parse_args()

    device = d2l.try_gpu() if args.use_gpu else d2l.cpu()
    print(f"*** Using device: {device} ***")

    model = LERS(max_timestep=args.max_timestep,
                 gnn_drop=args.gnn_drop,
                 num_decoder_layers_admission=args.num_decoder_layers_admission,
                 num_decoder_layers_labitem=args.num_decoder_layers_labitem,
                 hidden_dim=args.hidden_dim).to(device)

    # dataset
    root_path = os.path.join(args.root_path_dataset, f"batch_size_{args.batch_size_by_HADMID}")
    train_set = MyOwnDataset(root_path=root_path, usage="train")
    val_set = MyOwnDataset(root_path=root_path, usage="val") if args.val else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    t_loop = tqdm(range(args.epochs))
    for epoch in t_loop:
        model.train()
        metric_train = d2l.Accumulator(2)  # total_examples, total_loss
        for hg in tqdm(train_set, leave=False):
            optimizer.zero_grad()
            hg = hg.to(device)
            scores, labels = model(hg)  # Note: now scores and labels have the same shape
            loss = F.binary_cross_entropy_with_logits(scores, labels)
            loss.backward()
            optimizer.step()
            metric_train.add(scores.numel(), loss * (scores.numel()))

        if val_set is not None:
            model.eval()
            with torch.no_grad():
                metric_val = d2l.Accumulator(2)  # cnt, auc
                for hg in tqdm(val_set, leave=False):
                    hg = hg.to(device)
                    scores, labels = model(hg)
                    scores = torch.flatten(scores, start_dim=1).cpu().numpy()
                    labels = torch.flatten(labels, start_dim=1).cpu().numpy()
                    metric_val.add(1, roc_auc_score(labels, scores))

        train_loss = metric_train[1] / metric_train[0]
        avg_auc = metric_val[1] / metric_val[0]

        t_loop.set_description(f'Epoch [{epoch}/{args.epochs}]')
        t_loop.set_postfix_str(f'Training Loss: {train_loss:.4f}, Validation AUC: {avg_auc:.4f}')
