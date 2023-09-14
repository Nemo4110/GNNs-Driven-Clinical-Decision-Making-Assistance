import argparse
import os
import torch
import torch.nn.functional as F

from tqdm import tqdm
from d2l import torch as d2l
from sklearn.metrics import roc_auc_score

from dataset.hgs import MyOwnDataset
from model.lers import LERS
from utils.gird import GridOfRessults


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ********* Hyperparams ********* #
    # following arguments are model settings
    parser.add_argument("--max_timestep", type=int,                     default=20,         help="The maximum `TIMESTEP`")
    # NOTE: when max_timestep set to 30 or 50,
    #       would trigger the assert error "last timestep has not labels!"
    #       in `get_subgraph_by_timestep` in lers.py
    parser.add_argument("--gnn_type",                                   default="GINEConv", help="Specify the `conv` that being used as MessagePassing")
    parser.add_argument("--num_decoder_layers_admission", type=int,     default=6,          help="Number of decoder layers for admission")
    parser.add_argument("--num_decoder_layers_labitem", type=int,       default=6,          help="Number of decoder layers for labitem")
    parser.add_argument("--num_decoder_layers_drug", type=int,          default=6,          help="Number of decoder layers for drug")
    parser.add_argument("--hidden_dim", type=int,                       default=128,        help="hidden dimension")

    parser.add_argument("--lr", type=float,                             default=0.0003,     help="learning rate")

    # Experiment settings
    parser.add_argument("--epochs", type=int,                           default=5)
    parser.add_argument("--val", action="store_true",                   default=True,       help="specify whether do validating")
    parser.add_argument("--task",                                       default="MIX",      help="Specify the goal of the recommended task")
    parser.add_argument("--root_path_dataset",                          default=r"../datasets/mimic-iii-hgs-new",
                                                                                            help="path where dataset directory locates")  # in linux
    parser.add_argument("--batch_size_by_HADMID", type=int,             default=128,        help="specified the batch size that will be used for splitting the dataset by HADM_ID")
    parser.add_argument("--use_gpu", action="store_true",               default=True,       help="specify whether to use GPU")
    parser.add_argument("--num_gpu", type=int,                          default=0,          help="specify which GPU to be used firstly")
    parser.add_argument("--record_metrics", action="store_true",        default=True,       help="specify whether to record metrics")

    args = parser.parse_args()
    # print(args)

    # ********* Experiment ********* #
    device = d2l.try_gpu(args.num_gpu) if args.use_gpu else torch.device('cpu')
    print(f"*** Using device: {device} ***")

    # dataset
    root_path = os.path.join(args.root_path_dataset, f"batch_size_{args.batch_size_by_HADMID}")
    print(f"*** Choosing dataset at: {root_path} ***")
    train_set = MyOwnDataset(root_path=root_path, usage="train")
    val_set   = MyOwnDataset(root_path=root_path, usage="val") if args.val else None

    # model
    model = LERS(max_timestep=args.max_timestep,
                 gnn_type=args.gnn_type,
                 num_decoder_layers_admission=args.num_decoder_layers_admission,
                 num_decoder_layers_labitem=args.num_decoder_layers_labitem,
                 num_decoder_layers_drug=args.num_decoder_layers_drug,
                 hidden_dim=args.hidden_dim).to(device)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # with torch.autograd.detect_anomaly():

    gird      = GridOfRessults(max_timestep=args.max_timestep)
    gird4drug = GridOfRessults(max_timestep=args.max_timestep)

    t_loop = tqdm(range(args.epochs))
    for epoch in t_loop:
        model.train()
        metric_train = d2l.Accumulator(2)  # total_examples, total_loss
        t_loop_train_set = tqdm(train_set, leave=False)
        for hg in t_loop_train_set:
            optimizer.zero_grad()
            hg = hg.to(device)
            scores, labels, scores4drug, labels4drug = model(hg)  # Note: now scores and labels have the same shape

            # TODO: validate the `start_dim` shoule be 1 or 0?
            #       or, in other words, will the model performence improve
            #       after set `start_dim` to 0, and `end_dim` to 0?
            scores = torch.flatten(scores, start_dim=1)
            labels = torch.flatten(labels, start_dim=1)
            scores4drug = torch.flatten(scores4drug, start_dim=1)
            labels4drug = torch.flatten(labels4drug, start_dim=1)

            loss4labi = F.binary_cross_entropy_with_logits(scores, labels)
            loss4drug = F.binary_cross_entropy_with_logits(scores4drug, labels4drug)

            loss = loss4labi + 7*loss4drug  # with different weight
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()

            metric_train.add(scores.numel(), loss * (scores.numel()))
            t_loop_train_set.set_postfix_str(f'\033[32m Current loss: {loss:.4f} \033[0m')

        train_loss = metric_train[1] / metric_train[0]
        t_loop.set_postfix_str(f'\033[32m Training Loss: {train_loss:.4f}\033[0m')

    if val_set is not None:
        model.eval()
        with torch.no_grad():
            t_loop_val_set = tqdm(val_set, leave=False)
            for hg in t_loop_val_set:
                hg = hg.to(device)
                scores, labels, scores4drug, labels4drug = model(hg)

                gird.add_batch_result_per_timestep(scores, labels)
                auc = gird.get_curr_auc()

                gird4drug.add_batch_result_per_timestep(scores4drug, labels4drug)
                auc4drug = gird4drug.get_curr_auc()

                t_loop_val_set.set_postfix_str(f'AUC4LABITEM: {auc:.4f}, AUC4DRUG: {auc4drug:.4f}')

            avg_auc = 0.5 * (auc + auc4drug)

    pth_pts = os.path.join(".", "model", "hub")
    pth_res = os.path.join(".", "results", "dfs")
    os.mkdir(pth_pts) if not os.path.exists(pth_pts) else None
    os.mkdir(pth_res) if not os.path.exists(pth_res) else None

    str_prefix = f"4LABITEMS_gnn_type={args.gnn_type}_batch_size_by_HADMID={args.batch_size_by_HADMID}"
    gird.save_results(pth=pth_res, description=str_prefix)

    str_prefix4drug = f"4DRUGS_gnn_type={args.gnn_type}_batch_size_by_HADMID={args.batch_size_by_HADMID}"
    gird4drug.save_results(pth=pth_res, description=str_prefix4drug)

    model_saving_prefix = f"task={args.task}_gnn_type={args.gnn_type}_batch_size_by_HADMID={args.batch_size_by_HADMID}"
    torch.save(model.state_dict(), os.path.join(pth_pts, f"{str_prefix}_auc={avg_auc:.4f}.pt"))
