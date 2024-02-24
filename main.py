import argparse
import os
import torch
import torch.nn.functional as F

from tqdm import tqdm
from d2l import torch as d2l

from dataset.hgs import MyOwnDataset
from model.lers import LERS
from utils.metrics import Logger
from utils.best_thresholds import BestThreshldLogger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ********* Hyperparams ********* #
    # following arguments are model settings
    parser.add_argument("--max_timestep", type=int,                 default=20,         help="The maximum `TIMESTEP`")
    # NOTE: when max_timestep set to 30 or 50,
    #       would trigger the assert error "last timestep has not labels!"
    #       in `get_subgraph_by_timestep` in lers.py
    parser.add_argument("--gnn_type",                                 default="GENConv", help="Specify the `conv` that being used as MessagePassing")
    parser.add_argument("--num_decoder_layers_admission", type=int,   default=6,         help="Number of decoder layers for admission")
    parser.add_argument("--num_decoder_layers_labitem",   type=int,   default=6,         help="Number of decoder layers for labitem")
    parser.add_argument("--num_decoder_layers_drug",      type=int,   default=6,         help="Number of decoder layers for drug")
    parser.add_argument("--hidden_dim",                   type=int,   default=128,       help="hidden dimension")
    parser.add_argument("--lr",                           type=float, default=0.0003,    help="learning rate")

    # Paths
    # parser.add_argument("--root_path_dataset",  default=r"../datasets/mimic-iii-hgs-new", help="path where dataset directory locates")  # in linux
    parser.add_argument("--root_path_dataset",  default=r"../datasets/mimic-iii-hgs", help="path where dataset directory locates")  # in linux
    parser.add_argument("--path_dir_model_hub", default=r"./model/hub",               help="path where models save")
    parser.add_argument("--path_dir_results",   default=r"./results",                 help="path where results save")
    parser.add_argument("--path_dir_thresholds",default=r"./thresholds",              help="path where thresholds save")

    # Experiment settings
    parser.add_argument("--task",                                   default="MIX",     help="Specify the goal of the recommended task")
    parser.add_argument("--epochs",                       type=int, default=3)
    parser.add_argument("--train",            action="store_true",  default=False,     help="specify whether do training")
    parser.add_argument("--test",             action="store_true",  default=False,     help="specify whether do testing")
    parser.add_argument("--test_model_state_dict",                  default=None,      help="test only model's state_dict file name")  # must be specified when --train=False!
    parser.add_argument("--test_num",                     type=int, default=-1,        help="number of testing")
    parser.add_argument("--use_gpu",          action="store_true",  default=False,     help="specify whether to use GPU")
    parser.add_argument("--num_gpu",                      type=int, default=0,         help="specify which GPU to be used firstly")
    parser.add_argument("--batch_size_by_HADMID",         type=int, default=128,       help="specified the batch size that will be used for splitting the dataset by HADM_ID")
    parser.add_argument("--neg_smp_strategy",             type=int, default=0,         help="the stratege of negative sampling")
    args = parser.parse_args()

    if not os.path.exists(args.path_dir_model_hub):
        os.mkdir(args.path_dir_model_hub)
    root_path = os.path.join(args.root_path_dataset, f"batch_size_{args.batch_size_by_HADMID}")
    resl_path = os.path.join(args.path_dir_results, f"#{args.test_num}")
    if not os.path.exists(resl_path):
        os.mkdir(resl_path)

    device = d2l.try_gpu(args.num_gpu) if args.use_gpu else torch.device('cpu')

    # model
    model = LERS(max_timestep=args.max_timestep,
                 gnn_type=args.gnn_type,
                 num_decoder_layers_admission=args.num_decoder_layers_admission,
                 num_decoder_layers_labitem=args.num_decoder_layers_labitem,
                 num_decoder_layers_drug=args.num_decoder_layers_drug,
                 hidden_dim=args.hidden_dim,
                 neg_smp_strategy=args.neg_smp_strategy).to(device)

    if args.train:
        logger4item_best_thresholds = BestThreshldLogger(max_timestep=args.max_timestep, save_dir_path=args.path_dir_thresholds)
        logger4drug_best_thresholds = BestThreshldLogger(max_timestep=args.max_timestep, save_dir_path=args.path_dir_thresholds)

        train_set = MyOwnDataset(root_path=root_path, usage="train")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        # train
        # with torch.autograd.detect_anomaly():
        for epoch in tqdm(range(args.epochs)):
            model.train()
            t_loop_train_set = tqdm(train_set, leave=False)
            for hg in t_loop_train_set:
                optimizer.zero_grad()
                hg = hg.to(device)
                list_scores4item, list_labels4item, list_scores4drug, list_labels4drug, _ = model(hg)

                if epoch == (args.epochs - 1):
                    logger4item_best_thresholds.log(list_scores4item, list_labels4item)
                    logger4drug_best_thresholds.log(list_scores4drug, list_labels4drug)

                # calculating loss
                loss4item = torch.tensor(0.0).to(device)
                for scores4item, labels4item in zip(list_scores4item, list_labels4item):
                    loss4item += F.binary_cross_entropy_with_logits(scores4item, labels4item)
                loss4drug = torch.tensor(0.0).to(device)
                for scores4drug, labels4drug in zip(list_scores4drug, list_labels4drug):
                    loss4drug += F.binary_cross_entropy_with_logits(scores4drug, labels4drug)
                loss = loss4item + loss4drug  # with different weight
                loss.backward()

                # optimizing
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                optimizer.step()

                t_loop_train_set.set_postfix_str(f'\033[32m Current loss: {loss:.4f} \033[0m')

        # save trained model
        model_saving_prefix = f"task={args.task}_gnn_type={args.gnn_type}_batch_size_by_HADMID={args.batch_size_by_HADMID}"
        torch.save(model.state_dict(), os.path.join(args.path_dir_model_hub, f"{model_saving_prefix}_loss={loss:.4f}.pt"))

        logger4item_best_thresholds.save(prefix=f"4ITEMS_gnn_type={args.gnn_type}_batch_size_by_HADMID={args.batch_size_by_HADMID}")
        logger4drug_best_thresholds.save(prefix=f"4DRUGS_gnn_type={args.gnn_type}_batch_size_by_HADMID={args.batch_size_by_HADMID}")

    # testing
    if args.test:
        str_prefix4item = f"4ITEMS_gnn_type={args.gnn_type}_batch_size_by_HADMID={args.batch_size_by_HADMID}"
        str_prefix4drug = f"4DRUGS_gnn_type={args.gnn_type}_batch_size_by_HADMID={args.batch_size_by_HADMID}"

        test_set = MyOwnDataset(root_path=root_path, usage="test")

        if not args.train:
            model_state_dict = torch.load(os.path.join(args.path_dir_model_hub, f"{args.test_model_state_dict}.pt"), map_location=device)
            model.load_state_dict(model_state_dict)

        # metrics loggers
        logger4item = Logger(max_timestep=args.max_timestep, save_dir_path=resl_path, best_thresholdspath=os.path.join(args.path_dir_thresholds, f"{str_prefix4item}_best_thresholds.pickle"))
        logger4drug = Logger(max_timestep=args.max_timestep, save_dir_path=resl_path, best_thresholdspath=os.path.join(args.path_dir_thresholds, f"{str_prefix4drug}_best_thresholds.pickle"), is_calc_ddi=True)

        model.eval()
        with torch.no_grad():
            for hg in tqdm(test_set):
                hg = hg.to(device)
                list_scores4item, list_labels4item, list_scores4drug, list_labels4drug, list_edge_indices4drug = model(hg)
                logger4item.log(list_scores4item, list_labels4item)
                logger4drug.log(list_scores4drug, list_labels4drug, list_edge_indices4drug)

        logger4item.save(description=str_prefix4item)
        logger4drug.save(description=str_prefix4drug)
