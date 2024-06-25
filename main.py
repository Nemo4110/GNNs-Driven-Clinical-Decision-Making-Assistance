import argparse
import os
import torch
import torch.nn.functional as F

import utils.constant as constant

from tqdm import tqdm
from d2l import torch as d2l

from dataset.hgs import MyOwnDataset
from model.lers import LERS
from utils.metrics import Logger
from utils.best_thresholds import BestThreshldLogger
from utils.misc import calc_loss, node_type_to_prefix
from utils.config import HeteroGraphConfig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ********* Hyperparams ********* #
    # following arguments are model settings
    parser.add_argument("--max_timestep", type=int,                 default=20,         help="The maximum `TIMESTEP`")
    # NOTE: when max_timestep set to 30 or 50,
    #       would trigger the assert error "last timestep has not labels!"
    #       in `get_subgraph_by_timestep` in lers.py (bigger max_timestep can be support in future)
    parser.add_argument("--gnn_type",                                 default="GENConv", help="Specify the `conv` that being used as MessagePassing")
    parser.add_argument("--gnn_layer_num",                type=int,   default=2,         help="Number of gnn layers")
    parser.add_argument("--num_decoder_layers",           type=int,   default=6,         help="Number of decoder layers")
    parser.add_argument("--hidden_dim",                   type=int,   default=128,       help="hidden dimension")
    parser.add_argument("--lr",                           type=float, default=0.0003,    help="learning rate")

    # Paths
    parser.add_argument("--root_path_dataset",  default=constant.PATH_MIMIC_III_HGS_OUTPUT, help="path where dataset directory locates")  # in linux
    parser.add_argument("--path_dir_model_hub", default=r"./model/hub",                     help="path where models save")
    parser.add_argument("--path_dir_results",   default=r"./results",                       help="path where results save")
    parser.add_argument("--path_dir_thresholds",default=r"./thresholds",                    help="path where thresholds save")

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

    root_path = os.path.join(args.root_path_dataset, f"batch_size_{args.batch_size_by_HADMID}")
    resl_path = os.path.join(args.path_dir_results, f"#{args.test_num}")

    if not os.path.exists(resl_path):               os.mkdir(resl_path)
    if not os.path.exists(args.path_dir_model_hub): os.mkdir(args.path_dir_model_hub)
    if not os.path.exists(args.path_dir_thresholds):os.mkdir(args.path_dir_thresholds)

    device = d2l.try_gpu(args.num_gpu) if args.use_gpu else torch.device('cpu')

    # Heterogeneous graph config
    node_types, edge_types = HeteroGraphConfig.use_all_edge_type() if args.task=="MIX" else HeteroGraphConfig.use_one_edge_type(item_type=args.task)

    # model
    model = LERS(max_timestep=args.max_timestep,
                 gnn_type=args.gnn_type,
                 gnn_layer_num=args.gnn_layer_num,
                 node_types=node_types,
                 edge_types=edge_types,
                 num_decoder_layers=args.num_decoder_layers,
                 hidden_dim=args.hidden_dim,
                 neg_smp_strategy=args.neg_smp_strategy).to(device)

    if args.train:
        best_threshold_loggers = {
            node_type: BestThreshldLogger(max_timestep=args.max_timestep, save_dir_path=args.path_dir_thresholds)
            for node_type in node_types if node_type != 'admission'
        }

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
                dict_every_day_pred = model(hg)
                loss = calc_loss(dict_every_day_pred, node_types, device)  # calculating loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                optimizer.step()  # optimizing

                # log the best_threshold
                if epoch == (args.epochs - 1):
                    for node_type, best_threshold_logger in best_threshold_loggers.items():
                        best_threshold_logger.log(dict_every_day_pred[node_type]["scores"],
                                                  dict_every_day_pred[node_type]["labels"])
                        best_threshold_logger.save(prefix=f"4{node_type_to_prefix[node_type]}_gnn_type={args.gnn_type}_batch_size_by_HADMID={args.batch_size_by_HADMID}")

                t_loop_train_set.set_postfix_str(f'\033[32m Current loss: {loss:.4f} \033[0m')

        # save trained model
        model_saving_prefix = f"task={args.task}_gnn_type={args.gnn_type}_batch_size_by_HADMID={args.batch_size_by_HADMID}"
        torch.save(model.state_dict(), os.path.join(args.path_dir_model_hub, f"{model_saving_prefix}_loss={loss:.4f}.pt"))

    # testing
    if args.test:
        test_set = MyOwnDataset(root_path=root_path, usage="test")

        if not args.train:
            model_state_dict = torch.load(os.path.join(args.path_dir_model_hub, f"{args.test_model_state_dict}.pt"), map_location=device)
            model.load_state_dict(model_state_dict)

        # metrics loggers
        metrics_loggers = {}
        for node_type in node_types:
            if node_type == 'admission':
                continue
            metrics_loggers[node_type] = Logger(max_timestep=args.max_timestep,
                                                save_dir_path=resl_path,
                                                best_thresholdspath=os.path.join(args.path_dir_thresholds, f"4{node_type_to_prefix[node_type]}_gnn_type={args.gnn_type}_batch_size_by_HADMID={args.batch_size_by_HADMID}_best_thresholds.pickle"),
                                                is_calc_ddi=True if node_type == 'drug' else False)

        model.eval()
        with torch.no_grad():
            for hg in tqdm(test_set):
                hg = hg.to(device)
                dict_every_day_pred = model(hg)

                for node_type, metrics_logger in metrics_loggers.items():
                    metrics_logger.log(dict_every_day_pred[node_type]["scores"],
                                       dict_every_day_pred[node_type]["labels"],
                                       dict_every_day_pred[node_type]["indices"] if node_type == 'drug' else None)

        for node_type, metrics_logger in metrics_loggers.items():
            metrics_logger.save(description=f"4{node_type_to_prefix[node_type]}_gnn_type={args.gnn_type}_batch_size_by_HADMID={args.batch_size_by_HADMID}")
