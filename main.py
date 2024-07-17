import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.constant as constant

from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader

from dataset.adm_to_hg import OneAdmOneHetero, collect_hgs
from model.backbone import BackBoneV2
from model.layers import MaskedBCEWithLogitsLoss
from utils.best_thresholds import BestThresholdLoggerV2
from utils.misc import calc_loss, node_type_to_prefix, get_latest_model_ckpt
from utils.config import HeteroGraphConfig, GNNConfig
from utils.metrics import calc_metrics_for_curr_adm_v2
from typing import List


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ********* Hyperparams ********* #
    # following arguments are model settings

    # NOTE: when max_timestep set to 30 or 50,
    #       would trigger the assert error "last timestep has not labels!"
    #       in `get_subgraph_by_timestep` (bigger max_timestep can be support in future)
    parser.add_argument("--gnn_type",                                 default="GENConv")
    parser.add_argument("--gnn_layer_num",                type=int,   default=2)
    parser.add_argument("--num_decoder_layers",           type=int,   default=6)
    parser.add_argument("--num_encoder_layers",           type=int,   default=6)
    parser.add_argument("--decoder_choice",                           default="TransformerDecoder")
    parser.add_argument("--hidden_dim",                   type=int,   default=256)
    parser.add_argument("--lr",                           type=float, default=0.0003)
    parser.add_argument("--use_seq_rec",      action="store_true",    default=False,                help="whether to use sequntial recommendation (without GNN)")
    parser.add_argument("--is_gnn_only",      action="store_true",    default=False,                help="whether to only use GNN")
    parser.add_argument("--is_seq_pred",      action="store_true",    default=False,                help="whether to enable seq pred")

    # Paths
    parser.add_argument("--root_path_dataset",  default=constant.PATH_MIMIC_III_HGS_OUTPUT, help="path where dataset directory locates")  # in linux
    parser.add_argument("--path_dir_model_hub", default=r"./model/hub",                     help="path where models save")
    parser.add_argument("--path_dir_results",   default=r"./results",                       help="path where results save")
    parser.add_argument("--path_dir_thresholds",default=r"./thresholds",                    help="path where thresholds save")

    # Experiment settings
    parser.add_argument("--task",                                   default="MIX",     help="the goal of the recommended task, in ['MIX', 'drug', 'labitem']")
    parser.add_argument("--epochs",                       type=int, default=3)
    parser.add_argument("--train",            action="store_true",  default=False)

    parser.add_argument("--test",             action="store_true",  default=False)
    parser.add_argument("--test_model_state_dict",                  default=None,      help="test only model's state_dict file name")  # must be specified when --train=False!
    parser.add_argument("--model_ckpt",                             default=None,      help="the .pt filename where stores the state_dict of model")
    parser.add_argument("--test_num",                     type=int, default=-1,        help="number of testing")

    parser.add_argument("--use_gpu",          action="store_true",  default=False)
    parser.add_argument("--batch_size",                   type=int, default=32)

    args = parser.parse_args()

    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # 数据集位置

    # Heterogeneous graph config
    # if args.task == "MIX":
    #     node_types, edge_types = HeteroGraphConfig.use_all_edge_type()
    # else:
    #     node_types, edge_types = HeteroGraphConfig.use_one_edge_type(item_type=args.task)
    node_types, edge_types = HeteroGraphConfig.use_all_edge_type()

    # model
    gnn_conf = GNNConfig("GINEConv", 3, node_types, edge_types)
    model = BackBoneV2(args.hidden_dim, gnn_conf, device, args.num_encoder_layers).to(device)

    # --- train ---
    if args.train:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        loss_f = MaskedBCEWithLogitsLoss()

        # BestThresholdLogger
        if not os.path.exists(args.path_dir_thresholds):
            os.mkdir(args.path_dir_thresholds)
        bth_logger = BestThresholdLoggerV2(args.path_dir_thresholds)

        train_dataset = OneAdmOneHetero(args.root_path_dataset, "train")
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      collate_fn=collect_hgs, pin_memory=True, shuffle=False)
        # train
        for epoch in tqdm(range(args.epochs)):
            if args.use_gpu:
                torch.cuda.empty_cache()
            t_loop = tqdm(train_dataloader, leave=False)
            for batch_hgs, batch_d_flat, adm_lens in t_loop:
                input_hgs = [hgs[:-1] for hgs in batch_hgs]

                # to device
                for i, hgs in enumerate(input_hgs):
                    input_hgs[i] = [hg.to(device) for hg in hgs]
                batch_d_flat = batch_d_flat.to(device)
                adm_lens = adm_lens.to(device)

                logits = model(input_hgs)
                labels = batch_d_flat[:, 1:, :]
                can_prd_days = adm_lens - 1  # 能预测的天数为住院时长减去第一天
                loss = loss_f(logits, labels, adm_lens=can_prd_days)

                optimizer.zero_grad()
                loss.sum().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                optimizer.step()  # optimizing

                with torch.no_grad():
                    # TODO: at last epoch, log the best_threshold
                    if epoch == (args.epochs - 1):
                        bth_logger.log_cur_batch(logits, labels, can_prd_days)
                t_loop.set_postfix_str(f'\033[32m loss: {loss.sum(-1).mean().item():.4f} on {str(device)} \033[0m')

        # train done
        bth_logger.save()

        # save trained model
        if not os.path.exists(args.path_dir_model_hub):
            os.mkdir(args.path_dir_model_hub)
        torch.save(model.state_dict(), os.path.join(args.path_dir_model_hub,
                                                    f"loss_{loss.sum(-1).mean().item():.4f}_{model.__class__.__name__}.pt"))

    # --- test ---
    if args.test:
        test_dataset = OneAdmOneHetero(args.root_path_dataset, "test")
        test_dataloader = DataLoader(test_dataset, batch_size=1,
                                      collate_fn=collect_hgs, pin_memory=True, shuffle=False)

        if not args.train:
            # auto load latest save model from hub
            if not args.model_ckpt:
                latest_model_ckpt = get_latest_model_ckpt(args.path_dir_model_hub)
                print(f"auto using latest ckpt: {latest_model_ckpt}")
                sd_path = os.path.join(args.path_dir_model_hub, f"{latest_model_ckpt}")
            else:
                sd_path = os.path.join(args.path_dir_model_hub, f"{args.model_ckpt}")
            sd = torch.load(sd_path, map_location=device)
            model.load_state_dict(sd)

        model.eval()
        with torch.no_grad():
            results: List[pd.DataFrame] = []
            for idx, (batch_data) in tqdm(enumerate(test_dataloader)):
                batch_hgs, batch_d_flat, adm_lens = batch_data

                B, max_adm_len, _ = batch_d_flat.size()  # B = 1

                input_hgs = [hgs[:-1] for hgs in batch_hgs]  # 输入模型的数据不包括最后一天（因为没有下一天去预测了）

                # to device
                for i, hgs in enumerate(input_hgs):
                    input_hgs[i] = [hg.to(device) for hg in hgs]
                batch_d_flat = batch_d_flat.to(device)
                adm_lens = adm_lens.to(device)

                logits = model(input_hgs)  # 生成了此患者从第二天开始的药物集合预测
                labels = batch_d_flat[:, 1:, :]

                result = calc_metrics_for_curr_adm_v2(idx, logits, labels)
                results.append(result)

            results: pd.DataFrame = pd.concat(results)
            results.to_csv()  # TODO: specify path
