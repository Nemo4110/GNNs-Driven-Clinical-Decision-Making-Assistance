import argparse
import os
import pickle
import pandas as pd
import torch
import utils.constant as constant

from d2l import torch as d2l
from typing import List

from dataset.unified import SourceDataFrames, OneAdmOneHG
from model.backbone import BackBoneV2
from utils.best_thresholds import BestThresholdLoggerV2
from utils.misc import get_latest_model_ckpt, get_latest_threshold
from utils.config import HeteroGraphConfig, GNNConfig
from utils.metrics import convert2df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # following arguments are model settings
    parser.add_argument("--gnn_type",                                 default="GENConv")
    parser.add_argument("--gnn_layer_num",                type=int,   default=3)
    parser.add_argument("--num_encoder_layers",           type=int,   default=3)
    parser.add_argument("--decoder_choice",                           default="TransformerDecoder")
    parser.add_argument("--hidden_dim",                   type=int,   default=256)
    parser.add_argument("--lr",                           type=float, default=0.001)
    parser.add_argument("--is_gnn_only",      action="store_true",    default=False,                help="whether to only use GNN")
    # TODO: 增加只使用GNN的模型（消融）

    # Paths
    parser.add_argument("--root_path_dataset",  default=constant.PATH_MIMIC_III_ETL_OUTPUT, help="path where dataset directory locates")  # in linux
    parser.add_argument("--path_dir_model_hub", default=r"./model/hub",                     help="path where models save")
    parser.add_argument("--path_dir_results",   default=r"./results",                       help="path where results save")
    parser.add_argument("--path_dir_thresholds",default=r"./thresholds",                    help="path where thresholds save")

    # Experiment settings
    parser.add_argument("--item_type",                              default="MIX")
    parser.add_argument("--goal",                                   default="drug",     help="the goal of the recommended task, in ['drug', 'labitem']")
    parser.add_argument("--epochs",                       type=int, default=5)
    parser.add_argument("--train",            action="store_true",  default=False)
    parser.add_argument("--test",             action="store_true",  default=False)
    parser.add_argument("--test_model_state_dict",                  default=None,      help="test only model's state_dict file name")  # must be specified when --train=False!
    parser.add_argument("--model_ckpt",                             default=None,      help="the .pt filename where stores the state_dict of model")
    parser.add_argument("--use_gpu",          action="store_true",  default=False)
    parser.add_argument("--batch_size",                   type=int, default=32)

    args = parser.parse_args()

    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    sources_dfs = SourceDataFrames(args.root_path_dataset)

    if args.item_type == "MIX":
        node_types, edge_types = HeteroGraphConfig.use_all_edge_type()
    else:
        node_types, edge_types = HeteroGraphConfig.use_one_edge_type(item_type=args.item_type)
    gnn_conf = GNNConfig(args.gnn_type, args.gnn_layer_num, node_types, edge_types)
    model = BackBoneV2(sources_dfs, args.goal, args.hidden_dim, gnn_conf, device, args.num_encoder_layers).to(device)

    if args.train:
        train_dataset = OneAdmOneHG(sources_dfs, "train")
        valid_dataset = OneAdmOneHG(sources_dfs, "val")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        for epoch in range(args.epochs):
            if args.use_gpu:
                torch.cuda.empty_cache()
            train_metric = d2l.Accumulator(2)  # train loss, iter num
            model.train()
            for i, hg in enumerate(train_dataset):
                hg = hg.to(device)
                logits, labels = model(hg)
                loss = BackBoneV2.get_loss(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                with torch.no_grad():
                    train_metric.add(loss.detach().item(), 1)
                    print(f'#{i/len(train_dataset)*100:02.3f}%, '
                          f'train loss: {loss.item():.4f}', end='\r')
            print(f"epoch #{epoch:02}, train loss: {train_metric[0] / train_metric[1]:.4f}")

            if args.use_gpu:
                torch.cuda.empty_cache()
            valid_metric = d2l.Accumulator(2)
            model.eval()
            with torch.no_grad():
                for i, hg in enumerate(valid_dataset):
                    logits, labels = model(hg)
                    loss = BackBoneV2.get_loss(logits, labels)
                    valid_metric.add(loss.detach().item(), 1)
                    print(f'#{i/len(valid_dataset)*100:02.3f}%, '
                          f'valid loss: {loss.item():.4f}', end='\r')
                valid_loss = valid_metric[0] / valid_metric[1]
                print(f"epoch #{epoch:02}, valid loss: {valid_loss:.4f}")

        # save trained model (loss if valid loss)
        os.makedirs(args.path_dir_model_hub, exist_ok=True)
        model_name = f"loss_{valid_loss:.4f}_{model.__class__.__name__}_goal_{args.goal}.pt"
        torch.save(model.state_dict(), os.path.join(args.path_dir_model_hub, model_name))

    if args.test:
        test_dataset = OneAdmOneHG(sources_dfs, "test")

        if not args.train:
            # auto load latest save model from hub
            if not args.model_ckpt:
                ckpt_filename = get_latest_model_ckpt(args.path_dir_model_hub)
            else:
                ckpt_filename = args.model_ckpt
            assert ckpt_filename is not None
            print(f"using saved model: {ckpt_filename}")
            sd_path = os.path.join(args.path_dir_model_hub, ckpt_filename)
            sd = torch.load(sd_path, map_location=device)
            model.load_state_dict(sd)
        else:
            ckpt_filename = model_name

        # TODO：用0.5作为固定阈值试试看（模型输出的scores/logits在预测时都会过sigmoid）
        model.eval()
        with torch.no_grad():
            collector: List[pd.DataFrame] = []
            for idx, hg in enumerate(test_dataset):
                hg = hg.to(device)
                logits, labels = model(hg)

                # 把预测结果全部收集成DataFrame，后面再单独写notebook/脚本进行细致的指标计算
                collector.append(convert2df(logits, labels))

        results: pd.DataFrame = pd.concat(collector, axis=0)
        results.to_csv(os.path.join(args.path_dir_results, f"{ckpt_filename}.csv.gz"), compression='gzip')
