import argparse
import os
import pandas as pd
import torch
import torch.nn.functional as F
import utils.constant as constant

from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from dataset.adm_to_hg import OneAdmOneHetero, collect_fn
from model.seq_backbone import SeqBackBone
from model.layers import MaskedSoftmaxCELoss
from utils.metrics import calc_metrics_for_curr_adm
from utils.misc import get_latest_model_ckpt
from utils.config import HeteroGraphConfig, GNNConfig
from d2l import torch as d2l
from typing import List


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ********* Hyperparams ********* #
    # following arguments are model settings

    parser.add_argument("--gnn_type",                                 default="GENConv")
    parser.add_argument("--gnn_layer_num",                type=int,   default=2)
    parser.add_argument("--num_decoder_layers",           type=int,   default=6)
    parser.add_argument("--decoder_choice",                           default="TransformerDecoder")
    parser.add_argument("--hidden_dim",                   type=int,   default=64)
    parser.add_argument("--lr",                           type=float, default=0.0003)

    # Paths
    parser.add_argument("--root_path_dataset",  default=constant.PATH_MIMIC_III_HGS_OUTPUT, help="path where dataset directory locates")  # in linux
    parser.add_argument("--path_dir_model_hub", default=r"./model/hub",                     help="path where models save")
    parser.add_argument("--path_dir_results",   default=r"./results",                       help="path where results save")

    # Experiment settings
    parser.add_argument("--task",                                   default="MIX",     help="the goal of the recommended task, in ['MIX', 'drug', 'labitem']")
    parser.add_argument("--epochs",                       type=int, default=10)
    parser.add_argument("--train",            action="store_true",  default=False)
    parser.add_argument("--test",             action="store_true",  default=False)
    parser.add_argument("--model_ckpt",                             default=None,      help="the .pt filename where stores the state_dict of model")
    parser.add_argument("--use_gpu",          action="store_true",  default=False)
    parser.add_argument("--batch_size",                   type=int, default=16)

    args = parser.parse_args()

    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    node_types, edge_types = HeteroGraphConfig.use_all_edge_type()
    gnn_conf = GNNConfig("GINEConv", 3, node_types, edge_types)
    model = SeqBackBone(h_dim=args.hidden_dim, gnn_conf=gnn_conf, device=device).to(device)

    if args.train:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        loss_f = MaskedSoftmaxCELoss()

        train_dataset = OneAdmOneHetero(args.root_path_dataset, "train")
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      collate_fn=collect_fn, pin_memory=True)
        for epoch in tqdm(range(args.epochs)):
            # 清一下占用VRAM的临时变量
            if args.use_gpu:
                torch.cuda.empty_cache()

            t_loop = tqdm(train_dataloader, leave=False)
            metric = d2l.Accumulator(2)  # 训练损失总和，token数量
            for batch in t_loop:
                batch_hgs, \
                    batch_l, batch_l_mask, batch_l_lens, \
                    batch_d, batch_d_mask, batch_d_lens = batch

                B, max_adm_len, d_max_num = batch_d.size()

                # to device
                for i, hgs in enumerate(batch_hgs):
                    batch_hgs[i] = [hg.to(device) for hg in hgs]
                batch_d = batch_d.to(device)
                batch_d_mask = batch_d_mask.to(device)
                batch_d_lens = batch_d_lens.to(device)

                # 从第二天开始
                label = (batch_d[:, 1:, :]).contiguous()
                d_valid_len = (batch_d_lens[:, 1:]).contiguous()

                optimizer.zero_grad()

                logits = model(batch_hgs, batch_d, batch_d_mask)

                loss = loss_f(logits.view(B * (max_adm_len - 1), d_max_num, -1),
                              label.view(B * (max_adm_len - 1), d_max_num), valid_len=d_valid_len.view(-1))
                loss.sum().backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                optimizer.step()

                torch.cuda.synchronize()

                num_tokens = d_valid_len.detach().sum().item()
                metric.add(loss.detach().sum().item(), num_tokens)

                t_loop.set_postfix_str(f'\033[32m loss: {metric[0] / metric[1]:.4f}/token on {str(device)} \033[0m')

        # save model parameters
        torch.save(model.state_dict(), os.path.join(args.path_dir_model_hub,
                                                    f"loss_{metric[0] / metric[1]:.4f}_{model.__class__.__name__}"))

    if args.test:
        test_dataset = OneAdmOneHetero(args.root_path_dataset, "test")
        test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=collect_fn, pin_memory=True)

        if not args.train:
            # auto load latest save model from hub
            if not args.model_ckpt:
                sd_path = os.path.join(args.path_dir_model_hub, f"{get_latest_model_ckpt(args.path_dir_model_hub)}")
            else:
                sd_path = os.path.join(args.path_dir_model_hub, f"{args.model_ckpt}")
            sd = torch.load(sd_path, map_location=device)
            model.load_state_dict(sd)

        model.eval()
        with torch.no_grad():
            results: List[pd.DataFrame] = []
            for idx, (batch) in tqdm(enumerate(test_dataloader)):  # 一个一个患者
                batch_hgs, \
                    batch_l, batch_l_mask, batch_l_lens, \
                    batch_d, batch_d_mask, batch_d_lens = batch

                B, max_adm_len, d_max_num = batch_d.size()  # B = 1

                # to device
                for i, hgs in enumerate(batch_hgs):
                    batch_hgs[i] = [hg.to(device) for hg in hgs]
                batch_d = batch_d.to(device)
                batch_d_mask = batch_d_mask.to(device)
                batch_d_lens = batch_d_lens.to(device)

                # 先获取患者在每天的病情表示
                enc_input = [hgs[:-1] for hgs in batch_hgs]  # encode输入排除最后一天（最后一天不需要预测下一天了）
                patients_condition = model.encode(enc_input)

                # 一天一天地生成预测
                all_day_preds: List[List[int]] = []
                all_day_probs: List[List[torch.tensor]] = []
                all_day_labels: List[List[int]] = []
                for cur_day in tqdm(range(1, max_adm_len), leave=False):  # 从第二天开始预测[1, max_adm_len)
                    # 构建当天的起始<bos> / SOS
                    dec_input = torch.full((1, 1, 1), model.d_SOS, device=device, dtype=torch.int32)
                    cur_condition = patients_condition[:, cur_day, :].unsqueeze(1)  # (1, h_dim) -> (1, 1, h_dim)

                    cur_day_d_length = batch_d_lens[0, cur_day].int().item()
                    cur_day_labels = batch_d[0, cur_day, :cur_day_d_length].tolist()

                    cur_day_preds = []
                    cur_day_probs = []
                    for _ in range(d_max_num):  # 一个一个地生成当天的药物预测
                        dec_output = model.decode(dec_input, None, cur_condition)
                        logits = model.d_proj(dec_output)  # (1, 1, ?, d_voc_size), B = 1

                        # TODO: beam search ?
                        prob = F.softmax(logits[:, :, -1:, :], dim=-1)  # (1, 1, 1, d_voc_size + 3)
                        pred = torch.argmax(prob, dim=-1).type(torch.int32)  # (1, 1, 1)

                        # update 下一次的输入
                        dec_input = torch.cat([dec_input, pred], dim=-1)

                        if pred.view(-1).item() == model.d_EOS:
                            break

                        cur_day_preds.append(pred.view(-1).item())
                        cur_day_probs.append(prob.view(-1))  # TODO: 这里的probs应为softmax之后的logits？还是原始的logits？

                    all_day_preds.append(cur_day_preds)
                    all_day_probs.append(cur_day_probs)
                    all_day_labels.append(cur_day_labels)

                # TODO: calculate metrics for current admission
                result_cur_adm = calc_metrics_for_curr_adm(idx, all_day_preds, all_day_probs, all_day_labels)
                results.append(result_cur_adm)

            results: pd.DataFrame = pd.concat(results)
            results.to_csv()
