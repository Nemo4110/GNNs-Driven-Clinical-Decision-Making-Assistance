import sys; sys.path.append('..')
import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import to_hetero
from typing import List
from torch.utils.data.dataloader import DataLoader
from dataset.hgs import DiscreteTimeHeteroGraph
from dataset.adm_to_hg import OneAdmOneHetero, collect_hgs, get_batch_d_seq_to_be_judged_and_01_labels
from utils.config import HeteroGraphConfig, GNNConfig
from utils.misc import sequence_mask
from model.layers import LinksPredictor, SingelGnn


class BackBoneV2(nn.Module):
    def __init__(self, h_dim: int, gnn_conf: GNNConfig, device, num_enc_layers: int = 6):
        super().__init__()
        self.h_dim = h_dim
        self.gnn_conf = gnn_conf
        self.device = device

        self.node_types = self.gnn_conf.node_types
        self.edge_types = self.gnn_conf.edge_types

        self.med_vocab_size = self.gnn_conf.mapper.node_type_to_node_num['drug']
        self.itm_vocab_size = self.gnn_conf.mapper.node_type_to_node_num['labitem']

        # Embedding
        self.d_emb = nn.Embedding(self.med_vocab_size, self.h_dim)  # lab-items
        self.l_emb = nn.Embedding(self.itm_vocab_size, self.h_dim)  # drugs / medications
        self.nid_emb = nn.ModuleDict({
            'labitem': self.l_emb,
            'drug': self.d_emb
        })

        # Projector
        # - node feature proj
        self.nf_proj = nn.ModuleDict({
            node_type: nn.Linear(self.gnn_conf.mapper.node_type_to_node_feat_dim_in[node_type], self.h_dim)
            for node_type in self.node_types
        })
        # - edge feature proj
        self.ef_proj = nn.ModuleDict({
            "_".join(edge_type): nn.Linear(self.gnn_conf.mapper.edge_type_to_edge_feat_dim_in[edge_type], self.h_dim)
            for edge_type in self.edge_types
        })

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.h_dim, nhead=8, batch_first=True)
        self.patient_condition_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_enc_layers)

        self.gnn = SingelGnn(self.h_dim, self.gnn_conf.gnn_type, self.gnn_conf.gnn_layer_num)
        self.gnn = to_hetero(self.gnn, metadata=(self.gnn_conf.node_types, self.gnn_conf.edge_types))

        # Final links predictor
        self.d_lp = LinksPredictor(self.h_dim)

    def forward(self, batch_hgs,
                batch_d_seq_to_be_judged: List[List[torch.tensor]],
                batch_01_labels: List[List[torch.tensor]] = None):
        B = len(batch_hgs)
        adm_lens = [len(hgs) for hgs in batch_hgs]
        max_adm_len = max(adm_lens)

        # 将结点&边特征映射到同一的h_dim维度
        for hgs in batch_hgs:  # 每个患者
            for hg in hgs:  # 住院的每天
                # node feature
                for node_type in self.node_types:
                    if node_type != 'admission':
                        hg[node_type].x = self.nf_proj[node_type](hg[node_type].x.float()) + \
                                          self.nid_emb[node_type](hg[node_type].node_id)
                    else:  # adm node (just 1)
                        hg[node_type].x = self.nf_proj[node_type](hg[node_type].x.float())

                # edge feature
                for edge_type in self.edge_types:
                    hg[edge_type].x = self.ef_proj["_".join(edge_type)](hg[edge_type].x.float())

        # 将每个患者每天的图打包成 mini-batch 供GNN处理
        batch_packed_hgs = [DiscreteTimeHeteroGraph.pack_batch(hgs, len(hgs)) for hgs in batch_hgs]

        batch_packed_x_collection = [packed_hgs.collect('x') for packed_hgs in batch_packed_hgs]

        batch_packed_node_feats_ori = []
        batch_packed_edge_feats_ori = []
        for x_collection in batch_packed_x_collection:
            batch_packed_node_feats_ori.append({k: x for k, x in x_collection.items() if k in self.node_types})
            batch_packed_edge_feats_ori.append({k: x for k, x in x_collection.items() if k in self.edge_types})

        patients_condition = []  # [B, ?, h_dim]
        for node_feats_ori, packed_hgs, edge_feats_ori in \
                zip(batch_packed_node_feats_ori, batch_packed_hgs, batch_packed_edge_feats_ori):  # 遍历病人
            node_feats_enc = self.gnn(node_feats_ori, packed_hgs.edge_index_dict, edge_feats_ori)  # (?, h_dim)
            patients_condition.append(node_feats_enc["admission"])  # 此时长度不一，需要PAD

        # PAD
        patients_condition = pad_sequence(patients_condition, batch_first=True, padding_value=0).to(self.device)
        padding_mask = sequence_mask(patients_condition, valid_len=torch.tensor(adm_lens, device=self.device))

        subsequent_mask = nn.Transformer.generate_square_subsequent_mask(patients_condition.size(1)).to(self.device)
        patients_condition = self.patient_condition_encoder(
            patients_condition, mask=subsequent_mask, src_key_padding_mask=padding_mask)  # (B, max_adm_len, h_dim)

        # 目前先用对应天的患者病情表示 与 对应药物Embedding进行相乘 获取分数
        logits = []
        for i in range(B):
            cur_adm_logits: List[torch.tensor] = []
            for j in range(adm_lens[i]):  # 遍历住院天
                cur_day_pc = patients_condition[i, j]  # 当前患者当前天的病情表示（使用之前天的信息）
                cur_day_logits = self.d_lp(cur_day_pc, self.d_emb(batch_d_seq_to_be_judged[i][j]))
                cur_adm_logits.append(cur_day_logits)
            logits.append(cur_adm_logits)

        if batch_01_labels is not None:
            loss = self.get_loss(logits, batch_01_labels)
            return logits, loss

        else:
            return logits

    def get_loss(self, logits, batch_01_labels):
        total_loss = torch.tensor(0.0).to(self.device)
        for cur_adm_logits, cur_adm_labels in zip(logits, batch_01_labels):
            t_cur_adm_logits = torch.cat(cur_adm_logits)
            t_cur_adm_labels = torch.cat(cur_adm_labels)
            cur_adm_loss = F.binary_cross_entropy_with_logits(t_cur_adm_logits, t_cur_adm_labels)
            total_loss += cur_adm_loss
        return total_loss  # 当前批次的总loss


if __name__ == "__main__":
    test_dataset = OneAdmOneHetero(r"..\data", "test")
    test_dataloader = DataLoader(test_dataset, batch_size=4, collate_fn=collect_hgs, pin_memory=True)

    node_types, edge_types = HeteroGraphConfig.use_all_edge_type()
    gnn_conf = GNNConfig("GINEConv", 3, node_types, edge_types)
    model = BackBoneV2(64, gnn_conf, torch.device('cpu'))

    for batch in test_dataloader:
        batch_hgs, batch_d_seq, batch_d_neg, adm_lens = batch

        # 输入要排除住院的最后一天，因为这一天后没有下一天去预测了
        input_hgs = [hgs[:-1] for hgs in batch_hgs]

        batch_d_seq_to_be_judged, batch_01_labels = \
            get_batch_d_seq_to_be_judged_and_01_labels(batch_d_seq, batch_d_neg)

        logits, loss = model(input_hgs, batch_d_seq_to_be_judged, batch_01_labels)
