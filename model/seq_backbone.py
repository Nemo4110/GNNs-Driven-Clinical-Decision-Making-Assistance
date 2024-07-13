import sys; sys.path.append('..')
import torch
import torch.nn as nn

from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero
from typing import List
from dataset.hgs import DiscreteTimeHeteroGraph
from dataset.adm_to_hg import OneAdmOneHetero, collect_fn
from utils.config import GNNConfig, HeteroGraphConfig
from utils.misc import sequence_mask
from model.layers import SingelGnn


class SeqBackBone(nn.Module):
    def __init__(self, h_dim: int, gnn_conf: GNNConfig):
        """model for medications recommendation sequential predict task
        """
        super().__init__()

        self.h_dim = h_dim
        self.gnn_conf = gnn_conf

        self.node_types = self.gnn_conf.node_types
        self.edge_types = self.gnn_conf.edge_types

        self.med_vocab_size = self.gnn_conf.mapper.node_type_to_node_num['drug']
        self.itm_vocab_size = self.gnn_conf.mapper.node_type_to_node_num['labitem']

        self.d_PAD = self.med_vocab_size + 2
        self.l_PAD = self.itm_vocab_size + 2

        self.d_SOS = self.med_vocab_size
        self.l_SOS = self.itm_vocab_size

        # Embedding
        # + 3 for SOS, EOS, PAD
        self.d_emb = nn.Embedding(self.med_vocab_size + 3, self.h_dim, padding_idx=self.d_PAD)  # lab-items
        self.l_emb = nn.Embedding(self.itm_vocab_size + 3, self.h_dim, padding_idx=self.l_PAD)  # drugs / medications
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

        # --- Encode ---
        self.patient_condition_encoder = nn.TransformerEncoderLayer(d_model=self.h_dim, nhead=1, batch_first=True)

        self.gnn = SingelGnn(hidden_dim=self.h_dim,
                             gnn_type=self.gnn_conf.gnn_type, gnn_layer_num=self.gnn_conf.gnn_layer_num)
        self.gnn = to_hetero(self.gnn, metadata=(self.gnn_conf.node_types, self.gnn_conf.edge_types))

        self.l_encoder = nn.TransformerEncoderLayer(d_model=self.h_dim, nhead=1, batch_first=True)  # 为方便，暂设nhead=1
        self.d_encoder = nn.TransformerEncoderLayer(d_model=self.h_dim, nhead=1, batch_first=True)
        # some cross-seq attention encoder? NO. cross-seq info agg is done by gnn

        # --- Decode ---
        self.d_decoder = nn.TransformerDecoderLayer(d_model=self.h_dim, nhead=1, batch_first=True)

        self.d_proj = nn.Linear(self.h_dim, self.med_vocab_size, bias=False)
        self.d_proj.weight = self.d_emb.weight  # weight sharing

    def forward(self, batch_hgs: List[List[HeteroData]],
                batch_d: torch.tensor, batch_d_mask: torch.tensor):
        B = len(batch_hgs)
        max_adm_len = batch_d.size(1)

        # --- ENCODE ---
        # encode输入排除最后一天（最后一天不需要预测下一天了）
        enc_input = [hgs[:-1] for hgs in batch_hgs]
        patients_condition = self.encode(enc_input)

        # --- DECODE ---
        # 训练时的INPUT:
        #   1. 每一天的最新病情信息
        #       （已过滤仅住院1天的记录，因为第1天之前无信息可获取
        #   2. 每一天中，以<bos> / SOS 这种标识序列开头的特殊token，加上像左偏移[..., :-1]（除了最后一个<eos> / EOS）
        #
        # decode的预测从第二天开始
        bos      = torch.full((B, max_adm_len-1, 1), self.d_SOS,     device=batch_d.device)
        bos_mask = torch.full((B, max_adm_len-1, 1), False, device=batch_d.device, dtype=torch.bool)

        # 从第二天开始预测，[..., :-1] 将每天的序列往前偏移1位，以便和bos拼接（强制教学）
        dec_input      = torch.cat((bos,           batch_d[:, 1:, :-1]), dim=-1)
        dec_input_mask = torch.cat((bos_mask, batch_d_mask[:, 1:, :-1]), dim=-1)

        dec_output = self.decode(dec_input, dec_input_mask, patients_condition)

        logits = self.d_proj(dec_output)  # 将最后一维映射回 med_vocab_size + 3，方便softmax做预测

        return logits

    def gnn_enc(self, batch_hgs):
        """使用GNN得到患者在住院期间每天的病情表示"""
        B = len(batch_hgs)
        adm_lens = [len(hgs) for hgs in batch_hgs]
        max_adm_len = max(adm_lens)

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

        batch_packed_x_collection = [hgs.collect('x') for hgs in batch_packed_hgs]

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
        patients_condition = pad_sequence(patients_condition, batch_first=True, padding_value=0)
        padding_mask = sequence_mask(patients_condition, valid_len=torch.tensor(adm_lens))

        return patients_condition, padding_mask  # (B, T, h_dim), (B, T)

    def encode(self, batch_hgs):
        # batch_l, batch_l_mask, batch_l_lens,
        # batch_d, batch_d_mask, batch_d_lens):

        # TODO: 消融实验用——不使用GNN，使用药物、检验项目序列seq建模患者病情表示
        # B, max_adm_len, l_max_num = batch_l.size()
        # B, max_adm_len, d_max_num = batch_d.size()
        #
        # # (B, T, X) -> (B, T, X, h_dim)
        # batch_l_emd = self.l_emb(batch_l)
        # batch_d_emd = self.d_emb(batch_d)
        #
        # # (B, T, X, h_dim) -> (B * T, X, h_dim) for encoding
        # batch_l_emd_flat = batch_l_emd.view(B * max_adm_len, l_max_num, -1)
        # batch_d_emd_flat = batch_d_emd.view(B * max_adm_len, d_max_num, -1)
        #
        # batch_l_mask_flat = batch_l_mask.view(B * max_adm_len, l_max_num)
        # batch_d_mask_flat = batch_l_mask.view(B * max_adm_len, d_max_num)
        #
        # # 每一天内序列的自注意力
        # batch_l_emd_flat = self.l_encoder(batch_l_emd_flat, src_key_padding_mask=batch_l_mask_flat)
        # batch_d_emd_flat = self.d_encoder(batch_d_emd_flat, src_key_padding_mask=batch_d_mask_flat)
        #
        # batch_l_emd = batch_l_emd_flat.view(B, max_adm_len, l_max_num, -1)
        # batch_d_emd = batch_d_emd_flat.view(B, max_adm_len, d_max_num, -1)

        # --- GNN ---
        # 不同序列之间的信息聚合由GNN完成
        patients_condition, padding_mask = self.gnn_enc(batch_hgs)  # (B, T, h_dim)

        # 使用自注意力机制，为每一个天的患者病情表示，聚合 之前天 的病情信息 （即不同天之间的注意力）
        sequence_mask = nn.Transformer.generate_square_subsequent_mask(patients_condition.size(1))
        sequence_mask = sequence_mask.to(patients_condition.device)  # same device
        patients_condition = self.patient_condition_encoder(
            patients_condition, src_mask=sequence_mask, src_key_padding_mask=padding_mask)

        return patients_condition

    def decode(self,
               dec_input, dec_input_mask,
               patients_condition):
        """从入院的第2天（idx=1）开始预测

        训练时：
            dec_input，每天的药物序列开头应为BOS，直到EOS（不包括EOS）
            对应的label为，从第一个实际的药物开始，直到EOS（包括EOS），也就是self.forward得到的原始batch_d
        测试时（此时B=1）：
            一开始的dec_input，是一个形状（1，T，1）的全BOS tensor，
        """

        B, max_adm_len, d_max_num = dec_input.size()

        dec_input = self.d_emb(dec_input)  # (B, T, X, h_dim)
        dec_input_flat = dec_input.view(B * max_adm_len, d_max_num, -1)  # (B*T, X, h_dim)
        dec_input_mask_flat = dec_input_mask.view(B * max_adm_len, d_max_num)  # (B*T, X)

        # 防止当前位置看到之后位置的信息
        dec_input_sequence_mask = nn.Transformer.generate_square_subsequent_mask(dec_input_mask_flat.size(1))
        dec_input_sequence_mask = dec_input_sequence_mask.to(dec_input.device)  # same device

        # 来自encoder的输出，作为memory
        patients_condition = patients_condition.unsqueeze(dim=2)  # (B, T, h_dim) -> (B, T, 1, h_dim)
        patients_condition_flat = patients_condition.view(B * max_adm_len, 1, -1)  # -> (B*T, 1, h_dim)

        dec_output = self.d_decoder(tgt=dec_input_flat, memory=patients_condition_flat,
                                    tgt_mask=dec_input_sequence_mask, tgt_key_padding_mask=dec_input_mask_flat)
        dec_output = dec_output.view(B, max_adm_len, d_max_num, -1)  # reshape 回去
        return dec_output


if __name__ == "__main__":
    # test forward method
    test_dataset = OneAdmOneHetero(r"..\data", "test")
    test_dataloader = DataLoader(test_dataset, batch_size=16, collate_fn=collect_fn, pin_memory=True)

    node_types, edge_types = HeteroGraphConfig.use_all_edge_type()
    gnn_conf = GNNConfig("GINEConv", 3, node_types, edge_types)
    model = SeqBackBone(h_dim=64, gnn_conf=gnn_conf)

    for batch in test_dataloader:
        batch_hgs, \
            batch_l, batch_l_mask, batch_l_lens, \
            batch_d, batch_d_mask, batch_d_lens = batch

        logits = model(batch_hgs, batch_d, batch_d_mask)

        break