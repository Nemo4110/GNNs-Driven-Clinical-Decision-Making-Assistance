import sys; sys.path.append('..')
import torch.nn as nn
import torch

from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero
from typing import List, Tuple
from torch.utils.data.dataloader import DataLoader
from dataset.hgs import DiscreteTimeHeteroGraph
from dataset.adm_to_hg import OneAdmOneHetero, collect_hgs
from utils.config import HeteroGraphConfig, MappingManager, GNNConfig
from utils.misc import sequence_mask
from model.layers import LinksPredictor, PositionalEncoding, SingelGnn, get_decoder_by_choice, decode, MaskedBCEWithLogitsLoss
from deprecated import deprecated


@deprecated("This model is not suitable for current ond adm on hg dataset")
class BackBone(nn.Module):
    def __init__(self,
                 max_timestep: int,
                 gnn_type: str,
                 neg_smp_strategy: int,

                 node_types: List[str],
                 edge_types: List[Tuple[str, str, str]],

                 decoder_choice: str = "TransformerDecoder",
                 num_decoder_layers=6,
                 hidden_dim: int = 128,
                 is_gnn_only: bool = False,
                 gnn_layer_num: int = 2):
        super().__init__()

        self.max_timestep = max_timestep
        
        self.gnn_type = gnn_type
        self.gnn_layer_num = gnn_layer_num
        self.is_gnn_only = is_gnn_only

        self.neg_smp_strategy = neg_smp_strategy

        self.node_types = node_types
        self.edge_types = edge_types

        self.decoder_choice = decoder_choice
        self.num_decoder_layers = num_decoder_layers

        self.hidden_dim = hidden_dim

        # EMBD
        # ~~Think about the first args of `nn.Embedding` here should equal to~~
        #  - max of total `HADM_ID`?
        #  - current batch_size of `HADM_ID`?
        #  √ Or just deprecate the `nn.Embedding` as we already have the node_features
        self.module_dict_embedding = nn.ModuleDict({
            node_type: nn.Embedding(MappingManager.node_type_to_node_num[node_type], self.hidden_dim) \
            for node_type in self.node_types if node_type != 'admission'
        })

        # LIKES_PREDICTOR
        self.module_dict_links_predictor = nn.ModuleDict({
            node_type: LinksPredictor(hidden_dim=self.hidden_dim) \
            for node_type in self.node_types if node_type != "admission"
        })

        # PROJ
        self.module_dict_node_feat_proj = nn.ModuleDict({
            node_type: nn.Linear(in_features=MappingManager.node_type_to_node_feat_dim_in[node_type], out_features=self.hidden_dim) \
            for node_type in self.node_types
        })
        self.module_dict_edge_feat_proj = nn.ModuleDict({
            # TypeError: module name should be a string. Got tuple
            "_".join(edge_type): nn.Linear(in_features=MappingManager.edge_type_to_edge_feat_dim_in[edge_type], out_features=self.hidden_dim) \
            for edge_type in self.edge_types
        })

        self.position_encoding = PositionalEncoding(hidden_dim=self.hidden_dim, max_timestep=self.max_timestep)

        self.gnn = SingelGnn(hidden_dim=self.hidden_dim, gnn_type=self.gnn_type, gnn_layer_num=self.gnn_layer_num)
        self.gnn = to_hetero(self.gnn, metadata=(self.node_types, self.edge_types))

        # DECODER
        self.module_dict_decoder = nn.ModuleDict({
            node_type: get_decoder_by_choice(choice=self.decoder_choice, hidden_dim=self.hidden_dim, num_layers=self.num_decoder_layers)
            for node_type in self.node_types
        })

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.shape) > 1:
                nn.init.xavier_normal_(weight)

    def forward(self, hg: HeteroData):
        num_adm = hg['admission'].node_id.size(0)

        # EMB & PRJ
        for node_type in self.node_types:
            if node_type != 'admission':
                hg[node_type].x = self.module_dict_node_feat_proj[node_type](hg[node_type].x.float()) + self.module_dict_embedding[node_type](hg[node_type].node_id)
            else:
                hg[node_type].x = self.module_dict_node_feat_proj[node_type](hg[node_type].x.float())

        for edge_type in self.edge_types:
            if 'rev' in edge_type[1]: continue
            hg[edge_type].x = self.module_dict_edge_feat_proj["_".join(edge_type)](hg[edge_type].x.float())

        hgs = [DiscreteTimeHeteroGraph.get_subgraph_by_timestep(hg, timestep=t, neg_smp_strategy=self.neg_smp_strategy)
               for t in range(self.max_timestep)]

        # pack to batch for mini-batch processing in time (days) axis
        batch_hgs = DiscreteTimeHeteroGraph.pack_batch(hgs, self.max_timestep)

        # go through gnn
        x_collection = batch_hgs.collect('x')
        node_feats_ori = {k: x for k, x in x_collection.items() if k in self.node_types}
        edge_feats_ori = {k: x for k, x in x_collection.items() if k in self.edge_types}

        # encoded node_feats, is a dict, every value has shape [T * B, H]
        node_feats_enc = self.gnn(node_feats_ori, batch_hgs.edge_index_dict, edge_feats_ori)

        # reshape (unpack)
        for node_type in node_feats_enc.keys():
            if node_type == 'admission':
                node_feats_enc[node_type] = node_feats_enc[node_type].view(-1, num_adm, self.hidden_dim)  # [T, B, H]
                node_feats_ori[node_type] = node_feats_ori[node_type].view(-1, num_adm, self.hidden_dim)  # [T, B, H]
            else:
                node_feats_enc[node_type] = node_feats_enc[node_type].view(-1, MappingManager.node_type_to_node_num[node_type], self.hidden_dim)
                node_feats_ori[node_type] = node_feats_ori[node_type].view(-1, MappingManager.node_type_to_node_num[node_type], self.hidden_dim)

        # decode
        node_feat_dec = {}
        if not self.is_gnn_only:
            for node_type, node_feat in node_feats_enc.items():
                node_feat_ori = node_feats_ori[node_type][0].unsqueeze(0)
                node_feat_pos = self.position_encoding(node_feat)
                node_feat_dec[node_type] = decode(self.module_dict_decoder[node_type], input_seq=node_feat_pos, h_0=node_feat_ori)  # update

        dict_every_day_pred = {}
        for edge_type in self.edge_types:
            if 'rev' in edge_type[1]:
                continue
            else:
                user_node_type = edge_type[0]
                item_node_type = edge_type[-1]
                dict_every_day_pred[item_node_type] = {
                    "scores": [
                        # Link predicting:
                        self.module_dict_links_predictor[item_node_type](
                            node_feat_dec[user_node_type][curr_timestep],
                            node_feat_dec[item_node_type][curr_timestep],
                            hg[edge_type].labels_index
                        ) for curr_timestep, hg in enumerate(hgs)
                    ],
                    "labels": [hg[edge_type].labels for hg in hgs],
                    "indices": [hg[edge_type].labels_index for hg in hgs]
                }

        return dict_every_day_pred


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

        self.d_proj = nn.Linear(self.h_dim, self.med_vocab_size, bias=False)

    def forward(self, batch_hgs):
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

        # 将最后的一维映射回药物集大小
        logits = self.d_proj(patients_condition.view(-1, self.h_dim))
        logits = logits.view(B, max_adm_len, self.med_vocab_size)

        return logits


if __name__ == "__main__":
    test_dataset = OneAdmOneHetero(r"..\data", "test")
    test_dataloader = DataLoader(test_dataset, batch_size=16, collate_fn=collect_hgs, pin_memory=True)

    node_types, edge_types = HeteroGraphConfig.use_all_edge_type()
    gnn_conf = GNNConfig("GINEConv", 3, node_types, edge_types)
    model = BackBoneV2(64, gnn_conf, torch.device('cpu'))
    loss_f = MaskedBCEWithLogitsLoss()

    for batch_hgs, batch_d_flat, adm_lens in test_dataloader:
        input_hgs = [hgs[:-1] for hgs in batch_hgs]
        logits = model(input_hgs)
        labels = batch_d_flat[:, 1:, :]
        can_prd_days = adm_lens - 1  # 能预测的天数为住院时长减去第一天
        loss = loss_f(logits, labels, adm_len=can_prd_days)
        print(loss.sum().item())
        break
