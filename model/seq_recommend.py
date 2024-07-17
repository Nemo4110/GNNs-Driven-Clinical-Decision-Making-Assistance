import sys; sys.path.append('..')
import torch
import torch.nn as nn

from typing import List, Dict, Tuple
from torch_geometric.data import HeteroData

from model.layers import LinksPredictor, PositionalEncoding, decode, get_decoder_by_choice
from dataset.hgs import DiscreteTimeHeteroGraph
from utils.config import MappingManager, HeteroGraphConfig
from deprecated import deprecated


@deprecated("This model is not suitable for current ond adm on hg dataset")
class SeqRecommend(nn.Module):
    def __init__(self,
                 max_timestep: int,
                 neg_smp_strategy: int,

                 node_types: List[str],
                 edge_types: List[Tuple[str, str, str]],

                 decoder_choice: str = "TransformerDecoder",
                 num_layers: int = 6,
                 hidden_dim: int = 128, **kwargs):
        super().__init__()

        self.max_timestep = max_timestep
        self.neg_smp_strategy = neg_smp_strategy
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim

        self.embedding = nn.ModuleDict({
            node_type: nn.Embedding(MappingManager.node_type_to_node_num[node_type], self.hidden_dim) \
            for node_type in self.node_types if node_type != 'admission'
        })

        self.node_feat_proj = nn.ModuleDict({
            node_type: nn.Linear(in_features=MappingManager.node_type_to_node_feat_dim_in[node_type], out_features=self.hidden_dim) \
            for node_type in self.node_types
        })

        self.position_encoding = PositionalEncoding(hidden_dim=self.hidden_dim, max_timestep=self.max_timestep)
        self.decoder = nn.ModuleDict({
            node_type: get_decoder_by_choice(choice=decoder_choice, hidden_dim=hidden_dim, num_layers=num_layers) \
            for node_type in self.node_types
        })

        self.predictor = nn.ModuleDict({
            node_type: LinksPredictor(hidden_dim=self.hidden_dim) \
            for node_type in self.node_types if node_type != "admission"
        })

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.shape) > 1:
                nn.init.xavier_normal_(weight)

    def forward(self, hg: HeteroData):
        hgs = [DiscreteTimeHeteroGraph.get_subgraph_by_timestep(hg, timestep=t, neg_smp_strategy=self.neg_smp_strategy)
               for t in range(self.max_timestep)]

        list_dict_node_feats = [{
            node_type: self.node_feat_proj[node_type](hg[node_type].x) + self.embedding[node_type](hg[node_type].node_id) \
                if node_type != 'admission' \
                else self.node_feat_proj[node_type](hg[node_type].x) \
            for node_type in self.node_types
        } for hg in hgs]
        list_edge_index_dict = [{
            edge_type: edge_index \
            for edge_type, edge_index in hg.edge_index_dict.items() if edge_type in self.edge_types
        } for hg in hgs]

        dict_node_feat = {
            node_type: torch.stack([dict_node_feats[node_type] for dict_node_feats in list_dict_node_feats]) \
            for node_type in self.node_types
        }

        # without going through GNN

        # decode
        for node_type, node_feat in dict_node_feat.items():
            node_feat_ori = node_feat[0].unsqueeze(0)
            node_feat = self.position_encoding(node_feat)  # Add position encoding
            dict_node_feat[node_type] = decode(decoder=self.decoder[node_type], input_seq=node_feat, h_0=node_feat_ori)

        # Link predicting:
        dict_every_day_pred = {}
        for node_type in self.node_types:
            if node_type == "admission":  # admission is not the goal of prediction
                continue
            elif node_type == "drug":
                dict_every_day_pred[node_type] = {
                    "scores": [self.predictor[node_type](dict_node_feat["admission"][curr_timestep],
                                                         dict_node_feat["drug"][curr_timestep],
                                                         hg.labels4drug_index) \
                               for curr_timestep, hg in enumerate(hgs)],
                    "labels":  [hg.labels4drug       for hg in hgs],
                    "indices": [hg.labels4drug_index for hg in hgs]
                }
            elif node_type == "labitem":
                dict_every_day_pred[node_type] = {
                    "scores": [self.predictor[node_type](dict_node_feat["admission"][curr_timestep],
                                                         dict_node_feat["labitem"][curr_timestep],
                                                         hg.lables4item_index) \
                               for curr_timestep, hg in enumerate(hgs)],
                    "labels":  [hg.lables4item       for hg in hgs],
                    "indices": [hg.lables4item_index for hg in hgs]
                }
            else:
                raise f"check the node_types config! curr: {self.node_types}"

        return dict_every_day_pred


if __name__ == '__main__':
    node_types, edge_types = HeteroGraphConfig.use_all_edge_type()
    seq_model = SeqRecommend(
        max_timestep=20,
        neg_smp_strategy=0,
        node_types=node_types,
        edge_types=edge_types,
        decoder_choice="TransformerDecoder",
        num_layers=6
    )
    print(seq_model)

