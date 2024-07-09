import sys; sys.path.append('..')
import torch
import torch.nn as nn

from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero
from typing import List, Tuple

from dataset.hgs import DiscreteTimeHeteroGraph
from utils.config import HeteroGraphConfig, MappingManager, max_seq_length
from utils.misc import calc_loss
from model.layers import LinksPredictor, PositionalEncoding, SingelGnn, get_decoder_by_choice, decode


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
                 is_seq_pred: bool = False,
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

        self.is_seq_pred = is_seq_pred
        self.hidden_dim = hidden_dim

        # EMBD
        if self.is_seq_pred:
            # +1 for PAD
            self.module_dict_embedding = nn.ModuleDict({
                node_type: nn.Embedding(MappingManager.node_type_to_node_num[node_type] + 1, self.hidden_dim) \
                for node_type in self.node_types if node_type != 'admission'
            })

            self.module_dict_ffc = nn.ModuleDict({
                node_type: nn.Linear(self.hidden_dim, MappingManager.node_type_to_node_num[node_type] + 1)
                for node_type in self.node_types if node_type != 'admission'
            })

            # share weights
            for node_type in self.node_types:
                if node_type == 'admission':
                    continue
                else:
                    self.module_dict_ffc[node_type].weight = self.module_dict_embedding[node_type].weight
        else:
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
                if self.is_seq_pred:
                    node_feat = node_feat_dec[user_node_type].unsqueeze(-2).repeat(1, 1, max_seq_length, 1)
                    batch_size = node_feat.size(1)
                    node_feat = self.module_dict_ffc[item_node_type](node_feat.view(-1, self.hidden_dim))
                    scores = node_feat.view(self.max_timestep, batch_size, max_seq_length, -1)
                    dict_every_day_pred[item_node_type] = {
                        "scores": scores,  # [T, B, X, C]
                        "labels": torch.stack([hg[edge_type].seq_labels for hg in hgs], dim=0),  # [T, B, X]
                        "indices": [hg[edge_type].labels_index for hg in hgs]
                    }
                else:
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


if __name__ == "__main__":
    node_types, edge_types = HeteroGraphConfig.use_all_edge_type()
    model = BackBone(
        max_timestep=20,
        gnn_type="GENConv",
        node_types=node_types,
        edge_types=edge_types,
        num_decoder_layers=6,
        neg_smp_strategy=0,
        hidden_dim=32,
        is_seq_pred=True
    )

    train_set = DiscreteTimeHeteroGraph(root_path=r"..\dataset\batch_size_128", usage="train")
    dict_every_day_pred = model(train_set[0])
    # loss = calc_loss(dict_every_day_pred, node_types, device=torch.device('cpu'), is_seq_pred=True)
    # print(loss)
