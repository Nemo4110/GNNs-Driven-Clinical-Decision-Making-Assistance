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


class MultiGnns(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 max_timestep: int,
                 gnn_type: str,
                 node_types: List[str],
                 edge_types: List[Tuple[str, str, str]],
                 gnn_layer_num: int = 2,
                 ):
        super().__init__()

        # RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!
        # Solution: use the `nn.ModuleList` instead of list
        self.gnns = nn.ModuleList([SingelGnn(hidden_dim=hidden_dim, gnn_type=gnn_type, gnn_layer_num=gnn_layer_num)
                                   for _ in range(max_timestep)])  # as many fc as max_timestep

        # !!! Warning: here must use `self.gnns[i]`, if we use `gnn` will cause failure of `to_hetero`,
        #              because the `gnn` are temp parameter!
        for i, gnn in enumerate(self.gnns):
            self.gnns[i] = to_hetero(gnn, metadata=(node_types, edge_types))

    def forward(self, list_dict_node_feats, list_edge_index_dict, list_dict_edge_attrs):
        for i in range(len(self.gnns)):
            list_dict_node_feats[i] = self.gnns[i](list_dict_node_feats[i],
                                                   list_edge_index_dict[i],
                                                   list_dict_edge_attrs[i])

        return list_dict_node_feats


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
            # +3 for SOS, EOS, PAD
            self.module_dict_embedding = nn.ModuleDict({
                node_type: nn.Embedding(MappingManager.node_type_to_node_num[node_type] + 3, self.hidden_dim) \
                for node_type in self.node_types if node_type != 'admission'
            })

            self.module_dict_ffc = nn.ModuleDict({
                node_type: nn.Linear(self.hidden_dim, MappingManager.node_type_to_node_num[node_type] + 3)
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

        self.gnns = MultiGnns(hidden_dim=self.hidden_dim, max_timestep=self.max_timestep,
                              gnn_type=self.gnn_type, gnn_layer_num=self.gnn_layer_num,
                              node_types=self.node_types, edge_types=self.edge_types)

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
        hgs = [DiscreteTimeHeteroGraph.get_subgraph_by_timestep(hg, timestep=t, neg_smp_strategy=self.neg_smp_strategy) for t in range(self.max_timestep)]

        list_dict_node_feats = [{
           node_type: self.module_dict_node_feat_proj[node_type](hg[node_type].x) + self.module_dict_embedding[node_type](hg[node_type].node_id) if node_type != 'admission' \
                 else self.module_dict_node_feat_proj[node_type](hg[node_type].x) \
            for node_type in self.node_types
        } for hg in hgs]
        list_edge_index_dict = [{
            edge_type: edge_index \
            for edge_type, edge_index in hg.edge_index_dict.items() if edge_type in self.edge_types
        } for hg in hgs]
        list_dict_edge_attrs = [{
            edge_type: self.module_dict_edge_feat_proj["_".join(edge_type)](hg[edge_type].x)
            for edge_type in self.edge_types
        } for hg in hgs]

        dict_node_feat_ori = {
            node_type: torch.stack([dict_node_feats[node_type] for dict_node_feats in list_dict_node_feats])
            for node_type in self.node_types
        }

        # go through gnns
        list_dict_node_feats = self.gnns(list_dict_node_feats, list_edge_index_dict, list_dict_edge_attrs)

        dict_node_feat = {
            node_type: torch.stack([dict_node_feats[node_type] for dict_node_feats in list_dict_node_feats]) \
            for node_type in self.node_types
        }

        # decode
        if not self.is_gnn_only:
            for node_type, node_feat in dict_node_feat.items():
                node_feat_ori = dict_node_feat_ori[node_type][0].unsqueeze(0)
                node_feat = self.position_encoding(node_feat)  # Add position encoding
                dict_node_feat[node_type] = decode(self.module_dict_decoder[node_type], input_seq=node_feat, h_0=node_feat_ori)  # update
                # [T, B, H]

        dict_every_day_pred = {}
        for edge_type in self.edge_types:
            if 'rev' in edge_type[1]:
                continue
            else:
                user_node_type = edge_type[0]
                item_node_type = edge_type[-1]
                if self.is_seq_pred:
                    node_feat = dict_node_feat[user_node_type].unsqueeze(-2).repeat(1, 1, max_seq_length, 1)
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
                                dict_node_feat[user_node_type][curr_timestep],
                                dict_node_feat[item_node_type][curr_timestep],
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
