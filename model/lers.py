import sys; sys.path.append('..')
import torch
import torch.nn as nn
import torch_geometric.transforms as T
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn.conv import GINEConv, GENConv, GATConv
from torch_geometric.nn import to_hetero
from torch_geometric.utils import negative_sampling
from typing import List, Tuple

from dataset.hgs import DiscreteTimeHeteroGraph
from utils.config import HeteroGraphConfig, MappingManager
from model.layers import LinksPredictor, PositionalEncoding


class SingelGnn(nn.Module):
    r"""Chosen from this cheatsheet: <https://pytorch-geometric.readthedocs.io/en/latest/notes/cheatsheet.html>"""
    def __init__(self, hidden_dim, gnn_type, gnn_layer_num: int = 2):
        super().__init__()
        assert gnn_layer_num > 0
        self.hidden_dim = hidden_dim

        if gnn_type == "GINEConv":
            self.layers = nn.ModuleList([GINEConv(nn=nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim))
                                         for _ in range(gnn_layer_num)])
        elif gnn_type == "GENConv":
            self.layers = nn.ModuleList([GENConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim, msg_norm=True)
                                         for _ in range(gnn_layer_num)])
        elif gnn_type == "GATConv":
            self.layers = nn.ModuleList([GATConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim, add_self_loops=False, edge_dim=self.hidden_dim)
                                         for _ in range(gnn_layer_num)])
        else:
            raise f"Do not support arg:gnn_type={gnn_type} now!"

    def forward(self, node_feats, edge_index, edge_attrs):
        for conv in self.layers:
            node_feats = conv(x=node_feats, edge_index=edge_index, edge_attr=edge_attrs).relu()
        return node_feats


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


class LinksPredictor(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.re_weight_a = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.re_weight_b = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)

    def forward(self, node_features_a, node_features_b, edge_label_index):
        node_features_a_selected = node_features_a[edge_label_index[0]]
        node_features_b_selected = node_features_b[edge_label_index[1]]

        node_features_a_selected = self.re_weight_a(node_features_a_selected)
        node_features_b_selected = self.re_weight_b(node_features_b_selected)

        return (node_features_a_selected * node_features_b_selected).sum(dim=-1)


class LERS(nn.Module):
    def __init__(self,
                 max_timestep: int,
                 gnn_type: str,
                 neg_smp_strategy: int,

                 node_types: List[str],
                 edge_types: List[Tuple[str, str, str]],

                 num_decoder_layers=6,
                 hidden_dim: int = 128,
                 gnn_layer_num: int = 2):
        super().__init__()

        self.max_timestep = max_timestep
        self.gnn_type = gnn_type
        self.gnn_layer_num = gnn_layer_num
        self.neg_smp_strategy = neg_smp_strategy

        self.node_types = node_types
        self.edge_types = edge_types

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
            node_type: nn.TransformerDecoder(
                nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=8, dim_feedforward=512),
                num_layers=self.num_decoder_layers
            ) for node_type in self.node_types
        })

        # LIKES_PREDICTOR
        self.module_dict_links_predictor = nn.ModuleDict({
            node_type: LinksPredictor(hidden_dim=self.hidden_dim) \
            for node_type in self.node_types if node_type != "admission"
        })

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.shape) > 1:
                nn.init.xavier_normal_(weight)

    @staticmethod
    def get_subgraph_by_timestep(hg: HeteroData, timestep: int, neg_smp_strategy: int=0):
        r"""
        Params:
            - `neg_smp_stratege`: the stratege of negative sampling
                - \in [1, 100], assume each patient is assigned with `neg_smp_stratege` negative edges
                - 0 for 1: 2 (positive: negative
                - -1 for full-set
        """
        device = hg["admission", "did", "labitem"].timestep.device

        # https://discuss.pytorch.org/t/typeerror-expected-tensoroptions-dtype-float-device-cpu-layout-strided-requires-grad-false-default-pinned-memory-false-default-memory-format-nullopt/159558
        mask4item = (hg["admission", "did", "labitem"].timestep == timestep).to(device)
        eidx4item = hg["admission", "did", "labitem"].edge_index[:, mask4item]
        ex4item   = hg["admission", "did", "labitem"].x[mask4item, :]

        mask4drug = (hg["admission", "took", "drug"].timestep == timestep).to(device)
        eidx4drug = hg["admission", "took", "drug"].edge_index[:, mask4drug]
        ex4drug   = hg["admission", "took", "drug"].x[mask4drug, :]

        sub_hg = HeteroData()

        # Nodes
        sub_hg["admission"].node_id = hg["admission"].node_id.clone()
        sub_hg["admission"].x       = hg["admission"].x.clone().float()

        sub_hg["labitem"].node_id = hg["labitem"].node_id.clone()
        sub_hg["labitem"].x       = hg["labitem"].x.clone().float()

        sub_hg["drug"].node_id = hg["drug"].node_id.clone()
        sub_hg["drug"].x       = hg["drug"].x.clone().float()

        # Edges
        sub_hg["admission", "did", "labitem"].edge_index = eidx4item.clone()
        sub_hg["admission", "did", "labitem"].x          = ex4item.clone().float()

        sub_hg["admission", "took", "drug"].edge_index = eidx4drug.clone()
        sub_hg["admission", "took", "drug"].x          = ex4drug.clone().float()

        assert timestep < torch.max(hg["admission", "did", "labitem"].timestep), "last timestep has not labels!"
        assert timestep < torch.max(hg["admission", "took", "drug"].timestep),   "last timestep has not labels!"

        mask_next_t4item = (hg["admission", "did", "labitem"].timestep == (timestep+1)).to(device)
        sub_hg.labels4item_pos_index = hg["admission", "did", "labitem"].edge_index[:, mask_next_t4item].clone()
        if neg_smp_strategy == 0:
            sub_hg.labels4item_neg_index = negative_sampling(
                sub_hg.labels4item_pos_index,
                num_neg_samples=sub_hg.labels4item_pos_index.shape[1] * 2,
                num_nodes=(sub_hg["admission"].node_id.shape[0], sub_hg["labitem"].node_id.shape[0])
            ).to(device)
        elif 1 <= neg_smp_strategy <= 100:
            sub_hg.labels4item_neg_index = negative_sampling(
                sub_hg.labels4item_pos_index,
                num_neg_samples=sub_hg["admission"].node_id.shape[0] * neg_smp_strategy,
                num_nodes=(sub_hg["admission"].node_id.shape[0], sub_hg["labitem"].node_id.shape[0])
            ).to(device)
        elif neg_smp_strategy == -1:  # use the full labitem set
            sub_hg.labels4item_neg_index = negative_sampling(
                sub_hg.labels4item_pos_index,
                num_neg_samples=sub_hg["admission"].node_id.shape[0] * sub_hg["labitem"].node_id.shape[0],
                num_nodes=(sub_hg["admission"].node_id.shape[0], sub_hg["labitem"].node_id.shape[0])
            ).to(device)
        else:
            raise ValueError
        sub_hg.lables4item_index = torch.cat((sub_hg.labels4item_pos_index, sub_hg.labels4item_neg_index), dim=1)
        sub_hg.lables4item = torch.cat((torch.ones(sub_hg.labels4item_pos_index.shape[1]),
                                        torch.zeros(sub_hg.labels4item_neg_index.shape[1])), dim=0).to(device)
        index4item_shuffle = torch.randperm(sub_hg.lables4item_index.shape[1]).to(device)
        sub_hg.lables4item_index = sub_hg.lables4item_index[:, index4item_shuffle]
        sub_hg.lables4item = sub_hg.lables4item[index4item_shuffle]

        mask_next_t4drug = (hg["admission", "took", "drug"].timestep == (timestep+1)).to(device)
        sub_hg.labels4drug_pos_index = hg["admission", "took", "drug"].edge_index[:, mask_next_t4drug].clone()
        if neg_smp_strategy == 0:
            sub_hg.labels4drug_neg_index = negative_sampling(
                sub_hg.labels4drug_pos_index,
                num_neg_samples=sub_hg.labels4drug_pos_index.shape[1] * 2,
                num_nodes=(sub_hg["admission"].node_id.shape[0], sub_hg["drug"].node_id.shape[0])
            ).to(device)
        elif 1 <= neg_smp_strategy <= 100:
            sub_hg.labels4drug_neg_index = negative_sampling(
                sub_hg.labels4drug_pos_index,
                num_neg_samples=sub_hg["admission"].node_id.shape[0] * neg_smp_strategy,
                num_nodes=(sub_hg["admission"].node_id.shape[0], sub_hg["drug"].node_id.shape[0])
            ).to(device)
        elif neg_smp_strategy == -1:  # use the full labitem set
            sub_hg.labels4drug_neg_index = negative_sampling(
                sub_hg.labels4drug_pos_index,
                num_neg_samples=sub_hg["admission"].node_id.shape[0] * sub_hg["drug"].node_id.shape[0],
                num_nodes=(sub_hg["admission"].node_id.shape[0], sub_hg["drug"].node_id.shape[0])
            ).to(device)
        else:
            raise ValueError
        sub_hg.labels4drug_index = torch.cat((sub_hg.labels4drug_pos_index, sub_hg.labels4drug_neg_index), dim=1)
        sub_hg.labels4drug = torch.cat((torch.ones(sub_hg.labels4drug_pos_index.shape[1]),
                                        torch.zeros(sub_hg.labels4drug_neg_index.shape[1])), dim=0).to(device)
        index4drug_shuffle = torch.randperm(sub_hg.labels4drug_index.shape[1]).to(device)
        sub_hg.labels4drug_index = sub_hg.labels4drug_index[:, index4drug_shuffle]
        sub_hg.labels4drug = sub_hg.labels4drug[index4drug_shuffle]

        # We also need to make sure to add the reverse edges from labitems to admission
        # in order to let a GNN be able to pass messages in both directions.
        # We can leverage the `T.ToUndirected()` transform for this from PyG:
        sub_hg = T.ToUndirected()(sub_hg)

        return sub_hg

    def forward(self, hg: HeteroData):
        hgs = [self.get_subgraph_by_timestep(hg, timestep=t, neg_smp_strategy=self.neg_smp_strategy) for t in range(self.max_timestep)]

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

        # go through gnns
        list_dict_node_feats = self.gnns(list_dict_node_feats, list_edge_index_dict, list_dict_edge_attrs)

        # decode
        dict_node_feat = {
            node_type: torch.stack([dict_node_feats[node_type] for dict_node_feats in list_dict_node_feats]) \
            for node_type in self.node_types
        }
        for node_type, node_feat in dict_node_feat.items():
            node_feat = self.position_encoding(node_feat)  # Add position encoding
            tgt_mask = memory_mask = nn.Transformer.generate_square_subsequent_mask(self.max_timestep).to(node_feat.device)
            node_feat = self.module_dict_decoder[node_type](tgt=node_feat, memory=node_feat, tgt_mask=tgt_mask, memory_mask=memory_mask)
            dict_node_feat[node_type] = node_feat  # update

        # Link predicting:
        dict_every_day_pred = {}
        for node_type in self.node_types:
            if node_type != "admission":  # admission is not the goal of prediction
                if node_type == "drug":
                    dict_every_day_pred[node_type] = {
                        "scores": [self.module_dict_links_predictor[node_type](dict_node_feat["admission"][curr_timestep],
                                                                               dict_node_feat["drug"][curr_timestep],
                                                                               hg.labels4drug_index) \
                                   for curr_timestep, hg in enumerate(hgs)],
                        "labels":  [hg.labels4drug       for hg in hgs],
                        "indices": [hg.labels4drug_index for hg in hgs]
                    }
                elif node_type == "labitem":
                    dict_every_day_pred[node_type] = {
                        "scores": [self.module_dict_links_predictor[node_type](dict_node_feat["admission"][curr_timestep],
                                                                               dict_node_feat["labitem"][curr_timestep],
                                                                               hg.lables4item_index) \
                                   for curr_timestep, hg in enumerate(hgs)],
                        "labels":  [hg.lables4item       for hg in hgs],
                        "indices": [hg.lables4item_index for hg in hgs]
                    }
                else:
                    raise f"check the node_types config! curr: {self.node_types}"

        return dict_every_day_pred


if __name__ == "__main__":
    node_types, edge_types = HeteroGraphConfig.use_all_edge_type()
    model = LERS(max_timestep=20,
                 gnn_type="GENConv",
                 node_types=node_types,
                 edge_types=edge_types,
                 num_decoder_layers=6,
                 hidden_dim=128,
                 neg_smp_strategy=0)
    print(model)
