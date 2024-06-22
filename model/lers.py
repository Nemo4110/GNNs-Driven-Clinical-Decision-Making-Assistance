import sys; sys.path.append('..')
import torch
import torch.nn as nn
import torch_geometric.transforms as T
import torch.nn.functional as F
import numpy as np

from torch_geometric.data import HeteroData
from torch_geometric.nn.conv import GINEConv, GENConv, GATConv
from torch_geometric.nn import to_hetero
from torch_geometric.utils import negative_sampling

from dataset.hgs import MyOwnDataset
from model.position_encoding import PositionalEncoding


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
            node_feats = conv.forward(x=node_feats, edge_index=edge_index, edge_attr=edge_attrs).relu()
        return node_feats


class MultiGnns(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 max_timestep: int,
                 gnn_type: str,
                 gnn_layer_num: int = 2):
        super().__init__()

        # RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!
        # Solution: use the `nn.ModuleList` instead of list
        self.gnns = nn.ModuleList([SingelGnn(hidden_dim=hidden_dim, gnn_type=gnn_type, gnn_layer_num=gnn_layer_num)
                                   for _ in range(max_timestep)])  # as many fc as max_timestep

        node_types = ['admission', 'labitem', 'drug']
        edge_types = [('admission', 'did', 'labitem'), ('labitem', 'rev_did', 'admission'),
                      ("admission", "took", "drug"), ("drug", "rev_took", "admission")]

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
                 # num_admissions: int,
                 neg_smp_strategy: int,
                 num_labitems: int = 753,
                 num_drugs: int = 4294,
                 num_decoder_layers_admission=6,
                 num_decoder_layers_labitem=6,
                 num_decoder_layers_drug=6,
                 hidden_dim: int = 128,
                 gnn_layer_num: int = 2):
        super().__init__()

        self.max_timestep = max_timestep
        self.gnn_type = gnn_type
        self.gnn_layer_num = gnn_layer_num

        # self.num_admissions = num_admissions
        self.neg_smp_strategy = neg_smp_strategy

        self.num_labitems = num_labitems
        self.num_drugs    = num_drugs

        self.num_decoder_layers_admission = num_decoder_layers_admission
        self.num_decoder_layers_labitem   = num_decoder_layers_labitem
        self.num_decoder_layers_drug      = num_decoder_layers_drug

        self.hidden_dim = hidden_dim

        # DONE: Think about the first args of `nn.Embedding` here should equal to
        #  - max of total `HADM_ID`?
        #  - current batch_size of `HADM_ID`?
        #  √ Or just deprecate the `nn.Embedding` as we already have the node_features
        # self.admission_embedding = nn.Embedding(self.num_admissions, self.hidden_dim)
        self.labitem_embedding = nn.Embedding(self.num_labitems, self.hidden_dim)
        self.drug_embedding    = nn.Embedding(self.num_drugs, self.hidden_dim)

        self.position_encoding = PositionalEncoding(hidden_dim=self.hidden_dim, max_timestep=self.max_timestep)

        # will use MLP as proj better?
        self.proj_admission      = nn.Linear(in_features=8, out_features=self.hidden_dim)
        self.proj_labitem        = nn.Linear(in_features=2, out_features=self.hidden_dim)
        self.proj_drug           = nn.Linear(in_features=8, out_features=self.hidden_dim)

        self.proj_edge_attr      = nn.Linear(in_features=2, out_features=self.hidden_dim)
        self.proj_edge_attr4drug = nn.Linear(in_features=7, out_features=self.hidden_dim)

        self.gnns = MultiGnns(hidden_dim=self.hidden_dim, max_timestep=self.max_timestep, gnn_type=self.gnn_type, gnn_layer_num=self.gnn_layer_num)

        self.decoder_layers_admission = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=8, dim_feedforward=512)
        self.decoder_layers_labitem   = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=8, dim_feedforward=512)
        self.decoder_layers_drug      = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=8, dim_feedforward=512)

        self.decoder_admission = nn.TransformerDecoder(self.decoder_layers_admission, num_layers=self.num_decoder_layers_admission)
        self.decoder_labitem   = nn.TransformerDecoder(self.decoder_layers_labitem,   num_layers=self.num_decoder_layers_labitem)
        self.decoder_drug      = nn.TransformerDecoder(self.decoder_layers_drug,      num_layers=self.num_decoder_layers_drug)

        self.links_predictor4item = LinksPredictor(hidden_dim=self.hidden_dim)
        self.links_predictor4drug = LinksPredictor(hidden_dim=self.hidden_dim)

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
                - 0 for 1: 2 (positive: negative)
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

        list_dict_node_feats = [{'admission': self.proj_admission(hg['admission'].x),
                                 'labitem':   self.proj_labitem(hg['labitem'].x) + self.labitem_embedding(hg['labitem'].node_id),
                                 'drug':      self.proj_drug(hg['drug'].x) +       self.drug_embedding(hg['drug'].node_id)}
                                for hg in hgs]
        list_edge_index_dict = [hg.edge_index_dict for hg in hgs]
        list_dict_edge_attrs = [{('admission', 'did', 'labitem'):     self.proj_edge_attr(hg["admission", "did", "labitem"].x),
                                 ('labitem', 'rev_did', 'admission'): self.proj_edge_attr(hg['labitem', 'rev_did', 'admission'].x),
                                 ("admission", "took", "drug"):       self.proj_edge_attr4drug(hg["admission", "took", "drug"].x),
                                 ("drug", "rev_took", "admission"):   self.proj_edge_attr4drug(hg["drug", "rev_took", "admission"].x)}
                                for hg in hgs]

        list_dict_node_feats = self.gnns(list_dict_node_feats, list_edge_index_dict, list_dict_edge_attrs)

        admission_node_feats = torch.stack([dict_node_feats['admission'] for dict_node_feats in list_dict_node_feats])
        labitem_node_feats   = torch.stack([dict_node_feats['labitem']   for dict_node_feats in list_dict_node_feats])
        drug_node_feats      = torch.stack([dict_node_feats['drug']      for dict_node_feats in list_dict_node_feats])

        # Add position encoding:
        admission_node_feats = self.position_encoding(admission_node_feats)
        labitem_node_feats   = self.position_encoding(labitem_node_feats)
        drug_node_feats      = self.position_encoding(drug_node_feats)

        device = admission_node_feats.device
        tgt_mask = memory_mask = nn.Transformer.generate_square_subsequent_mask(self.max_timestep).to(device)

        tgt_admi = mem_admi = admission_node_feats
        tgt_labi = mem_labi = labitem_node_feats
        tgt_drug = mem_drug = drug_node_feats

        admission_node_feats = self.decoder_admission(tgt=tgt_admi, memory=mem_admi, tgt_mask=tgt_mask, memory_mask=memory_mask)
        labitem_node_feats   = self.decoder_labitem(  tgt=tgt_labi, memory=mem_labi, tgt_mask=tgt_mask, memory_mask=memory_mask)
        drug_node_feats      = self.decoder_drug(     tgt=tgt_drug, memory=mem_drug, tgt_mask=tgt_mask, memory_mask=memory_mask)

        # Link predicting:
        list_labels4item = []; list_scores4item = []
        list_labels4drug = []; list_scores4drug = []; list_edge_indices4drug = []
        for curr_timestep, hg in enumerate(hgs):
            list_labels4item.append(hg.lables4item)
            list_labels4drug.append(hg.labels4drug)
            list_scores4item.append(self.links_predictor4item(admission_node_feats[curr_timestep], labitem_node_feats[curr_timestep], hg.lables4item_index))
            list_scores4drug.append(self.links_predictor4drug(admission_node_feats[curr_timestep], drug_node_feats[curr_timestep],    hg.labels4drug_index))
            list_edge_indices4drug.append(hg.labels4drug_index)

        return list_scores4item, list_labels4item, list_scores4drug, list_labels4drug, list_edge_indices4drug


if __name__ == "__main__":
    pass