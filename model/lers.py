import sys; sys.path.append('..')
import torch
import torch.nn as nn
import torch_geometric.transforms as T
import torch.nn.functional as F
import d2l.torch as d2l
import numpy as np

from torch_geometric.data import HeteroData
from torch_geometric.nn.conv import GINEConv, GENConv, GATConv
from torch_geometric.nn import to_hetero

from dataset.hgs import MyOwnDataset
from model.position_encoding import PositionalEncoding


class SingelGnn(nn.Module):
    def __init__(self, hidden_dim, gnn_type):
        super().__init__()
        self.hidden_dim = hidden_dim

        if gnn_type == "GINEConv":
            self.conv1 = GINEConv(nn=nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim))
            self.conv2 = GINEConv(nn=nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim))
        elif gnn_type == "GENConv":
            self.conv1 = GENConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim, msg_norm=True)
            self.conv2 = GENConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim, msg_norm=True)
        elif gnn_type == "GATConv":
            self.conv1 = GATConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim, add_self_loops=False, edge_dim=self.hidden_dim)
            self.conv2 = GATConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim, add_self_loops=False, edge_dim=self.hidden_dim)
        else:
            raise f"Do not support arg:gnn_type={gnn_type} now!"

    def forward(self, node_feats, edge_index, edge_attrs):
        node_feats = self.conv1(x=node_feats, edge_index=edge_index, edge_attr=edge_attrs).relu()
        node_feats = self.conv2(x=node_feats, edge_index=edge_index, edge_attr=edge_attrs).relu()
        return node_feats


class MultiGnns(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 max_timestep: int,
                 gnn_type: str):
        super().__init__()

        # RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!
        # Solution: use the `nn.ModuleList` instead of list
        self.gnns = nn.ModuleList([SingelGnn(hidden_dim=hidden_dim, gnn_type=gnn_type)
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


class LERS(nn.Module):
    def __init__(self,
                 max_timestep: int,
                 gnn_type: str,
                 # num_admissions: int,
                 num_labitems: int = 753,
                 num_drugs: int = 4294,
                 num_decoder_layers_admission=6,
                 num_decoder_layers_labitem=6,
                 num_decoder_layers_drug=6,
                 hidden_dim: int = 128):
        super().__init__()

        self.max_timestep = max_timestep
        self.gnn_type = gnn_type

        # self.num_admissions = num_admissions
        self.num_labitems = num_labitems
        self.num_drugs    = num_drugs

        self.num_decoder_layers_admission = num_decoder_layers_admission
        self.num_decoder_layers_labitem   = num_decoder_layers_labitem
        self.num_decoder_layers_drug      = num_decoder_layers_drug

        self.hidden_dim = hidden_dim

        # TODO: Think about the first args of `nn.Embedding` here should equal to
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

        self.gnns = MultiGnns(hidden_dim=self.hidden_dim, max_timestep=self.max_timestep, gnn_type=self.gnn_type)

        self.decoder_layers_admission = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=8, dim_feedforward=512)
        self.decoder_layers_labitem   = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=8, dim_feedforward=512)
        self.decoder_layers_drug      = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=8, dim_feedforward=512)

        self.decoder_admission = nn.TransformerDecoder(self.decoder_layers_admission, num_layers=self.num_decoder_layers_admission)
        self.decoder_labitem   = nn.TransformerDecoder(self.decoder_layers_labitem,   num_layers=self.num_decoder_layers_labitem)
        self.decoder_drug      = nn.TransformerDecoder(self.decoder_layers_drug,      num_layers=self.num_decoder_layers_drug)

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.shape) > 1:
                nn.init.xavier_normal_(weight)

    def forward(self, hg: HeteroData):
        hgs = [self.get_subgraph_by_timestep(hg, timestep=t) for t in range(self.max_timestep)]

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
        # tgt_mask = memory_mask = torch.triu(torch.full((self.max_timestep, self.max_timestep), -1e+31, device=device), diagonal=1)

        tgt_admi = mem_admi = admission_node_feats
        tgt_labi = mem_labi = labitem_node_feats
        tgt_drug = mem_drug = drug_node_feats

        admission_node_feats = self.decoder_admission(tgt=tgt_admi, memory=mem_admi, tgt_mask=tgt_mask, memory_mask=memory_mask)
        labitem_node_feats   = self.decoder_labitem(  tgt=tgt_labi, memory=mem_labi, tgt_mask=tgt_mask, memory_mask=memory_mask)
        drug_node_feats      = self.decoder_drug(     tgt=tgt_drug, memory=mem_drug, tgt_mask=tgt_mask, memory_mask=memory_mask)

        # Link predictions
        scores      = torch.matmul(admission_node_feats, labitem_node_feats.transpose(1, 2))
        scores4drug = torch.matmul(admission_node_feats, drug_node_feats.transpose(1, 2))

        labels      = torch.stack([self.to_dense_adj(hg.labels,      shape=(scores.shape[-2], scores.shape[-1]))       for hg in hgs])
        labels4drug = torch.stack([self.to_dense_adj(hg.labels4drug, shape=(scores4drug.shape[-2], scores4drug.shape[-1])) for hg in hgs])

        return scores, labels, scores4drug, labels4drug

    @staticmethod
    def get_subgraph_by_timestep(hg: HeteroData, timestep: int):
        device = hg["admission", "did", "labitem"].timestep.device

        # https://discuss.pytorch.org/t/typeerror-expected-tensoroptions-dtype-float-device-cpu-layout-strided-requires-grad-false-default-pinned-memory-false-default-memory-format-nullopt/159558
        mask = (hg["admission", "did", "labitem"].timestep == timestep).to(device)
        eidx = hg["admission", "did", "labitem"].edge_index[:, mask]
        ex   = hg["admission", "did", "labitem"].x[mask, :]

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
        sub_hg["admission", "did", "labitem"].edge_index = eidx.clone()
        sub_hg["admission", "did", "labitem"].x          = ex.clone().float()

        sub_hg["admission", "took", "drug"].edge_index = eidx4drug.clone()
        sub_hg["admission", "took", "drug"].x          = ex4drug.clone().float()

        # Ensure that the current time step does not exceed the maximum time step recorded
        # for this batch of hospitalized patients.
        assert timestep < torch.max(hg["admission", "did", "labitem"].timestep), "last timestep has not labels!"
        assert timestep < torch.max(hg["admission", "took", "drug"].timestep),   "last timestep has not labels!"

        mask_next_t = (hg["admission", "did", "labitem"].timestep == (timestep+1)).to(device)
        sub_hg.labels = hg["admission", "did", "labitem"].edge_index[:, mask_next_t].clone()

        mask_next_t4drug = (hg["admission", "took", "drug"].timestep == (timestep+1)).to(device)
        sub_hg.labels4drug = hg["admission", "took", "drug"].edge_index[:, mask_next_t4drug].clone()

        sub_hg = T.ToUndirected()(sub_hg)
        return sub_hg

    @staticmethod
    def to_dense_adj(edge_index: torch.tensor, shape):
        # here must ensure all tensor be on same device
        dense_adj = torch.zeros(shape[0], shape[1], dtype=torch.float, device=edge_index.device)

        # Note: Due to the heterogeneity of nodes (there are many different nodes) this adjacency matrix is a rectangle, 
        #       not a square like a homogeneous graph.
        for src, tgt in zip(edge_index[0], edge_index[1]):
            dense_adj[src][tgt] = 1

        return dense_adj


if __name__ == "__main__":
    pass