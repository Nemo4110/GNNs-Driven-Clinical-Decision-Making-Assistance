import sys; sys.path.append('..')

import torch
import torch.nn as nn
import torch_geometric.transforms as T
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn.conv import GINEConv
from torch_geometric.nn import to_hetero
from dataset.hgs import MyOwnDataset


class SingelGnn(nn.Module):
    def __init__(self, hidden_dim, dropout_rate, nn_type="fc"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.drop = nn.Dropout(p=self.dropout_rate)

        if nn_type == "fc":
            self.nn1 = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
            self.nn2 = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        else:
            raise f"Do not support {nn_type} now!"

        self.conv1 = GINEConv(nn=self.nn1, train_eps=True, edge_dim=2)
        self.conv2 = GINEConv(nn=self.nn2, train_eps=True, edge_dim=2)

    def forward(self, node_feats, edge_index, edge_attrs):
        node_feats = self.conv1(x=node_feats, edge_index=edge_index, edge_attr=edge_attrs).relu()
        node_feats = self.drop(node_feats)
        node_feats = self.conv2(x=node_feats, edge_index=edge_index, edge_attr=edge_attrs).relu()

        return node_feats


class MultiGnns(nn.Module):
    r"""

    Args:

    """
    def __init__(self,
                 hidden_dim: int,
                 max_timestep: int,
                 dropout_rate: float):
        super().__init__()

        # as many fc as max_timestep
        self.gnns = [SingelGnn(hidden_dim=hidden_dim, dropout_rate=dropout_rate) for t in range(max_timestep)]

        node_types = ['admission', 'labitem']
        edge_types = [('admission', 'did', 'labitem'), ('labitem', 'rev_did', 'admission')]

        # !!! Warning: here must use `self.gnns[i]`, if we use `gnn` will cause failure of `to_hetero`,
        #              because the `gnn` are temp parameter!
        for i, gnn in enumerate(self.gnns):
            self.gnns[i] = to_hetero(gnn, metadata=(node_types, edge_types))

    def forward(self, list_dict_node_feats, list_edge_index_dict, list_dict_edge_attrs):
        for i, gnn in enumerate(self.gnns):
            list_dict_node_feats[i] = gnn(list_dict_node_feats[i],
                                          list_edge_index_dict[i],
                                          list_dict_edge_attrs[i])

        return list_dict_node_feats


class LERS(nn.Module):
    r"""

    Args:
    """
    def __init__(self,
                 max_timestep: int,
                 gnn_drop: int = 0.3,
                 # num_admissions: int,
                 num_labitems: int = 753,
                 num_decoder_layers_admission=3,
                 num_decoder_layers_labitem=3,
                 hidden_dim: int = 128):
        super().__init__()

        self.max_timestep = max_timestep
        self.gnn_drop = gnn_drop
        # self.num_admissions = num_admissions
        self.num_labitems = num_labitems
        self.num_decoder_layers_admission = num_decoder_layers_admission
        self.num_decoder_layers_labitem = num_decoder_layers_labitem
        self.hidden_dim = hidden_dim

        # TODO: Think about the first args of `nn.Embedding` here should equal to
        #  - max of total `HADM_ID`?
        #  - current batch_size of `HADM_ID`?
        #  - Or just deprecate the `nn.Embedding` as we already have the node_features? √√√
        # self.admission_embedding = nn.Embedding(self.num_admissions, self.hidden_dim)

        self.labitem_embedding = nn.Embedding(self.num_labitems, self.hidden_dim)

        self.admission_proj = nn.Linear(in_features=8, out_features=self.hidden_dim)
        self.labitem_proj = nn.Linear(in_features=2, out_features=self.hidden_dim)

        self.gnns = MultiGnns(hidden_dim=self.hidden_dim, max_timestep=self.max_timestep, dropout_rate=self.gnn_drop)

        self.decoder_layers_admission = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=8)
        self.decoder_layers_labitem = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=8)
        self.decoder_admission = nn.TransformerDecoder(self.decoder_layers_admission, num_layers=self.num_decoder_layers_admission)
        self.decoder_labitem = nn.TransformerDecoder(self.decoder_layers_labitem, num_layers=self.num_decoder_layers_labitem)

    def forward(self, hg: HeteroData):
        hgs = [self.get_subgraph_by_timestep(hg, timestep=t) for t in range(self.max_timestep)]

        list_dict_node_feats = [{
            'admission': self.admission_proj(hg['admission'].x),
            'labitem': self.labitem_proj(hg['labitem'].x) + self.labitem_embedding(hg['labitem'].node_id)
        } for hg in hgs]
        list_edge_index_dict = [hg.edge_index_dict for hg in hgs]
        list_dict_edge_attrs = [{
            ('admission', 'did', 'labitem'): hg["admission", "did", "labitem"].x.float(),
            ('labitem', 'rev_did', 'admission'): hg['labitem', 'rev_did', 'admission'].x.float()
        } for hg in hgs]

        list_dict_node_feats = self.gnns(list_dict_node_feats, list_edge_index_dict, list_dict_edge_attrs)

        admission_node_feats = torch.stack([dict_node_feats['admission'] for dict_node_feats in list_dict_node_feats])
        labitem_node_feats = torch.stack([dict_node_feats['labitem'] for dict_node_feats in list_dict_node_feats])

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.max_timestep)
        memory_mask = nn.Transformer.generate_square_subsequent_mask(self.max_timestep)
        admission_node_feats = self.decoder_admission(tgt=admission_node_feats, memory=admission_node_feats,
                                                      tgt_mask=tgt_mask, memory_mask=memory_mask)
        labitem_node_feats = self.decoder_labitem(tgt=labitem_node_feats, memory=labitem_node_feats,
                                                  tgt_mask=tgt_mask, memory_mask=memory_mask)

        # Link predictions
        scores = torch.matmul(admission_node_feats, labitem_node_feats.permute(0, 2, 1))#.sigmoid()
        labels = torch.stack([self.to_dense_adj(hg.labels, shape=(scores.shape[-2], scores.shape[-1])) for hg in hgs])

        return scores, labels

    @staticmethod
    def get_subgraph_by_timestep(hg, timestep):
        mask = torch.BoolTensor(hg["admission", "did", "labitem"].timestep == timestep)
        eidx = hg["admission", "did", "labitem"].edge_index[:, mask]
        ex = hg["admission", "did", "labitem"].x[mask, :]

        sub_hg = HeteroData()

        # Nodes
        sub_hg["admission"].node_id = hg["admission"].node_id.clone()
        sub_hg["admission"].x = hg["admission"].x.clone().float()

        sub_hg["labitem"].node_id = hg["labitem"].node_id.clone()
        sub_hg["labitem"].x = hg["labitem"].x.clone().float()

        # Edges
        sub_hg["admission", "did", "labitem"].edge_index = eidx.clone()
        sub_hg["admission", "did", "labitem"].x = ex.clone().float()

        assert timestep < torch.max(hg["admission", "did", "labitem"].timestep), "last timestep has not labels!"
        mask_next_t = torch.BoolTensor(hg["admission", "did", "labitem"].timestep == (timestep+1))
        sub_hg.labels = hg["admission", "did", "labitem"].edge_index[:, mask_next_t].clone()

        sub_hg = T.ToUndirected()(sub_hg)
        return sub_hg

    @staticmethod
    def to_dense_adj(edge_index: torch.tensor, shape: (int, int)):
        dense_adj = torch.zeros(shape[0], shape[1], dtype=torch.float)
        for src, tgt in zip(edge_index[0], edge_index[1]):
            dense_adj[src][tgt] = 1

        return dense_adj

if __name__ == "__main__":
    train_set = MyOwnDataset(root_path=r"D:\Datasets\mimic\mimic-iii-hgs\batch_size_128", usage="train")
    model = LERS(max_timestep=20)
    scores, labels = model(train_set[0])

    # print(scores)  # torch.Size([20, 128, 753])
    # print(scores[0])

    # print(labels.shape)
    print(F.binary_cross_entropy_with_logits(scores, labels))
