import torch
import torch.nn as nn
import math

from torch_geometric.nn.conv import GINEConv, GENConv, GATConv


class PositionalEncoding(nn.Module):
    r"""
        Add position encoding information for each timestep.
    Refs:
        - <https://jalammar.github.io/illustrated-transformer/>
        - <https://zhuanlan.zhihu.com/p/338592312>
        - <https://zhuanlan.zhihu.com/p/454482273>
    """
    def __init__(self, hidden_dim, max_timestep):
        super().__init__()

        pe = torch.zeros(max_timestep, hidden_dim)

        position = torch.arange(0, max_timestep, dtype=torch.float).unsqueeze(1)  # [max_timestep, 1]

        # *** the temperature modefied due to the experiment result of DAB-DETR ***
        # div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(20.0) / hidden_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)  # [len, batch, hidden_dim]
        pe.requires_grad = False

        self.register_buffer('pe', pe)
        # print(self.pe.shape)  # torch.Size([20, 1, 64])

    def forward(self, x):
        if self.pe.device != x.device:
            self.pe.to(x.device)
        # return x + self.pe[:x.size(0), :]
        return x + self.pe  # using broadcast mechanism


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


def get_decoder_by_choice(choice: str, hidden_dim: int, num_layers: int = 1):
    if choice == "TransformerDecoder":
        chosen_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=512),
            num_layers=num_layers
        )
    elif choice == "RNN":
        chosen_decoder = nn.RNN(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1)
    elif choice == "GRU":
        chosen_decoder = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1)
    elif choice == "LSTM":
        chosen_decoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1)
    else:
        raise NotImplementedError

    return chosen_decoder


def decode(decoder: nn.Module, input_seq: torch.tensor):
    """
    Args:
        decoder:
        input_seq:
            has shape (seq_len, n_samples, n_features)
    """
    if isinstance(decoder, nn.TransformerDecoder):
        tgt_mask = memory_mask = nn.Transformer\
            .generate_square_subsequent_mask(input_seq.shape[0])\
            .to(input_seq.device)
        output_seq = decoder(tgt=input_seq, memory=input_seq, tgt_mask=tgt_mask, memory_mask=memory_mask)
    elif isinstance(decoder, nn.RNN):
        # TypeError: RNN.forward() got an unexpected keyword argument 'h_0'
        output_seq, h_n = decoder(input=input_seq, hx=input_seq[0].unsqueeze(0))
    elif isinstance(decoder, nn.GRU):
        output_seq, h_n = decoder(input=input_seq, hx=input_seq[0].unsqueeze(0))
    elif isinstance(decoder, nn.LSTM):
        h_0, c_0 = input_seq[0].unsqueeze(0), input_seq[0].unsqueeze(0)
        output_seq, h_n = decoder(input=input_seq, hx=(h_0, c_0))
    else:
        raise NotImplementedError

    return output_seq


if __name__ == "__main__":
    pe = PositionalEncoding(64, 20)
    node_feats = torch.rand(20, 753, 64)
    node_feats_pe = pe(node_feats)
