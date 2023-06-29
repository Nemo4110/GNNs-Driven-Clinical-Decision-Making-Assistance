import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    r"""

    Refs:
        - <https://jalammar.github.io/illustrated-transformer/>
        - <https://zhuanlan.zhihu.com/p/338592312>
        - <https://zhuanlan.zhihu.com/p/454482273>
    """
    def __init__(self, hidden_dim, max_timestep):
        super().__init__()

        pe = torch.zeros(max_timestep, hidden_dim)

        position = torch.arange(0, max_timestep, dtype=torch.float).unsqueeze(1)  # [max_timestep, 1]
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


if __name__ == "__main__":
    pe = PositionalEncoding(64, 20)
    node_feats = torch.rand(20, 753, 64)
    node_feats_pe = pe(node_feats)