import torch
import torch.nn as nn

from model.abstract_recommender import GeneralRecommender
from model.init import xavier_normal_initialization
from dataset.unified import get_pos_or_neg_shard


class BPR(GeneralRecommender):
    def __init__(self, config, dataset):
        super(BPR, self).__init__(config, dataset)
        self.loss = BPRLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self, interaction):
        user_e, item_e = self.embedding_layer(interaction)
        user_e = self.uf_aligner(user_e.flatten(start_dim=1))
        item_e = self.if_aligner(item_e.flatten(start_dim=1))
        return user_e, item_e

    def calculate_loss(self, interaction):
        pos_shard = get_pos_or_neg_shard(interaction, is_pos=True)
        neg_shard = get_pos_or_neg_shard(interaction, is_pos=False)

        pos_user_e, pos_item_e = self.forward(pos_shard)
        neg_user_e, neg_item_e = self.forward(neg_shard)

        less_size = min(pos_item_e.size(0) * 2, neg_item_e.size(0))
        pos_item_score = torch.mul(pos_user_e, pos_item_e).sum(dim=1).repeat(2)[:less_size]  # 2：1负采样策略
        neg_item_score = torch.mul(neg_user_e, neg_item_e).sum(dim=1)[:less_size]

        # 注意BPRLoss需要两个输入形状相同
        loss = self.loss(pos_item_score, neg_item_score)
        return loss

    def predict(self, interaction):
        user_e, item_e = self.forward(interaction)
        return torch.mul(user_e, item_e).sum(dim=1)


class BPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss
