import torch
import torch.nn as nn

from torch.nn.init import xavier_normal_, constant_

from model.abstract_recommender import ContextRecommender
from model.layers import MLPLayers
from dataset.unified import SourceDataFrames, SingleItemType
from utils.enum_type import FeatureType


class DSSM(ContextRecommender):
    def __init__(self, config, dataset):
        super(DSSM, self).__init__(config, dataset)

        # load parameters info
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.dropout_prob = config["dropout_prob"]

        self.user_feature_num = (self.embedding_layer.user_token_field_num + (self.embedding_layer.user_float_field_num > 0))
        self.item_feature_num = (self.embedding_layer.item_token_field_num + (self.embedding_layer.item_float_field_num > 0))

        # define layers and loss
        user_size_list = [self.embedding_size * self.user_feature_num] + self.mlp_hidden_size
        item_size_list = [self.embedding_size * self.item_feature_num] + self.mlp_hidden_size
        self.user_mlp_layers = MLPLayers(user_size_list, self.dropout_prob, activation="tanh", bn=True)
        self.item_mlp_layers = MLPLayers(item_size_list, self.dropout_prob, activation="tanh", bn=True)
        self.loss = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, interaction):
        embed_result = self.embedding_layer.double_tower_embed_input_fields(interaction)
        user_sparse_embedding, user_dense_embedding = embed_result[:2]
        item_sparse_embedding, item_dense_embedding = embed_result[2:]

        user = []
        if user_sparse_embedding is not None:
            user.append(user_sparse_embedding)
        if user_dense_embedding is not None and len(user_dense_embedding.shape) == 3:
            user.append(user_dense_embedding)

        item = []
        if item_sparse_embedding is not None:
            item.append(item_sparse_embedding)
        if item_dense_embedding is not None and len(item_dense_embedding.shape) == 3:
            item.append(item_dense_embedding)

        embed_user = torch.cat(user, dim=1)
        embed_item = torch.cat(item, dim=1)

        batch_size = embed_item.shape[0]

        user_dnn_out = self.user_mlp_layers(embed_user.view(batch_size, -1))
        item_dnn_out = self.item_mlp_layers(embed_item.view(batch_size, -1))
        score = torch.cosine_similarity(user_dnn_out, item_dnn_out, dim=1)
        return score.squeeze(-1)

    def calculate_loss(self, interaction):
        label = torch.from_numpy(interaction[self.LABEL].values)\
            .float().to(self.device)
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction))
