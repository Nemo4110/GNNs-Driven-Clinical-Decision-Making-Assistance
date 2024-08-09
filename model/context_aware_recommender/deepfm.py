import torch.nn as nn
import torch

from torch.nn.init import xavier_normal_, constant_

from model.abstract_recommender import ContextRecommender
from model.layers import BaseFactorizationMachine, MLPLayers


class DeepFM(ContextRecommender):
    def __init__(self, config, dataset):
        super(DeepFM, self).__init__(config, dataset)

        # load parameters info
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.dropout_prob = config["dropout_prob"]

        # define layers and loss
        self.fm = BaseFactorizationMachine(reduce_sum=True)
        size_list = [self.embedding_size * self.embedding_layer.num_feature_field] + self.mlp_hidden_size
        self.mlp_layers = MLPLayers(size_list, self.dropout_prob)
        self.deep_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)  # Linear product to the final score
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

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
        deepfm_all_embeddings = self.embedding_layer.concat_embed_input_fields(interaction)  # [batch_size, num_field, embed_dim]

        batch_size = deepfm_all_embeddings.shape[0]

        fo_linear = self.embedding_layer.first_order_linear(deepfm_all_embeddings)  # [batch_size, num_field, 1]
        fo_linear = torch.sum(fo_linear, dim=1)

        y_fm = fo_linear + self.fm(deepfm_all_embeddings)

        y_deep = self.deep_predict_layer(
            self.mlp_layers(deepfm_all_embeddings.view(batch_size, -1))
        )
        y = y_fm + y_deep
        return y.squeeze(-1)

    def calculate_loss(self, interaction):
        label = torch.from_numpy(interaction[self.LABEL].values)\
            .float().to(self.device)
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction))

