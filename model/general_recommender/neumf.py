import torch
import torch.nn as nn

from torch.nn.init import normal_
from model.abstract_recommender import GeneralRecommender
from model.layers import MLPLayers


class NeuMF(GeneralRecommender):
    def __init__(self, config, dataset):
        super(NeuMF, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config["LABEL_FIELD"]

        # load parameters info
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.dropout_prob = config["dropout_prob"]

        # define layers and loss
        self.mlp_layers = MLPLayers([2 * self.embedding_size] + self.mlp_hidden_size, self.dropout_prob)
        self.predict_layer = nn.Linear(self.embedding_size + self.mlp_hidden_size[-1], 1)

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    def forward(self, interaction):
        user_e, item_e = self.embedding_layer(interaction)

        user_mf_e = self.uf_aligner(user_e.flatten(start_dim=1))
        item_mf_e = self.if_aligner(item_e.flatten(start_dim=1))

        user_mlp_e = self.uf_aligner(user_e.flatten(start_dim=1))
        item_mlp_e = self.if_aligner(item_e.flatten(start_dim=1))

        mf_output = torch.mul(user_mf_e, item_mf_e)  # [batch_size, embedding_size]
        mlp_output = self.mlp_layers(torch.cat((user_mlp_e, item_mlp_e), -1))  # [batch_size, layers[-1]]

        output = self.predict_layer(torch.cat((mf_output, mlp_output), -1))
        return output.squeeze(-1)

    def calculate_loss(self, interaction):
        label = torch.from_numpy(interaction[self.LABEL].values).float().to(self.device)
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        predict = self.sigmoid(self.forward(interaction))
        return predict
