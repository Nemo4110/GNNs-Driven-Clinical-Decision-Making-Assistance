import torch
import torch.nn as nn

from model.abstract_recommender import SequentialRecommender
from model.layers import MLPLayers, SequentialEmbeddingLayer


class SASRec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(SASRec, self).__init__(config, dataset)

        # load parameters
        self.LABEL_FIELD = config.get("LABEL_FIELD", "label")
        self.n_layers = config.get("n_layers", 2)
        self.n_heads = config.get("n_heads", 2)
        self.hidden_size = self.embedding_size = config["embedding_size"]
        self.hidden_dropout_prob = config.get("hidden_dropout_prob", 0.5)
        self.attn_dropout_prob = config.get("attn_dropout_prob", 0.5)
        self.layer_norm_eps = config.get("layer_norm_eps", 1e-5)

        # define layers and loss
        self.embedding_layer = SequentialEmbeddingLayer(config, dataset)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.mask_mat = torch.arange(self.max_seq_length).to(self.device).view(1, -1)
        self.trm_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=self.n_heads,
                batch_first=True,
                layer_norm_eps=self.layer_norm_eps,
                dropout=self.attn_dropout_prob
            ),
            num_layers=self.n_layers
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.dnn_predict_layers = nn.Linear(3 * self.hidden_size, 1)

        # 将user & item embedding的最后一维映射回self.hidden_size
        if len(self.embedding_layer.user_float_field_names) > 0:
            num_user_feature = len(self.embedding_layer.user_token_field_names) + 1
        else:
            num_user_feature = len(self.embedding_layer.user_token_field_names)
        self.ue_fc = nn.Linear(num_user_feature * self.embedding_size, self.hidden_size)

        if len(self.embedding_layer.item_float_field_names) > 0:
            num_item_feature = len(self.embedding_layer.item_token_field_names) + 1 + 1  # for conceited id emb
        else:
            num_item_feature = len(self.embedding_layer.item_token_field_names) + 1

        # RuntimeError: Trying to backward through the graph a second time
        # https://discuss.pytorch.org/t/pytorch-runtimeerror-trying-to-backward-through-the-graph-a-second-time/191396/2
        self.hst_ie_fc = nn.Linear(num_item_feature * self.embedding_size, self.hidden_size)
        self.tgt_ie_fc = nn.Linear(num_item_feature * self.embedding_size, self.hidden_size)

        # 与原论文保持一致，使用二分类CE
        self.loss_fct = nn.BCEWithLogitsLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, interaction):
        B = len(interaction)

        user_embedding, item_seqs_embedding = self.embedding_layer(interaction)
        target_item_feat_emb, history_item_feat_emd = torch.split(
            item_seqs_embedding, [1, self.max_seq_length], dim=1)
        target_item_feat_emb = target_item_feat_emb.squeeze(1)  # [B, ?]

        # 注意上面的这些_embedding的最后一维并不是self.hidden_size
        user_embedding = self.ue_fc(user_embedding)
        target_item_feat_emb = self.tgt_ie_fc(target_item_feat_emb)
        history_item_feat_emd = self.hst_ie_fc(history_item_feat_emd)

        # position information
        position_ids = torch.arange(history_item_feat_emd.size(1), dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(B, -1)
        position_embedding = self.position_embedding(position_ids)

        input_emb = history_item_feat_emd + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        item_seq_len = torch.from_numpy(interaction[self.ITEM_SEQ_LEN].values)
        item_seq_len = torch.clamp(item_seq_len, max=self.max_seq_length).to(self.device)

        padding_mask = self.mask_mat.repeat(B, 1)
        padding_mask = padding_mask >= item_seq_len.unsqueeze(1)  # padding mask
        subsequent_mask = nn.Transformer.generate_square_subsequent_mask(self.max_seq_length).to(self.device)

        # [B, max_seq_length, h]
        trm_output = self.trm_encoder(input_emb, mask=subsequent_mask, src_key_padding_mask=padding_mask)

        # 从自注意完的历史item列表emb收集最后一个（有效序列长度-1）
        trm_output = self.gather_indexes(trm_output, item_seq_len - 1)  # [B, H]

        scores = self.dnn_predict_layers(
            torch.cat([trm_output, target_item_feat_emb, user_embedding], dim=-1))

        return scores.squeeze(1)

    def calculate_loss(self, interaction):
        label = torch.from_numpy(interaction[self.LABEL_FIELD].values)\
            .float().to(self.device)
        output = self.forward(interaction)
        loss = self.loss_fct(output, label)
        return loss

    def predict(self, interaction):
        scores = self.sigmoid(self.forward(interaction))
        return scores
