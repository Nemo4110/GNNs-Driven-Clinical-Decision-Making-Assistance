import torch
import torch.nn as nn
import numpy as np

from dataset.unified import SingleItemType
from model.layers import FMEmbedding
from utils.misc import set_color
from utils.enum_type import FeatureSource, FeatureType


class AbstractRecommender(nn.Module):
    def __init__(self):
        super(AbstractRecommender, self).__init__()

    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data."""
        raise NotImplementedError

    def predict(self, interaction):
        r"""Predict the scores between users and items."""
        raise NotImplementedError

    def __str__(self):
        """Model prints with number of trainable parameters"""
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return (
            super().__str__()
            + set_color("\nTrainable parameters", "blue")
            + f": {params}"
        )


class GeneralRecommender(AbstractRecommender):
    def __init__(self, config, dataset):
        super(GeneralRecommender, self).__init__()

        # 获取数据集信息
        self.USER_ID = config["USER_ID_FIELD"]  # 记录用户id的列名
        self.ITEM_ID = config["ITEM_ID_FIELD"]  # 记录物品id的列名
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items

        self.device = config["device"]


class SequentialRecommender(AbstractRecommender):
    def __init__(self, config, dataset):
        super(SequentialRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]

        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]
        self.n_items = dataset.num(self.ITEM_ID)

        # load parameters info
        self.device = config["device"]


class ContextRecommender(AbstractRecommender):
    def __init__(self, config, dataset: SingleItemType):
        super(ContextRecommender, self).__init__()
        self.device = config["device"]
        self.embedding_size = config["embedding_size"]

        self.double_tower = config["double_tower"]
        if self.double_tower is None:
            self.double_tower = False

        self.LABEL = config["LABEL_FIELD"]
        self.field_names = dataset.fields(source=[
                FeatureSource.INTERACTION,
                FeatureSource.USER,
                FeatureSource.USER_ID,
                FeatureSource.ITEM,
                FeatureSource.ITEM_ID,
            ])
        self.numerical_features = config["numerical_features"]

        if self.double_tower:  # 也就是说分物品塔和用户塔
            # user_id不应该作为用户塔的特征，
            # 否则按目前以adm划分训练、测试、验证的做法，在后面两个集合的user_id对应的embbeding向量不会被优化到
            self.user_field_names = dataset.fields(source=[FeatureSource.USER,])
            self.item_field_names = dataset.fields(source=[FeatureSource.ITEM, FeatureSource.ITEM_ID])
            self.field_names = self.user_field_names + self.item_field_names  # 先用户，后物品

            # 统计各类型的特征列数量
            # 用户塔
            self.user_token_field_num = 0
            self.user_float_field_num = 0
            for field_name in self.user_field_names:
                if dataset.source_dfs.field2type[field_name] == FeatureType.TOKEN:
                    self.user_token_field_num += 1
                elif dataset.source_dfs.field2type[field_name] == FeatureType.FLOAT:
                    self.user_float_field_num += 1
                else:
                    raise NotImplementedError
            # 物品塔
            self.item_token_field_num = 0
            self.item_float_field_num = 0
            for field_name in self.item_field_names:
                if dataset.source_dfs.field2type[field_name] == FeatureType.TOKEN:
                    self.item_token_field_num += 1
                elif dataset.source_dfs.field2type[field_name] == FeatureType.FLOAT:
                    self.item_float_field_num += 1
                else:
                    raise NotImplementedError

        # 统计各类型特征列名 + token类型特征列有多少种不同值的数量
        self.token_field_names = []
        self.token_field_dims = []
        self.float_field_names = []
        self.num_feature_field = 0
        for field_name in self.field_names:
            if field_name == self.LABEL:
                continue

            if dataset.source_dfs.field2type[field_name] == FeatureType.TOKEN:
                self.token_field_names.append(field_name)
                self.token_field_dims.append(dataset.num(field_name))
            elif (dataset.source_dfs.field2type[field_name] == FeatureType.FLOAT and
                  field_name in self.numerical_features):
                self.float_field_names.append(field_name)
            else:
                continue

            self.num_feature_field += 1

        # 给token fields 加偏移，这样便于放到一个统一的embedding_table中
        if len(self.token_field_dims) > 0:
            self.token_field_offsets = np.array(
                (0, *np.cumsum(self.token_field_dims)[:-1]), dtype=np.long)
            self.token_embedding_table = FMEmbedding(
                self.token_field_dims, self.token_field_offsets, self.embedding_size)

        # float fields 过一层fc
        if len(self.float_field_names) > 0:
            self.float_embedding_table = nn.Linear(len(self.float_field_names), self.embedding_size)

        # FM类模型用的first_order_linear
        self.first_order_linear = nn.Linear(self.embedding_size, 1, bias=True)

    def double_tower_embed_input_fields(self, interaction):
        """Embed the whole feature columns in a double tower way.

        Returns:
            torch.FloatTensor: The embedding tensor of token sequence columns in the first part.
            torch.FloatTensor: The embedding tensor of float sequence columns in the first part.
            torch.FloatTensor: The embedding tensor of token sequence columns in the second part.
            torch.FloatTensor: The embedding tensor of float sequence columns in the second part.
        """
        assert self.double_tower

        sparse_embedding, dense_embedding = self.embed_input_fields(interaction)

        if dense_embedding is not None:
            first_dense_embedding, second_dense_embedding = torch.split(dense_embedding,
                [self.user_float_field_num, self.item_float_field_num], dim=1)
        else:
            first_dense_embedding, second_dense_embedding = None, None

        if sparse_embedding is not None:
            sizes = [self.user_token_field_num, self.item_token_field_num]  # 先用户，后物品，OK
            first_sparse_embedding, second_sparse_embedding = torch.split(sparse_embedding, sizes, dim=1)
        else:
            first_sparse_embedding, second_sparse_embedding = None, None

        return (
            first_sparse_embedding,
            first_dense_embedding,
            second_sparse_embedding,
            second_dense_embedding,
        )

    def embed_input_fields(self, interaction):
        """Embed the whole feature columns.

        Returns:
            torch.FloatTensor: The embedding tensor of token sequence columns.
            torch.FloatTensor: The embedding tensor of float sequence columns.
        """

        # float fields
        float_fields = []
        for field_name in self.float_field_names:  # 先用户，后物品
            cur_float_field_values = interaction[field_name].values
            cur_float_field_tensor = torch.from_numpy(cur_float_field_values).unsqueeze(1)
            float_fields.append(cur_float_field_tensor)
        if len(float_fields) > 0:
            float_fields = torch.cat(float_fields, dim=1)  # [batch_size, num_float_field]
        else:
            float_fields = None
        # float fields 过一层全连接层转换到self.embedding_size
        dense_embedding = self.embed_float_fields(float_fields)  # [batch_size, embed_dim] or None
        dense_embedding = dense_embedding.unsqueeze(1) if dense_embedding is not None else dense_embedding

        # token fields
        token_fields = []
        for field_name in self.token_field_names:  # 先用户，后物品
            cur_token_field_values = interaction[field_name].values
            cur_token_field_tensor = torch.from_numpy(cur_token_field_values).unsqueeze(1)
            token_fields.append(cur_token_field_tensor)
        if len(token_fields) > 0:
            token_fields = torch.cat(token_fields, dim=1)  # [batch_size, num_token_field]
        else:
            token_fields = None
        sparse_embedding = self.embed_token_fields(token_fields)  # [batch_size, num_token_field, embed_dim] or None

        # sparse_embedding shape: [batch_size, num_token_field, embed_dim] or None
        # dense_embedding shape:  [batch_size, 1,               embed_dim] or None
        return sparse_embedding, dense_embedding

    def embed_float_fields(self, float_fields: torch.FloatTensor):
        if float_fields is None:
            return None
        float_embedding = self.float_embedding_table(float_fields)

        return float_embedding

    def embed_token_fields(self, token_fields: torch.LongTensor):
        if token_fields is None:
            return None
        token_embedding = self.token_embedding_table(token_fields)

        return token_embedding

    def concat_embed_input_fields(self, interaction):
        sparse_embedding, dense_embedding = self.embed_input_fields(interaction)
        all_embeddings = []
        if sparse_embedding is not None:
            all_embeddings.append(sparse_embedding)
        if dense_embedding is not None and len(dense_embedding.shape) == 3:
            all_embeddings.append(dense_embedding)
        return torch.cat(all_embeddings, dim=1)  # [batch_size, num_field, embed_dim]
