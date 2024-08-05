"""用于跑各个模型的简单测试"""
import torch

from model import context_aware_recommender, general_recommender, sequential_recommender
from model.layers import SequentialEmbeddingLayer
from dataset.unified import SourceDataFrames, SingleItemType, SingleItemTypeForContextAwareRec, SingleItemTypeForSequentialRec
from utils.enum_type import FeatureType, FeatureSource

if __name__ == '__main__':
    sources_dfs = SourceDataFrames(r"data\mimic-iii-clinical-database-1.4")

    """↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ context_aware_recommender ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓"""
    # dataset = SingleItemTypeForContextAwareRec(sources_dfs, "val", "labitem")

    # DSSM
    # config = {
    #     "mlp_hidden_size": [128, 64, 32],
    #     "dropout_prob": 0.1,
    #     "embedding_size": 32,
    #     "double_tower": True,
    #     "device": torch.device('cpu'),
    #     "LABEL_FIELD": "label",
    #     "numerical_features": dataset.fields(ftype=[FeatureType.FLOAT])
    # }
    # net = context_aware_recommender.DSSM(config, dataset)

    # DeepFM
    # config = {
    #     "mlp_hidden_size": [128, 64, 32],
    #     "dropout_prob": 0.1,
    #     "embedding_size": 32,
    #     "device": torch.device('cpu'),
    #     "LABEL_FIELD": "label",
    #     "numerical_features": dataset.fields(ftype=[FeatureType.FLOAT])
    # }
    # net = context_aware_recommender.DeepFM(config, dataset)

    """↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ GeneralRecommender ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓"""
    # dataset = SingleItemType(sources_dfs, "val", "drug")
    # NeuMF
    # config = {
    #     "mlp_hidden_size": [128, 64, 32],
    #     "dropout_prob": 0.1,
    #     "embedding_size": 32,
    #     "device": torch.device('cpu'),
    #     "LABEL_FIELD": "label",
    # }
    # net = general_recommender.NeuMF(config, dataset)
    # loss = net.calculate_loss(dataset[2])
    # BPR
    # config = {
    #     "embedding_size": 32,
    #     "device": torch.device('cpu'),
    #     "LABEL_FIELD": "label",
    # }
    # net = general_recommender.BPR(config, dataset)
    # loss = net.calculate_loss(dataset[2])

    """↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ SequentialRecommender ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓"""
    dataset = SingleItemTypeForSequentialRec(sources_dfs, "test", "drug")
    interaction = dataset[2]

    # SequentialEmbeddingLayer
    # config = {
    #     "dropout_prob": 0.1,
    #     "embedding_size": 32,
    #     "device": torch.device('cpu'),
    #     "LABEL_FIELD": "label",
    #     "MAX_HISTORY_ITEM_ID_LIST_LENGTH": 100,
    # }
    # emb_layer = SequentialEmbeddingLayer(config, dataset)
    # interaction = dataset[1]
    # user_embedding, item_seqs_embedding = emb_layer(interaction)

    # DIN
    config = {
        "mlp_hidden_size": [128, 64, 32],
        "dropout_prob": 0.1,
        "embedding_size": 32,
        "device": torch.device('cpu'),
        "LABEL_FIELD": "label",
        "MAX_HISTORY_ITEM_ID_LIST_LENGTH": 100,
    }
    net = sequential_recommender.DIN(config, dataset)
    loss = net.calculate_loss(interaction)
    print(loss)
