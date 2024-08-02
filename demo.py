"""用于跑各个模型的简单测试"""
import torch

from model import context_aware_recommender, general_recommender, sequential_recommender
from dataset.unified import SourceDataFrames, SingleItemType
from utils.enum_type import FeatureType, FeatureSource

if __name__ == '__main__':
    sources_dfs = SourceDataFrames(r"data\mimic-iii-clinical-database-1.4")
    dataset = SingleItemType(sources_dfs, "val", "labitem")

    """DSSM"""
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

    """DeepFM"""
    config = {
        "mlp_hidden_size": [128, 64, 32],
        "dropout_prob": 0.1,
        "embedding_size": 32,
        "device": torch.device('cpu'),
        "LABEL_FIELD": "label",
        "numerical_features": dataset.fields(ftype=[FeatureType.FLOAT])
    }
    net = context_aware_recommender.DeepFM(config, dataset)

    loss = net.calculate_loss(dataset[2])
    print(loss)
