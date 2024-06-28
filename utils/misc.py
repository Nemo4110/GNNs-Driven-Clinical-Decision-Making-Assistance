import torch
import torch.nn.functional as F

from typing import List, Dict, Tuple


def calc_loss(dict_every_day_pred, node_types: List[str], device):
    loss = torch.tensor(0.0).to(device)
    for node_type in node_types:
        if node_type == "admission":
            continue
        for scores, labels in zip(dict_every_day_pred[node_type]["scores"],
                                  dict_every_day_pred[node_type]["labels"]):
            loss += F.binary_cross_entropy_with_logits(scores, labels)
    return loss


node_type_to_prefix = {
    'labitem':"ITEMS",
    'drug': "DRUGS"
}
