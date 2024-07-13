import torch
import torch.nn.functional as F

from typing import List, Dict, Tuple


def calc_loss(dict_every_day_pred, node_types: List[str], device, loss_f):
    loss = torch.tensor(0.0).to(device)
    for node_type in node_types:
        if node_type == "admission":
            continue
        for scores, labels in zip(dict_every_day_pred[node_type]["scores"],
                                    dict_every_day_pred[node_type]["labels"]):
            loss += loss_f(scores, labels)
    return loss


def sequence_mask(sequence: torch.tensor, valid_len: torch.tensor):
    """为序列生成相应的mask， 标记pad

    Args:
        sequence: (B, L) or (B, L, h_dim)
        valid_len: (B,)
    """
    max_len = sequence.size(1)
    mask = torch.arange((max_len), dtype=torch.float32,
                        device=sequence.device)[None, :] < valid_len[:, None]
    return ~mask


node_type_to_prefix = {
    'labitem':"ITEMS",
    'drug': "DRUGS"
}
