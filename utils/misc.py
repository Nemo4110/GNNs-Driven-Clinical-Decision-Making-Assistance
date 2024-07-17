import torch
import torch.nn.functional as F
import os
import glob

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


def get_latest_threshold(bth_path):
    files = glob.glob(os.path.join(bth_path, '*.pickle'))
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    return os.path.basename(latest_file)


def get_latest_model_ckpt(folder_path):
    # 获取指定文件夹下的所有.pt文件路径
    files = glob.glob(os.path.join(folder_path, '*.pt'))

    # 如果文件夹为空，返回 None
    if not files:
        return None

        # 找到最新的文件
    latest_file = max(files, key=os.path.getmtime)

    # 返回最新文件的文件名
    return os.path.basename(latest_file)


if __name__ == "__main__":
    latest_model_ckpt = get_latest_model_ckpt(r"..\model\hub")
    print(latest_model_ckpt)
