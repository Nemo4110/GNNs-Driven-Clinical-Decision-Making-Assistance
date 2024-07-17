import os
import torch
import sys; sys.path.append("..")
import utils.config as conf

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from utils.config import MappingManager
from dataset.hgs import DiscreteTimeHeteroGraph


class OneAdmOneHetero(Dataset):
    """一次住院 admission 过程用一个 Hetero graph表示"""
    def __init__(self, path, split) -> None:
        super().__init__()
        assert os.path.exists(path)
        assert split in ("train", "test")

        self.path = os.path.join(path, split)

        pt_files = [f for f in os.listdir(self.path) if f.endswith('.pt')]
        pt_files.sort(key=lambda filename: int(filename[:-3]))

        self.data = [torch.load(os.path.join(self.path, pt)) for pt in pt_files]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def collect_fn(batch):
    batch_hgs = [DiscreteTimeHeteroGraph.split_by_day(hg) for hg in batch]  # 每个患者的住院天数

    B = len(batch)   # batch shape: [B, ?], ?表示每个患者住院天数不一
    adm_lens = [len(hgs) for hgs in batch_hgs]
    max_adm_len = max(adm_lens)

    l_max_num = 0
    d_max_num = 0

    batch_l_seq, batch_l_lens = [], []
    batch_d_seq, batch_d_lens = [], []

    # get vocab size
    l_vocab_size = batch[0]["labitem"].node_id.size(0)
    d_vocab_size = batch[0]["drug"].node_id.size(0)

    for hgs in batch_hgs:
        # add EOS at the end of each day's seq
        # 约定：最终的embedding向量形状为 (vocab_size + 3, h_dim)
        #   SOS = vocab_size,
        #   EOS = vocab_size + 1,
        #   PAD = vocab_size + 2
        lbe = [torch.cat((hg["admission", "did", "labitem"].edge_index[1, :], torch.tensor([l_vocab_size + 1])), dim=0) for hg in hgs]
        prs = [torch.cat((hg["admission", "took",   "drug"].edge_index[1, :], torch.tensor([d_vocab_size + 1])), dim=0) for hg in hgs]

        batch_l_seq.append(lbe)
        batch_d_seq.append(prs)

        # 收集每天的序列长度
        l_lens = [l_seq.size(0) for l_seq in lbe]
        d_lens = [d_seq.size(0) for d_seq in prs]

        batch_l_lens.append(l_lens)
        batch_d_lens.append(d_lens)

        # 不断更新每天中的最长序列长度
        l_max_num = max(l_max_num, max(l_lens))
        d_max_num = max(d_max_num, max(d_lens))

    # 用相应的PAD id进行填充
    batch_l = torch.full((B, max_adm_len, l_max_num), l_vocab_size + 2)
    batch_d = torch.full((B, max_adm_len, d_max_num), d_vocab_size + 2)

    batch_l_mask = torch.full((B, max_adm_len, l_max_num), True, dtype=torch.bool)
    batch_d_mask = torch.full((B, max_adm_len, d_max_num), True, dtype=torch.bool)

    T_batch_l_lens = torch.full((B, max_adm_len), 0)
    T_batch_d_lens = torch.full((B, max_adm_len), 0)

    for i in range(B):  # adm
        for j in range(adm_lens[i]):  # day
            batch_l[i, j, :batch_l_lens[i][j]] = batch_l_seq[i][j]
            batch_d[i, j, :batch_d_lens[i][j]] = batch_d_seq[i][j]

            batch_l_mask[i, j, :batch_l_lens[i][j]] = False
            batch_d_mask[i, j, :batch_d_lens[i][j]] = False

            T_batch_l_lens[i, j] = batch_l_lens[i][j]
            T_batch_d_lens[i, j] = batch_d_lens[i][j]

    return batch_hgs, \
        batch_l, batch_l_mask, T_batch_l_lens, \
        batch_d, batch_d_mask, T_batch_d_lens


def collect_hgs(batch):
    batch_hgs = [DiscreteTimeHeteroGraph.split_by_day(hg) for hg in batch]
    B = len(batch)   # batch shape: [B, ?], ?表示每个患者住院天数不一
    d_vocab_size = MappingManager.node_type_to_node_num['drug']

    adm_lens = [len(hgs) for hgs in batch_hgs]

    max_adm_len = max(adm_lens)
    max_adm_len = min(max_adm_len, conf.max_adm_length)  # 对最长住院天数长度进行截断，存在极端个例，住院长达873天；暂限制<=50

    batch_hgs = [hgs[:max_adm_len] for hgs in batch_hgs]  # 取前max_adm_len天
    for i, a_l in enumerate(adm_lens):  # 更新住院长度记录
        if a_l > max_adm_len:
            adm_lens[i] = max_adm_len

    batch_d_seq = []
    for hgs in batch_hgs:
        prs = [hg["admission", "took",   "drug"].edge_index[1, :] for hg in hgs]
        batch_d_seq.append(prs)

    batch_d_flat = torch.full((B, max_adm_len, d_vocab_size), 0.)
    for i in range(B):
        for j in range(adm_lens[i]):
            batch_d_flat[i, j, batch_d_seq[i][j]] = 1.

    return batch_hgs, batch_d_flat, torch.tensor(adm_lens)


if __name__ == "__main__":
    test_dataset = OneAdmOneHetero(r"..\data", "test")
    test_dataloader = DataLoader(test_dataset, batch_size=16, collate_fn=collect_fn, pin_memory=True)

    for batch in test_dataloader:
        batch_hgs, \
            batch_l, batch_l_mask, batch_l_lens, \
            batch_d, batch_d_mask, batch_d_lens = batch
        break
