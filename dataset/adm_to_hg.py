import os
import torch
import sys; sys.path.append("..")
import utils.config as conf

from torch_geometric.utils import negative_sampling
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from utils.config import MappingManager, neg_sample_strategy
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
    l_vocab_size = MappingManager.node_type_to_node_num['labitem']

    adm_lens = [len(hgs) for hgs in batch_hgs]

    max_adm_len = max(adm_lens)
    max_adm_len = min(max_adm_len, conf.max_adm_length)  # 对最长住院天数长度进行截断，存在极端个例，住院长达873天；暂限制<=50

    batch_hgs = [hgs[:max_adm_len] for hgs in batch_hgs]  # 取前max_adm_len天
    for i, a_l in enumerate(adm_lens):  # 更新住院长度记录
        if a_l > max_adm_len:
            adm_lens[i] = max_adm_len

    batch_d_seq, batch_d_neg = [], []
    batch_l_seq, batch_l_neg = [], []
    for hgs in batch_hgs:
        all_day_d_seq, all_day_d_neg = [], []  # 收集当前住院过程每天的药物处方列表（正样本），没有安排的药物处方列表（负样本）
        all_day_l_seq, all_day_l_neg = [], [] 
        for hg in hgs:  # 遍历每天的图
            d_pos_indices = hg["admission", "took", "drug"].edge_index
            l_pos_indices = hg["admission", "did", "labitem"].edge_index
            
            d_neg_indices = neg_sample_for_cur_day(d_pos_indices, num_itm_nodes=d_vocab_size, strategy=neg_sample_strategy)
            l_neg_indices = neg_sample_for_cur_day(l_pos_indices, num_itm_nodes=l_vocab_size, strategy=neg_sample_strategy)

            cur_day_d_seq = d_pos_indices[1, :].tolist()
            cur_day_d_neg = d_neg_indices[1, :].tolist()
            
            cur_day_l_seq = l_pos_indices[1, :].tolist()
            cur_day_l_neg = l_neg_indices[1, :].tolist()

            all_day_d_seq.append(cur_day_d_seq)
            all_day_d_neg.append(cur_day_d_neg)
            
            all_day_l_seq.append(cur_day_l_seq)
            all_day_l_neg.append(cur_day_l_neg)

        batch_d_seq.append(all_day_d_seq)
        batch_d_neg.append(all_day_d_neg)
        
        batch_l_seq.append(all_day_l_seq)
        batch_l_neg.append(all_day_l_neg)

    return (batch_hgs, adm_lens,
            batch_d_seq, batch_d_neg,
            batch_l_seq, batch_l_neg) 


def neg_sample_for_cur_day(pos_indices, num_itm_nodes: int, strategy: int = 2):
    """
    Args:
        strategy: int
            - 2: 2:1 (neg:pos)
            - >=10 and < num_itm_nodes: sample `strategy` negative samples
            - -1: full item set
    """
    # 一张图里的病人结点固定为1个
    num_pos_edges = pos_indices.size(1)

    if num_pos_edges == 0:  # 若当前天没有正样本
        num_neg_samples = 10  # 保证最少有10个负样本
    else:
        if strategy == 2:
            num_neg_samples = num_pos_edges * 2
        elif 10 <= strategy < num_itm_nodes:
            num_neg_samples = strategy
        elif strategy == -1:
            num_neg_samples = num_itm_nodes
        else:
            raise f"invalid negative sample `strategy` args: {strategy}!"

    neg_indices = negative_sampling(pos_indices, (1, num_itm_nodes), num_neg_samples=num_neg_samples)

    return neg_indices


def get_batch_seq_to_be_judged_and_01_labels(batch_pos_seq, batch_neg_seq, device=torch.device('cpu')):
    # 使用正负样本构造[0,1] labels
    batch_pos_labels = [pos_seq[1:] for pos_seq in batch_pos_seq]  # 排除第一天
    batch_neg_labels = [neg_seq[1:] for neg_seq in batch_neg_seq]

    batch_seq_to_be_judged = []
    batch_01_labels = []
    for i, (pos_labels, neg_labels) in enumerate(zip(batch_pos_labels, batch_neg_labels)):  # 遍历病人
        all_day_seq_to_be_judged = []
        all_day_01_labels = []
        for j, (cur_day_pos, cur_day_neg) in enumerate(zip(pos_labels, neg_labels)):  # 遍历每天
            # *** 将上一天安排的药物/检验项目加入当前天的评估队列中 ***
            # 这里的j表示扣除第一天的序列中的天，实际上的表示j + 1天
            pre_day_pos = batch_pos_seq[i][j]  # 因此这样就能获取到前一天的正样本
            # 获取前一天正样本 与 当前天正样本 的差集，作为强负样本加入
            cur_pre_dec = list(set(pre_day_pos) - set(cur_day_pos))

            # 将差集加入到当前天的负样本中，注意去重
            cur_day_neg += cur_pre_dec
            cur_day_neg = list(set(cur_day_neg))

            t_cur_day_pos = torch.tensor(cur_day_pos, dtype=torch.int)
            t_cur_day_neg = torch.tensor(cur_day_neg, dtype=torch.int)

            cur_day_seq_to_be_judged = torch.cat((t_cur_day_pos, t_cur_day_neg))
            cur_day_01_labels = torch.cat((torch.ones(len(cur_day_pos)),
                                           torch.zeros(len(cur_day_neg))))

            # 打乱顺序
            index_shuffle = torch.randperm(cur_day_01_labels.size(0))

            all_day_seq_to_be_judged.append(cur_day_seq_to_be_judged[index_shuffle].to(device))
            all_day_01_labels.append(cur_day_01_labels[index_shuffle].to(device))

        batch_seq_to_be_judged.append(all_day_seq_to_be_judged)
        batch_01_labels.append(all_day_01_labels)

    return batch_seq_to_be_judged, batch_01_labels


if __name__ == "__main__":
    test_dataset = OneAdmOneHetero(r"..\data", "test")
    test_dataloader = DataLoader(test_dataset, batch_size=4, collate_fn=collect_hgs, pin_memory=True)

    for batch in test_dataloader:
        batch_hgs, adm_lens, \
            batch_d_seq, batch_d_neg, \
            batch_l_seq, batch_l_neg = batch

        batch_d_seq_to_be_judged, batch_d_01_labels = \
            get_batch_seq_to_be_judged_and_01_labels(batch_d_seq, batch_d_neg)
        batch_l_seq_to_be_judged, batch_l_01_labels = \
            get_batch_seq_to_be_judged_and_01_labels(batch_l_seq, batch_l_neg)

        break
