import os
import sys; sys.path.append("..")
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.data import Dataset, HeteroData
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from utils.config import MappingManager, max_seq_length
from utils.constant import MASK
from typing import List


class DiscreteTimeHeteroGraph(Dataset):
    def __init__(self, root_path: str, usage: str):
        super().__init__()

        assert os.path.exists(root_path), f"The dataset root path >>>{root_path}<<< isn't exist!"
        assert usage in ["train", "test"], "Only support usage in ['train', 'test']!"

        self.root_path = os.path.join(root_path, usage)
        self.pt_files = [f for f in os.listdir(self.root_path) if f[-3:]==".pt"]

    def len(self):
        return len(self.pt_files)

    def get(self, idx):
        return torch.load(os.path.join(self.root_path, self.pt_files[idx]))

    @staticmethod
    def pack_batch(hgs: List[HeteroData], batch_size: int):
        r"""
        Args:
            hgs:
            batch_size: how many sub graphs (days) to pack into a batch
        """
        loader = DataLoader(hgs, batch_size=batch_size)
        return next(iter(loader))

    @staticmethod
    def get_subgraph_by_timestep(hg: HeteroData, timestep: int, neg_smp_strategy: int=0):
        r"""
        Params:
            - `neg_smp_stratege`: the stratege of negative sampling
                - \in [1, 100], assume each patient is assigned with `neg_smp_stratege` negative edges
                - 0 for 1: 2 (positive: negative
                - -1 for full-set
        """
        device = hg["admission", "did", "labitem"].timestep.device

        # https://discuss.pytorch.org/t/typeerror-expected-tensoroptions-dtype-float-device-cpu-layout-strided-requires-grad-false-default-pinned-memory-false-default-memory-format-nullopt/159558

        sub_hg = HeteroData()

        # NODE (copied directly)
        for node_type in hg.node_types:
            sub_hg[node_type].node_id = hg[node_type].node_id.clone()
            sub_hg[node_type].x = hg[node_type].x.clone().float()

        # Edges
        for edge_type in hg.edge_types:
            mask = (hg[edge_type].timestep == timestep).to(device)
            eidx = hg[edge_type].edge_index[:, mask]
            ex = hg[edge_type].x[mask, :]

            sub_hg[edge_type].edge_index = eidx.clone()
            sub_hg[edge_type].x = ex.clone().float()

            assert timestep < torch.max(hg[edge_type].timestep), "last timestep has not labels!"

            mask_next_t = (hg[edge_type].timestep == (timestep+1)).to(device)
            sub_hg[edge_type].pos_index = hg[edge_type].edge_index[:, mask_next_t].clone()

            num_pos_edges = sub_hg[edge_type].pos_index.shape[1]
            num_adm_nodes = sub_hg["admission"].node_id.shape[0]
            num_itm_nodes = sub_hg[edge_type[-1]].node_id.shape[0]
            if neg_smp_strategy == 0:
                sub_hg[edge_type].neg_index = negative_sampling(sub_hg[edge_type].pos_index, (num_adm_nodes, num_itm_nodes), num_neg_samples=num_pos_edges * 2).to(device)
            elif 1 <= neg_smp_strategy <= 100:
                sub_hg[edge_type].neg_index = negative_sampling(sub_hg[edge_type].pos_index, (num_adm_nodes, num_itm_nodes), num_neg_samples=num_pos_edges * neg_smp_strategy).to(device)
            elif neg_smp_strategy == -1:
                sub_hg[edge_type].neg_index = negative_sampling(sub_hg[edge_type].pos_index, (num_adm_nodes, num_itm_nodes), num_neg_samples=num_adm_nodes * num_itm_nodes).to(device)
            else:
                raise ValueError

            sub_hg[edge_type].labels_index = torch.cat((sub_hg[edge_type].pos_index, sub_hg[edge_type].neg_index), dim=1)
            sub_hg[edge_type].labels       = torch.cat(
                (torch.ones(sub_hg[edge_type].pos_index.shape[1]),
                          torch.zeros(sub_hg[edge_type].neg_index.shape[1])),
                dim=0
            ).to(device)
            index_shuffle = torch.randperm(sub_hg[edge_type].labels_index.shape[1]).to(device)
            sub_hg[edge_type].labels_index = sub_hg[edge_type].labels_index[:, index_shuffle]
            sub_hg[edge_type].labels = sub_hg[edge_type].labels[index_shuffle]

        # We also need to make sure to add the reverse edges from labitems to admission
        # in order to let a GNN be able to pass messages in both directions.
        # We can leverage the `T.ToUndirected()` transform for this from PyG:
        sub_hg = T.ToUndirected()(sub_hg)

        return sub_hg

    @staticmethod
    def split_by_day(hg: HeteroData):
        hgs = []

        last_lbe_day = hg["admission", "did", "labitem"].timestep.max().int().item()
        last_pre_day = hg["admission", "took", "drug"].timestep.max().int().item()
        adm_len = max(last_lbe_day, last_pre_day)

        device = hg["admission", "did", "labitem"].timestep.device

        for cur_day in range(adm_len + 1):
            sub_hg = HeteroData()

            # NODE (copied directly)
            for node_type in hg.node_types:
                sub_hg[node_type].node_id = hg[node_type].node_id.clone()
                sub_hg[node_type].x = hg[node_type].x.clone().float()

            # Edges
            for edge_type in hg.edge_types:
                mask = (hg[edge_type].timestep == cur_day).to(device)

                edge_index = hg[edge_type].edge_index[:, mask]
                ex = hg[edge_type].x[mask, :]

                sub_hg[edge_type].edge_index = edge_index.clone()
                sub_hg[edge_type].x = ex.clone().float()

            sub_hg = T.ToUndirected()(sub_hg)

            hgs.append(sub_hg)

        return hgs


if __name__ == "__main__":
    pass
    # dataset = DiscreteTimeHeteroGraph(root_path=r".\batch_size_128", usage="train")
    # hg = dataset[0]
    # max_timestep = 20

    # hgs = [DiscreteTimeHeteroGraph.get_subgraph_by_timestep(hg, timestep=t, neg_smp_strategy=0) for t in range(max_timestep)]
    # loader = DataLoader(hgs, batch_size=max_timestep)
    # batch_hg = next(iter(loader))
    # print(batch_hg)
    # for k, v in batch_hg.collect('x').items():
    #     print(k, v.shape)
