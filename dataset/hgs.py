import os
import torch
import torch_geometric.transforms as T

from torch_geometric.data import Dataset, HeteroData
from torch_geometric.utils import negative_sampling
from tqdm import tqdm


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
        mask4item = (hg["admission", "did", "labitem"].timestep == timestep).to(device)
        eidx4item = hg["admission", "did", "labitem"].edge_index[:, mask4item]
        ex4item   = hg["admission", "did", "labitem"].x[mask4item, :]

        mask4drug = (hg["admission", "took", "drug"].timestep == timestep).to(device)
        eidx4drug = hg["admission", "took", "drug"].edge_index[:, mask4drug]
        ex4drug   = hg["admission", "took", "drug"].x[mask4drug, :]

        sub_hg = HeteroData()

        # Nodes
        sub_hg["admission"].node_id = hg["admission"].node_id.clone()
        sub_hg["admission"].x       = hg["admission"].x.clone().float()

        sub_hg["labitem"].node_id = hg["labitem"].node_id.clone()
        sub_hg["labitem"].x       = hg["labitem"].x.clone().float()

        sub_hg["drug"].node_id = hg["drug"].node_id.clone()
        sub_hg["drug"].x       = hg["drug"].x.clone().float()

        # Edges
        sub_hg["admission", "did", "labitem"].edge_index = eidx4item.clone()
        sub_hg["admission", "did", "labitem"].x          = ex4item.clone().float()

        sub_hg["admission", "took", "drug"].edge_index = eidx4drug.clone()
        sub_hg["admission", "took", "drug"].x          = ex4drug.clone().float()

        assert timestep < torch.max(hg["admission", "did", "labitem"].timestep), "last timestep has not labels!"
        assert timestep < torch.max(hg["admission", "took", "drug"].timestep),   "last timestep has not labels!"

        mask_next_t4item = (hg["admission", "did", "labitem"].timestep == (timestep+1)).to(device)
        sub_hg.labels4item_pos_index = hg["admission", "did", "labitem"].edge_index[:, mask_next_t4item].clone()
        if neg_smp_strategy == 0:
            sub_hg.labels4item_neg_index = negative_sampling(
                sub_hg.labels4item_pos_index,
                num_neg_samples=sub_hg.labels4item_pos_index.shape[1] * 2,
                num_nodes=(sub_hg["admission"].node_id.shape[0], sub_hg["labitem"].node_id.shape[0])
            ).to(device)
        elif 1 <= neg_smp_strategy <= 100:
            sub_hg.labels4item_neg_index = negative_sampling(
                sub_hg.labels4item_pos_index,
                num_neg_samples=sub_hg["admission"].node_id.shape[0] * neg_smp_strategy,
                num_nodes=(sub_hg["admission"].node_id.shape[0], sub_hg["labitem"].node_id.shape[0])
            ).to(device)
        elif neg_smp_strategy == -1:  # use the full labitem set
            sub_hg.labels4item_neg_index = negative_sampling(
                sub_hg.labels4item_pos_index,
                num_neg_samples=sub_hg["admission"].node_id.shape[0] * sub_hg["labitem"].node_id.shape[0],
                num_nodes=(sub_hg["admission"].node_id.shape[0], sub_hg["labitem"].node_id.shape[0])
            ).to(device)
        else:
            raise ValueError
        sub_hg.lables4item_index = torch.cat((sub_hg.labels4item_pos_index, sub_hg.labels4item_neg_index), dim=1)
        sub_hg.lables4item = torch.cat((torch.ones(sub_hg.labels4item_pos_index.shape[1]),
                                        torch.zeros(sub_hg.labels4item_neg_index.shape[1])), dim=0).to(device)
        index4item_shuffle = torch.randperm(sub_hg.lables4item_index.shape[1]).to(device)
        sub_hg.lables4item_index = sub_hg.lables4item_index[:, index4item_shuffle]
        sub_hg.lables4item = sub_hg.lables4item[index4item_shuffle]

        mask_next_t4drug = (hg["admission", "took", "drug"].timestep == (timestep+1)).to(device)
        sub_hg.labels4drug_pos_index = hg["admission", "took", "drug"].edge_index[:, mask_next_t4drug].clone()
        if neg_smp_strategy == 0:
            sub_hg.labels4drug_neg_index = negative_sampling(
                sub_hg.labels4drug_pos_index,
                num_neg_samples=sub_hg.labels4drug_pos_index.shape[1] * 2,
                num_nodes=(sub_hg["admission"].node_id.shape[0], sub_hg["drug"].node_id.shape[0])
            ).to(device)
        elif 1 <= neg_smp_strategy <= 100:
            sub_hg.labels4drug_neg_index = negative_sampling(
                sub_hg.labels4drug_pos_index,
                num_neg_samples=sub_hg["admission"].node_id.shape[0] * neg_smp_strategy,
                num_nodes=(sub_hg["admission"].node_id.shape[0], sub_hg["drug"].node_id.shape[0])
            ).to(device)
        elif neg_smp_strategy == -1:  # use the full labitem set
            sub_hg.labels4drug_neg_index = negative_sampling(
                sub_hg.labels4drug_pos_index,
                num_neg_samples=sub_hg["admission"].node_id.shape[0] * sub_hg["drug"].node_id.shape[0],
                num_nodes=(sub_hg["admission"].node_id.shape[0], sub_hg["drug"].node_id.shape[0])
            ).to(device)
        else:
            raise ValueError
        sub_hg.labels4drug_index = torch.cat((sub_hg.labels4drug_pos_index, sub_hg.labels4drug_neg_index), dim=1)
        sub_hg.labels4drug = torch.cat((torch.ones(sub_hg.labels4drug_pos_index.shape[1]),
                                        torch.zeros(sub_hg.labels4drug_neg_index.shape[1])), dim=0).to(device)
        index4drug_shuffle = torch.randperm(sub_hg.labels4drug_index.shape[1]).to(device)
        sub_hg.labels4drug_index = sub_hg.labels4drug_index[:, index4drug_shuffle]
        sub_hg.labels4drug = sub_hg.labels4drug[index4drug_shuffle]

        # We also need to make sure to add the reverse edges from labitems to admission
        # in order to let a GNN be able to pass messages in both directions.
        # We can leverage the `T.ToUndirected()` transform for this from PyG:
        sub_hg = T.ToUndirected()(sub_hg)

        return sub_hg


if __name__ == "__main__":
    train_set = DiscreteTimeHeteroGraph(root_path=r"D:\Datasets\mimic\mimic-iii-hgs-new\batch_size_128", usage="val")
    for hg in tqdm(train_set):
        assert max(hg["admission"].node_id) >= max(hg["admission", "did", "labitem"].edge_index[0])