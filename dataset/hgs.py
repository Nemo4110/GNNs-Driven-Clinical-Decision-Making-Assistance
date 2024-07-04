import os
import sys; sys.path.append("..")
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.data import Dataset, HeteroData
from torch_geometric.utils import negative_sampling

from utils.config import MappingManager, max_seq_length


class SeqLabelConverter:
    r""" [B, X]

    X is set to 160, as max length (of someday) of
        - lab items: 119
        - drug: 143
    """
    @staticmethod
    def convert(sub_hg: HeteroData, edge_type, next_t_edge_index):
        """
        Note that `hg` here should not be converted to undirected graph by `T.ToUndirected()`
        """
        label = []
        for i in sub_hg["admission"].node_id:  # iterating nodes (each patient)
            mask = next_t_edge_index[0, :] == i
            curr_seq = next_t_edge_index[1, mask]

            curr_seq = F.pad(curr_seq, pad=(1, 0), mode='constant', value=MappingManager.node_type_to_special_token[edge_type[-1]].SOS)
            curr_seq = F.pad(curr_seq, pad=(0, 1), mode='constant', value=MappingManager.node_type_to_special_token[edge_type[-1]].EOS)
            curr_seq = F.pad(curr_seq, pad=(0, max_seq_length - curr_seq.size(0)),
                                                   mode='constant', value=MappingManager.node_type_to_special_token[edge_type[-1]].PAD)

            label.append(curr_seq)

        return torch.stack(label, dim=0)  # [B, X]


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

            sub_hg[edge_type].seq_labels = SeqLabelConverter.convert(sub_hg, edge_type, next_t_edge_index=sub_hg[edge_type].pos_index)

        # We also need to make sure to add the reverse edges from labitems to admission
        # in order to let a GNN be able to pass messages in both directions.
        # We can leverage the `T.ToUndirected()` transform for this from PyG:
        sub_hg = T.ToUndirected()(sub_hg)

        return sub_hg


if __name__ == "__main__":
    train_set = DiscreteTimeHeteroGraph(root_path=r".\batch_size_128", usage="train")
    for hg in train_set:
        sub_hg = DiscreteTimeHeteroGraph.get_subgraph_by_timestep(hg, 3)
        drug_seq_labels = sub_hg['admission', 'took', 'drug'].seq_labels
        print(drug_seq_labels.shape)
        print(drug_seq_labels[7])
        break
