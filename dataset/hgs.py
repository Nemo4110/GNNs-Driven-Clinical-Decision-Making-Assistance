import os
import torch

from torch_geometric.data import Dataset
from tqdm import tqdm


class MyOwnDataset(Dataset):
    def __init__(self, root_path: str, usage: str):
        super().__init__()

        assert os.path.exists(root_path), f"The dataset root path >>>{root_path}<<< isn't exist!"
        assert usage in ["train", "val", "test"], "Only support usage in ['train', 'val', 'test']!"

        self.root_path = os.path.join(root_path, usage)
        self.pt_files = [f for f in os.listdir(self.root_path) if f[-3:]==".pt"]

    def len(self):
        return len(self.pt_files)

    def get(self, idx):
        return torch.load(os.path.join(self.root_path, self.pt_files[idx]))


if __name__ == "__main__":
    train_set = MyOwnDataset(root_path=r"D:\Datasets\mimic\mimic-iii-hgs\batch_size_128", usage="train")
    for hg in tqdm(train_set):
        assert max(hg["admission"].node_id) >= max(hg["admission", "did", "labitem"].edge_index[0])