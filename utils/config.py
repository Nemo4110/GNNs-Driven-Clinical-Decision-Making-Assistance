from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass
class MappingManager:
    node_type_to_node_num = {
        "labitem": 1622,
        "drug": 5967,
    }
    node_type_to_node_feat_dim_in = {
        "admission": 7,
        "labitem": 2,
        "drug": 8
    }
    edge_type_to_edge_feat_dim_in = {
        ('admission', 'did', 'labitem'): 2,
        ('labitem', 'rev_did', 'admission'): 2,
        ("admission", "took", "drug"): 7,
        ("drug", "rev_took", "admission"): 7
    }


class HeteroGraphConfig:
    @staticmethod
    def use_all_edge_type():
        node_types = ['admission', 'labitem', 'drug']
        edge_types = [('admission', 'did', 'labitem'), ('labitem', 'rev_did', 'admission'),
                      ("admission", "took", "drug"), ("drug", "rev_took", "admission")]
        return node_types, edge_types

    @staticmethod
    def use_one_edge_type(item_type: str):
        r"""
        Args:
            - item_type: 'drug' or 'labitem'
        """
        if item_type == 'drug':
            node_types = ['admission', 'drug']
            edge_types = [("admission", "took", "drug"), ("drug", "rev_took", "admission")]
        elif item_type == 'labitem':
            node_types = ['admission', 'labitem']
            edge_types = [('admission', 'did', 'labitem'), ('labitem', 'rev_did', 'admission')]
        else:
            raise NotImplementedError

        return node_types, edge_types


max_seq_length = 150
max_adm_length = 50
neg_sample_strategy = 2  # 默认使用2：1负采样策略


@dataclass
class GNNConfig:
    gnn_type: str
    gnn_layer_num: int
    node_types: List[str]
    edge_types: List[Tuple[str, str, str]]
    mapper = MappingManager()


if __name__ == "__main__":
    print(MappingManager.node_type_to_node_feat_dim_in['admission'])