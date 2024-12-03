import sys; sys.path.append('..')
import torch.nn as nn
import torch.nn.functional as F
import torch

from d2l import torch as d2l
from torch_geometric.nn import to_hetero
from tqdm import tqdm
from dataset.unified import (SourceDataFrames,
                             OneAdmOneHG,
                             list_selected_admission_columns,
                             list_selected_labitems_columns,
                             list_selected_drug_ndc_columns,
                             list_selected_labevents_columns,
                             list_selected_prescriptions_columns)
from utils.config import HeteroGraphConfig, MappingManager, GNNConfig, max_adm_length
from utils.enum_type import FeatureType
from utils.misc import init_seed
from model.layers import LinksPredictor, SingelGnn, GraphEmbeddingLayer, AdditiveAttention
from model.init import str2init


class BackBoneV2(nn.Module):
    def __init__(self,
                 source_dfs: SourceDataFrames,
                 goal: str,
                 h_dim: int,
                 gnn_conf: GNNConfig,
                 device,
                 num_enc_layers: int = 6,
                 embedding_size: int = 10,
                 is_gnn_only: bool = False,
                 init_method: str = "xavier_normal",
                 **kwargs):
        super().__init__()
        self.source_dfs = source_dfs  # 提供有用的信息
        assert goal in ["drug", "labitem"]
        self.goal = goal

        self.h_dim = h_dim
        self.embedding_size = embedding_size
        self.gnn_conf = gnn_conf  # 可以用来控制纳入考虑的边类型
        self.device = device
        self.is_gnn_only = is_gnn_only

        self.node_types = self.gnn_conf.node_types
        self.edge_types = self.gnn_conf.edge_types

        self.item_vocab_size = {
            node_type: self.gnn_conf.mapper.node_type_to_node_num[node_type]
            for node_type in self.node_types if node_type != "admission"
        }
        self.item_id_embedding = nn.ModuleDict({
            node_type: nn.Embedding(self.item_vocab_size[node_type], self.embedding_size)
            for node_type in self.node_types if node_type != "admission"
        })
        self.node_features_embedding = nn.ModuleDict({
            node_type: GraphEmbeddingLayer(self.embedding_size, *self._get_node_feat_dims(node_type))
            for node_type in self.node_types
        })
        self.edge_features_embedding = nn.ModuleDict({
            "_".join(edge_type): GraphEmbeddingLayer(self.embedding_size, *self._get_edge_feat_dims(edge_type))
            for edge_type in self.edge_types if "rev" not in edge_type[1]
        })

        # 用于对齐emb完的维度
        self.node_features_aligner = nn.ModuleDict({
            node_type: nn.Linear(
                self.embedding_size * (
                        len(self.node_features_embedding[node_type].token_field_dims) +
                        int(self.node_features_embedding[node_type].float_field_nums > 0) +
                        int(node_type != "admission")
                ),
                self.h_dim
            )
            for node_type in self.node_types
        })
        self.edge_features_aligner = nn.ModuleDict({
            "_".join(edge_type): nn.Linear(
                self.embedding_size * (
                        len(self.edge_features_embedding["_".join(edge_type)].token_field_dims) +
                        int(self.edge_features_embedding["_".join(edge_type)].float_field_nums > 0)
                ),
                self.h_dim
            )
            for edge_type in self.edge_types if "rev" not in edge_type[1]
        })

        if not self.is_gnn_only:
            self.attention = AdditiveAttention(num_hiddens=self.h_dim, dropout=0.1)

        self.gnn = SingelGnn(self.h_dim, self.gnn_conf.gnn_type, self.gnn_conf.gnn_layer_num)
        self.gnn = to_hetero(self.gnn, metadata=(self.gnn_conf.node_types, self.gnn_conf.edge_types))

        # Final links predictor
        self.lp = LinksPredictor(self.h_dim, "mul")

        # parameters initialization
        self.apply(str2init[init_method])

    def _get_node_feat_dims(self, node_type):
        if node_type == "admission":
            fields = list_selected_admission_columns
        elif node_type == "labitem":
            fields = list_selected_labitems_columns
        elif node_type == "drug":
            fields = list_selected_drug_ndc_columns
        else:
            raise NotImplementedError

        token_field_dims = []
        float_field_nums = 0
        for field in fields:
            if self.source_dfs.field2type[field] == FeatureType.TOKEN:
                token_field_dims.append(
                    len(self.source_dfs.tokenfields2mappedid[field])
                )
            elif self.source_dfs.field2type[field] == FeatureType.FLOAT:
                float_field_nums += 1
            else:
                raise NotImplementedError

        return token_field_dims, float_field_nums

    def _get_edge_feat_dims(self, edge_type):
        if edge_type == ('admission', 'did', 'labitem'):
            fields = list_selected_labevents_columns
        elif edge_type == ("admission", "took", "drug"):
            fields = list_selected_prescriptions_columns
        else:
            raise NotImplementedError

        token_field_dims = []
        float_field_nums = 0
        for field in fields:
            if self.source_dfs.field2type[field] == FeatureType.TOKEN:
                token_field_dims.append(
                    len(self.source_dfs.tokenfields2mappedid[field]))
            elif self.source_dfs.field2type[field] == FeatureType.FLOAT:
                float_field_nums += 1
            else:
                raise NotImplementedError

        return token_field_dims, float_field_nums

    def forward(self, hg):
        # emb
        for node_type in self.node_types:
            if node_type != "admission":
                emb_node_features = self.node_features_embedding[node_type](hg[node_type].x)
                emb_node_id = self.item_id_embedding[node_type](hg[node_type].node_id).unsqueeze(1)
                emb_features = torch.cat([emb_node_id, emb_node_features], dim=1)
                emb_features = self.node_features_aligner[node_type](emb_features.flatten(1))
                hg[node_type].x = emb_features
            else:  # admission
                emb_node_features = self.node_features_embedding[node_type](hg[node_type].x)
                emb_features = self.node_features_aligner[node_type](emb_node_features.flatten(1))
                hg[node_type].x = emb_features

        for edge_type in self.edge_types:
            if "rev" in edge_type[1]:  # 没有按时间分划的原始图中，没有反向连接的边，因此不用处理
                continue
            else:
                emb_edge_features = self.edge_features_embedding["_".join(edge_type)](hg[edge_type].x)
                emb_features = self.edge_features_aligner["_".join(edge_type)](emb_edge_features.flatten(1))
                hg[edge_type].x = emb_features

        # 按天进行分割，获取离散时间动态图
        total_hgs = OneAdmOneHG.split_by_day(hg)
        total_hgs = total_hgs[:max_adm_length]  # 设置最长长度限制

        # 打包成 mini-batch 供GNN并行处理
        packed_hgs = OneAdmOneHG.pack_batch(total_hgs, len(total_hgs))

        packed_x_collection = packed_hgs.collect('x')
        packed_node_feats_ori = {k: x for k, x in packed_x_collection.items() if k in self.node_types}
        packed_edge_feats_ori = {k: x for k, x in packed_x_collection.items() if k in self.edge_types}

        # 这里面有batch norm，这也是为什么至少要有2天及以上的住院时长（确保batch_size > 1）
        node_feats_enc = self.gnn(packed_node_feats_ori, packed_hgs.edge_index_dict, packed_edge_feats_ori)

        # 拆分每天的物品特征，每个图的物品结点数量固定
        item_feats_enc = {
            node_type: x.view(-1, self.gnn_conf.mapper.node_type_to_node_num[node_type], self.h_dim)
            for node_type, x in node_feats_enc.items() if (node_type != "admission" and node_type in self.node_types)
        }

        patient_conditions = node_feats_enc["admission"].unsqueeze(0)  # batch size = 1

        logits = []
        labels = []
        items = []
        for d, cur_day_hg in enumerate(total_hgs[1:]):  # 这里要扣除第一天，因为我们预测从第二天开始的序列
            pre_day_patient_conditions = patient_conditions[:, :d+1, :]  # 之前天的病情表示
            pre_day_item_feats_enc = item_feats_enc[self.goal][d, :, :]  # 前一天的物品emb

            cur_day_seq_to_be_judged, cur_day_01_labels = self._get_cur_day_seq_to_be_judged_and_labels(cur_day_hg)
            cur_day_seq_to_be_judged_emb = pre_day_item_feats_enc[cur_day_seq_to_be_judged]  # 取出相应行

            bsz = cur_day_seq_to_be_judged_emb.size(0)
            cur_day_seq_to_be_judged_emb = cur_day_seq_to_be_judged_emb.view(bsz, 1, -1)
            pre_day_patient_conditions = pre_day_patient_conditions.repeat(bsz, 1, 1)

            # 注意力计算
            # queries：当前天的目标物品（待判断的药品/检验项目）
            # keys: 之前天的患者病情序列
            # values: 之前天的患者病情序列
            # 相当于用当前天的目标物品，去问“要怎样注意对之前天的病情？”
            if not self.is_gnn_only:
                att_patient_conditions = self.attention(
                    queries=cur_day_seq_to_be_judged_emb,
                    keys=pre_day_patient_conditions,
                    values=pre_day_patient_conditions,
                    valid_lens=None
                )
            else:
                # 不用注意力的话，就直接取昨天的患者病情表示
                att_patient_conditions = pre_day_patient_conditions[:, -1, :]

            cur_day_logits = self.lp(
                att_patient_conditions.view(bsz, -1),
                cur_day_seq_to_be_judged_emb.view(bsz, -1)
            )

            logits.append(cur_day_logits)
            labels.append(cur_day_01_labels)
            items.append(cur_day_seq_to_be_judged)

        return logits, labels, items  # 按天收集

    def _get_cur_day_seq_to_be_judged_and_labels(self, hg):
        r"""获取当天需要判断的物品序列"""

        # STEP 1: 负采样
        if self.goal == "drug":
            pos_indices = hg["admission", "took", "drug"].edge_index
            neg_indices = OneAdmOneHG.neg_sample_for_cur_day(
                pos_indices, num_itm_nodes=self.gnn_conf.mapper.node_type_to_node_num["drug"])
        else:  # "labitem"
            pos_indices = hg["admission", "did", "labitem"].edge_index
            neg_indices = OneAdmOneHG.neg_sample_for_cur_day(
                pos_indices, num_itm_nodes=self.gnn_conf.mapper.node_type_to_node_num["labitem"])

        # STEP 2：构建正负序列及标签
        cur_day_pos = pos_indices[1, :]
        cur_day_neg = neg_indices[1, :]

        cur_day_seq_to_be_judged = torch.cat([cur_day_pos, cur_day_neg])
        cur_day_01_labels = torch.cat([torch.ones(cur_day_pos.size(0)),
                                      torch.zeros(cur_day_neg.size(0))])

        # STEP 3：打乱顺序
        index_shuffle = torch.randperm(cur_day_01_labels.size(0))
        cur_day_seq_to_be_judged = cur_day_seq_to_be_judged[index_shuffle].to(self.device)
        cur_day_01_labels = cur_day_01_labels[index_shuffle].to(self.device)

        return cur_day_seq_to_be_judged, cur_day_01_labels

    @staticmethod
    def get_loss(logits, labels):
        logits = torch.cat(logits)
        labels = torch.cat(labels)
        return F.binary_cross_entropy_with_logits(logits, labels)


if __name__ == '__main__':
    init_seed(10043)

    sources_dfs = SourceDataFrames(r"..\data\mimic-iii-clinical-database-1.4")
    dataset = OneAdmOneHG(sources_dfs, "val")

    hidden_dim, device = 64, torch.device('cpu')
    node_types, edge_types = HeteroGraphConfig.use_all_edge_type()
    # node_types, edge_types = HeteroGraphConfig.use_one_edge_type("drug")
    gnn_conf = GNNConfig("GINEConv", 3, node_types, edge_types)
    model = BackBoneV2(sources_dfs, "drug", hidden_dim,
                       gnn_conf, device, 3, 10)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    accumulation_steps = 16

    for epoch in range(10):
        metric = d2l.Accumulator(2)
        loop = tqdm(enumerate(dataset), total=len(dataset), leave=False)
        for i, hg in loop:
            # https://github.com/pytorch/pytorch/issues/118499
            # with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            #     logits, labels = model(hg)
            #     loss = BackBoneV2.get_loss(logits, labels)
            logits, labels = model(hg)
            loss = BackBoneV2.get_loss(logits, labels)
            metric.add(loss.detach().item(), 1)
            loop.set_postfix_str(f"curr loss: {loss.detach().item():.4f}, avg. loss of epoch #{epoch:02}: {metric[0] / metric[1]:.4f}")
            loss = loss / accumulation_steps
            loss.backward()
            if (i+1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
