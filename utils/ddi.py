r"""
Class used to calculate the DDI.
"""

import pandas as pd
import os
import pickle
import torch

from queue import Queue


class DDICalculator:
    def __init__(self, 
                 path_ddi_dataset=r"/data/data2/041/datasets/DDI") -> None:
        self.df_map_of_idx4ndc_rxcui_atc4_cids = pd.read_csv(os.path.join(path_ddi_dataset, "MAP_IDX4NDC_RXCUI_ATC4_CIDS.csv"))
        self.df_map_of_idx4ndc_rxcui_atc4_cids.sort_values(by='idx', inplace=True)

        with open(os.path.join(path_ddi_dataset, "ddi_adj_matrix.pickle"), 'rb') as f:
            self.ddi_adj = pickle.load(f)

    def calc_ddis_for_batch_admi(self, edge_labels, edge_indices):
        existing_edge_indices = torch.index_select(edge_indices, dim=1, index=torch.nonzero(edge_labels).flatten())
        set_admi = existing_edge_indices[0].unique()

        ddis = []
        for curr_admi in set_admi:
            indices_curr_hadm = torch.nonzero(existing_edge_indices[0] == curr_admi).flatten()
            durg_idxes_curr_admi = torch.index_select(existing_edge_indices, dim=1, index=indices_curr_hadm)[1]
            ddis.append(self.calc_ddi_rate(durg_idxes_curr_admi))

        return ddis

    def calc_ddi_rate(self, durg_idxes_curr_admi):
        mask = self.df_map_of_idx4ndc_rxcui_atc4_cids.idx.isin(durg_idxes_curr_admi)
        df_drugs_curr_admi = self.df_map_of_idx4ndc_rxcui_atc4_cids.loc[mask]
        df_drugs_can_calc_ddi = df_drugs_curr_admi[df_drugs_curr_admi.list_cid_idx.notnull()]

        q = Queue()
        for list_cid_idx in list(df_drugs_can_calc_ddi.list_cid_idx.values):
            q.put(list_cid_idx)

        cnt_all_pair = 0
        cnt_ddi_pair = 0
        while q.qsize() > 1:
            list_curr_cid_idx = q.get()
            for curr_cid_idx in eval(list_curr_cid_idx):
                for list_other_cid_idx in q.queue:
                    for other_cid_idx in eval(list_other_cid_idx):
                        if self.ddi_adj[curr_cid_idx][other_cid_idx] > 0 or \
                           self.ddi_adj[other_cid_idx][curr_cid_idx] > 0:
                            cnt_ddi_pair += 1
                        cnt_all_pair += 1
                    
        return 0 if cnt_all_pair == 0 else cnt_ddi_pair / cnt_all_pair
