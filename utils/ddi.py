r"""
Class used to calculate the DDI.
"""

import os
import pickle
import torch
import dill
import pandas as pd
# https://stackoverflow.com/questions/20625582/
pd.options.mode.chained_assignment = None  # default='warn'


class DDICalculator:
    def __init__(self, 
                 path_ddi_dataset=r"/data/data2/041/datasets/DDI") -> None:
        # use `index_col=0` argument to avoid `unnamed :0`
        # <https://stackoverflow.com/questions/53988226/pd-read-csv-add-column-named-unnamed-0>
        self.df_map_of_idx4ndc_rxcui_atc4_cids = pd.read_csv(os.path.join(path_ddi_dataset, "MAP_IDX4NDC_RXCUI_ATC4_CIDS.csv"), index_col=0)
        self.df_map_of_idx4ndc_rxcui_atc4_cids = self.df_map_of_idx4ndc_rxcui_atc4_cids.drop(columns=['list_cid', 'list_cid_idx'])

        self.df_map_of_idx4ndc_rxcui_atc4_cids['ATC3'] = self.df_map_of_idx4ndc_rxcui_atc4_cids['ATC4'].map(lambda x: x[:4], na_action='ignore')

        self.df_map_of_idx4ndc_rxcui_atc4_cids.sort_values(by='idx', inplace=True)

        with open(os.path.join(path_ddi_dataset, "voc_final.pkl"), 'rb') as f:
            self.med_voc = dill.load(f)['med_voc']
        self.med_unique_word = list(self.med_voc.word2idx.keys())

        with open(os.path.join(path_ddi_dataset, "ddi_A_final.pkl"), 'rb') as f:
            self.ddi_adj = dill.load(f)

    def calc_ddis_for_batch_admi(self, edge_labels, edge_indices):
        existing_edge_indices = torch.index_select(edge_indices, dim=1, index=torch.nonzero(edge_labels).flatten())
        set_admi = existing_edge_indices[0].unique()

        ddis = []
        for curr_admi in set_admi:
            indices_curr_hadm = torch.nonzero(existing_edge_indices[0] == curr_admi).flatten()
            durg_idxes_curr_admi = torch.index_select(existing_edge_indices, dim=1, index=indices_curr_hadm)[1]
            ddis.append(self.calc_ddi_rate(durg_idxes_curr_admi))

        return ddis

    def calc_ddi_rate(self, durg_idxes_curr_admi: torch.tensor):
        # `durg_idxes_curr_admi`: the indeices of drugs of current patient, waiting to calculate the DDI score
        mask = self.df_map_of_idx4ndc_rxcui_atc4_cids.idx.isin(durg_idxes_curr_admi.tolist())  # MUST tolist !!!
        df_drugs_curr_admi = self.df_map_of_idx4ndc_rxcui_atc4_cids.loc[mask]

        df_drugs_can_calc_ddi = df_drugs_curr_admi[df_drugs_curr_admi.ATC3.notnull()]
        df_drugs_can_calc_ddi = df_drugs_can_calc_ddi[df_drugs_can_calc_ddi.ATC3.isin(self.med_unique_word)]
        atc3s = df_drugs_can_calc_ddi.ATC3.unique()

        cnt_all = 0
        cnt_ddi = 0
        for i, atc3_i in enumerate(atc3s):
            idx_drug_i = self.med_voc.word2idx[atc3_i]

            for j, atc3_j in enumerate(atc3s):
                if j <= i: continue
                cnt_all += 1

                idx_drug_j = self.med_voc.word2idx[atc3_j]
                if self.ddi_adj[idx_drug_i, idx_drug_j] == 1 or \
                   self.ddi_adj[idx_drug_j, idx_drug_i] == 1:
                    cnt_ddi += 1

        if cnt_all == 0:
            return 0
        return cnt_ddi / cnt_all
