r"""separate script for dropping repeated edges"""


import pandas as pd
import numpy as np
import os

from tqdm import tqdm


if __name__ == '__main__':
    path_dataset = r"/data/data2/041/datasets/mimic-iii-clinical-database-1.4"
    df_new_labevents = pd.read_csv(os.path.join(path_dataset, "LABEVENTS_NEW.csv.gz"))

    gb_hadmid = df_new_labevents.groupby("HADM_ID")

    drop_indexs = []
    for hadm_id in tqdm(df_new_labevents.HADM_ID.unique()):
        df_curr_hadmid = df_new_labevents[df_new_labevents.HADM_ID == hadm_id]

        for timestep in range(int(df_curr_hadmid.TIMESTEP.max()) + 1):
            df_curr_hadmid_curr_timestep = df_curr_hadmid[df_curr_hadmid.TIMESTEP == timestep]

            sr_itemid_value_counts = df_curr_hadmid_curr_timestep.ITEMID.value_counts()
            sr_itemid_repeat = sr_itemid_value_counts[sr_itemid_value_counts > 1]

            for itemid_repeat in list(sr_itemid_repeat.index):
                deprecate_entry_rowid = df_curr_hadmid_curr_timestep[
                                            df_curr_hadmid_curr_timestep.ITEMID == itemid_repeat].sort_values(
                    by="CHARTTIME").ROW_ID.iloc[0:-1]
                deprecate_entry_index = list(deprecate_entry_rowid.index)
                drop_indexs.extend(deprecate_entry_index)

        if len(drop_indexs) > 13333:
            df_new_labevents.drop(drop_indexs, inplace=True)
            drop_indexs = []

    df_new_labevents.drop(drop_indexs, inplace=True)
    df_new_labevents.to_csv(os.path.join(path_dataset, "LABEVENTS_NEW_remove_duplicate_edges.csv.gz"))  # DONE
