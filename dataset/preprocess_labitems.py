r"""
A separate script for preprocessing data, same function as `processes_labitems.ipynb`.
"""
import sys; sys.path.append("..")
import os.path as path
import numpy as np
import pandas as pd
import utils.constant as constant

from tqdm import tqdm
from datetime import datetime, timedelta


def preprocess_admission(src_csv_path, dst_csv_path, value_na=0):
    df_admissions = pd.read_csv(src_csv_path)
    df_admissions.columns = df_admissions.columns.str.upper()
    df_admissions["ADMITTIME"] = pd.to_datetime(df_admissions["ADMITTIME"], format="%Y-%m-%d %H:%M:%S")
    df_admissions["DISCHTIME"] = pd.to_datetime(df_admissions["DISCHTIME"], format="%Y-%m-%d %H:%M:%S")

    # Node_feature
    list_str_type_columns = ['ADMISSION_TYPE',
                             'ADMISSION_LOCATION',
                             'DISCHARGE_LOCATION',
                             'INSURANCE',
                             'LANGUAGE',
                             'MARITAL_STATUS',
                             'RACE']

    for c in list_str_type_columns:
        m = df_admissions[c].value_counts()
        m = pd.Series(index=m.index, data=range(1, len(m) + 1))
        df_admissions[c] = df_admissions[c].map(m)

    values_fillna = {}
    for c in list_str_type_columns:
        values_fillna[c] = value_na
    df_admissions.fillna(value=values_fillna, inplace=True)
    df_admissions.to_csv(dst_csv_path)


def preprocess_labitems(src_csv_path, dst_csv_path, value_na=0):
    df_d_labitems = pd.read_csv(src_csv_path)
    df_d_labitems.columns = df_d_labitems.columns.str.upper()
    list_str_type_columns = ['FLUID', 'CATEGORY']
    for c in list_str_type_columns:
        m = df_d_labitems[c].value_counts()
        m = pd.Series(index=m.index, data=range(1, len(m) + 1))
        df_d_labitems[c] = df_d_labitems[c].map(m)

    # In fact, there is none nan in this 2 columns, so following code can be ignored.
    values_fillna = {}
    for c in list_str_type_columns:
        values_fillna[c] = value_na
    df_d_labitems.fillna(value=values_fillna, inplace=True)

    df_d_labitems.to_csv(dst_csv_path)


def preprocess_labevents(src_csv_path, dst_csv_path, src_csv_path_admi, value_na=0):
    # read csv files
    df_labevents = pd.read_csv(src_csv_path)
    df_labevents.columns = df_labevents.columns.str.upper()
    df_labevents['ROW_ID'] = df_labevents.index  # mimic-iv 2.2 没有row_id列
    df_labevents.dropna(subset=['HADM_ID', 'ITEMID'], inplace=True)
    df_labevents.sort_values(by=["HADM_ID", "ITEMID"], inplace=True)
    df_labevents["CHARTTIME"] = pd.to_datetime(df_labevents["CHARTTIME"], format="%Y-%m-%d %H:%M:%S")

    df_admissions = pd.read_csv(src_csv_path_admi)
    df_admissions.columns = df_admissions.columns.str.upper()
    df_admissions["ADMITTIME"] = pd.to_datetime(df_admissions["ADMITTIME"], format="%Y-%m-%d %H:%M:%S")
    df_admissions["DISCHTIME"] = pd.to_datetime(df_admissions["DISCHTIME"], format="%Y-%m-%d %H:%M:%S")

    # group by ITEMID
    grouped_by_itemid_value_type_only = df_labevents[df_labevents.VALUENUM.notnull()].groupby("ITEMID")
    grouped_by_itemid_not_value_type = df_labevents[df_labevents.VALUENUM.isnull()].groupby("ITEMID")

    # **************************************************************************************************************** #
    print("*** Solving multi value type ***")
    set_itemid_value_type = list(grouped_by_itemid_value_type_only.groups.keys())
    set_itemid_not_value_type = list(grouped_by_itemid_not_value_type.groups.keys())

    set_itemid_pure_value_type = np.setdiff1d(set_itemid_value_type, set_itemid_not_value_type)
    set_itemid_mixed_value_type = np.setdiff1d(set_itemid_value_type, set_itemid_pure_value_type)
    set_itemid_pure_non_value_type = np.setdiff1d(set_itemid_not_value_type, set_itemid_mixed_value_type)

    # >>> Pure non-value type itemid <<< Need to re-map by categories
    for itemid in tqdm(set_itemid_pure_non_value_type):
        s = df_labevents[df_labevents.ITEMID == itemid].VALUE.value_counts()
        m = pd.Series(index=s.index, data=range(1, len(s) + 1))
        df_labevents.loc[df_labevents.ITEMID == itemid, 'CATAGORY'] = df_labevents[df_labevents.ITEMID == itemid].VALUE.map(m)

    df_labevents.CATAGORY.fillna(value_na, inplace=True)

    # >>> Mix value type itemid <<<
    for itemid in tqdm(set_itemid_mixed_value_type):
        mask = df_labevents[df_labevents.ITEMID == itemid].VALUENUM.isnull()
        list_index = df_labevents[df_labevents.ITEMID == itemid].loc[mask, :].index
        df_labevents.drop(list_index, inplace=True)

    # **************************************************************************************************************** #
    print("*** Z-SCORE ***")

    def three_theta(data: pd.Series):
        assert len(data) > 1

        std = data.std()
        mean = data.mean()

        up_bound = mean + 3 * std
        low_bound = mean - 3 * std

        mask = (data > low_bound) & (data < up_bound)
        return mask

    def z_score_4_value_type_labitem(df_grp):
        dfx = df_grp.copy()
        dfx_normal = dfx[dfx.FLAG.isnull()]

        if len(dfx_normal) > 1:
            mask = three_theta(dfx_normal.VALUENUM)
            dfx_normal_filted = dfx_normal.loc[mask, :]
            if len(dfx_normal_filted) > 1:
                mean = dfx_normal_filted.VALUENUM.mean()
                std = dfx_normal_filted.VALUENUM.std() + 1e-7
            elif len(dfx_normal_filted) == 1:
                mean = dfx_normal_filted.VALUENUM.mean()
                std = dfx_normal_filted.VALUENUM.std(ddof=0) + 1e-7
            else:  # meaning that the normal entries have same values 0
                mean = 0
                std = 0 + 1e-7
            dfx['VALUENUM_Z-SCORED'] = dfx['VALUENUM'].apply(lambda x: (x - mean) / std)
        elif len(dfx_normal) == 1:
            mean = dfx_normal.VALUENUM.mean()
            std = dfx_normal.VALUENUM.std(ddof=0) + 1e-7
            dfx['VALUENUM_Z-SCORED'] = dfx['VALUENUM'].apply(lambda x: (x - mean) / std)
        else:  # 0, meaning all enties' FLAG="abnormal"
            dfx['VALUENUM_Z-SCORED'] = 3  # 3 or -3? worth discussing, but here set to 3 for convenience.

        return dfx

    # df_itemid_value_type_only_zscore = grouped_by_itemid_value_type_only.apply(z_score_4_value_type_labitem)
    df_itemid_value_type_only_zscore = pd.DataFrame()
    list_df_itemid_value_type_only_zscore = []
    for k, df in tqdm(grouped_by_itemid_value_type_only):
        list_df_itemid_value_type_only_zscore.append(z_score_4_value_type_labitem(df))
        if len(list_df_itemid_value_type_only_zscore) % 3000 == 0:
            list_df_itemid_value_type_only_zscore.append(df_itemid_value_type_only_zscore)
            df_itemid_value_type_only_zscore = pd.concat(list_df_itemid_value_type_only_zscore)
            list_df_itemid_value_type_only_zscore = []
    list_df_itemid_value_type_only_zscore.append(df_itemid_value_type_only_zscore)
    df_itemid_value_type_only_zscore = pd.concat(list_df_itemid_value_type_only_zscore).reset_index(drop=True)

    df_labevents = df_labevents.merge(df_itemid_value_type_only_zscore[['ROW_ID', 'VALUENUM_Z-SCORED']], how='left', on='ROW_ID')
    df_labevents['VALUENUM_Z-SCORED'].fillna(value_na, inplace=True)

    # **************************************************************************************************************** #
    print("*** Adding TIMESTEP ***")
    grouped_by_hadmid = df_labevents.groupby("HADM_ID")

    def add_timestep_per_hadmid(df_grouped_by_hadmid: pd.DataFrame):
        interval_hour = 24  # chosen interval
        df_grouped_by_hadmid = df_grouped_by_hadmid.sort_values(by="CHARTTIME")

        hadm_id = df_grouped_by_hadmid.HADM_ID.unique()[0]
        st = df_admissions[df_admissions.HADM_ID == hadm_id].ADMITTIME.iloc[0]
        et = df_grouped_by_hadmid.CHARTTIME.iloc[-1]  # et <- end time

        st = datetime.strptime(f"{st.year}-{st.month}-{st.day} {st.hour // interval_hour * interval_hour:2}:00:00", "%Y-%m-%d %H:%M:%S")
        et = datetime.strptime(f"{et.year}-{et.month}-{et.day} {(((et.hour // interval_hour) + 1) * interval_hour) - 1:2}:59:59", "%Y-%m-%d %H:%M:%S")

        interval = timedelta(hours=interval_hour)

        dfx = df_grouped_by_hadmid.copy()
        dfx = dfx[dfx.CHARTTIME >= st]
        dfx.insert(len(dfx.columns), "TIMESTEP", np.NaN)

        timestep = 0
        while st < et:
            mask = (st <= dfx.CHARTTIME) & (dfx.CHARTTIME <= st + interval)
            if len(dfx.loc[mask]) > 0:
                dfx.loc[mask, 'TIMESTEP'] = timestep

            timestep += 1
            st += interval

        return dfx

    # df_grouped_by_hadmid_timestep_added = grouped_by_hadmid.apply(add_timestep_per_hadmid)
    df_grouped_by_hadmid_timestep_added = pd.DataFrame()
    list_df_grouped_by_hadmid_timestep_added = []
    for k, df in tqdm(grouped_by_hadmid):
        list_df_grouped_by_hadmid_timestep_added.append(add_timestep_per_hadmid(df))
        if len(list_df_grouped_by_hadmid_timestep_added) % 3000 == 0:
            list_df_grouped_by_hadmid_timestep_added.append(df_grouped_by_hadmid_timestep_added)
            df_grouped_by_hadmid_timestep_added = pd.concat(list_df_grouped_by_hadmid_timestep_added)
            list_df_grouped_by_hadmid_timestep_added = []
    list_df_grouped_by_hadmid_timestep_added.append(df_grouped_by_hadmid_timestep_added)
    df_grouped_by_hadmid_timestep_added = pd.concat(list_df_grouped_by_hadmid_timestep_added).reset_index(drop=True)

    df_labevents = df_labevents.merge(df_grouped_by_hadmid_timestep_added[['ROW_ID', 'TIMESTEP']], how='left', on='ROW_ID', copy=False)
    df_labevents = df_labevents[df_labevents.TIMESTEP.notnull()]  # above `merge` step will introduce `nan`

    # **************************************************************************************************************** #
    print("*** Merging repeat edges ***")
    gb_hadmid = df_labevents.groupby("HADM_ID")
    drop_indexs = []
    for hadm_id in tqdm(df_labevents.HADM_ID.unique()):
        df_curr_hadmid = df_labevents[df_labevents.HADM_ID == hadm_id]

        for timestep in range(int(df_curr_hadmid.TIMESTEP.max()) + 1):
            df_curr_hadmid_curr_timestep = df_curr_hadmid[df_curr_hadmid.TIMESTEP == timestep]

            sr_itemid_value_counts = df_curr_hadmid_curr_timestep.ITEMID.value_counts()
            sr_itemid_repeat = sr_itemid_value_counts[sr_itemid_value_counts > 1]

            for itemid_repeat in list(sr_itemid_repeat.index):
                deprecate_entry_rowid = df_curr_hadmid_curr_timestep[df_curr_hadmid_curr_timestep.ITEMID == itemid_repeat]\
                                            .sort_values(by="CHARTTIME")\
                                            .ROW_ID.iloc[0:-1]
                deprecate_entry_index = list(deprecate_entry_rowid.index)
                drop_indexs.extend(deprecate_entry_index)

    df_labevents.drop(drop_indexs, inplace=True)
    df_labevents.to_csv(dst_csv_path)


if __name__ == "__main__":
    path_dataset = r"/root/autodl-tmp/mimic-iv-2.2"
    path_dst_csv = r"/root/autodl-tmp/mimic-iv-clinical-database-2.2"
    preprocess_admission(src_csv_path=path.join(path_dataset, "admissions.csv.gz"),
                         dst_csv_path=path.join(path_dst_csv, "ADMISSIONS_NEW.csv.gz"))

    preprocess_labitems(src_csv_path=path.join(path_dataset, "d_labitems.csv.gz"),
                        dst_csv_path=path.join(path_dst_csv, "D_LABITEMS_NEW.csv.gz"))

    preprocess_labevents(src_csv_path=path.join(path_dataset, "labevents.csv.gz"),
                         dst_csv_path=path.join(path_dst_csv, "LABEVENTS_PREPROCESSED.csv.gz"),
                         src_csv_path_admi=path.join(path_dataset, "admissions.csv.gz"))
