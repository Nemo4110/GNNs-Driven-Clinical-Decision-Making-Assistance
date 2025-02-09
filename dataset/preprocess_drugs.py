import pandas as pd
import os
import numpy as np
import sys; sys.path.append("..")
import utils.constant as constant

from tqdm import tqdm
from datetime import datetime, timedelta


path_ddi_dataset = r"../dataset/ddi/"
path_dataset = r"../data/mimic-iv-2.2/"
path_etl_output = r"../data/mimic-iv-clinical-database-2.2/"

df_admissions = pd.read_csv(os.path.join(path_dataset, "admissions.csv.gz"))
df_admissions.columns = df_admissions.columns.str.upper()
df_admissions["ADMITTIME"] = pd.to_datetime(df_admissions["ADMITTIME"], format="%Y-%m-%d %H:%M:%S")
df_admissions["DISCHTIME"] = pd.to_datetime(df_admissions["DISCHTIME"], format="%Y-%m-%d %H:%M:%S")


def ndc0_mapping_handler(df_prescriptions):
    _df_ndc_not_equal_0 = df_prescriptions[df_prescriptions.NDC != 0].copy()
    _df_ndc_equal_0 = df_prescriptions[df_prescriptions.NDC == 0].copy()
    remap_4_ndc_equal_0 = _df_ndc_equal_0.DRUG.value_counts()
    remap_4_ndc_equal_0 = pd.Series(index=remap_4_ndc_equal_0.index, data=range(1, len(remap_4_ndc_equal_0)+1))
    _df_ndc_equal_0['NDC'] = _df_ndc_equal_0['DRUG'].map(remap_4_ndc_equal_0)

    return pd.concat([_df_ndc_equal_0, _df_ndc_not_equal_0])


def duration_issuse_handler(df_prescriptions):
    interval_1day = timedelta(hours=24)

    mask = (df_prescriptions.STOPTIME - df_prescriptions.STARTTIME) > interval_1day
    df_duration_longer_than_1 = df_prescriptions.loc[mask].copy()
    df_duration_equal_to_1day = df_prescriptions.loc[~mask].copy()

    df_duration_longer_than_1 = df_duration_longer_than_1.reindex(
        df_duration_longer_than_1.index.repeat(
            (df_duration_longer_than_1.STOPTIME - df_duration_longer_than_1.STARTTIME) / interval_1day
        )
    ).reset_index(drop=True)

    gb_duration_longer_than_1 = df_duration_longer_than_1.groupby("ROW_ID")

    def modify_timestamp_4_duration_longer_than_1(df):
        num_duration = len(df)
        dfx = df.copy()
        interval_1day = timedelta(hours=24)

        curr_idx = 0
        curr_st = dfx.iloc[curr_idx].STARTTIME
        curr_et = curr_st + interval_1day

        while curr_idx < num_duration:
            dfx["STARTTIME"].iloc[curr_idx] = curr_st
            dfx["STOPTIME"].iloc[curr_idx] = curr_et

            curr_idx += 1
            curr_st += interval_1day
            curr_et += interval_1day

        return dfx

    df_prescriptions_duration_solved = pd.DataFrame()
    list_df_prescriptions_duration_solved = []
    for key, df in tqdm(gb_duration_longer_than_1):
        list_df_prescriptions_duration_solved.append(modify_timestamp_4_duration_longer_than_1(df))
        if len(list_df_prescriptions_duration_solved) % 10000 == 0:
            list_df_prescriptions_duration_solved.append(df_prescriptions_duration_solved)  # previous results
            df_prescriptions_duration_solved = pd.concat(list_df_prescriptions_duration_solved)
            list_df_prescriptions_duration_solved = []  # clear

    list_df_prescriptions_duration_solved.append(df_prescriptions_duration_solved)
    df_prescriptions_duration_solved = pd.concat(list_df_prescriptions_duration_solved)
    df_prescriptions_duration_solved = pd.concat([df_prescriptions_duration_solved, df_duration_equal_to_1day]).reset_index(drop=True)

    return df_prescriptions_duration_solved


def ncv_converting_handler(df_prescriptions):
    ncv_cols_global = ["DRUG_TYPE", "DOSE_UNIT_RX", "FORM_UNIT_DISP", "ROUTE"]

    for c in ncv_cols_global:
        m = df_prescriptions[c].value_counts()
        m = pd.Series(index=m.index, data=range(1, len(m) + 1))
        df_prescriptions[c] = df_prescriptions[c].map(m)

    values_fillna = {}
    for c in ncv_cols_global:
        values_fillna[c] = 0  # use 0 to fill na
    df_prescriptions.fillna(value=values_fillna, inplace=True)

    # below cols are special with each NDC
    ncv_cols = ["PROD_STRENGTH", "DOSE_VAL_RX", "FORM_VAL_DISP"]

    def convert_ncv(df_curr_ndc):
        dfx = df_curr_ndc.copy()

        for col in ncv_cols:
            m = df_curr_ndc[col].value_counts()
            m = pd.Series(index=m.index, data=range(1, len(m) + 1))
            dfx[col] = dfx[col].map(m)

        return dfx

    gb_ndc = df_prescriptions.groupby("NDC")

    list_dfx = []
    dfxs_ncv_converted = pd.DataFrame()
    for k, df in tqdm(gb_ndc):
        list_dfx.append(convert_ncv(df))
        if len(list_dfx) % 100 == 0:
            list_dfx.append(dfxs_ncv_converted)
            dfxs_ncv_converted = pd.concat(list_dfx)
            list_dfx = []
    list_dfx.append(dfxs_ncv_converted)
    dfxs_ncv_converted = pd.concat(list_dfx)

    values_fillna = {}
    for c in ncv_cols:
        values_fillna[c] = 0  # use 0 to fill na
    dfxs_ncv_converted.fillna(value=values_fillna, inplace=True)

    return dfxs_ncv_converted


def adding_timestep_handler(df_prescriptions):
    grouped_by_hadmid = df_prescriptions.groupby("HADM_ID")

    def add_timestep_per_hadmid(df_grouped_by_hadmid: pd.DataFrame):
        interval_hour = 24  # chosen interval
        df_grouped_by_hadmid = df_grouped_by_hadmid.sort_values(by=["STARTTIME", "STOPTIME"])

        hadm_id = df_grouped_by_hadmid.HADM_ID.unique()[0]
        st = df_admissions[df_admissions.HADM_ID == hadm_id].ADMITTIME.iloc[0]
        et = df_grouped_by_hadmid[df_grouped_by_hadmid.HADM_ID == hadm_id].STOPTIME.iloc[-1]
        # The end time of the prescription may be after the discharge time

        st = datetime.strptime(f"{st.year}-{st.month}-{st.day} {st.hour // interval_hour * interval_hour:2}:00:00", "%Y-%m-%d %H:%M:%S")
        et = datetime.strptime(f"{et.year}-{et.month}-{et.day} {(((et.hour // interval_hour) + 1) * interval_hour) - 1:2}:59:59", "%Y-%m-%d %H:%M:%S")

        interval = timedelta(hours=interval_hour)

        dfx = df_grouped_by_hadmid.copy()

        # filter out records whose `STARTTIME` are earlier than `ADMITTIME` which will cause `nan` issues
        dfx = dfx[dfx.STARTTIME >= st]

        dfx.insert(len(dfx.columns), "TIMESTEP", np.NaN)

        timestep = 0
        while st < et:
            mask = (st <= dfx.STARTTIME) & (dfx.STARTTIME <= st + interval)
            if len(dfx.loc[mask]) > 0:
                dfx.loc[mask, 'TIMESTEP'] = timestep

            timestep += 1
            st += interval

        return dfx

    df_prescriptions_preprocessed = pd.DataFrame()  # blank DataFrame
    list_df_prescriptions_preprocessed = []
    for k, df in tqdm(grouped_by_hadmid):
        list_df_prescriptions_preprocessed.append(add_timestep_per_hadmid(df))
        if len(list_df_prescriptions_preprocessed) % 1000 == 0:
            list_df_prescriptions_preprocessed.append(df_prescriptions_preprocessed)
            df_prescriptions_preprocessed = pd.concat(list_df_prescriptions_preprocessed)
            list_df_prescriptions_preprocessed = []

    list_df_prescriptions_preprocessed.append(df_prescriptions_preprocessed)
    df_prescriptions_preprocessed = pd.concat(list_df_prescriptions_preprocessed)

    return df_prescriptions_preprocessed


def repeating_edges_handler(df_prescriptions):
    gb_hadmid = df_prescriptions.groupby("HADM_ID")

    drop_indexs = []
    for k, df_curr_hadmid in tqdm(gb_hadmid):
        for timestep in range(int(df_curr_hadmid.TIMESTEP.max()) + 1):
            df_curr_hadmid_curr_timestep = df_curr_hadmid[df_curr_hadmid.TIMESTEP == timestep]

            sr_ndc_value_counts = df_curr_hadmid_curr_timestep.NDC.value_counts()
            sr_ndc_repeat = sr_ndc_value_counts[sr_ndc_value_counts > 1]

            for ndc_repeat in list(sr_ndc_repeat.index):
                deprecate_entry_index = list(
                    df_curr_hadmid_curr_timestep[df_curr_hadmid_curr_timestep.NDC == ndc_repeat] \
                    .sort_values(by=["STARTTIME", "STOPTIME"]) \
                    .index[0:-1])
                drop_indexs.extend(deprecate_entry_index)
        if len(drop_indexs) > 10000:
            df_prescriptions.drop(drop_indexs, inplace=True)
            drop_indexs = []
    df_prescriptions.drop(drop_indexs, inplace=True)

    # post-checkout
    def repeatability_check(df_curr_hadmid: pd.DataFrame):
        for timestep in range(int(df_curr_hadmid.TIMESTEP.max()) + 1):
            df_curr_hadmid_curr_timestep = df_curr_hadmid[df_curr_hadmid.TIMESTEP == timestep]
            sr_itemid_value_counts = df_curr_hadmid_curr_timestep.NDC.value_counts()
            sr_itemid_repeat = sr_itemid_value_counts[sr_itemid_value_counts > 1]
            assert len(sr_itemid_repeat) == 0

    _gb_hadmid = df_prescriptions.groupby("HADM_ID")
    for k, df in tqdm(_gb_hadmid):
        repeatability_check(df)

    return df_prescriptions


def drugs_node_features_handler(df_prescriptions_preprocessed: pd.DataFrame):
    gb_ndc_all = df_prescriptions_preprocessed.groupby("NDC")

    df_drug_node_feature_per_ndc = pd.DataFrame(columns=["NDC",
                                                         "DRUG_TYPE_MAIN_Proportion",
                                                         "DRUG_TYPE_BASE_Proportion",
                                                         "DRUG_TYPE_ADDITIVE_Proportion",
                                                         "FORM_UNIT_DISP_Freq_1",
                                                         "FORM_UNIT_DISP_Freq_2",
                                                         "FORM_UNIT_DISP_Freq_3",
                                                         "FORM_UNIT_DISP_Freq_4",
                                                         "FORM_UNIT_DISP_Freq_5"])

    for k, df in tqdm(gb_ndc_all):
        ser_drug_type = df.DRUG_TYPE.value_counts()
        sum_drug_type = ser_drug_type.sum()
        len_drug_type = len(ser_drug_type)

        ser_form_unit = df.FORM_UNIT_DISP.value_counts()
        len_form_unit = len(ser_form_unit)

        df_drug_node_feature_per_ndc.loc[len(df_drug_node_feature_per_ndc.index)] = [
            k,

            ser_drug_type.iloc[0] / sum_drug_type,
            ser_drug_type.iloc[1] / sum_drug_type if len_drug_type == 2 else 0,
            ser_drug_type.iloc[2] / sum_drug_type if len_drug_type == 3 else 0,

            ser_form_unit.index[0],
            ser_form_unit.index[1] if len_form_unit == 2 else 0,
            ser_form_unit.index[2] if len_form_unit == 3 else 0,
            ser_form_unit.index[3] if len_form_unit == 4 else 0,
            ser_form_unit.index[4] if len_form_unit == 5 else 0,
        ]

    with open(os.path.join(path_ddi_dataset, "ndc2rxnorm_mapping.txt"), 'r') as f:
        ndc2rxnorm = eval(f.read())

    df_drug_node_feature_per_ndc.insert(len(df_drug_node_feature_per_ndc.columns), "rxnorm_id", np.NaN)

    df_drug_node_feature_per_ndc_real = df_drug_node_feature_per_ndc[df_drug_node_feature_per_ndc.NDC > 1000]
    df_drug_node_feature_per_ndc_fake = df_drug_node_feature_per_ndc[df_drug_node_feature_per_ndc.NDC < 1000]
    df_drug_node_feature_per_ndc_real["NDC"] = df_drug_node_feature_per_ndc_real["NDC"].map("{:0>11.0f}".format)

    df_drug_node_feature_per_ndc_final = pd.concat([df_drug_node_feature_per_ndc_real,
                                                    df_drug_node_feature_per_ndc_fake])
    df_drug_node_feature_per_ndc_final["rxnorm_id"] = df_drug_node_feature_per_ndc_final["NDC"].map(ndc2rxnorm, na_action='ignore')
    mask = df_drug_node_feature_per_ndc_final.rxnorm_id == ''
    df_drug_node_feature_per_ndc_final.loc[mask, "rxnorm_id"] = np.NaN

    return df_drug_node_feature_per_ndc_final


if __name__ == "__main__":
    df_prescriptions = pd.read_csv(os.path.join(path_dataset, "prescriptions.csv.gz"))
    df_prescriptions.columns = df_prescriptions.columns.str.upper()
    df_prescriptions['ROW_ID'] = df_prescriptions.index  # mimic-iv 2.2 没有row_id列
    df_prescriptions["STARTTIME"] = pd.to_datetime(df_prescriptions["STARTTIME"], format="%Y-%m-%d %H:%M:%S")
    df_prescriptions["STOPTIME"]   = pd.to_datetime(df_prescriptions["STOPTIME"],   format="%Y-%m-%d %H:%M:%S")

    # Handling of null values and outliers
    df_prescriptions = df_prescriptions.loc[df_prescriptions.STARTTIME.notnull() &
                                            df_prescriptions.STOPTIME.notnull()]
    df_prescriptions = df_prescriptions[df_prescriptions.NDC.notnull()]
    df_prescriptions = df_prescriptions.loc[df_prescriptions.STARTTIME < df_prescriptions.STOPTIME]

    # string columns
    df_prescriptions['DRUG']           = df_prescriptions['DRUG'          ].astype("string").str.lower()
    df_prescriptions["DRUG_TYPE"]      = df_prescriptions["DRUG_TYPE"     ].astype("string").str.upper()
    df_prescriptions["PROD_STRENGTH"]  = df_prescriptions["PROD_STRENGTH" ].astype("string").str.lower()
    df_prescriptions["DOSE_UNIT_RX"]   = df_prescriptions["DOSE_UNIT_RX"  ].astype("string").str.lower()
    df_prescriptions["FORM_UNIT_DISP"] = df_prescriptions["FORM_UNIT_DISP"].astype("string").str.lower()
    df_prescriptions["ROUTE"]          = df_prescriptions["ROUTE"         ].astype("string").str.upper()

    df_prescriptions = ndc0_mapping_handler(df_prescriptions)
    df_prescriptions = duration_issuse_handler(df_prescriptions)#; df_prescriptions.to_csv(os.path.join(path_etl_output, "PRESCRIPTIONS_DURATION_SOLVED.csv.gz"))
    df_prescriptions = ncv_converting_handler(df_prescriptions) #; df_prescriptions.to_csv(os.path.join(path_etl_output, "PRESCRIPTIONS_NCV_SOLVED.csv.gz"))
    df_prescriptions = adding_timestep_handler(df_prescriptions)#; df_prescriptions.to_csv(os.path.join(path_etl_output, "PRESCRIPTIONS_TIMESTEP_SOLVED.csv.gz"))
    df_prescriptions = repeating_edges_handler(df_prescriptions)

    df_prescriptions.to_csv(os.path.join(path_etl_output, "PRESCRIPTIONS_PREPROCESSED.csv.gz"))

    df_drug_node_feature_per_ndc_final = drugs_node_features_handler(pd.read_csv(os.path.join(path_etl_output, "PRESCRIPTIONS_PREPROCESSED.csv.gz")))
    df_drug_node_feature_per_ndc_final.to_csv(os.path.join(path_etl_output, "DRUGS_NDC_FEAT.csv.gz"))

