import sys; sys.path.append("..")
import pandas as pd
import os
import shutil
import torch
import utils.constant as constant

from tqdm import tqdm
from torch_geometric.data import HeteroData
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    path_dataset = constant.PATH_MIMIC_III_ETL_OUTPUT
    path_hgs     = constant.PATH_MIMIC_III_HGS_OUTPUT
    path_hgs_train = os.path.join(path_hgs, "train")
    path_hgs_test  = os.path.join(path_hgs, "test")

    df_admissions    = pd.read_csv(os.path.join(path_dataset, "ADMISSIONS_NEW.csv.gz"))
    df_labitems      = pd.read_csv(os.path.join(path_dataset, "D_LABITEMS_NEW.csv.gz"))
    df_labevents     = pd.read_csv(os.path.join(path_dataset, "LABEVENTS_PREPROCESSED.csv.gz"))
    df_prescriptions = pd.read_csv(os.path.join(path_dataset, "PRESCRIPTIONS_PREPROCESSED.csv.gz"))
    df_drug_ndc_feat = pd.read_csv(os.path.join(path_dataset, "DRUGS_NDC_FEAT.csv.gz"))

    length_per_hadm_l = df_labevents.groupby('HADM_ID')[['TIMESTEP']].max()
    length_per_hadm_p = df_prescriptions.groupby('HADM_ID')[['TIMESTEP']].max()

    # filter out patient whose admission length is not more than 1 day
    length_per_hadm_l_multidays = length_per_hadm_l[length_per_hadm_l.TIMESTEP > 1]
    length_per_hadm_p_multidays = length_per_hadm_p[length_per_hadm_p.TIMESTEP > 1]

    adm_l = set(list(length_per_hadm_l_multidays.index))
    adm_p = set(list(length_per_hadm_p_multidays.index))

    both = list(set.intersection(adm_l, adm_p))

    adm_train, adm_test = train_test_split(both, test_size=0.05, random_state=10043)
    print(len(adm_train), len(adm_test))

    g_admi = df_admissions.groupby('HADM_ID')
    g_labe = df_labevents.groupby('HADM_ID')
    g_pres = df_prescriptions.groupby('HADM_ID')

    list_selected_admission_columns = ['ADMISSION_TYPE',
                                       'ADMISSION_LOCATION',
                                       'DISCHARGE_LOCATION',
                                       'INSURANCE',
                                       'LANGUAGE',
                                       'RELIGION',
                                       'MARITAL_STATUS',
                                       'ETHNICITY']
    list_selected_labitems_columns = ['FLUID', 'CATEGORY']
    list_selected_drug_ndc_columns = ["DRUG_TYPE_MAIN_Proportion",
                                      "DRUG_TYPE_BASE_Proportion",
                                      "DRUG_TYPE_ADDITIVE_Proportion",
                                      "FORM_UNIT_DISP_Freq_1",
                                      "FORM_UNIT_DISP_Freq_2",
                                      "FORM_UNIT_DISP_Freq_3",
                                      "FORM_UNIT_DISP_Freq_4",
                                      "FORM_UNIT_DISP_Freq_5"]
    list_selected_prescriptions_columns = ["DRUG_TYPE",
                                           "PROD_STRENGTH",
                                           "DOSE_VAL_RX",
                                           "DOSE_UNIT_RX",
                                           "FORM_VAL_DISP",
                                           "FORM_UNIT_DISP",
                                           "ROUTE"]
    list_selected_labevents_columns = ['CATAGORY', 'VALUENUM_Z-SCORED']

    df_labitems.sort_values(by='ITEMID', inplace=True)
    df_drug_ndc_feat.sort_values(by='NDC', inplace=True)

    unique_item_id = df_labitems.ITEMID.sort_values().unique()
    unique_item_id = pd.DataFrame(data={
        'ITEMID': unique_item_id,
        'mappedID': pd.RangeIndex(len(unique_item_id)),
    })

    unique_ndc_id = df_drug_ndc_feat.NDC.sort_values().unique()
    unique_ndc_id = pd.DataFrame(data={
        'NDC': unique_ndc_id,
        'mappedID': pd.RangeIndex(len(unique_ndc_id)),
    })

    # --- train set ---
    if os.path.exists(path_hgs_train) and os.path.isdir(path_hgs_train):
        shutil.rmtree(path_hgs_train)
        os.mkdir(path_hgs_train)

    for idx, id in tqdm(enumerate(adm_train), total=len(adm_train)):
        curr_id_df_admi = g_admi.get_group(id)
        curr_id_df_labe = g_labe.get_group(id)
        curr_id_df_pres = g_pres.get_group(id)

        # add ROW_ID for seq order info
        curr_id_df_labe = curr_id_df_labe.sort_values(by=["TIMESTEP", "CHARTTIME", "ROW_ID"])
        curr_id_df_pres = curr_id_df_pres.sort_values(by=["TIMESTEP", "STARTDATE", "ENDDATE", "ROW_ID"])

        # --- get corr tensor shards ---
        # nodes
        nodes_feature_admission_curr = torch.from_numpy(curr_id_df_admi[list_selected_admission_columns].values)
        nodes_feature_labitems = torch.from_numpy(df_labitems[list_selected_labitems_columns].values)
        nodes_feature_drug_ndc = torch.from_numpy(df_drug_ndc_feat[list_selected_drug_ndc_columns].values)

        # edges
        # Edge indexes
        unique_hadm_id = curr_id_df_admi.HADM_ID.sort_values().unique()
        unique_hadm_id = pd.DataFrame(data={
            'HADM_ID': unique_hadm_id,
            'mappedID': pd.RangeIndex(len(unique_hadm_id)),
        })

        # Perform merge to obtain the edges from HADM_ID and ITEMID:
        ratings_hadm_id = pd.merge(
            curr_id_df_labe['HADM_ID'], unique_hadm_id, left_on='HADM_ID', right_on='HADM_ID', how='left')
        ratings_item_id = pd.merge(
            curr_id_df_labe['ITEMID'], unique_item_id, left_on='ITEMID', right_on='ITEMID', how='left')

        # FOR `df_prescriptions_curr`
        ratings_hadm_id_drug = pd.merge(
            curr_id_df_pres['HADM_ID'], unique_hadm_id, left_on='HADM_ID', right_on='HADM_ID', how='left')
        ratings_ndc_id = pd.merge(
            curr_id_df_pres['NDC'], unique_ndc_id, left_on='NDC', right_on='NDC', how='left')

        ratings_hadm_id = torch.from_numpy(ratings_hadm_id['mappedID'].values)
        ratings_item_id = torch.from_numpy(ratings_item_id['mappedID'].values)
        ratings_hadm_id_drug = torch.from_numpy(ratings_hadm_id_drug['mappedID'].values)
        ratings_ndc_id = torch.from_numpy(ratings_ndc_id['mappedID'].values)

        edge_index_hadm_to_item = torch.stack([ratings_hadm_id, ratings_item_id], dim=0)
        edge_index_hadm_to_ndc = torch.stack([ratings_hadm_id_drug, ratings_ndc_id], dim=0)

        # Edge features
        # FOR `df_labevents_curr`
        edges_feature_labevents = torch.from_numpy(curr_id_df_labe[list_selected_labevents_columns].values)
        # FOR `df_prescriptions_curr`
        edges_feature_prescriptions = torch.from_numpy(curr_id_df_pres[list_selected_prescriptions_columns].values)

        # Timesteps:
        edges_timestep = torch.from_numpy(curr_id_df_labe['TIMESTEP'].values)
        edges_timestep_prescriptions = torch.from_numpy(curr_id_df_pres['TIMESTEP'].values)

        ############################################## assemble #####################################################
        data = HeteroData()

        # Node
        # node indices
        data["admission"].node_id = torch.arange(len(unique_hadm_id))
        data["labitem"].node_id = torch.arange(len(unique_item_id))
        data["drug"].node_id = torch.arange(len(unique_ndc_id))
        # node features:
        data["admission"].x = nodes_feature_admission_curr
        data["labitem"].x = nodes_feature_labitems
        data["drug"].x = nodes_feature_drug_ndc

        # edge:
        data["admission", "did", "labitem"].edge_index = edge_index_hadm_to_item
        data["admission", "did", "labitem"].x = edges_feature_labevents
        data["admission", "did", "labitem"].timestep = edges_timestep

        data["admission", "took", "drug"].edge_index = edge_index_hadm_to_ndc
        data["admission", "took", "drug"].x = edges_feature_prescriptions
        data["admission", "took", "drug"].timestep = edges_timestep_prescriptions

        torch.save(data, f'{os.path.join(path_hgs_train, str(idx))}.pt')

    # --- test ---
    if os.path.exists(path_hgs_test) and os.path.isdir(path_hgs_test):
        shutil.rmtree(path_hgs_test)
        os.mkdir(path_hgs_test)

    for idx, id in tqdm(enumerate(adm_test), total=len(adm_test)):
        curr_id_df_admi = g_admi.get_group(id)
        curr_id_df_labe = g_labe.get_group(id)
        curr_id_df_pres = g_pres.get_group(id)

        # add ROW_ID for seq order info
        curr_id_df_labe = curr_id_df_labe.sort_values(by=["TIMESTEP", "CHARTTIME", "ROW_ID"])
        curr_id_df_pres = curr_id_df_pres.sort_values(by=["TIMESTEP", "STARTDATE", "ENDDATE", "ROW_ID"])

        # --- get corr tensor shards ---
        # nodes
        nodes_feature_admission_curr = torch.from_numpy(curr_id_df_admi[list_selected_admission_columns].values)
        nodes_feature_labitems = torch.from_numpy(df_labitems[list_selected_labitems_columns].values)
        nodes_feature_drug_ndc = torch.from_numpy(df_drug_ndc_feat[list_selected_drug_ndc_columns].values)

        # edges
        # Edge indexes
        unique_hadm_id = curr_id_df_admi.HADM_ID.sort_values().unique()
        unique_hadm_id = pd.DataFrame(data={
            'HADM_ID': unique_hadm_id,
            'mappedID': pd.RangeIndex(len(unique_hadm_id)),
        })

        # Perform merge to obtain the edges from HADM_ID and ITEMID:
        ratings_hadm_id = pd.merge(
            curr_id_df_labe['HADM_ID'], unique_hadm_id, left_on='HADM_ID', right_on='HADM_ID', how='left')
        ratings_item_id = pd.merge(
            curr_id_df_labe['ITEMID'], unique_item_id, left_on='ITEMID', right_on='ITEMID', how='left')

        # FOR `df_prescriptions_curr`
        ratings_hadm_id_drug = pd.merge(
            curr_id_df_pres['HADM_ID'], unique_hadm_id, left_on='HADM_ID', right_on='HADM_ID', how='left')
        ratings_ndc_id = pd.merge(
            curr_id_df_pres['NDC'], unique_ndc_id, left_on='NDC', right_on='NDC', how='left')

        ratings_hadm_id = torch.from_numpy(ratings_hadm_id['mappedID'].values)
        ratings_item_id = torch.from_numpy(ratings_item_id['mappedID'].values)
        ratings_hadm_id_drug = torch.from_numpy(ratings_hadm_id_drug['mappedID'].values)
        ratings_ndc_id = torch.from_numpy(ratings_ndc_id['mappedID'].values)

        edge_index_hadm_to_item = torch.stack([ratings_hadm_id, ratings_item_id], dim=0)
        edge_index_hadm_to_ndc = torch.stack([ratings_hadm_id_drug, ratings_ndc_id], dim=0)

        # Edge features
        # FOR `df_labevents_curr`
        edges_feature_labevents = torch.from_numpy(curr_id_df_labe[list_selected_labevents_columns].values)
        # FOR `df_prescriptions_curr`
        edges_feature_prescriptions = torch.from_numpy(curr_id_df_pres[list_selected_prescriptions_columns].values)

        # Timesteps:
        edges_timestep = torch.from_numpy(curr_id_df_labe['TIMESTEP'].values)
        edges_timestep_prescriptions = torch.from_numpy(curr_id_df_pres['TIMESTEP'].values)

        ############################################## assemble #####################################################
        data = HeteroData()

        # Node
        # node indices
        data["admission"].node_id = torch.arange(len(unique_hadm_id))
        data["labitem"].node_id = torch.arange(len(unique_item_id))
        data["drug"].node_id = torch.arange(len(unique_ndc_id))
        # node features:
        data["admission"].x = nodes_feature_admission_curr
        data["labitem"].x = nodes_feature_labitems
        data["drug"].x = nodes_feature_drug_ndc

        # edge:
        data["admission", "did", "labitem"].edge_index = edge_index_hadm_to_item
        data["admission", "did", "labitem"].x = edges_feature_labevents
        data["admission", "did", "labitem"].timestep = edges_timestep

        data["admission", "took", "drug"].edge_index = edge_index_hadm_to_ndc
        data["admission", "took", "drug"].x = edges_feature_prescriptions
        data["admission", "took", "drug"].timestep = edges_timestep_prescriptions

        torch.save(data, f'{os.path.join(path_hgs_test, str(idx))}.pt')
