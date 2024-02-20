import sys; sys.path.append('..')
import pandas as pd
import os, shutil
import torch
import numpy as np

from tqdm import tqdm
from torch_geometric.data import HeteroData


def get_list_total_hadmid(*list_df_single_edges_type: list):
    r"""
    Get the interset of hadmid from df(s) which record(s) the edge connection.
    """
    if len(list_df_single_edges_type) <= 1:
        return list(list_df_single_edges_type[0].HADM_ID.sort_values().unique())
    else:
        list_set_hadmid_single_edges_type = [set(list(df_single_edges_type.HADM_ID.sort_values().unique()))
                                             for df_single_edges_type in list_df_single_edges_type]
        return list(set.intersection(*list_set_hadmid_single_edges_type))


def get_train_test_hadmid_list(list_total_hadmid, split_ratio: float, shuffle: bool=False):
    np.random.shuffle(list_total_hadmid) if shuffle else None

    length = len(list_total_hadmid)
    list_train_hadmid = list_total_hadmid[0:int(length * split_ratio)]
    list_val_hadmid = list_total_hadmid[int(length * split_ratio):]

    return list(list_train_hadmid), list(list_val_hadmid)


def batches_spliter(list_hadmid: list, batch_size: int, *dfs):
    r"""
    Split the df(s) into many batches by the `HADM_ID` batch.

    Note: the order of returned list_dfs is consistent with the order of dfs passed in.
    """
    idx = 0
    length = len(list_hadmid)
    batches_hadmids = []
    while (idx + batch_size) <= length:
        batches_hadmids += [list_hadmid[idx:idx + batch_size]]
        idx += batch_size
    if idx < length:
        batches_hadmids += [list_hadmid[idx:]]

    list_list_dfs = [[] for _ in range(len(dfs))]
    for batch_hadmids in tqdm(batches_hadmids):
        for i, df in enumerate(dfs):
            list_list_dfs[i].append(
                df[df.HADM_ID.isin(batch_hadmids)].copy()
            )

    return list_list_dfs


def construct_dynamic_hetero_graph(df_admissions_curr,
                                   df_labitems_curr,
                                   df_labevents_curr,
                                   df_drug_ndc_feat_curr,
                                   df_prescriptions_curr):
    ############################################### Nodes #######################################################
    ## admission
    df_admissions_curr.sort_values(by='HADM_ID', inplace=True)
    list_selected_admission_columns = ['ADMISSION_TYPE',
                                       'ADMISSION_LOCATION',
                                       'DISCHARGE_LOCATION',
                                       'INSURANCE',
                                       'LANGUAGE',
                                       'RELIGION',
                                       'MARITAL_STATUS',
                                       'ETHNICITY']
    nodes_feature_admission_curr = torch.from_numpy(df_admissions_curr[list_selected_admission_columns].values)

    ## labitems
    df_labitems_curr.sort_values(by='ITEMID', inplace=True)
    list_selected_labitems_columns = ['FLUID', 'CATEGORY']
    nodes_feature_labitems = torch.from_numpy(df_labitems_curr[list_selected_labitems_columns].values)

    ## drug_ndc
    df_drug_ndc_feat_curr.sort_values(by='NDC', inplace=True)
    list_selected_drug_ndc_columns = ["DRUG_TYPE_MAIN_Proportion",
                                      "DRUG_TYPE_BASE_Proportion",
                                      "DRUG_TYPE_ADDITIVE_Proportion",
                                      "FORM_UNIT_DISP_Freq_1",
                                      "FORM_UNIT_DISP_Freq_2",
                                      "FORM_UNIT_DISP_Freq_3",
                                      "FORM_UNIT_DISP_Freq_4",
                                      "FORM_UNIT_DISP_Freq_5"]
    nodes_feature_drug_ndc = torch.from_numpy(df_drug_ndc_feat_curr[list_selected_drug_ndc_columns].values)

    ############################################### Edges #######################################################
    df_labevents_curr.sort_values(by=["HADM_ID", "ITEMID"], inplace=True)
    df_prescriptions_curr.sort_values(by=["HADM_ID", "NDC"], inplace=True)

    ## Edge indexes
    ### Create a mapping from unique hadm_id indices to range [0, num_hadm_nodes):
    unique_hadm_id = df_admissions_curr.HADM_ID.sort_values().unique()
    unique_hadm_id = pd.DataFrame(data={
        'HADM_ID': unique_hadm_id,
        'mappedID': pd.RangeIndex(len(unique_hadm_id)),
    })
    ### Create a mapping from unique ITEMID indices to range [0, num_labitem_nodes):
    unique_item_id = df_labitems_curr.ITEMID.sort_values().unique()
    unique_item_id = pd.DataFrame(data={
        'ITEMID': unique_item_id,
        'mappedID': pd.RangeIndex(len(unique_item_id)),
    })
    ### Create a mapping from unique NDC indices to range [0, num_hadm_nodes):
    unique_ndc_id = df_drug_ndc_feat_curr.NDC.sort_values().unique()
    unique_ndc_id = pd.DataFrame(data={
        'NDC': unique_ndc_id,
        'mappedID': pd.RangeIndex(len(unique_ndc_id)),
    })

    ### Perform merge to obtain the edges from HADM_ID and ITEMID:

    #### FOR `df_labevents_curr`
    ratings_hadm_id = pd.merge(df_labevents_curr['HADM_ID'], unique_hadm_id, left_on='HADM_ID', right_on='HADM_ID', how='left')
    ratings_item_id = pd.merge(df_labevents_curr['ITEMID'],  unique_item_id, left_on='ITEMID',  right_on='ITEMID',  how='left')
    ratings_hadm_id = torch.from_numpy(ratings_hadm_id['mappedID'].values)
    ratings_item_id = torch.from_numpy(ratings_item_id['mappedID'].values)

    #### FOR `df_prescriptions_curr`
    ratings_hadm_id_drug = pd.merge(df_prescriptions_curr['HADM_ID'], unique_hadm_id, left_on='HADM_ID', right_on='HADM_ID', how='left')
    ratings_ndc_id       = pd.merge(df_prescriptions_curr['NDC'],     unique_ndc_id,  left_on='NDC',     right_on='NDC',     how='left')
    ratings_hadm_id_drug = torch.from_numpy(ratings_hadm_id_drug['mappedID'].values)
    ratings_ndc_id       = torch.from_numpy(ratings_ndc_id['mappedID'].values)

    edge_index_hadm_to_item = torch.stack([ratings_hadm_id, ratings_item_id], dim=0)
    edge_index_hadm_to_ndc  = torch.stack([ratings_hadm_id_drug, ratings_ndc_id], dim=0)

    ## Edge features

    ### FOR `df_labevents_curr`
    list_selected_labevents_columns = ['CATAGORY', 'VALUENUM_Z-SCORED']
    edges_feature_labevents = torch.from_numpy(df_labevents_curr[list_selected_labevents_columns].values)

    ### FOR `df_prescriptions_curr`
    list_selected_prescriptions_columns = ["DRUG_TYPE",
                                           "PROD_STRENGTH",
                                           "DOSE_VAL_RX",
                                           "DOSE_UNIT_RX",
                                           "FORM_VAL_DISP",
                                           "FORM_UNIT_DISP",
                                           "ROUTE"]
    edges_feature_prescriptions = torch.from_numpy(df_prescriptions_curr[list_selected_prescriptions_columns].values)

    ## Timesteps:
    edges_timestep = torch.from_numpy(df_labevents_curr['TIMESTEP'].values)
    edges_timestep_prescriptions = torch.from_numpy(df_prescriptions_curr['TIMESTEP'].values)

    ############################################## assemble #####################################################
    data = HeteroData()

    ## Node
    ### node indices
    data["admission"].node_id = torch.arange(len(unique_hadm_id))
    data["labitem"].node_id   = torch.arange(len(unique_item_id))
    data["drug"].node_id      = torch.arange(len(unique_ndc_id))
    ### node features:
    data["admission"].x = nodes_feature_admission_curr
    data["labitem"].x   = nodes_feature_labitems
    data["drug"].x      = nodes_feature_drug_ndc

    ## edge:
    data["admission", "did", "labitem"].edge_index = edge_index_hadm_to_item
    data["admission", "did", "labitem"].x          = edges_feature_labevents
    data["admission", "did", "labitem"].timestep   = edges_timestep

    data["admission", "took", "drug"].edge_index = edge_index_hadm_to_ndc
    data["admission", "took", "drug"].x          = edges_feature_prescriptions
    data["admission", "took", "drug"].timestep   = edges_timestep_prescriptions

    ############################################# debug NaN #####################################################
    assert not data["admission"].node_id.isnan().any()
    assert not data["admission"].x.isnan().any()

    assert not data["labitem"].node_id.isnan().any()
    assert not data["labitem"].x.isnan().any()

    assert not data["drug"].node_id.isnan().any()
    assert not data["drug"].x.isnan().any()

    assert     data["admission", "did", "labitem"].edge_index.shape[-1] > 0
    assert not data["admission", "did", "labitem"].edge_index.isnan().any()
    assert not data["admission", "did", "labitem"].x.isnan().any()

    assert     data["admission", "took", "drug"].edge_index.shape[-1] > 0
    assert not data["admission", "took", "drug"].edge_index.isnan().any()
    assert not data["admission", "took", "drug"].x.isnan().any()

    return data


if __name__ == "__main__":
    path_dataset = r"/data/data2/041/datasets/mimic-iii-clinical-database-1.4"

    df_admissions    = pd.read_csv(os.path.join(path_dataset, "ADMISSIONS_NEW.csv.gz"))
    df_labitems      = pd.read_csv(os.path.join(path_dataset, "D_LABITEMS_NEW.csv.gz"))
    df_labevents     = pd.read_csv(os.path.join(path_dataset, "LABEVENTS_PREPROCESSED.csv.gz"))
    df_prescriptions = pd.read_csv(os.path.join(path_dataset, "PRESCRIPTIONS_PREPROCESSED.csv.gz"))
    df_drug_ndc_feat = pd.read_csv(os.path.join(path_dataset, "DRUGS_NDC_FEAT.csv.gz"))

    list_total_hadmid = get_list_total_hadmid(df_labevents, df_prescriptions)
    list_train_hadmid, list_test_hadmid = get_train_test_hadmid_list(list_total_hadmid, 0.95)

    batch_size = 128
    list_df_admissions_single_batch_train, list_df_labevents_single_batch_train, list_df_prescriptions_single_batch_train = batches_spliter(list_train_hadmid, batch_size, df_admissions, df_labevents, df_prescriptions)
    list_df_admissions_single_batch_test,  list_df_labevents_single_batch_test,  list_df_prescriptions_single_batch_test  = batches_spliter(list_test_hadmid,  batch_size, df_admissions, df_labevents, df_prescriptions)

    train_hgs = [construct_dynamic_hetero_graph(df_admissions_single_batch,
                                                df_labitems,
                                                df_labevents_single_batch,
                                                df_drug_ndc_feat,
                                                df_prescriptions_single_batch) \
                 for df_admissions_single_batch, df_labevents_single_batch, df_prescriptions_single_batch in tqdm(
            zip(list_df_admissions_single_batch_train,
                list_df_labevents_single_batch_train,
                list_df_prescriptions_single_batch_train)
        )]

    val_hgs = [construct_dynamic_hetero_graph(df_admissions_single_batch,
                                              df_labitems,
                                              df_labevents_single_batch,
                                              df_drug_ndc_feat,
                                              df_prescriptions_single_batch) \
               for df_admissions_single_batch, df_labevents_single_batch, df_prescriptions_single_batch in tqdm(
            zip(list_df_admissions_single_batch_test,
                list_df_labevents_single_batch_test,
                list_df_prescriptions_single_batch_test)
        )]

    path_hgs = r"/data/data2/041/datasets/mimic-iii-hgs"
    # path_hgs = r"/data/data2/041/datasets/mimic-iii-hgs-new"
    path_hgs_curr = os.path.join(path_hgs, f'batch_size_{batch_size}')

    if os.path.isdir(path_hgs_curr):
        shutil.rmtree(path_hgs_curr)

    os.mkdir(path_hgs_curr)
    os.mkdir(os.path.join(path_hgs_curr, "train"))
    for idx, train_hg in enumerate(train_hgs):
        torch.save(train_hg, f'{os.path.join(os.path.join(path_hgs_curr, "train"), str(idx))}.pt')

    os.mkdir(os.path.join(path_hgs_curr, "test"))
    for idx, train_hg in enumerate(val_hgs):
        torch.save(train_hg, f'{os.path.join(os.path.join(path_hgs_curr, "test"), str(idx))}.pt')