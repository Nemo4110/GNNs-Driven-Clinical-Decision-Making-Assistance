# Installation of environment

## Prerequisites :warning:

- A Nvidia `GPU` with enough memory.
  - In our case, we use 3090 that has about 24 GB memory, and the CUDA version is 11.3.
  - In fact, the memory does not need to be so large.
- `conda`
- Get the dataset from the [official website](https://mimic.mit.edu/) of mimic-iii dataset:
  - the completion of an ethics course is required.
  - then you need to sign the *PhysioNet Credentialed Health
Data Use Agreement 1.5.0 - Data Use Agreement for the
MIMIC-III Clinical Database (v1.4)* to get the accessment of dataset. 

## Steps

In the CMD:

1. Create conda environment:

    ```shell
    conda create --name <env_name> python==3.10.12
    conda activate <env_name>
    ```

2. Install dependencies:

    ```shell
    # NOTE: following line shoule be adjusted according to your CUDA version
    # see <https://pytorch.org/get-started/previous-versions/> for detail
    # conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3  # for our old 2xRTX3090 server which CUDA<11.3
    pip install -q torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
    # ⚠ use pytorch > 2.0 for compile feature that accelerates training speed, 
    # ⚠ will cause gpu training loss nan problem for unknown reasons
    # conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   
    pip install torch_geometric -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
    # pip install torch_geometric
    pip install d2l tqdm scikit-learn dill
    ```

# Usage

## data preprocess pipeline

1. Modify path configuration in `utils.constant`
2. Download following files, and store them in `constant.PATH_DDI_DATA` directory:
   1. `ndc2rxnorm_mapping.txt`, `drug-atc.csv`, `drug-DDI.csv`, `ndc2atc_level4.csv`, `idx2drug.pkl`
      - from https://github.com/BarryRun/COGNet/blob/master/data/
   2. `TWOSIDES.csv.gz` 
      - from http://tatonettilab-resources.s3-website-us-west-1.amazonaws.com/?p=nsides/
3. Run cmd `python processing.py` under WWW'22 COGNet repo (need to `git clone` it in another directory) to obtain the `voc_final.pkl` file for DDI calculation
4. In our proj, `CD ./dataset`, then run cmd `preprocess_drugs.py`, `preprocess_labitems.py`, `construct_hgs.py` for generating data
   - In the `construct_graph.py`, `batch_size` is the hyperparemeter specifies how many patient nodes contian in a single $\mathcal{DTDG}$. 
   - It's set to $128$ by default due to our experiment has shown this number is appreciate.

## train and test

After finishing above, modify the default value of `root_path_dataset` in `main.py` according to the path where you store the $\mathcal{DTDG}$s.

TODO: Then modify or just run the CMDs in `expr.sh` script for training or testing.
