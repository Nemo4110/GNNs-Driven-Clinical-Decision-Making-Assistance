# LERS

The acronym of **Lab Events Recommender Systems**, due to the file name "LABEVENTS.csv.gz" of core data.

## Overview

This project aims to use sequential recommendation systems technique to help clinicians when scheduling laboratory items for patients.

We propose an objective model that leverages patients' historical results from Electronic Medical Records (EMR) to provide clinicians with supplementary advice on scheduling laboratory items. By transforming the recommendation problem into edges prediction problem on dynamic graphs, utilizing the natural heterogeneity of EMR data, our model demonstrates high efficiency and practicality.

## Installation of environment

### Prerequisites :warning:

- A Nvidia `GPU` with enough memory.
  - In our case, we use 3090 that has about 24 GB memory, and the CUDA version is 11.3.
  - In fact, the memory does not need to be so large.
- `conda`
- Get the dataset from the [official website](https://mimic.mit.edu/) of mimic-iii dataset:
  - the completion of an ethics course is required.
  - then you need to sign the *PhysioNet Credentialed Health
Data Use Agreement 1.5.0 - Data Use Agreement for the
MIMIC-III Clinical Database (v1.4)* to get the accessment of dataset. 

### Steps

In the CMD:

1. Create conda environment:

    ```shell
    conda create LERS python==3.9.0
    conda activate LERS
    ```

2. Install dependencies:

    ```shell
    # NOTE: following line shoule be adjusted according to your CUDA version
    # see <https://pytorch.org/get-started/previous-versions/> for detail
    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3

    conda install pyg -c pyg

    pip install d2l
    pip install tqdm
    pip install scikit-learn
    ```

    Alternative:

    ```shell
    pip install -r requirements.txt
    ```

## Usage

### data preprocess

- Modify the `path_dataset` parameter in the `preprocess.py` according to the path where you store the `mimic-iii v1.4` dataset.
- Run the `python preprocess.py` CMD in the `dataset` directory.
  - (OPTIONAL) For running `processes.ipynb`, you need modify the `path_dataset` parameter according to the path where you store the `mimic-iii v1.4` dataset.
  - Some unit cells in `processes.ipynb` may run out of time, this is why we recommend to use the equivalent `preprocess.py` script instead.

### dataset split

In the `construct_graph.ipynb`, there are $3$ parameter need to modify:

- `path_dataset` is the path where you store the `mimic-iii v1.4` dataset
- `batch_size` is the hyperparemeter specifies how many patient nodes contian in a single $\mathcal{DTDG}$. It's set to $128$ by default due to our experiment has shown this number is appreciate.
- `path_hgs` specifies where you want to store the $\mathcal{DTDG}$s.
  - We recommend you set these path according to the **Project Structure** above

Run all unit cells! DONE.

### train and evaluate

After finishing above, modify the default value of `root_path_dataset` in `main.py` according to the path where you store the $\mathcal{DTDG}$s, then just run the following CMD in `LERS` directory:

```shell
python main.py --gnn_type=GENConv --batch_size_by_HADMID=128
```

### about evaluate result

By defult, the .csv files that contain the evaluate result will store in the `results/dfs` directory, and the `evaluate.ipynb` is used for visualizing them.
