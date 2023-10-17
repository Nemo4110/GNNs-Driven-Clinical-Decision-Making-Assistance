# MedDG

[TOC]

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

## Usage

### data preprocess pipeline

- Modify the `path_dataset` parameter in the both `preprocess_drugs.py` adn `preprocess_labitems.py` according to the path where you store the `mimic-iii v1.4` dataset.
  - as well as modify the `path_ddi_dataset` parameter in `preprocess_drugs.py`
- Run the `python preprocess_drugs.py` and `python preprocess_labitems.py` CMD in the `dataset` directory.
  - (OPTIONAL) For running `processes_drugs.ipynb` and `processes_labitems.ipynb`, you need modify all parameters relate to path according to the path where you store the `mimic-iii v1.4` dataset and the `DDI` directory.
  - Some unit cells in `processes.ipynb` may run out of time, this is why we recommend to use the equivalent `preprocess_drugs.py` adn `preprocess_labitems.py` scripts instead.

### dataset split

In the `construct_graph.py`, there are $3$ parameter need to modify:

- `path_dataset` is the path where you store the `mimic-iii v1.4` dataset
- `batch_size` is the hyperparemeter specifies how many patient nodes contian in a single $\mathcal{DTDG}$. It's set to $128$ by default due to our experiment has shown this number is appreciate.
- `path_hgs` specifies where you want to store the $\mathcal{DTDG}$s.

Run all unit cells! DONE.

### train and evaluate

After finishing above, modify the default value of `root_path_dataset` in `main.py` according to the path where you store the $\mathcal{DTDG}$s.

Then modify or just run the CMDs in `expr.sh` script for training or evaluating.
