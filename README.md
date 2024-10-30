# Environment

```bash
pip install -q torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -q torch_geometric -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install -q d2l tqdm scikit-learn dill torcheval
```

# How to use

1. Get the dataset from the [official website](https://mimic.mit.edu/) of mimic-iii dataset:
  - the completion of an ethics course is required.
  - then you need to sign the *PhysioNet Credentialed Health Data Use Agreement 1.5.0 - Data Use Agreement for the MIMIC-III Clinical Database (v1.4)* to get the accessment of dataset.
2. modify the paths in `utils/constant.py`
3. run data preprocessing scripts in the `dataset/preprocess*.py`
4. cmds: 
  - train: `python run_backbone.py --train --use_gpu`
  - test: `python run_backbone.py --test --use_gpu --model_ckpt=<specify *.pt file in model/hub>`