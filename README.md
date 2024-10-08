# NOTE

- Get the dataset from the [official website](https://mimic.mit.edu/) of mimic-iii dataset:
  - the completion of an ethics course is required.
  - then you need to sign the *PhysioNet Credentialed Health
Data Use Agreement 1.5.0 - Data Use Agreement for the
MIMIC-III Clinical Database (v1.4)* to get the accessment of dataset. 
- data preprocessing scripts are located in the dataset/preprocess*
- The scripts for running the experiments on Google Colab are located in the .ipynb files under the notebook directory:
  - run_backbone.ipynb
  - run_baseline*.ipynb
- Paths should be modified in utils/constant.py (we used Google Drive and mounted it to the Colab runtime).


# UPDATE

After a series of experiments, we found that the main reasons the previous results of the main model did not achieve optimal performance may be as follows:

- Learning rate related (the part with the greatest benefit)
  - Our model based on GNN and attention mechanisms, with a previous learning rate setting of 0.001, was able to converge in the early stages. However, in the later stages of training, it oscillated around the optimal parameter solution and did not converge effectively towards the optimum.
  Therefore, we lowered the learning rate to between 0.0003 and 0.0001, and employed a cosine annealing learning rate adjustment strategy to gradually decrease the learning rate, which led to the optimal results we obtained.

- Parameter initialization
  - Previously, we initialized the model parameters using a normal distribution with a mean of 0 and a standard deviation of 0.01.
  - After consulting various tuning guidelines online, we are now using Xavier normal initialization.
- Attention weight smoothing in the MHA layer
  - The previous MHA layer applied attention weights to represent the patient condition across different snapshots of days. However, we found that when using GNN type GINEConv, the attention mechanism did not seem to enhance performance.
    - After visual inspection, we observed that it applied almost the same weight to each day, indicating that this aspect was not adequately trained.
  - In the current model, we optimized the way the attention mechanism is used—treating the target item as the query, the historical condition representation sequence as the key-value, and using additive attention for learning.

---

# Latest Result on Drug Recommend Task!

 **Model** | **AUC**    | **AP**  | **ACC** | **Precision** | **Recall** | **F1**  
-----------|------------|---------|---------|---------------|------------|---------
 BPR       | 0.7035     | 0.5622  |  0.6744 |  0.5109       |  0.5432    |  0.5266 
 NeuMF     | 0.7108     |  0.5824 |  0.7159 |  0.6626       | 0.3011     |  0.4140 
 DeepFM    | 0.9595     |  0.9204 | 0.8983  | 0.8406        | 0.8575     |  0.8490 
 DSSM      | 0.9258     | 0.8617  |  0.8812 |  0.8250       |  0.8168    | 0.8209  
 SASRec    | 0.9420     |  0.8879 |  0.8754 |  0.8162       | 0.8083     | 0.8122  
 DIN       | 0.9933     | 0.9884  | 0.9643  |  0.9525       | 0.9398     |  0.9461 
 **Ours**     | **0.9954** | **0.9917**  | **0.9703**  | **0.9677**        | **0.9410**     | **0.9542**  

