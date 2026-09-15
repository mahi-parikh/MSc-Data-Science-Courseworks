# Machine Learning for Data Science — Group Coursework

**Module:** COMM075 – Machine Learning for Data Science
**Group:** MacLearnDS6

This repository contains our group's coursework for the Machine Learning for Data Science module. We applied four supervised learning algorithms and two additional learning frameworks to two binary classification datasets, then compared their performance.

## Datasets

| Dataset | Task | Source |
|---|---|---|
| **Adult Income Census** | Predict whether an individual earns `>50K` or `<=50K` per year | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult), loaded via the `ucimlrepo` package |
| **IBM HR Employee Attrition** | Predict whether an employee will leave the company (`Attrition = Yes/No`) | [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) (Kaggle) |

Raw data files are not included in this repository (see [Reproducing the results](#reproducing-the-results) below).

## Methods

**Algorithms**
- **K-Nearest Neighbours (KNN)**
- **Support Vector Machine (SVM)**
- **Multi-Layer Perceptron (MLP)**
- **Decision Tree**

**Frameworks**
- **Inductive Logic Programming (ILP)** — learns human-readable logical rules from the data (FOIL-style separate-and-conquer rule learning)
- **Reinforcement Learning (Q-Learning)** — frames classification as a sequential decision problem solved via Q-learning

Each algorithm/framework was applied independently to both datasets, tuned via grid/randomised search where applicable, and evaluated with accuracy, balanced accuracy, F1, confusion matrices, and ROC/AUC.

## Repository structure

```
notebooks/
├── 01_eda/                      Exploratory data analysis for each dataset
│   ├── AdultIncome_EDA.ipynb
│   └── IBM_Attrition_EDA.ipynb
├── 02_classical_ml/              KNN, SVM, MLP, Decision Tree models
│   ├── Decision_Tree_AdultIncome.ipynb
│   ├── Decision_Tree_IBM_Attrition.ipynb
│   ├── KNN_Adult_IBM.ipynb
│   ├── SVM_Adult_IBM.ipynb
│   └── MLP_Adult_IBM.ipynb
├── 03_ilp/                       Inductive Logic Programming models
│   ├── AdultIncome_ILP.ipynb
│   └── IBM_Attrition_ILP_Model.ipynb
├── 04_reinforcement_learning/    Q-learning models
│   ├── AdultIncome_QLearning_RL.ipynb
│   └── IBM_Attrition_QLearning_RL.ipynb
└── 05_results/                   Combined results and comparison plots
    └── ResultsNotebook_Plots.ipynb
```

## Reproducing the results

The notebooks were developed and run in **Google Colab**, with cleaned/preprocessed train and test splits read from Google Drive (e.g. `train_data.csv`, `adult_income_train_preprocessed_cleaned.csv`, `train_data_ibm.csv`). Each model notebook writes its own metrics/ROC values to CSV, which `05_results/ResultsNotebook_Plots.ipynb` then reads back in to build the comparison plots.

To run these notebooks outside Colab:

1. Install dependencies: `pip install -r requirements.txt`
2. Download the two datasets from the sources listed above.
3. Run the relevant `01_eda` notebook first to generate the cleaned train/test CSVs.
4. Update the hardcoded Colab/Drive paths (`/content/drive/MyDrive/...`, `drive.mount(...)`) at the top of each notebook to point to your local copies of those CSVs.
5. Run the model notebooks (`02`–`04`), then `05_results/ResultsNotebook_Plots.ipynb` to reproduce the comparison plots.

## Authors

MacLearnDS6 — *add group member names here*

## Acknowledgements

Coursework submitted for COMM075: Machine Learning for Data Science.
