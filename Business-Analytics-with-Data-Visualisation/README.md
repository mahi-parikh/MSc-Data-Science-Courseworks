# Business Analytics — Early Warning System for Student Dropout

MSc Data Science group coursework. This project builds an early-warning system
that predicts whether a student will drop out, stay enrolled, or graduate,
using only information available at the point of enrolment — deliberately
excluding any semester-level academic performance data, since the goal is to
flag at-risk students before they've had a chance to struggle academically.

> The full written report is kept in a separate private repository. This repo
> holds only the code.

## Dataset

["Predict Students' Dropout and Academic Success"](https://doi.org/10.24432/C5MC89)
(Realinho et al., 2022) from the UCI Machine Learning Repository — 4,424
students from a Portuguese higher education institution, 36 enrolment-time
features (demographics, application details, financial indicators, parental
background, macroeconomic context), plus 12 semester-level columns that were
excluded by design.

## Approach

The group followed CRISP-DM throughout. After shared EDA and preprocessing,
each team member trained two models — one individually chosen algorithm plus
a shared Logistic Regression baseline — tuned via GridSearchCV with 5-fold
stratified cross-validation, then evaluated on a common set of metrics
(balanced accuracy, macro F1, weighted F1, ROC-AUC, confusion matrices).

## Project Overview

- `notebooks/01_group_eda_preprocessing.ipynb` — shared exploratory analysis
  and preprocessing: loading and cleaning, univariate/bivariate analysis of
  binary and continuous features against the outcome, statistical feature
  selection (ANOVA, chi-square, Cramér's V), outlier capping, encoding, and
  an 80/20 stratified train/test split with Random Forest feature importance
  ranking (71 features selected from 36 original).
- `notebooks/02_mahi_decision_tree.ipynb` — Decision Tree + shared Logistic
  Regression baseline, GridSearchCV tuning over depth/split/leaf parameters.
- `notebooks/03_azad_naive_bayes.ipynb` — Bernoulli Naive Bayes + shared
  Logistic Regression baseline, Laplace smoothing tuning, feature influence
  analysis.
- `notebooks/04_pavan_lda.ipynb` — Linear Discriminant Analysis + shared
  Logistic Regression baseline, solver/shrinkage tuning.
- `notebooks/05_neha_svm.ipynb` — Support Vector Machine (RBF kernel) +
  shared Logistic Regression baseline, grid search over C/gamma/class weight,
  learning curve analysis.
- `notebooks/06_vinayak_knn.ipynb` — k-Nearest Neighbours + shared Logistic
  Regression baseline, tuning over k and distance weighting.
- `notebooks/07_group_model_comparison.ipynb` — brings together every
  member's results into shared comparison tables, per-class performance
  breakdowns, and cross-model visualisations.

Each individual notebook loads the same preprocessed train/test split
produced by the group EDA notebook, trains one member's chosen algorithm
alongside a shared Logistic Regression baseline, and evaluates both with a
common set of metrics for fair comparison across the group.

## Key results

Financial indicators were the strongest predictors across every model and
every team member's analysis — tuition fee status alone identifies dropout
risk at 86.6%, with scholarship and debtor status also standing out. Balanced
accuracy across the individually tuned models ranged from roughly 0.55 to
0.58, with ROC-AUC around 0.76. The hardest class to predict, consistently
across every algorithm tried (distance-based, probability-based, and
kernel-based), was "Enrolled" — these students haven't reached a final
outcome yet, so their profiles genuinely overlap with both dropouts and
graduates. The group's recommendation for deployment was a balanced
multinomial Logistic Regression, favoring interpretability (so advisors can
see why a student was flagged) over the marginal accuracy gains some other
models offered.

## Note

This was **group coursework**. Only the code is published here; the full
written report remains in a private repository.

## Reproducing

```bash
pip install -r requirements.txt
```

The group notebook was run on Google Colab and mounts Google Drive to load
`data.csv` from a shared project folder. To run it locally, replace the
`drive.mount(...)` cell with a local path to the dataset (e.g. a `data/`
folder alongside the notebook).
