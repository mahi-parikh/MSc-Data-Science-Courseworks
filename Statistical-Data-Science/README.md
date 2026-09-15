# Statistical Data Science — Leaf Species Classification

MSc Data Science coursework. This project builds and compares logistic
regression models (linear and polynomial) to classify leaf species from leaf
length and width measurements.

## Project Overview

- **Data prep**: 80/20 train/test split of the leaf dataset.
- **Linear logistic regression**: baseline model, evaluated with a confusion
  matrix, precision, recall, and F1 score.
- **Polynomial logistic regression**: the same evaluation repeated for
  degree-2, degree-3, and degree-4 polynomial feature expansions, to see how
  model complexity affects the fit.
- **Visualization**: decision-boundary plots for each model degree, showing
  underfitting at low degrees and overfitting at degree 4.

## Key results

The linear logistic regression model underperformed, misclassifying a
meaningful share of points due to underfitting — the true boundary between
species isn't linear. Increasing to a degree-2 polynomial gave no real
accuracy improvement over the linear model. Degree-3 was the best fit: test
accuracy rose to 0.8025, and the decision-boundary plot showed a
noticeably cleaner separation between classes. Degree-4 overfit the data,
producing a visibly more convoluted decision boundary without a
corresponding gain in generalization.

## Files

- `notebooks/leaf_classification_logistic_regression.ipynb` — full analysis
  (R notebook)
- `data/leaf_data.csv` — leaf length/width measurements with species labels

## Reproducing

```r
install.packages(c("caret", "lattice", "ggplot2"))
```

Then open `notebooks/leaf_classification_logistic_regression.ipynb` in
Jupyter with an R kernel (or in RStudio) and run top to bottom — it reads
`leaf_data.csv` from the `data/` folder via a relative path.
