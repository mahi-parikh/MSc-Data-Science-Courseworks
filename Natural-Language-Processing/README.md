# Natural Language Processing — Sentiment & Sarcasm Classification (BESSTIE)

MSc Data Science group coursework. This project classifies sentiment and sarcasm
across three varieties of English — Indian, Australian, and UK — using the
BESSTIE benchmark dataset (Srirag et al., 2025), sourced from Google Reviews
and Reddit posts.

> The full written report is kept in a separate private repository. This repo
> holds only the code: the experimentation notebook and the deployed Flask app.

## Project Overview

**Exploratory analysis**: dataset composition by variety/source, sentiment and
sarcasm class balance, vocabulary size and overlap between varieties (Jaccard
similarity, Jensen-Shannon divergence), and dependency-parse-based sarcasm
inspection.

**Three experimental setups**, compared across the three English varieties:

1. **Baseline vs. pre-trained transformer** — TF-IDF + Logistic Regression vs.
   RoBERTa-base, for both sentiment and sarcasm classification.
2. **Cross-variety evaluation** — fine-tuning XLM-RoBERTa on each variety
   individually and testing on the other two, to measure how well a model
   trained on one variety of English transfers to another.
3. **LLM fine-tuning via LoRA** — Qwen2.5-1.5B-Instruct with variety-specific
   LoRA adapters (applied to q/k/v/o projection layers, rank 8), trained with a
   prompt-answer format for sarcasm detection, evaluated in-domain and
   cross-domain.

**Deployment**: a Flask web app (`app.py`, `index.html`) that serves the
TF-IDF/Logistic Regression model for sentiment and the Qwen + LoRA adapters for
sarcasm, letting a user pick the English variety and task and get a live
prediction. Model artifacts are hosted on Hugging Face and downloaded at
runtime.

## Key results

- **Baseline vs. transformer**: RoBERTa-base outperformed the TF-IDF +
  Logistic Regression baseline on sentiment (0.88 vs. 0.81 accuracy) and on
  sarcasm (0.86 vs. 0.75 accuracy). Sarcasm remained harder for both models —
  macro F1 sat at 0.46–0.64, well below sentiment's ~0.89, due to class
  imbalance (most samples are non-sarcastic).
- **Cross-variety transfer**: models trained and tested on the same variety
  performed best (e.g. UK→UK Macro-F1 of 0.9364), but the model trained on
  Indian English generalized best to UK English, and generalization was
  consistently weaker on Indian English as a target.
- **LoRA adapters**: the Australian-English adapter generalized best overall
  (F1 0.7715 in-domain, 0.5618 and 0.4916 cross-domain), and was the only
  adapter to reliably detect sarcasm (F1 0.69 in-domain). The Indian-English
  adapter was weakest, and in its own-variety evaluation failed to detect any
  sarcastic instances at all.

## Files

- `code/app.py` — Flask backend serving the sentiment and sarcasm models
- `code/index.html` — front-end template for the app
- `notebooks/main.ipynb` — full experimentation: EDA, baseline, cross-variety
  fine-tuning, and LoRA training/evaluation

## Reproducing

```bash
pip install -r requirements.txt
```

`main.ipynb` runs top to bottom and expects the BESSTIE dataset to be
available via the `datasets` library. `app.py` downloads trained model
artifacts from Hugging Face Hub at runtime rather than bundling them —
run with `python app.py` once dependencies are installed.

## Note

This was **group coursework**. Only the code is published here; the full
written report remains in a private repository.
