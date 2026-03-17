# Baseline: Lagged-Feature MLP for Stock Market Signal

## Task

- Competition: **Kaggle – Stock Market Signal: Predict Next-Day Returns**  
- Goal: For each stock and each date, predict whether the **next-day return** will be positive (up) or negative (down).  
- In this baseline, we **do not use RNNs**.  
- Instead, we build a **plain MLP** that takes a fixed window of past days as input features (lagged features).

***

## 1. Data loading and basic preprocessing

- Load the Kaggle competition data (train/test CSVs).  
- For today, **focus on a small subset of stocks** (e.g., top 10 by volume) to keep experiments light.  
- Select a few basic features per day, for example:
  - `Open`, `Close`, `High`, `Low`, `Volume`  
- Sort data by `(Stock, Date)` and fill or drop missing values.

***

## 2. Build lagged features (fixed window)

- Choose a **window length** \(T\) (e.g., 10 days).  
- For each stock and date \(D\):
  - Construct an input vector by stacking the last \(T\) days of features:  
    \[
    x_D = [f_{D-T+1}, f_{D-T+2}, \dots, f_{D}]
    \]
    where \(f_d\) is the feature vector for day \(d\).  
- If there are not enough past days (start of series), **drop** those rows.  
- Target label:
  - 1 if **next-day return** \(r_{D+1} > 0\)  
  - 0 otherwise.

***

## 3. Train / validation split

- Split the time series **chronologically** (no shuffling):
  - First 80% of dates → **train**  
  - Last 20% of dates → **validation**  
- Important: Do **not** leak future information into the past.

***

## 4. MLP model

- Input dimension: `T * num_features`.  
- Architecture (example):
  - Dense(128) → ReLU  
  - Dense(64) → ReLU  
  - Dense(1) → Sigmoid  
- Loss: Binary Cross-Entropy.  
- Optimizer: Adam.  
- Metric: Accuracy, AUC (optional).

***

## 5. Training loop (PyTorch or Keras)

- Normalize input features (e.g., per-feature standardization).  
- Batch training with mini-batches (e.g., batch size 256).  
- Train for a small number of epochs (e.g., 10–20) and monitor validation metrics.  
- Save the **best** model checkpoint based on validation performance.

***

## 6. Evaluation and discussion

- Report:
  - Validation accuracy (and/or AUC).  
  - Confusion matrix for “up” vs “down” predictions.  
- Short analysis:
  - Does the model seem to use lagged information?  
  - Are there signs of overfitting (train vs validation gap)?  
- This baseline will be the **reference** when we later introduce GRU / minGRU models on the same data.

***

## 7. What to submit in the notebook

Your Jupyter notebook should contain:

1. Data loading and basic EDA for the chosen stocks.  
2. Code for creating lagged features with window length \(T\).  
3. MLP model definition and training code.  
4. Validation metrics and one or two simple plots (e.g., loss curve).  
5. A short markdown cell (5–10 lines) discussing:
   - Why lagged features allow an MLP to handle time series.  
   - What its limitations are compared to a “real” sequence model.
