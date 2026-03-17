# Lab Task: Tesla Directional Move – Lagged MLP vs RNNs

## 0. Context

In this lab, you will work on the Kaggle competition:

https://www.kaggle.com/competitions/tesla-directional-move-prediction-from-technical-indicators

The dataset contains **38 columns** in total: **1 ID**, **1 target**, and **36 numeric features** (technical indicators).  
We provide a complete baseline using a **lagged-feature MLP**, and your goal is to implement and compare **sequence models (GRU / minGRU)** under a **fair, controlled setting**.

This is a **group competition** inside the class: groups will be compared based on both **predictive performance** and **quality of analysis**, not leaderboard position alone.

***

## 1. Data and features

You are given a single time series of Tesla data ordered by an integer **ID** (treat it as time index).  
The target column **`y`** is categorical with values such as `"up"` and `"neutral"` (we will map this to a binary label).

Feature columns (36 technical indicators) include:

- `'SMA', 'EMA', 'ADX', 'Aroon', 'TDI-1', 'TDI-2', 'Donchian-1', 'Donchian-2', 'Donchian-3', 'VHF', 'CCI', 'MFI', 'OBV', 'MACD-1', 'MACD-2', 'RSI', 'Stochastics-1', 'Stochastics-2', 'Stochastics-3', 'CMO', 'KST-1', 'KST-2', 'TRIX-1', 'TRIX-2', 'ROC', 'DVI-1', 'DVI-2', 'DVI-3', 'Momentum', 'Bbands-1', 'Bbands-2', 'Bbands-3', 'Volatility', 'Pbands-1', 'Pbands-2', 'Pbands-3'`

Notable groups of indicators:

- Trend: SMA, EMA, ADX, Aroon  
- Momentum / oscillators: RSI, CCI, ROC, Momentum, CMO, TRIX-*, KST-*, Stochastics-*  
- Volume / flow: OBV, MFI  
- Channels / bands: Donchian-*, Bbands-*, Pbands-*  
- Volatility proxy: Volatility  

You should assume that:

- Features may be **correlated** and live on **very different scales**.  
- Some features are **bounded oscillators** (e.g. RSI), others are **unbounded** or can be very large (e.g. OBV).

For this lab, we treat this as a **binary classification** problem:

- Map `y = "up"` to label `1`  
- Map all other values of `y` (e.g. `"neutral"`, `"down"`) to label `0`.

***

## 2. Baseline: Lagged-Feature MLP (provided)

The baseline notebook includes a complete implementation of a **lagged-feature MLP**:

- We sort the data by `ID` and interpret it as a single time series.  
- For each time step \(t\), we take a fixed window of the **previous \(T\)** steps’ features as input, flatten them into a single vector, and predict whether the **next** step \(t+1\) will be `"up"` or not.  
- This gives an input of shape \(T \times \text{num\_features}\), which is fed into a standard MLP.

The baseline code already handles:

- Data loading from Kaggle.  
- Lagged feature construction.  
- Feature scaling.  
- MLP model definition and training loop.

Your first task is simply to **run the baseline**, record its **validation metrics** and understand:

- How lagged features are constructed from the time series.  
- How the label is defined (up vs non-up).  
- What performance you get as a starting point.

***

## 3. Your main task: RNN models under fair comparison

You will implement **sequence models** (RNNs) that consume the same history window as a **sequence**, rather than a flattened vector.

### 3.1 Input representation

- For the MLP, the input is of shape:  
  - \((\text{batch}, T \times F)\) where \(T\) is window length and \(F\) is number of features.  
- For RNNs, you must reshape this into:  
  - \((\text{batch}, T, F)\) and feed it step-by-step through the recurrent model.

You must use the **same window length \(T\)** as the MLP baseline, unless you explicitly justify and document a different choice.

### 3.2 Models to implement

You must implement and train at least:

- **GRU classifier**  
- **minGRU classifier** (minimal GRU cell from the lecture)

LSTM is optional for extra exploration and discussion.

***

## 4. Experimental protocol and fairness constraints

To ensure a **fair comparison** between groups and models, all experiments must respect the following constraints:

- **Single train/validation split**:  
  - Use a **time-based split** on the ID (first 80% of IDs for train, last 20% for validation).  
  - No random reshuffling over time.

- **Fixed batch size**:  
  - Use the same `batch_size` for **all** models (MLP, GRU, minGRU, optional LSTM).  
  - The default lab value is `batch_size = 256` (you may change it only if you clearly justify it and keep it consistent across models).

- **Maximum training epochs**:  
  - You may train each model for up to **300 epochs**.  
  - You do not have to use all 300 epochs; you may use fewer and/or early stopping, but you may not exceed 300 for any model.

- **Same optimizer class and base learning rate family**:  
  - Use the **Adam** optimizer for all models.  
  - Learning rates can differ slightly between models if you justify the choice, but you should keep them in a similar range.

- **Same feature preprocessing**:  
  - Use the **same scaling** (e.g. StandardScaler fitted on train) for all models.  
  - Do not apply model-specific feature engineering that changes the input distribution for only one model, unless you clearly document and justify it.

Your goal is not just to “overfit one model”, but to compare **architectures** under comparable conditions.

***

## 5. Evaluation metrics

You must report at least the following metrics on the **validation set**:

- **Accuracy**  
- **F1 score** for the **positive class** (`up` = label 1)

F1 is computed with `up` as the positive class, using the standard binary F1 definition.  
You may also report additional metrics (e.g. confusion matrix, ROC AUC), but **accuracy and F1** are mandatory and will be used for comparison across groups.

For each model (MLP, GRU, minGRU, optional LSTM), you must report:

- Validation accuracy.  
- Validation F1 (positive class).  
- Number of trainable parameters.  
- Approximate training time per epoch or total training time.

***

## 6. What you must deliver (in your notebook)

Your final Kaggle notebook submission must include:

1. **Baseline MLP section**  
   - Code that runs the lagged-feature MLP baseline.  
   - A short written recap (5–10 sentences) describing the input, target, and baseline performance.

2. **RNN implementation section**  
   - GRU and minGRU implementations (and optionally LSTM).  
   - Code that reshapes inputs into \((\text{batch}, T, F)\).  
   - Training loops respecting the **300-epoch cap**, fixed batch size, and the shared train/validation split.

3. **Evaluation and comparison section**  
   - Validation metrics (accuracy, F1) for all models.  
   - A small comparison table summarizing:
     - Model type  
     - Parameter count  
     - Accuracy  
     - F1 (positive class)  
     - Training time

4. **Analysis / discussion section**  
   - At least 8–10 sentences discussing:
     - How the RNN models (GRU, minGRU) compare to the MLP.  
     - Whether sequence modeling seems to capture additional temporal patterns beyond lagged features.  
     - How the metrics (accuracy vs F1) change across models, and why that might be.  
     - Any signs of overfitting or underfitting.  
     - Trade-offs between model complexity, training time, and performance.

***

## 7. Group competition and grading

This is a **group-based challenge**. Your group will be evaluated on:

- Correctness and robustness of your implementations.  
- Respecting the **fairness constraints** (same split, batch size, epoch cap).  
- Quality of your analysis and written discussion.  
- Relative performance (accuracy and F1) compared to other groups, within reasonable variance.

Leaderboards are not everything: a slightly weaker model with an excellent, honest analysis is preferred over a marginally better metric with no insight.
