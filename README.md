# Telecom Churn Prediction

Binary classifier that predicts whether a telecom customer will cancel(churn),
using an MLP (PyTorch) on the IBM Telco Customer Churn dataset.

## Model
- Preprocessing: `ColumnTransformer` (median imputation + StandardScaler for numeric,
  most-frequent imputation + OneHotEncoder for categorical)
- Architecture: Input → Dense(64) + ReLU + Dropout(0.3) → Dense(32) + ReLU + Dropout(0.3) → Dense(1)
- Loss: `BCEWithLogitsLoss` with `pos_weight` to handle class imbalance (~26% churn)
- Optimizer: Adam (lr = 0.001), 100 epochs

## Results (test set, 20% stratified split)

| Metric | Value |
| --- | --- |
| AUC-ROC | 0.8395 |
| Recall (churn) | 0.78 |
| Precision (churn) | 0.52 |
| Accuracy | 0.75 |

The model favors recall: it catches 78% of customers who cancel, at the cost of
more false alarms (precision 0.51).

## How to run
```bash
pip install -r requirements.txt
python churn_telecom.py
```

## Dataset
[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM, Kaggle).
