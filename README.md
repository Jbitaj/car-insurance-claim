# Car Insurance Claim Prediction

Data mining course project. Binary classification: predict whether a driver filed an insurance claim (`OUTCOME`), comparing a Decision Tree and a Linear SVM.

## Dataset

- Source: [Car Insurance Data on Kaggle](https://www.kaggle.com/datasets/sagnik1511/car-insurance-data)
- ~10,000 rows, 19 columns (17 input features + `OUTCOME` after dropping `ID`)
- Imbalanced: ~69% no claim (0), ~31% claim (1)
- The CSV is not included in this repo. Download it from Kaggle and place it next to the notebook.

## Pipeline

All fitted steps use the training set only, so there is no data leakage.

1. Drop `ID`, inspect types, missing values, outliers and class balance
2. Stratified 80/20 train/test split
3. Remove outliers (IQR) from the training set, only for `CREDIT_SCORE` and `ANNUAL_MILEAGE`
4. Balance classes by undersampling the training set (2018 samples per class)
5. Preprocessing inside a `Pipeline`: mean imputation + `MinMaxScaler` for numeric columns, `OneHotEncoder` for categorical columns
6. Train Linear SVM and Decision Tree (`max_depth=5`)
7. Tune both with `GridSearchCV` (5-fold, F1)

## Results (test set, 1630 samples)

| Model | Accuracy | Balanced Acc | F1 | MAE | MSE |
|---|---|---|---|---|---|
| Linear SVM | 0.8117 | 0.8173 | 0.7333 | 0.1883 | 0.1883 |
| Decision Tree | 0.8092 | 0.8231 | 0.7371 | 0.1908 | 0.1908 |
| Tuned SVM (linear, C=0.1) | 0.8104 | 0.8175 | 0.7329 | 0.1896 | 0.1896 |
| Tuned Tree (entropy, depth 5, min leaf 5) | 0.8092 | 0.8231 | 0.7371 | 0.1908 | 0.1908 |

For 0/1 labels, MAE and MSE both equal the error rate (1 - accuracy).

## Key findings

- All models reach ~81% accuracy, well above the 69% majority-class baseline. Differences between models are only a few samples, so none is clearly better.
- The tree has slightly higher recall on claims (~0.86), so it misses fewer real claims. Most errors are false alarms, a side effect of training on balanced data and testing on imbalanced data.
- Hyperparameter tuning barely helped and `rbf` did not beat the linear kernel.
- Most important features in both models: low driving experience (0-9 years), vehicle ownership, vehicle year, postal code. Feature importance does not imply causation, and `POSTAL_CODE` is a region code, not a numeric value.

## Run

```bash
pip install -r requirements.txt
```

Then run the notebook from top to bottom.
