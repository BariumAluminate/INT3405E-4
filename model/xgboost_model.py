
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sys

from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
)



# 1. PATHS AND CONFIGURATION

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "DataSolve" / "output"
MODEL_DIR = ROOT / "model" / "output"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "DataSolve"))

TARGET = "is_fraud"
ID = "id"

RANDOM_STATE = 42


# 2. LOAD DATA


print("Loading datasets...")

train_df = pd.read_csv(
    DATA_DIR / "raw_fe" / "train_fe.csv"
)

val_df = pd.read_csv(
    DATA_DIR / "raw_fe" / "validation_fe.csv"
)

y_train = train_df[TARGET].to_numpy()
y_val = val_df[TARGET].to_numpy()



# 3. PREPROCESSING


print("Loading preprocessor...")

preprocessor = joblib.load(
    DATA_DIR
    / "preprocessors"
    / "preprocessor_v2_raw_plus_fe.pkl"
)

transformer = preprocessor.named_steps["transform"]

X_train = transformer.transform(train_df)
X_val = transformer.transform(val_df)

feature_names = transformer.get_feature_names_out()

X_train = pd.DataFrame(
    X_train,
    columns=feature_names
)

X_val = pd.DataFrame(
    X_val,
    columns=feature_names
)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)



# 4. HANDLE CLASS IMBALANCE


negative_count = np.sum(y_train == 0)
positive_count = np.sum(y_train == 1)

scale_pos_weight = negative_count / positive_count

print("Negative samples:", negative_count)
print("Positive samples:", positive_count)
print("Scale pos weight:", scale_pos_weight)



# 5. TRAIN XGBOOST


params = {
    "n_estimators": 1500,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "gamma": 0,
    "reg_lambda": 1.0,
    "scale_pos_weight": scale_pos_weight,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "early_stopping_rounds": 50,
}

print("\nTraining XGBoost...")
print("Parameters:", params)

model = XGBClassifier(**params)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
)

print("Training completed!")



# 6. EVALUATE MODEL


train_probs = model.predict_proba(X_train)[:, 1]
val_probs = model.predict_proba(X_val)[:, 1]

print("\n===== XGBOOST =====")

print(
    "Train ROC-AUC:",
    roc_auc_score(y_train, train_probs)
)

print(
    "Validation ROC-AUC:",
    roc_auc_score(y_val, val_probs)
)

val_preds = model.predict(X_val)

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, val_preds))

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        val_preds,
        zero_division=0
    )
)



# 7. SAVE MODEL


model_path = MODEL_DIR / "xgboost_model.json"

model.save_model(model_path)

print("\nModel saved at:", model_path)



# 8. CREATE KAGGLE SUBMISSION


print("\nLoading Kaggle test data...")

test_df = pd.read_csv(
    DATA_DIR / "raw_fe" / "test_fe.csv"
)

test_ids = test_df[ID]

X_test = transformer.transform(test_df)

X_test = pd.DataFrame(
    X_test,
    columns=feature_names
)

test_probs = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    ID: test_ids,
    TARGET: test_probs,
})

RESULT_DIR = ROOT / "xgboost_model_result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

submission_path = RESULT_DIR / "submission.csv"

submission.to_csv(
    submission_path,
    index=False
)

print("\nSubmission saved at:", submission_path)
print(submission.head())
