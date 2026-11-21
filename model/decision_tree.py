
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
)



# 1. PATHS


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "DataSolve" / "output"
MODEL_DIR = ROOT / "model" / "output"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Required for loading the saved preprocessor
sys.path.insert(0, str(ROOT / "DataSolve"))



# 2. CONFIGURATION


TARGET = "is_fraud"
ID = "id"

BEST_PARAMS = {
    "criterion": "entropy",
    "max_depth": 6,
    "min_samples_leaf": 50,
    "min_samples_split": 50,
    "class_weight": "balanced",
    "random_state": 42,
}



# 3. LOAD FEATURE-ENGINEERED DATA


print("Loading v2_raw_plus_fe datasets...")

train_df = pd.read_csv(DATA_DIR / "raw_fe" / "train_fe.csv")
val_df = pd.read_csv(DATA_DIR / "raw_fe" / "validation_fe.csv")

y_train = train_df[TARGET]
y_val = val_df[TARGET]

print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)



# 4. LOAD PREPROCESSOR AND TRANSFORM DATA


print("\nLoading v2 preprocessor...")

preprocessor = joblib.load(
    DATA_DIR / "preprocessors" / "preprocessor_v2_raw_plus_fe.pkl"
)

# Data is already cleaned and feature-engineered in raw_fe.
# Therefore, apply only the final ColumnTransformer.
transformer = preprocessor.named_steps["transform"]

X_train = transformer.transform(train_df)
X_val = transformer.transform(val_df)

feature_names = transformer.get_feature_names_out()

X_train = pd.DataFrame(X_train, columns=feature_names)
X_val = pd.DataFrame(X_val, columns=feature_names)

if list(X_train.columns) != list(X_val.columns):
    raise ValueError("Train and validation features do not match!")

print("Feature consistency: OK")
print("Number of features:", X_train.shape[1])



# 5. TRAIN DECISION TREE


print("\nTraining Decision Tree with v2_raw_plus_fe...")
print("Parameters:", BEST_PARAMS)

model = DecisionTreeClassifier(**BEST_PARAMS)
model.fit(X_train, y_train)

print("Training completed!")



# 6. EVALUATE MODEL


train_probs = model.predict_proba(X_train)[:, 1]
val_probs = model.predict_proba(X_val)[:, 1]

train_auc = roc_auc_score(y_train, train_probs)
val_auc = roc_auc_score(y_val, val_probs)

print("\n===== DECISION TREE - V2 RAW + FEATURE ENGINEERING =====")
print("Train ROC-AUC:", train_auc)
print("Validation ROC-AUC:", val_auc)

val_preds = model.predict(X_val)

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, val_preds))

print("\nClassification Report:")
print(classification_report(y_val, val_preds, zero_division=0))



# 7. SAVE MODEL


model_path = MODEL_DIR / "decision_tree_best.pkl"

joblib.dump(model, model_path)

print("\nModel saved at:", model_path)



# 8. CREATE KAGGLE SUBMISSION


print("\nLoading feature-engineered Kaggle test data...")

test_df = pd.read_csv(DATA_DIR / "raw_fe" / "test_fe.csv")

test_ids = test_df[ID]

X_test = transformer.transform(test_df)
X_test = pd.DataFrame(X_test, columns=feature_names)

if list(X_test.columns) != list(X_train.columns):
    raise ValueError("Test and train features do not match!")

if X_test.isnull().any().any():
    raise ValueError("Test data contains missing values!")

print("Test shape:", X_test.shape)

# ROC-AUC requires predicted probabilities, not binary labels.
test_probs = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    ID: test_ids,
    TARGET: test_probs,
})

submission_path = MODEL_DIR / "submission.csv"
submission.to_csv(submission_path, index=False)

print("\n===== SUBMISSION COMPLETED =====")
print(submission.head())
print("Submission shape:", submission.shape)
print("Submission saved at:", submission_path)
