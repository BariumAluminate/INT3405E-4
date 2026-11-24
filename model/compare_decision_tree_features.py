
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



# 1. Paths and imports


ROOT = Path(__file__).resolve().parents[1]

# Needed to load the saved preprocessor
sys.path.insert(0, str(ROOT / "DataSolve"))

OUTPUT = ROOT / "DataSolve" / "output"
MODEL_OUTPUT = ROOT / "model" / "output"
MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)



# 2. Decision Tree configuration


PARAMS = {
    "criterion": "entropy",
    "max_depth": 6,
    "min_samples_leaf": 50,
    "min_samples_split": 50,
    "class_weight": "balanced",
    "random_state": 42,
}



# 3. Load Model A: v1_raw (54 features)


print("Loading Model A: v1_raw...")

train_a = pd.read_csv(OUTPUT / "train.csv")
val_a = pd.read_csv(OUTPUT / "validation.csv")

X_train_a = train_a.drop(columns=["id", "is_fraud"])
y_train_a = train_a["is_fraud"]

X_val_a = val_a.drop(columns=["id", "is_fraud"])
y_val_a = val_a["is_fraud"]



# 4. Load Model B: v2_raw_plus_fe (70 features)


print("Loading Model B: v2_raw_plus_fe...")

train_b_raw = pd.read_csv(OUTPUT / "raw_fe" / "train_fe.csv")
val_b_raw = pd.read_csv(OUTPUT / "raw_fe" / "validation_fe.csv")

# Ensure both feature sets use exactly the same rows
if not np.array_equal(train_a["id"].to_numpy(),
                      train_b_raw["id"].to_numpy()):
    raise ValueError("Train IDs do not match between Model A and B.")

if not np.array_equal(val_a["id"].to_numpy(),
                      val_b_raw["id"].to_numpy()):
    raise ValueError("Validation IDs do not match between Model A and B.")

y_train_b = train_b_raw["is_fraud"]
y_val_b = val_b_raw["is_fraud"]

# Load saved v2 preprocessor
preprocessor = joblib.load(
    OUTPUT / "preprocessors" / "preprocessor_v2_raw_plus_fe.pkl"
)

# raw_fe files already contain cleaned and engineered features.
# Therefore, use only the final ColumnTransformer stage.
transformer = preprocessor.named_steps["transform"]

X_train_b = transformer.transform(train_b_raw)
X_val_b = transformer.transform(val_b_raw)

feature_names_b = transformer.get_feature_names_out()

X_train_b = pd.DataFrame(
    X_train_b,
    columns=feature_names_b
)

X_val_b = pd.DataFrame(
    X_val_b,
    columns=feature_names_b
)



# 5. Train and evaluate


def evaluate_model(name, X_train, y_train, X_val, y_val):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    model = DecisionTreeClassifier(**PARAMS)
    model.fit(X_train, y_train)

    train_probs = model.predict_proba(X_train)[:, 1]
    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds = model.predict(X_val)

    print(f"Number of features: {X_train.shape[1]}")
    print(f"Train ROC-AUC:      {roc_auc_score(y_train, train_probs):.6f}")
    print(f"Validation ROC-AUC: {roc_auc_score(y_val, val_probs):.6f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, val_preds))

    print("\nClassification Report:")
    print(classification_report(y_val, val_preds, digits=4))

    return model, roc_auc_score(y_val, val_probs)



# 6. Compare A and B


model_a, auc_a = evaluate_model(
    "MODEL A - v1_raw",
    X_train_a, y_train_a,
    X_val_a, y_val_a
)

model_b, auc_b = evaluate_model(
    "MODEL B - v2_raw_plus_fe",
    X_train_b, y_train_b,
    X_val_b, y_val_b
)

print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)
print(f"Model A (54 features): {auc_a:.6f}")
print(f"Model B (70 features): {auc_b:.6f}")
print(f"Difference (B - A):   {auc_b - auc_a:+.6f}")

if auc_b > auc_a:
    print("v2_raw_plus_fe performs better on validation ROC-AUC.")
elif auc_b < auc_a:
    print("v1_raw performs better on validation ROC-AUC.")
else:
    print("Both feature sets have equal validation ROC-AUC.")

print("\nExperiment completed.")
print("No existing model or submission file was overwritten.")
