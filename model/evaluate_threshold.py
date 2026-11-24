
import sys
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "DataSolve"))

OUTPUT = ROOT / "DataSolve" / "output"
MODEL_OUTPUT = ROOT / "model" / "output"

# Load validation data
val_raw = pd.read_csv(OUTPUT / "raw_fe" / "validation_fe.csv")
y_val = val_raw["is_fraud"]

# Load preprocessor and transform validation data
preprocessor = joblib.load(
    OUTPUT / "preprocessors" / "preprocessor_v2_raw_plus_fe.pkl"
)
transformer = preprocessor.named_steps["transform"]

X_val = transformer.transform(val_raw)

# Load Model B
model = joblib.load(MODEL_OUTPUT / "decision_tree_best.pkl")

# IMPORTANT:
# decision_tree_best.pkl currently contains Model A.
# So we train Model B here using the same parameters as the comparison.
# This makes the script independent of the existing saved model.

train_raw = pd.read_csv(OUTPUT / "raw_fe" / "train_fe.csv")
y_train = train_raw["is_fraud"]
X_train = transformer.transform(train_raw)

from sklearn.tree import DecisionTreeClassifier

model_b = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=6,
    min_samples_leaf=50,
    min_samples_split=50,
    class_weight="balanced",
    random_state=42,
)

model_b.fit(X_train, y_train)

# Predict probabilities once
val_probs = model_b.predict_proba(X_val)[:, 1]

print(f"Validation ROC-AUC: {roc_auc_score(y_val, val_probs):.6f}")

thresholds = [0.3, 0.4, 0.5, 0.6]

print("\nTHRESHOLD COMPARISON")
print("-" * 85)
print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'False Pos.':<13}"
    f"{'Fraud caught':<13}"
)

for threshold in thresholds:
    predictions = (val_probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    recall = recall_score(
        y_val, predictions, zero_division=0
    )
    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    print(
        f"{threshold:<12.2f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
        f"{fp:<13}"
        f"{tp:<13}"
    )

print("\nExperiment completed. No existing model or submission was changed.")
