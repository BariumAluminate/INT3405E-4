
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# 1. DECISION TREE IMPLEMENTED FROM SCRATCH

class DecisionTreeScratch:
    def __init__(
        self,
        max_depth=6,
        min_samples_split=50,
        min_samples_leaf=100,
        class_weight="balanced",
        max_bins=32,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.max_bins = max_bins
        self.tree_ = None

    def _gini(self, y, weights):
        total = weights.sum()
        if total == 0:
            return 0

        p = np.array([
            weights[y == 0].sum() / total,
            weights[y == 1].sum() / total,
        ])
        return 1 - np.sum(p ** 2)

    def _best_split(self, X, y, weights):
        n_samples, n_features = X.shape
        parent_gini = self._gini(y, weights)

        best_gain = 0
        best_split = None

        positions = np.linspace(
            self.min_samples_leaf,
            n_samples - self.min_samples_leaf,
            self.max_bins,
            dtype=int,
        )
        positions = np.unique(positions)

        for feature in range(n_features):
            order = np.argsort(X[:, feature])
            values = X[order, feature]
            labels = y[order]
            w = weights[order]

            # Include actual feature values at sampled positions
            # and both endpoints. This preserves binary splits (0 vs 1).
            candidate_values = np.unique(np.concatenate([
                values[positions - 1],
                values[positions],
                [values[0], values[-1]],
            ]))

            if len(candidate_values) < 2:
                continue

            thresholds = (
                candidate_values[:-1] + candidate_values[1:]
            ) / 2.0

            # Convert thresholds into actual split positions.
            split_positions = np.searchsorted(
                values, thresholds, side="right"
            )

            valid = (
                (split_positions >= self.min_samples_leaf)
                & (split_positions <= n_samples - self.min_samples_leaf)
            )

            split_positions = split_positions[valid]
            thresholds = thresholds[valid]

            if len(split_positions) == 0:
                continue

            # Remove duplicate split positions.
            unique_idx = np.unique(split_positions, return_index=True)[1]
            split_positions = split_positions[unique_idx]
            thresholds = thresholds[unique_idx]

            cum_pos = np.cumsum(w * (labels == 1))
            cum_neg = np.cumsum(w * (labels == 0))

            left_pos = cum_pos[split_positions - 1]
            left_neg = cum_neg[split_positions - 1]

            right_pos = cum_pos[-1] - left_pos
            right_neg = cum_neg[-1] - left_neg

            left_weight = left_pos + left_neg
            right_weight = right_pos + right_neg
            total_weight = left_weight + right_weight

            left_gini = 1 - (
                (left_pos / left_weight) ** 2
                + (left_neg / left_weight) ** 2
            )

            right_gini = 1 - (
                (right_pos / right_weight) ** 2
                + (right_neg / right_weight) ** 2
            )

            child_gini = (
                left_weight * left_gini
                + right_weight * right_gini
            ) / total_weight

            gains = parent_gini - child_gini
            idx = np.argmax(gains)

            if gains[idx] > best_gain:
                best_gain = gains[idx]
                best_split = (feature, thresholds[idx])

        return best_split

    def _build_tree(self, X, y, weights, depth):
        total = weights.sum()
        probability = weights[y == 1].sum() / total

        leaf = {"probability": probability}

        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(y) < 2 * self.min_samples_leaf
            or len(np.unique(y)) == 1
        ):
            return leaf

        split = self._best_split(X, y, weights)
        if split is None:
            return leaf

        feature, threshold = split
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        if (
            left_mask.sum() < self.min_samples_leaf
            or right_mask.sum() < self.min_samples_leaf
        ):
            return leaf

        return {
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(
                X[left_mask], y[left_mask],
                weights[left_mask], depth + 1
            ),
            "right": self._build_tree(
                X[right_mask], y[right_mask],
                weights[right_mask], depth + 1
            ),
        }

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values.")

        if not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError("Training labels must contain both 0 and 1.")

        if self.class_weight == "balanced":
            counts = np.bincount(y, minlength=2)
            class_weights = len(y) / (2 * counts)
            weights = class_weights[y]
        elif self.class_weight is None:
            weights = np.ones(len(y))
        else:
            weights = np.array([
                self.class_weight.get(int(label), 1.0)
                for label in y
            ])

        self.n_features_in_ = X.shape[1]
        self.tree_ = self._build_tree(X, y, weights, 0)
        return self

    def _predict_one(self, row, node):
        while "feature" in node:
            if row[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["probability"]

    def predict_proba(self, X):
        if self.tree_ is None:
            raise RuntimeError("Model has not been fitted.")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Input features do not match training data.")

        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values.")

        probabilities = np.array([
            self._predict_one(row, self.tree_) for row in X
        ])

        return np.column_stack((1 - probabilities, probabilities))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)


# 2. PATHS AND CONFIGURATION

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "DataSolve" / "output"
MODEL_DIR = ROOT / "model" / "output"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "DataSolve"))

TARGET = "is_fraud"
ID = "id"

BEST_PARAMS = {
    "max_depth": 6,
    "min_samples_leaf": 100,
    "min_samples_split": 50,
    "class_weight": "balanced",
    "max_bins": 256,
}


# 3. LOAD AND TRANSFORM DATA

print("Loading datasets...")

train_df = pd.read_csv(DATA_DIR / "raw_fe" / "train_fe.csv")
val_df = pd.read_csv(DATA_DIR / "raw_fe" / "validation_fe.csv")

y_train = train_df[TARGET].to_numpy()
y_val = val_df[TARGET].to_numpy()

preprocessor = joblib.load(
    DATA_DIR / "preprocessors" / "preprocessor_v2_raw_plus_fe.pkl"
)
transformer = preprocessor.named_steps["transform"]

X_train = transformer.transform(train_df)
X_val = transformer.transform(val_df)

feature_names = transformer.get_feature_names_out()

X_train = pd.DataFrame(X_train, columns=feature_names)
X_val = pd.DataFrame(X_val, columns=feature_names)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)



# 4. TRAIN MODEL


print("\nTraining Decision Tree from scratch...")
print("Parameters:", BEST_PARAMS)

model = DecisionTreeScratch(**BEST_PARAMS)
model.fit(X_train, y_train)

print("Training completed!")



# 5. EVALUATE MODEL


train_probs = model.predict_proba(X_train)[:, 1]
val_probs = model.predict_proba(X_val)[:, 1]

print("\n===== CUSTOM DECISION TREE =====")
print("Train ROC-AUC:", roc_auc_score(y_train, train_probs))
print("Validation ROC-AUC:", roc_auc_score(y_val, val_probs))

val_preds = model.predict(X_val)

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, val_preds))

print("\nClassification Report:")
print(classification_report(y_val, val_preds, zero_division=0))



# 6. SAVE MODEL

model_path = MODEL_DIR / "decision_tree_best.pkl"
model.save(model_path)

print("\nModel saved at:", model_path)


# 7. CREATE KAGGLE SUBMISSION


print("\nLoading Kaggle test data...")

test_df = pd.read_csv(DATA_DIR / "raw_fe" / "test_fe.csv")
test_ids = test_df[ID]

X_test = transformer.transform(test_df)
X_test = pd.DataFrame(X_test, columns=feature_names)

if X_test.isnull().any().any():
    raise ValueError("Test data contains missing values!")

test_probs = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    ID: test_ids,
    TARGET: test_probs,
})

RESULT_DIR = ROOT / "decision_tree_result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

submission_path = RESULT_DIR / "submission.csv"
submission.to_csv(submission_path, index=False)

print("\n===== SUBMISSION COMPLETED =====")
print(submission.head())
print("Submission shape:", submission.shape)
print("Submission saved at:", submission_path)
