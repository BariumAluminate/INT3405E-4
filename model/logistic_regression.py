import math
import numpy as np
from pathlib import Path

def _sigmoid(n):
    clipped = np.clip(n ,-700, None) #Không chặn dưới thì bị tràn số, đã thử nghiệm
    return 1.0 / (1.0 + np.exp(-clipped))

#cross entropy for biary 
def _binary_cross_entropy(y_true, y_prob, eps=1e-15):
    true_prob = np.clip(y_true, eps, 1 - eps)
    prob = np.clip(y_prob, eps, 1 - eps)
    return -(true_prob * np.log(prob) + (1 - true_prob) * np.log(1 - prob))

#create mini batches for epoch
def _iter_minibatches(X, y, sample_weight, batch_size, shuffle, rng):
    if batch_size <= 0:
        raise ValueError("Batch size must be greater than 0!")
    if X.shape[0] != y.shape[0]:
        raise ValueError("Length of X and y is different!")
    if sample_weight is not None and sample_weight.shape[0] != X.shape[0]:
        raise ValueError("Length of sample weight must be equal to X")
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    if shuffle == True:
        generator = rng if rng is not None else np.random.default_rng()
        indices = generator.permutation(n_samples)
    for start in range(0, n_samples, batch_size):
        batch_idx = indices[start : start + batch_size]
        weight_batch = None if sample_weight is None else sample_weight[batch_idx]
        yield X[batch_idx], y[batch_idx], weight_batch


class logistic_regression:
    def __init__(
        self,
        learning_rate=0.05,
        reg_lambda=1e-4,
        n_epochs=30,
        batch_size=2048,
        class_weight=None,
        shuffle=True,
        random_state=42,
        dtype=np.float64,
    ):
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.class_weight = class_weight
        self.shuffle = shuffle
        self.random_state = random_state
        self.dtype = dtype
        self.rng = np.random.default_rng(random_state)

    def _compute_sample_weight(self, y):
        if self.class_weight is None:
            return None
        if isinstance(self.class_weight, dict):
            weights = np.ones_like(y, dtype=self.dtype)
            for label, factor in self.class_weight.items():
                weights[y == label] = factor
            return weights
        raise ValueError("class_weight must be None or a dictionary")
    
    def fit(self, X, y, sample_weight=None, X_val=None, y_val=None):
        X = np.asanyarray(X, dtype=self.dtype)
        y = np.asanyarray(y, dtype=self.dtype)

        _, n_features = X.shape
        sample_weight = self._compute_sample_weight(y)
        self.weights = np.zeros(n_features, dtype=self.dtype)
        self.bias = self.dtype(0.0)
        self.history_ = []

        for _ in range(self.n_epochs):
            for X_batch, y_batch, w_batch in _iter_minibatches(
                X, y, sample_weight, self.batch_size, self.shuffle,self.rng
            ):
                batch_size_actual = X_batch.shape[0]
                z_batch = X_batch @ self.weights + self.bias
                p_hat = _sigmoid(z_batch)
                error = p_hat - y_batch
                weighted_error = error if w_batch is None else w_batch * error
                grad_w = (X_batch.T @ weighted_error) / batch_size_actual
                grad_w = grad_w + self.reg_lambda * self.weights
                grad_b = np.sum(weighted_error) / batch_size_actual

                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b
            full_logits = X @ self.weights + self.bias
            full_probs = _sigmoid(full_logits)
            epoch_loss = _binary_cross_entropy(y, full_probs, sample_weight=sample_weight)
            self.history_.append(epoch_loss)
        return self

    def _check_is_fitted(self):
        if not (hasattr(self, "weights_") and hasattr(self, "bias_")):
            raise RuntimeError(
                "Model haven't been trained!"
            )

    def predict_proba(self, X):
        self._check_is_fitted()
        X_arr = np.asanyarray(X, dtype=self.dtype)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D, but get the shape {X_arr.shape}.")
        if X_arr.shape[1] != self.weights_.shape[0]:
            raise ValueError(
                f"The number of features in X ({X_arr.shape[1]}) does not match the model."
                f"({self.weights_.shape[0]})."
            )
        z = X_arr @ self.weights + self.bias
        return _sigmoid(z)

    def predict(self, X, threshold=0.5):
        pass

    def save(self, filepath):
        pass

    @classmethod
    def load(cls, filepath):
        pass

if __name__ == "__main__":
    print(_binary_cross_entropy(1, 0))