import math
import numpy as np

def _sigmoid(n):
    clipped = np.clip(n ,-700, None) #Không chặn dưới thì bị tràn số, đã thử nghiệm
    return 1.0 / (1.0 + np.exp(-clipped))

#cross entropy for biary 
def _binary_cross_entropy(y_true, y_prob, eps=1e-15):
    true_prob = np.clip(y_true, eps, 1 - eps)
    prob = np.clip(y_prob, eps, 1 - eps)
    return -(true_prob * np.log(prob) + (1 - true_prob) * np.log(1 - prob))

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

    def fit(self, X, y, sample_weight=None, X_val=None, y_val=None):
        pass

    def predict_proba(self, X):
        pass

    def predict(self, X, threshold=0.5):
        pass

    def save(self, filepath):
        pass

    @classmethod
    def load(cls, path):
        pass

if __name__ == "__main__":
    print(_binary_cross_entropy(1, 0))