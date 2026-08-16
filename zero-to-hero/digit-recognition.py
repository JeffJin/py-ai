import numpy as np
import pandas as pd

# Load data
data = pd.read_csv("../data/image-recognizer-train.csv")
data = np.array(data)

# Use a subset first, because full kernel methods are O(n^2) memory and O(n^3) solve
data = data[:5000]
np.random.shuffle(data)

# Split
data_dev = data[:1000]
data_train = data[1000:]

y_dev = data_dev[:, 0].astype(int)
X_dev = data_dev[:, 1:] / 255.0

y_train = data_train[:, 0].astype(int)
X_train = data_train[:, 1:] / 255.0

print(X_train.shape, y_train.shape)
print(X_dev.shape, y_dev.shape)


def rbf_kernel(X, Z, sigma=None):
    """
    X: (n_samples_x, n_features)
    Z: (n_samples_z, n_features)
    returns K: (n_samples_x, n_samples_z)
    """

    X_norms = np.sum(X ** 2, axis=1, keepdims=True)
    Z_norms = np.sum(Z ** 2, axis=1, keepdims=True).T

    dist_sq = X_norms + Z_norms - 2 * X @ Z.T
    dist_sq = np.maximum(dist_sq, 0.0)

    if sigma is None:
        # Median heuristic on a subset
        subset = X[:1000]
        subset_norms = np.sum(subset ** 2, axis=1, keepdims=True)
        subset_dist_sq = subset_norms + subset_norms.T - 2 * subset @ subset.T
        sigma = np.sqrt(np.median(subset_dist_sq[subset_dist_sq > 0]))

    K = np.exp(-dist_sq / (2 * sigma ** 2))
    return K, sigma


class RBFKernelClassifier:
    def __init__(self, lam=1e-2, sigma=None):
        self.lam = lam
        self.sigma = sigma
        self.X_train = None
        self.B = None  # shape: (n_train, 10)

    def fit(self, X, y):
        self.X_train = X
        n = X.shape[0]

        K, self.sigma = rbf_kernel(X, X, self.sigma)

        # One-hot labels: (n, 10)
        Y = np.zeros((n, 10))
        Y[np.arange(n), y] = 1.0

        # Kernel ridge regression:
        # (K + λI) B = Y
        self.B = np.linalg.solve(K + self.lam * np.eye(n), Y)

    def predict_scores(self, X):
        K_test, _ = rbf_kernel(X, self.X_train, self.sigma)
        return K_test @ self.B

    def predict(self, X):
        scores = self.predict_scores(X)
        return np.argmax(scores, axis=1)


def accuracy(preds, y):
    return np.mean(preds == y)


model = RBFKernelClassifier(lam=1e-2)
model.fit(X_train, y_train)

train_preds = model.predict(X_train)
dev_preds = model.predict(X_dev)

print(f"sigma: {model.sigma:.4f}")
print(f"Train accuracy: {accuracy(train_preds, y_train) * 100:.2f}%")
print(f"Dev accuracy: {accuracy(dev_preds, y_dev) * 100:.2f}%")