import numpy as np
import pandas as pd

data = pd.read_csv('./data/image-recognizer-train.csv')
data = np.array(data)[0:5000]
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
y_dev = data_dev[0]
x_dev = data_dev[1:n] / 255

data_train = data[1000:m].T
y_train = data_train[0]
x_train = data_train[1:n] / 255
_, m_train = x_train.shape

class KernelLMS:
  def __init__(self, alpha=0.01, lam = 0.1, epochs=100):
    self.alpha = alpha
    self.epochs = epochs
    self.beta = None
    self.X_train = None
    self.lam = lam

  def _rbf_kernel_matrix(self, X, Z):
    # Computes K(x, z) = exp(-||x - z||^2 / sigma**2) for all pairs
    # Uses ||x-z||^2 = ||x||^2 + ||z||^2 - 2x^Tz to avoid a (n,n,d) intermediate tensor
    X_T = X.T
    Z_T = Z.T
    X_sub = X_T[:1000, :]
    X_sub_norms = np.sum(X_sub ** 2, axis=1)
    sub_dist_sq = X_sub_norms[:, None] + X_sub_norms[None, :] - 2 * X_sub @ X_sub.T
    sigma_sq = np.median(sub_dist_sq)
    X_norms = np.sum(X_T ** 2, axis=1)
    Z_norms = np.sum(Z_T ** 2, axis=1)
    dist_sq = X_norms[:, None] + Z_norms[None, :] - 2 * X_T @ Z_T.T
    return np.exp(-dist_sq / (2 * sigma_sq))

  def _linear_kernel_matrix(self, X, Z):
    # Computes K(x, z) = X @ T.T
    return X @ Z.T

  def train(self, X, y):
    self.X_train = X
    n_samples = X.shape[1]

    # 1. Initialize beta to zeros
    self.beta = np.zeros(n_samples)

    # 2. Pre-compute the n x n Kernel Matrix
    K = self._rbf_kernel_matrix(X, X)


    # Exact kernel ridge regression: (K + lam*I) @ beta = y
    # Gradient descent diverges unless alpha < 2/lambda_max(K)^2 (~1e-7 for n=4000)
    self.beta = np.linalg.solve(K + self.lam * np.eye(n_samples), y)

  def predict(self, X_test):
    # To predict, we need the kernel between test data and training data
    # K_test shape: (n_test, n_train)
    K_test = self._rbf_kernel_matrix(X_test, self.X_train)

    # Prediction is the weighted sum of similarities
    return K_test @ self.beta

def get_accuracy(predictions, Y):
  return np.sum(predictions == Y) / Y.size

# --- One-vs-all: train one KernelLMS per digit ---
print("--- Training Kernel LMS (one-vs-all) ---")
num_classes = 10
models = []
for k in range(num_classes):
    print(f"  Training class {k}...")
    y_binary = (y_train == k).astype(float)
    clf = KernelLMS()
    clf.train(x_train, y_binary)
    models.append(clf)

def predict_all(X, models):
    scores = np.column_stack([m.predict(X) for m in models])  # (n_samples, 10)
    return np.argmax(scores, axis=1)

print("\n--- Testing Kernel LMS ---")
train_preds = predict_all(x_train, models)
print(f"Kernel LMS Training Accuracy: {get_accuracy(train_preds, y_train) * 100:.2f}%")

dev_preds = predict_all(x_dev, models)
print(f"Kernel LMS Dev/Validation Accuracy: {get_accuracy(dev_preds, y_dev) * 100:.2f}%")








