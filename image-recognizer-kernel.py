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
    X_T = X.T
    Z_T = Z.T
    X_sub = X_T[:1000, :]
    sub_dist_sq = np.linalg.norm(X_sub[:, np.newaxis] - X_sub, axis=2) ** 2
    sigma_sq = np.median(sub_dist_sq)
    dist_sq = np.linalg.norm(X_T[:, np.newaxis] - Z_T, axis=2) ** 2
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


    # 3. np.linalg.solve(A, b) => solves the equation Ax = b for x, where A is a square matrix.
    # self.beta = np.linalg.solve(K + self.lam * np.eye(n_samples), y)
    # 3. Training Loop (Batch Update)
    for _ in range(self.epochs):
      # Prediction: y_hat = K @ beta
      predictions = K @ self.beta

      # Error: y - y_hat
      error = y - predictions

      # Update: beta = beta + alpha * error
      self.beta += self.alpha * error
      # self.beta += self.alpha * (error - self.lam * self.beta)

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








