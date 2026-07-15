import numpy as np
import pandas as pd

data = pd.read_csv('../data/image-recognizer-train.csv')
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

  def train(self, X, y):
    self.X_train = X
    n_samples = X.shape[1]
    n_classes = 10  # For MNIST

    # 1. One-Hot Encode y: (n_samples,) -> (n_samples, 10)
    # This turns '3' into [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y_one_hot = np.eye(n_classes)[y.astype(int)]

    # 2. Initialize Beta as a Matrix: (n_samples, 10)
    self.beta = np.zeros((n_samples, n_classes))

    # 3. Pre-compute K: (n_samples, n_samples)
    K = self._rbf_kernel_matrix(X, X)

    # 4. Training Loop
    for _ in range(self.epochs):
      self.beta -= self.kernel_softmax_update(self.beta, K, y_one_hot, self.alpha, self.lam)

  def kernel_softmax_update(beta, K, y_one_hot, alpha, lam):
    # 1. Predictions using the precalculated K
    z = K @ beta  # K is (m, m), beta is (m, k)

    # 2. Softmax (Probability)
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    probs = np.exp(z_shifted) / np.sum(np.exp(z_shifted), axis=1, keepdims=True)

    # 3. The Update (Gradient in Kernel Space)
    # The 'error' is (probs - y_one_hot)
    # We update beta directly by the error
    gradient = (1 / m) * (probs - y_one_hot)

    # 4. Apply update with Kernel Regularization
    return beta - alpha * (gradient + lam * beta)

  def predict(self, X_test):
    K_test = self._rbf_kernel_matrix(X_test, self.X_train)

    # Raw scores: (n_test, 10)
    scores = K_test @ self.beta

    # Pick the index of the highest score (the predicted digit)
    return np.argmax(scores, axis=1)

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








