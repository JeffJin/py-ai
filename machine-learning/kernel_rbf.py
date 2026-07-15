import numpy as np
import pandas as pd

data = pd.read_csv('../data/image-recognizer-train.csv')
data = np.array(data)[0:2000]
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:2000].T
y_dev = data_dev[0]
x_dev = data_dev[1:n] / 255

data_train = data[2000:m].T
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

      ## Stochastic update for one random image 'i'
      # i = np.random.randint(n_samples)
      # error_i = y[i] - (K[i] @ self.beta)
      #
      # # The m (n_samples) appears here to scale the regularization correctly
      # self.beta[i] += self.alpha * (error_i - n_samples * self.lam * self.beta[i])

  def predict(self, X_test):
    # To predict, we need the kernel between test data and training data
    # K_test shape: (n_test, n_train)
    K_test = self._rbf_kernel_matrix(X_test, self.X_train)

    # Prediction is the weighted sum of similarities
    return K_test @ self.beta

def get_accuracy(predictions, Y):
  return np.sum(predictions == Y) / Y.size

kms = KernelLMS()

# --- Execute the GDA ---
print("--- Training Kernel LMS ---")
# Training is just a single function call! No gradient descent.
kms.train(x_train, y_train)

print("\n--- Testing Kernel LMS ---")
train_preds = np.round(kms.predict(x_train)).astype(int)
print(f"Kernel LMS Training Accuracy: {get_accuracy(train_preds, y_train) * 100:.2f}%")

dev_preds = kms.predict(x_dev)
print(f"Kernel LMS Dev/Validation Accuracy: {get_accuracy(dev_preds, y_dev) * 100:.2f}%")








