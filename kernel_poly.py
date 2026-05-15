import numpy as np


class KernelLMS:
  def __init__(self, alpha=0.01, epochs=100):
    self.alpha = alpha
    self.epochs = epochs
    self.beta = None
    self.X_train = None

  def _poly_kernel_matrix(self, X1, X2):
    # Computes K(x, z) = (x.T @ z)^2 for all pairs
    return (X1 @ X2.T) ** 2

  def train(self, X, y):
    self.X_train = X
    n_samples = X.shape[0]

    # 1. Initialize beta to zeros
    self.beta = np.zeros(n_samples)

    # 2. Pre-compute the n x n Kernel Matrix
    K = self._poly_kernel_matrix(X, X)

    # 3. Training Loop (Batch Update)
    for _ in range(self.epochs):
      # Prediction: y_hat = K @ beta
      predictions = K @ self.beta

      # Error: y - y_hat
      error = y - predictions

      # Update: beta = beta + alpha * error
      self.beta += self.alpha * error

  def predict(self, X_test):
    # To predict, we need the kernel between test data and training data
    # K_test shape: (n_test, n_train)
    K_test = self._poly_kernel_matrix(X_test, self.X_train)

    # Prediction is the weighted sum of similarities
    return K_test @ self.beta