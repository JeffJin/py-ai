import numpy as np


class SimpleSVM:
  def __init__(self, C=1.0, tol=1e-3, max_passes=5):
    self.C = C
    self.tol = tol
    self.max_passes = max_passes

  def fit(self, X, y):
    n_samples, n_features = X.shape
    self.alpha = np.zeros(n_samples)
    self.b = 0
    self.X = X
    self.y = y

    passes = 0
    while passes < self.max_passes:
      num_changed_alphas = 0
      for i in range(n_samples):
        # Calculate error for sample i
        f_i = np.sum(self.alpha * self.y * self._kernel(self.X, self.X[i])) + self.b
        E_i = f_i - self.y[i]

        if (self.y[i] * E_i < -self.tol and self.alpha[i] < self.C) or \
                (self.y[i] * E_i > self.tol and self.alpha[i] > 0):

          # Select j != i randomly
          j = np.random.choice([idx for idx in range(n_samples) if idx != i])
          f_j = np.sum(self.alpha * self.y * self._kernel(self.X, self.X[j])) + self.b
          E_j = f_j - self.y[j]

          # Save old alphas
          old_ai, old_aj = self.alpha[i], self.alpha[j]

          # Compute bounds L and H
          if self.y[i] != self.y[j]:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
          else:
            L = max(0, self.alpha[i] + self.alpha[j] - self.C)
            H = min(self.C, self.alpha[i] + self.alpha[j])

          if L == H: continue

          # Eta: second derivative of objective
          eta = 2 * self._kernel(self.X[i:i + 1], self.X[j:j + 1]) - \
                self._kernel(self.X[i:i + 1], self.X[i:i + 1]) - \
                self._kernel(self.X[j:j + 1], self.X[j:j + 1])

          if eta >= 0: continue

          # Update alpha j
          self.alpha[j] -= (self.y[j] * (E_i - E_j)) / eta
          self.alpha[j] = np.clip(self.alpha[j], L, H)

          if abs(self.alpha[j] - old_aj) < 1e-5: continue

          # Update alpha i
          self.alpha[i] += self.y[i] * self.y[j] * (old_aj - self.alpha[j])

          # Compute b
          b1 = self.b - E_i - self.y[i] * (self.alpha[i] - old_ai) * self._kernel(self.X[i:i + 1],
                                                                                  self.X[i:i + 1]) - \
               self.y[j] * (self.alpha[j] - old_aj) * self._kernel(self.X[i:i + 1], self.X[j:j + 1])
          b2 = self.b - E_j - self.y[i] * (self.alpha[i] - old_ai) * self._kernel(self.X[i:i + 1],
                                                                                  self.X[j:j + 1]) - \
               self.y[j] * (self.alpha[j] - old_aj) * self._kernel(self.X[j:j + 1], self.X[j:j + 1])

          self.b = (b1 + b2) / 2
          num_changed_alphas += 1

      passes = passes + 1 if num_changed_alphas == 0 else 0

  def _kernel(self, x1, x2):
    # Linear kernel for simplicity; replace with RBF if needed
    return x1 @ x2.T

  def predict(self, X):
    return np.sign(np.sum(self.alpha * self.y * self._kernel(self.X, X), axis=1) + self.b)
