import numpy as np


# SimpleSVM: a minimal SMO-based solver for the (soft-margin) SVM dual problem.
#
# Mathematical background (dual formulation):
#   Maximize W(\alpha) = \sum_i \alpha_i - 1/2 \sum_i\sum_j \alpha_i \alpha_j y_i y_j K(x_i,x_j)
#   subject to 0 <= \alpha_i <= C,  and  \sum_i \alpha_i y_i = 0
#
# Where K(x_i,x_j) is the kernel (here linear: K(x,x') = x^T x').
# The decision function for a new point x is:
#   f(x) = \sum_i \alpha_i y_i K(x_i, x) + b
# and the predicted label is sign(f(x)).
#
# This class implements a very small subset of Platt's SMO algorithm:
# - pick a violating example i, randomly select second index j
# - compute bounds L/H for alpha_j from box constraints and equality constraint
# - compute eta (related to second derivative) and update alpha_j,
#   then clip to [L,H], update alpha_i to satisfy the equality constraint,
#   and recompute the bias term b using the two candidate b values.

class SimpleSVM:
  def __init__(self, C=1.0, tol=1e-3, max_passes=5):
    # C: regularization parameter (box constraint on alphas)
    # tol: numerical tolerance for KKT violation checks
    # max_passes: number of passes with no alpha changes before stopping
    self.C = C
    self.tol = tol
    self.max_passes = max_passes

  def fit(self, X, y):
    # X: (n_samples, n_features)
    # y: labels in {-1, +1}, shape (n_samples,)
    # Note: we store data and labels to compute kernel products in updates.
    n_samples, n_features = X.shape
    self.alpha = np.zeros(n_samples)   # dual variables \alpha_i
    self.b = 0                         # bias term
    self.X = X
    self.y = y

    passes = 0
    # main SMO loop: iterate until convergence (no changes for max_passes passes)
    while passes < self.max_passes:
      num_changed_alphas = 0
      for i in range(n_samples):
        # f_i = \sum_k \alpha_k y_k K(x_k, x_i) + b
        f_i = np.sum(self.alpha * self.y * self._kernel(self.X, self.X[i])) + self.b
        # E_i = f_i - y_i (prediction error for i)
        E_i = f_i - self.y[i]

        # Check if example i violates KKT conditions enough to consider updating.
        # Conditions derived from complementary slackness:
        #  - if y_i * f_i < 1 and alpha_i < C  => can increase alpha_i
        #  - if y_i * f_i > 1 and alpha_i > 0  => can decrease alpha_i
        # Here a tolerance 'tol' is used; the code uses an equivalent check on E_i.
        if (self.y[i] * E_i < -self.tol and self.alpha[i] < self.C) or \
                (self.y[i] * E_i > self.tol and self.alpha[i] > 0):

          # choose j != i (simple heuristic: random choice)
          j = np.random.choice([idx for idx in range(n_samples) if idx != i])
          f_j = np.sum(self.alpha * self.y * self._kernel(self.X, self.X[j])) + self.b
          E_j = f_j - self.y[j]

          # save current values
          old_ai, old_aj = self.alpha[i], self.alpha[j]

          # Compute L and H, the feasible interval for new alpha_j given box
          # constraints 0 <= alpha <= C and equality constraint on \sum alpha_i y_i.
          if self.y[i] != self.y[j]:
            # If labels differ:
            #   L = max(0, alpha_j - alpha_i)
            #   H = min(C, C + alpha_j - alpha_i)
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
          else:
            # If labels equal:
            #   L = max(0, alpha_i + alpha_j - C)
            #   H = min(C, alpha_i + alpha_j)
            L = max(0, self.alpha[i] + self.alpha[j] - self.C)
            H = min(self.C, self.alpha[i] + self.alpha[j])

          # If L == H no move is possible
          if L == H: continue

          # Eta encodes curvature: typical SMO sets
          #   eta_typical = K_ii + K_jj - 2 K_ij
          # This implementation computes eta = 2 K_ij - K_ii - K_jj = -eta_typical
          # and then checks eta >= 0 to skip non-descending directions.
          # The update below uses this sign convention so algebra matches.
          eta = 2 * self._kernel(self.X[i:i + 1], self.X[j:j + 1]) - \
                self._kernel(self.X[i:i + 1], self.X[i:i + 1]) - \
                self._kernel(self.X[j:j + 1], self.X[j:j + 1])

          # If eta >= 0 (i.e. eta_typical <= 0) the objective is not suitable for
          # the simple greedy update (numerical instability / non-positive curvature).
          if eta >= 0: continue

          # SMO update for alpha_j (using this file's eta sign):
          #   alpha_j_new = alpha_j_old - y_j (E_i - E_j) / eta
          # With the alternative sign convention above, this matches the usual
          # alpha_j += y_j (E_i - E_j) / (K_ii + K_jj - 2K_ij).
          self.alpha[j] -= (self.y[j] * (E_i - E_j)) / eta

          # clip to [L, H] to enforce box constraints
          self.alpha[j] = np.clip(self.alpha[j], L, H)

          # if change is too small, skip
          if abs(self.alpha[j] - old_aj) < 1e-5: continue

          # Update alpha_i to preserve equality constraint \sum alpha y = 0:
          #   alpha_i_new = alpha_i_old + y_i y_j (old_aj - alpha_j_new)
          self.alpha[i] += self.y[i] * self.y[j] * (old_aj - self.alpha[j])

          # Recompute bias term b. Two candidate biases come from the KKT
          # conditions for the two updated examples; average them for stability.
          # Formulas derive from solving for b in y_i (f_i + b) = 1 (for support
          # vectors on the margin) with the updated alphas.
          b1 = self.b - E_i - self.y[i] * (self.alpha[i] - old_ai) * self._kernel(self.X[i:i + 1],
                                                                                  self.X[i:i + 1]) - \
               self.y[j] * (self.alpha[j] - old_aj) * self._kernel(self.X[i:i + 1], self.X[j:j + 1])
          b2 = self.b - E_j - self.y[i] * (self.alpha[i] - old_ai) * self._kernel(self.X[i:i + 1],
                                                                                  self.X[j:j + 1]) - \
               self.y[j] * (self.alpha[j] - old_aj) * self._kernel(self.X[j:j + 1], self.X[j:j + 1])

          # Choose the average of the two candidate b's (common SMO heuristic)
          self.b = (b1 + b2) / 2
          num_changed_alphas += 1

      # If no alphas changed in this full pass, increment passes; otherwise reset
      passes = passes + 1 if num_changed_alphas == 0 else 0

  def _kernel(self, x1, x2):
    # Linear kernel (dot product): K(x, x') = x^T x'
    # Accepts (m, d) and (n, d) arrays and returns (m, n) Gram matrix.
    return x1 @ x2.T

  def predict(self, X):
    # Compute decision values f(x) = \sum_i alpha_i y_i K(x_i, x) + b for each
    # row in X and return sign(f).
    # Note: internal storage uses training examples self.X; kernel with new X
    # returns shape (n_train, n_test), so sum over training axis and transpose.
    return np.sign(np.sum(self.alpha * self.y * self._kernel(self.X, X), axis=1) + self.b)
