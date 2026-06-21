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

def logsumexp(log_x, axis):
  # Numerically stable log(sum(exp(log_x))) along `axis`
  c = np.max(log_x, axis=axis, keepdims=True)
  return (c + np.log(np.sum(np.exp(log_x - c), axis=axis, keepdims=True))).squeeze(axis)

class GMM:
  # Gaussian Mixture Model with diagonal covariances, fit by Expectation-Maximization.
  # Diagonal covariance keeps each component to O(d) parameters instead of O(d^2),
  # which is essential for the 784-dim pixel vectors (a full 784x784 covariance would
  # be singular with only a few hundred samples per digit).
  def __init__(self, n_components=5, max_iters=50, reg=1e-2, tol=1e-3):
    self.n_components = n_components
    self.max_iters = max_iters
    self.reg = reg          # variance floor; also guards against collapsing components
    self.tol = tol
    self.weights = None     # (K,)   mixing coefficients
    self.means = None       # (K, d) component means
    self.variances = None   # (K, d) diagonal variances

  def _log_gaussian(self, X):
    # Per-sample, per-component log density of a diagonal Gaussian.
    # X: (N, d) -> returns (N, K)
    # log N(x; mu, var) = -0.5 * sum_d [ log(2*pi*var_d) + (x_d - mu_d)^2 / var_d ]
    log_norm = -0.5 * np.sum(np.log(2 * np.pi * self.variances), axis=1)  # (K,)
    # (N, K) squared Mahalanobis distance using broadcasting over components
    diff = X[:, None, :] - self.means[None, :, :]                          # (N, K, d)
    maha = np.sum(diff ** 2 / self.variances[None, :, :], axis=2)          # (N, K)
    return log_norm[None, :] - 0.5 * maha

  def _log_resp(self, X):
    # E-step core: weighted log densities and their log-normalizer per sample.
    log_prob = self._log_gaussian(X) + np.log(self.weights)[None, :]       # (N, K)
    log_norm = logsumexp(log_prob, axis=1)                                 # (N,)
    return log_prob, log_norm

  def fit(self, X):
    # X: (N, d)
    N, d = X.shape
    K = self.n_components

    # Initialize: random samples as means, global variance, uniform weights
    rng_idx = np.random.choice(N, K, replace=False)
    self.means = X[rng_idx].copy()
    global_var = np.var(X, axis=0) + self.reg
    self.variances = np.tile(global_var, (K, 1))
    self.weights = np.full(K, 1.0 / K)

    prev_ll = -np.inf
    for it in range(self.max_iters):
      # E-step: responsibilities r[i,k] = p(component k | x_i)
      log_prob, log_norm = self._log_resp(X)
      resp = np.exp(log_prob - log_norm[:, None])                          # (N, K)

      # M-step: update weights, means, variances from soft assignments
      Nk = resp.sum(axis=0) + 1e-12                                        # (K,)
      self.weights = Nk / N
      self.means = (resp.T @ X) / Nk[:, None]                              # (K, d)
      # E[(x-mu)^2] under responsibilities = E[x^2] - mu^2
      mean_sq = (resp.T @ (X ** 2)) / Nk[:, None]                          # (K, d)
      self.variances = mean_sq - self.means ** 2 + self.reg

      # Convergence check on total log-likelihood
      ll = np.sum(log_norm)
      if abs(ll - prev_ll) < self.tol * abs(prev_ll):
        break
      prev_ll = ll

    return self

  def score_samples(self, X):
    # Log-likelihood log p(x) of each sample under the mixture. X: (N, d) -> (N,)
    _, log_norm = self._log_resp(X)
    return log_norm

def get_accuracy(predictions, Y):
  return np.sum(predictions == Y) / Y.size

# --- Generative classifier: fit one GMM per digit, classify by Bayes rule ---
# argmax_k [ log p(x | class k) + log P(class k) ]
print("--- Training GMM via EM (one model per digit) ---")
num_classes = 10
X_train = x_train.T  # (N, d) — GMM works sample-major
models = []
log_priors = np.zeros(num_classes)
for k in range(num_classes):
    print(f"  Fitting GMM for class {k}...")
    X_k = X_train[y_train == k]
    log_priors[k] = np.log(X_k.shape[0] / X_train.shape[0])
    gmm = GMM(n_components=5)
    gmm.fit(X_k)
    models.append(gmm)

def predict_all(X, models, log_priors):
    X = X.T  # (N, d)
    # (N, 10) class-conditional log-likelihood plus class log-prior
    scores = np.column_stack([m.score_samples(X) for m in models]) + log_priors[None, :]
    return np.argmax(scores, axis=1)

print("\n--- Testing GMM classifier ---")
train_preds = predict_all(x_train, models, log_priors)
print(f"GMM Training Accuracy: {get_accuracy(train_preds, y_train) * 100:.2f}%")

dev_preds = predict_all(x_dev, models, log_priors)
print(f"GMM Dev/Validation Accuracy: {get_accuracy(dev_preds, y_dev) * 100:.2f}%")
