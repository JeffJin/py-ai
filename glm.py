import numpy as np


def update_step(theta, X, y, alpha, model_type='linear'):
  # Linear predictor
  m = X.shape[0]
  eta = X @ theta

  # Apply the specific response function (g-inverse)
  if model_type == 'linear':
    mu = eta
  elif model_type == 'logistic':
    mu = 1 / (1 + np.exp(-eta))
  elif model_type == 'poisson':
    mu = np.exp(eta)

  # THE UNIVERSAL GRADIENT
  gradient = (1 / m) * X.T @ (mu - y)

  return theta - alpha * gradient

def update_step_softmax(Theta, X, y_one_hot, alpha, lam):
  """
  Theta: (n_features, n_classes)
  X: (m_samples, n_features)
  y_one_hot: (m_samples, n_classes) - The 0/1 labels
  """
  m = X.shape[0]

  # 1. Calculate raw scores (logits): (m, k)
  z = X @ Theta

  # 2. Softmax for probabilities: (m, k)
  # Subtract max for numerical stability on your 3090
  z_shifted = z - np.max(z, axis=1, keepdims=True)
  probs = np.exp(z_shifted) / np.sum(np.exp(z_shifted), axis=1, keepdims=True)

  # 3. The Universal Gradient: (n, k)
  # This is (Predictions - Truth) dot X
  # Using X.T (n, m) @ (probs - y_one_hot) (m, k) gives the correct (n, k) shape
  gradient = (1 / m) * (X.T @ (probs - y_one_hot))

  # 4. Apply update with Regularization (Weight Decay)
  # Theta_new = Theta_old - alpha * (Gradient + Penalty)
  return Theta - alpha * (gradient + lam * Theta)