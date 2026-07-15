import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# --- [YOUR EXACT DATA LOADING CODE HERE] ---
data = pd.read_csv('../data/image-recognizer-train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
y_dev = data_dev[0]
x_dev = data_dev[1:n] / 255

data_train = data[1000:m].T
y_train = data_train[0]
x_train = data_train[1:n] / 255
_, m_train = x_train.shape


# ------------------------------------------
def fit_qda(X, Y, num_classes=10):
  """Trains the GDA by calculating the Mean and Covariance for all classes."""
  n_features, m_samples = X.shape

  phis = np.zeros(num_classes)
  mus = np.zeros((n_features, num_classes))
  Sigmas = []

  print("1. Calculating Means (mu) and Priors (phi)...")
  for k in range(num_classes):
    # Filter data for the specific digit 'k'
    # Y == 1 generates a new True/False array comparing every label to 1.
    # X[:, Y == 1] grabs all the images of ones.
    X_k = X[:, Y == k]
    m_k = X_k.shape[1]

    # P(y=k): What percentage of the data is this digit?
    phis[k] = m_k / m_samples

    # Calculate the "Average Image" for this digit
    mu_k = np.mean(X_k, axis=1, keepdims=True)
    mus[:, k:k + 1] = mu_k

    # Center the data by subtracting the correct class mean from every sample
    # shifting data so the class mean is at zero.
    X_k_centered = X_k - mu_k
    # This is E[ (Z-E[Z])(Z-E[Z])^T ]
    # The .dot() with the transpose handles the outer products for all samples
    # Vectorized Covariance calculation: (X - mu)(X - mu)^T / m
    sigma_k = (1 / m_k) * X_k_centered.dot(X_k_centered.T)

    # CRITICAL FIX: Add a tiny value to the diagonal to prevent a Singular Matrix
    epsilon = 1e-4
    sigma_k += epsilon * np.eye(n_features)
    Sigmas.append(sigma_k)

  return phis, mus, Sigmas

def predict_qda(X, phis, mus, Sigmas, num_classes=10):
  n_features, m_samples = X.shape
  # Store the log-likelihood for each class for each sample
  scores = np.zeros((num_classes, m_samples))

  for k in range(num_classes):
    mu_k = mus[:, k:k + 1]
    Sigma_k = Sigmas[k]
    Sigma_inv = np.linalg.inv(Sigma_k)

    # 1. The Log-Determinant term (measures the 'volume' of the class)
    # np.linalg.slogdet is more stable than log(det)
    _, logdet = np.linalg.slogdet(Sigma_k)

    # 2. The Prior term
    log_phi = np.log(phis[k])

    # 3. The Quadratic term: -0.5 * (x - mu)^T * Sigma_inv * (x - mu)
    # This is the 'Mahalanobis Distance' part
    X_centered = X - mu_k  # Shape: (n_features, m_samples)

    # Efficiently calculate (x-mu)^T * Sigma_inv * (x-mu) for all samples:
    # Step A: Sigma_inv * (X - mu)
    dist_part = Sigma_inv.dot(X_centered)  # (n_features, m_samples)
    # Step B: (X - mu) * dist_part (element-wise) then sum down columns
    mahalanobis_dist = np.sum(X_centered * dist_part, axis=0)  # (m_samples,)
    # mahalanobis_dist = np.diag(np.dot(X_centered.T, dist_part)) # very inefficient way, we only need the diagonal entries

    # Full QDA Score Formula:
    scores[k, :] = -0.5 * logdet - 0.5 * mahalanobis_dist + log_phi

  return np.argmax(scores, axis=0)

def get_accuracy(predictions, Y):
  return np.sum(predictions == Y) / Y.size

# --- Execute QDA
print("--- Training QDA ---")
# Training is just a single function call! No gradient descent.
phis_qda, mus_qda, Sigmas = fit_qda(x_train, y_train)

print("\n--- Testing QDA ---")
train_preds_qda = predict_qda(x_train, phis_qda, mus_qda, Sigmas)
print(f"QDA Training Accuracy: {get_accuracy(train_preds_qda, y_train) * 100:.2f}%")

dev_preds_qda = predict_qda(x_dev, phis_qda, mus_qda, Sigmas)
print(f"QDA Dev/Validation Accuracy: {get_accuracy(dev_preds_qda, y_dev) * 100:.2f}%")