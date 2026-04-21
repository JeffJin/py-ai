import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# --- [YOUR EXACT DATA LOADING CODE HERE] ---
data = pd.read_csv('./data/image-recognizer-train.csv')
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

def fit_gda(X, Y, num_classes=10):
  """Trains the GDA by calculating the Mean and Covariance for all classes."""
  n_features, m_samples = X.shape

  phis = np.zeros(num_classes)
  mus = np.zeros((n_features, num_classes))

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
    mus[:, k:k + 1] = np.mean(X_k, axis=1, keepdims=True)

  print("2. Calculating Shared Covariance Matrix (Sigma)...")
  # Center the data by subtracting the correct class mean from every sample
  X_centered = np.zeros_like(X)
  for i in range(m_samples):
    k = Y[i]
    X_centered[:, i:i + 1] = X[:, i:i + 1] - mus[:, k:k + 1]

  # Vectorized Covariance calculation: (X - mu)(X - mu)^T / m
  Sigma = (1 / m_samples) * X_centered.dot(X_centered.T)

  # CRITICAL FIX: Add a tiny value to the diagonal to prevent a Singular Matrix
  epsilon = 1e-4
  Sigma += epsilon * np.eye(n_features)

  return phis, mus, Sigma


def predict_gda(X, phis, mus, Sigma, num_classes=10):
  """Makes predictions by calculating the Log-Likelihood for each class."""
  # We only need to invert the Covariance matrix once!
  Sigma_inv = np.linalg.inv(Sigma)

  m_samples = X.shape[1]
  predictions = np.zeros((num_classes, m_samples))

  for k in range(num_classes):
    mu_k = mus[:, k:k + 1]

    # ---------------------------------------------------------
    # THE MAGIC TRICK: This simplifies into W * X + b!
    # Because Sigma is shared, the complex Gaussian math
    # mathematically collapses into a standard linear equation.
    # ---------------------------------------------------------
    W_k = Sigma_inv.dot(mu_k)
    b_k = -0.5 * mu_k.T.dot(Sigma_inv).dot(mu_k) + np.log(phis[k])

    # Calculate the linear score for all samples at once
    predictions[k, :] = W_k.T.dot(X) + b_k

  # The prediction is simply the class that scored the highest Likelihood
  return np.argmax(predictions, axis=0)


def get_accuracy(predictions, Y):
  return np.sum(predictions == Y) / Y.size


# --- Execute the GDA ---
print("--- Training GDA ---")
# Training is just a single function call! No gradient descent.
phis, mus, Sigma = fit_gda(x_train, y_train)

print("\n--- Testing GDA ---")
train_preds = predict_gda(x_train, phis, mus, Sigma)
print(f"Training Accuracy: {get_accuracy(train_preds, y_train) * 100:.2f}%")

dev_preds = predict_gda(x_dev, phis, mus, Sigma)
print(f"Dev/Validation Accuracy: {get_accuracy(dev_preds, y_dev) * 100:.2f}%")