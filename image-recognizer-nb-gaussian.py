import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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

def fit_nb(X, Y, num_classes=10):
    """
    Trains Gaussian Naive Bayes by computing per-class, per-pixel
    mean (mu) and variance (sigma^2). The 'naive' assumption is that
    pixels are conditionally independent given the class label — this is
    false for images (adjacent pixels are spatially correlated even within
    a class), but the model still works and avoids a full covariance matrix.
    """
    n_features, m_samples = X.shape

    phis = np.zeros(num_classes)
    mus = np.zeros((n_features, num_classes))
    sigmas_sq = np.zeros((n_features, num_classes))

    for k in range(num_classes):
        X_k = X[:, Y == k]
        m_k = X_k.shape[1]

        # P(y=k): prior probability of this class
        phis[k] = m_k / m_samples

        # Per-pixel mean for class k: shape (n_features,)
        mus[:, k] = np.mean(X_k, axis=1)

        # Per-pixel variance for class k: shape (n_features,)
        # Add epsilon to prevent log(0) and division by zero
        sigmas_sq[:, k] = np.var(X_k, axis=1) + 1e-9

    return phis, mus, sigmas_sq


def predict_nb(X, phis, mus, sigmas_sq, num_classes=10):
    """
    Predicts class via log-likelihood under the Gaussian NB model:
      log P(y=k | x) ∝ log P(y=k) + Σ_i log N(x_i; μ_{i,k}, σ²_{i,k})

    Expanding log N: -0.5 * log(2π σ²) - 0.5 * (x - μ)² / σ²
    The constant -0.5 * log(2π) is shared across all classes and dropped.
    """
    n_features, m_samples = X.shape
    scores = np.zeros((num_classes, m_samples))

    for k in range(num_classes):
        mu_k = mus[:, k:k+1]           # (n_features, 1)
        sigma_sq_k = sigmas_sq[:, k:k+1]  # (n_features, 1)

        # Log-prior
        log_prior = np.log(phis[k])

        # Per-pixel Gaussian log-likelihood, summed across all pixels
        # shape of each term: (n_features, m_samples) -> sum -> (m_samples,)
        log_likelihood = np.sum(
            -0.5 * np.log(sigma_sq_k)
            - 0.5 * (X - mu_k) ** 2 / sigma_sq_k,
            axis=0
        )

        scores[k, :] = log_prior + log_likelihood

    return np.argmax(scores, axis=0)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


def test_prediction(index, phis, mus, sigmas_sq):
    current_image = x_train[:, index:index+1]
    prediction = predict_nb(current_image, phis, mus, sigmas_sq)[0]
    label = int(y_train[index])
    print("Prediction: ", prediction)
    print("Label: ", label)

    img = current_image.reshape(28, 28) * 255
    plt.gray()
    plt.imshow(img, interpolation='nearest')
    plt.show()


print("--- Training Naive Bayes ---")
phis, mus, sigmas_sq = fit_nb(x_train, y_train)

print("\n--- Testing Naive Bayes ---")
train_preds = predict_nb(x_train, phis, mus, sigmas_sq)
print(f"NB Training Accuracy: {get_accuracy(train_preds, y_train) * 100:.2f}%")

dev_preds = predict_nb(x_dev, phis, mus, sigmas_sq)
print(f"NB Dev/Validation Accuracy: {get_accuracy(dev_preds, y_dev) * 100:.2f}%")

for i in range(8):
    test_prediction(i, phis, mus, sigmas_sq)
