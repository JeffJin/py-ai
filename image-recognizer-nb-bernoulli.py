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

print('x shape = ', x_train.shape)


def fit_nb(X, Y, num_classes=10):
    """
    Trains Bernoulli Naive Bayes using the MLE formulas from the notes,
    generalized from binary (y in {0,1}) to 10 classes (y in {0,...,9}).

    Notes formula -> 10-class generalization:
      phi_k       = P(y=k)          = # samples with y=k / total
      phi_j|y=k   = P(x_j=1 | y=k) = # samples with x_j=1 AND y=k / # samples with y=k

    Pixels are binarized (0 or 1) to satisfy the Bernoulli assumption.
    Laplace smoothing (+1 / +2) prevents log(0) for pixels never seen in a class.
    """
    n_features, m_samples = X.shape
    X_bin = (X > 0.5).astype(float) # key for bernoulli nb implementation

    phis = np.zeros(num_classes)                         # phi_k: P(y=k)
    phi_j_given_k = np.zeros((n_features, num_classes))  # phi_{j|y=k}: P(x_j=1 | y=k)

    for k in range(num_classes):
        X_k = X_bin[:, Y == k]
        m_k = X_k.shape[1]

        # phi_k = |{y=k}| / n
        phis[k] = m_k / m_samples

        # phi_{j|y=k} = sum(x_j=1 AND y=k) / sum(y=k)  with Laplace smoothing
        phi_j_given_k[:, k] = (np.sum(X_k, axis=1) + 1) / (m_k + 2)

    return phis, phi_j_given_k


def predict_nb(X, phis, phi_j_given_k, num_classes=10):
    """
    Prediction uses Bayes rule (from notes), extended to 10 classes:

      P(y=k | x) ∝ P(y=k) * prod_j P(x_j | y=k)

    In log space to avoid underflow:
      log P(y=k | x) ∝ log phi_k + sum_j [ x_j*log(phi_{j|y=k}) + (1-x_j)*log(1-phi_{j|y=k}) ]
    """
    X_bin = (X > 0.5).astype(float)
    scores = np.zeros((num_classes, X_bin.shape[1]))

    for k in range(num_classes):
        phi_k = phi_j_given_k[:, k:k+1]  # P(x_j=1 | y=k), shape (n_features, 1)

        log_prior = np.log(phis[k])

        # Bernoulli log-likelihood summed across all pixels
        log_likelihood = np.sum(
            X_bin * np.log(phi_k) + (1 - X_bin) * np.log(1 - phi_k),
            axis=0
        )

        scores[k, :] = log_prior + log_likelihood

    return np.argmax(scores, axis=0)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


def test_prediction(index, phis, phi_j_given_k):
    current_image = x_train[:, index:index+1]
    prediction = predict_nb(current_image, phis, phi_j_given_k)[0]
    label = int(y_train[index])
    print("Prediction: ", prediction)
    print("Label: ", label)

    img = current_image.reshape(28, 28) * 255
    plt.gray()
    plt.imshow(img, interpolation='nearest')
    plt.show()


print("--- Training Naive Bayes (Bernoulli) ---")
phis, phi_j_given_k = fit_nb(x_train, y_train)

print("\n--- Testing Naive Bayes (Bernoulli) ---")
train_preds = predict_nb(x_train, phis, phi_j_given_k)
print(f"NB Training Accuracy: {get_accuracy(train_preds, y_train) * 100:.2f}%")

dev_preds = predict_nb(x_dev, phis, phi_j_given_k)
print(f"NB Dev/Validation Accuracy: {get_accuracy(dev_preds, y_dev) * 100:.2f}%")

for i in range(8):
    test_prediction(i, phis, phi_j_given_k)
