import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('./data/image-recognizer-train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
y_dev = data_dev[0]
# Keep raw integer pixel values (0-255) — these are the "word counts" in the multinomial model
x_dev = data_dev[1:n].astype(float)

data_train = data[1000:m].T
y_train = data_train[0]
x_train = data_train[1:n].astype(float)
_, m_train = x_train.shape

print('x shape = ', x_train.shape)


def fit_nb(X, Y, num_classes=10):
    """
    Multinomial Naive Bayes — directly applies the notes' MLE formula to images.

    Analogy to text: each image is a "document", each pixel position j is a "word",
    and the raw pixel intensity x_j is the "word count" (how many times word j appears).

    Notes MLE formula adapted to images:
      phi_{j|k} = total intensity of pixel j across all class-k images
                  ──────────────────────────────────────────────────────
                  total intensity of ALL pixels across all class-k images

    Numerator  = Σ_i x_j^(i) * 1{y^(i) = k}     (matches notes: count of word k in class)
    Denominator = Σ_i Σ_j x_j^(i) * 1{y^(i) = k} (matches notes: total words in class)

    Laplace smoothing: +1 numerator, +n_features denominator.
    """
    n_features, m_samples = X.shape

    phis          = np.zeros(num_classes)
    phi_j_given_k = np.zeros((n_features, num_classes))

    for k in range(num_classes):
        X_k = X[:, Y == k]
        m_k = X_k.shape[1]

        phis[k] = m_k / m_samples

        # Sum of each pixel's intensity across all class-k images — shape (n_features,)
        pixel_sums = np.sum(X_k, axis=1)

        # Total intensity across all pixels in all class-k images — scalar
        total_intensity = np.sum(X_k)

        phi_j_given_k[:, k] = (pixel_sums + 1) / (total_intensity + n_features)

    return phis, phi_j_given_k


def predict_nb(X, phis, phi_j_given_k, num_classes=10):
    """
    Multinomial log-likelihood (from notes):
      log P(x | y=k) = Σ_j x_j * log phi_{j|k}

    The multinomial coefficient (d! / Π x_j!) is the same for all classes and dropped.
    Full score:
      log P(y=k | x) ∝ log phi_k + Σ_j x_j * log phi_{j|k}

    Compare to Bernoulli NB:  Σ_j [ x_j*log(phi) + (1-x_j)*log(1-phi) ]   (binary x_j)
    Compare to Gaussian NB:   Σ_j [ -0.5*log(σ²) - 0.5*(x-μ)²/σ² ]        (continuous x_j)
    """
    scores = np.zeros((num_classes, X.shape[1]))

    for k in range(num_classes):
        log_phi_k = np.log(phi_j_given_k[:, k:k+1])  # (n_features, 1)

        log_prior = np.log(phis[k])

        # Multinomial log-likelihood: each pixel contributes x_j * log(phi_{j|k})
        log_likelihood = np.sum(X * log_phi_k, axis=0)  # (m_samples,)

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

    img = current_image.reshape(28, 28)
    plt.gray()
    plt.imshow(img, interpolation='nearest')
    plt.show()


print("--- Training Multinomial Naive Bayes ---")
phis, phi_j_given_k = fit_nb(x_train, y_train)

print("\n--- Testing Multinomial Naive Bayes ---")
train_preds = predict_nb(x_train, phis, phi_j_given_k)
print(f"NB Training Accuracy: {get_accuracy(train_preds, y_train) * 100:.2f}%")

dev_preds = predict_nb(x_dev, phis, phi_j_given_k)
print(f"NB Dev/Validation Accuracy: {get_accuracy(dev_preds, y_dev) * 100:.2f}%")

for i in range(8):
    test_prediction(i, phis, phi_j_given_k)
