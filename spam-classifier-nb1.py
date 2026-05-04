import numpy as np
import pandas as pd
import re
from collections import Counter

# Dataset: SMS Spam Collection
# Download: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
# Place at ./data/spam.csv (columns: v1=label, v2=text)
data = pd.read_csv('./data/spam.csv', encoding='latin-1')[['v1', 'v2']]
data.columns = ['label', 'text']
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
y = (data['label'] == 'spam').astype(int).values

MAX_FEATURES = 3000


def tokenize(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()


def build_vocab(texts, max_features):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    return {word: i for i, (word, _) in enumerate(counter.most_common(max_features))}


def texts_to_tf(texts, vocab):
    """Normalized word count (TF) features, shape (n_features, m_samples)."""
    X = np.zeros((len(vocab), len(texts)))
    for i, text in enumerate(texts):
        counts = Counter(tokenize(text))
        total = max(sum(counts.values()), 1)
        for word, count in counts.items():
            if word in vocab:
                X[vocab[word], i] = count / total
    return X


split = int(0.8 * len(data))
train_texts = data['text'][:split].values
dev_texts   = data['text'][split:].values
y_train, y_dev = y[:split], y[split:]

vocab   = build_vocab(train_texts, MAX_FEATURES)
x_train = texts_to_tf(train_texts, vocab)  # (n_features, m_train)
x_dev   = texts_to_tf(dev_texts,   vocab)  # (n_features, m_dev)

print(f'x_train shape = {x_train.shape}')


def fit_nb(X, Y):
    """
    Gaussian Naive Bayes — models each word's TF score as an independent Gaussian.
    The naive assumption (conditional pixel independence) is violated for text too:
    word co-occurrences within a class are real (e.g., 'free' and 'win' co-occur in spam),
    but the model still works well in practice.

    Per-class, per-word:
      mu_{j|k}      = E[x_j | y=k]   — mean TF score of word j in class k
      sigma^2_{j|k} = Var[x_j | y=k] — variance of TF score of word j in class k
    """
    n_features, m_samples = X.shape

    phis      = np.zeros(2)
    mus       = np.zeros((n_features, 2))
    sigmas_sq = np.zeros((n_features, 2))

    for k in range(2):
        X_k = X[:, Y == k]
        m_k = X_k.shape[1]

        phis[k]         = m_k / m_samples
        mus[:, k]       = np.mean(X_k, axis=1)
        sigmas_sq[:, k] = np.var(X_k, axis=1) + 1e-9

    return phis, mus, sigmas_sq


def predict_nb(X, phis, mus, sigmas_sq):
    """
    log P(y=k | x) ∝ log phi_k + Σ_j log N(x_j; mu_{j|k}, sigma^2_{j|k})

    Gaussian log N expanded (dropping the shared -0.5*log(2pi) constant):
      -0.5 * log(sigma^2) - 0.5 * (x - mu)^2 / sigma^2
    """
    scores = np.zeros((2, X.shape[1]))

    for k in range(2):
        mu_k       = mus[:, k:k+1]
        sigma_sq_k = sigmas_sq[:, k:k+1]

        log_prior = np.log(phis[k])
        log_likelihood = np.sum(
            -0.5 * np.log(sigma_sq_k) - 0.5 * (X - mu_k) ** 2 / sigma_sq_k,
            axis=0
        )
        scores[k, :] = log_prior + log_likelihood

    return np.argmax(scores, axis=0)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


print("--- Training Gaussian Naive Bayes ---")
phis, mus, sigmas_sq = fit_nb(x_train, y_train)

print("\n--- Testing Gaussian Naive Bayes ---")
train_preds = predict_nb(x_train, phis, mus, sigmas_sq)
print(f"NB Training Accuracy: {get_accuracy(train_preds, y_train) * 100:.2f}%")

dev_preds = predict_nb(x_dev, phis, mus, sigmas_sq)
print(f"NB Dev/Validation Accuracy: {get_accuracy(dev_preds, y_dev) * 100:.2f}%")
