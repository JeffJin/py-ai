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

# GDA needs to invert an (n_features x n_features) matrix — keep vocabulary small
MAX_FEATURES = 500


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


def fit_gda(X, Y):
    """
    Binary GDA (y in {0: ham, 1: spam}).

    Parameters:
      phi   = P(y=1)         — prior probability of spam
      mu_0  = E[x | y=0]    — mean feature vector for ham
      mu_1  = E[x | y=1]    — mean feature vector for spam
      Sigma = shared covariance matrix (GDA assumption)
    """
    n_features, m_samples = X.shape

    phi  = np.mean(Y)
    mu_0 = np.mean(X[:, Y == 0], axis=1, keepdims=True)
    mu_1 = np.mean(X[:, Y == 1], axis=1, keepdims=True)

    # Center each sample by its class mean, then compute shared Sigma
    X_centered = np.where(Y == 1, X - mu_1, X - mu_0)
    Sigma = (1 / m_samples) * X_centered.dot(X_centered.T)
    Sigma += 1e-4 * np.eye(n_features)  # prevent singular matrix

    return phi, mu_0, mu_1, Sigma


def predict_gda(X, phi, mu_0, mu_1, Sigma):
    """
    Shared Sigma collapses Gaussian math into linear scores (same trick as image GDA):
      score_k = W_k^T * x + b_k
    Predict spam (1) if score_1 > score_0.
    """
    Sigma_inv = np.linalg.inv(Sigma)

    W_1 = Sigma_inv.dot(mu_1)
    b_1 = -0.5 * mu_1.T.dot(Sigma_inv).dot(mu_1) + np.log(phi)

    W_0 = Sigma_inv.dot(mu_0)
    b_0 = -0.5 * mu_0.T.dot(Sigma_inv).dot(mu_0) + np.log(1 - phi)

    score_1 = (W_1.T.dot(X) + b_1).flatten()
    score_0 = (W_0.T.dot(X) + b_0).flatten()

    return (score_1 > score_0).astype(int)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


print("--- Training GDA ---")
phi, mu_0, mu_1, Sigma = fit_gda(x_train, y_train)

print("\n--- Testing GDA ---")
train_preds = predict_gda(x_train, phi, mu_0, mu_1, Sigma)
print(f"GDA Training Accuracy: {get_accuracy(train_preds, y_train) * 100:.2f}%")

dev_preds = predict_gda(x_dev, phi, mu_0, mu_1, Sigma)
print(f"GDA Dev/Validation Accuracy: {get_accuracy(dev_preds, y_dev) * 100:.2f}%")
