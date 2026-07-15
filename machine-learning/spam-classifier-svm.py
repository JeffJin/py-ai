import numpy as np
import pandas as pd
import re
from collections import Counter

data = pd.read_csv('../data/spam.csv', encoding='latin-1')[['label', 'email']]
data.columns = ['label', 'text']
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
y = data['label'].values  # already 0/1 integers

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
    tokens = tokenize(text)
    counts = Counter(tokens)
    total = max(len(tokens), 1)
    for word, count in counts.items():
      if word in vocab:
        X[vocab[word], i] = count / total
  return X


split = int(0.8 * len(data))
train_texts = data['text'][:split].values
dev_texts = data['text'][split:].values
y_train, y_dev = y[:split], y[split:]

vocab = build_vocab(train_texts, MAX_FEATURES)
x_train = texts_to_tf(train_texts, vocab)  # (n_features, m_train)
x_dev = texts_to_tf(dev_texts, vocab)  # (n_features, m_dev)

print(f'Train: {x_train.shape[1]} samples | Dev: {x_dev.shape[1]} samples | Vocab: {len(vocab)} words')


# ─── Gaussian Naive Bayes ────────────────────────────────────────────────────

def fit_nb(X, Y):
  """
  Gaussian Naive Bayes over TF features.

  For each class k and feature j:
    mu_{j|k}  = mean TF score of word j in class k
    var_{j|k} = variance of TF score of word j in class k
  """
  n_features, m = X.shape
  phis = np.zeros(2)
  mus = np.zeros((n_features, 2))
  sigmas_sq = np.zeros((n_features, 2))

  for k in range(2):
    # [(word_index in vocab, mail_index)] with label spam or ham
    # shape (n_features, m_k): rows=word indices, cols=samples with label k
    X_k = X[:, Y == k]
    phis[k] = X_k.shape[1] / m
    mus[:, k] = np.nan_to_num(np.mean(X_k, axis=1), nan=0.0) # average value of TF for spam and ham
    sigmas_sq[:, k] = np.nan_to_num(np.var(X_k, axis=1), nan=0.0) + 1e-9

  return phis, mus, sigmas_sq


def predict_nb(X, phis, mus, sigmas_sq):
  """
  log P(y=k|x) ∝ log phi_k + Σ_j log N(x_j; mu_{j|k}, var_{j|k})

  Gaussian log-likelihood (constant -0.5*log(2π) dropped, same for both classes):
    -0.5 * log(var) - 0.5 * (x - mu)^2 / var
  """
  scores = np.zeros((2, X.shape[1]))
  for k in range(2):
    mu_k = mus[:, k:k + 1]
    var_k = sigmas_sq[:, k:k + 1]
    scores[k] = np.log(phis[k]) + np.sum(
      -0.5 * np.log(var_k) - 0.5 * (X - mu_k) ** 2 / var_k,
      axis=0
    )
  return np.argmax(scores, axis=0)


# ─── Linear SVM (SGD) ────────────────────────────────────────────────────────

def fit_svm(X, Y, C=1.0, lr=0.01, epochs=100):
  """
  Linear SVM trained with SGD on hinge loss + L2 regularization.

  Objective: min_{w,b}  0.5*||w||^2 + C * Σ max(0, 1 - y_i*(w·x_i + b))

  Labels are converted to {-1, +1} for the margin formulation.

  SGD update per sample (x_i, y_i):
    if y_i*(w·x_i + b) >= 1  (correctly classified with margin):
        w ← w - lr * w          (L2 shrinkage only)
    else (within margin or misclassified):
        w ← w - lr * (w - C*y_i*x_i)
        b ← b + lr * C * y_i
  """
  n_features, m = X.shape
  w = np.zeros(n_features)
  b = 0.0
  labels = 2 * Y - 1  # {0,1} → {-1,+1}

  for epoch in range(epochs):
    perm = np.random.permutation(m)
    for i in perm:
      xi, yi = X[:, i], labels[i]
      margin = yi * (w.dot(xi) + b)
      if margin >= 1:
        w -= lr * w
      else:
        w -= lr * (w - C * yi * xi)
        b += lr * C * yi

  return w, b


def predict_svm(X, w, b):
  return (w.dot(X) + b >= 0).astype(int)


# ─── Evaluation ──────────────────────────────────────────────────────────────

def get_accuracy(predictions, Y):
  return np.sum(predictions == Y) / Y.size


def report(name, train_preds, dev_preds, y_train, y_dev):
  print(f"\n{'─' * 40}")
  print(f"  {name}")
  print(f"{'─' * 40}")
  print(f"  Train accuracy : {get_accuracy(train_preds, y_train) * 100:.2f}%")
  print(f"  Dev accuracy   : {get_accuracy(dev_preds, y_dev) * 100:.2f}%")


print("\n--- Gaussian Naive Bayes ---")
phis, mus, sigmas_sq = fit_nb(x_train, y_train)
report(
  "Gaussian Naive Bayes",
  predict_nb(x_train, phis, mus, sigmas_sq),
  predict_nb(x_dev, phis, mus, sigmas_sq),
  y_train, y_dev,
)

print("\n--- Linear SVM (SGD, C=1) ---")
w, b = fit_svm(x_train, y_train, C=1.0, lr=0.01, epochs=30)
report(
  "Linear SVM",
  predict_svm(x_train, w, b),
  predict_svm(x_dev, w, b),
  y_train, y_dev,
)
