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


def texts_to_binary_bow(texts, vocab):
    """Binary word presence features, shape (n_features, m_samples)."""
    X = np.zeros((len(vocab), len(texts)))
    for i, text in enumerate(texts):
        for word in set(tokenize(text)):
            if word in vocab:
                X[vocab[word], i] = 1.0
    return X


split = int(0.8 * len(data))
train_texts = data['text'][:split].values
dev_texts   = data['text'][split:].values
y_train, y_dev = y[:split], y[split:]

vocab   = build_vocab(train_texts, MAX_FEATURES)
x_train = texts_to_binary_bow(train_texts, vocab)  # (n_features, m_train)
x_dev   = texts_to_binary_bow(dev_texts,   vocab)  # (n_features, m_dev)

print(f'x_train shape = {x_train.shape}')


def fit_nb(X, Y):
    """
    Bernoulli Naive Bayes — direct application of the notes' MLE formulas
    to spam classification (binary: y in {0: ham, 1: spam}).

    Notes formula:
      phi_{j|y=1} = sum(x_j=1 AND y=1) / sum(y=1)   [fraction of spam emails containing word j]
      phi_{j|y=0} = sum(x_j=1 AND y=0) / sum(y=0)   [fraction of ham emails containing word j]
      phi_y       = sum(y=1) / n                      [prior: fraction of emails that are spam]

    Laplace smoothing (+1 / +2) prevents log(0) for words never seen in one class.
    """
    X_spam = X[:, Y == 1]
    X_ham  = X[:, Y == 0]
    m_spam, m_ham = X_spam.shape[1], X_ham.shape[1]

    phi_y      = m_spam / (m_spam + m_ham)
    phi_j_spam = (np.sum(X_spam, axis=1) + 1) / (m_spam + 2)
    phi_j_ham  = (np.sum(X_ham,  axis=1) + 1) / (m_ham  + 2)

    return phi_y, phi_j_spam, phi_j_ham


def predict_nb(X, phi_y, phi_j_spam, phi_j_ham):
    """
    Applies the notes' prediction rule extended to 2 classes:
      log P(y=1|x) ∝ log phi_y     + Σ_j [x_j*log(phi_j_spam) + (1-x_j)*log(1-phi_j_spam)]
      log P(y=0|x) ∝ log(1-phi_y)  + Σ_j [x_j*log(phi_j_ham)  + (1-x_j)*log(1-phi_j_ham)]
    """
    phi_spam = phi_j_spam[:, None]  # (n_features, 1) for broadcasting
    phi_ham  = phi_j_ham[:, None]

    log_score_spam = np.log(phi_y) + np.sum(
        X * np.log(phi_spam) + (1 - X) * np.log(1 - phi_spam), axis=0
    )
    log_score_ham = np.log(1 - phi_y) + np.sum(
        X * np.log(phi_ham) + (1 - X) * np.log(1 - phi_ham), axis=0
    )

    return (log_score_spam > log_score_ham).astype(int)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


print("--- Training Bernoulli Naive Bayes ---")
phi_y, phi_j_spam, phi_j_ham = fit_nb(x_train, y_train)

print("\n--- Testing Bernoulli Naive Bayes ---")
train_preds = predict_nb(x_train, phi_y, phi_j_spam, phi_j_ham)
print(f"NB Training Accuracy: {get_accuracy(train_preds, y_train) * 100:.2f}%")

dev_preds = predict_nb(x_dev, phi_y, phi_j_spam, phi_j_ham)
print(f"NB Dev/Validation Accuracy: {get_accuracy(dev_preds, y_dev) * 100:.2f}%")

# Show the top words most indicative of spam vs ham
top_n = 10
spam_ratio = np.log(phi_j_spam / phi_j_ham)
vocab_inv = {i: w for w, i in vocab.items()}
top_spam_words = [vocab_inv[i] for i in np.argsort(spam_ratio)[-top_n:][::-1]]
top_ham_words  = [vocab_inv[i] for i in np.argsort(spam_ratio)[:top_n]]
print(f"\nTop {top_n} spam-indicative words: {top_spam_words}")
print(f"Top {top_n} ham-indicative words:  {top_ham_words}")
