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

n_input = len(vocab)  # 3000
n1, n2  = 256, 64     # hidden layer sizes
n_out   = 2           # ham=0, spam=1


def init_params():
    # Xavier/Glorot initialization — same as image-recognizer-mlp.py
    w1 = np.random.randn(n1,    n_input) * np.sqrt(2.0 / (n_input + n1))
    w2 = np.random.randn(n2,    n1)      * np.sqrt(2.0 / (n1 + n2))
    w3 = np.random.randn(n_out, n2)      * np.sqrt(2.0 / (n2 + n_out))
    b1 = np.zeros((n1,    1))
    b2 = np.zeros((n2,    1))
    b3 = np.zeros((n_out, 1))
    return w1, b1, w2, b2, w3, b3


def ReLU(z):
    return np.maximum(0, z)


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=0)


def forward_propagation(w1, b1, w2, b2, w3, b3, X):
    z1 = w1.dot(X) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = ReLU(z2)
    z3 = w3.dot(a2) + b3
    a3 = softmax(z3)
    return z1, a1, z2, a2, z3, a3


def one_hot(Y, num_classes=2):
    oh = np.zeros((Y.shape[0], num_classes))
    oh[np.arange(Y.shape[0]), Y] = 1
    return oh.T


def deriv_ReLU(z):
    return (z > 0).astype(int)


def back_propagation(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    m = Y.shape[0]
    onehot_Y = one_hot(Y)

    dZ3 = A3 - onehot_Y
    dW3 = 1/m * dZ3.dot(A2.T)
    db3 = 1/m * np.sum(dZ3, axis=1, keepdims=True)

    dZ2 = W3.T.dot(dZ3) * deriv_ReLU(Z2)
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3


def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 -= alpha * dW1;  b1 -= alpha * db1
    W2 -= alpha * dW2;  b2 -= alpha * db2
    W3 -= alpha * dW3;  b3 -= alpha * db3
    return W1, b1, W2, b2, W3, b3


def get_predictions(A3):
    return np.argmax(A3, axis=0)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, alpha, iterations, epsilon=1e-5):
    W1, b1, W2, b2, W3, b3 = init_params()
    prev_loss = float('inf')

    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_propagation(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = back_propagation(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)

        m = Y.shape[0]
        current_loss = -1/m * np.sum(one_hot(Y) * np.log(A3 + 1e-8))

        if abs(prev_loss - current_loss) < epsilon:
            print(f"Converged at iteration {i}")
            break

        prev_loss = current_loss

        if i % 100 == 0:
            print(f"Iteration {i} | Loss: {current_loss:.4f} | "
                  f"Train Accuracy: {get_accuracy(get_predictions(A3), Y) * 100:.2f}%")

    return W1, b1, W2, b2, W3, b3


def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_propagation(W1, b1, W2, b2, W3, b3, X)
    return get_predictions(A3)


W1, b1, W2, b2, W3, b3 = gradient_descent(x_train, y_train, alpha=0.10, iterations=500)

dev_predictions = make_predictions(x_dev, W1, b1, W2, b2, W3, b3)
print(f"\nMLP Dev/Validation Accuracy: {get_accuracy(dev_predictions, y_dev) * 100:.2f}%")
