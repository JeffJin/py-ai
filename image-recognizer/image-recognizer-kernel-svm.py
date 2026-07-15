import numpy as np
import pandas as pd

data = pd.read_csv('../data/image-recognizer-train.csv')
data = np.array(data)[0:5000]
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
y_dev = data_dev[0]
x_dev = data_dev[1:n] / 255

data_train = data[1000:m].T
y_train = data_train[0]
x_train = data_train[1:n] / 255
_, m_train = x_train.shape


class KernelSVM:
    def __init__(self, C=1.0, epochs=15, tol=1e-3):
        self.C = C
        self.epochs = epochs
        self.tol = tol
        self.alpha = None
        self.b = 0.0
        self.X_train = None
        self.y_train = None
        self.sigma_sq = None

    def _rbf_kernel_matrix(self, X, Z):
        # K(x, z) = exp(-||x - z||^2 / 2*sigma^2) for all pairs
        # X, Z are (n_features, n_samples); result is (n_X, n_Z)
        X_T = X.T
        Z_T = Z.T
        dist_sq = np.linalg.norm(X_T[:, np.newaxis] - Z_T, axis=2) ** 2
        return np.exp(-dist_sq / (2 * self.sigma_sq))

    def train(self, X, y):
        self.X_train = X
        # SVM needs +1/-1 labels, not 0/1
        self.y_train = np.where(y == 1, 1.0, -1.0)
        n = X.shape[1]
        self.alpha = np.zeros(n)
        self.b = 0.0

        # Estimate sigma from a subsample of training data
        X_T = X.T
        X_sub = X_T[:min(500, n), :]
        sub_dist_sq = np.linalg.norm(X_sub[:, np.newaxis] - X_sub, axis=2) ** 2
        self.sigma_sq = max(np.median(sub_dist_sq), 1e-8)

        # Precompute full kernel matrix K[i,j] = K(x_i, x_j): (n, n)
        K = self._rbf_kernel_matrix(X, X)

        # f_kernel tracks K @ (alpha * y) incrementally to avoid O(n^2) recompute
        # Decision value: f(x_i) = f_kernel[i] + b
        f_kernel = np.zeros(n)

        for epoch in range(self.epochs):
            num_changed = 0

            for i in range(n):
                E_i = f_kernel[i] + self.b - self.y_train[i]

                # KKT violation: misclassified (y*f < 1) or alpha at wrong bound
                kkt_violated = (
                    (self.y_train[i] * E_i < -self.tol and self.alpha[i] < self.C) or
                    (self.y_train[i] * E_i > self.tol and self.alpha[i] > 0)
                )
                if not kkt_violated:
                    continue

                # Pick a second alpha j to co-optimize (random heuristic)
                j = np.random.randint(0, n)
                while j == i:
                    j = np.random.randint(0, n)

                E_j = f_kernel[j] + self.b - self.y_train[j]

                alpha_i_old = self.alpha[i]
                alpha_j_old = self.alpha[j]

                # Box constraints: L <= alpha_j <= H
                # Derived from 0 <= alpha <= C and the equality constraint sum(alpha*y)=0
                if self.y_train[i] == self.y_train[j]:
                    L = max(0.0, alpha_i_old + alpha_j_old - self.C)
                    H = min(self.C, alpha_i_old + alpha_j_old)
                else:
                    L = max(0.0, alpha_j_old - alpha_i_old)
                    H = min(self.C, self.C + alpha_j_old - alpha_i_old)

                if L >= H:
                    continue

                # Second derivative of the dual objective along alpha_j
                # eta < 0 means objective is concave -> has a maximum
                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                # Unconstrained update for alpha_j, then clip to [L, H]
                self.alpha[j] -= self.y_train[j] * (E_i - E_j) / eta
                self.alpha[j] = np.clip(self.alpha[j], L, H)

                if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                    continue


                # alpha_i update is determined analytically from the equality constraint
                self.alpha[i] += self.y_train[i] * self.y_train[j] * (alpha_j_old - self.alpha[j])

                # Bias update: choose b so that f(x_i)=y_i or f(x_j)=y_j for support vectors
                delta_i = self.alpha[i] - alpha_i_old
                delta_j = self.alpha[j] - alpha_j_old
                b1 = (self.b - E_i
                      - self.y_train[i] * delta_i * K[i, i]
                      - self.y_train[j] * delta_j * K[i, j])
                b2 = (self.b - E_j
                      - self.y_train[i] * delta_i * K[i, j]
                      - self.y_train[j] * delta_j * K[j, j])

                if 0 < self.alpha[i] < self.C:
                    self.b = b1
                elif 0 < self.alpha[j] < self.C:
                    self.b = b2
                else:
                    self.b = (b1 + b2) / 2

                # Incrementally update f_kernel: O(n) instead of O(n^2) recompute
                f_kernel += (self.y_train[i] * delta_i * K[i, :]
                             + self.y_train[j] * delta_j * K[j, :])

                num_changed += 1

            if num_changed == 0:
                break

    def predict(self, X_test):
        # Returns raw decision scores (not thresholded) for one-vs-rest aggregation
        K_test = self._rbf_kernel_matrix(X_test, self.X_train)
        return K_test @ (self.alpha * self.y_train) + self.b


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


# --- One-vs-rest: train one KernelSVM per digit ---
print("--- Training Kernel SVM (one-vs-rest, SMO) ---")
num_classes = 10
models = []
for k in range(num_classes):
    print(f"  Training class {k}...")
    y_binary = (y_train == k).astype(float)   # +1 for class k, 0 otherwise
    clf = KernelSVM(C=1.0, epochs=15)
    clf.train(x_train, y_binary)
    models.append(clf)


def predict_all(X, models):
    # Each model returns a raw score; highest score wins (one-vs-rest)
    scores = np.column_stack([model.predict(X) for model in models])  # (n_samples, 10)
    return np.argmax(scores, axis=1)


print("\n--- Testing Kernel SVM ---")
train_preds = predict_all(x_train, models)
print(f"Kernel SVM Training Accuracy: {get_accuracy(train_preds, y_train) * 100:.2f}%")

dev_preds = predict_all(x_dev, models)
print(f"Kernel SVM Dev/Validation Accuracy: {get_accuracy(dev_preds, y_dev) * 100:.2f}%")
