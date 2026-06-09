import numpy as np


class MultiClassPrimalSVM:
  def __init__(self, num_classes=10, C=1.0, learning_rate=0.05, epochs=50, batch_size=64):
    self.num_classes = num_classes
    self.C = C  # Regularization scaling hyperparameter
    self.lr = learning_rate
    self.epochs = epochs
    self.batch_size = batch_size

  def fit(self, X, y):
    """
    X shape: (784, m_samples) -> Matching your MLP data orientation
    y shape: (m_samples,)
    """
    n_features, m = X.shape
    # Initialize weights (shape: 10 x 784) and biases (shape: 10 x 1)
    self.W = np.zeros((self.num_classes, n_features))
    self.b = np.zeros((self.num_classes, 1))

    prev_loss = float('inf')

    for epoch in range(self.epochs):
      # Shuffle at the start of each epoch
      permutation = np.random.permutation(m)
      X_shuffled = X[:, permutation]
      y_shuffled = y[permutation]

      epoch_loss = 0
      num_batches = 0

      for start in range(0, m, self.batch_size):
        X_batch = X_shuffled[:, start:start + self.batch_size]
        y_batch = y_shuffled[start:start + self.batch_size]
        batch_m = y_batch.shape[1] if len(y_batch.shape) > 1 else y_batch.shape[0]

        # 1. Forward Pass: Compute scores for all 10 classes
        # Z shape: (10, batch_size)
        Z = self.W.dot(X_batch) + self.b

        # Extract the scores of the correct classes
        # Using advanced indexing to get Z[y_batch[i], i]
        correct_class_scores = Z[y_batch, np.arange(batch_m)]

        # 2. Compute Margins: Z_j - Z_correct + 1
        # this is the key difference between SVM and Linear Regression,
        # keeping the minimum safety margin at 1
        margins = Z - correct_class_scores + 1.0
        margins[y_batch, np.arange(batch_m)] = 0  # Zero out the correct class spots

        # Max(0, margin)
        loss_mask = margins > 0
        batch_loss = np.sum(margins[loss_mask]) / batch_m

        # Add L2 Regularization component to loss
        l2_loss = 0.5 * (1.0 / m) * np.sum(self.W ** 2)
        epoch_loss += (self.C * batch_loss) + l2_loss
        num_batches += 1

        # 3. Backward Pass (Subgradient Calculation)
        # dZ tracks indicators for margin violations
        dZ = np.zeros(Z.shape)
        dZ[loss_mask] = 1.0
        # Correct class updates gather the sum of negative violation counts
        # Because dZ[loss_mask] = 1.0 sets a penalty of 1.0 across all rows that violated
        # the margin, we need a fast, vectorized way to jump straight to the correct class row
        # for each image, add up all the mistakes made on that specific image,
        # and assign the balancing negative penalty. This indexing trick allows you to
        # update the correct class for thousands of images simultaneously without
        # introducing a single slow Python for loop.
        dZ[y_batch, np.arange(batch_m)] = -np.sum(dZ, axis=0)

        # Scale gradient by batch size and parameter C
        dZ = (self.C / batch_m) * dZ

        # Gradients with respect to weights and bias (plus L2 weight decay)
        dW = dZ.dot(X_batch.T) + (1.0 / m) * self.W
        db = np.sum(dZ, axis=1, keepdims=True)

        # 4. Parameter Updates
        self.W -= self.lr * dW
        self.b -= self.lr * db

      current_loss = epoch_loss / num_batches

      if epoch % 10 == 0:
        preds = self.predict(X)
        accuracy = np.sum(preds == y) / y.size
        print(f"Epoch {epoch} | Total Multi-SVM Loss: {current_loss:.4f} | Training Acc: {accuracy:.4f}")

    return self.W, self.b

  def predict(self, X):
    """ X shape: (784, any_m) """
    scores = self.W.dot(X) + self.b
    return np.argmax(scores, axis=0)