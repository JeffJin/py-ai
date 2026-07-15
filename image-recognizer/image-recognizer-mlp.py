import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('../data/image-recognizer-train.csv')

data = np.array(data)
m, n = data.shape
# print(m, n, data[:, 0])
np.random.shuffle(data)

data_dev = data[0:1000].T
y_dev = data_dev[0]
x_dev = data_dev[1:n]
x_dev = x_dev / 255

data_train = data[1000:m].T
y_train = data_train[0]
x_train = data_train[1:n]
x_train = x_train / 255
_, m_train = x_train.shape

# print(y_train, m_train)
#     - The syntax `:` means you are selecting all rows in the array.
#     - The syntax `0` means you are selecting the first column in those rows.
print('x shape = ', x_train.shape)


def init_params(n1, n2, n3):
  """
  Initialize parameters using Xavier/Glorot initialization.
  n1: input size (784 for 28x28 images)
  n2: first hidden layer size (256)
  n3: second hidden layer size (128)
  """
  # He initialization for weights (correct for ReLU activations)
  # Formula: sqrt(2.0 / fan_in)
  w1 = np.random.randn(n2, n1) * np.sqrt(2.0 / n1)
  w2 = np.random.randn(n3, n2) * np.sqrt(2.0 / n2)
  w3 = np.random.randn(10, n3) * np.sqrt(2.0 / n3)

  # Initialize biases to zero (common practice)
  b1 = np.zeros((n2, 1))
  b2 = np.zeros((n3, 1))
  b3 = np.zeros((10, 1))

  return w1, b1, w2, b2, w3, b3


def ReLU(z):
  return np.maximum(0, z)

def sigmoid(z):
  return 1 / (1 + np.exp(-z))


def softmax(z):
  # exp(z2[0][0]) / (exp(z2[0][0]) + exp(z2[1][0]) + ... + exp(z2[9][0]))
  z_shifted = z - np.max(z, axis=0)
  return np.exp(z_shifted) / np.sum(np.exp(z_shifted), axis=0)


def forward_propagation(w1, b1, w2, b2, w3, b3, x):
    # First hidden layer: 784 -> 256
    z1 = w1.dot(x) + b1 # the result shape is [256, m]
    a1 = sigmoid(z1) # the result shape is [256, m]
    # Second hidden layer: 256 -> 128
    z2 = w2.dot(a1) + b2  # the result shape is [128, m]
    a2 = sigmoid(z2) # the result shape is [128, m]
    # Output layer: 128 -> 10
    z3 = w3.dot(a2) + b3 # the result shape is [10, m]
    a3 = softmax(z3) # the result shape is [10, m]
    return z1, a1, z2, a2, z3, a3


def one_hot(y):
  y_onehot = np.zeros((y.shape[0], 10))  # y.shape[0] is m, training sample size
  y_onehot[np.arange(y.shape[0]), y] = 1
  y_onehot = y_onehot.T
  return y_onehot


# print(one_hot(np.array([0, 3, 2, 6, 3, 7, 9, 0, 2, 2, 3, 4, 5])))

def deriv_ReLU(z):
  return (z > 0).astype(int)

def deriv_Sigmoid(z):
  return sigmoid(z) * (1 - sigmoid(z))


def back_propagation(Z1, A1, Z2, A2, Z3, A3,
                     W1, W2, W3, X, Y, lambd):
    m = Y.shape[0]
    onehot_Y = one_hot(Y)

    # Gradient calculation with L2 term added: (lambd / m) * W
    dZ3 = A3 - onehot_Y
    dW3 = 1 / m * dZ3.dot(A2.T) + (lambd / m) * W3
    db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)

    dZ2 = W3.T.dot(dZ3) * deriv_Sigmoid(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T) + (lambd / m) * W2
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * deriv_Sigmoid(Z1)
    dW1 = 1 / m * dZ1.dot(X.T) + (lambd / m) * W1
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3


def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3


def get_predictions(A3):
  return np.argmax(A3, 0)


def get_accuracy(predictions, Y):
  return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, alpha, epochs, batch_size=64, epsilon=1e-5, lambd=0.1):
  W1, b1, W2, b2, W3, b3 = init_params(784, 256, 128)
  m = Y.shape[0]
  prev_loss = float('inf')

  for i in range(epochs):
    # Shuffle at the start of each epoch
    permutation = np.random.permutation(m)
    X_shuffled = X[:, permutation]
    Y_shuffled = Y[permutation]

    epoch_loss = 0
    num_batches = 0
    for start in range(0, m, batch_size):
      X_batch = X_shuffled[:, start:start + batch_size]
      Y_batch = Y_shuffled[start:start + batch_size]
      batch_m = Y_batch.shape[0]

      Z1, A1, Z2, A2, Z3, A3 = forward_propagation(W1, b1, W2, b2, W3, b3, X_batch)
      dW1, db1, dW2, db2, dW3, db3 = back_propagation(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X_batch, Y_batch, lambd)
      W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)

      batch_loss = -1 / batch_m * np.sum(one_hot(Y_batch) * np.log(A3 + 1e-8))
      epoch_loss += batch_loss
      num_batches += 1

    current_loss = epoch_loss / num_batches

    if abs(prev_loss - current_loss) < epsilon:
      print(f"Converged at epoch {i}")
      break

    prev_loss = current_loss

    if i % 10 == 0:
      preds = get_predictions(forward_propagation(W1, b1, W2, b2, W3, b3, X)[5])
      print(f"Epoch {i} | Loss: {current_loss:.4f} | Acc: {get_accuracy(preds, Y):.4f}")

  return W1, b1, W2, b2, W3, b3


def make_predictions(X, W1, b1, W2, b2, W3, b3):
  _, _, _, _, _, A3 = forward_propagation(W1, b1, W2, b2, W3, b3, X)
  predictions = get_predictions(A3)
  return predictions


def test_prediction(index, W1, b1, W2, b2, W3, b3):
  current_image = x_train[:, index, None]
  prediction = make_predictions(x_train[:, index, None], W1, b1, W2, b2, W3, b3)
  label = y_train[index]
  print("Prediction: ", prediction)
  print("Label: ", label)

  current_image = current_image.reshape((28, 28)) * 255
  plt.gray()
  plt.imshow(current_image, interpolation='nearest')
  plt.show()


W1, b1, W2, b2, W3, b3 = gradient_descent(x_train, y_train, 0.10, epochs=50, batch_size=64)
print(f"W1 shape: {W1.shape}, b1 shape: {b1.shape}")
print(f"W2 shape: {W2.shape}, b2 shape: {b2.shape}")
print(f"W3 shape: {W3.shape}, b3 shape: {b3.shape}")

test_prediction(0, W1, b1, W2, b2, W3, b3)
test_prediction(1, W1, b1, W2, b2, W3, b3)
test_prediction(2, W1, b1, W2, b2, W3, b3)
test_prediction(3, W1, b1, W2, b2, W3, b3)
test_prediction(4, W1, b1, W2, b2, W3, b3)
test_prediction(5, W1, b1, W2, b2, W3, b3)
test_prediction(6, W1, b1, W2, b2, W3, b3)
test_prediction(7, W1, b1, W2, b2, W3, b3)

dev_predictions = make_predictions(x_dev, W1, b1, W2, b2, W3, b3)
print(f"MLP Dev/Validation Accuracy: {get_accuracy(dev_predictions, y_dev) * 100:.2f}%")
