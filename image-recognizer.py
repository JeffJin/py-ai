import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('./data/image-recognizer-train.csv')

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
_,m_train = x_train.shape

# print(y_train, m_train)
#     - The syntax `:` means you are selecting all rows in the array.
#     - The syntax `0` means you are selecting the first column in those rows.
print('x shape = ', x_train.shape)

def init_params(n1, n2):
    w1 = np.random.rand(10, n1) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    w2 = np.random.rand(10, n2) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    # print('init_params', w1, b1, w2, b2)
    return w1, b1, w2, b2

def ReLU(z):
    return np.maximum(0, z)

def softmax(z):
    # exp(z2[0][0]) / (exp(z2[0][0]) + exp(z2[1][0]) + ... + exp(z2[9][0]))
    return np.exp(z) / np.sum(np.exp(z), axis=0)

def forward_prop(w1, b1, w2, b2, x):
    z1 = w1.dot(x) + b1
    # a1 = np.tanh(z1)  #
    a1 = ReLU(z1) # ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def one_hot(y):
    y_onehot = np.zeros((y.shape[0], y.max() + 1)) #y.shape[0] is m, training sample size
    y_onehot[np.arange(y.shape[0]), y] = 1
    y_onehot = y_onehot.T
    return y_onehot
# print(one_hot(np.array([0, 3, 2, 6, 3, 7, 9, 0, 2, 2, 3, 4, 5])))

def deriv_tanh(z):
    return 1 - np.tanh(z)**2

def deriv_ReLU(z):
    return (z > 0).astype(int)

def deriv_softmax(z):
    s = z.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)

def back_propagation_v1(Z1,  A1, Z2, A2, W1, W2, X, Y):
    m = Y.shape[0]
    onehot_Y = one_hot(Y) # one_hot is expected value(real value in the image)
    dZ2 = A2 - onehot_Y # negative value for the valid ones and positive for invalid ones (-1 to +1)
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    # dZ11 = dZ2.T.dot(W2).T * deriv_ReLU(Z1)
    # print(dZ1 == dZ11)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

# TODO - implement another version with other activation functions
def back_propagation_v2(Z1,  A1, Z2, A2, W1, W2, X, Y):
    m = Y.shape[0]
    onehot_Y = one_hot(Y) # one_hot is expected value(real value in the image)
    dZ2 = A2 - onehot_Y # negative value for the valid ones and positive for invalid ones (-1 to +1)
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_tanh(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params(784, 10)
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_propagation_v1(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


def test_prediction(index, W1, b1, W2, b2):
    current_image = x_train[:, index, None]
    prediction = make_predictions(x_train[:, index, None], W1, b1, W2, b2)
    label = y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

W1, b1, W2, b2 = gradient_descent(x_train, y_train, 0.10, 500)
# print(W1.shape, b1.shape, W2.shape, b2.shape)

test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)
test_prediction(4, W1, b1, W2, b2)
test_prediction(5, W1, b1, W2, b2)
test_prediction(6, W1, b1, W2, b2)
test_prediction(7, W1, b1, W2, b2)

dev_predictions = make_predictions(x_dev, W1, b1, W2, b2)
get_accuracy(dev_predictions, y_dev)