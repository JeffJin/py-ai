import numpy as np
import pandas as pd


class DigitRecognizer:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        # Initialize Weights (He Initialization for ReLU) and Biases (d)
        # Layer 1: W1 (128, 784), d1 (128, 1)
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2. / input_size)
        self.d1 = np.zeros((hidden_size, 1))

        # Layer 2: W2 (10, 128), d2 (10, 1)
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2. / hidden_size)
        self.d2 = np.zeros((output_size, 1))

    # --- Activation Functions ---
    def relu(self, Z):
        return np.maximum(0, Z)

    def softmax(self, Z):
        # Subtract max for numerical stability (prevents overflow)
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def relu_deriv(self, Z):
        return Z > 0

    # --- Forward Propagation ---
    def forward(self, X):
        # Layer 1
        Z1 = np.dot(self.W1, X) + self.d1
        A1 = self.relu(Z1)

        # Layer 2
        Z2 = np.dot(self.W2, A1) + self.d2
        A2 = self.softmax(Z2)

        # Cache values for backprop
        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache

    # --- Backpropagation ---
    def backward(self, X, Y, cache, learning_rate=0.1):
        m = X.shape[1]  # Number of training examples
        A2 = cache["A2"]
        A1 = cache["A1"]
        Z1 = cache["Z1"]

        # 1. Output Layer Gradients
        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        dd2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        # 2. Hidden Layer Gradients
        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * self.relu_deriv(Z1)
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        dd1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        # 3. Update Parameters (Gradient Descent)
        self.W1 -= learning_rate * dW1
        self.d1 -= learning_rate * dd1
        self.W2 -= learning_rate * dW2
        self.d2 -= learning_rate * dd2

    # --- One Hot Encoding ---
    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        return one_hot_Y.T

    # --- Training Loop ---
    def train(self, data_path, epochs=500, learning_rate=0.1):
        # 1. Read Data (Skipping header as per your screenshot format)
        data = pd.read_csv(data_path)
        data = np.array(data)
        m, n = data.shape
        np.random.shuffle(data)

        # 2. Split X (Pixels) and Y (Labels)
        # Transpose so columns are examples, rows are features
        data_dev = data[0:1000].T  # Validation set (first 1000)
        Y_dev = data_dev[0]
        X_dev = data_dev[1:n] / 255.0

        data_train = data[1000:m].T  # Training set
        Y_train = data_train[0]
        X_train = data_train[1:n] / 255.0

        # Convert Y to One-Hot
        Y_train_encoded = self.one_hot(Y_train)

        print(f"Training on {m - 1000} examples...")

        for i in range(epochs):
            # A. Forward
            A2, cache = self.forward(X_train)

            # B. Backward & Update
            self.backward(X_train, Y_train_encoded, cache, learning_rate)

            # C. Progress Log
            if i % 50 == 0:
                predictions = np.argmax(A2, axis=0)
                accuracy = np.mean(predictions == Y_train)
                print(f"Epoch {i}: Accuracy = {accuracy:.2%}")

    # --- Prediction ---
    def predict(self, image_data):
        # Expects image_data to be shape (784, 1) or (784, m)
        output, _ = self.forward(image_data)
        return np.argmax(output, axis=0)

# --- Usage Example ---
# Save the code below to a file (e.g., main.py) and run it with your CSV
# nn = DigitRecognizer()
# nn.train('your_file.csv')