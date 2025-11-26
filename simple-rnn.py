import numpy as np

# 1. Define the Fixed Input Size
# For simple emails, 50-100 is enough.
# For complex NLP, this might be 10,000+.
VECTOR_SIZE = 50


def features_to_vector(feature_dict, size):
    # Initialize a vector of all zeros
    x_vector = np.zeros(size)

    for key, value in feature_dict.items():
        # A. Get the Hash of the string (returns a large integer)
        h = hash(key)

        # B. Find the slot (Modulo)
        # abs() is needed because hash() can be negative in Python
        index = abs(h) % size

        # C. Assign value
        # We use += instead of = to handle "Collisions"
        # (rare case where two different features land in the same slot)
        x_vector[index] += value

    return x_vector


# --- Example Workflow ---

# A. Your Template Extraction (From before)
raw_email = "jeff@gmail.com"
feature_dict = {
    'suffix=com': 1,
    'len>5': 1,
    'count_@=1': 1,
    'has_dot': 1
}

# B. Convert to Neural Net Input
X_input = features_to_vector(feature_dict, VECTOR_SIZE)

print(f"Dictionary keys: {list(feature_dict.keys())}")
print(f"Vector shape: {X_input.shape}")
print(f"Non-zero indices: {np.where(X_input != 0)[0]}")


# Example Output: Non-zero indices: [ 4 12 33 48 ]


# --- Helper Functions ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(p):
    # The derivative of sigmoid is p * (1 - p)
    return p * (1 - p)


# --- The Neural Network Class ---
class SimpleNeuralNet:
    def __init__(self, input_size, hidden_size, output_size):
        # 1. Initialize Weights Randomly
        # Layer 1 (Input -> Hidden)
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))

        # Layer 2 (Hidden -> Output)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        """
        Move data forward and SAVE the values (cache)
        because we need them for the backward pass.
        """
        # Step 1: Input -> Hidden
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)  # Hidden Layer Activation

        # Step 2: Hidden -> Output
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = sigmoid(self.z2)  # Final Prediction

        return self.output

    def backward(self, X, y, learning_rate):
        """
        The Core Backpropagation Logic
        """
        m = X.shape[0]  # Number of examples (batch size)

        # --- LAYER 2 (OUTPUT) GRADIENTS ---
        # 1. Calculate Output Error (p - y)
        # This matches the Logistic Regression gradient we found earlier
        error_output = self.output - y

        # 2. Calculate Gradients for W2 and b2
        # dW2 = (Hidden_State_Transposed dot Error)
        dW2 = np.dot(self.a1.T, error_output)
        db2 = np.sum(error_output, axis=0, keepdims=True)

        # --- LAYER 1 (HIDDEN) GRADIENTS ---
        # 3. Propagate Error Backwards
        # How much did Layer 1 contribute to the error?
        # We take the error from Layer 2 and multiply by W2 weights (transposed)
        error_hidden = np.dot(error_output, self.W2.T)

        # 4. Multiply by Derivative of Activation Function (Chain Rule)
        # derivative of sigmoid(a1) = a1 * (1 - a1)
        delta_hidden = error_hidden * sigmoid_derivative(self.a1)

        # 5. Calculate Gradients for W1 and b1
        dW1 = np.dot(X.T, delta_hidden)
        db1 = np.sum(delta_hidden, axis=0, keepdims=True)

        # --- UPDATE WEIGHTS ---
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1


# --- Testing the Implementation ---

# 1. Dummy Data (4 examples, 3 features)
# Features: [Has_@, Has_., Length>5]
X = np.array([
    [1, 1, 1],  # Valid
    [0, 1, 1],  # No @
    [1, 0, 1],  # No .
    [1, 1, 0]  # Too Short
])

# Labels (1 = Valid, 0 = Invalid)
y = np.array([
    [1],
    [0],
    [0],
    [0]
])

# 2. Create Network
# Input: 3 features -> Hidden: 4 neurons -> Output: 1 prediction
nn = SimpleNeuralNet(input_size=3, hidden_size=4, output_size=1)

# 3. Training Loop
print(f"{'Epoch':<6} | {'Loss':<10} | {'Prediction for Valid Email'}")
print("-" * 45)

for i in range(5000):
    # A. Forward Pass
    preds = nn.forward(X)

    # B. Backward Pass (updates weights internally)
    nn.backward(X, y, learning_rate=0.1)

    # C. Monitor Progress (every 1000 epochs)
    if i % 1000 == 0:
        # Calculate Mean Squared Error just for display
        loss = np.mean(np.square(y - preds))
        print(f"{i:<6} | {loss:.5f}    | {preds[0][0]:.4f}")

print("-" * 45)
print("Final Predictions:\n", nn.forward(X))