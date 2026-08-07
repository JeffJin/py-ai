import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load and preprocess data
data = pd.read_csv('../data/image-recognizer-train.csv')
data = np.array(data)
m, n = data.shape

# Shuffle data
np.random.shuffle(data)

# Split into dev and train sets
data_dev = data[0:1000].T
y_dev = data_dev[0].astype(int)
x_dev = data_dev[1:n] / 255.0

data_train = data[1000:m].T
y_train = data_train[0].astype(int)
x_train = data_train[1:n] / 255.0

# Convert to PyTorch tensors
x_train = torch.tensor(x_train.T, dtype=torch.float32).to(device)  # (m_train, 784)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)  # (m_train,)
x_dev = torch.tensor(x_dev.T, dtype=torch.float32).to(device)  # (1000, 784)
y_dev = torch.tensor(y_dev, dtype=torch.long).to(device)  # (1000,)

print(f'x_train shape: {x_train.shape}')
print(f'y_train shape: {y_train.shape}')

# Define the neural network using PyTorch
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(NeuralNetwork, self).__init__()
        # Define layers: 784 -> 256 -> 128 -> 10
        self.fc1 = nn.Linear(input_size, hidden1_size)    # 784 -> 256
        self.fc2 = nn.Linear(hidden1_size, hidden2_size) # 256 -> 128
        self.fc3 = nn.Linear(hidden2_size, output_size)  # 128 -> 10
        
        # Initialize weights using Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        # Initialize biases to zero (default in PyTorch, but explicit here)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
    
    def forward(self, x):
        # Forward pass: Input -> Hidden1 -> Hidden2 -> Output
        z1 = self.fc1(x)   # 784 -> 256
        a1 = F.sigmoid(z1)    # ReLU activation
        z2 = self.fc2(a1)  # 256 -> 128
        a2 = F.sigmoid(z2)    # ReLU activation
        z3 = self.fc3(a2)  # 128 -> 10
        # Note: We'll apply softmax in the loss function (CrossEntropyLoss does this)
        return z3

# Create model: 784 -> 256 -> 128 -> 10
model = NeuralNetwork(input_size=784, hidden1_size=256, hidden2_size=128, output_size=10).to(device)
print(f'Model:\n{model}')

# Define loss function and optimizer
# CrossEntropyLoss includes softmax, so we don't need to apply it manually
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.10)

# Training function
def train_model(model, x_train, y_train, epochs, print_interval=10):
    model.train()  # Set model to training mode
    for epoch in range(epochs):
        # Forward pass
        outputs = model(x_train)  # Shape: (batch_size, 10)
        
        # Compute loss
        loss = criterion(outputs, y_train)
        
        # Backward pass and optimization
        optimizer.zero_grad()  # Clear gradients
        loss.backward()         # Compute gradients (automatic!)
        optimizer.step()        # Update parameters
        
        # Print progress
        if epoch % print_interval == 0:
            with torch.no_grad():
                # Get predictions
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == y_train).float().mean().item()
                print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

# Train the model
print('\nStarting training...')
train_model(model, x_train, y_train, epochs=1500, print_interval=10)

# Evaluation function
def evaluate_model(model, x_data, y_data):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        outputs = model(x_data)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_data).float().mean().item()
        return accuracy, predicted

# Evaluate on dev set
print('\nEvaluating on dev set...')
dev_accuracy, dev_predictions = evaluate_model(model, x_dev, y_dev)
print(f'Dev Accuracy: {dev_accuracy:.4f}')

# Test prediction function
def test_prediction(model, x_train, y_train, index):
    model.eval()
    with torch.no_grad():
        # Get single image
        current_image = x_train[index:index+1]  # Keep batch dimension
        output = model(current_image)
        _, prediction = torch.max(output, 1)
        label = y_train[index]
        
        print(f"Prediction: {prediction.item()}")
        print(f"Label: {label.item()}")
        
        # Visualize image
        img = current_image.cpu().numpy().reshape(28, 28) * 255
        plt.gray()
        plt.imshow(img, interpolation='nearest')
        plt.title(f'Predicted: {prediction.item()}, Actual: {label.item()}')
        plt.show()

# Test some predictions
print('\nTesting predictions...')
for i in range(8):
    test_prediction(model, x_train, y_train, i)
