import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

data = pd.read_csv('../data/image-recognizer-train.csv')
data = data.sample(frac=1).reset_index(drop=True)

data_dev = data.iloc[0:1000]
y_dev = torch.tensor(data_dev.iloc[:, 0].values, dtype=torch.long)
x_dev = torch.tensor(data_dev.iloc[:, 1:].values / 255.0, dtype=torch.float32)

data_train = data.iloc[1000:]
y_train = torch.tensor(data_train.iloc[:, 0].values, dtype=torch.long)
x_train = torch.tensor(data_train.iloc[:, 1:].values / 255.0, dtype=torch.float32)

print('x shape = ', x_train.shape)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

        # Xavier/Glorot initialization to match numpy version
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # CrossEntropyLoss applies softmax internally
        return x


def get_accuracy(predictions, Y):
    return (predictions == Y).float().mean().item()


def gradient_descent(X, Y, alpha, iterations, epsilon=1e-5):
    model = MLP()
    optimizer = optim.SGD(model.parameters(), lr=alpha)
    loss_fn = nn.CrossEntropyLoss()

    prev_loss = float('inf')

    for i in range(iterations):
        model.train()
        optimizer.zero_grad()

        logits = model(X)
        loss = loss_fn(logits, Y)
        loss.backward()
        optimizer.step()

        current_loss = loss.item()

        if abs(prev_loss - current_loss) < epsilon:
            print(f"Converged at iteration {i}")
            break

        prev_loss = current_loss

        if i % 100 == 0:
            print("Iteration: ", i)
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=1)
                print(get_accuracy(predictions, Y))

    return model


def make_predictions(X, model):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        return torch.argmax(logits, dim=1)


def test_prediction(index, model):
    current_image = x_train[index].unsqueeze(0)
    prediction = make_predictions(current_image, model)
    label = y_train[index].item()
    print("Prediction: ", prediction.item())
    print("Label: ", label)

    img = current_image.squeeze().detach().reshape(28, 28) * 255
    plt.gray()
    plt.imshow(img, interpolation='nearest')
    plt.show()


model = gradient_descent(x_train, y_train, alpha=0.10, iterations=500)

for i in range(8):
    test_prediction(i, model)

dev_predictions = make_predictions(x_dev, model)
print(f"MLP (PyTorch) Dev/Validation Accuracy: {get_accuracy(dev_predictions, y_dev) * 100:.2f}%")
