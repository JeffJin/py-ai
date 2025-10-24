#%%
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

trueW = np.array([1, 2, 3, 4, 5])
def generateData():
    x = np.random.randn(len(trueW))
    y = trueW.dot(x) + np.random.randn()
    # print(f"example: x={x}, y={y}")
    return (x, y)

trainingData = [generateData() for _ in range(10000)]

def phi(x):
    return np.array(x)

def initW():
    return np.zeros(len(trueW))

def trainLoss(w):
    return 1 / len(trainingData) * sum((w.dot(phi(x)) - y) ** 2 for x, y in trainingData)

# this results in a 1 x 2 matrix, [0, 0], [0.47, 1.27] ...
def gradiantTrainLoss(w):
    # w = [w1, w2]
    # phi(x) =[1, x] => [1, 1], [1, 2], [1, 4]
    # y = w1 * 1 + w2 * x
    return 1 / len(trainingData) * sum(2 * (w.dot(phi(x)) - y) * phi(x) for x, y in trainingData)

def stochasticTrainLoss(w, i):
    # w = [w1, w2]
    # phi(x) =[1, x] => [1, 1], [1, 2], [1, 4]
    if i >= 0:
        [x, y] = trainingData[i]
        return 2 * (w.dot(phi(x)) - y) * phi(x)
    else:
        [x, y] = random.choice(trainingData)
        return 2 * (w.dot(phi(x)) - y) * phi(x)

def drawData(w_values):
    # Plot the line using coordinates from w_values
    w_df = pd.DataFrame(w_values, columns=["x", "y"])
    plt.plot(w_df["x"], w_df["y"], marker="o", label="Line")
    plt.title("Line Plot from Vector Array (w_values)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

def gradiantDescent(gradiantFunc, step):
    w = initW()
    w_values = []
    iteration = 0  # Manual iteration counter

    while True:
        gradiant = gradiantFunc(w)
        w = w - step * gradiant
        if np.linalg.norm(gradiant) < 1e-9:  # Convergence condition
            break
        if iteration > 500:  # Iteration limit to prevent infinite loops
            print("Iteration limit reached. Stopping.")
            break
        w_values.append(w.copy())
        print(f"Loss at iteration {iteration}: w={w}, gradiant={gradiant}")
        iteration += 1  # Increment iteration counter

def stochasticGradiantDescent(stGradiantFunc, n):
    w = initW()
    iteration = 0  # Manual iteration counter
    w_values = []

    # option 1
    for _ in range(10):
        for i in range(n):
            gradiant = stGradiantFunc(w, i)
            iteration += 1  # Increment iteration counter
            step = 1.0 / math.sqrt(iteration)
            w = w - step * gradiant
            if np.linalg.norm(gradiant) < 1e-5:  # Convergence condition
              break
            w_values.append(w.copy())
            print(f"Loss at iteration {iteration}: n={n}, w={w}, step={step}, gradiant={gradiant}")

    # option 2
    # while True:
    #     gradiant = stGradiantFunc(w, -1)
    #     iteration += 1  # Increment iteration counter
    #     step = 1.0 / math.sqrt(iteration)
    #     # step = 0.01
    #     w = w - step * gradiant
    #     if np.linalg.norm(gradiant) < 1e-9:  # Convergence condition
    #         print("Reached the accuracy 1e-5. Stopping.\n"),
    #         break
    #     if iteration > 50000:
    #         print("Iteration limit reached. Stopping.\n"),
    #         break
    #     print(f"Loss at iteration {iteration}: n={n}, w={w}, step={step}, gradiant={gradiant}")
    # drawData(w_values)

gradiantDescent(gradiantTrainLoss, 0.1)
# stochasticGradiantDescent(stochasticTrainLoss, len(trainingData))

#%%

#%%

#%%
