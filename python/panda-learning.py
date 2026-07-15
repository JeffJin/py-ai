import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create a range of x values
x1 = np.linspace(-10, 10, 1000)  # 100 evenly spaced points between -10 and 10
x2 = np.linspace(-10, 10, 1000)  # 100 evenly spaced points between -10 and 10
# y = np.linspace(0, 1, 5)  # 100 evenly spaced points between -10 and 10
phi = np.array([x1, x2, x1**2 + x2**2])

y = np.sign(np.array([2, 2, -1]).dot(phi))

# Step 3: Store the values in a pandas DataFrame
data = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})

# Step 4: Plot the data
plt.plot(data['x1'], data['x2'], data['y'], label="y = f(x)")
plt.title("Quadratic Function")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()
