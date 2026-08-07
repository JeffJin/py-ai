import matplotlib
matplotlib.use('Agg')  # headless backend — saves to file instead of GUI
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import pandas as pd
# Make 1000 samples
n_samples = 1000
# Create circles
X, y = make_circles(n_samples,
                    noise=0.03, # a little bit of noise to the dots
                    random_state=42) # keep random state so we get the same values
print(f"First 5 X features:\n{X[:5]}")
print(f"\nFirst 5 y labels:\n{y[:5]}")
circles = pd.DataFrame({"X1": X[:, 0],
    "X2": X[:, 1],
    "label": y
})
print("circle value count", circles.label.value_counts())
print("Start plotting the circles ...")
plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap=plt.cm.RdYlBu)
plt.title("Binary Classification - Circles")
plt.savefig("circles.png", dpi=150, bbox_inches='tight')
print("Plot saved to: circles.png")
