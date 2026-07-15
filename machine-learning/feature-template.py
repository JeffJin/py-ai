from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

# Sample data
data = np.array([[1], [2], [3], [4], [100]])

# Min-Max Normalization
minmax_scaler = MinMaxScaler()
data_minmax = minmax_scaler.fit_transform(data)
print("Min-Max Normalized:\n", data_minmax)

# Z-Score Normalization
standard_scaler = StandardScaler()
data_standard = standard_scaler.fit_transform(data)
print("Z-Score Normalized:\n", data_standard)