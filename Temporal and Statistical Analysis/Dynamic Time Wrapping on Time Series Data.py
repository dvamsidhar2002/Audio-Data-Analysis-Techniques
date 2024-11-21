import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Generate example time-series data
np.random.seed(42)
time_series_1 = np.cumsum(np.random.randn(100))
time_series_2 = np.cumsum(np.random.randn(120))

# Compute DTW
distance_matrix = cdist(time_series_1[:, np.newaxis], time_series_2[:, np.newaxis], metric='euclidean')
n, m = distance_matrix.shape
cost_matrix = np.zeros((n, m))
cost_matrix[0, 0] = distance_matrix[0, 0]

# Fill the cost matrix
for i in range(1, n):
    cost_matrix[i, 0] = cost_matrix[i - 1, 0] + distance_matrix[i, 0]
for j in range(1, m):
    cost_matrix[0, j] = cost_matrix[0, j - 1] + distance_matrix[0, j]
for i in range(1, n):
    for j in range(1, m):
        cost_matrix[i, j] = distance_matrix[i, j] + min(
            cost_matrix[i - 1, j],
            cost_matrix[i, j - 1],
            cost_matrix[i - 1, j - 1],
        )

# Backtrack for the optimal path
path = []
i, j = n - 1, m - 1
while i > 0 and j > 0:
    path.append((i, j))
    steps = [
        (cost_matrix[i - 1, j], i - 1, j),
        (cost_matrix[i, j - 1], i, j - 1),
        (cost_matrix[i - 1, j - 1], i - 1, j - 1),
    ]
    _, i, j = min(steps)
path.append((0,0))
path.reverse()
path_x, path_y = zip(*path)

# Plot the results
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(time_series_1, label="Time Series 1")
plt.plot(time_series_2, label="Time Series 2")
plt.legend()
plt.title("Original Time-Series Data")

plt.subplot(2, 1, 2)
plt.imshow(distance_matrix.T, origin='lower', cmap='viridis', interpolation='nearest')
plt.plot(path_x, path_y, color='red')
plt.title("DTW Distance Matrix and Optimal Path")
plt.xlabel("Time Series 1")
plt.ylabel("Time Series 2")
plt.colorbar()
plt.tight_layout()
plt.show()