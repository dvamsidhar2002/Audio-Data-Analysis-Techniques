import numpy as np
from scipy.spatial.distance import cdist

# Example sequences
sequence_1 = np.array([1,2,3,4,5])
sequence_2 = np.array([1,1,2,3,4,4,5])

# Compute distance matrix
distance_matrix = cdist(sequence_1[:, np.newaxis], sequence_2[:, np.newaxis], metric='euclidean')

n, m = distance_matrix.shape
cost_matrix = np.zeros((n, m))
cost_matrix[0, 0] = distance_matrix[0, 0]

# Fill the cost matrix
for i in range(1, n):
    cost_matrix[i, 0] = cost_matrix[i - 1, 0] + distance_matrix[i, 0]

for j in range(1, m):
    cost_matrix[0, j] = cost_matrix[0, j-1] + distance_matrix[0, j]

for i in range(1, n):
    for j in range(1, m):
        cost_matrix[i, j] = distance_matrix[i, j] + min(
            cost_matrix[i - 1, j],  # insertion
            cost_matrix[i, j - 1],  # deletion
            cost_matrix[i - 1, j - 1] # match
        )

# Backtrack to find the optimal path
path = []
i, j = n - 1, m - 1
while i > 0 and j > 0:
    path.append((i, j))
    steps = [
        (cost_matrix[i - 1, j], i - 1, j),  # insertion
        (cost_matrix[i, j - 1], i, j - 1),  # deletion
        (cost_matrix[i - 1, j - 1], i - 1, j - 1),  # match
    ]
    _, i, j = min(steps)
path.append((0, 0))
path.reverse()

# Print results
print("Distance Matrix:")
print(distance_matrix)
print("\nCost Matrix:")
print(cost_matrix)
print("\nOptimal Path:", path)