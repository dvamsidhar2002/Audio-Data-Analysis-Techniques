import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the audio file
file_path = ""  # Change this to your audio file
y, sr = librosa.load(file_path, sr=None)

# Extract MFCC features
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # 13 MFCC coefficients

# Standardize the MFCCs
scaler = StandardScaler()
mfccs_scaled = scaler.fit_transform(mfccs.T)  # Transpose for proper scaling

# Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 principal components for visualization
mfccs_pca = pca.fit_transform(mfccs_scaled)

# Plot the PCA-transformed MFCCs
plt.figure(figsize=(8, 6))
plt.scatter(mfccs_pca[:, 0], mfccs_pca[:, 1], alpha=0.5, c=np.arange(len(mfccs_pca)), cmap='coolwarm')
plt.colorbar(label="Time Frames")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of MFCC Features")
plt.show()

# Print explained variance ratio
print("Explained Variance Ratio:", pca.explained_variance_ratio_)