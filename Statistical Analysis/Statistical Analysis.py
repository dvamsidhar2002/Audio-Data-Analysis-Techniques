import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load an audio file
audio_file = r"H:\Audio Data Analysis\GTZAN Data\genres_original\disco\disco.00000.wav"
y, sr = librosa.load(audio_file, sr=None)  # y is the audio signal, sr is the sample rate

# Extract some features:
# 1. Spectral Centroid (brightness of the sound)
spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

# 2. Spectral Roll-off (frequency below which a certain percentage of total spectral energy lies)
# Ensure that `rolloff` is used correctly based on the version of librosa.
# For librosa v0.8.0 and earlier, `rolloff` might work as expected
spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)

# 3. Zero-Crossing Rate (how often the signal crosses zero)
zero_crossings = librosa.feature.zero_crossing_rate(y=y)

# 4. Root Mean Square Energy (loudness)
rmse = librosa.feature.rms(y=y)

# Calculate basic statistics on the features
def calculate_statistics(feature):
    mean = np.mean(feature)
    std = np.std(feature)
    median = np.median(feature)
    min_val = np.min(feature)
    max_val = np.max(feature)
    return mean, std, median, min_val, max_val

# Compute statistics for each feature
centroid_stats = calculate_statistics(spectral_centroid)
rolloff_stats = calculate_statistics(spectral_rolloff)
zero_crossing_stats = calculate_statistics(zero_crossings)
rmse_stats = calculate_statistics(rmse)

print("Spectral Centroid Statistics:", centroid_stats)
print("Spectral Roll-off Statistics:", rolloff_stats)
print("Zero Crossing Rate Statistics:", zero_crossing_stats)
print("Root Mean Square Energy Statistics:", rmse_stats)

# Visualize features for better understanding
plt.figure(figsize=(12, 8))

# Spectral Centroid
plt.subplot(4, 1, 1)
plt.plot(spectral_centroid.T)
plt.title('Spectral Centroid')

# Spectral Roll-off
plt.subplot(4, 1, 2)
plt.plot(spectral_rolloff.T)
plt.title('Spectral Roll-off')

# Zero-Crossing Rate
plt.subplot(4, 1, 3)
plt.plot(zero_crossings.T)
plt.title('Zero Crossing Rate')

# Root Mean Square Energy
plt.subplot(4, 1, 4)
plt.plot(rmse.T)
plt.title('Root Mean Square Energy')

plt.tight_layout()
plt.show()
