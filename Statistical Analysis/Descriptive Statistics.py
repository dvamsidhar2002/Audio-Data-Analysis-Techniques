# importing necessary libraries
import librosa
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


# Load the audio file
audio_file = r"H:\Audio Data Analysis\TESS Toronto emotional speech set data\OAF_angry\OAF_back_angry.wav"
y, sr = librosa.load(audio_file, sr=16000)


# Compute descriptive statistics
mean_amplitude = np.mean(y)
variance_amplitude = np.var(y)
std_dev_amplitude = np.std(y)
skewness_amplitude = stats.skew(y)
kurtosis_amplitude = stats.kurtosis(y)
min_amplitude = np.min(y)
max_amplitude = np.max(y)
rms_amplitude = np.sqrt(np.mean(y**2))

# Display the results
print(f"Sampling Rate: {sr} Hz")
print(f"Mean Amplitude: {mean_amplitude}")
print(f"Variance: {variance_amplitude}")
print(f"Standard Deviation: {std_dev_amplitude}")
print(f"Skewness: {skewness_amplitude}")
print(f"Kurtosis: {kurtosis_amplitude}")
print(f"Min Amplitude: {min_amplitude}")
print(f"Max Amplitude: {max_amplitude}")
print(f"RMS Amplitude: {rms_amplitude}")

# Visualize the waveform
plt.figure(figsize=(10, 4))
plt.plot(y, color='blue')
plt.title("Audio Waveform")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.show()