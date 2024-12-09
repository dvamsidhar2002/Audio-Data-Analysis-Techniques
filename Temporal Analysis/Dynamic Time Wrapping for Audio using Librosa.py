import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Load an audio file
audio_path = r'H:\Audio Data Analysis\TESS Toronto emotional speech set data\OAF_angry\OAF_back_angry.wav'  # Replace with your file
audio, sr = librosa.load(audio_path, sr=None)

# Split into two segments
split_point = len(audio) // 2
audio_1 = audio[:split_point]
audio_2 = audio[split_point:]

# Compute MFCC features
mfcc_1 = librosa.feature.mfcc(y=audio_1, sr=sr, n_mfcc=13)
mfcc_2 = librosa.feature.mfcc(y=audio_2, sr=sr, n_mfcc=13)

# Compute DTW between the MFCC features
D, wp = librosa.sequence.dtw(mfcc_1.T, mfcc_2.T, metric='cosine')

# Visualize the results
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
librosa.display.specshow(mfcc_1, x_axis='time', sr=sr)
plt.title("MFCC - Segment 1")
plt.colorbar()

plt.subplot(2, 1, 2)
librosa.display.specshow(mfcc_2, x_axis='time', sr=sr)
plt.title("MFCC - Segment 2")
plt.colorbar()

plt.figure(figsize=(8, 6))
plt.imshow(D, origin='lower', cmap='gray')
plt.plot(np.array(wp)[:, 0], np.array(wp)[:, 1], color='red')
plt.title("DTW Alignment Cost Matrix with Path")
plt.xlabel("Segment 1")
plt.ylabel("Segment 2")
plt.colorbar()
plt.show()
