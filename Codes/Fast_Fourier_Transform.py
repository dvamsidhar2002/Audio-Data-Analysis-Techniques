import numpy as np
import librosa
import matplotlib.pyplot as plt

FIG_SIZE = (15,10)

file = "H:\Audio Data Analysis\TESS Toronto emotional speech set data\OAF_angry\OAF_back_angry.wav"

signal, sampling_rate = librosa.load(file, sr=22050)

fft = np.fft.fft(signal)
spectrum = np.abs(fft)
f = np.linspace(0, sampling_rate, len(spectrum))
left_spectrum = spectrum[:int(len(spectrum)/2)]
left_f = f[:int(len(spectrum)/2)]
plt.figure(figsize=FIG_SIZE)
plt.plot(left_f, left_spectrum, alpha=0.4)
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.title('Power Spectrum')
plt.show()