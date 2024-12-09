# importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import librosa
#import ipython_genutils as ipd

# specifying the paths of audio files
blues_file = 'H:/Audio Data Analysis\GTZAN Data/genres_original/blues/blues.00000.wav'
classical_file = 'H:/Audio Data Analysis/GTZAN Data/genres_original/classical/classical.00000.wav'
country_file = 'H:/Audio Data Analysis/GTZAN Data/genres_original/country/country.00000.wav'
disco_file = 'H:/Audio Data Analysis/GTZAN Data/genres_original/disco/disco.00000.wav'
hiphop_file = 'H:/Audio Data Analysis/GTZAN Data/genres_original/hiphop/hiphop.00000.wav'
jazz_file = 'H:/Audio Data Analysis/GTZAN Data/genres_original/jazz/jazz.00000.wav'
metal_file = 'H:/Audio Data Analysis/GTZAN Data/genres_original/metal/metal.00000.wav'
pop_file = 'H:/Audio Data Analysis/GTZAN Data/genres_original/pop/pop.00000.wav'
reggae_file = 'H:/Audio Data Analysis/GTZAN Data/genres_original/reggae/reggae.00000.wav'
rock_file = 'H:/Audio Data Analysis/GTZAN Data/genres_original/rock/rock.00000.wav'

#loading the audio files with librosa
blues, sr = librosa.load(blues_file, duration=30)
classical, _ = librosa.load(classical_file, duration=30)
country, _ = librosa.load(country_file, duration=30)
disco, _ = librosa.load(disco_file, duration=30)
hiphop, _ = librosa.load(hiphop_file, duration=30)
jazz, _ = librosa.load(jazz_file, duration=30)
metal, _ = librosa.load(metal_file, duration=30)
pop, _ = librosa.load(pop_file, duration=30)
reggae, _ = librosa.load(reggae_file, duration=30)
rock, _ = librosa.load(rock_file, duration=30)

#Specifying Frame Size and Hop Length
FRAME_SIZE = 1024
HOP_LENGTH = 512

#Extracting zero crossing rate for each music genre song using Librosa
zcr_classical = librosa.feature.zero_crossing_rate(classical, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
zcr_blues = librosa.feature.zero_crossing_rate(blues, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
zcr_reggae = librosa.feature.zero_crossing_rate(reggae, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
zcr_rock = librosa.feature.zero_crossing_rate(rock, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
zcr_jazz = librosa.feature.zero_crossing_rate(jazz, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
zcr_country = librosa.feature.zero_crossing_rate(country, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
zcr_disco = librosa.feature.zero_crossing_rate(disco, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
zcr_hiphop = librosa.feature.zero_crossing_rate(hiphop, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
zcr_metal = librosa.feature.zero_crossing_rate(metal, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
zcr_pop = librosa.feature.zero_crossing_rate(pop, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]


print(f'Blues ZCR: {zcr_blues}')
print(f'Classical ZCR: {zcr_classical}')
print(f'Country ZCR: {zcr_country}')
print(f'Disco ZCR: {zcr_disco}')
print(f'HipHop ZCR: {zcr_hiphop}')
print(f'Jazz ZCR: {zcr_jazz}')
print(f'Metal ZCR: {zcr_metal}')
print(f'Pop ZCR: {zcr_pop}')
print(f'Reggea ZCR: {zcr_reggae}')
print(f'Rock ZCR: {zcr_rock}')


frames = range(len(zcr_classical))
t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)


#Visualizing normalised Zero-Crossing Rate of different music genre songs
plt.figure(figsize=(20,20))

ax = plt.subplot(5, 2, 1)
librosa.display.waveshow(blues, alpha=0.5)
plt.plot(t, zcr_blues, color='b')
plt.ylim((-1,1))
plt.title("Blues Music Genre Song")

plt.subplot(5, 2, 2)
librosa.display.waveshow(classical, alpha=0.5)
plt.plot(t, zcr_classical, color='g')
plt.ylim((-1,1))
plt.title("Classical Music Genre Song")

plt.subplot(5, 2, 3)
librosa.display.waveshow(country, alpha=0.5)
plt.plot(t, zcr_country, color='r')
plt.ylim((-1,1))
plt.title("Country Music Genre Song")

plt.subplot(5, 2, 4)
librosa.display.waveshow(disco, alpha=0.5)
plt.plot(t, zcr_disco, color='y')
plt.ylim((-1,1))
plt.title("Disco Music Genre Song")

plt.subplot(5, 2, 5)
librosa.display.waveshow(disco, alpha=0.5)
plt.plot(t, zcr_disco, color='m')
plt.ylim((-1,1))
plt.title("Disco Music Genre Song")

plt.subplot(5, 2, 6)
librosa.display.waveshow(hiphop, alpha=0.5)
plt.plot(t, zcr_hiphop, color='#E9967A')
plt.ylim((-1,1))
plt.title("Hip Hop Music Genre Song")

plt.subplot(5, 2, 7)
librosa.display.waveshow(jazz, alpha=0.5)
plt.plot(t, zcr_jazz, color='#7FFF00')
plt.ylim((-1,1))
plt.title("Jazz Music Genre Song")

plt.subplot(5, 2, 8)
librosa.display.waveshow(metal, alpha=0.5)
plt.plot(t, zcr_metal, color='#FFB90F')
plt.ylim((-1,1))
plt.title("Metal Music Genre Song")

plt.subplot(5, 2, 9)
librosa.display.waveshow(pop, alpha=0.5)
plt.plot(t, zcr_pop, color='#458B00')
plt.ylim((-1,1))
plt.title("Pop Music Genre Song")

plt.subplot(5, 2, 10)
librosa.display.waveshow(reggae, alpha=0.5)
plt.plot(t, zcr_reggae, color='k')
plt.ylim((-1,1))
plt.title("Reggae Music Genre Song")
plt.subplots_adjust(hspace=1.00)

plt.show()