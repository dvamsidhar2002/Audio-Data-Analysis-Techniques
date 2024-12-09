# importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import librosa 

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

# Extracting root mean square (RMS) energy for each genre using Librosa
rms_classical = librosa.feature.rms(y=classical, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
rms_blues = librosa.feature.rms(y=blues, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
rms_reggae = librosa.feature.rms(y=reggae, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
rms_rock = librosa.feature.rms(y=rock, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
rms_jazz = librosa.feature.rms(y=jazz, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
rms_country = librosa.feature.rms(y=country, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
rms_disco = librosa.feature.rms(y=disco, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
rms_hiphop = librosa.feature.rms(y=hiphop, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
rms_metal = librosa.feature.rms(y=metal, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
rms_pop = librosa.feature.rms(y=pop, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]

# Print the RMS values for each genre
print(f'Blues rms: {rms_blues}')
print(f'Classical rms: {rms_classical}')
print(f'Country rms: {rms_country}')
print(f'Disco rms: {rms_disco}')
print(f'HipHop rms: {rms_hiphop}')
print(f'Jazz rms: {rms_jazz}')
print(f'Metal rms: {rms_metal}')
print(f'Pop rms: {rms_pop}')
print(f'Reggaa rms: {rms_reggae}')
print(f'Rock rms: {rms_rock}')
