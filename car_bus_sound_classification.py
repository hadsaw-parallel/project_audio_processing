import numpy as np
import soundfile as sf
import sounddevice as sd
import librosa as lb
from scipy.fftpack import fft, ifft, dct
from scipy import signal
import matplotlib.pyplot as plt
import os
from feature_extraction import *


# Reading and labeling input data
# 1) Read sound files
# 2) Add label to read sound files as tuple (sound, fs, label) for each file: 0 for car, 1 for bus
car = [] # [(sound, fs, label)]
bus = [] # [(sound, fs, label)]

# Define directories containing multiple audio files
car_dir = "dataset/car-sounds"    # Replace with your car audio directory
bus_dir = "dataset/bus-sounds"    # Replace with your bus audio directory

# Load car audio files and label as 1
for filename in os.listdir(car_dir):
    if filename.endswith(('.wav', '.WAV')):  # Add more audio formats if needed
        file_path = os.path.join(car_dir, filename)
        car_audio, car_sr = sf.read(file_path)
        car.append((car_audio, car_sr, 1))    # Label 1 for car

# Load bus audio files and label as 0
for filename in os.listdir(bus_dir):
    if filename.endswith(('.wav', '.WAV')):  # Add more audio formats if needed
        file_path = os.path.join(bus_dir, filename)
        bus_audio, bus_sr = sf.read(file_path)
        bus.append((bus_audio, bus_sr, 0))    # Label 0 for bus

#print(f"Loaded {len(car)} car audio files and {len(bus)} bus audio files")
#print(car[0])
#print(bus[0])
#sd.play(car[0][0], car[0][1])
#sd.wait()
print(f'Finish reading {len(car)} car audios  and {len(bus)} bus audios.\n')


# Preprocessing
# 1) Normalize the dataset
# 2) Feature extractions:
# 2.1) Mel Spectrogram
# 2.2) MFCC
# 2.3) Energy: RMS
# 2.4) Zero-crossing rate (zcr)
# output example: list[ dict(mel spectrogram, MFCC, RMS, zcr), ...]
Fs = 44100
L = Fs*5
n_fft = 2048
win_size = 1024
hop_size = win_size//2
n_mels = 128

car_features = extract_feature(audios=car, Fs=Fs, audio_length=L, n_fft=n_fft, win_size=win_size, hop_size=hop_size, n_mels=n_mels)
bus_features = extract_feature(audios=bus, Fs=Fs, audio_length=L, n_fft=n_fft, win_size=win_size, hop_size=hop_size, n_mels=n_mels)

print("Feature extraction done.")
print(f'Usable car audios: {len(car_features)}. Usable bus audios: {len(bus_features)}')
print(f'car_features[0] ->', car_features[0].keys(), "\n")





# Split the dataset into train, validation, and test data
# 1) Train dataset
# 2) Validation dataset
# 3) Test dataset



# Specify the model
