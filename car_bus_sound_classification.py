#import numpy as np
import soundfile as sf
import sounddevice as sd
import librosa as lb
from scipy.fftpack import fft, ifft
from scipy import signal
import matplotlib.pyplot as plt
import os


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

print(f"Loaded {len(car)} car audio files and {len(bus)} bus audio files")
print(car[0])
print(bus[0])
sd.play(car[0][0], car[0][1])
sd.wait()

# Preprocessing
# 1) Normalize the dataset
# 2) Feature extractions:
# 2.1) Mel Spectrogram
# 2.2) MFCC
# 2.3) Energy: RMS
# 2.4) Zero-crossing rate (zcr)
# output example: [mel spectrogram, MFCC, RMS, zcr]
car_features = []
bus_features = []




# Split the dataset into train, validation, and test data
# 1) Train dataset
# 2) Validation dataset
# 3) Test dataset



# Specify the model
