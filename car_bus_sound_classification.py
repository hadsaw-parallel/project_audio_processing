import numpy as np
import soundfile as sf
import sounddevice as sd
import librosa as lb
from scipy.fftpack import fft, ifft, dct
from scipy import signal
import matplotlib.pyplot as plt
import os
from feature_extraction import *


# ----------------------------------Reading and labeling input data------------------------------------
# 1) Read sound files
# 2) Add label to read sound files as tuple (sound, fs, label) for each file: 1 for car, 0 for bus
# -----------------------------------------------------------------------------------------------------
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


# ---------------------------------------Preprocessing -------------------------------------
# 1) Resample the audio signal to a fixed sampling rate
# 2) Normalize the dataset
# 3) Feature extractions:
# 3.1) Mel Spectrogram
# 3.2) MFCC
# 3.3) Energy: RMS
# 3.4) Zero-crossing rate (zcr)
# output example: list[ (dict(mel spectrogram, MFCC, RMS, zcr), label), ...]
# -------------------------------------------------------------------------------------------
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
print(f'car_features[0] ->', car_features[0][0].keys(),  f', label={car_features[0][1]}', "\n")

# Combine car and bus features into one dataset
all_features = car_features + bus_features

# Separate features and labels
X = np.array([np.concatenate([v.flatten() if isinstance(v, np.ndarray) else [v] 
              for v in features[0].values()]) for features in all_features])
y = np.array([label for _, label in all_features])

# Split the dataset into train, validation, and test sets (60-20-20 split)
from sklearn.model_selection import train_test_split

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Train SVM model
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Create and train SVM model
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svm_model.predict(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Model Performance Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Bus', 'Car']))




# ---------------------------Split the dataset into train, validation, and test data-----------------------
# 1) Train dataset
# 2) Validation dataset
# 3) Test dataset
# ---------------------------------------------------------------------------------------------------------



# Specify the model
