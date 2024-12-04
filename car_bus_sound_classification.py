import numpy as np
import soundfile as sf
import sounddevice as sd
import librosa as lb
from scipy.fftpack import fft, ifft, dct
from scipy import signal
import matplotlib.pyplot as plt
import os
from feature_extraction import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier


# ----------------------------------Reading and labeling input data------------------------------------
# 1) Read sound files
# 2) Add label to read sound files as tuple (sound, fs, label) for each file: 1 for car, 0 for bus
# -----------------------------------------------------------------------------------------------------
def load_audio_files(car_dir, bus_dir):
    """
    Load audio files from specified car and bus directories
    
    Args:
        car_dir (str): Path to directory containing car audio files
        bus_dir (str): Path to directory containing bus audio files
        
    Returns:
        tuple: Lists of car and bus audio data as (audio_data, sample_rate, label)
    """
    car = []  # [(sound, fs, label)]
    bus = []  # [(sound, fs, label)]
    
    # Load car audio files and label as 1
    for filename in os.listdir(car_dir):
        if filename.endswith(('.wav', '.WAV')):
            file_path = os.path.join(car_dir, filename)
            car_audio, car_sr = sf.read(file_path)
            car.append((car_audio, car_sr, 1))
    
    # Load bus audio files and label as 0
    for filename in os.listdir(bus_dir):
        if filename.endswith(('.wav', '.WAV')):
            file_path = os.path.join(bus_dir, filename)
            bus_audio, bus_sr = sf.read(file_path)
            bus.append((bus_audio, bus_sr, 0))
            
    print(f'Loaded {len(car)} car audios and {len(bus)} bus audios.')
    return car, bus

# Usage example:
# Load training data
train_car_dir = "dataset/car-sounds/training"
train_bus_dir = "dataset/bus-sounds/training"
train_car, train_bus = load_audio_files(train_car_dir, train_bus_dir)

# Load validation data
val_car_dir = "dataset/car-sounds/validation"
val_bus_dir = "dataset/bus-sounds/validation"
val_car, val_bus = load_audio_files(val_car_dir, val_bus_dir)

# Load test data
test_car_dir = "dataset/car-sounds/testing"
test_bus_dir = "dataset/bus-sounds/testing"
test_car, test_bus = load_audio_files(test_car_dir, test_bus_dir)


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

# Extract features for training data
train_car_features = extract_feature(audios=train_car, Fs=Fs, audio_length=L, n_fft=n_fft, win_size=win_size, hop_size=hop_size, n_mels=n_mels)
train_bus_features = extract_feature(audios=train_bus, Fs=Fs, audio_length=L, n_fft=n_fft, win_size=win_size, hop_size=hop_size, n_mels=n_mels)

# Extract features for validation data
val_car_features = extract_feature(audios=val_car, Fs=Fs, audio_length=L, n_fft=n_fft, win_size=win_size, hop_size=hop_size, n_mels=n_mels)
val_bus_features = extract_feature(audios=val_bus, Fs=Fs, audio_length=L, n_fft=n_fft, win_size=win_size, hop_size=hop_size, n_mels=n_mels)

# Extract features for testing data
test_car_features = extract_feature(audios=test_car, Fs=Fs, audio_length=L, n_fft=n_fft, win_size=win_size, hop_size=hop_size, n_mels=n_mels)
test_bus_features = extract_feature(audios=test_bus, Fs=Fs, audio_length=L, n_fft=n_fft, win_size=win_size, hop_size=hop_size, n_mels=n_mels)

print("Feature extraction done.")
print(f'Train data: usable car audios: {len(train_car_features)}, usable bus audios: {len(train_bus_features)}')
print(f'Validation data: usable car audios: {len(val_car_features)}, usable bus audios: {len(val_bus_features)}')
print(f'Test data: usable car audios: {len(test_car_features)}, usable bus audios: {len(test_bus_features)}')
print(f'train_car_features[0] ->', train_car_features[0][0].keys(), f', label={train_car_features[0][1]}', "\n")

# Combine features for each dataset separately
train_features = train_car_features + train_bus_features
val_features = val_car_features + val_bus_features
test_features = test_car_features + test_bus_features

# Prepare training data with error handling and debugging
X_train = []
max_length = 0  # Track the maximum length of feature vectors

for features in train_features:
    try:
        feature_vector = []
        for v in features[0].values():
            if isinstance(v, np.ndarray):
                feature_vector.append(v.flatten())
            else:
                feature_vector.append(np.array([v]))
        concatenated_vector = np.concatenate(feature_vector)
        X_train.append(concatenated_vector)
        max_length = max(max_length, len(concatenated_vector))
    except Exception as e:
        print(f"Error processing feature: {features[0].keys()}")
        print(f"Feature shapes: {[v.shape if isinstance(v, np.ndarray) else type(v) for v in features[0].values()]}")
        raise e

# Pad feature vectors to ensure consistent length
X_train = np.array([np.pad(x, (0, max_length - len(x)), 'constant') for x in X_train])
y_train = np.array([label for _, label in train_features])

# Create labels for validation data
y_val = np.array([label for _, label in val_features])

# Create labels for test data
y_test = np.array([label for _, label in test_features])

# Apply the same pattern for validation and test data
X_val = []
for features in val_features:
    feature_vector = []
    for v in features[0].values():
        if isinstance(v, np.ndarray):
            feature_vector.append(v.flatten())
        else:
            feature_vector.append(np.array([v]))
    concatenated_vector = np.concatenate(feature_vector)
    X_val.append(np.pad(concatenated_vector, (0, max_length - len(concatenated_vector)), 'constant'))
X_val = np.array(X_val)

X_test = []
for features in test_features:
    feature_vector = []
    for v in features[0].values():
        if isinstance(v, np.ndarray):
            feature_vector.append(v.flatten())
        else:
            feature_vector.append(np.array([v]))
    concatenated_vector = np.concatenate(feature_vector)
    X_test.append(np.pad(concatenated_vector, (0, max_length - len(concatenated_vector)), 'constant'))
X_test = np.array(X_test)


# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


print("#"*20, "Support Vector Machine model", "#"*20)
# Create and train SVM model
svm_model = SVC(
    kernel='rbf',  # Try 'linear', 'poly', or 'sigmoid'
    C=1.0,        # Try different values like 0.1, 1, 10, 100
    gamma='scale', # Try 'auto' or specific values
    class_weight='balanced', # Add this to handle class imbalance
    random_state=42
)
svm_model.fit(X_train_scaled, y_train)

# Evaluate on validation set
val_predictions = svm_model.predict(X_val_scaled)
val_accuracy = accuracy_score(y_val, val_predictions)
print("\nValidation Set Performance:")
print(f"Validation Accuracy: {val_accuracy:.4f} or {val_accuracy*100:.2f}%")
print("\nValidation Classification Report:")
print(classification_report(y_val, val_predictions, 
                          target_names=['Bus', 'Car'],
                          zero_division=0))

# Final evaluation on test set
test_predictions = svm_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, test_predictions)
print("\nTest Set Performance:")
print(f"Test Accuracy: {test_accuracy:.4f} or {test_accuracy*100:.2f}%")
print("\nTest Classification Report:")
print(classification_report(y_test, test_predictions, 
                          target_names=['Bus', 'Car'],
                          zero_division=0))
print("#"*100)


# Also, let's add some diagnostic information
print("\nData Distribution:")
print(f"Training set - Bus: {sum(y_train == 0)}, Car: {sum(y_train == 1)}")
print(f"Validation set - Bus: {sum(y_val == 0)}, Car: {sum(y_val == 1)}")
print(f"Test set - Bus: {sum(y_test == 0)}, Car: {sum(y_test == 1)}")


print()
print("#"*20, "Random Forest model", "#"*20)
# Or try other classifiers
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)

# Evaluate on validation set
val_predictions = rf_model.predict(X_val_scaled)
val_accuracy = accuracy_score(y_val, val_predictions)
print("\nValidation Set Performance:")
print(f"Validation Accuracy: {val_accuracy:.4f} or {val_accuracy*100:.2f}%")
print("\nValidation Classification Report:")
print(classification_report(y_val, val_predictions, 
                          target_names=['Bus', 'Car'],
                          zero_division=0))

# Final evaluation on test set
test_predictions = rf_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, test_predictions)
print("\nTest Set Performance:")
print(f"Test Accuracy: {test_accuracy:.4f} or {test_accuracy*100:.2f}%")
print("\nTest Classification Report:")
print(classification_report(y_test, test_predictions, 
                          target_names=['Bus', 'Car'],
                          zero_division=0))
print("#"*100)


def plot_learning_curves(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Plot learning curves for training, validation, and test sets
    """
    train_scores = []
    val_scores = []
    test_scores = []
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    # Get indices for each class
    class_0_idx = np.where(y_train == 0)[0]
    class_1_idx = np.where(y_train == 1)[0]
    
    for size in train_sizes:
        try:
            # Calculate how many samples we need from each class
            n_samples = int(len(X_train) * size)
            n_samples_per_class = n_samples // 2
            
            # Get balanced subset of indices
            subset_0_idx = np.random.choice(class_0_idx, n_samples_per_class, replace=False)
            subset_1_idx = np.random.choice(class_1_idx, n_samples_per_class, replace=False)
            subset_idx = np.concatenate([subset_0_idx, subset_1_idx])
            
            # Create balanced subset
            X_train_subset = X_train[subset_idx]
            y_train_subset = y_train[subset_idx]
            
            # Train model on subset
            model.fit(X_train_subset, y_train_subset)
            
            # Calculate scores
            train_scores.append(accuracy_score(y_train_subset, model.predict(X_train_subset)))
            val_scores.append(accuracy_score(y_val, model.predict(X_val)))
            test_scores.append(accuracy_score(y_test, model.predict(X_test)))
        except Exception as e:
            print(f"Error at size {size}: {str(e)}")
            continue
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes * 100, train_scores, label='Training', marker='o')
    plt.plot(train_sizes * 100, val_scores, label='Validation', marker='s')
    plt.plot(train_sizes * 100, test_scores, label='Test', marker='^')
    
    plt.xlabel('Percentage of Training Data')
    plt.ylabel('Accuracy Score')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot learning curves for SVM model
print("\nPlotting Learning Curves for SVM Model:")
plot_learning_curves(
    SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42),
    X_train_scaled, y_train,
    X_val_scaled, y_val,
    X_test_scaled, y_test
)

# Plot learning curves for Random Forest model
print("\nPlotting Learning Curves for Random Forest Model:")
plot_learning_curves(
    RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    X_train_scaled, y_train,
    X_val_scaled, y_val,
    X_test_scaled, y_test
)





