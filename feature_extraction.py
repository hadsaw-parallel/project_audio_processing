import numpy as np
import librosa as lb
from scipy.fftpack import dct

def extract_feature(audios, Fs, audio_length, n_fft, win_size, hop_size, n_mels, window_name="hamming"):
    features = []
    for i in range(len(audios)):
        feature = dict()
        audio, fs, label = audios[i]
        
        # Check if audio is a 1d array
        if (audio.ndim == 1):
            # Resample the audio to a fixed sampling rate Fs
            if fs != Fs:
                audio = lb.resample(audio, orig_sr=fs, target_sr=Fs)
            
            # Trim the audio to a fixed length of 5 seconds
            if len(audio) != audio_length:
                audio = audio[:audio_length]

            # Add checks for valid audio data
            if len(audio) == 0 or np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
                print(f"Skipping invalid audio sample")
                continue
            
            # Safer normalization
            if np.max(audio) != np.min(audio):
                audio = 2 * ((audio - np.min(audio)) / (np.max(audio) - np.min(audio))) - 1
            else:
                audio = np.zeros_like(audio)
            
            # 2.1) Mel spectrogram in log scale
            mel_spectro = lb.feature.melspectrogram(
                y=audio, sr=Fs, n_fft=n_fft, n_mels=n_mels, 
                win_length=win_size, window=window_name, hop_length=hop_size)
            
            mel_spectro_db = lb.power_to_db(mel_spectro, ref=np.max)

            # 2.2) MFCC
            #mfcc = lb.feature.mfcc(
            #    y=audio, sr=Fs, n_mfcc=n_mels, n_mels=n_mels, n_fft=n_fft, 
            #    win_length=win_size, window="hamming", hop_length=hop_size)
            mfcc = dct(mel_spectro_db, axis=0)
            
            # 2.3) RMS
            rms = lb.feature.rms(y=audio, frame_length=audio_length//hop_size, hop_length=hop_size)
            
            # 2.4) zcr
            zcr = lb.feature.zero_crossing_rate(y=audio, frame_length=audio_length//hop_size, hop_length=hop_size)

            # Append the features to car_features
            feature["mel"] = mel_spectro_db
            feature["mfcc"] = mfcc
            feature["rms"] = rms
            feature["zcr"] = zcr

            features.append((feature, label))

    
    return features