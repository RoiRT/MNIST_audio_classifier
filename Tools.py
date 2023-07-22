import os
import torch
import librosa
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from IPython.display import Audio

def load_audio_dataset(root_dir):
    file_paths = []
    labels = []

    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(folder_path, file_name)
                    file_paths.append(file_path)
                    labels.append(file_name[0])

    dataset = (file_paths, labels)

    return dataset

def adjust_audio_shape(waveform):
    if len(waveform.shape) > 1:
        return waveform.reshape(-1)

    return waveform

def adjust_audio_duration(waveform, sample_rate, target_duration=0.9):
    max_len = int(target_duration * sample_rate)

    if waveform.shape[0] <= max_len:
        begin_len = random.randint(0, max_len - waveform.shape[0])
        end_len = max_len - waveform.shape[0] - begin_len

        audio = np.concatenate((np.zeros(begin_len), waveform, np.zeros(end_len)), axis=0)
        return audio

    return waveform[:max_len]

def audio_roll(waveform):
    shift = int(random.uniform(-0.5, 0.5) * 0.25 * waveform.shape[0])
    return np.roll(waveform, shift)

def convert_mel_spectrogram(waveform, sample_rate):
    top_db = 80

    spec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)

    # Convert to decibels
    spec_db = librosa.power_to_db(spec, ref=np.max, top_db=top_db)

    return spec_db

class AudioDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        waveform, sample_rate = librosa.load(file_path, sr=None)
        label = self.labels[index]

        return waveform, label