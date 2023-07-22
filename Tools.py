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
    spec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=64)

    spec_db = librosa.power_to_db(spec, ref=np.max, top_db=80)

    return spec_db

def data_masking(spec, per_mask=0.05, n_freq_masks=1, n_time_masks=1):
    n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec.copy()

    freq_mask_param = int(per_mask * n_mels)
    for _ in range(n_freq_masks):
        f_start = np.random.randint(0, n_mels - freq_mask_param)
        aug_spec[f_start:f_start + freq_mask_param, :] = mask_value

    time_mask_param = int(per_mask * n_steps)
    for _ in range(n_time_masks):
        t_start = np.random.randint(0, n_steps - time_mask_param)
        aug_spec[:, t_start:t_start + time_mask_param] = mask_value

    return aug_spec

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

        waveform_sh = adjust_audio_shape(waveform)
        waveform_adp = adjust_audio_duration(waveform_sh, sample_rate, target_duration=0.9)
        waveform_rld = audio_roll(waveform_adp)
        spec = convert_mel_spectrogram(waveform_rld, sample_rate)
        spec_mask = data_masking(spec, per_mask=0.05, n_freq_masks=1, n_time_masks=1)

        return spec_mask, label