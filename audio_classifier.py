import os
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
from IPython.display import Audio


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



'''data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
'''
'''for batch_waveforms, batch_labels in data_loader:
    print(batch_labels)
    break'''
