import os
import shutil
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class AudioFeaturesDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        features = np.load(self.files[idx])
        return torch.tensor(features, dtype=torch.float32)

def copy_files_to_split_folders(dataset, train_dir, test_dir, train_indices, test_indices):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    for idx in train_indices:
        shutil.copy(dataset.files[idx], train_dir)
    
    for idx in test_indices:
        shutil.copy(dataset.files[idx], test_dir)

data_directory = '/home/broiron/Desktop/TPoS/dataset/mels'
train_directory = '/home/broiron/Desktop/TPoS/dataset/mel_train' # train
test_directory = '/home/broiron/Desktop/TPoS/dataset/mel_test' # test


dataset = AudioFeaturesDataset(data_directory)

total_size = len(dataset)
train_size = int(total_size * 0.8)  
test_size = total_size - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

copy_files_to_split_folders(dataset, train_directory, test_directory, train_dataset.indices, test_dataset.indices)


# DataLoader
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)