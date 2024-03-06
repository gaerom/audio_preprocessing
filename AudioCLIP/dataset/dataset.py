from torch.utils.data import Dataset
import librosa


class AudioDataset(Dataset):
    def __init__(self, audio_features, labels):
        self.audio_features = audio_features
        self.labels = labels

    def __len__(self):
        return len(self.audio_features)

    def __getitem__(self, idx):
        # audio feature와 lable을 반환
        return self.audio_features[idx], self.labels[idx]
