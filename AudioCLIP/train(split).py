"""
1. vggsound dataset(일단 sample로 5개만 사용): raw audio(10s, .wav 형식) -> segment(2s, .wav 형식)
2. 각각의 segment -> audio encoder를 통과 시킴. 이때 차원은 [1, 1024]
2-1. text encoder(CLIP text encoder)로 text embedding 뽑음. 이때 차원은 [1, 77, 1024]
3. 2, 2-1 사이에 MSE를 적용 -> audio embedding[1, 1024]가 MLP를 거쳤을 때 text embedding[1, 77, 1024]가 되도록
"""
""" To-do: validation까지 적용 """

import os
import sys
import glob

import librosa
import librosa.display

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(f'{os.getcwd()}/..'))

from model import AudioCLIP # audio encoder
from model.model_final import FrozenCLIPTextEmbedder # text encoder
from dataset import AudioDataset # dataset

from utils.transforms import ToTensor1D

torch.set_grad_enabled(True)
device = "cuda" if torch.cuda.is_available() else "cpu"

max_length = 77
input_dim = 1024
output_dim = 1024
epochs = 10


class Mapping_Model(nn.Module):
    def __init__(self, input_dim=1024, output_dim=1024, max_length=77):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, output_dim * 2)  
        self.linear2 = nn.Linear(output_dim * 2, output_dim * max_length) 
        self.act = nn.ReLU()  
        self.drop = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.drop(self.act(self.linear1(x)))
        x = self.linear2(x)
        return x.view(-1, max_length, output_dim) 


model = Mapping_Model(input_dim, output_dim, max_length).to(device)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


MODEL_FILENAME = 'AudioCLIP-Full-Training.pt'
SAMPLE_RATE = 44100 


# audio encoder
audio_encoder = AudioCLIP(pretrained=f'../assets/{MODEL_FILENAME}').eval().to(device) # 학습시키는게 아니니까
# text encoder, 나중에 ViT 기반으로 교체
text_encoder = FrozenCLIPTextEmbedder(version='RN50', device=device).eval().to(device)


# 아래 경로는 각각의 10초 raw audio를 2초 segment로 잘라서 저장해놓은 경로
# segments_2_val: 38,430개
audio_segments_path = '/home/broiron/Desktop/AudioCLIP/data/test_2/*.wav' # 여기에 2초짜리 audio segment(.wav)
audio_paths = glob.glob(audio_segments_path)


audio = list()
# audio_data = [] # {}
audio_transforms = ToTensor1D()
audio_data = [audio_transforms(librosa.load(path, sr=SAMPLE_RATE, dtype=np.float32)[0].reshape(1, -1)) for path in tqdm(audio_paths, desc='Audio loading')]

# for path in tqdm(glob.glob(audio_segments_path), desc='Audio loading: '):
#     video_id = '_'.join(path.split('/')[-1].split('_')[:-2])
#     track, _ = librosa.load(path, sr=SAMPLE_RATE, dtype=np.float32)
#     transformed_track = audio_transforms(track.reshape(1, -1))
    
#     if video_id not in audio_data:
#         audio_data[video_id] = [transformed_track]
#     else:
#         audio_data[video_id].append(transformed_track)


# Load labels from file
# dataset 이름에서 바로 추출해야 함, 중복 제거 없이
labels_path = '/home/broiron/Desktop/AudioCLIP/data/label/test_labels.txt'
with open(labels_path, 'r') as file:
    labels = [line.strip() for line in file]


### train / validation 분리
audio_data_train, audio_data_val, labels_train, labels_val = train_test_split(audio_data, labels, test_size=0.2, random_state=42)

# Audio encoding function
def encode_audio_data(audio_data_list):
    encoded_features = []
    for audio_sample in tqdm(audio_data_list, desc='Audio encoding: '):
        with torch.no_grad():
            ((audio_features, _, _), _), _ = audio_encoder(audio=audio_sample)
            # audio_features = audio_encoder(audio_sample.unsqueeze(0).to(device))[0]
            print(f'encoder 통과한 audio embedding shape: {audio_features.shape}') # [1, 1024]
            encoded_features.append(audio_features)
    return torch.stack(encoded_features)

# train, validation data 각각 encoding 처리
audio_features_train = encode_audio_data(audio_data_train)
audio_features_val = encode_audio_data(audio_data_val)



# # audio encoder 통과
# all_audio_features = [] 
# for i, (video_id, segments) in tqdm(enumerate(audio_data.items()), desc='Audio processing: '):
#     for segment in segments:
#         audio_sample = segment.unsqueeze(0)
#         # print(f'통과 전 audio feature shape: {audio_sample.shape}') # (1, 1, 88200)
        
#         with torch.no_grad():
#             ((audio_features, _, _), _), _ = audio_encoder(audio=audio_sample)
#         print(f'encoder 통과한 audio embedding shape {i+1}: {audio_features.shape}') # [1, 1024]
#         all_audio_features.append(audio_features) 
        
# all_audio_features = torch.stack(all_audio_features) # audio embedding


batch_size = 1
shuffle = True
num_workers = 0

# Data Loader
train_dataset = AudioDataset(audio_features_train, labels_train)
val_dataset = AudioDataset(audio_features_val, labels_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def evaluate(model, data_loader, loss_function):
    model.eval()
    total_loss = 0
    with torch.no_grad(): 
        for audio_features, labels_batch in data_loader:
            audio_features = audio_features.to(device).float()
            text_embeddings = torch.stack([text_encoder.encode([label]).to(device).float() for label in labels_batch])

            predictions = model(audio_features)
            loss = loss_function(predictions, text_embeddings.squeeze(1))
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss



""" Audio embedding과 Text embedding MSE 적용"""
patience = 20
cnt = 0
best_loss = float('inf')  # 최소 손실 값을 저장하기 위한 변수 초기화
model_save_path = '/home/broiron/Desktop/AudioCLIP/assets/train_best_model.pth'  

# training
model.train()
for epoch in range(epochs):
    total_loss = 0
    for audio_features, labels_batch in train_loader:
        audio_features = audio_features.to(device).float() 

        # text embedding 추출
        text_embeddings = torch.stack([text_encoder.encode([label]).to(device).float() for label in labels_batch])
        print(f'text embedding shape: {text_embeddings.shape}')

        optimizer.zero_grad()
        predictions = model(audio_features) # MLP 통과한 embedding
        loss = loss_function(predictions, text_embeddings.squeeze(1)) # 앞서 stack으로 쌓아서 차원이 하나 더 생김(?) -> squeeze로 제거
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    ### Validation
    avg_val_loss = evaluate(model, val_loader, loss_function)

    print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss}, Val Loss = {avg_val_loss}')

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), model_save_path)
        print(f"Best model saved at epoch: {epoch+1}, loss: {best_loss}")
        cnt = 0
    else:
        cnt += 1

    if cnt >= patience:
        print(f"Early stopping {epoch+1}")
        break
    
