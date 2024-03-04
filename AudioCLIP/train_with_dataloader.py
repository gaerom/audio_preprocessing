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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(f'{os.getcwd()}/..'))

from model import AudioCLIP # audio encoder
from model.model_final import FrozenCLIPTextEmbedder # text encoder
from dataset import AudioTextDataset


from utils.transforms import ToTensor1D

torch.set_grad_enabled(True)
device = "cuda" if torch.cuda.is_available() else "cpu"

max_length = 77
input_dim = 1024
output_dim = 1024
epochs = 50

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
SAMPLE_RATE = 44100 # 초당 44100개의 sample을 사용하여 audio signal을 디지털화

# UnAV-100 annotation과 같은 역할(VGGSound/data/Common.txt에 있는 label과 동일)
# LABELS = ['beat boxing', 'cat purring', 'cattle, bovinae cowbell', 'fire truck siren', 'playing violin, fiddle'] # vggsound sample audio에 해당하는 label

# audio encoder
audio_encoder = AudioCLIP(pretrained=f'../assets/{MODEL_FILENAME}')
audio_encoder.eval() # 어차피 audio encoder를 학습시키는게 아니니까

# text encoder
text_encoder = FrozenCLIPTextEmbedder(version='RN50', device=device)

#paths_to_audio = glob.glob('./vggsound/raw_audios/*.wav')
# 아래 경로는 각각의 10초 raw audio를 2초 segment로 잘라서 저장해놓은 경로
# 40만개는 load하다가 process 죽음 -> segments_2_val: 38,430개
audio_segments_path = '/home/broiron/Desktop/AudioCLIP/data/segments_2_val/*.wav' # 여기에 2초짜리 audio segment(.wav)

audio = list()
audio_data = {}
audio_transforms = ToTensor1D() # audio feature를 1D tensor로 변환하기 위한 것, 왜..? -> 일단 pytorch에서 다루기 쉽게 만들기 위해서
for path in tqdm(glob.glob(audio_segments_path), desc='Audio loading: '):
    video_id = '_'.join(path.split('/')[-1].split('_')[:-2])
    track, _ = librosa.load(path, sr=SAMPLE_RATE, dtype=np.float32)
    transformed_track = audio_transforms(track.reshape(1, -1))
    
    if video_id not in audio_data:
        audio_data[video_id] = [transformed_track]
    else:
        audio_data[video_id].append(transformed_track)

print(device)

# Load labels from file
labels_path = '/home/broiron/Desktop/AudioCLIP/data/label/labels.txt'
with open(labels_path, 'r') as file:
    labels = [line.strip() for line in file]

dataset = AudioTextDataset(audio_paths=audio_segments_path, labels=labels, transform=ToTensor1D())

all_audio_features = [] 
for i, (video_id, segments) in tqdm(enumerate(audio_data.items()), desc='Audio processing: '):
    for segment in segments:
        audio_sample = segment.unsqueeze(0)
        # print(f'통과 전 audio feature shape: {audio_sample.shape}') # (1, 1, 88200)
        # audio encoder 통과
        with torch.no_grad():
            ((audio_features, _, _), _), _ = audio_encoder(audio=audio_sample)
        # print(f'encoder 통과한 audio embedding shape {i+1}: {audio_features.shape}')
        all_audio_features.append(audio_features.squeeze(0)) 

all_audio_features = torch.stack(all_audio_features)

# Dataloader
train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

""" Audio embedding과 Text embedding MSE 적용"""
patience = 20
cnt = 0
best_loss = float('inf')  # 최소 손실 값을 저장하기 위한 변수 초기화
model_save_path = '/home/broiron/Desktop/AudioCLIP/assets/train_best_model.pth'  

model.train()
for epoch in range(epochs):
    total_loss = 0
    for audio_features, labels_batch in train_dataloader:
        audio_features = audio_features.to(device)

        # 레이블에 대한 텍스트 임베딩 생성
        text_embeddings = torch.stack([text_encoder.encode([label]).to(device) for label in labels_batch])

        optimizer.zero_grad()
        outputs = model(audio_features)
        loss = loss_function(outputs, text_embeddings)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch+1}, Loss: {avg_loss}')

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), model_save_path)
        print(f"Best model saved at epoch: {epoch+1}, loss: {best_loss}")
        cnt = 0  
    else:
        cnt += 1

    if cnt >= patience:
        print(f"Early stopping {epoch+1}")
        break


# ### Normalize Embedding
# all_audio_features = all_audio_features / torch.linalg.norm(all_audio_features, dim=-1, keepdim=True)
# text_features = text_features / torch.linalg.norm(text_features, dim=-1, keepdim=True)

# ### Similarity 계산
# scale_audio_text = torch.clamp(aclp.logit_scale_at.exp(), min=1.0, max=100.0)
# logits_audio_text = scale_audio_text * all_audio_features @ text_features.T

# # Audio Classification
# confidence = logits_audio_text.softmax(dim=1)

# # 출력 부분 수정
# print('\t\tFilename, Audio\t\t\tTextual Label (Confidence)', end='\n\n')
# for audio_idx, path in enumerate(paths_to_audio_segments):
#     conf_values, ids = confidence[audio_idx].topk(3)
#     query = f'{os.path.basename(path):>30s} ->\t\t'
#     results = ', '.join([f'{LABELS[i]:>15s} ({v:06.2%})' for v, i in zip(conf_values, ids)])
#     print(query + results)
