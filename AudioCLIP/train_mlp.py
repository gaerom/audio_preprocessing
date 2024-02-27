import os
import sys
import glob

import librosa
import librosa.display

import numpy as np

import torch
import torchvision as tv

import matplotlib.pyplot as plt

from PIL import Image
from IPython.display import Audio, display

sys.path.append(os.path.abspath(f'{os.getcwd()}/..'))

from model import AudioCLIP # audio encoder
from model.model_final import FrozenCLIPTextEmbedder # text encoder
from utils.transforms import ToTensor1D


torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_FILENAME = 'AudioCLIP-Full-Training.pt'
# derived from ESResNeXt
SAMPLE_RATE = 44100 # 초당 44100개의 sample을 사용하여 audio signal을 디지털화

# UnAV-100 annotation과 같은 것(VGGSound/data/Common.txt에 있는 label과 동일)
LABELS = ['beat boxing', 'cat purring', 'cattle, bovinae cowbell', 'fire truck siren', 'playing violin, fiddle'] # vggsound sample audio에 해당하는 label

# audio encoder
audio_encoder = AudioCLIP(pretrained=f'../assets/{MODEL_FILENAME}')
audio_encoder.eval() # 어차피 audio encoder를 학습시키는게 아니니까

# text encoder
text_encoder = FrozenCLIPTextEmbedder(version='RN50', device=device)

#paths_to_audio = glob.glob('./vggsound/raw_audios/*.wav') # audio input은 .wav / vggsound data로 경로 수정
paths_to_audio_segments = glob.glob('./vggsound/segments/*.wav')

audio = list()
audio_data = {}
audio_transforms = ToTensor1D() # audio feature를 1D tensor로 변환하기 위한 것, 왜..? -> 일단 pytorch에서 다루기 쉽게 만들기 위해서
for path in paths_to_audio_segments:
    video_id = '_'.join(path.split('/')[-1].split('_')[:-2])
    
    track, _ = librosa.load(path, sr=SAMPLE_RATE, dtype=np.float32)
    transformed_track = audio_transforms(track.reshape(1, -1))
    
    if video_id not in audio_data:
        audio_data[video_id] = [transformed_track]
    else:
        audio_data[video_id].append(transformed_track)


all_audio_features = [] 

for i, (video_id, segments) in enumerate(audio_data.items()):
    for segment in segments:
        audio_sample = segment.unsqueeze(0)
        print(f'통과 전 audio feature shape: {audio_sample.shape}') # (1, 1, 88200)
        # audio encoder 통과
        with torch.no_grad():
            ((audio_features, _, _), _), _ = audio_encoder(audio=audio_sample)
        print(f'encoder 통과한 audio embedding shape {i+1}: {audio_features.shape}')
        all_audio_features.append(audio_features.squeeze(0)) 

all_audio_features = torch.stack(all_audio_features)

# text = [[label] for label in LABELS] 
for i in LABELS:
    text_features = text_encoder.encode(i)
    print(f'Text embedding shape(GT): {text_features.shape}')  # (1, 77, 1024)


""" Audio embedding과 Text embedding MSE 적용"""






























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



"""
1. audio encoder를 통과 시켰음 -> audio embedding 차원 확인 완료
2. text encoder(CLIP text encoder)로 text embedding 뽑아야 함



3. 1, 2 사이의 MSE 적용해야 함
"""