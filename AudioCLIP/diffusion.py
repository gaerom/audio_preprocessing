""" MLP를 거쳐 mapping 된 text embedding을 input으로 사용하여 diffusion model output 확인해보기 
https://huggingface.co/stabilityai/stable-diffusion-2-1-base """

import torch
import numpy as np
import librosa
import sys
import os
import glob
sys.path.append(os.path.abspath(f'{os.getcwd()}/..'))
from model import AudioCLIP
from train_mlp import Mapping_Model

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler # diffusion

max_length = 77
input_dim = 1024
output_dim = 1024

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_FILENAME = 'AudioCLIP-Full-Training.pt'
audio_encoder = AudioCLIP(pretrained=f'../assets/{MODEL_FILENAME}')

# 학습때 사용한 model instance 생성
model_instance = Mapping_Model(input_dim, output_dim, max_length).to(device)

# 학습때 얻은 가중치 load
model_path = '/home/broiron/Desktop/AudioCLIP/assets/model.pth' 
state_dict = torch.load(model_path)

model_instance.load_state_dict(state_dict) # 이렇게 해야 eval이 가능

# baseline (수정 필요)
model_instance.eval()
audio_data = glob.glob('/home/broiron/Desktop/AudioCLIP/data/real_test/*.wav')
sample_rate = 44100 # 학습때와 동일

# output embedding 저장할 list
embeddings = []
audio_encoder.eval()

for audio_file in audio_data:

    audio, _ = librosa.load(audio_file, sr=sample_rate)
    audio_tensor = torch.tensor(audio).unsqueeze(0).float().to(device)
    
    with torch.no_grad():  # 학습 x
        # audio encoder 통과
        audio_outputs = audio_encoder(audio_tensor)
        audio_feature = audio_outputs[0].squeeze(0) 
        
        mlp_embedding = model_instance(audio_feature.unsqueeze(0))
        embeddings.append(mlp_embedding.squeeze().cpu().numpy())

# embedding 저장
all_mlp_embeddings_array = np.array(embeddings)
embeddings_tensor = torch.from_numpy(all_mlp_embeddings_array).float() # embedding -> tensor로 변환
embeddings_tensor = embeddings_tensor.to(device)



# disk에 저장
# embeddings_path = '/embedding 저장 경로/embeddings.npy'
# np.save(embeddings_path, all_mlp_embeddings_array)


""" input dim [1, 77, 1024]를 갖는 diffusion model과 연결 """
""" prerequisite """
# pip install diffusers transformers accelerate scipy safetensors (설치 완료)
 
model_id = "stabilityai/stable-diffusion-2-1-base" # hugging face model id
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

# 수정해야 함 -> embedding을 구하지 않고 그냥 return할 수 있도록(__call__ 수정 완료)
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
