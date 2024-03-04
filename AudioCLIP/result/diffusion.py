""" MLP를 거쳐 mapping 된 text embedding을 input으로 사용하여 diffusion model output 확인해보기 
https://huggingface.co/stabilityai/stable-diffusion-2-1-base """

import torch
import numpy as np
import librosa

from model import AudioCLIP

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler # diffusion

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_FILENAME = 'AudioCLIP-Full-Training.pt'
audio_encoder = AudioCLIP(pretrained=f'../assets/{MODEL_FILENAME}')

# 학습때 얻은 가중치 load
model_path = 'model/path/.pth' # 아직 생성 X
model = torch.load(model_path)

# baseline (수정 필요)
model.eval()
audio_data = 'audio segment 몇개 사용'
sample_rate = 44100 # 학습때와 동일

# output embedding 저장할 list
embeddings = []

for audio_file in audio_data:

    audio, _ = librosa.load(audio_file, sr=sample_rate)
    audio_tensor = torch.tensor(audio).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        # audio encoder 통과
        audio_feature = audio_encoder(audio_tensor)[0][0] 
        
        mlp_embedding = model(audio_feature.unsqueeze(0))
        embeddings.append(mlp_embedding.squeeze(0).cpu().numpy())

# embedding 저장
all_mlp_embeddings_array = np.array(embeddings)
embeddings_tensor = torch.from_numpy(all_mlp_embeddings_array).float() # embedding -> tensor로 변환
embeddings_tensor = embeddings_tensor.to(device)

# disk에 저장
# embeddings_path = '/embedding 저장 경로/embeddings.npy'
# np.save(embeddings_path, all_mlp_embeddings_array)


""" input dim [1, 77, 1024]를 갖는 diffusion model과 연결 """
""" prerequisite """
# pip install diffusers transformers accelerate scipy safetensors
 
model_id = "stabilityai/stable-diffusion-2-1-base" # hugging face model id
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

# 수정해야 함 -> embedding을 구하지 않고 그냥 return할 수 있도록(__call__ 수정 완료)
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")


# text 관련은 필요 없음
# text = ""

# tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer", use_fast=False,)
# text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")


# text -> 대신에 뽑아낸 embedding
# text_inputs = tokenizer(["a photo of a cat"], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
# input  = all_mlp_embeddings_array
# text_features = text_encoder(**text_inputs)  
# print(f'shape: {all_mlp_embeddings_array.shape}')  # [1, 77, 1024]

generated_images = pipe(prompt_embeds=embeddings_tensor) # error 날거임
generated_images.save('결과 저장해/.png')
