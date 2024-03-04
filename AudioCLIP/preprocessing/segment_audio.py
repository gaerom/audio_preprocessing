import os
import glob
import librosa
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm

import sys 
sys.path.append('/home/broiron/Desktop/AudioCLIP')

from model import AudioCLIP
from utils.transforms import ToTensor1D

sample_rate = 44100
segment_length = 2 * sample_rate

audio_dir = 'data/vggsound/raw_audio_val' # 뽑아낸 오디오 경로
output_dir = 'data/segments_2_val' # 새로 만들 segment 저장 경로

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

paths_to_audio = glob.glob(os.path.join(audio_dir, '*.wav'))

audio_transforms = ToTensor1D()

for path_to_audio in tqdm(paths_to_audio, desc='segmentation: '):
    track, sr = librosa.load(path_to_audio, sr=sample_rate, dtype=np.float32)
    
    # 2초 segment로 분할
    segments = [track[i:i+segment_length] for i in range(0, len(track), segment_length) if len(track[i:i+segment_length]) == segment_length]
    
    for idx, segment in enumerate(segments):
        # 각각의 audio로부터 나온 segment를 구별할 수 있도록
        output_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(path_to_audio))[0]}_segment_{idx}.wav")
        sf.write(output_filename, segment, sample_rate)
