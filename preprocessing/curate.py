from glob import glob
import librosa
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import parmap
import os

""" label에 해당하는 소리가 나오는 2s segment들 mel-spectrogram으로 변환 """
# 이거는 일단 raw_audio(.wav) 각각을 10초 단위로 끊은 다음 적용해야 함 (48s -> 10s * 4 + 8s), 8s audio에 대해서는 나머지 2초 padding 적용
audio_lists = glob("../dataset/segment_2/*.wav") # .wav
data_length = len(audio_lists)
os.makedirs("../dataset/mels", exist_ok=True) # 없으면 만들어라

def func(idx):
    try:
        wav_name = audio_lists[idx]        
        
        name = wav_name.split("/")[-1].split(".")[0]
        path = f"../dataset/mels/{name}" # 2초 segment들이 저장될 위치

        if not os.path.exists(path):
            y, sr = librosa.load(wav_name, sr=44100)
            audio_inputs = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            audio_inputs = librosa.power_to_db(audio_inputs, ref=np.max) / 80.0 + 1
            audio_inputs = np.array([audio_inputs])
            np.save(path, audio_inputs)
        #os.remove(wav_name)
    except:
        print(wav_name)
    finally:
        return 0

result = parmap.map(func, range(data_length), pm_pbar=True, pm_processes=16)