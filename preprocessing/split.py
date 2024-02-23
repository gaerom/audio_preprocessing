import librosa
import numpy as np
import soundfile as sf
import os
from typing import List, Dict

def divide_and_save(source_directory: str, target_directory: str, segment_duration: float = 10.0):
    """ 모든 audio -> 10s segment로 분할하고 별도의 파일로 저장 """

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    
    audio_files = [f for f in os.listdir(source_directory) if f.endswith('.wav')]
    
    for file_name in audio_files:
        audio_path = os.path.join(source_directory, file_name)
        
        audio, sr = librosa.load(audio_path, sr=None)
        total_duration = len(audio) / sr  
        
        num_segments = int(np.ceil(total_duration / segment_duration))
        
        for i in range(num_segments):
            start_sample = int(i * segment_duration * sr)
            end_sample = int(min((i + 1) * segment_duration * sr, len(audio)))
        
            # segment 추출
            audio_segment = audio[start_sample:end_sample]
            
            # segment가 10초보다 짧은 경우 padding 추가
            if len(audio_segment) < segment_duration * sr:
                audio_segment = np.pad(audio_segment, (0, int(segment_duration * sr) - len(audio_segment)), 'constant')
            
            # 각 segment(10s) 저장
            segment_file_name = f"{file_name[:-4]}_segment_{i}.wav"
            target_path = os.path.join(target_directory, segment_file_name)
            sf.write(target_path, audio_segment, sr)
            print(f"Saved: {target_path}")

source_directory = '/home/broiron/Desktop/TPoS/dataset/raw_audios_2/raw_audios'
target_directory = '/home/broiron/Desktop/TPoS/dataset/segment_10' # 생성된 10초 segment 저장할 dir

divide_and_save(source_directory, target_directory)
