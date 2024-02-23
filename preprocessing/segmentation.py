import librosa
import numpy as np
import soundfile as sf
import os
import json
from typing import Dict


def load_data_from_json(json_file_path: str) -> Dict:
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data

def filter_and_save_segments(json_file_path: str, source_directory: str, target_directory: str, segment_duration: float = 2.0):
    """ 10s audio -> 2s seg로 나누고, annotation과 비교하여 label이 등장하는 segment만 저장 """

    data = load_data_from_json(json_file_path)
    
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    
    # segment_10
    audio_files = [f for f in os.listdir(source_directory) if f.endswith('.wav')]
    
    for file_name in audio_files:
        video_id = file_name.split("_segment")[0]  
        
        if video_id in data["database"]: 
            annotations = data["database"][video_id]["annotations"]
            audio_path = os.path.join(source_directory, file_name)
            
            audio, sr = librosa.load(audio_path, sr=None)
            
            # segment 생성
            for i in range(5):  # 2초씩 5개로 나눔
                start_sample = int(i * segment_duration * sr)
                end_sample = int((i + 1) * segment_duration * sr)
                audio_segment = audio[start_sample:end_sample]
                
                segment_start_time = i * segment_duration
                segment_end_time = (i + 1) * segment_duration
                
                # segment가 annotation에 명시된 label 소리 등장 시간과 겹치는지 확인
                if any(segment_start_time < anno["segment"][1] and segment_end_time > anno["segment"][0] for anno in annotations):
                    # 겹치는 경우(소리 등장), segment 저장
                    segment_file_name = f"{file_name[:-4]}_subsegment_{i}.wav"
                    target_path = os.path.join(target_directory, segment_file_name)
                    sf.write(target_path, audio_segment, sr)
                    print(f"Saved: {target_path}")

json_file_path = '/home/broiron/Desktop/TPoS/dataset/annotations/unav100_annotations.json'
source_directory = '/home/broiron/Desktop/TPoS/dataset/segment_10'
target_directory = '/home/broiron/Desktop/TPoS/dataset/segment_2'

filter_and_save_segments(json_file_path, source_directory, target_directory)
