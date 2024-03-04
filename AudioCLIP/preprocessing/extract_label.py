""" 파일명에서 text label 추출 """
import os
from tqdm import tqdm

video_dir = '/home/broiron/Desktop/AudioCLIP/data/vggsound/train'
label_dir = '/home/broiron/Desktop/AudioCLIP/data/label'

if not os.path.exists(label_dir):
    os.makedirs(label_dir)

labels_file_path = os.path.join(label_dir, 'labels.txt')

# 중복 제거
labels_set = set()

with open(labels_file_path, 'w') as labels_file:
    for filename in tqdm(os.listdir(video_dir), desc='Extracting labels'):
        label = filename.split('_')[0]
 
        if label not in labels_set:
            labels_file.write(label + '\n')
            labels_set.add(label)
