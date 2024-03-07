""" 가진 dataset에 대해 순서대로 label 추출 """
import os
from tqdm import tqdm

seg_dir = '/home/broiron/Desktop/AudioCLIP/data/segments'
label_dir = '/home/broiron/Desktop/AudioCLIP/data/label'

# Ensure the label directory exists
if not os.path.exists(label_dir):
    os.makedirs(label_dir)

labels_file_path = os.path.join(label_dir, 'labels.txt')

# file 순서대로 label 추출 (dataset 개수와 동일)
sorted_filenames = sorted(os.listdir(seg_dir))

with open(labels_file_path, 'w') as labels_file:
    for filename in tqdm(sorted_filenames, desc='Extracting labels'):
        label = filename.split('_')[0]
        labels_file.write(label + '\n')
