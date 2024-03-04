import subprocess
import os
from glob import glob

video_dir = 'data/vggsound/val'
audio_dir = 'data/vggsound/raw_audio_val'

os.makedirs(audio_dir, exist_ok=True)

video_files = glob(os.path.join(video_dir, '*.mp4'))

for video_path in video_files:
    base_name = os.path.basename(video_path)
    file_name = os.path.splitext(base_name)[0]

    audio_output_path = os.path.join(audio_dir, file_name + '.wav')

    # .wav 형식으로 encoding
    command = f'ffmpeg -i "{video_path}" -acodec pcm_s16le -ar 44100 -ac 2 "{audio_output_path}"'
    subprocess.run(command, shell=True)
